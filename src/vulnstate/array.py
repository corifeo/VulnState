"""
CVD Array - Vectorized batch container for vulnerabilities

Provides:
- CVDArray: Numpy-based batch operations on multiple vulnerabilities
  - Factory methods: zeros(), ones(), random()
  - Boolean masking and filtering
  - Batch event application
  - State distribution analysis

Key features:
- O(N) vectorized operations
- Dirty tracking for efficient sync
- Analytics via CVDAnalyzer.analyze()

Layer: Batch
Dependencies: constants.py, states.py, models.py, vulnerability.py
Used by: io.py, analyzer.py
"""

import warnings
from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd

    from .constants import AntiDesiderataBit, DesiderataBit, FixPath, ThreatState
    from .models import AnalysisResult

from .constants import (
    CVDEvent,
    get_all_valid_states,
    get_state_label,
    state_int_to_string,
    string_to_state_int,
)
from .models import (
    ArrayAnalytics,
    ArrayCVDAnalytics,
    ArrayEnrichment,
    ArrayIdentifiers,
    ArrayMetadata,
    ArrayScoring,
    ArrayState,
    ArrayTimestamps,
    compute_history_id,
    compute_pair_mask,
)
from .vulnerability import CVDVulnerability


class CVDArray:
    """
    Numpy-based batch container for CVD vulnerabilities.

    Stores multiple vulnerabilities with efficient vectorized operations.
    Provides O(N) algorithms with good cache locality and numpy acceleration.

    Storage Layout:
        - states: np.ndarray (N,) uint8 - internal integer state for each vuln
        - vuln_ids: np.ndarray (N,) object - vulnerability identifiers
        - event_timestamps: Dict[CVDEvent, np.ndarray] - per-event timestamps
        - metadata: Dict[str, np.ndarray] - metadata fields

    Example:
        arr = CVDArray([vuln1, vuln2, vuln3])
        arr.apply_event_batch(CVDEvent.V)  # Apply to all eligible
        has_vendor_aware = arr.has_event_occurred(CVDEvent.V)
        critical_only = arr[arr.metadata['cvss'] > 8.0]
    """

    # ==================== INITIALIZATION ====================

    def __init__(
        self, vulnerabilities: Optional[list[CVDVulnerability]] = None, fixed_size: bool = False
    ):
        """
        Initialize array from list of vulnerabilities or create empty.

        Args:
            vulnerabilities: List of CVDVulnerability objects, or None for empty array
            fixed_size: If True, array has fixed capacity and cannot grow via imports.
                       Use False (default) for dynamic arrays that grow automatically.

        Examples:
            Dynamic array (grows automatically):
            >>> arr = CVDArray()
            >>> arr.import_nvd('data.json')  # Array grows to fit data

            Fixed-size array (strict capacity):
            >>> arr = CVDArray.zeros(100)  # Pre-allocated 100 slots
            >>> arr.import_nvd('data.json')  # ValueError if >100 items
        """
        # Live vulnerability objects (can be expunged after analysis)
        self._vulnerabilities: np.ndarray = np.array([], dtype=object)

        # Data components (dataclasses)
        self.identifiers = ArrayIdentifiers()  # vuln_id + cve_id
        self.timestamps = ArrayTimestamps()  # Event timestamps (V, F, D, P, X, A)
        self.scoring = ArrayScoring()
        self.enrichment = ArrayEnrichment()
        self.analytics = ArrayAnalytics()  # Computed metrics cache
        self.cvd_analytics = ArrayCVDAnalytics()  # Precomputed pair_mask, history_id
        self._metadata_data = ArrayMetadata()  # Legacy metadata storage

        # State dataclass (uint8 bitmask array)
        self.state = ArrayState()

        # Metadata encoders (for categorical optimization - not yet implemented)
        self._metadata_encoders: dict[str, Any] = {}

        # Dirty tracking
        self._dirty_indices: set[int] = set()

        # Analysis cache (computed on first access)
        self._analysis_cache: Optional[AnalysisResult] = None

        # Bitmask dirty tracking (recompute pair_mask/history_id when timestamps change)
        self._pair_mask_dirty = True
        self._history_id_dirty = True

        # Lazy CVSS metric parsing flag (like _analysis_cache pattern)
        self._cvss_metrics_parsed = False

        # Fixed-size semantics
        self._fixed_size = fixed_size

        if vulnerabilities:
            self._from_list(vulnerabilities)
        else:
            self._init_empty()

    def _init_empty(self) -> None:
        """Initialize empty arrays (dataclasses already initialized with empty arrays)."""
        # Dataclasses (cvd_state, identifiers, cvd_analytics, scoring, enrichment, analytics)
        # are already initialized with empty arrays in __init__
        pass

    def _from_list(self, vulnerabilities: list[CVDVulnerability]) -> None:
        """Build arrays from list of vulnerabilities."""
        # Store current fixed_size flag (preserve during rebuilds)
        was_fixed = getattr(self, "_fixed_size", False)

        n = len(vulnerabilities)

        # Live vulnerability objects
        self._vulnerabilities = np.array(vulnerabilities, dtype=object)

        # State array
        self.state.bitmask = np.array(
            [v.state.state_encoded for v in vulnerabilities], dtype=np.uint8
        )

        # Timestamps
        self.timestamps.V = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.F = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.D = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.P = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.X = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.A = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")

        # Identifiers
        self.identifiers.vuln_id = np.array(
            [v.identity.vuln_id for v in vulnerabilities], dtype=object
        )
        self.identifiers.cve_id = np.array(
            [v.identity.cve_id for v in vulnerabilities], dtype=object
        )

        # Initialize analytics arrays (will be filled by sync)
        self.analytics.severities = np.empty(n, dtype=object)
        self.analytics.fix_path = np.empty(n, dtype=np.uint8)
        self.analytics.threat_state = np.empty(n, dtype=np.uint8)
        self.analytics.is_zero_day = np.zeros(n, dtype=bool)
        self.analytics.is_fix_available = np.zeros(n, dtype=bool)
        self.analytics.is_fix_deployed = np.zeros(n, dtype=bool)
        self.analytics.is_weaponized = np.zeros(n, dtype=bool)
        self.analytics.is_under_attack = np.zeros(n, dtype=bool)
        self.analytics.is_premature_disclosure = np.zeros(n, dtype=bool)
        self.analytics.disclosure_window_days = np.full(n, np.nan, dtype=np.float32)
        self.analytics.fix_lag_days = np.full(n, np.nan, dtype=np.float32)
        self.analytics.deployment_lag_days = np.full(n, np.nan, dtype=np.float32)
        self.analytics.violated_orderings_count = np.zeros(n, dtype=np.int32)
        self.analytics.desiderata_score = np.zeros(n, dtype=np.float32)
        self.analytics.skill_score = np.full(n, np.nan, dtype=np.float32)

        # Trigger initial sync to populate arrays
        self._dirty_indices = set(range(n))
        self.sync()

        # Event timestamps as parallel arrays (backward compatibility dict)
        self._metadata_data.event_timestamps_absolute = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for event in CVDEvent:
                timestamps = []
                for v in vulnerabilities:
                    ts = v.state.events.get(event)
                    if ts:
                        timestamps.append(np.datetime64(ts, "us"))
                    else:
                        timestamps.append(np.datetime64("NaT", "us"))
                self._metadata_data.event_timestamps_absolute[event] = np.array(
                    timestamps, dtype="datetime64[us]"
                )

        # Metadata - extract all fields including cvss_score
        if not vulnerabilities:
            return

        # Extract cvss_score from scoring dataclass
        cvss_scores = [
            v.scoring.cvss_base_score if v.scoring.cvss_base_score is not None else np.nan
            for v in vulnerabilities
        ]
        self._metadata_data.raw["cvss_score"] = np.array(cvss_scores, dtype=np.float32)

        # Extract scoring data (CVSS score only - metrics parsed lazily)
        self.scoring.cvss_score = np.array(
            [
                v.scoring.cvss_base_score if v.scoring.cvss_base_score is not None else np.nan
                for v in vulnerabilities
            ],
            dtype=np.float32,
        )

        # Store raw CVSS vectors for lazy parsing (avoid parsing during import)
        self.scoring.cve_vector = np.array(
            [v.scoring.cve_vector for v in vulnerabilities], dtype=object
        )

        # Initialize CVSS metric arrays as empty (populated on first access)
        self.scoring.attack_vector = np.empty(n, dtype=object)
        self.scoring.attack_complexity = np.empty(n, dtype=object)
        self.scoring.privileges_required = np.empty(n, dtype=object)
        self.scoring.user_interaction = np.empty(n, dtype=object)
        self.scoring.scope = np.empty(n, dtype=object)
        self.scoring.confidentiality_impact = np.empty(n, dtype=object)
        self.scoring.integrity_impact = np.empty(n, dtype=object)
        self.scoring.availability_impact = np.empty(n, dtype=object)

        # Reset lazy parsing flag (metrics not yet parsed)
        self._cvss_metrics_parsed = False

        # Extract enrichment data
        self.enrichment.epss = np.array(
            [
                v.enrichment.epss if v.enrichment.epss is not None else np.nan
                for v in vulnerabilities
            ],
            dtype=np.float32,
        )

        self.enrichment.epss_percentile = np.array(
            [
                v.enrichment.epss_percentile if v.enrichment.epss_percentile is not None else np.nan
                for v in vulnerabilities
            ],
            dtype=np.float32,
        )

        self.enrichment.kev = np.array([v.enrichment.kev for v in vulnerabilities], dtype=bool)

        self.enrichment.kev_date = np.array(
            [
                (
                    np.datetime64(v.enrichment.kev_date)
                    if v.enrichment.kev_date
                    else np.datetime64("NaT")
                )
                for v in vulnerabilities
            ],
            dtype="datetime64[s]",
        )

        # Extract EPSS scores from enrichment dataclass
        epss_scores = [
            v.enrichment.epss if v.enrichment.epss is not None else np.nan for v in vulnerabilities
        ]
        self._metadata_data.raw["epss"] = np.array(epss_scores, dtype=np.float32)

        # Extract kev flags from enrichment dataclass
        kev_flags = [v.enrichment.kev for v in vulnerabilities]
        self._metadata_data.raw["kev"] = np.array(kev_flags, dtype=bool)

        # Extract cve_vector strings from scoring dataclass
        cve_vectors = [
            v.scoring.cve_vector if v.scoring.cve_vector else None for v in vulnerabilities
        ]
        self._metadata_data.raw["cve_vector"] = np.array(cve_vectors, dtype=object)

        # Store cve_ids for reconstruction (essential for preserving user-provided IDs)
        cve_ids = [v.identity.cve_id if v.identity.cve_id else None for v in vulnerabilities]
        self._metadata_data.raw["_cve_id"] = np.array(cve_ids, dtype=object)

        # Extract other metadata fields
        if vulnerabilities[0].metadata:
            for key in vulnerabilities[0].metadata:
                values = [v.metadata.get(key, np.nan) for v in vulnerabilities]
                try:
                    self._metadata_data.raw[key] = np.array(values, dtype=np.float32)
                except (ValueError, TypeError):
                    self._metadata_data.raw[key] = np.array(values, dtype=object)

        # Restore fixed_size flag (preserve during rebuilds)
        self._fixed_size = was_fixed

    # ==================== REPRESENTATION ====================

    def __len__(self) -> int:
        """Number of vulnerabilities in array."""
        return len(self.state_ints)

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        unique_states = len(self.count_by_state()) if len(self) > 0 else 0
        return f"CVDArray(n={len(self)}, unique_states={unique_states})"

    def __str__(self) -> str:
        """User-friendly string (shows summary)."""
        return self.summary

    @property
    def is_fixed_size(self) -> bool:
        """
        Check if array has fixed-size semantics.

        Returns:
            True if array was created with zeros()/ones()/random() and cannot grow.
            False if array is dynamic and can grow via imports.

        Examples:
            >>> arr = CVDArray.zeros(100)
            >>> arr.is_fixed_size
            True

            >>> arr = CVDArray()
            >>> arr.is_fixed_size
            False
        """
        return getattr(self, "_fixed_size", False)

    # ==================== DATACLASS PROPERTY ACCESSORS ====================

    @property
    def state_ints(self) -> np.ndarray:
        """Raw state bitmask array (uint8, VFDPXA bit positions)."""
        return self.state.bitmask

    @state_ints.setter
    def state_ints(self, value: np.ndarray) -> None:
        """Raw state bitmask array (uint8, VFDPXA bit positions)."""
        self.state.bitmask = value

    @property
    def vuln_ids(self) -> np.ndarray:
        """Vulnerability UUID array (object dtype)."""
        return self.identifiers.vuln_id

    @vuln_ids.setter
    def vuln_ids(self, value: np.ndarray) -> None:
        """Vulnerability UUID array (object dtype)."""
        self.identifiers.vuln_id = value

    @property
    def cve_ids(self) -> np.ndarray:
        """
        Get CVE identifiers array.

        Primary identifier for vulnerabilities. Users select and filter
        vulnerabilities by CVE ID.

        Returns:
            Array of CVE ID strings (object dtype). None values for
            vulnerabilities without CVE IDs.

        Examples:
            >>> arr.cve_ids
            array(['CVE-2024-001', 'CVE-2024-002', None], dtype=object)

            >>> mask = arr.cve_ids == 'CVE-2024-001'
            >>> critical = arr[mask]
        """
        return self.identifiers.cve_id

    @property
    def V_timestamps(self) -> np.ndarray:
        """Vendor awareness timestamps (datetime64[us], NaT if not occurred)."""
        return self.timestamps.V

    @property
    def F_timestamps(self) -> np.ndarray:
        """Fix ready timestamps (datetime64[us], NaT if not occurred)."""
        return self.timestamps.F

    @property
    def D_timestamps(self) -> np.ndarray:
        """Deployment timestamps (datetime64[us], NaT if not occurred)."""
        return self.timestamps.D

    @property
    def P_timestamps(self) -> np.ndarray:
        """Public disclosure timestamps (datetime64[us], NaT if not occurred)."""
        return self.timestamps.P

    @property
    def X_timestamps(self) -> np.ndarray:
        """Exploit public timestamps (datetime64[us], NaT if not occurred)."""
        return self.timestamps.X

    @property
    def A_timestamps(self) -> np.ndarray:
        """Attack observed timestamps (datetime64[us], NaT if not occurred)."""
        return self.timestamps.A

    @property
    def events(self) -> dict[CVDEvent, np.ndarray]:
        """
        Get event timestamp arrays as dict (matches CVDVulnerability.events pattern).

        Returns dict mapping each event to its timestamp array.
        Equivalent to accessing V_timestamps, F_timestamps, etc. individually.

        Returns:
            Dict mapping CVDEvent to np.ndarray of datetime64[us] timestamps

        Examples:
            >>> arr.events[CVDEvent.V]  # Same as arr.V_timestamps
            >>> arr.events[CVDEvent.P]  # Same as arr.P_timestamps
        """
        return {
            CVDEvent.V: self.V_timestamps,
            CVDEvent.F: self.F_timestamps,
            CVDEvent.D: self.D_timestamps,
            CVDEvent.P: self.P_timestamps,
            CVDEvent.X: self.X_timestamps,
            CVDEvent.A: self.A_timestamps,
        }

    @property
    def pair_mask(self) -> np.ndarray:
        """
        Get precomputed pair ordering mask (uint16 bitmask).

        Computed once and cached. Recomputed when timestamps change (after sync()).
        Used for fast O(1) vectorized queries on event pair relationships.

        Returns:
            np.ndarray[uint16]: Bitmask where bit i indicates if pair i is satisfied.
                Bit 0 = V≺F, Bit 1 = V≺D, ..., Bit 14 = X≺A

        Examples:
            >>> # Check if vendor aware before public (V≺P, bit 2)
            >>> is_coordinated = (arr.pair_mask & (1 << 2)) != 0

            >>> # Check for zero-day exploit (V≺X bit clear)
            >>> is_zero_day_exploit = (arr.pair_mask & (1 << 3)) == 0

        See:
            docs/design/2026-01-20-cvd-state-storage-architecture.md
        """
        if self._pair_mask_dirty or len(self.cvd_analytics.pair_mask) != len(self):
            # Compute pair_mask from state + timestamps
            self.cvd_analytics.pair_mask = compute_pair_mask(self.state, self.timestamps)
            self._pair_mask_dirty = False

        return self.cvd_analytics.pair_mask

    @property
    def history_id(self) -> np.ndarray:
        """
        Get history IDs for complete disclosure histories (0-69, or 255 for incomplete).

        For vulnerabilities where all 6 events (V, F, D, P, X, A) occurred,
        returns the canonical index (0-69) into VALID_HISTORIES based on
        the chronological ordering of events.

        For incomplete histories (fewer than 6 events), returns 255.

        Computed lazily and cached. Recomputed when timestamps change (after sync()).

        Returns:
            np.ndarray[uint8]: History ID array
                - 0: AXPVFD (worst case - attacks first)
                - 69: VFDPXA (perfect CVD)
                - 255: Incomplete history

        Examples:
            >>> # Find all perfect CVD disclosures
            >>> perfect = arr[arr.history_id == 69]

            >>> # Filter to only complete histories for analysis
            >>> complete = arr[arr.history_id != 255]

        See:
            - vulnstate.constants.VALID_HISTORIES for full list
            - docs/ref/cvd-histories.md for interpretation
        """
        if self._history_id_dirty or len(self.cvd_analytics.history_id) != len(self):
            # Compute history_id from state + timestamps
            self.cvd_analytics.history_id = compute_history_id(self.state, self.timestamps)
            self._history_id_dirty = False

        return self.cvd_analytics.history_id

    @property
    def severities(self) -> np.ndarray:
        """CVSS severity labels (object array: 'CRITICAL'/'HIGH'/'MEDIUM'/'LOW'/'NONE').

        Returns:
            np.ndarray: Severity strings derived from CVSS base scores.

        Note:
            Stored in ArrayAnalytics, updated during sync().
        """
        return self.analytics.severities

    @property
    def fix_path(self) -> np.ndarray:
        """VFD dimension as FixPath enum values (uint8 array).

        Returns:
            np.ndarray: 0=NO_AWARENESS, 1=VENDOR_AWARE, 3=FIX_READY, 7=REMEDIATED

        Note:
            Computed by CVDAnalyzer on first access, cached until invalidated.
        """
        return self.analysis.fix_path_int

    @property
    def threat_state(self) -> np.ndarray:
        """PXA dimension as ThreatState enum values (uint8 array).

        Returns:
            np.ndarray: 0=LATENT, 1=DISCLOSED, 2=WEAPONIZED, ..., 7=ACTIVE_ATTACK

        Note:
            Computed by CVDAnalyzer on first access, cached until invalidated.
        """
        return self.analysis.threat_state_int

    @property
    def is_zero_day(self) -> np.ndarray:
        """True if exploit (X) or attack (A) before vendor awareness (V) (bool array).

        Returns:
            np.ndarray[bool]: Zero-day status for each vulnerability.

        Note:
            Computed by CVDAnalyzer on first access, cached until invalidated.
        """
        return self.analysis.is_zero_day

    @property
    def is_fix_available(self) -> np.ndarray:
        """True if fix is ready (F event occurred) (bool array).

        Returns:
            np.ndarray[bool]: Derived directly from state bitmask.
        """
        return self.has_event_occurred(CVDEvent.F)

    @property
    def is_fix_deployed(self) -> np.ndarray:
        """True if fix is deployed (D event occurred) (bool array).

        Returns:
            np.ndarray[bool]: Derived directly from state bitmask.
        """
        return self.has_event_occurred(CVDEvent.D)

    @property
    def is_weaponized(self) -> np.ndarray:
        """True if public exploit exists (X event occurred) (bool array).

        Returns:
            np.ndarray[bool]: From AnalysisResult.

        Note:
            Computed by CVDAnalyzer on first access, cached until invalidated.
        """
        return self.analysis.is_weaponized

    @property
    def is_under_attack(self) -> np.ndarray:
        """True if under active attack (A event occurred) (bool array).

        Returns:
            np.ndarray[bool]: Derived directly from state bitmask.
        """
        return self.has_event_occurred(CVDEvent.A)

    @property
    def is_premature_disclosure(self) -> np.ndarray:
        """True if public disclosure (P) before fix ready (F) (bool array).

        Returns:
            np.ndarray[bool]: From AnalysisResult.

        Note:
            Computed by CVDAnalyzer on first access, cached until invalidated.
        """
        return self.analysis.is_premature_disclosure

    @property
    def is_zero_day_exploit(self) -> np.ndarray:
        """
        Check if exploit public before vendor awareness (zero-day exploit).

        True if X event occurred before V event. Indicates attacker had
        working exploit before vendor knew vulnerability existed.

        Uses precomputed pair_mask for O(1) lookup (bit 3: V≺X).

        Returns:
            Boolean array indicating zero-day exploit status

        Examples:
            >>> arr.is_zero_day_exploit
            array([False, True, False, ...], dtype=bool)
        """
        # V≺X (bit 3): if clear, then X before V (zero-day exploit)
        # Check if both events occurred first
        v_times = self.V_timestamps
        x_times = self.X_timestamps
        has_both = ~np.isnat(v_times) & ~np.isnat(x_times)

        # Use pair_mask: bit 3 clear means X before V
        result: np.ndarray = np.zeros(len(self), dtype=bool)
        result[has_both] = (self.pair_mask[has_both] & (1 << 3)) == 0
        return result

    @property
    def is_zero_day_attack(self) -> np.ndarray:
        """
        Check if attacks before vendor awareness (zero-day attack).

        True if A event occurred before V event. Indicates attacks were
        observed before vendor knew vulnerability existed.

        Uses precomputed pair_mask for O(1) lookup (bit 4: V≺A).

        Returns:
            Boolean array indicating zero-day attack status

        Examples:
            >>> arr.is_zero_day_attack
            array([False, True, False, ...], dtype=bool)
        """
        # V≺A (bit 4): if clear, then A before V (zero-day attack)
        # Check if both events occurred first
        v_times = self.V_timestamps
        a_times = self.A_timestamps
        has_both = ~np.isnat(v_times) & ~np.isnat(a_times)

        # Use pair_mask: bit 4 clear means A before V
        result: np.ndarray = np.zeros(len(self), dtype=bool)
        result[has_both] = (self.pair_mask[has_both] & (1 << 4)) == 0
        return result

    @property
    def is_coordinated(self) -> np.ndarray:
        """
        Check if vendor aware before public disclosure (coordinated disclosure).

        True if V event occurred before P event. Indicates proper coordination
        where vendor had awareness before public disclosure.

        Uses precomputed pair_mask for O(1) lookup (bit 2: V≺P).

        Returns:
            Boolean array indicating coordinated disclosure status

        Examples:
            >>> arr.is_coordinated
            array([True, False, True, ...], dtype=bool)
        """
        # V≺P (bit 2): if set, then V before P (coordinated)
        # Check if both events occurred first
        v_times = self.V_timestamps
        p_times = self.P_timestamps
        has_both = ~np.isnat(v_times) & ~np.isnat(p_times)

        # Use pair_mask: bit 2 set means V before P
        result: np.ndarray = np.zeros(len(self), dtype=bool)
        result[has_both] = (self.pair_mask[has_both] & (1 << 2)) != 0
        return result

    @property
    def is_responsible_disclosure(self) -> np.ndarray:
        """
        Check if V→F→P ordering maintained (responsible disclosure).

        True if vendor awareness (V) preceded fix ready (F) which preceded
        public disclosure (P). This is the ideal disclosure sequence.

        Uses precomputed pair_mask for O(1) lookup (bits 0, 6: V≺F and F≺P).

        Returns:
            Boolean array indicating responsible disclosure status

        Examples:
            >>> arr.is_responsible_disclosure
            array([True, False, True, ...], dtype=bool)
        """
        # V≺F (bit 0) and F≺P (bit 6): both must be set
        # Check if all three events occurred first
        v_times = self.V_timestamps
        f_times = self.F_timestamps
        p_times = self.P_timestamps
        has_all = ~np.isnat(v_times) & ~np.isnat(f_times) & ~np.isnat(p_times)

        # Use pair_mask: bits 0 and 6 must both be set
        result: np.ndarray = np.zeros(len(self), dtype=bool)
        mask_vf = (self.pair_mask[has_all] & (1 << 0)) != 0  # V≺F
        mask_fp = (self.pair_mask[has_all] & (1 << 6)) != 0  # F≺P
        result[has_all] = mask_vf & mask_fp
        return result

    @property
    def has_fix_before_exploit(self) -> np.ndarray:
        """
        Check if fix ready before exploit public.

        True if F event occurred before X event. Indicates vendor had
        fix ready before exploit became publicly available.

        Uses precomputed pair_mask for O(1) lookup (bit 7: F≺X).

        Returns:
            Boolean array indicating fix-before-exploit status

        Examples:
            >>> arr.has_fix_before_exploit
            array([True, False, True, ...], dtype=bool)
        """
        # F≺X (bit 7): if set, then F before X
        # Check if both events occurred first
        f_times = self.F_timestamps
        x_times = self.X_timestamps
        has_both = ~np.isnat(f_times) & ~np.isnat(x_times)

        # Use pair_mask: bit 7 set means F before X
        result: np.ndarray = np.zeros(len(self), dtype=bool)
        result[has_both] = (self.pair_mask[has_both] & (1 << 7)) != 0
        return result

    @property
    def has_fix_before_attack(self) -> np.ndarray:
        """
        Check if fix ready before attacks observed.

        True if F event occurred before A event. Indicates vendor had
        fix ready before attacks were observed.

        Uses precomputed pair_mask for O(1) lookup (bit 8: F≺A).

        Returns:
            Boolean array indicating fix-before-attack status

        Examples:
            >>> arr.has_fix_before_attack
            array([True, False, True, ...], dtype=bool)
        """
        # F≺A (bit 8): if set, then F before A
        # Check if both events occurred first
        f_times = self.F_timestamps
        a_times = self.A_timestamps
        has_both = ~np.isnat(f_times) & ~np.isnat(a_times)

        # Use pair_mask: bit 8 set means F before A
        result: np.ndarray = np.zeros(len(self), dtype=bool)
        result[has_both] = (self.pair_mask[has_both] & (1 << 8)) != 0
        return result

    @property
    def has_deployment_before_exploit(self) -> np.ndarray:
        """
        Check if fix deployed before exploit public.

        True if D event occurred before X event. Indicates fix was
        deployed before exploit became publicly available.

        Uses precomputed pair_mask for O(1) lookup (bit 10: D≺X).

        Returns:
            Boolean array indicating deployment-before-exploit status

        Examples:
            >>> arr.has_deployment_before_exploit
            array([True, False, True, ...], dtype=bool)
        """
        # D≺X (bit 10): if set, then D before X
        # Check if both events occurred first
        d_times = self.D_timestamps
        x_times = self.X_timestamps
        has_both = ~np.isnat(d_times) & ~np.isnat(x_times)

        # Use pair_mask: bit 10 set means D before X
        result: np.ndarray = np.zeros(len(self), dtype=bool)
        result[has_both] = (self.pair_mask[has_both] & (1 << 10)) != 0
        return result

    @property
    def has_deployment_before_attack(self) -> np.ndarray:
        """
        Check if fix deployed before attacks observed.

        True if D event occurred before A event. Indicates fix was
        deployed before attacks were observed.

        Uses precomputed pair_mask for O(1) lookup (bit 11: D≺A).

        Returns:
            Boolean array indicating deployment-before-attack status

        Examples:
            >>> arr.has_deployment_before_attack
            array([True, False, True, ...], dtype=bool)
        """
        # D≺A (bit 11): if set, then D before A
        # Check if both events occurred first
        d_times = self.D_timestamps
        a_times = self.A_timestamps
        has_both = ~np.isnat(d_times) & ~np.isnat(a_times)

        # Use pair_mask: bit 11 set means D before A
        result: np.ndarray = np.zeros(len(self), dtype=bool)
        result[has_both] = (self.pair_mask[has_both] & (1 << 11)) != 0
        return result

    @property
    def is_private_attack(self) -> np.ndarray:
        """
        Check if attacks without public exploit (targeted/private attack).

        True if A event occurred without X event. Indicates targeted
        attacks without publicly available exploit code.

        Returns:
            Boolean array indicating private attack status

        Examples:
            >>> arr.is_private_attack
            array([False, True, False, ...], dtype=bool)
        """
        x_times = self.X_timestamps
        a_times = self.A_timestamps

        # A occurred but not X
        has_a = ~np.isnat(a_times)
        has_x = ~np.isnat(x_times)

        result: np.ndarray = has_a & ~has_x
        return result

    @property
    def is_mass_exploitation(self) -> np.ndarray:
        """
        Check if both exploit public and attacks observed (mass exploitation).

        True if both X and A events occurred. Indicates widespread
        exploitation with both public exploit code and observed attacks.

        Returns:
            Boolean array indicating mass exploitation status

        Examples:
            >>> arr.is_mass_exploitation
            array([False, True, False, ...], dtype=bool)
        """
        x_times = self.X_timestamps
        a_times = self.A_timestamps

        # Both X and A occurred
        has_x = ~np.isnat(x_times)
        has_a = ~np.isnat(a_times)

        result: np.ndarray = has_x & has_a
        return result

    @property
    def disclosure_window_days(self) -> np.ndarray:
        """Days between V (vendor awareness) and P (public disclosure) (float32).

        Returns:
            np.ndarray[float32]: Window in days. NaN if V or P not occurred.

        Note:
            Computed by CVDAnalyzer on first access, cached until invalidated.
        """
        return self.analysis.disclosure_window_days

    @property
    def fix_lag_days(self) -> np.ndarray:
        """Days between V (vendor awareness) and F (fix ready) (float32).

        Returns:
            np.ndarray[float32]: Lag in days. NaN if V or F not occurred.

        Note:
            Computed by CVDAnalyzer on first access, cached until invalidated.
        """
        return self.analysis.fix_lag_days

    @property
    def deployment_lag_days(self) -> np.ndarray:
        """Days between F (fix ready) and D (fix deployed) (float32).

        Returns:
            np.ndarray[float32]: Lag in days. NaN if F or D not occurred.

        Note:
            Computed by CVDAnalyzer on first access, cached until invalidated.
        """
        return self.analysis.deployment_lag_days

    @property
    def violated_orderings_count(self) -> np.ndarray:
        """Number of violated event ordering constraints (int32, 0-12).

        Returns:
            np.ndarray[int32]: Count of set bits in anti_desiderata_mask.
        """
        # Count bits set in anti_desiderata_mask
        mask = self.analysis.anti_desiderata_mask
        count = np.zeros(len(mask), dtype=np.int32)
        for i in range(12):  # 12 desiderata pairs
            count += ((mask >> i) & 1).astype(np.int32)
        return count

    @property
    def desiderata_score(self) -> np.ndarray:
        """Fraction of satisfied desiderata (float32, 0.0-1.0).

        Returns:
            np.ndarray[float32]: desiderata_count / 12 for each vulnerability.

        Note:
            Computed by CVDAnalyzer on first access, cached until invalidated.
        """
        return self.analysis.desiderata_score

    @property
    def desiderata_mask(self) -> np.ndarray:
        """Get desiderata satisfaction as uint16 bitmask array.

        Reads from AnalysisResult.desiderata_mask computed by CVDAnalyzer.
        Each bit corresponds to a DesiderataBit. Use bitwise ops for filtering:
            arr.desiderata_mask & (1 << DesiderataBit.D1_V_P)
        """
        return self.analysis.desiderata_mask

    @property
    def anti_desiderata_mask(self) -> np.ndarray:
        """Anti-desiderata (violations) as uint16 bitmask array.

        Returns:
            np.ndarray[uint16]: Each bit corresponds to an AntiDesiderataBit.
                Complement of desiderata_mask.

        Note:
            Computed by CVDAnalyzer on first access, cached until invalidated.
        """
        return self.analysis.anti_desiderata_mask

    def get_satisfied_desiderata(self) -> list[list[str]]:
        """Get labels of satisfied desiderata for each vulnerability.

        Returns:
            List of lists, one per vulnerability, containing human-readable
            labels for satisfied desiderata (e.g., "Coordinated Disclosure").
        """
        from .constants import get_desiderata_labels

        return [get_desiderata_labels(int(m)) for m in self.desiderata_mask]

    def get_violated_desiderata(self) -> list[list[str]]:
        """Get labels of violated desiderata (anti-desiderata) for each vulnerability.

        Returns:
            List of lists, one per vulnerability, containing human-readable
            labels for violations (e.g., "Zero-Day Exploit").
        """
        from .constants import get_anti_desiderata_labels

        return [get_anti_desiderata_labels(int(m)) for m in self.anti_desiderata_mask]

    def where_desiderata_satisfied(self, *desiderata: "DesiderataBit") -> np.ndarray:
        """Return boolean mask where all specified desiderata are satisfied.

        Args:
            *desiderata: DesiderataBit values to check (AND logic)

        Returns:
            Boolean array where True means all specified desiderata satisfied.

        Example:
            >>> from vulnstate.constants import DesiderataBit
            >>> coordinated = arr.where_desiderata_satisfied(DesiderataBit.D1_V_P)
            >>> arr[coordinated]  # Filter to coordinated disclosures
        """
        required = sum(1 << d for d in desiderata)
        return (self.desiderata_mask & required) == required

    def where_desiderata_violated(self, *anti_desiderata: "AntiDesiderataBit") -> np.ndarray:
        """Return boolean mask where all specified anti-desiderata are violated.

        Args:
            *anti_desiderata: AntiDesiderataBit values to check (AND logic)

        Returns:
            Boolean array where True means all specified anti-desiderata violated.

        Example:
            >>> from vulnstate.constants import AntiDesiderataBit
            >>> zero_days = arr.where_desiderata_violated(AntiDesiderataBit.U2_X_V)
            >>> arr[zero_days]  # Filter to zero-day exploits
        """
        required = sum(1 << a for a in anti_desiderata)
        return (self.anti_desiderata_mask & required) == required

    @property
    def skill_score(self) -> np.ndarray:
        """Stakeholder skill score (float32, 0.0-1.0).

        Returns:
            np.ndarray[float32]: Weighted desiderata score. NaN if insufficient data.

        Note:
            Computed by CVDAnalyzer on first access, cached until invalidated.
        """
        return self.analysis.skill_score

    @property
    def _metadata_raw(self) -> dict[str, np.ndarray]:
        """Get metadata raw dict (backward compatibility)."""
        return self._metadata_data.raw

    @property
    def _event_timestamps_absolute(self) -> dict[CVDEvent, np.ndarray]:
        """Get event timestamps absolute dict (backward compatibility)."""
        return self._metadata_data.event_timestamps_absolute

    # ==================== CORE PROPERTIES ====================

    @property
    def shape(self) -> tuple[int]:
        """(N,) tuple."""
        return (len(self),)

    @property
    def ndim(self) -> int:
        """Always 1."""
        return 1

    @property
    def size(self) -> int:
        """Number of vulnerabilities (same as len())."""
        return len(self)

    @property
    def metadata(self) -> dict[str, np.ndarray]:
        """
        Get metadata with lazy decoding from categorical mode if optimized.

        Returns:
            Dict[str, np.ndarray] with decoded values (strings if categorical encoded)
            regardless of internal storage mode.
        """
        result = {}
        for field_name, values in self._metadata_raw.items():
            if field_name in self._metadata_encoders:
                # Decode categorical
                result[field_name] = self._metadata_encoders[field_name].inverse_transform(
                    field_name, values
                )
            else:
                result[field_name] = values
        return result

    def pluck(self, path: str) -> npt.NDArray[np.object_]:
        """
        Extract nested metadata values as numpy array using dot notation.

        Args:
            path: Dot-separated path to metadata field (e.g., 'kev.dateAdded')

        Returns:
            Numpy array with values (None/NaN for missing data).
            Numeric arrays convert None to np.nan for filtering compatibility.

        Example:
            >>> kev_dates = arr.pluck('kev.dateAdded')
            >>> high_epss = arr[arr.pluck('epss.score') > 0.8]
        """
        import numpy as np

        result: list[Any] = []
        keys = path.split(".")

        if self._vulnerabilities is not None and len(self._vulnerabilities) > 0:
            # Fast path: extract from objects
            for vuln in self._vulnerabilities:
                value = vuln.metadata
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        value = None
                        break
                result.append(value)
        else:
            # Expunged or empty: return appropriate array
            result = [None] * len(self)

        # Convert None to np.nan for numeric arrays to enable boolean filtering
        if result and all(isinstance(v, (int, float)) or v is None for v in result):
            numeric_result = [v if v is not None else np.nan for v in result]
            return np.array(numeric_result, dtype=float)

        return np.array(result, dtype=object)

    def __getitem__(
        self, idx: Union[int, slice, np.ndarray]
    ) -> Union[CVDVulnerability, "CVDArray"]:
        """
        Index into array to get vulnerability or subset.

        Args:
            idx: Integer index, slice, or boolean mask

        Returns:
            CVDVulnerability if idx is int, CVDArray if idx is slice/mask

        Raises:
            IndexError: If integer index is out of bounds
        """
        if isinstance(idx, (int, np.integer)):
            # Validate index is in bounds
            if idx < -len(self) or idx >= len(self):
                raise IndexError(f"Index {idx} out of range for array of size {len(self)}")

            # Single index: return live object
            # Use arr.get(i) for explicit intent, arr[i:i+1] for 1-element CVDArray
            return self._vulnerabilities[idx]

        elif isinstance(idx, (slice, list, np.ndarray)):
            # Return subset as new CVDArray
            subset = CVDArray()
            subset.state_ints = self.state_ints[idx]
            subset.vuln_ids = self.vuln_ids[idx]
            subset._vulnerabilities = self._vulnerabilities[idx]

            # Copy timestamp data
            subset._metadata_data.event_timestamps_absolute = {
                event: arr[idx] for event, arr in self._event_timestamps_absolute.items()
            }

            # Copy metadata (raw)
            subset._metadata_data.raw = {key: arr[idx] for key, arr in self._metadata_raw.items()}
            subset._metadata_encoders = self._metadata_encoders  # Share encoder references

            return subset

        else:
            raise TypeError(f"Invalid index type: {type(idx)}")

    def __setitem__(self, key: int, value: CVDVulnerability) -> None:
        """Set vulnerability at index."""
        if not isinstance(key, (int, np.integer)):
            raise TypeError("CVDArray only supports integer assignment")

        self._vulnerabilities[key] = value
        self._dirty_indices.add(key)

    def __iter__(self) -> Iterator[CVDVulnerability]:
        """Iterate over vulnerabilities in the array."""
        return iter(self._vulnerabilities)

    # ==================== INDEXING & SLICING ====================
    # (above: __getitem__)

    def get(self, idx: int) -> CVDVulnerability:
        """Get live CVDVulnerability object at index for mutation.

        Use this when you need to modify a specific vulnerability and have
        the changes reflected in the array (after sync()).

        Args:
            idx: Integer index (supports negative indexing)

        Returns:
            Live CVDVulnerability object

        Raises:
            IndexError: If index is out of bounds

        Example:
            vuln = arr.get(5)
            vuln.apply_event(CVDEvent.V)
            arr.sync()  # Update cached arrays
        """
        if idx < -len(self) or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for array of size {len(self)}")
        # Mark as dirty since get() is for mutation
        actual_idx = idx if idx >= 0 else len(self) + idx
        self._dirty_indices.add(actual_idx)
        return self._vulnerabilities[idx]

    # ==================== VECTORIZED STATE QUERIES ====================

    def has_event_occurred(self, event: CVDEvent) -> np.ndarray:
        """
        Check which vulnerabilities have event occurred (vectorized).

        Args:
            event: Event to check

        Returns:
            Boolean array of shape (N,) where True = event occurred
        """
        bit_pos = event
        return (self.state_ints & (1 << bit_pos)) != 0

    def event_year(self, event: CVDEvent) -> np.ndarray:
        """Year of event occurrence as int array (0 where event not occurred).

        Args:
            event: CVDEvent to extract year from.

        Returns:
            np.ndarray[int16]: Year values. 0 for vulnerabilities where
                the event has not occurred (NaT timestamps).

        Example:
            >>> published_2024 = arr[arr.event_year(CVDEvent.P) == 2024]
        """
        timestamps = self._event_timestamps_absolute[event]
        valid = ~np.isnat(timestamps)
        result: np.ndarray = np.zeros(len(self), dtype=np.int16)
        if valid.any():
            result[valid] = timestamps[valid].astype("datetime64[Y]").astype(int) + 1970
        return result

    def event_month(self, event: CVDEvent) -> np.ndarray:
        """Month of event occurrence as int array (0 where event not occurred).

        Args:
            event: CVDEvent to extract month from.

        Returns:
            np.ndarray[int8]: Month values (1-12). 0 for vulnerabilities where
                the event has not occurred (NaT timestamps).

        Example:
            >>> q1_disclosures = arr[arr.event_month(CVDEvent.P) <= 3]
        """
        timestamps = self._event_timestamps_absolute[event]
        valid = ~np.isnat(timestamps)
        result: np.ndarray = np.zeros(len(self), dtype=np.int8)
        if valid.any():
            # months since epoch mod 12, +1 for 1-based
            result[valid] = (
                timestamps[valid].astype("datetime64[M]").astype(int) % 12 + 1
            ).astype(np.int8)
        return result

    def event_age_days(self, event: CVDEvent) -> np.ndarray:
        """Days elapsed since event occurrence (float32, NaN where not occurred).

        Computes the number of days between the event timestamp and now,
        useful for vulnerability age calculations and SLA tracking.

        Args:
            event: CVDEvent to compute age from.

        Returns:
            np.ndarray[float32]: Days since event. NaN for vulnerabilities
                where the event has not occurred (NaT timestamps).

        Example:
            >>> old_vulns = arr[arr.event_age_days(CVDEvent.P) > 90]
        """
        timestamps = self._event_timestamps_absolute[event]
        valid = ~np.isnat(timestamps)
        result: np.ndarray = np.full(len(self), np.nan, dtype=np.float32)
        if valid.any():
            now = np.datetime64("now", "us")
            delta = now - timestamps[valid]
            result[valid] = delta.astype("timedelta64[s]").astype(np.float64) / 86400.0
        return result

    @property
    def terminal_mask(self) -> np.ndarray:
        """
        Boolean mask indicating which vulnerabilities are in terminal state (VFDPXA).

        Returns:
            Boolean array of shape (N,)
        """
        return self.state_ints == 0b111111

    @property
    def states(self) -> np.ndarray:
        """
        All states as string array.

        Returns:
            np.ndarray of shape (N,) with dtype=object, values are state strings
        """
        return np.array([state_int_to_string(s) for s in self.state_ints], dtype=object)

    @property
    def state_labels(self) -> np.ndarray:
        """
        All state labels as string array (announced events only).

        Returns:
            np.ndarray of shape (N,) with dtype=object, values are state labels

        Example:
            >>> arr.states  # Full state strings
            array(['VFdpxa', 'VFdpxa', 'VFDPXA'], dtype=object)
            >>> arr.state_labels  # Announced events only
            array(['VF', 'VF', 'VFDPXA'], dtype=object)
        """
        return np.array(
            [get_state_label(state_int_to_string(s)) for s in self.state_ints], dtype=object
        )

    def count_by_state(self) -> dict[str, int]:
        """
        Count vulnerabilities by current state.

        Returns:
            Dictionary mapping state strings to counts
        """
        unique_states, counts = np.unique(self.state_ints, return_counts=True)
        return {
            state_int_to_string(state): int(count) for state, count in zip(unique_states, counts)
        }

    def count_by_fix_path(self) -> "dict[FixPath, int]":
        """
        Count vulnerabilities by fix path dimension.

        Returns:
            Dictionary mapping FixPath enum values to counts.

        Example:
            >>> counts = arr.count_by_fix_path()
            >>> counts[FixPath.REMEDIATED]
            42
        """
        from .constants import FixPath

        fix_path_arr = self.fix_path
        result: dict[FixPath, int] = {}
        for fp in FixPath:
            count = int((fix_path_arr == fp).sum())
            if count > 0:
                result[fp] = count
        return result

    def count_by_threat_state(self) -> "dict[ThreatState, int]":
        """
        Count vulnerabilities by threat state dimension.

        Returns:
            Dictionary mapping ThreatState enum values to counts.

        Example:
            >>> counts = arr.count_by_threat_state()
            >>> counts[ThreatState.WEAPONIZED]
            15
        """
        from .constants import ThreatState

        threat_arr = self.threat_state
        result: dict[ThreatState, int] = {}
        for ts in ThreatState:
            count = int((threat_arr == ts).sum())
            if count > 0:
                result[ts] = count
        return result

    @property
    def event_occurrence_counts(self) -> dict[str, int]:
        """
        Count how many vulnerabilities have each event occurred.

        Returns:
            Dictionary mapping event names to counts
        """
        return {event.name: int(self.has_event_occurred(event).sum()) for event in CVDEvent}

    @property
    def event_occurrence_rates(self) -> dict[str, float]:
        """
        Percentage of vulnerabilities with each event occurred.

        Returns:
            Dictionary mapping event names to percentages (0-100)
        """
        n = len(self)
        if n == 0:
            return {}

        return {
            event.name: float(self.has_event_occurred(event).sum()) / n * 100 for event in CVDEvent
        }

    @property
    def state_distribution_labeled(self) -> dict[str, int]:
        """
        Distribution of vulnerabilities by state label (cube names).

        Returns the count of vulnerabilities for each unique state label,
        where state labels are the announced event combinations (e.g., 'VFD', 'VFDPXA').

        Returns:
            Dictionary mapping state labels to counts

        Example:
            >>> arr = CVDArray([vuln1, vuln2, vuln3])  # states: VFdpxa, VFdpxa, VFDPXA
            >>> arr.state_distribution_labeled
            {'VF': 2, 'VFDPXA': 1}
        """
        label_counts: dict[str, int] = {}
        state_strings = self.states
        for state_str in state_strings:
            label = get_state_label(state_str)
            label_counts[label] = label_counts.get(label, 0) + 1

        return label_counts

    # ==================== BATCH OPERATIONS ====================

    def apply_event(
        self,
        index: int,
        event: CVDEvent,
        timestamp: Optional[datetime] = None,
        sync: bool = False,
    ) -> "CVDArray":
        """
        Apply event to vulnerability at index.

        Args:
            index: Index of vulnerability
            event: Event to apply
            timestamp: Optional timestamp
            sync: If True, sync immediately after applying

        Returns:
            self (for chaining)
        """
        # Apply to live object
        self._vulnerabilities[index].apply_event(event, timestamp=timestamp)

        # Mark as dirty
        self._dirty_indices.add(index)

        # Auto-sync if requested
        if sync:
            self.sync()

        return self

    def sync(self, indices: Optional[np.ndarray] = None) -> "CVDArray":
        """
        Update cached arrays from dirty objects.

        Syncs state, timestamps, and enrichment data from live vulnerability
        objects to cached arrays.

        Args:
            indices: Optional specific indices to sync (uses _dirty_indices if None)

        Returns:
            self (for chaining)
        """
        # Determine which indices to sync
        to_sync = set(indices) if indices is not None else self._dirty_indices.copy()

        # Sync each dirty index
        for idx in to_sync:
            vuln = self._vulnerabilities[idx]

            # Update state and identifiers
            self.state.bitmask[idx] = vuln.state.state_encoded
            self.identifiers.vuln_id[idx] = vuln.identity.vuln_id
            self.identifiers.cve_id[idx] = vuln.identity.cve_id

            # Extract timestamps to exploded arrays
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for event, attr_name in [
                    (CVDEvent.V, "V"),
                    (CVDEvent.F, "F"),
                    (CVDEvent.D, "D"),
                    (CVDEvent.P, "P"),
                    (CVDEvent.X, "X"),
                    (CVDEvent.A, "A"),
                ]:
                    ts = vuln.state.events.get(event)
                    if ts:
                        getattr(self.timestamps, attr_name)[idx] = np.datetime64(ts, "us")
                    else:
                        getattr(self.timestamps, attr_name)[idx] = np.datetime64("NaT")

            # Sync enrichment data (EPSS, KEV) - only if arrays are initialized
            if len(self.enrichment.epss) > idx:
                if vuln.enrichment.epss is not None:
                    self.enrichment.epss[idx] = vuln.enrichment.epss
                if vuln.enrichment.epss_percentile is not None:
                    self.enrichment.epss_percentile[idx] = vuln.enrichment.epss_percentile
                self.enrichment.kev[idx] = vuln.enrichment.kev
                if vuln.enrichment.kev_date is not None:
                    self.enrichment.kev_date[idx] = np.datetime64(vuln.enrichment.kev_date, "s")

        # Clear dirty flags
        self._dirty_indices -= to_sync

        # Mark bitmasks as dirty (timestamps changed)
        if to_sync:
            self._pair_mask_dirty = True
            self._history_id_dirty = True

        return self

    def expunge(self) -> "CVDArray":
        """
        Release vulnerability objects from memory.

        After expunge(), individual vulnerabilities cannot be accessed via
        indexing. Use this after sync() when you only need array operations.

        Returns:
            self (for chaining)
        """
        self._vulnerabilities = np.empty(0, dtype=object)
        return self

    @property
    def analysis(self) -> "AnalysisResult":
        """
        Computed analytics for this array (lazy, cached).

        Returns AnalysisResult dataclass with all computed analytics.
        Automatically triggers CVDAnalyzer.analyze() on first access.
        Cache is invalidated when array is modified.

        Returns:
            AnalysisResult with all 11 groups of analytics
        """
        from .analyzer import CVDAnalyzer

        # Check if cached and valid
        if self._analysis_cache is None:
            self._analysis_cache = CVDAnalyzer.analyze(self)
        return self._analysis_cache

    def invalidate_analysis(self) -> None:
        """Invalidate the cached analysis, forcing recomputation on next access.

        Note:
            Call after modifying state_ints or timestamps directly (bypassing
            apply_event/sync). The analysis property will recompute automatically.
        """
        self._analysis_cache = None

    def reanalyze(self, infer: bool = True) -> "AnalysisResult":
        """Force re-run of analysis after manual data changes.

        Use this after importing new data or manually modifying timestamps
        to re-run inference rules and recompute all analytics.

        Args:
            infer: If True (default), apply inference rules before analysis.
                   If False, analyze raw data without inferring missing events.

        Example:
            # Load initial data
            arr = CVDArray(vulns)
            _ = arr.analysis  # First access triggers analysis

            # Import vendor CSV with updated V timestamps
            for i, vuln in enumerate(arr._vulnerabilities):
                if vuln.cve_id in vendor_data:
                    vuln.events[CVDEvent.V] = vendor_data[vuln.cve_id]

            # Force re-analysis with new data
            arr.sync()
            result = arr.reanalyze()  # Re-runs inference + analytics

            # Or analyze raw data without inference
            raw_result = arr.reanalyze(infer=False)

        Returns:
            Fresh AnalysisResult with all computed analytics
        """
        from .analyzer import CVDAnalyzer

        self._analysis_cache = CVDAnalyzer.analyze(self, infer=infer)
        return self._analysis_cache

    def apply_event_batch(
        self,
        event: CVDEvent,
        mask: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        """
        Apply event to multiple vulnerabilities at once (vectorized).

        Validates V→F→D constraints before applying.

        Args:
            event: Event to apply
            mask: Optional boolean mask (N,) indicating which to update.
                  If None, applies to all that don't have event and satisfy constraints.
            timestamp: Timestamp for the event (default: now)

        Returns:
            Boolean mask (N,) indicating which vulnerabilities were updated
        """
        if mask is None:
            # Create mask: vulnerabilities that don't have event
            mask = ~self.has_event_occurred(event)

            # Apply V→F→D constraints (vectorized version of CVDEvent.validate_vfd_constraint)
            if event == CVDEvent.F:
                # F requires V
                mask &= self.has_event_occurred(CVDEvent.V)
            elif event == CVDEvent.D:
                # D requires F
                mask &= self.has_event_occurred(CVDEvent.F)

        # Apply event by setting bit
        bit_pos = event
        self.state_ints[mask] |= 1 << bit_pos

        # Update timestamps - always update _event_timestamps_absolute
        # (if in delta mode, optimization will be re-applied by user)
        ts = timestamp or datetime.now()
        ts_dt64 = np.datetime64(ts, "us")

        # Only set where not already set
        existing = self._event_timestamps_absolute[event]
        new_ts = np.where(np.isnat(existing) & mask, ts_dt64, existing)
        self._event_timestamps_absolute[event] = new_ts

        return mask

    # ==================== CONVERSIONS ====================

    def to_matrix(self) -> np.ndarray:
        """
        Convert all states to binary feature matrix for ML.

        Each row is a vulnerability, each column is an event (V, F, D, P, X, A).

        Returns:
            np.ndarray of shape (N, 6) with dtype=uint8
            Values are 0 or 1 indicating event occurrence
        """
        n = len(self)
        matrix = np.zeros((n, 6), dtype=np.uint8)

        for i, event in enumerate(CVDEvent):
            matrix[:, i] = self.has_event_occurred(event).astype(np.uint8)

        return matrix

    @staticmethod
    def from_matrix(matrix: np.ndarray, vuln_ids: Optional[list[str]] = None) -> "CVDArray":
        """
        Create CVDArray from binary feature matrix.

        Args:
            matrix: np.ndarray of shape (N, 6) with binary values
            vuln_ids: Optional list of vulnerability IDs

        Returns:
            New CVDArray instance

        Raises:
            ValueError: If matrix shape is not (N, 6)
        """
        # Validate matrix shape
        if matrix.ndim != 2 or matrix.shape[1] != 6:
            raise ValueError(f"Matrix must have shape (N, 6) for 6 CVD events, got {matrix.shape}")
        n = matrix.shape[0]
        arr = CVDArray()

        # Convert rows to state integers
        states = np.zeros(n, dtype=np.uint8)
        for i, event in enumerate(CVDEvent):
            states = (states | (matrix[:, i].astype(np.uint8) << int(event))).astype(np.uint8)

        # Create vulnerability objects from states
        vulnerabilities = []
        ids = vuln_ids if vuln_ids else [f"V{i}" for i in range(n)]
        for i, state_int in enumerate(states):
            vuln = CVDVulnerability(ids[i])
            # Set state by applying events based on state bits
            for event in CVDEvent:
                if state_int & (1 << event):
                    vuln.apply_event(event, timestamp=None)
            vulnerabilities.append(vuln)

        arr._vulnerabilities = np.array(vulnerabilities, dtype=object)
        arr.state_ints = states
        arr.vuln_ids = np.array(ids, dtype=object)
        arr._metadata_data.event_timestamps_absolute = {
            event: np.full(n, np.datetime64("NaT"), dtype="datetime64[us]") for event in CVDEvent
        }
        arr._metadata_data.raw = {}

        return arr

    # ==================== ANALYSIS & DISPLAY ====================

    @property
    def summary(self) -> str:
        """
        Formatted summary of entire vulnerability array.

        Returns:
            Multi-line string with statistics
        """
        n = len(self)
        if n == 0:
            return "Empty CVDArray (0 vulnerabilities)"

        lines = ["CVDArray Summary", "=" * 50, f"Total Vulnerabilities: {n}", ""]

        # State distribution
        state_counts = self.count_by_state()
        lines.append("State Distribution:")
        for state, count in sorted(state_counts.items()):
            pct = count / n * 100
            lines.append(f"  {state}: {count} ({pct:.1f}%)")
        lines.append("")

        # Event occurrence rates
        lines.append("Event Occurrence Rates:")
        for event_name, pct in self.event_occurrence_rates.items():
            lines.append(f"  {event_name}: {pct:.1f}%")

        # Terminal states
        terminal_count = int(self.terminal_mask.sum())
        lines.append(f"\nComplete (VFDPXA): {terminal_count}/{n} ({terminal_count / n * 100:.1f}%)")

        return "\n".join(lines)

    # ==================== BATCH SERIALIZATION METHODS ====================

    def to_dict_batch(self, include_computed: bool = False) -> list[dict[str, Any]]:
        """
        Convert all vulnerabilities to list of dicts.

        Args:
            include_computed: Include computed properties in each dict

        Returns:
            List of vulnerability dictionaries
        """
        return [vuln.to_dict(include_computed=include_computed) for vuln in self]

    def to_dataframe(
        self,
        include_analytics: bool = True,
        include_events: bool = False,
        explode_cvss: bool = True,
        explode_metadata: bool = True,
    ) -> "pd.DataFrame":
        """
        Convert array to pandas DataFrame with comprehensive data.

        Args:
            include_analytics: Include computed metrics (default True)
            include_events: Add V, F, D, P, X, A columns as 0/1 flags (default False)
            explode_cvss: CVSS vector as separate columns (default True)
            explode_metadata: Unpack metadata dicts into columns (default True)

        Returns:
            pandas DataFrame with one row per vulnerability

        Example:
            >>> df = arr.to_dataframe(include_events=True)
            >>> df[['cve_id', 'V', 'F', 'D', 'P', 'X', 'A']].head()
        """
        from vulnstate.io import CVDIO

        return CVDIO.to_dataframe(
            self, include_analytics, include_events, explode_cvss, explode_metadata
        )

    def to_json_batch(self, filepath: str, include_computed: bool = False) -> None:
        """
        Save all vulnerabilities to JSON file.

        Args:
            filepath: Path to save .json file
            include_computed: Include computed properties
        """
        import json

        # Use to_json() for each vulnerability to handle type conversions
        json_list = [json.loads(vuln.to_json(include_computed=include_computed)) for vuln in self]
        with open(filepath, "w") as f:
            json.dump(json_list, f, indent=2)

    @classmethod
    def from_json_batch(cls, filepath: str) -> "CVDArray":
        """
        Load multiple vulnerabilities from JSON file.

        Args:
            filepath: Path to .json file

        Returns:
            CVDArray with loaded vulnerabilities
        """
        import json

        with open(filepath) as f:
            data_list = json.load(f)
        vulns = [CVDVulnerability.from_dict(data) for data in data_list]
        return cls(vulns)

    def import_nvd(
        self,
        source: Union[str, list[dict[str, Any]]],
        apply_event: bool = True,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        skip_existing: bool = False,
    ) -> None:
        """
        Import NVD vulnerability data.

        Args:
            source: Path to NVD JSON file or list of CVE items
            apply_event: Apply event P (Public) from publishedDate (default: True)
            import_metadata: Store full NVD record in metadata['nvd'] (default: False)
            include: Only import these metadata fields
            exclude: Skip these metadata fields
            skip_existing: Skip CVEs already in array (default: False)

        Example:
            >>> arr = CVDArray()
            >>> with open("nvdcve-1.1-2024.json") as f:
            ...     data = json.load(f)
            >>> arr.import_nvd(data["CVE_Items"])
        """
        from vulnstate.io import CVDIO

        CVDIO.import_nvd(
            self, source, apply_event, import_metadata, include, exclude, skip_existing
        )

    def import_nvd_glob(
        self,
        pattern: str,
        apply_event: bool = True,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        skip_existing: bool = True,
    ) -> int:
        """
        Import NVD data from multiple files matching a glob pattern.

        Args:
            pattern: Glob pattern for NVD JSON files (e.g., 'nvdcve-*.json')
            apply_event: Apply event P (Public) from publishedDate (default: True)
            import_metadata: Store full NVD record in metadata['nvd'] (default: False)
            include: Only import these metadata fields
            exclude: Skip these metadata fields
            skip_existing: Skip CVEs already in array (default: True)

        Returns:
            Number of files processed

        Example:
            >>> arr = CVDArray.zeros(0)
            >>> count = arr.import_nvd_glob('nvdcve-2.0-*.json')
            >>> print(f"Loaded from {count} files")
        """
        from vulnstate.io import CVDIO

        return CVDIO.import_nvd_glob(
            self, pattern, apply_event, import_metadata, include, exclude, skip_existing
        )

    def save_pickle_batch(self, filepath: str) -> None:
        """
        Save all vulnerabilities to pickle file (fast).

        Args:
            filepath: Path to save .pkl file
        """
        import pickle

        with open(filepath, "wb") as f:
            pickle.dump(list(self), f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load_pickle_batch(cls, filepath: str) -> "CVDArray":
        """
        Load multiple vulnerabilities from pickle file.

        Args:
            filepath: Path to .pkl file

        Returns:
            CVDArray with loaded vulnerabilities
        """
        import pickle

        with open(filepath, "rb") as f:
            vulns = pickle.load(f)
        return cls(vulns)

    # ==================== DATA IMPORT ====================

    def import_epss(
        self,
        source: Union[str, dict[str, Union[float, dict[str, Any]]]],
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """
        Import EPSS (Exploit Prediction Scoring System) data.

        Delegates to CVDIO for implementation.

        Args:
            source: Filepath to EPSS CSV or dict mapping CVE IDs to EPSS data
            import_metadata: Store full EPSS data in vuln.metadata['epss'] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)

        Example:
            >>> epss_scores = {'CVE-2024-001': 0.85, 'CVE-2024-002': 0.42}
            >>> arr.import_epss(epss_scores)
            >>> # With metadata
            >>> epss_data = {'CVE-2024-001': {'score': 0.85, 'percentile': 0.95}}
            >>> arr.import_epss(epss_data, import_metadata=True)
        """
        from .io import CVDIO

        CVDIO.import_epss(
            self, source, import_metadata=import_metadata, include=include, exclude=exclude
        )

    def import_kev(
        self,
        source: Union[str, dict[str, dict[str, Any]]],
        apply_event: bool = True,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """
        Import KEV catalog data.

        Args:
            source: Filepath to KEV CSV or dict mapping CVE IDs to KEV data
            apply_event: Apply event A with dateAdded timestamp (default True)
            import_metadata: Store full KEV data in metadata['kev'] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)

        Example:
            >>> arr.import_kev("kev.csv")
            >>> arr.import_kev("kev.csv", import_metadata=True)
        """
        from .io import CVDIO

        CVDIO.import_kev(self, source, apply_event, import_metadata, include, exclude)

    def import_csv(
        self,
        source: Union[str, list[dict[str, Any]]],
        cve_column: str,
        event: CVDEvent,
        timestamp_column: str,
        apply_event: bool = True,
        metadata_namespace: Optional[str] = None,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """
        Generic CSV import that applies any CVD event with timestamps.

        This method enables importing vendor patch advisories (event F),
        threat intel (event A), or any custom timeline data. It can match
        CVEs and apply the specified event with a timestamp, optionally
        storing the full row data in metadata.

        Delegates to CVDIO for implementation.

        Args:
            source: Filepath to CSV or list of row dicts
            cve_column: Column name containing CVE IDs
            event: CVD event to apply (V, F, D, P, X, or A)
            timestamp_column: Column name containing event timestamps
            apply_event: Apply event with timestamp (default True)
            metadata_namespace: Namespace for storing row metadata (default: None)
            import_metadata: Store full row in vuln.metadata[namespace] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)

        Example:
            # Import vendor patch dates (event F)
            >>> arr.import_csv(
            ...     source='vendor_patches.csv',
            ...     cve_column='cve_id',
            ...     event=CVDEvent.F,
            ...     timestamp_column='patch_date',
            ...     apply_event=True
            ... )

            # Import threat intel with metadata
            >>> arr.import_csv(
            ...     source='threat_intel.csv',
            ...     cve_column='cve',
            ...     event=CVDEvent.A,
            ...     timestamp_column='attack_date',
            ...     apply_event=True,
            ...     import_metadata=True,
            ...     metadata_namespace='threat_intel'
            ... )

            # Import disclosure dates (event P)
            >>> arr.import_csv(
            ...     source='disclosures.csv',
            ...     cve_column='vuln_id',
            ...     event=CVDEvent.P,
            ...     timestamp_column='disclosure_date'
            ... )
        """
        from .io import CVDIO

        CVDIO.import_csv(
            self,
            source,
            cve_column,
            event,
            timestamp_column,
            apply_event,
            metadata_namespace,
            import_metadata,
            include,
            exclude,
        )

    def import_json(
        self,
        source: Union[str, list[dict[str, Any]]],
        cve_field: str,
        event: CVDEvent,
        timestamp_field: str,
        apply_event: bool = True,
        metadata_namespace: Optional[str] = None,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """
        Generic JSON import that applies any CVD event with timestamps.

        Supports nested field access using dot notation (e.g., "vulnerability.cve_id"
        accesses {"vulnerability": {"cve_id": "CVE-2024-001"}}).

        This method enables importing data from JSON files or lists of dicts
        with complex nested structures. It can match CVEs using nested paths
        and apply the specified event with a timestamp, optionally storing
        the full record in metadata.

        Delegates to CVDIO for implementation.

        Args:
            source: Filepath to JSON or list of dicts
            cve_field: Field path to CVE IDs (supports dot notation for nested fields)
            event: CVD event to apply (V, F, D, P, X, or A)
            timestamp_field: Field path to event timestamps (supports dot notation)
            apply_event: Apply event with timestamp (default True)
            metadata_namespace: Namespace for storing row metadata (default: None)
            import_metadata: Store full row in vuln.metadata[namespace] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)

        Example:
            # Import from nested JSON structure
            >>> arr.import_json(
            ...     source='threat_intel.json',
            ...     cve_field='vulnerability.cve_id',
            ...     event=CVDEvent.A,
            ...     timestamp_field='threat_intel.first_observed',
            ...     apply_event=True
            ... )

            # Import vendor patches with metadata
            >>> arr.import_json(
            ...     source='vendor_data.json',
            ...     cve_field='cve.id',
            ...     event=CVDEvent.F,
            ...     timestamp_field='patch.release_date',
            ...     apply_event=True,
            ...     import_metadata=True,
            ...     metadata_namespace='vendor'
            ... )

            # Import from list of dicts (API response)
            >>> threat_data = [
            ...     {"vuln": {"id": "CVE-2024-001"}, "observed": "2024-03-01"}
            ... ]
            >>> arr.import_json(
            ...     source=threat_data,
            ...     cve_field='vuln.id',
            ...     event=CVDEvent.A,
            ...     timestamp_field='observed'
            ... )
        """
        from .io import CVDIO

        CVDIO.import_json(
            self,
            source,
            cve_field,
            event,
            timestamp_field,
            apply_event,
            metadata_namespace,
            import_metadata,
            include,
            exclude,
        )

    # ==================== FACTORY METHODS ====================

    @classmethod
    def zeros(cls, n: int, vuln_id_prefix: Optional[str] = None) -> "CVDArray":
        """
        Create fixed-size array of n vulnerabilities in initial state (vfdpxa).

        All vulnerabilities start in the initial 'vfdpxa' state with no events applied.
        Fixed-size arrays have strict capacity limits - attempting to import more items
        than capacity raises ValueError. Use CVDArray() for dynamic growth.

        Args:
            n: Number of vulnerabilities to create (fixed capacity)
            vuln_id_prefix: Optional prefix for auto-generated CVE IDs (e.g., 'ZERO')
                If provided, creates IDs like ZERO-00000, ZERO-00001, etc.

        Returns:
            Fixed-size CVDArray with n vulnerabilities in vfdpxa state

        Example:
            >>> arr = CVDArray.zeros(100)  # 100 initial vulnerabilities
            >>> arr.is_fixed_size
            True
            >>> len(arr)
            100
        """
        vulns = []
        for i in range(n):
            cve_id = f"{vuln_id_prefix}-{i:05d}" if vuln_id_prefix else None
            # Create with no awareness flags (defaults to initial state vfdpxa)
            vuln = CVDVulnerability(cve_id=cve_id)
            vulns.append(vuln)
        return cls(vulns, fixed_size=True)

    @classmethod
    def ones(cls, n: int, vuln_id_prefix: Optional[str] = None) -> "CVDArray":
        """
        Create fixed-size array of n vulnerabilities in terminal state (VFDPXA).

        All vulnerabilities start with all events announced (VFDPXA state).
        This is useful for testing and scenarios where all phases are complete.
        Fixed-size arrays have strict capacity limits - attempting to import more items
        than capacity raises ValueError. Use CVDArray() for dynamic growth.

        Args:
            n: Number of vulnerabilities to create (fixed capacity)
            vuln_id_prefix: Optional prefix for auto-generated CVE IDs (e.g., 'TERM')
                If provided, creates IDs like TERM-00000, TERM-00001, etc.

        Returns:
            Fixed-size CVDArray with n vulnerabilities in VFDPXA state

        Example:
            >>> arr = CVDArray.ones(100)  # 100 terminal state vulnerabilities
            >>> arr.is_fixed_size
            True
        """
        vulns = []
        for i in range(n):
            cve_id = f"{vuln_id_prefix}-{i:05d}" if vuln_id_prefix else None
            # Create with all awareness flags to reach terminal state VFDPXA
            vuln = CVDVulnerability(
                cve_id=cve_id,
                vendor_aware=True,
                fix_aware=True,
                deployed_aware=True,
                public_aware=True,
                exploit_aware=True,
                attack_aware=True,
                create_timestamps=False,
            )
            vulns.append(vuln)
        return cls(vulns, fixed_size=True)

    @classmethod
    def random(
        cls, n: int, vuln_id_prefix: Optional[str] = None, seed: Optional[int] = None
    ) -> "CVDArray":
        """
        Create fixed-size array of n vulnerabilities with random valid states.

        Each vulnerability is assigned a random valid state from the 32 possible CVD states.
        Useful for testing and simulations.
        Fixed-size arrays have strict capacity limits - attempting to import more items
        than capacity raises ValueError. Use CVDArray() for dynamic growth.

        Args:
            n: Number of vulnerabilities to create (fixed capacity)
            vuln_id_prefix: Optional prefix for auto-generated CVE IDs (e.g., 'RAND')
                If provided, creates IDs like RAND-00000, RAND-00001, etc.
            seed: Optional random seed for reproducibility

        Returns:
            Fixed-size CVDArray with n vulnerabilities in random valid states

        Example:
            >>> arr = CVDArray.random(1000)  # 1000 random vulnerabilities
            >>> arr.is_fixed_size
            True
        """
        if seed is not None:
            np.random.seed(seed)

        valid_states = get_all_valid_states()
        vulns = []
        for i in range(n):
            cve_id = f"{vuln_id_prefix}-{i:05d}" if vuln_id_prefix else None
            # Select random valid state
            random_state_str = np.random.choice(valid_states)
            # Convert state string to awareness flags for constructor
            state_int = string_to_state_int(random_state_str)
            vuln = CVDVulnerability(
                cve_id=cve_id,
                vendor_aware=bool(state_int & (1 << CVDEvent.V)),
                fix_aware=bool(state_int & (1 << CVDEvent.F)),
                deployed_aware=bool(state_int & (1 << CVDEvent.D)),
                public_aware=bool(state_int & (1 << CVDEvent.P)),
                exploit_aware=bool(state_int & (1 << CVDEvent.X)),
                attack_aware=bool(state_int & (1 << CVDEvent.A)),
                create_timestamps=False,
            )
            vulns.append(vuln)
        return cls(vulns, fixed_size=True)

    @classmethod
    def generate(
        cls,
        size: int,
        event_probs: Optional[dict[CVDEvent, float]] = None,
        cvss_range: tuple[float, float] = (3.0, 10.0),
        vendors: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> "CVDArray":
        """
        Generate a realistic sample dataset with configurable distributions.

        Creates vulnerabilities with randomized events, CVSS scores, and vendor
        assignments based on the provided probability distributions. Unlike random(),
        this method produces realistic timelines with proper event ordering.

        Args:
            size: Number of vulnerabilities to generate
            event_probs: Probability of each event occurring. Defaults:
                V=1.0, F=0.7, D=0.4, P=0.5, X=0.2, A=0.1
            cvss_range: Min/max CVSS base score range (default: 3.0-10.0)
            vendors: List of vendor names to assign. Default:
                ["VendorA", "VendorB", "VendorC", "VendorD", "VendorE"]
            seed: Optional random seed for reproducibility

        Returns:
            CVDArray with generated vulnerabilities (not fixed-size)

        Example:
            >>> arr = CVDArray.generate(1000, seed=42)
            >>> arr = CVDArray.generate(500, event_probs={CVDEvent.X: 0.5, CVDEvent.A: 0.3})
        """
        import random as rand_mod

        if seed is not None:
            np.random.seed(seed)
            rand_mod.seed(seed)

        default_probs: dict[CVDEvent, float] = {
            CVDEvent.V: 1.0,
            CVDEvent.F: 0.7,
            CVDEvent.D: 0.4,
            CVDEvent.P: 0.5,
            CVDEvent.X: 0.2,
            CVDEvent.A: 0.1,
        }
        if event_probs:
            default_probs.update(event_probs)
        probs = default_probs

        if vendors is None:
            vendors = ["VendorA", "VendorB", "VendorC", "VendorD", "VendorE"]

        cvss_min, cvss_max = cvss_range

        vulns: list[CVDVulnerability] = []
        for i in range(size):
            cve_id = f"CVE-2024-{i:05d}"
            vendor = vendors[i % len(vendors)]
            cvss = round(rand_mod.uniform(cvss_min, cvss_max), 1)

            vuln = CVDVulnerability(
                cve_id=cve_id,
                vendor=vendor,
                cvss_score=cvss,
            )

            # Base timestamp for this vulnerability
            base = datetime(2024, 1, 1) + timedelta(days=i % 365)

            # Apply events based on probabilities, respecting V→F→D constraint
            if rand_mod.random() < probs[CVDEvent.V]:
                vuln.apply_event(CVDEvent.V, timestamp=base)

                if rand_mod.random() < probs[CVDEvent.F]:
                    vuln.apply_event(
                        CVDEvent.F,
                        timestamp=base + timedelta(days=rand_mod.randint(5, 30)),
                    )

                    if rand_mod.random() < probs[CVDEvent.D]:
                        vuln.apply_event(
                            CVDEvent.D,
                            timestamp=base + timedelta(days=rand_mod.randint(20, 60)),
                        )

            # P, X, A are independent of V→F→D
            if rand_mod.random() < probs[CVDEvent.P]:
                vuln.apply_event(
                    CVDEvent.P,
                    timestamp=base + timedelta(days=rand_mod.randint(1, 45)),
                )

            if rand_mod.random() < probs[CVDEvent.X]:
                vuln.apply_event(
                    CVDEvent.X,
                    timestamp=base + timedelta(days=rand_mod.randint(0, 30)),
                )

            if rand_mod.random() < probs[CVDEvent.A]:
                vuln.apply_event(
                    CVDEvent.A,
                    timestamp=base + timedelta(days=rand_mod.randint(5, 45)),
                )

            vulns.append(vuln)

        return cls(vulns)

    # ==================== SCORING PROPERTIES ====================

    def _ensure_cvss_metrics_parsed(self) -> None:
        """Lazy parse all CVSS vectors on first metric access (batch).

        Parses all vectors at once to populate all 8 CVSS metric arrays.
        This avoids parsing during import, deferring the cost to first access.
        """
        if self._cvss_metrics_parsed:
            return

        from .parsers import NVDParser

        # Get raw vectors from scoring dataclass
        vectors: np.ndarray = (
            self.scoring.cve_vector
            if hasattr(self.scoring, "cve_vector")
            else np.array([], dtype=object)
        )

        if len(vectors) == 0:
            # Empty array, nothing to parse
            self._cvss_metrics_parsed = True
            return

        # Parse all vectors at once
        metrics = [NVDParser.parse_cvss_vector(v) for v in vectors]

        # Populate ALL scoring arrays
        self.scoring.attack_vector = np.array([m["AV"] for m in metrics], dtype=object)
        self.scoring.attack_complexity = np.array([m["AC"] for m in metrics], dtype=object)
        self.scoring.privileges_required = np.array([m["PR"] for m in metrics], dtype=object)
        self.scoring.user_interaction = np.array([m["UI"] for m in metrics], dtype=object)
        self.scoring.scope = np.array([m["S"] for m in metrics], dtype=object)
        self.scoring.confidentiality_impact = np.array([m["C"] for m in metrics], dtype=object)
        self.scoring.integrity_impact = np.array([m["I"] for m in metrics], dtype=object)
        self.scoring.availability_impact = np.array([m["A"] for m in metrics], dtype=object)

        self._cvss_metrics_parsed = True

    @property
    def cvss_scores(self) -> np.ndarray:
        """CVSS base scores (float32, 0.0-10.0)."""
        return self.scoring.cvss_score

    @property
    def cve_vectors(self) -> np.ndarray:
        """CVSS vector strings (object array, e.g. 'CVSS:3.1/AV:N/AC:L/...')."""
        return self._metadata_raw.get("cve_vector", np.array([], dtype=object))

    @property
    def attack_vector(self) -> np.ndarray:
        """CVSS AV metric (object: N/A/L/P, None if unparsed)."""
        self._ensure_cvss_metrics_parsed()
        return self.scoring.attack_vector

    @property
    def attack_complexity(self) -> np.ndarray:
        """CVSS AC metric (object: L/H, None if unparsed)."""
        self._ensure_cvss_metrics_parsed()
        return self.scoring.attack_complexity

    @property
    def privileges_required(self) -> np.ndarray:
        """CVSS PR metric (object: N/L/H, None if unparsed)."""
        self._ensure_cvss_metrics_parsed()
        return self.scoring.privileges_required

    @property
    def user_interaction(self) -> np.ndarray:
        """CVSS UI metric (object: N/R, None if unparsed)."""
        self._ensure_cvss_metrics_parsed()
        return self.scoring.user_interaction

    @property
    def scope(self) -> np.ndarray:
        """CVSS S metric (object: U/C, None if unparsed)."""
        self._ensure_cvss_metrics_parsed()
        return self.scoring.scope

    @property
    def confidentiality_impact(self) -> np.ndarray:
        """CVSS C metric (object: N/L/H, None if unparsed)."""
        self._ensure_cvss_metrics_parsed()
        return self.scoring.confidentiality_impact

    @property
    def integrity_impact(self) -> np.ndarray:
        """CVSS I metric (object: N/L/H, None if unparsed)."""
        self._ensure_cvss_metrics_parsed()
        return self.scoring.integrity_impact

    @property
    def availability_impact(self) -> np.ndarray:
        """CVSS A metric (object: N/L/H, None if unparsed)."""
        self._ensure_cvss_metrics_parsed()
        return self.scoring.availability_impact

    # ==================== ENRICHMENT PROPERTIES ====================

    @property
    def epss(self) -> np.ndarray:
        """EPSS exploitation probability (float32, 0.0-1.0)."""
        return self.enrichment.epss

    @property
    def kev(self) -> np.ndarray:
        """CISA KEV catalog membership (bool array)."""
        return self.enrichment.kev

    @property
    def epss_percentile(self) -> np.ndarray:
        """EPSS percentile rank (float32, 0.0-1.0)."""
        return self.enrichment.epss_percentile

    @property
    def kev_dates(self) -> np.ndarray:
        """KEV catalog date added (datetime64[s])."""
        return self.enrichment.kev_date
