"""CVD Array - Vectorized batch container for vulnerabilities.

Provides:
- CVDArray: Numpy-based batch operations on multiple vulnerabilities
  - Factory methods: zeros(), ones(), random(), generate()
  - Boolean masking and filtering
  - Batch event application
  - Lifecycle namespace for state access

Lifecycle Integration
---------------------
CVDArray provides vectorized lifecycle operations through arr.lifecycle:

    arr.lifecycle.state           # State strings: ["VFdpxa", "vfdPxa", ...]
    arr.lifecycle.fix_path        # FixPath values as uint8 array
    arr.lifecycle.threat_state    # ThreatState values as uint8 array
    arr.lifecycle.timestamps.V    # Vendor awareness timestamps (datetime64)
    arr.lifecycle.apply_event(CVDEvent.P)  # Apply to all
    arr.lifecycle.apply_event(CVDEvent.X, mask=high_risk)  # Apply to subset
    arr.lifecycle.desiderata      # DesiderataExtractor for analytics

On indexing (arr[i]), CVDArray reconstructs a ScalarLifecycle from columnar data
and returns a CVDVulnerability with that lifecycle as its _lifecycle attribute.

Storage Layout (columnar for vectorized ops):
    - state_ints: uint8 array of 6-bit bitmasks
    - internal_ids/cve_ids: object arrays
    - timestamps.V/F/D/P/X/A: datetime64 arrays
    - _vulnerabilities: object array of live CVDVulnerability refs

Key Features:
- O(N) vectorized operations with numpy acceleration
- Dirty tracking for efficient sync between live objects and columnar data
- Analytics via arr.lifecycle.desiderata

Layer: Batch
Dependencies: constants.py, models.py, lifecycle.py, vulnerability.py
Used by: io.py
"""

import warnings
from collections.abc import Iterator
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd

    from .constants import AntiDesiderataBit, DesiderataBit, FixPath, ThreatState
    from .models import AnalysisResult, CVSSMetrics

from . import factories as _factories
from .constants import (
    CVDEvent,
    get_state_label,
    state_int_to_string,
)
from .lifecycle import LifecycleNamespace
from .models import (
    ArrayCVDAnalytics,
    ArrayIdentifiers,
    ArrayMetadata,
    ArraySource,
    ArrayState,
    ArrayTimestamps,
    CVSSScore,
    KEVEntry,
    compute_history_id,
    compute_pair_mask,
)
from .transforms import Transform
from .vulnerability import CVDVulnerability


class CVDArray:
    """
    Numpy-based batch container for CVD vulnerabilities.

    Stores multiple vulnerabilities with efficient vectorized operations.
    Provides O(N) algorithms with good cache locality and numpy acceleration.

    Storage Layout:
        - states: np.ndarray (N,) uint8 - internal integer state for each vuln
        - internal_ids: np.ndarray (N,) object - vulnerability identifiers
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
        self.identifiers = ArrayIdentifiers()  # internal_id + cve_id
        self.timestamps = ArrayTimestamps()  # Event timestamps (V, F, D, P, X, A)
        self.cvd_analytics = ArrayCVDAnalytics()  # Precomputed pair_mask, history_id
        self._metadata_data = ArrayMetadata()  # Legacy metadata storage

        # State dataclass (uint8 bitmask array)
        self.state = ArrayState()

        # Metadata encoders (for categorical optimization - not yet implemented)
        self._metadata_encoders: dict[str, Any] = {}

        # Dirty tracking
        self._dirty_indices: set[int] = set()

        # Bitmask dirty tracking (recompute pair_mask/history_id when timestamps change)
        self._pair_mask_dirty = True
        self._history_id_dirty = True

        # ETL: Computed data (populated by _recompute)
        self._analysis: Optional[AnalysisResult] = None
        self._cvss_metrics: Optional[CVSSMetrics] = None
        self._kev_dates: Optional[np.ndarray] = None

        # Fixed-size semantics
        self._fixed_size = fixed_size

        # Transform infrastructure (API v2)
        # _source: Holds object arrays (lists of CVSSScore, EPSSScore, etc.)
        # _cache: Holds computed numpy arrays from transforms
        # _transforms: List of registered transforms to run
        self._source = ArraySource(
            cvss_scores=np.array([], dtype=object),
            epss_scores=np.array([], dtype=object),
            cwes=np.array([], dtype=object),
            cpes=np.array([], dtype=object),
            kev=np.array([], dtype=object),
            exploits=np.array([], dtype=object),
        )
        self._cache: dict[str, np.ndarray] = {}
        self._transforms: list[Transform[Any]] = []

        # Transform registry for lazy dispatch via __getattr__ (API v2)
        self._transform_registry: dict[str, Any] = {}
        self._transform_cache: dict[str, Any] = {}
        self._register_core_transforms()

        # Transform state tracking (API v2)
        self._is_transformed: bool = False
        self._transformed_at: Optional[datetime] = None
        self._stale: bool = False

        if vulnerabilities:
            self._from_list(vulnerabilities)
        else:
            self._init_empty()
            self._recompute()

    def _init_empty(self) -> None:
        """Initialize empty arrays (dataclasses already initialized with empty arrays)."""
        # Dataclasses (cvd_state, identifiers, cvd_analytics, scoring, enrichment, analytics)
        # are already initialized with empty arrays in __init__
        pass

    def _register_core_transforms(self) -> None:
        """Register built-in transforms."""
        from vulnstate.transforms import ScoreExtractor

        self._transform_registry["scores"] = ScoreExtractor()

    def __getattr__(self, name: str) -> Any:
        """Lazy dispatch to registered transforms."""
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        if name in self._transform_registry:
            if name not in self._transform_cache:
                self._transform_cache[name] = self._transform_registry[name].apply(self)
            return self._transform_cache[name]
        raise AttributeError(f"No transform '{name}' registered")

    def invalidate(self, name: Optional[str] = None) -> None:
        """Clear transform cache.

        Args:
            name: Specific transform to invalidate, or None to clear all.
        """
        if name is None:
            self._transform_cache.clear()
        else:
            self._transform_cache.pop(name, None)

    def _from_list(self, vulnerabilities: list[CVDVulnerability]) -> None:
        """Build arrays from list of vulnerabilities."""
        # Store current fixed_size flag (preserve during rebuilds)
        was_fixed = getattr(self, "_fixed_size", False)

        n = len(vulnerabilities)

        # Live vulnerability objects
        self._vulnerabilities = np.array(vulnerabilities, dtype=object)

        # State array - get bitmask from lifecycle
        self.state.bitmask = np.array(
            [v._lifecycle.bitmask for v in vulnerabilities], dtype=np.uint8
        )

        # Initialize _source with empty lists for each vulnerability (API v2)
        self._source = ArraySource(
            cvss_scores=np.empty(n, dtype=object),
            epss_scores=np.empty(n, dtype=object),
            cwes=np.empty(n, dtype=object),
            cpes=np.empty(n, dtype=object),
            kev=np.empty(n, dtype=object),
            exploits=np.empty(n, dtype=object),
        )
        for i in range(n):
            self._source.cvss_scores[i] = []
            self._source.epss_scores[i] = []
            self._source.cwes[i] = []
            self._source.cpes[i] = []
            self._source.kev[i] = None
            self._source.exploits[i] = []

        # Timestamps
        self.timestamps.V = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.F = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.D = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.P = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.X = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.A = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")

        # Identifiers
        self.identifiers.internal_id = np.array(
            [v.identity.internal_id for v in vulnerabilities], dtype=object
        )
        self.identifiers.cve_id = np.array(
            [v.identity.cve_id for v in vulnerabilities], dtype=object
        )

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
                    ts = v._lifecycle._timestamps.get(event)
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

        # Populate _source with enrichment data from vulnerabilities (API v2)
        for i, v in enumerate(vulnerabilities):
            self._source.cvss_scores[i] = v.cvss_scores.copy() if v.cvss_scores else []
            self._source.epss_scores[i] = v.epss_scores.copy() if v.epss_scores else []
            self._source.cwes[i] = v.cwes.copy() if v.cwes else []
            self._source.cpes[i] = v.cpes.copy() if v.cpes else []
            self._source.kev[i] = v.kev_entry
            self._source.exploits[i] = v.exploits.copy() if v.exploits else []

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

        # ETL: Compute derived data
        self._recompute()

    def _recompute(self) -> None:
        """Recompute all derived data (ETL approach).

        Called by __init__, _from_list(), and sync().
        """
        from .models import CVSSMetrics
        from .transforms.desiderata import DesiderataExtractor

        n = len(self)
        if n == 0:
            self._analysis = DesiderataExtractor.analyze(self, infer=False)
            self._cvss_metrics = CVSSMetrics.empty(0)
            self._kev_dates = np.array([], dtype="datetime64[s]")
            return

        # Compute analysis (infer=False to avoid side effects on timestamps)
        self._analysis = DesiderataExtractor.analyze(self, infer=False)

        # Parse CVSS vectors
        self._cvss_metrics = self._parse_cvss_vectors()

        # Extract KEV dates
        self._kev_dates = self._extract_kev_dates()

    def _parse_cvss_vectors(self) -> "CVSSMetrics":
        """Parse CVSS vectors into metric arrays."""
        from .models import CVSSMetrics
        from .parsers import NVDParser

        n = len(self)
        metrics = CVSSMetrics.empty(n)

        for i in range(n):
            cvss_list: list[CVSSScore] = self._source.cvss_scores[i]  # type: ignore[assignment]
            if cvss_list:
                vector = cvss_list[0].vector
                if vector:
                    parsed = NVDParser.parse_cvss_vector(vector)
                    metrics.attack_vector[i] = parsed.get("AV")
                    metrics.attack_complexity[i] = parsed.get("AC")
                    metrics.privileges_required[i] = parsed.get("PR")
                    metrics.user_interaction[i] = parsed.get("UI")
                    metrics.scope[i] = parsed.get("S")
                    metrics.confidentiality_impact[i] = parsed.get("C")
                    metrics.integrity_impact[i] = parsed.get("I")
                    metrics.availability_impact[i] = parsed.get("A")

        return metrics

    def _extract_kev_dates(self) -> np.ndarray:
        """Extract KEV added dates into array."""
        n = len(self)
        dates: np.ndarray = np.empty(n, dtype="datetime64[s]")
        dates[:] = np.datetime64("NaT")

        for i in range(n):
            kev_entry: Optional[KEVEntry] = self._source.kev[i]  # type: ignore[assignment]
            if kev_entry is not None and kev_entry.added_at:
                dates[i] = np.datetime64(kev_entry.added_at, "s")

        return dates

    # ==================== TRANSFORM INFRASTRUCTURE ====================

    def register_transform(self, transform: Transform[Any]) -> None:
        """Register a transform to be run on this array.

        Transforms are pluggable computation units that extract data from
        _source and populate _cache with computed numpy arrays.

        Args:
            transform: Transform instance implementing the Transform protocol.

        Example:
            >>> arr = CVDArray.generate(100)
            >>> arr.register_transform(ScoreExtractor())
            >>> arr.run_transforms()
            >>> arr._cache["cvss_score"]  # Populated by transform
        """
        self._transforms.append(transform)

    def run_transforms(self) -> None:
        """Run all registered transforms and populate cache.

        Iterates through registered transforms, calling apply() on each
        and updating _cache with the returned slots.

        Example:
            >>> arr.register_transform(ScoreExtractor())
            >>> arr.run_transforms()
            >>> print(arr._cache.keys())  # ['cvss_score', 'epss_probability', ...]
        """
        for transform in self._transforms:
            slots = transform.apply(self)
            self._cache.update(slots)

    def transform(
        self,
        only: Optional[list[str]] = None,
        force: bool = False,
    ) -> "CVDArray":
        """Run transforms and compute derived data.

        Args:
            only: List of transform names to run (None = all)
            force: If True, re-run even if already transformed and not stale

        Returns:
            self (for chaining)

        Example:
            >>> arr = CVDArray.generate(100)
            >>> arr.transform()  # Run all transforms
            >>> arr.is_transformed
            True
            >>> arr.transform()  # Skips (already transformed, not stale)
            >>> arr.transform(force=True)  # Forces re-run
        """
        # Skip if already transformed, not stale, and not forced
        if self._is_transformed and not self._stale and not force and only is None:
            return self

        # Run selected or all transforms
        transforms_to_run = self._transforms
        if only:
            transforms_to_run = [t for t in self._transforms if t.name in only]

        for transform_obj in transforms_to_run:
            slots = transform_obj.apply(self)
            self._cache.update(slots)

        # Update state tracking
        self._is_transformed = True
        self._stale = False
        self._transformed_at = datetime.now()

        return self

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

    # ==================== TRANSFORM STATE TRACKING (API v2) ====================

    @property
    def is_transformed(self) -> bool:
        """True if transform() has been called at least once."""
        return self._is_transformed

    @property
    def stale(self) -> bool:
        """True if mutations occurred since last transform()."""
        return self._stale

    @property
    def transformed_at(self) -> Optional[datetime]:
        """Timestamp of last transform() call, or None if never called."""
        return self._transformed_at

    def _require_transform(self, property_name: str) -> None:
        """Raise TransformNotRunError if transform() has not been called.

        Args:
            property_name: Name of the property being accessed (for error message).

        Raises:
            TransformNotRunError: If transform() has not been called.
        """
        if not self._is_transformed:
            from .constants import TransformNotRunError

            raise TransformNotRunError(property_name)

    # ==================== DATACLASS PROPERTY ACCESSORS ====================

    @property
    def lifecycle(self) -> LifecycleNamespace:
        """CVD lifecycle public API (arr.lifecycle.*).

        Provides human-readable state access:
        - arr.lifecycle.state - State strings (e.g., "VFdpXa")
        - arr.lifecycle.summary - Human-readable summaries
        - arr.lifecycle.fix_path - FixPath enum values
        - arr.lifecycle.threat_state - ThreatState enum values
        - arr.lifecycle.timestamps.V/F/D/P/X/A - Event timestamps
        - arr.lifecycle.desiderata - Desiderata analysis
        - arr.lifecycle.bitmask - Raw state bitmask (advanced)
        - arr.lifecycle.pair_mask - Raw pair ordering mask (advanced)
        """
        return LifecycleNamespace(self)

    @property
    def state_ints(self) -> np.ndarray:
        """Raw state bitmask array (uint8, VFDPXA bit positions)."""
        return self.state.bitmask

    @state_ints.setter
    def state_ints(self, value: np.ndarray) -> None:
        """Raw state bitmask array (uint8, VFDPXA bit positions)."""
        self.state.bitmask = value

    @property
    def internal_ids(self) -> np.ndarray:
        """Vulnerability UUID array (object dtype)."""
        return self.identifiers.internal_id

    @internal_ids.setter
    def internal_ids(self, value: np.ndarray) -> None:
        """Vulnerability UUID array (object dtype)."""
        self.identifiers.internal_id = value

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
    def fix_path(self) -> np.ndarray:
        """VFD dimension as FixPath enum values (uint8 array).

        Returns:
            np.ndarray: 0=NO_AWARENESS, 1=VENDOR_AWARE, 3=FIX_READY, 7=REMEDIATED

        Note:
            Computed by DesiderataExtractor on first access, cached until invalidated.
        """
        return self.analysis.fix_path_int

    @property
    def threat_state(self) -> np.ndarray:
        """PXA dimension as ThreatState enum values (uint8 array).

        Returns:
            np.ndarray: 0=LATENT, 1=DISCLOSED, 2=WEAPONIZED, ..., 7=ACTIVE_ATTACK

        Note:
            Computed by DesiderataExtractor on first access, cached until invalidated.
        """
        return self.analysis.threat_state_int

    @property
    def is_zero_day(self) -> np.ndarray:
        """True if exploit (X) or attack (A) before vendor awareness (V)."""
        self._require_transform("is_zero_day")
        return self.lifecycle.desiderata.is_zero_day

    @property
    def is_fix_available(self) -> np.ndarray:
        """True if fix is ready (F event occurred)."""
        self._require_transform("is_fix_available")
        return self.lifecycle.desiderata.is_fix_available

    @property
    def is_fix_deployed(self) -> np.ndarray:
        """True if fix is deployed (D event occurred)."""
        self._require_transform("is_fix_deployed")
        return self.lifecycle.desiderata.is_fix_deployed

    @property
    def is_weaponized(self) -> np.ndarray:
        """True if public exploit exists (X event occurred)."""
        self._require_transform("is_weaponized")
        return self.lifecycle.desiderata.is_weaponized

    @property
    def is_under_attack(self) -> np.ndarray:
        """True if under active attack (A event occurred)."""
        self._require_transform("is_under_attack")
        return self.lifecycle.desiderata.is_under_attack

    @property
    def is_premature_disclosure(self) -> np.ndarray:
        """True if public disclosure (P) before fix ready (F)."""
        self._require_transform("is_premature_disclosure")
        return self.lifecycle.desiderata.is_premature_disclosure

    @property
    def is_zero_day_exploit(self) -> np.ndarray:
        """True if exploit (X) before vendor awareness (V)."""
        self._require_transform("is_zero_day_exploit")
        return self.lifecycle.desiderata.is_zero_day_exploit

    @property
    def is_zero_day_attack(self) -> np.ndarray:
        """True if attack (A) before vendor awareness (V)."""
        self._require_transform("is_zero_day_attack")
        return self.lifecycle.desiderata.is_zero_day_attack

    @property
    def is_coordinated(self) -> np.ndarray:
        """True if vendor aware (V) before public disclosure (P)."""
        self._require_transform("is_coordinated")
        return self.lifecycle.desiderata.is_coordinated

    @property
    def is_responsible_disclosure(self) -> np.ndarray:
        """True if V→F→P ordering maintained."""
        self._require_transform("is_responsible_disclosure")
        return self.lifecycle.desiderata.is_responsible_disclosure

    @property
    def has_fix_before_exploit(self) -> np.ndarray:
        """True if fix ready (F) before exploit public (X)."""
        self._require_transform("has_fix_before_exploit")
        return self.lifecycle.desiderata.has_fix_before_exploit

    @property
    def has_fix_before_attack(self) -> np.ndarray:
        """True if fix ready (F) before attacks observed (A)."""
        self._require_transform("has_fix_before_attack")
        return self.lifecycle.desiderata.has_fix_before_attack

    @property
    def has_deployment_before_exploit(self) -> np.ndarray:
        """True if fix deployed (D) before exploit public (X)."""
        self._require_transform("has_deployment_before_exploit")
        return self.lifecycle.desiderata.has_deployment_before_exploit

    @property
    def has_deployment_before_attack(self) -> np.ndarray:
        """True if fix deployed (D) before attacks observed (A)."""
        self._require_transform("has_deployment_before_attack")
        return self.lifecycle.desiderata.has_deployment_before_attack

    @property
    def is_private_attack(self) -> np.ndarray:
        """True if attacks (A) without public exploit (X)."""
        self._require_transform("is_private_attack")
        return self.lifecycle.desiderata.is_private_attack

    @property
    def is_mass_exploitation(self) -> np.ndarray:
        """True if both exploit public (X) and attacks observed (A)."""
        self._require_transform("is_mass_exploitation")
        return self.lifecycle.desiderata.is_mass_exploitation

    @property
    def disclosure_window_days(self) -> np.ndarray:
        """Days between V (vendor awareness) and P (public disclosure) (float32).

        Returns:
            np.ndarray[float32]: Window in days. NaN if V or P not occurred.

        Note:
            Computed by DesiderataExtractor on first access, cached until invalidated.
        """
        return self.analysis.disclosure_window_days

    @property
    def fix_lag_days(self) -> np.ndarray:
        """Days between V (vendor awareness) and F (fix ready) (float32).

        Returns:
            np.ndarray[float32]: Lag in days. NaN if V or F not occurred.

        Note:
            Computed by DesiderataExtractor on first access, cached until invalidated.
        """
        return self.analysis.fix_lag_days

    @property
    def deployment_lag_days(self) -> np.ndarray:
        """Days between F (fix ready) and D (fix deployed) (float32).

        Returns:
            np.ndarray[float32]: Lag in days. NaN if F or D not occurred.

        Note:
            Computed by DesiderataExtractor on first access, cached until invalidated.
        """
        return self.analysis.deployment_lag_days

    @property
    def desiderata_score(self) -> np.ndarray:
        """Fraction of satisfied desiderata (float32, 0.0-1.0).

        Returns:
            np.ndarray[float32]: desiderata_count / 12 for each vulnerability.

        Note:
            Computed by DesiderataExtractor on first access, cached until invalidated.
        """
        return self.analysis.desiderata_score

    @property
    def desiderata_mask(self) -> np.ndarray:
        """Get desiderata satisfaction as uint16 bitmask array.

        Reads from AnalysisResult.desiderata_mask computed by DesiderataExtractor.
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
            Computed by DesiderataExtractor on first access, cached until invalidated.
        """
        return self.analysis.anti_desiderata_mask

    def get_satisfied_desiderata(self) -> list[list[str]]:
        """Get labels of satisfied desiderata for each vulnerability."""
        return self.lifecycle.desiderata.get_satisfied_labels()

    def get_violated_desiderata(self) -> list[list[str]]:
        """Get labels of violated desiderata for each vulnerability."""
        return self.lifecycle.desiderata.get_violated_labels()

    def where_desiderata_satisfied(self, *desiderata: "DesiderataBit") -> np.ndarray:
        """Return boolean mask where all specified desiderata are satisfied."""
        return self.lifecycle.desiderata.where_satisfied(*desiderata)

    def where_desiderata_violated(self, *anti_desiderata: "AntiDesiderataBit") -> np.ndarray:
        """Return boolean mask where all specified anti-desiderata are violated."""
        return self.lifecycle.desiderata.where_violated(*anti_desiderata)

    @property
    def skill_score(self) -> np.ndarray:
        """Stakeholder skill score (float32, 0.0-1.0).

        Returns:
            np.ndarray[float32]: Weighted desiderata score. NaN if insufficient data.

        Note:
            Computed by DesiderataExtractor on first access, cached until invalidated.
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
            subset.internal_ids = self.internal_ids[idx]
            subset._vulnerabilities = self._vulnerabilities[idx]

            # Copy timestamp data
            subset._metadata_data.event_timestamps_absolute = {
                event: arr[idx] for event, arr in self._event_timestamps_absolute.items()
            }

            # Copy metadata (raw)
            subset._metadata_data.raw = {key: arr[idx] for key, arr in self._metadata_raw.items()}
            subset._metadata_encoders = self._metadata_encoders  # Share encoder references

            # Slice transform infrastructure (API v2)
            subset._source = self._source[idx]
            subset._cache = {k: v[idx] for k, v in self._cache.items()}
            subset._transforms = self._transforms.copy()
            subset._transform_registry = self._transform_registry.copy()
            subset._transform_cache = {}  # Clear lazy cache, will recompute

            # Copy transform state (API v2)
            subset._is_transformed = self._is_transformed
            subset._transformed_at = self._transformed_at
            subset._stale = False  # Fresh subset starts not stale

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
        """Check which vulnerabilities have event occurred (vectorized)."""
        return self.lifecycle.has_event(event)

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
            result[valid] = (timestamps[valid].astype("datetime64[M]").astype(int) % 12 + 1).astype(
                np.int8
            )
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
        mask: Optional[np.ndarray] = None,
        sync: bool = False,
    ) -> "CVDArray":
        """
        Apply event to vulnerability at index, or to multiple vulnerabilities if mask provided.

        Args:
            index: Index of vulnerability (ignored if mask is provided)
            event: Event to apply
            timestamp: Optional timestamp for the event
            mask: Optional boolean mask - if provided, applies event to all indices where True
            sync: If True, sync immediately after applying

        Returns:
            self (for chaining)
        """
        if mask is not None:
            # Batch mode: apply to all indices where mask is True
            indices = np.where(mask)[0]
            for idx in indices:
                self._vulnerabilities[idx].apply_event(event, timestamp=timestamp)
                self._dirty_indices.add(idx)
        else:
            # Single index mode
            self._vulnerabilities[index].apply_event(event, timestamp=timestamp)
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

            # Update state and identifiers from lifecycle
            self.state.bitmask[idx] = vuln._lifecycle.bitmask
            self.identifiers.internal_id[idx] = vuln.identity.internal_id
            self.identifiers.cve_id[idx] = vuln.identity.cve_id

            # Extract timestamps to exploded arrays from lifecycle
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
                    ts = vuln._lifecycle._timestamps.get(event)
                    if ts:
                        getattr(self.timestamps, attr_name)[idx] = np.datetime64(ts, "us")
                    else:
                        getattr(self.timestamps, attr_name)[idx] = np.datetime64("NaT")

            # Sync enrichment data to _source (API v2)
            if len(self._source.cvss_scores) > idx:
                self._source.cvss_scores[idx] = vuln.cvss_scores.copy() if vuln.cvss_scores else []
                self._source.epss_scores[idx] = vuln.epss_scores.copy() if vuln.epss_scores else []
                self._source.cwes[idx] = vuln.cwes.copy() if vuln.cwes else []
                self._source.cpes[idx] = vuln.cpes.copy() if vuln.cpes else []
                self._source.kev[idx] = vuln.kev_entry
                self._source.exploits[idx] = vuln.exploits.copy() if vuln.exploits else []

            # Invalidate transform cache (data changed)
            self._cache.clear()

        # Clear dirty flags
        self._dirty_indices -= to_sync

        # Mark bitmasks as dirty (timestamps changed)
        if to_sync:
            self._pair_mask_dirty = True
            self._history_id_dirty = True
            # ETL: Recompute derived data
            self._recompute()

            # Mark stale after mutations synced (API v2)
            if self._is_transformed:
                self._stale = True

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
        """Computed analytics for this array.

        Populated at construction and updated by sync()/apply_event_batch().
        """
        if self._analysis is None:
            self._recompute()
        return self._analysis

    def invalidate_analysis(self) -> None:
        """Invalidate the cached analysis, forcing recomputation on next access.

        Note:
            Call after modifying state_ints or timestamps directly (bypassing
            apply_event/sync). The analysis property will recompute automatically.
        """
        self._analysis = None

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
        from .transforms.desiderata import DesiderataExtractor

        self._analysis = DesiderataExtractor.analyze(self, infer=infer)
        return self._analysis

    def apply_event_batch(
        self,
        event: CVDEvent,
        mask: Optional[np.ndarray] = None,
        timestamp: Optional[datetime] = None,
    ) -> np.ndarray:
        """Apply event to multiple vulnerabilities at once (vectorized)."""
        return self.lifecycle.apply_event(event, mask, timestamp)

    # ==================== ANALYSIS & DISPLAY ====================

    @property
    def summary(self) -> str:
        """Formatted summary of entire vulnerability array."""
        from .formatting import CVDFormatter

        return CVDFormatter.array_summary_text(self)

    # ==================== BATCH SERIALIZATION METHODS ====================

    def to_dict_batch(self, include_computed: bool = False) -> list[dict[str, Any]]:
        """Convert all vulnerabilities to list of dicts."""
        from .io import CVDIO

        return CVDIO.array_to_dicts(self, include_computed)

    def to_dataframe(
        self,
        include_analytics: bool = True,
        include_events: bool = False,
        explode_cvss: bool = True,
        explode_metadata: bool = True,
    ) -> "pd.DataFrame":
        """Convert array to pandas DataFrame."""
        from .io import CVDIO

        return CVDIO.to_dataframe(
            self, include_analytics, include_events, explode_cvss, explode_metadata
        )

    def to_json_batch(self, filepath: str, include_computed: bool = False) -> None:
        """Save all vulnerabilities to JSON file."""
        from .io import CVDIO

        CVDIO.to_json_file(self, filepath, include_computed)

    @classmethod
    def from_json_batch(cls, filepath: str) -> "CVDArray":
        """Load multiple vulnerabilities from JSON file."""
        from .io import CVDIO

        return CVDIO.from_json_file(filepath)

    def import_nvd(
        self,
        source: Union[str, list[dict[str, Any]]],
        apply_event: bool = True,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        skip_existing: bool = False,
        infer_vendor: bool = True,
        infer_timestamps: bool = True,
        include_rejected: bool = False,
    ) -> None:
        """Import NVD vulnerability data."""
        from .io import CVDIO

        CVDIO.import_nvd(
            self,
            source,
            apply_event,
            import_metadata,
            include,
            exclude,
            skip_existing,
            infer_vendor=infer_vendor,
            infer_timestamps=infer_timestamps,
            include_rejected=include_rejected,
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
        """Import NVD data from multiple files matching a glob pattern."""
        from .io import CVDIO

        return CVDIO.import_nvd_glob(
            self, pattern, apply_event, import_metadata, include, exclude, skip_existing
        )

    def save_pickle_batch(self, filepath: str) -> None:
        """Save all vulnerabilities to pickle file (fast)."""
        from .io import CVDIO

        CVDIO.to_pickle_file(self, filepath)

    @classmethod
    def load_pickle_batch(cls, filepath: str) -> "CVDArray":
        """Load multiple vulnerabilities from pickle file."""
        from .io import CVDIO

        return CVDIO.from_pickle_file(filepath)

    # ==================== DATA IMPORT ====================

    def import_epss(
        self,
        source: Union[str, dict[str, Union[float, dict[str, Any]]]],
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """Import EPSS data. See CVDIO.import_epss for full docs."""
        from .io import CVDIO

        CVDIO.import_epss(self, source, import_metadata, include, exclude)

    def import_kev(
        self,
        source: Union[str, dict[str, dict[str, Any]]],
        apply_event: bool = True,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """Import KEV catalog data. See CVDIO.import_kev for full docs."""
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
        """Generic CSV import. See CVDIO.import_csv for full docs."""
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
        self.sync()

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
        """Generic JSON import. See CVDIO.import_json for full docs."""
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

    def import_dict(
        self,
        data: list[dict[str, Any]],
        on_error: str = "skip",
    ) -> "CVDArray":
        """Import vulnerabilities from list of dicts.

        Enables dict roundtrip workflow:
        >>> exported = arr.to_dict_batch()
        >>> arr2 = CVDArray()
        >>> arr2.import_dict(exported)

        Args:
            data: List of vulnerability dicts (from to_dict_batch or manual).
                  Each dict should have at minimum 'cve_id' or 'internal_id'.
                  Optional fields: 'state', 'cvss_scores', 'epss_scores', etc.
            on_error: Error handling mode:
                - "skip": Skip invalid records silently (default)
                - "raise": Raise exception on first invalid record
                - "collect": Skip invalid records but collect errors (future)

        Returns:
            self (for chaining)

        Example:
            >>> arr = CVDArray()
            >>> arr.import_dict([
            ...     {"cve_id": "CVE-2024-001", "state": "VFdpxa"},
            ...     {"cve_id": "CVE-2024-002", "state": "vfdPxa"},
            ... ])
            >>> len(arr)
            2
        """
        from .io import CVDIO

        CVDIO.import_dict(self, data, on_error)
        return self

    # ==================== FACTORY METHODS ====================
    # Implementations delegated to factories module; see factories.py for details

    @classmethod
    def zeros(cls, n: int, internal_id_prefix: Optional[str] = None) -> "CVDArray":
        """Create fixed-size array of n vulnerabilities in initial state (vfdpxa)."""
        return _factories.create_zeros(n, internal_id_prefix)

    @classmethod
    def ones(cls, n: int, internal_id_prefix: Optional[str] = None) -> "CVDArray":
        """Create fixed-size array of n vulnerabilities in terminal state (VFDPXA)."""
        return _factories.create_ones(n, internal_id_prefix)

    @classmethod
    def random(
        cls, n: int, internal_id_prefix: Optional[str] = None, seed: Optional[int] = None
    ) -> "CVDArray":
        """Create fixed-size array of n vulnerabilities with random valid states."""
        return _factories.create_random(n, internal_id_prefix, seed)

    @classmethod
    def generate(
        cls,
        size: int,
        event_probs: Optional[dict[CVDEvent, float]] = None,
        cvss_range: tuple[float, float] = (3.0, 10.0),
        vendors: Optional[list[str]] = None,
        seed: Optional[int] = None,
    ) -> "CVDArray":
        """Generate a realistic sample dataset with configurable distributions."""
        return _factories.generate(size, event_probs, cvss_range, vendors, seed)

    # ==================== SCORING PROPERTIES ====================

    @property
    def cvss_scores(self) -> np.ndarray:
        """CVSS base scores (float32, 0.0-10.0).

        Raises:
            TransformNotRunError: If transform() has not been called.
        """
        self._require_transform("cvss_scores")
        return self.scores["cvss_score"]

    @property
    def cvss_score(self) -> np.ndarray:
        """Best CVSS base score (float32, 0.0-10.0, prefer 3.1 > 4.0 > 3.0 > 2.0).

        Raises:
            TransformNotRunError: If transform() has not been called.
        """
        self._require_transform("cvss_score")
        return self.scores["cvss_score"]

    @property
    def cvss_max(self) -> np.ndarray:
        """Maximum CVSS score across all versions (float32, 0.0-10.0).

        Raises:
            TransformNotRunError: If transform() has not been called.
        """
        self._require_transform("cvss_max")
        return self.scores["cvss_max"]

    @property
    def has_exploit(self) -> np.ndarray:
        """Has known exploits (bool array).

        Raises:
            TransformNotRunError: If transform() has not been called.
        """
        self._require_transform("has_exploit")
        return self.scores["has_exploit"]

    @property
    def cve_vectors(self) -> np.ndarray:
        """CVSS vector strings (object array, e.g. 'CVSS:3.1/AV:N/AC:L/...')."""
        if "cve_vector" in self._cache:
            return self._cache["cve_vector"]
        n = len(self)
        vectors: np.ndarray = np.empty(n, dtype=object)
        for i in range(n):
            cvss_list: list[CVSSScore] = self._source.cvss_scores[i]  # type: ignore[assignment]
            vectors[i] = cvss_list[0].vector if cvss_list else None
        self._cache["cve_vector"] = vectors
        return vectors

    @property
    def attack_vector(self) -> np.ndarray:
        """CVSS Attack Vector values."""
        return (
            self._cvss_metrics.attack_vector if self._cvss_metrics else np.array([], dtype=object)
        )

    @property
    def attack_complexity(self) -> np.ndarray:
        """CVSS Attack Complexity values."""
        return (
            self._cvss_metrics.attack_complexity
            if self._cvss_metrics
            else np.array([], dtype=object)
        )

    @property
    def privileges_required(self) -> np.ndarray:
        """CVSS Privileges Required values."""
        return (
            self._cvss_metrics.privileges_required
            if self._cvss_metrics
            else np.array([], dtype=object)
        )

    @property
    def user_interaction(self) -> np.ndarray:
        """CVSS User Interaction values."""
        return (
            self._cvss_metrics.user_interaction
            if self._cvss_metrics
            else np.array([], dtype=object)
        )

    @property
    def scope(self) -> np.ndarray:
        """CVSS Scope values."""
        return self._cvss_metrics.scope if self._cvss_metrics else np.array([], dtype=object)

    @property
    def confidentiality_impact(self) -> np.ndarray:
        """CVSS Confidentiality Impact values."""
        return (
            self._cvss_metrics.confidentiality_impact
            if self._cvss_metrics
            else np.array([], dtype=object)
        )

    @property
    def integrity_impact(self) -> np.ndarray:
        """CVSS Integrity Impact values."""
        return (
            self._cvss_metrics.integrity_impact
            if self._cvss_metrics
            else np.array([], dtype=object)
        )

    @property
    def availability_impact(self) -> np.ndarray:
        """CVSS Availability Impact values."""
        return (
            self._cvss_metrics.availability_impact
            if self._cvss_metrics
            else np.array([], dtype=object)
        )

    # ==================== ENRICHMENT PROPERTIES ====================

    @property
    def epss(self) -> np.ndarray:
        """EPSS exploitation probability (float32, 0.0-1.0).

        Raises:
            TransformNotRunError: If transform() has not been called.
        """
        self._require_transform("epss")
        return self.scores["epss_probability"]

    @property
    def kev(self) -> np.ndarray:
        """CISA KEV catalog membership (bool array).

        Raises:
            TransformNotRunError: If transform() has not been called.
        """
        self._require_transform("kev")
        return self.scores["kev"]

    @property
    def epss_percentile(self) -> np.ndarray:
        """EPSS percentile rank (float32, 0.0-1.0).

        Raises:
            TransformNotRunError: If transform() has not been called.
        """
        self._require_transform("epss_percentile")
        return self.scores["epss_percentile"]

    @property
    def kev_dates(self) -> np.ndarray:
        """KEV catalog date added (datetime64[s])."""
        if "kev_date" in self._cache:
            return self._cache["kev_date"]
        n = len(self)
        dates: np.ndarray = np.empty(n, dtype="datetime64[s]")
        dates[:] = np.datetime64("NaT")
        for i in range(n):
            kev_entry: Optional[KEVEntry] = self._source.kev[i]  # type: ignore[assignment]
            if kev_entry is not None:
                dates[i] = np.datetime64(kev_entry.added_at, "s")
        self._cache["kev_date"] = dates
        return dates

    # ==================== DATA SCIENCE UX METHODS ====================

    def head(self, n: int = 5) -> "CVDArray":
        """Return first n items as CVDArray.

        Pandas-like convenience method for quick exploration.

        Args:
            n: Number of items (default 5)

        Returns:
            CVDArray with first n items (or all items if array is smaller)

        Example:
            >>> arr = CVDArray.generate(1000)
            >>> arr.head()  # First 5
            >>> arr.head(10)  # First 10
        """
        result = self[:n]
        # slice always returns CVDArray, cast for type checker
        return result  # type: ignore[return-value]

    def describe(self) -> dict[str, Any]:
        """Return summary statistics.

        Pandas-like method for quick data exploration. Returns basic
        statistics without requiring transform(), and richer statistics
        after transform() has been called.

        Returns:
            Dict with count, event rates, and (if transformed) score statistics

        Example:
            >>> arr = CVDArray.generate(100)
            >>> arr.describe()  # Basic stats
            >>> arr.transform().describe()  # Richer stats
        """
        result: dict[str, Any] = {
            "count": len(self),
            "events": self.event_occurrence_rates,
        }

        if self._is_transformed:
            cvss = self.scores["cvss_score"]
            valid_cvss = cvss[~np.isnan(cvss)]
            if len(valid_cvss) > 0:
                result["cvss_mean"] = float(np.mean(valid_cvss))
                result["cvss_std"] = float(np.std(valid_cvss))
                result["cvss_min"] = float(np.min(valid_cvss))
                result["cvss_max"] = float(np.max(valid_cvss))

            epss = self.scores["epss_probability"]
            valid_epss = epss[~np.isnan(epss)]
            if len(valid_epss) > 0:
                result["epss_mean"] = float(np.mean(valid_epss))

            result["kev_count"] = int(self.scores["kev"].sum())
            result["zero_day_count"] = int(self.is_zero_day.sum())

        return result

    def info(self) -> str:
        """Return shape, dtypes, memory usage info.

        Pandas-like method for inspecting array structure.

        Returns:
            Formatted string with array information

        Example:
            >>> arr = CVDArray.generate(1000)
            >>> print(arr.info())
        """
        lines = [
            f"CVDArray: {len(self)} vulnerabilities",
            f"Fixed size: {self._fixed_size}",
            f"Transformed: {self._is_transformed}",
            f"Stale: {self._stale}",
        ]

        # Memory estimate (~2KB per vuln as per design doc)
        mem_kb = len(self) * 2
        if mem_kb > 1024:
            lines.append(f"Memory: ~{mem_kb // 1024} MB")
        else:
            lines.append(f"Memory: ~{mem_kb} KB")

        return "\n".join(lines)
