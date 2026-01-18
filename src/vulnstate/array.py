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
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import pandas as pd

    from .models import AnalysisResult

from .constants import (
    CVDEvent,
    get_all_valid_states,
    get_state_label,
    state_int_to_string,
    string_to_state_int,
)
from .models import ArrayAnalytics, ArrayCoreData, ArrayMetadata, ArrayTimestamps
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

    def __init__(self, vulnerabilities: Optional[list[CVDVulnerability]] = None):
        """
        Initialize array from list of vulnerabilities or create empty.

        Args:
            vulnerabilities: List of CVDVulnerability objects, or None for empty array
        """
        # Data components (dataclasses)
        self.core = ArrayCoreData()
        self.timestamps = ArrayTimestamps()
        self.analytics = ArrayAnalytics()
        self._metadata_data = ArrayMetadata()

        # Metadata encoders (for categorical optimization - not yet implemented)
        self._metadata_encoders: dict[str, Any] = {}

        # Dirty tracking
        self._dirty_indices: set[int] = set()

        # Analysis cache (computed on first access)
        self._analysis_cache: Optional[AnalysisResult] = None

        if vulnerabilities:
            self._from_list(vulnerabilities)
        else:
            self._init_empty()

    def _init_empty(self) -> None:
        """Initialize empty arrays (dataclasses already initialized with empty arrays)."""
        # Dataclasses (self.core, self.timestamps, self.analytics, self.metadata)
        # are already initialized with empty arrays in __init__
        pass

    def _from_list(self, vulnerabilities: list[CVDVulnerability]) -> None:
        """Build arrays from list of vulnerabilities."""
        n = len(vulnerabilities)

        # Core data
        self.core.vulnerabilities = np.array(vulnerabilities, dtype=object)
        self.core.states = np.array(
            [v.event_data.state_encoded for v in vulnerabilities], dtype=np.uint8
        )
        self.core.vuln_ids = np.array([v.identity.vuln_id for v in vulnerabilities], dtype=object)

        # Initialize timestamp arrays (will be filled by sync)
        self.timestamps.V = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.F = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.D = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.P = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.X = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")
        self.timestamps.A = np.full(n, np.datetime64("NaT"), dtype="datetime64[us]")

        # Initialize analytics arrays (will be filled by sync)
        self.analytics.severities = np.empty(n, dtype=object)
        self.analytics.fix_path = np.empty(n, dtype=np.uint8)
        self.analytics.threat_state = np.empty(n, dtype=np.uint8)
        self.analytics.is_zero_day = np.zeros(n, dtype=bool)
        self.analytics.is_fix_available = np.zeros(n, dtype=bool)
        self.analytics.is_fix_deployed = np.zeros(n, dtype=bool)
        self.analytics.has_public_exploit = np.zeros(n, dtype=bool)
        self.analytics.is_under_attack = np.zeros(n, dtype=bool)
        self.analytics.premature_disclosure = np.zeros(n, dtype=bool)
        self.analytics.disclosure_window_days = np.full(n, np.nan, dtype=np.float32)
        self.analytics.fix_lag_days = np.full(n, np.nan, dtype=np.float32)
        self.analytics.deployment_lag_days = np.full(n, np.nan, dtype=np.float32)
        self.analytics.violated_orderings_count = np.zeros(n, dtype=np.int32)
        self.analytics.desiderata_scores = np.zeros(n, dtype=np.float32)
        self.analytics.skill_scores = np.full(n, np.nan, dtype=np.float32)

        # Trigger initial sync to populate arrays
        self._dirty_indices = set(range(n))
        self.sync()

        # Event timestamps as parallel arrays (backward compatibility dict)
        self._metadata_data.event_timestamps_absolute = {}
        for event in CVDEvent:
            timestamps = []
            for v in vulnerabilities:
                ts = v.event_data.events.get(event)
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

        # Extract EPSS scores from enrichment dataclass
        epss_scores = [
            v.enrichment.epss if v.enrichment.epss is not None else np.nan for v in vulnerabilities
        ]
        self._metadata_data.raw["epss"] = np.array(epss_scores, dtype=np.float32)

        # Extract is_kev flags from enrichment dataclass
        is_kev_flags = [v.enrichment.is_kev for v in vulnerabilities]
        self._metadata_data.raw["is_kev"] = np.array(is_kev_flags, dtype=bool)

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

    # ==================== REPRESENTATION ====================

    def __len__(self) -> int:
        """Number of vulnerabilities in array."""
        return len(self.states)

    def __repr__(self) -> str:
        """Developer-friendly representation."""
        unique_states = len(self.count_by_state()) if len(self) > 0 else 0
        return f"CVDArray(n={len(self)}, unique_states={unique_states})"

    def __str__(self) -> str:
        """User-friendly string (shows summary)."""
        return self.summary

    # ==================== DATACLASS PROPERTY ACCESSORS ====================

    @property
    def states(self) -> np.ndarray:
        """Get states array."""
        return self.core.states

    @states.setter
    def states(self, value: np.ndarray) -> None:
        """Set states array."""
        self.core.states = value

    @property
    def vuln_ids(self) -> np.ndarray:
        """Get vulnerability IDs array."""
        return self.core.vuln_ids

    @vuln_ids.setter
    def vuln_ids(self, value: np.ndarray) -> None:
        """Set vulnerability IDs array."""
        self.core.vuln_ids = value

    @property
    def _vulnerabilities(self) -> np.ndarray:
        """Get live vulnerability objects array."""
        return self.core.vulnerabilities

    @_vulnerabilities.setter
    def _vulnerabilities(self, value: np.ndarray) -> None:
        """Set live vulnerability objects array."""
        self.core.vulnerabilities = value

    @property
    def V_timestamps(self) -> np.ndarray:
        """Get V event timestamps."""
        return self.timestamps.V

    @property
    def F_timestamps(self) -> np.ndarray:
        """Get F event timestamps."""
        return self.timestamps.F

    @property
    def D_timestamps(self) -> np.ndarray:
        """Get D event timestamps."""
        return self.timestamps.D

    @property
    def P_timestamps(self) -> np.ndarray:
        """Get P event timestamps."""
        return self.timestamps.P

    @property
    def X_timestamps(self) -> np.ndarray:
        """Get X event timestamps."""
        return self.timestamps.X

    @property
    def A_timestamps(self) -> np.ndarray:
        """Get A event timestamps."""
        return self.timestamps.A

    @property
    def severities(self) -> np.ndarray:
        """Get severities analytics array.

        Note: Severities are stored in ArrayAnalytics, not computed by CVDAnalyzer.
        """
        return self.analytics.severities

    @property
    def fix_path(self) -> np.ndarray:
        """Get fix path as uint8 array (FixPath enum values).

        Reads from AnalysisResult.fix_path_int computed by CVDAnalyzer.
        """
        return self.analysis.fix_path_int

    @property
    def threat_state(self) -> np.ndarray:
        """Get threat state as uint8 array (ThreatState enum values).

        Reads from AnalysisResult.threat_state_int computed by CVDAnalyzer.
        """
        return self.analysis.threat_state_int

    @property
    def cubes(self) -> np.ndarray:
        """Deprecated: Use fix_path instead."""
        import warnings

        warnings.warn("cubes is deprecated, use fix_path", DeprecationWarning, stacklevel=2)
        return self.fix_path

    @property
    def is_zero_day(self) -> np.ndarray:
        """Get is_zero_day analytics array.

        Reads from AnalysisResult.is_zero_day computed by CVDAnalyzer.
        True if exploit (X) or attack (A) occurred before vendor awareness (V).
        """
        return self.analysis.is_zero_day

    @property
    def is_fix_available(self) -> np.ndarray:
        """Check if fix is available (F event occurred).

        Derived from state bits: True if F bit is set.
        """
        return self.has_event_occurred(CVDEvent.F)

    @property
    def is_fix_deployed(self) -> np.ndarray:
        """Check if fix is deployed (D event occurred).

        Derived from state bits: True if D bit is set.
        """
        return self.has_event_occurred(CVDEvent.D)

    @property
    def has_public_exploit(self) -> np.ndarray:
        """Check if public exploit exists (X event occurred).

        Reads from AnalysisResult.is_weaponized computed by CVDAnalyzer.
        """
        return self.analysis.is_weaponized

    @property
    def is_under_attack(self) -> np.ndarray:
        """Check if under active attack (A event occurred).

        Derived from state bits: True if A bit is set.
        """
        return self.has_event_occurred(CVDEvent.A)

    @property
    def premature_disclosure(self) -> np.ndarray:
        """Get premature_disclosure analytics array.

        Reads from AnalysisResult.is_premature_disclosure computed by CVDAnalyzer.
        True if P occurred before F (disclosure before fix ready).
        """
        return self.analysis.is_premature_disclosure

    @property
    def disclosure_window_days(self) -> np.ndarray:
        """Get disclosure_window_days analytics array.

        Reads from AnalysisResult.disclosure_window_days computed by CVDAnalyzer.
        Days between V (vendor awareness) and P (public disclosure).
        """
        return self.analysis.disclosure_window_days

    @property
    def fix_lag_days(self) -> np.ndarray:
        """Get fix_lag_days analytics array.

        Reads from AnalysisResult.fix_lag_days computed by CVDAnalyzer.
        Days between V (vendor awareness) and F (fix ready).
        """
        return self.analysis.fix_lag_days

    @property
    def deployment_lag_days(self) -> np.ndarray:
        """Get deployment_lag_days analytics array.

        Reads from AnalysisResult.deployment_lag_days computed by CVDAnalyzer.
        Days between F (fix ready) and D (fix deployed).
        """
        return self.analysis.deployment_lag_days

    @property
    def violated_orderings_count(self) -> np.ndarray:
        """Get count of violated ordering constraints.

        Computed from anti_desiderata_mask by counting set bits.
        """
        # Count bits set in anti_desiderata_mask
        mask = self.analysis.anti_desiderata_mask
        count = np.zeros(len(mask), dtype=np.int32)
        for i in range(12):  # 12 desiderata pairs
            count += ((mask >> i) & 1).astype(np.int32)
        return count

    @property
    def desiderata_scores(self) -> np.ndarray:
        """Get desiderata_scores analytics array (0.0 to 1.0).

        Reads from AnalysisResult.desiderata_score computed by CVDAnalyzer.
        Fraction of satisfied desiderata (desiderata_count / 12).
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
        """Get anti-desiderata (violations) as uint16 bitmask array.

        Reads from AnalysisResult.anti_desiderata_mask computed by CVDAnalyzer.
        Each bit corresponds to an AntiDesiderataBit.
        """
        return self.analysis.anti_desiderata_mask

    @property
    def skill_scores(self) -> np.ndarray:
        """Get skill_scores analytics array.

        Reads from AnalysisResult.skill_score computed by CVDAnalyzer.
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
        """Shape of array (numpy convention)."""
        return (len(self),)

    @property
    def ndim(self) -> int:
        """Number of dimensions (always 1 for CVDArray)."""
        return 1

    @property
    def size(self) -> int:
        """Total number of elements (numpy convention)."""
        return len(self)

    @property
    def event_timestamps(self) -> dict[CVDEvent, np.ndarray]:
        """
        DEPRECATED: Use exploded timestamp arrays instead.

        .. deprecated:: 2026-01-13
            Use direct array access instead:
            - arr.V_timestamps instead of arr.event_timestamps[CVDEvent.V]
            - arr.F_timestamps instead of arr.event_timestamps[CVDEvent.F]
            - arr.D_timestamps instead of arr.event_timestamps[CVDEvent.D]
            - arr.P_timestamps instead of arr.event_timestamps[CVDEvent.P]
            - arr.X_timestamps instead of arr.event_timestamps[CVDEvent.X]
            - arr.A_timestamps instead of arr.event_timestamps[CVDEvent.A]

        Returns:
            Dict[CVDEvent, np.ndarray] with absolute datetime64[us] arrays.
        """
        warnings.warn(
            "event_timestamps dict is deprecated. Use exploded timestamp arrays "
            "(V_timestamps, F_timestamps, D_timestamps, P_timestamps, "
            "X_timestamps, A_timestamps) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._event_timestamps_absolute

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
            subset.states = self.states[idx]
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
        return (self.states & (1 << bit_pos)) != 0

    @property
    def terminal_mask(self) -> np.ndarray:
        """
        Boolean mask indicating which vulnerabilities are in terminal state (VFDPXA).

        Returns:
            Boolean array of shape (N,)
        """
        return self.states == 0b111111

    @property
    def states_as_strings(self) -> np.ndarray:
        """
        All states as string array.

        Returns:
            np.ndarray of shape (N,) with dtype=object, values are state strings
        """
        return np.array([state_int_to_string(s) for s in self.states], dtype=object)

    def count_by_state(self) -> dict[str, int]:
        """
        Count vulnerabilities by current state.

        Returns:
            Dictionary mapping state strings to counts
        """
        unique_states, counts = np.unique(self.states, return_counts=True)
        return {
            state_int_to_string(state): int(count) for state, count in zip(unique_states, counts)
        }

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
        state_strings = self.states_as_strings
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

        Syncs state and timestamps from live vulnerability objects to cached arrays.

        Args:
            indices: Optional specific indices to sync (uses _dirty_indices if None)

        Returns:
            self (for chaining)
        """
        # Determine which indices to sync
        to_sync = set(indices) if indices is not None else self._dirty_indices.copy()

        # Sync each dirty index
        for idx in to_sync:
            vuln = self.core.vulnerabilities[idx]

            # Update state and vuln_id
            self.core.states[idx] = vuln.event_data.state_encoded
            self.core.vuln_ids[idx] = vuln.identity.vuln_id

            # Extract timestamps to exploded arrays
            for event, attr_name in [
                (CVDEvent.V, "V"),
                (CVDEvent.F, "F"),
                (CVDEvent.D, "D"),
                (CVDEvent.P, "P"),
                (CVDEvent.X, "X"),
                (CVDEvent.A, "A"),
            ]:
                ts = vuln.event_data.events.get(event)
                if ts:
                    getattr(self.timestamps, attr_name)[idx] = np.datetime64(ts, "us")
                else:
                    getattr(self.timestamps, attr_name)[idx] = np.datetime64("NaT")

        # Clear dirty flags
        self._dirty_indices -= to_sync

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
        """Invalidate the cached analysis (called after modifications)."""
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
        self.states[mask] |= 1 << bit_pos

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
        arr.states = states
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
        explode_cvss: bool = True,
        explode_metadata: bool = True,
    ) -> "pd.DataFrame":
        """
        Convert array to pandas DataFrame with comprehensive data.

        Args:
            include_analytics: Include computed metrics (default True)
            explode_cvss: CVSS vector as separate columns (default True)
            explode_metadata: Unpack metadata dicts into columns (default True)

        Returns:
            pandas DataFrame with one row per vulnerability

        Example:
            >>> df = arr.to_dataframe()
            >>> df = arr.to_dataframe(explode_metadata=False)
        """
        from vulnstate.io import CVDIO

        return CVDIO.to_dataframe(self, include_analytics, explode_cvss, explode_metadata)

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

    def import_nvd_file(self, filepath: str) -> None:
        """
        Import NVD data from JSON file (convenience wrapper).

        Parses NVD Feed 1.1 format and delegates to CVDIO.import_nvdcve().

        Args:
            filepath: Path to NVD JSON file

        Example:
            >>> arr.import_nvd_file('data/nvdcve-1.1-2024.json')
        """
        import json
        from typing import Any

        from .io import CVDIO

        with open(filepath) as f:
            nvd_feed = json.load(f)

        nvd_data: dict[str, dict[str, Any]] = {}
        for item in nvd_feed.get("CVE_Items", []):
            cve_id = item["cve"]["CVE_data_meta"]["ID"]
            nvd_info: dict[str, Any] = {}

            impact = item.get("impact", {})
            if "baseMetricV3" in impact:
                cvss_v3 = impact["baseMetricV3"]["cvssV3"]
                nvd_info["cvss_score"] = cvss_v3["baseScore"]
                nvd_info["cve_vector"] = cvss_v3.get("vectorString", "")
            elif "baseMetricV2" in impact:
                cvss_v2 = impact["baseMetricV2"]["cvssV2"]
                nvd_info["cvss_score"] = cvss_v2["baseScore"]
                nvd_info["cve_vector"] = cvss_v2.get("vectorString", "")

            if nvd_info:
                nvd_data[cve_id] = nvd_info

        CVDIO.import_nvdcve(self, nvd_data)

    # ==================== FACTORY METHODS ====================

    @classmethod
    def zeros(cls, n: int, vuln_id_prefix: Optional[str] = None) -> "CVDArray":
        """
        Create array of n vulnerabilities in initial state (vfdpxa).

        All vulnerabilities start in the initial 'vfdpxa' state with no events applied.

        Args:
            n: Number of vulnerabilities to create
            vuln_id_prefix: Optional prefix for auto-generated CVE IDs (e.g., 'ZERO')
                If provided, creates IDs like ZERO-00000, ZERO-00001, etc.

        Returns:
            CVDArray with n vulnerabilities in vfdpxa state

        Example:
            >>> arr = CVDArray.zeros(100)  # 100 initial vulnerabilities
            >>> arr = CVDArray.zeros(50, vuln_id_prefix='INIT')  # INIT-00000, INIT-00001, ...
        """
        vulns = []
        for i in range(n):
            cve_id = f"{vuln_id_prefix}-{i:05d}" if vuln_id_prefix else None
            # Create with no awareness flags (defaults to initial state vfdpxa)
            vuln = CVDVulnerability(cve_id=cve_id)
            vulns.append(vuln)
        return cls(vulns)

    @classmethod
    def ones(cls, n: int, vuln_id_prefix: Optional[str] = None) -> "CVDArray":
        """
        Create array of n vulnerabilities in terminal state (VFDPXA).

        All vulnerabilities start with all events announced (VFDPXA state).
        This is useful for testing and scenarios where all phases are complete.

        Args:
            n: Number of vulnerabilities to create
            vuln_id_prefix: Optional prefix for auto-generated CVE IDs (e.g., 'TERM')
                If provided, creates IDs like TERM-00000, TERM-00001, etc.

        Returns:
            CVDArray with n vulnerabilities in VFDPXA state

        Example:
            >>> arr = CVDArray.ones(100)  # 100 terminal state vulnerabilities
            >>> arr = CVDArray.ones(50, vuln_id_prefix='END')  # END-00000, END-00001, ...
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
        return cls(vulns)

    @classmethod
    def random(
        cls, n: int, vuln_id_prefix: Optional[str] = None, seed: Optional[int] = None
    ) -> "CVDArray":
        """
        Create array of n vulnerabilities with random valid states.

        Each vulnerability is assigned a random valid state from the 32 possible CVD states.
        Useful for testing and simulations.

        Args:
            n: Number of vulnerabilities to create
            vuln_id_prefix: Optional prefix for auto-generated CVE IDs (e.g., 'RAND')
                If provided, creates IDs like RAND-00000, RAND-00001, etc.
            seed: Optional random seed for reproducibility

        Returns:
            CVDArray with n vulnerabilities in random valid states

        Example:
            >>> arr = CVDArray.random(1000)  # 1000 random vulnerabilities
            >>> arr = CVDArray.random(100, vuln_id_prefix='RND', seed=42)  # Reproducible random
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
        return cls(vulns)
