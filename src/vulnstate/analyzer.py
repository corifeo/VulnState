"""
CVD Analyzer - Vectorized analytics computation for arrays

Provides:
- CVDAnalyzer: Single class for all array-level analytics
  - Fix path (VFD dimension)
  - Threat state (PXA dimension)
  - Pair analysis (validity, desiderata)
  - Inference engine
  - Duration metrics

Layer: Analytics
Dependencies: constants.py, models.py
Used by: array.py
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .array import CVDArray
    from .constants import CVDEvent
    from .models import AnalysisResult


class CVDAnalyzer:
    """
    Vectorized analytics for CVDArray.

    Loads array context once, provides all analytics as properties.
    Complex logic implemented for arrays; single-item wraps in 1-element array.

    Example:
        arr = CVDArray(vulnerabilities)
        analyzer = CVDAnalyzer(arr)

        valid_mask = analyzer.validity == HistoryValidity.VALID
        weaponized = analyzer.threat_state == 0b011
    """

    def __init__(self, arr: "CVDArray"):
        """
        Initialize analyzer with array context.

        Args:
            arr: CVDArray to analyze
        """
        self._arr = arr
        self.states = arr.state_ints
        self.timestamps = arr.timestamps
        self.n = len(arr)

        # Lazy caches
        self._fix_path: Optional[np.ndarray] = None
        self._threat_state: Optional[np.ndarray] = None
        self._pair_mask: Optional[np.ndarray] = None
        self._validity: Optional[np.ndarray] = None
        self._inferred_mask: Optional[np.ndarray] = None

    @property
    def fix_path(self) -> np.ndarray:
        """
        VFD dimension as uint8 array.

        Values: 0b000 (No Awareness), 0b001 (Vendor Aware),
                0b011 (Fix Ready), 0b111 (Remediated)
        """
        if self._fix_path is None:
            # Extract bits 0-2 (V, F, D)
            self._fix_path = (self.states & 0b000111).astype(np.uint8)
        return self._fix_path

    @property
    def threat_state(self) -> np.ndarray:
        """
        PXA dimension as uint8 array.

        Values: 0b000 (Latent), 0b001 (Disclosed), 0b011 (Weaponized),
                0b100 (Targeted), 0b101 (Under Attack), 0b111 (Active Threat)
        """
        if self._threat_state is None:
            # Extract bits 3-5 (P, X, A) and shift right
            self._threat_state = ((self.states >> 3) & 0b000111).astype(np.uint8)
        return self._threat_state

    @property
    def pair_mask(self) -> np.ndarray:
        """
        Pair analysis as uint16 bitmask array.

        Bit i is set if pair i is satisfied (earlier < later).
        See PAIR_BIT_POSITIONS for mapping.
        """
        if self._pair_mask is None:
            self._pair_mask = self._compute_pair_mask()
        return self._pair_mask

    def _compute_pair_mask(self) -> np.ndarray:
        """Compute pair satisfaction for all 15 pairs."""
        from .constants import PAIR_BIT_POSITIONS, CVDEvent

        result = np.zeros(self.n, dtype=np.uint16)

        # Get timestamp arrays
        ts = {
            CVDEvent.V: self.timestamps.V,
            CVDEvent.F: self.timestamps.F,
            CVDEvent.D: self.timestamps.D,
            CVDEvent.P: self.timestamps.P,
            CVDEvent.X: self.timestamps.X,
            CVDEvent.A: self.timestamps.A,
        }

        for (e1, e2), bit_pos in PAIR_BIT_POSITIONS.items():
            t1 = ts[e1]
            t2 = ts[e2]

            # Pair satisfied if:
            # - Either event missing (NaT), OR
            # - t1 < t2 (correct order)
            missing = np.isnat(t1) | np.isnat(t2)
            correct_order = t1 < t2
            satisfied = missing | correct_order

            # Set bit where satisfied
            result |= np.where(satisfied, 1 << bit_pos, 0).astype(np.uint16)

        return result  # type: ignore[no-any-return]

    @property
    def validity(self) -> np.ndarray:
        """
        History validity as uint8 enum array.

        Values: HistoryValidity.VALID (0), HistoryValidity.IMPOSSIBLE (1)
        """
        if self._validity is None:
            self._validity = self._compute_validity()
        return self._validity

    def _compute_validity(self) -> np.ndarray:
        """Check if required pairs are satisfied."""
        from .constants import REQUIRED_PAIRS_MASK, HistoryValidity

        pair_mask = self.pair_mask

        # Required pairs must be satisfied (bits set)
        required_satisfied = (pair_mask & REQUIRED_PAIRS_MASK) == REQUIRED_PAIRS_MASK

        return np.where(  # type: ignore[no-any-return]
            required_satisfied, HistoryValidity.VALID, HistoryValidity.IMPOSSIBLE
        ).astype(np.uint8)

    @property
    def inferred_mask(self) -> np.ndarray:
        """
        Inference provenance as uint8 bitmask array.

        Bit i set if event i was inferred (not observed).
        Same bit layout as state_encoded.
        """
        if self._inferred_mask is None:
            # Initialize to zero (nothing inferred yet)
            self._inferred_mask = np.zeros(self.n, dtype=np.uint8)
        return self._inferred_mask

    @property
    def desiderata_score(self) -> np.ndarray:
        """
        Count of satisfied desiderata (0-12) as uint8 array.

        Counts how many of the 12 desiderata pairs are satisfied.
        A pair is satisfied if the earlier event occurs before the later,
        or if either event is missing (NaT).
        """
        from .constants import DESIDERATA_PAIRS_BITS

        pair_mask = self.pair_mask
        scores = np.zeros(self.n, dtype=np.uint8)
        for bit in DESIDERATA_PAIRS_BITS:
            scores += ((pair_mask >> bit) & 1).astype(np.uint8)
        return scores  # type: ignore[no-any-return]

    @property
    def fix_path_labels(self) -> np.ndarray:
        """
        Human-readable fix path labels as object array.

        Values: "No Awareness", "Vendor Aware", "Fix Ready", "Remediated"
        """
        from .constants import FIX_PATH_LABELS

        return np.array(  # type: ignore[no-any-return]
            [FIX_PATH_LABELS.get(int(fp), "Unknown") for fp in self.fix_path],
            dtype=object,
        )

    @property
    def threat_labels(self) -> np.ndarray:
        """
        Human-readable threat state labels as object array.

        Values: "Latent", "Disclosed", "Weaponized", "Targeted",
                "Under Attack", "Active Threat"
        """
        from .constants import THREAT_LABELS

        return np.array(  # type: ignore[no-any-return]
            [THREAT_LABELS.get(int(ts), "Unknown") for ts in self.threat_state],
            dtype=object,
        )

    @staticmethod
    def explain_pair_violations(pair_mask: int) -> list[str]:
        """
        Return list of violated required pair names.

        Args:
            pair_mask: uint16 bitmask from pair_mask property

        Returns:
            List of pair names (e.g., ["V≺F", "F≺D"]) that are violated

        Example:
            >>> mask = 0b0000_0000_0010_0001  # V≺F and F≺D satisfied
            >>> CVDAnalyzer.explain_pair_violations(mask)
            ['V≺D']  # Only V≺D violated
        """
        from .constants import PAIR_NAMES, REQUIRED_PAIRS_MASK

        violations = ~pair_mask & REQUIRED_PAIRS_MASK
        return [PAIR_NAMES[i] for i in range(15) if violations & (1 << i)]

    def apply_inferences(self) -> "CVDAnalyzer":
        """
        Apply inference rules to fill missing timestamps.

        Mutates array timestamps in place. Tracks provenance in inferred_mask.

        Rules:
        - P without V → infer V at P timestamp
        - X without P → infer P at X timestamp
        - F without V → infer V at F timestamp
        - D without F → infer F at D timestamp

        Returns:
            self (for chaining)
        """
        from .constants import CVDEvent

        # Ensure inferred_mask is initialized
        _ = self.inferred_mask

        # Rule 1: P implies V (vendor learns from public)
        self._infer_event(source=CVDEvent.P, target=CVDEvent.V, target_bit=0)

        # Rule 2: X implies P (exploit publication = disclosure)
        self._infer_event(source=CVDEvent.X, target=CVDEvent.P, target_bit=3)

        # Rule 3: F implies V (causality)
        self._infer_event(source=CVDEvent.F, target=CVDEvent.V, target_bit=0)

        # Rule 4: D implies F (causality)
        self._infer_event(source=CVDEvent.D, target=CVDEvent.F, target_bit=1)

        # Invalidate caches that depend on timestamps
        self._pair_mask = None
        self._validity = None

        return self

    def _infer_event(self, source: "CVDEvent", target: "CVDEvent", target_bit: int) -> None:
        """Infer target event from source event timestamp."""

        source_ts = getattr(self.timestamps, source.name)
        target_ts = getattr(self.timestamps, target.name)

        # Where source exists but target missing
        infer_mask = ~np.isnat(source_ts) & np.isnat(target_ts)

        if not infer_mask.any():
            return

        # Copy source timestamp to target
        target_ts[infer_mask] = source_ts[infer_mask]

        # Update inferred_mask (guaranteed initialized by apply_inferences)
        assert self._inferred_mask is not None
        self._inferred_mask[infer_mask] |= 1 << target_bit

        # Update array states
        self._arr.state_ints[infer_mask] |= 1 << target_bit

    @staticmethod
    def analyze(arr: "CVDArray", infer: bool = True) -> "AnalysisResult":
        """
        Compute full analysis for an array.

        Optionally applies inference rules first (idempotent), then
        computes all analytics. Returns AnalysisResult dataclass with
        all 11 groups of analytics.

        Inference rules (when infer=True):
        - P without V → infer V at P timestamp (vendor learns from disclosure)
        - X without P → infer P at X timestamp (exploit = disclosure)
        - F without V → infer V at F timestamp (fix implies awareness)
        - D without F → infer F at D timestamp (deploy implies fix)

        Args:
            arr: CVDArray to analyze
            infer: If True (default), apply inference rules before analysis.
                   If False, analyze raw data without inferring missing events.

        Returns:
            AnalysisResult with computed analytics
        """
        from .models import AnalysisResult

        analyzer = CVDAnalyzer(arr)

        # Apply inference rules if enabled (idempotent - only fills missing timestamps)
        if infer:
            analyzer.apply_inferences()

        n = analyzer.n
        ts = analyzer.timestamps

        # Get timestamp arrays for comparisons
        V_ts = ts.V
        F_ts = ts.F
        D_ts = ts.D
        P_ts = ts.P
        X_ts = ts.X
        A_ts = ts.A

        # Helper: check if timestamp exists (not NaT)
        def has(t: np.ndarray) -> np.ndarray:
            return ~np.isnat(t)

        # Helper: check t1 < t2 (True if either missing or t1 < t2)
        def before(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
            both_exist = has(t1) & has(t2)
            return ~both_exist | (t1 < t2)

        # Helper: compute days between timestamps (NaN if either missing)
        def days_between(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
            result = np.full(n, np.nan, dtype=np.float32)
            both_exist = has(t1) & has(t2)
            delta = t2[both_exist].astype("datetime64[us]").astype(np.int64) - t1[
                both_exist
            ].astype("datetime64[us]").astype(np.int64)
            result[both_exist] = delta / (1e6 * 86400)  # microseconds to days
            return result

        # ==================== Group 1: Validity & Completeness ====================
        validity_int = analyzer.validity
        event_count = np.zeros(n, dtype=np.uint8)
        for event_ts in [V_ts, F_ts, D_ts, P_ts, X_ts, A_ts]:
            event_count += has(event_ts).astype(np.uint8)
        is_complete = event_count == 6

        # ==================== Group 2: Pair Analysis ====================
        pair_mask = analyzer.pair_mask
        # Desiderata mask: bits for the 12 desiderata pairs
        desiderata_mask = pair_mask  # Use full pair mask for now
        anti_desiderata_mask = ~pair_mask & 0x0FFF  # Inverse of desiderata
        # Observed mask: which pairs have both events
        pair_observed_mask = np.zeros(n, dtype=np.uint16)
        from .constants import PAIR_BIT_POSITIONS, CVDEvent

        ts_dict = {
            CVDEvent.V: V_ts,
            CVDEvent.F: F_ts,
            CVDEvent.D: D_ts,
            CVDEvent.P: P_ts,
            CVDEvent.X: X_ts,
            CVDEvent.A: A_ts,
        }
        for (e1, e2), bit_pos in PAIR_BIT_POSITIONS.items():
            both_observed = has(ts_dict[e1]) & has(ts_dict[e2])
            pair_observed_mask |= np.where(both_observed, 1 << bit_pos, 0).astype(np.uint16)

        # ==================== Group 3: Zero-Day Indicators ====================
        # Zero-day exploit: X before V
        is_zero_day_exploit = has(X_ts) & (~has(V_ts) | (X_ts < V_ts))
        # Zero-day attack: A before V
        is_zero_day_attack = has(A_ts) & (~has(V_ts) | (A_ts < V_ts))
        # Either type of zero-day
        is_zero_day = is_zero_day_exploit | is_zero_day_attack

        # ==================== Group 4: Coordination Quality ====================
        # Coordinated: V before P (vendor aware before disclosure)
        is_coordinated = before(V_ts, P_ts) & has(V_ts) & has(P_ts)
        # Premature disclosure: P before F
        is_premature_disclosure = has(P_ts) & (~has(F_ts) | (P_ts < F_ts))
        # Responsible disclosure: V before P before F
        is_responsible_disclosure = is_coordinated & before(P_ts, F_ts) & has(F_ts)

        # ==================== Group 5: Fix Effectiveness ====================
        has_fix_before_exploit = before(F_ts, X_ts) & has(F_ts)
        has_fix_before_attack = before(F_ts, A_ts) & has(F_ts)
        has_deployment_before_exploit = before(D_ts, X_ts) & has(D_ts)
        has_deployment_before_attack = before(D_ts, A_ts) & has(D_ts)

        # ==================== Group 6: Threat Characteristics ====================
        # Private attack: A without X (attack without public exploit)
        is_private_attack = has(A_ts) & ~has(X_ts)
        # Weaponized: X occurred
        is_weaponized = has(X_ts)
        # Mass exploitation: both X and A
        is_mass_exploitation = has(X_ts) & has(A_ts)

        # ==================== Group 7: Classification ====================
        fix_path_int = analyzer.fix_path
        threat_state_int = analyzer.threat_state
        # Outcome classification (simplified: based on final state)
        outcome_int = np.zeros(n, dtype=np.uint8)  # Placeholder
        # First event classification
        first_event_int = np.zeros(n, dtype=np.uint8)  # Placeholder
        disclosure_pattern_int = np.zeros(n, dtype=np.uint8)  # Placeholder

        # ==================== Group 8: Scores & Counts ====================
        desiderata_count = analyzer.desiderata_score
        desiderata_score_float = (desiderata_count / 12.0).astype(np.float32)
        skill_score = np.zeros(n, dtype=np.float32)  # Placeholder

        # ==================== Group 9: Time Deltas ====================
        fix_lag_days = days_between(V_ts, F_ts)
        deployment_lag_days = days_between(F_ts, D_ts)
        disclosure_window_days = days_between(V_ts, P_ts)
        exploit_window_days = days_between(P_ts, X_ts)
        attack_window_days = days_between(P_ts, A_ts)
        exposure_days = days_between(P_ts, D_ts)  # Time from disclosure to deployment
        exploit_to_attack_days = days_between(X_ts, A_ts)

        # ==================== Group 10: Transitions ====================
        # Can transition if event hasn't occurred yet
        can_transition_mask = ~arr.state_ints & 0b111111

        # ==================== Group 11: Probabilities ====================
        # Placeholder - these would need EPSS or other external data
        prob_X = np.zeros(n, dtype=np.float32)
        prob_A = np.zeros(n, dtype=np.float32)
        prob_threat = np.zeros(n, dtype=np.float32)

        return AnalysisResult(
            # Group 1
            validity_int=validity_int,
            event_count=event_count,
            is_complete=is_complete,
            # Group 2
            desiderata_mask=desiderata_mask,
            anti_desiderata_mask=anti_desiderata_mask,
            pair_observed_mask=pair_observed_mask,
            # Group 3
            is_zero_day=is_zero_day,
            is_zero_day_exploit=is_zero_day_exploit,
            is_zero_day_attack=is_zero_day_attack,
            # Group 4
            is_coordinated=is_coordinated,
            is_premature_disclosure=is_premature_disclosure,
            is_responsible_disclosure=is_responsible_disclosure,
            # Group 5
            has_fix_before_exploit=has_fix_before_exploit,
            has_fix_before_attack=has_fix_before_attack,
            has_deployment_before_exploit=has_deployment_before_exploit,
            has_deployment_before_attack=has_deployment_before_attack,
            # Group 6
            is_private_attack=is_private_attack,
            is_weaponized=is_weaponized,
            is_mass_exploitation=is_mass_exploitation,
            # Group 7
            fix_path_int=fix_path_int,
            threat_state_int=threat_state_int,
            outcome_int=outcome_int,
            first_event_int=first_event_int,
            disclosure_pattern_int=disclosure_pattern_int,
            # Group 8
            desiderata_count=desiderata_count,
            desiderata_score=desiderata_score_float,
            skill_score=skill_score,
            # Group 9
            fix_lag_days=fix_lag_days,
            deployment_lag_days=deployment_lag_days,
            disclosure_window_days=disclosure_window_days,
            exploit_window_days=exploit_window_days,
            attack_window_days=attack_window_days,
            exposure_days=exposure_days,
            exploit_to_attack_days=exploit_to_attack_days,
            # Group 10
            can_transition_mask=can_transition_mask,
            # Group 11
            prob_X=prob_X,
            prob_A=prob_A,
            prob_threat=prob_threat,
            # Group 12: Provenance (only populated if infer=True)
            inferred_mask=analyzer.inferred_mask if infer else np.zeros(n, dtype=np.uint8),
        )
