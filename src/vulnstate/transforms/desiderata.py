"""Desiderata analysis transform.

Provides:
- DesiderataExtractor: Full CVD desiderata analysis (pair_mask, validity, scores)

Absorbs functionality from former analyzer.py.

Layer: Analytics
Dependencies: constants.py, models.py
Used by: array.py (via transform registry)
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from vulnstate.array import CVDArray
    from vulnstate.constants import CVDEvent
    from vulnstate.models import AnalysisResult


class DesiderataExtractor:
    """Full CVD desiderata analysis.

    Computes pair_mask, validity, desiderata scores, and inference.
    Implements the Transform protocol for lazy dispatch via arr.desiderata.

    Example:
        >>> arr = CVDArray.generate(100)
        >>> desiderata = arr.desiderata  # Lazy via transform registry
        >>> desiderata.validity  # np.ndarray of validity enum values
        >>> desiderata.desiderata_score  # 0-12 score per vulnerability
    """

    name: str = "desiderata"

    def __init__(self, arr: "CVDArray"):
        """Initialize with array context.

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

    @classmethod
    def apply(cls, arr: "CVDArray") -> "DesiderataExtractor":
        """Transform protocol: return self for property access."""
        return cls(arr)

    # ==================== Fix Path (VFD dimension) ====================

    @property
    def fix_path(self) -> np.ndarray:
        """VFD dimension as uint8 array.

        Values: 0b000 (No Awareness), 0b001 (Vendor Aware),
                0b011 (Fix Ready), 0b111 (Remediated)
        """
        if self._fix_path is None:
            self._fix_path = (self.states & 0b000111).astype(np.uint8)
        return self._fix_path

    @property
    def fix_path_labels(self) -> np.ndarray:
        """Human-readable fix path labels as object array."""
        from vulnstate.constants import FIX_PATH_LABELS

        return np.array(
            [FIX_PATH_LABELS.get(int(fp), "Unknown") for fp in self.fix_path],
            dtype=object,
        )

    # ==================== Threat State (PXA dimension) ====================

    @property
    def threat_state(self) -> np.ndarray:
        """PXA dimension as uint8 array.

        Values: 0b000 (Latent), 0b001 (Disclosed), 0b011 (Weaponized),
                0b100 (Targeted), 0b101 (Under Attack), 0b111 (Active Threat)
        """
        if self._threat_state is None:
            self._threat_state = ((self.states >> 3) & 0b000111).astype(np.uint8)
        return self._threat_state

    @property
    def threat_labels(self) -> np.ndarray:
        """Human-readable threat state labels as object array."""
        from vulnstate.constants import THREAT_LABELS

        return np.array(
            [THREAT_LABELS.get(int(ts), "Unknown") for ts in self.threat_state],
            dtype=object,
        )

    # ==================== Pair Analysis ====================

    @property
    def pair_mask(self) -> np.ndarray:
        """Pair analysis as uint16 bitmask array.

        Bit i is set if pair i is satisfied (earlier < later).
        See PAIR_BIT_POSITIONS for mapping.
        """
        if self._pair_mask is None:
            self._pair_mask = self._compute_pair_mask()
        return self._pair_mask

    def _compute_pair_mask(self) -> np.ndarray:
        """Compute pair satisfaction for all 15 pairs."""
        from vulnstate.constants import PAIR_BIT_POSITIONS, CVDEvent

        result = np.zeros(self.n, dtype=np.uint16)

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

            # Pair satisfied if either event missing OR t1 < t2
            missing = np.isnat(t1) | np.isnat(t2)
            correct_order = t1 < t2
            satisfied = missing | correct_order

            result |= np.where(satisfied, 1 << bit_pos, 0).astype(np.uint16)

        return result

    # ==================== Validity ====================

    @property
    def validity(self) -> np.ndarray:
        """History validity as uint8 enum array.

        Values: HistoryValidity.VALID (0), HistoryValidity.IMPOSSIBLE (1)
        """
        if self._validity is None:
            self._validity = self._compute_validity()
        return self._validity

    def _compute_validity(self) -> np.ndarray:
        """Check if required pairs are satisfied."""
        from vulnstate.constants import REQUIRED_PAIRS_MASK, HistoryValidity

        pair_mask = self.pair_mask
        required_satisfied = (pair_mask & REQUIRED_PAIRS_MASK) == REQUIRED_PAIRS_MASK

        return np.where(
            required_satisfied, HistoryValidity.VALID, HistoryValidity.IMPOSSIBLE
        ).astype(np.uint8)

    # ==================== Desiderata Scores ====================

    @property
    def desiderata_score(self) -> np.ndarray:
        """Count of satisfied desiderata (0-12) as uint8 array."""
        from vulnstate.constants import DESIDERATA_PAIRS_BITS

        pair_mask = self.pair_mask
        scores = np.zeros(self.n, dtype=np.uint8)
        for bit in DESIDERATA_PAIRS_BITS:
            scores += ((pair_mask >> bit) & 1).astype(np.uint8)
        return scores

    # ==================== Inference ====================

    @property
    def inferred_mask(self) -> np.ndarray:
        """Inference provenance as uint8 bitmask array.

        Bit i set if event i was inferred (not observed).
        """
        if self._inferred_mask is None:
            self._inferred_mask = np.zeros(self.n, dtype=np.uint8)
        return self._inferred_mask

    def apply_inferences(self) -> "DesiderataExtractor":
        """Apply inference rules to fill missing timestamps.

        Rules:
        - P without V → infer V at P timestamp
        - X without P → infer P at X timestamp
        - F without V → infer V at F timestamp
        - D without F → infer F at D timestamp

        Returns:
            self (for chaining)
        """
        from vulnstate.constants import CVDEvent

        _ = self.inferred_mask  # Ensure initialized

        # Rule 1: P implies V
        self._infer_event(source=CVDEvent.P, target=CVDEvent.V, target_bit=0)
        # Rule 2: X implies P
        self._infer_event(source=CVDEvent.X, target=CVDEvent.P, target_bit=3)
        # Rule 3: F implies V
        self._infer_event(source=CVDEvent.F, target=CVDEvent.V, target_bit=0)
        # Rule 4: D implies F
        self._infer_event(source=CVDEvent.D, target=CVDEvent.F, target_bit=1)

        # Invalidate caches
        self._pair_mask = None
        self._validity = None

        return self

    def _infer_event(self, source: "CVDEvent", target: "CVDEvent", target_bit: int) -> None:
        """Infer target event from source event timestamp."""
        source_ts = getattr(self.timestamps, source.name)
        target_ts = getattr(self.timestamps, target.name)

        infer_mask = ~np.isnat(source_ts) & np.isnat(target_ts)

        if not infer_mask.any():
            return

        target_ts[infer_mask] = source_ts[infer_mask]

        assert self._inferred_mask is not None
        self._inferred_mask[infer_mask] |= 1 << target_bit
        self._arr.state_ints[infer_mask] |= 1 << target_bit

    # ==================== Filtering & Labels ====================

    def get_satisfied_labels(self) -> list[list[str]]:
        """Get labels of satisfied desiderata for each vulnerability.

        Returns:
            List of lists, one per vulnerability, containing human-readable
            labels for satisfied desiderata (e.g., "Coordinated Disclosure").
        """
        from vulnstate.constants import get_desiderata_labels

        return [get_desiderata_labels(int(m)) for m in self.pair_mask]

    def get_violated_labels(self) -> list[list[str]]:
        """Get labels of violated desiderata (anti-desiderata) for each vulnerability.

        Returns:
            List of lists, one per vulnerability, containing human-readable
            labels for violations (e.g., "Zero-Day Exploit").
        """
        from vulnstate.constants import get_anti_desiderata_labels

        anti_mask = ~self.pair_mask & 0x0FFF
        return [get_anti_desiderata_labels(int(m)) for m in anti_mask]

    def where_satisfied(self, *desiderata: int) -> np.ndarray:
        """Return boolean mask where all specified desiderata are satisfied.

        Args:
            *desiderata: DesiderataBit values to check (AND logic)

        Returns:
            Boolean array where True means all specified desiderata satisfied.

        Example:
            >>> from vulnstate.constants import DesiderataBit
            >>> coordinated = arr.lifecycle.desiderata.where_satisfied(DesiderataBit.D1_V_P)
        """
        required = sum(1 << d for d in desiderata)
        return (self.pair_mask & required) == required

    def where_violated(self, *anti_desiderata: int) -> np.ndarray:
        """Return boolean mask where all specified anti-desiderata are violated.

        Args:
            *anti_desiderata: AntiDesiderataBit values to check (AND logic)

        Returns:
            Boolean array where True means all specified anti-desiderata violated.

        Example:
            >>> from vulnstate.constants import AntiDesiderataBit
            >>> zero_days = arr.lifecycle.desiderata.where_violated(AntiDesiderataBit.U2_X_V)
        """
        required = sum(1 << a for a in anti_desiderata)
        anti_mask = ~self.pair_mask & 0x0FFF
        return (anti_mask & required) == required

    # ==================== Boolean State Properties ====================

    def _has(self, t: np.ndarray) -> np.ndarray:
        """Check if timestamps exist (not NaT)."""
        return ~np.isnat(t)

    def _before(self, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """Check if t1 < t2 (True if either is missing)."""
        both_exist = self._has(t1) & self._has(t2)
        return ~both_exist | (t1 < t2)

    # Event occurrence checks

    @property
    def is_fix_available(self) -> np.ndarray:
        """True if fix is ready (F event occurred)."""
        return self._has(self.timestamps.F)

    @property
    def is_fix_deployed(self) -> np.ndarray:
        """True if fix is deployed (D event occurred)."""
        return self._has(self.timestamps.D)

    @property
    def is_under_attack(self) -> np.ndarray:
        """True if under active attack (A event occurred)."""
        return self._has(self.timestamps.A)

    @property
    def is_weaponized(self) -> np.ndarray:
        """True if public exploit exists (X event occurred)."""
        return self._has(self.timestamps.X)

    # Zero-day indicators

    @property
    def is_zero_day_exploit(self) -> np.ndarray:
        """True if exploit (X) before vendor awareness (V)."""
        X_ts, V_ts = self.timestamps.X, self.timestamps.V
        return self._has(X_ts) & (~self._has(V_ts) | (X_ts < V_ts))

    @property
    def is_zero_day_attack(self) -> np.ndarray:
        """True if attack (A) before vendor awareness (V)."""
        A_ts, V_ts = self.timestamps.A, self.timestamps.V
        return self._has(A_ts) & (~self._has(V_ts) | (A_ts < V_ts))

    @property
    def is_zero_day(self) -> np.ndarray:
        """True if exploit (X) or attack (A) before vendor awareness (V)."""
        return self.is_zero_day_exploit | self.is_zero_day_attack

    # Coordination quality

    @property
    def is_coordinated(self) -> np.ndarray:
        """True if vendor aware (V) before public disclosure (P)."""
        V_ts, P_ts = self.timestamps.V, self.timestamps.P
        return self._before(V_ts, P_ts) & self._has(V_ts) & self._has(P_ts)

    @property
    def is_premature_disclosure(self) -> np.ndarray:
        """True if public disclosure (P) before fix ready (F)."""
        P_ts, F_ts = self.timestamps.P, self.timestamps.F
        return self._has(P_ts) & (~self._has(F_ts) | (P_ts < F_ts))

    @property
    def is_responsible_disclosure(self) -> np.ndarray:
        """True if V→F→P ordering maintained (F before P)."""
        F_ts, P_ts = self.timestamps.F, self.timestamps.P
        # V < P (coordinated) AND F < P (fix ready before disclosure) AND F exists
        both_fp = self._has(F_ts) & self._has(P_ts)
        return self.is_coordinated & both_fp & (F_ts < P_ts)

    # Fix effectiveness (require BOTH events to exist)

    @property
    def has_fix_before_exploit(self) -> np.ndarray:
        """True if fix ready (F) before exploit public (X). Both must exist."""
        F_ts, X_ts = self.timestamps.F, self.timestamps.X
        both_exist = self._has(F_ts) & self._has(X_ts)
        return both_exist & (F_ts < X_ts)

    @property
    def has_fix_before_attack(self) -> np.ndarray:
        """True if fix ready (F) before attacks observed (A). Both must exist."""
        F_ts, A_ts = self.timestamps.F, self.timestamps.A
        both_exist = self._has(F_ts) & self._has(A_ts)
        return both_exist & (F_ts < A_ts)

    @property
    def has_deployment_before_exploit(self) -> np.ndarray:
        """True if fix deployed (D) before exploit public (X). Both must exist."""
        D_ts, X_ts = self.timestamps.D, self.timestamps.X
        both_exist = self._has(D_ts) & self._has(X_ts)
        return both_exist & (D_ts < X_ts)

    @property
    def has_deployment_before_attack(self) -> np.ndarray:
        """True if fix deployed (D) before attacks observed (A). Both must exist."""
        D_ts, A_ts = self.timestamps.D, self.timestamps.A
        both_exist = self._has(D_ts) & self._has(A_ts)
        return both_exist & (D_ts < A_ts)

    # Threat characteristics

    @property
    def is_private_attack(self) -> np.ndarray:
        """True if attacks (A) without public exploit (X)."""
        return self._has(self.timestamps.A) & ~self._has(self.timestamps.X)

    @property
    def is_mass_exploitation(self) -> np.ndarray:
        """True if both exploit public (X) and attacks observed (A)."""
        return self._has(self.timestamps.X) & self._has(self.timestamps.A)

    # ==================== Static Helpers ====================

    @staticmethod
    def explain_pair_violations(pair_mask: int) -> list[str]:
        """Return list of violated required pair names.

        Args:
            pair_mask: uint16 bitmask from pair_mask property

        Returns:
            List of pair names (e.g., ["V≺F", "F≺D"]) that are violated
        """
        from vulnstate.constants import PAIR_NAMES, REQUIRED_PAIRS_MASK

        violations = ~pair_mask & REQUIRED_PAIRS_MASK
        return [PAIR_NAMES[i] for i in range(15) if violations & (1 << i)]

    # ==================== Full Analysis ====================

    @staticmethod
    def analyze(arr: "CVDArray", infer: bool = True) -> "AnalysisResult":
        """Compute full analysis for an array.

        Args:
            arr: CVDArray to analyze
            infer: If True, apply inference rules before analysis

        Returns:
            AnalysisResult with all computed analytics
        """
        from vulnstate.models import AnalysisResult

        extractor = DesiderataExtractor(arr)

        if infer:
            extractor.apply_inferences()

        n = extractor.n
        ts = extractor.timestamps

        V_ts = ts.V
        F_ts = ts.F
        D_ts = ts.D
        P_ts = ts.P
        X_ts = ts.X
        A_ts = ts.A

        def has(t: np.ndarray) -> np.ndarray:
            return ~np.isnat(t)

        def before(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
            both_exist = has(t1) & has(t2)
            return ~both_exist | (t1 < t2)

        def days_between(t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
            result = np.full(n, np.nan, dtype=np.float32)
            both_exist = has(t1) & has(t2)
            delta = t2[both_exist].astype("datetime64[us]").astype(np.int64) - t1[
                both_exist
            ].astype("datetime64[us]").astype(np.int64)
            result[both_exist] = delta / (1e6 * 86400)
            return result

        # Group 1: Validity & Completeness
        validity_int = extractor.validity
        event_count = np.zeros(n, dtype=np.uint8)
        for event_ts in [V_ts, F_ts, D_ts, P_ts, X_ts, A_ts]:
            event_count += has(event_ts).astype(np.uint8)
        is_complete = event_count == 6

        # Group 2: Pair Analysis
        pair_mask = extractor.pair_mask
        desiderata_mask = pair_mask
        anti_desiderata_mask = ~pair_mask & 0x0FFF
        pair_observed_mask = np.zeros(n, dtype=np.uint16)
        from vulnstate.constants import PAIR_BIT_POSITIONS, CVDEvent

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

        # Group 3: Zero-Day Indicators
        is_zero_day_exploit = has(X_ts) & (~has(V_ts) | (X_ts < V_ts))
        is_zero_day_attack = has(A_ts) & (~has(V_ts) | (A_ts < V_ts))
        is_zero_day = is_zero_day_exploit | is_zero_day_attack

        # Group 4: Coordination Quality
        is_coordinated = before(V_ts, P_ts) & has(V_ts) & has(P_ts)
        is_premature_disclosure = has(P_ts) & (~has(F_ts) | (P_ts < F_ts))
        is_responsible_disclosure = is_coordinated & before(P_ts, F_ts) & has(F_ts)

        # Group 5: Fix Effectiveness
        has_fix_before_exploit = before(F_ts, X_ts) & has(F_ts)
        has_fix_before_attack = before(F_ts, A_ts) & has(F_ts)
        has_deployment_before_exploit = before(D_ts, X_ts) & has(D_ts)
        has_deployment_before_attack = before(D_ts, A_ts) & has(D_ts)

        # Group 6: Threat Characteristics
        is_private_attack = has(A_ts) & ~has(X_ts)
        is_weaponized = has(X_ts)
        is_mass_exploitation = has(X_ts) & has(A_ts)

        # Group 7: Classification
        fix_path_int = extractor.fix_path
        threat_state_int = extractor.threat_state
        outcome_int = np.zeros(n, dtype=np.uint8)
        first_event_int = np.zeros(n, dtype=np.uint8)
        disclosure_pattern_int = np.zeros(n, dtype=np.uint8)

        # Group 8: Scores
        desiderata_count = extractor.desiderata_score
        desiderata_score_float = (desiderata_count / 12.0).astype(np.float32)
        skill_score = np.zeros(n, dtype=np.float32)

        # Group 9: Time Deltas
        fix_lag_days = days_between(V_ts, F_ts)
        deployment_lag_days = days_between(F_ts, D_ts)
        disclosure_window_days = days_between(V_ts, P_ts)
        exploit_window_days = days_between(P_ts, X_ts)
        attack_window_days = days_between(P_ts, A_ts)
        exposure_days = days_between(P_ts, D_ts)
        exploit_to_attack_days = days_between(X_ts, A_ts)

        # Group 10: Transitions
        can_transition_mask = ~arr.state_ints & 0b111111

        # Group 11: Probabilities (placeholders)
        prob_X = np.zeros(n, dtype=np.float32)
        prob_A = np.zeros(n, dtype=np.float32)
        prob_threat = np.zeros(n, dtype=np.float32)

        return AnalysisResult(
            validity_int=validity_int,
            event_count=event_count,
            is_complete=is_complete,
            desiderata_mask=desiderata_mask,
            anti_desiderata_mask=anti_desiderata_mask,
            pair_observed_mask=pair_observed_mask,
            is_zero_day=is_zero_day,
            is_zero_day_exploit=is_zero_day_exploit,
            is_zero_day_attack=is_zero_day_attack,
            is_coordinated=is_coordinated,
            is_premature_disclosure=is_premature_disclosure,
            is_responsible_disclosure=is_responsible_disclosure,
            has_fix_before_exploit=has_fix_before_exploit,
            has_fix_before_attack=has_fix_before_attack,
            has_deployment_before_exploit=has_deployment_before_exploit,
            has_deployment_before_attack=has_deployment_before_attack,
            is_private_attack=is_private_attack,
            is_weaponized=is_weaponized,
            is_mass_exploitation=is_mass_exploitation,
            fix_path_int=fix_path_int,
            threat_state_int=threat_state_int,
            outcome_int=outcome_int,
            first_event_int=first_event_int,
            disclosure_pattern_int=disclosure_pattern_int,
            desiderata_count=desiderata_count,
            desiderata_score=desiderata_score_float,
            skill_score=skill_score,
            fix_lag_days=fix_lag_days,
            deployment_lag_days=deployment_lag_days,
            disclosure_window_days=disclosure_window_days,
            exploit_window_days=exploit_window_days,
            attack_window_days=attack_window_days,
            exposure_days=exposure_days,
            exploit_to_attack_days=exploit_to_attack_days,
            can_transition_mask=can_transition_mask,
            prob_X=prob_X,
            prob_A=prob_A,
            prob_threat=prob_threat,
            inferred_mask=extractor.inferred_mask if infer else np.zeros(n, dtype=np.uint8),
        )
