"""
Data Models - Dataclasses and property descriptors

Provides:
- VulnerabilityIdentity: ID fields (vuln_id, cve_id)
- VulnerabilityScoringData: CVSS scoring fields
- VulnerabilityEnrichmentData: EPSS, KEV enrichment
- VulnerabilityEventData: State and event tracking
- AnalysisResult: Analytics output from CVDAnalyzer
- ArrayCoreData: Core array fields
- ArrayTimestamps: Event timestamp arrays (DEPRECATED - use ArrayCVDState)
- ArrayCVDState: CVD state machine data (source of truth)
- ArrayCVDAnalytics: Precomputed bitmasks for vectorized queries
- ArrayMetadata: Metadata storage
- compute_pair_mask(): Vectorized pair ordering computation

Layer: Core
Dependencies: constants.py
Used by: vulnerability.py, array.py, analyzer.py, serialization.py, io.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from .constants import CVDEvent

# ==================== PROPERTY DESCRIPTORS ====================


class CachedAnalyticsProperty:
    """
    Descriptor for cached analytics properties with lazy computation.

    Provides read-only access to analytics attributes that are computed
    lazily and cached. Triggers analytics computation on first access.

    Example:
        class CVDVulnerability:
            is_zero_day = CachedAnalyticsProperty('is_zero_day',
                "True if exploit/attack occurred before vendor awareness.")
    """

    def __init__(self, attr_name: str, doc: str):
        self.attr_name = attr_name
        self.__doc__ = doc

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        if obj is None:
            return self
        if obj._analytics is None:
            obj._compute_analytics()
        value = getattr(obj._analytics, self.attr_name)
        # Return scalar bool for single-item access
        return bool(value[0])

    def __set__(self, obj: Any, value: Any) -> None:
        raise AttributeError(f"{self.attr_name} is read-only")


# ==================== VULNERABILITY DATACLASSES ====================


@dataclass
class EventMetadata:
    """Metadata for a single event occurrence."""

    actor: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class VulnerabilityIdentity:
    """Identity information for a vulnerability."""

    vuln_id: str = field(default_factory=lambda: str(uuid4()))
    cve_id: Optional[str] = None
    vendor_id: Optional[str] = None


@dataclass
class VulnerabilityScoringData:
    """CVSS 3.1 scoring data (parsed from cve_vector during analysis)."""

    # Raw CVSS vector string (e.g., 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H')
    cve_vector: Optional[str] = None

    # CVSS 3.1 Scores (computed from metrics)
    cvss_base_score: Optional[float] = None
    cvss_temporal_score: Optional[float] = None
    cvss_environmental_score: Optional[float] = None

    # CVSS 3.1 Base Metrics (parsed from cve_vector)
    attack_vector: Optional[str] = None  # N (Network), A (Adjacent), L (Local), P (Physical)
    attack_complexity: Optional[str] = None  # L (Low), H (High)
    privileges_required: Optional[str] = None  # N (None), L (Low), H (High)
    user_interaction: Optional[str] = None  # N (None), R (Required)
    scope: Optional[str] = None  # U (Unchanged), C (Changed)
    confidentiality_impact: Optional[str] = None  # N (None), L (Low), H (High)
    integrity_impact: Optional[str] = None  # N (None), L (Low), H (High)
    availability_impact: Optional[str] = None  # N (None), L (Low), H (High)

    # CVSS 3.1 Temporal Metrics (optional, parsed from cve_vector)
    exploit_code_maturity: Optional[str] = None  # X, U, P, F, H
    remediation_level: Optional[str] = None  # X, O, T, W, U
    report_confidence: Optional[str] = None  # X, U, R, C


@dataclass
class VulnerabilityEnrichmentData:
    """External enrichment data from EPSS, KEV, NVD, etc."""

    epss: Optional[float] = None  # Exploit Prediction Scoring System (0.0-1.0)
    is_kev: bool = False  # CISA Known Exploited Vulnerabilities flag


@dataclass
class VulnerabilityEventData:
    """Event tracking and state information."""

    state_encoded: np.uint8 = field(default_factory=lambda: np.uint8(0))
    events: dict[CVDEvent, Optional[datetime]] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class VulnerabilityAnalytics:
    """Cached analytics computed from vulnerability state."""

    # Categorical analytics
    severity: Optional[str] = None  # none, low, medium, high, critical
    fix_path: int = (
        0  # FixPath enum value (0=NO_AWARENESS, 1=VENDOR_AWARE, 3=FIX_READY, 7=REMEDIATED)
    )
    threat_state: int = 0  # ThreatState enum value

    # Boolean analytics
    is_zero_day: bool = False
    is_fix_available: bool = False
    is_fix_deployed: bool = False
    is_weaponized: bool = False
    is_under_attack: bool = False
    is_premature_disclosure: bool = False

    # Time metrics (days)
    disclosure_window_days: Optional[float] = None
    fix_lag_days: Optional[float] = None
    deployment_lag_days: Optional[float] = None

    # Research tier analytics
    violated_orderings_count: int = 0
    desiderata_mask: int = 0  # uint16 bitmask
    anti_desiderata_mask: int = 0  # uint16 bitmask
    desiderata_count: int = 0  # Count of satisfied (popcount of mask)
    skill_score: Optional[float] = None


# ==================== ARRAY DATACLASSES ====================


@dataclass
class ArrayCoreData:
    """Core data arrays for CVDArray (vulnerabilities, states, identifiers)."""

    vulnerabilities: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    states: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    vuln_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))


@dataclass
class ArrayTimestamps:
    """Event timestamp arrays (6 events: V, F, D, P, X, A)."""

    V: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[us]"))
    F: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[us]"))
    D: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[us]"))
    P: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[us]"))
    X: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[us]"))
    A: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[us]"))

    def __getitem__(self, key: Any) -> "ArrayTimestamps":
        """Slice all arrays consistently."""
        return ArrayTimestamps(
            V=self.V[key],
            F=self.F[key],
            D=self.D[key],
            P=self.P[key],
            X=self.X[key],
            A=self.A[key],
        )


@dataclass
class ArrayIdentifiers:
    """Identifier arrays for vulnerabilities."""

    vuln_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    cve_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    vendor_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    def __getitem__(self, key: Any) -> "ArrayIdentifiers":
        """Slice all arrays consistently."""
        return ArrayIdentifiers(
            vuln_id=self.vuln_id[key],
            cve_id=self.cve_id[key],
            vendor_id=self.vendor_id[key],
        )


@dataclass
class ArrayScoring:
    """CVSS 3.1 scoring data arrays."""

    # Scores
    cvss_score: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    cvss_exploitability: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    cvss_impact: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    cvss_vector_int: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint16))

    # CVSS 3.1 Base Metrics
    attack_vector: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    attack_complexity: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    privileges_required: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    user_interaction: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    scope: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    confidentiality_impact: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    integrity_impact: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    availability_impact: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    def __getitem__(self, key: Any) -> "ArrayScoring":
        """Slice all arrays consistently."""
        return ArrayScoring(
            cvss_score=self.cvss_score[key],
            cvss_exploitability=self.cvss_exploitability[key],
            cvss_impact=self.cvss_impact[key],
            cvss_vector_int=self.cvss_vector_int[key],
            attack_vector=self.attack_vector[key],
            attack_complexity=self.attack_complexity[key],
            privileges_required=self.privileges_required[key],
            user_interaction=self.user_interaction[key],
            scope=self.scope[key],
            confidentiality_impact=self.confidentiality_impact[key],
            integrity_impact=self.integrity_impact[key],
            availability_impact=self.availability_impact[key],
        )


@dataclass
class ArrayEnrichment:
    """EPSS and KEV enrichment data arrays."""

    epss: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    epss_percentile: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    kev: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    kev_date: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[s]"))

    def __getitem__(self, key: Any) -> "ArrayEnrichment":
        """Slice all arrays consistently."""
        return ArrayEnrichment(
            epss=self.epss[key],
            epss_percentile=self.epss_percentile[key],
            kev=self.kev[key],
            kev_date=self.kev_date[key],
        )


@dataclass
class ArrayCVDState:
    """
    CVD state machine data - SINGLE SOURCE OF TRUTH.

    All CVD state information (history, event_order, pairs) can be derived
    from these two components: which events occurred + when they occurred.

    This is the minimal storage required to reconstruct all CVD analytics.
    See docs/design/2026-01-20-cvd-state-storage-architecture.md for details.
    """

    # Which events occurred (bitmask: bit 0=V, 1=F, 2=D, 3=P, 4=X, 5=A)
    states: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))

    # When each event occurred (NaT if event not occurred)
    V_timestamps: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[us]"))
    F_timestamps: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[us]"))
    D_timestamps: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[us]"))
    P_timestamps: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[us]"))
    X_timestamps: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[us]"))
    A_timestamps: np.ndarray = field(default_factory=lambda: np.array([], dtype="datetime64[us]"))

    def __getitem__(self, key: Any) -> "ArrayCVDState":
        """Slice all arrays consistently."""
        return ArrayCVDState(
            states=self.states[key],
            V_timestamps=self.V_timestamps[key],
            F_timestamps=self.F_timestamps[key],
            D_timestamps=self.D_timestamps[key],
            P_timestamps=self.P_timestamps[key],
            X_timestamps=self.X_timestamps[key],
            A_timestamps=self.A_timestamps[key],
        )


@dataclass
class ArrayCVDAnalytics:
    """
    Precomputed bitmasks for fast vectorized queries.

    Computed once during array construction or sync(), reused for all queries.
    These are DERIVED from ArrayCVDState - not independent source data.

    Memory cost: 3 bytes/vuln (2 for pair_mask + 1 for history_id)
    Performance gain: O(1) vectorized queries vs O(n log n) timestamp sorting

    Edge case - simultaneous events:
        If two events occur at the exact same timestamp (same microsecond),
        the pair is treated as "not satisfied" (bit clear). This means
        is_zero_day_exploit returns True if X and V occur simultaneously,
        since V did not occur strictly BEFORE X. This is intentional - if
        vendor awareness wasn't established before exploit, that's still
        a problematic disclosure pattern.

    See:
        - docs/design/2026-01-20-cvd-state-storage-architecture.md
        - vulnstate.constants.PAIR_BIT_POSITIONS for bit encoding
    """

    # Pair ordering mask (15 bits for 15 event pairs)
    # Bit i = 1 if pair i occurred in desired order (strictly less than)
    # Bit is 0 if: pair violated, events simultaneous, or one/both events missing
    # See constants.PAIR_BIT_POSITIONS for canonical bit layout
    pair_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint16))

    # History ID (0-69 for complete histories, 255 for incomplete)
    # Maps to one of 70 valid complete orderings when all 6 events occurred
    # See docs/ref/cvd-histories.md for the 70 valid orderings
    # Value 255 indicates incomplete history (fewer than 6 events)
    history_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))

    def __getitem__(self, key: Any) -> "ArrayCVDAnalytics":
        """Slice all arrays consistently."""
        return ArrayCVDAnalytics(
            pair_mask=self.pair_mask[key],
            history_id=self.history_id[key] if len(self.history_id) else self.history_id,
        )


@dataclass
class ArrayProbabilities:
    """Flexible probability inputs. NaN = not set."""

    base_threat: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    xa_split_ratio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    threat_multiplier: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    override_X: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    override_A: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    def __getitem__(self, key: Any) -> "ArrayProbabilities":
        """Slice all arrays consistently."""
        return ArrayProbabilities(
            base_threat=self.base_threat[key],
            xa_split_ratio=self.xa_split_ratio[key],
            threat_multiplier=self.threat_multiplier[key],
            override_X=self.override_X[key],
            override_A=self.override_A[key],
        )


@dataclass
class AnalysisResult:
    """All computed analytics from CVDAnalyzer (Groups 1-11)."""

    # Group 1: Validity & Completeness
    validity_int: np.ndarray  # uint8[N]
    event_count: np.ndarray  # uint8[N]
    is_complete: np.ndarray  # bool[N]

    # Group 2: Pair Analysis
    desiderata_mask: np.ndarray  # uint16[N]
    anti_desiderata_mask: np.ndarray  # uint16[N]
    pair_observed_mask: np.ndarray  # uint16[N]

    # Group 3: Zero-Day Indicators
    is_zero_day: np.ndarray  # bool[N]
    is_zero_day_exploit: np.ndarray  # bool[N]
    is_zero_day_attack: np.ndarray  # bool[N]

    # Group 4: Coordination Quality
    is_coordinated: np.ndarray  # bool[N]
    is_premature_disclosure: np.ndarray  # bool[N]
    is_responsible_disclosure: np.ndarray  # bool[N]

    # Group 5: Fix Effectiveness
    has_fix_before_exploit: np.ndarray  # bool[N]
    has_fix_before_attack: np.ndarray  # bool[N]
    has_deployment_before_exploit: np.ndarray  # bool[N]
    has_deployment_before_attack: np.ndarray  # bool[N]

    # Group 6: Threat Characteristics
    is_private_attack: np.ndarray  # bool[N]
    is_weaponized: np.ndarray  # bool[N]
    is_mass_exploitation: np.ndarray  # bool[N]

    # Group 7: Classification (optional, defaults to empty arrays)
    fix_path_int: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    threat_state_int: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    outcome_int: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    first_event_int: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    disclosure_pattern_int: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))

    # Group 8: Scores & Counts (optional)
    desiderata_count: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))
    desiderata_score: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    skill_score: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    # Group 9: Time Deltas (optional, float32 for NaN support)
    fix_lag_days: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    deployment_lag_days: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    disclosure_window_days: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    exploit_window_days: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    attack_window_days: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    exposure_days: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    exploit_to_attack_days: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )

    # Group 10: Transitions (optional)
    can_transition_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))

    # Group 11: Probabilities (optional)
    prob_X: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    prob_A: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    prob_threat: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    # Group 12: Provenance (optional)
    # Bitmask tracking which events were inferred vs observed.
    # Bit layout matches state encoding: V=0, F=1, D=2, P=3, X=4, A=5
    inferred_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))

    def __len__(self) -> int:
        return len(self.validity_int)

    def __getitem__(self, key: Any) -> "AnalysisResult":
        """Slice all arrays consistently."""
        return AnalysisResult(
            # Group 1
            validity_int=self.validity_int[key],
            event_count=self.event_count[key],
            is_complete=self.is_complete[key],
            # Group 2
            desiderata_mask=self.desiderata_mask[key],
            anti_desiderata_mask=self.anti_desiderata_mask[key],
            pair_observed_mask=self.pair_observed_mask[key],
            # Group 3
            is_zero_day=self.is_zero_day[key],
            is_zero_day_exploit=self.is_zero_day_exploit[key],
            is_zero_day_attack=self.is_zero_day_attack[key],
            # Group 4
            is_coordinated=self.is_coordinated[key],
            is_premature_disclosure=self.is_premature_disclosure[key],
            is_responsible_disclosure=self.is_responsible_disclosure[key],
            # Group 5
            has_fix_before_exploit=self.has_fix_before_exploit[key],
            has_fix_before_attack=self.has_fix_before_attack[key],
            has_deployment_before_exploit=self.has_deployment_before_exploit[key],
            has_deployment_before_attack=self.has_deployment_before_attack[key],
            # Group 6
            is_private_attack=self.is_private_attack[key],
            is_weaponized=self.is_weaponized[key],
            is_mass_exploitation=self.is_mass_exploitation[key],
            # Group 7
            fix_path_int=self.fix_path_int[key] if len(self.fix_path_int) else self.fix_path_int,
            threat_state_int=(
                self.threat_state_int[key] if len(self.threat_state_int) else self.threat_state_int
            ),
            outcome_int=self.outcome_int[key] if len(self.outcome_int) else self.outcome_int,
            first_event_int=(
                self.first_event_int[key] if len(self.first_event_int) else self.first_event_int
            ),
            disclosure_pattern_int=(
                self.disclosure_pattern_int[key]
                if len(self.disclosure_pattern_int)
                else self.disclosure_pattern_int
            ),
            # Group 8
            desiderata_count=(
                self.desiderata_count[key] if len(self.desiderata_count) else self.desiderata_count
            ),
            desiderata_score=(
                self.desiderata_score[key] if len(self.desiderata_score) else self.desiderata_score
            ),
            skill_score=self.skill_score[key] if len(self.skill_score) else self.skill_score,
            # Group 9
            fix_lag_days=self.fix_lag_days[key] if len(self.fix_lag_days) else self.fix_lag_days,
            deployment_lag_days=(
                self.deployment_lag_days[key]
                if len(self.deployment_lag_days)
                else self.deployment_lag_days
            ),
            disclosure_window_days=(
                self.disclosure_window_days[key]
                if len(self.disclosure_window_days)
                else self.disclosure_window_days
            ),
            exploit_window_days=(
                self.exploit_window_days[key]
                if len(self.exploit_window_days)
                else self.exploit_window_days
            ),
            attack_window_days=(
                self.attack_window_days[key]
                if len(self.attack_window_days)
                else self.attack_window_days
            ),
            exposure_days=(
                self.exposure_days[key] if len(self.exposure_days) else self.exposure_days
            ),
            exploit_to_attack_days=(
                self.exploit_to_attack_days[key]
                if len(self.exploit_to_attack_days)
                else self.exploit_to_attack_days
            ),
            # Group 10
            can_transition_mask=(
                self.can_transition_mask[key]
                if len(self.can_transition_mask)
                else self.can_transition_mask
            ),
            # Group 11
            prob_X=self.prob_X[key] if len(self.prob_X) else self.prob_X,
            prob_A=self.prob_A[key] if len(self.prob_A) else self.prob_A,
            prob_threat=self.prob_threat[key] if len(self.prob_threat) else self.prob_threat,
            # Group 12
            inferred_mask=(
                self.inferred_mask[key] if len(self.inferred_mask) else self.inferred_mask
            ),
        )

    def can_transition_V(self) -> np.ndarray:
        """Check if V transition is possible."""
        return (self.can_transition_mask & 0b000001) != 0

    def can_transition_F(self) -> np.ndarray:
        """Check if F transition is possible."""
        return (self.can_transition_mask & 0b000010) != 0

    def can_transition_D(self) -> np.ndarray:
        """Check if D transition is possible."""
        return (self.can_transition_mask & 0b000100) != 0

    def can_transition_P(self) -> np.ndarray:
        """Check if P transition is possible."""
        return (self.can_transition_mask & 0b001000) != 0

    def can_transition_X(self) -> np.ndarray:
        """Check if X transition is possible."""
        return (self.can_transition_mask & 0b010000) != 0

    def can_transition_A(self) -> np.ndarray:
        """Check if A transition is possible."""
        return (self.can_transition_mask & 0b100000) != 0

    def can_transition(self, event: int) -> np.ndarray:
        """Check if transition for given event is possible."""
        return (self.can_transition_mask & (1 << event)) != 0

    @property
    def inferred_V(self) -> np.ndarray:
        """Bool mask: True where V was inferred (not observed)."""
        if len(self.inferred_mask) == 0:
            return np.array([], dtype=bool)
        return (self.inferred_mask & 0b000001) != 0

    @property
    def inferred_F(self) -> np.ndarray:
        """Bool mask: True where F was inferred (not observed)."""
        if len(self.inferred_mask) == 0:
            return np.array([], dtype=bool)
        return (self.inferred_mask & 0b000010) != 0

    @property
    def inferred_P(self) -> np.ndarray:
        """Bool mask: True where P was inferred (not observed)."""
        if len(self.inferred_mask) == 0:
            return np.array([], dtype=bool)
        return (self.inferred_mask & 0b001000) != 0

    def where_desiderata(self, required: int = 0, forbidden: int = 0) -> np.ndarray:
        """Filter by desiderata bitmask.

        Args:
            required: Bitmask of desiderata that must be satisfied
            forbidden: Bitmask of desiderata that must not be satisfied

        Returns:
            Boolean mask of matching entries
        """
        mask: np.ndarray = np.ones(len(self), dtype=bool)
        if required:
            mask &= (self.desiderata_mask & required) == required
        if forbidden:
            mask &= (self.desiderata_mask & forbidden) == 0
        return mask


@dataclass
class ArrayAnalytics:
    """Cached analytics arrays (15 analytics fields)."""

    # Categorical fields
    severities: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    fix_path: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.uint8)
    )  # FixPath enum values
    threat_state: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.uint8)
    )  # ThreatState enum values

    # Boolean flags
    is_zero_day: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    is_fix_available: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    is_fix_deployed: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    is_weaponized: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    is_under_attack: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))
    is_premature_disclosure: np.ndarray = field(default_factory=lambda: np.array([], dtype=bool))

    # Numeric metrics (float32 for memory efficiency)
    disclosure_window_days: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    fix_lag_days: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    deployment_lag_days: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    desiderata_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    # Bitmask fields (uint16 for 12 bits)
    desiderata_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint16))
    anti_desiderata_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint16))

    skill_scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))

    # Integer count
    violated_orderings_count: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32)
    )

    # Deprecated alias for backward compatibility
    @property
    def cubes(self) -> np.ndarray:
        """Deprecated: Use fix_path instead."""
        import warnings

        warnings.warn("cubes is deprecated, use fix_path", DeprecationWarning, stacklevel=2)
        return self.fix_path


@dataclass
class ArrayMetadata:
    """Metadata and backward compatibility fields."""

    raw: dict[str, np.ndarray] = field(default_factory=dict)
    event_timestamps_absolute: dict[CVDEvent, np.ndarray] = field(
        default_factory=dict
    )  # Deprecated


# ==================== HELPER FUNCTIONS ====================


def compute_pair_mask(cvd_state: ArrayCVDState) -> np.ndarray:
    """
    Compute pair ordering mask from timestamps (vectorized).

    Computes which of 15 event pairs occurred in the desired chronological order.
    Returns uint16 bitmask where bit i indicates if pair i is satisfied.

    Args:
        cvd_state: ArrayCVDState with state_encoded and timestamps

    Returns:
        np.ndarray[uint16]: Bitmask array, one per vulnerability
            Bit 0 = V≺F, Bit 1 = V≺D, ..., Bit 14 = X≺A

    Example:
        >>> state = ArrayCVDState(...)
        >>> mask = compute_pair_mask(state)
        >>> is_coordinated = (mask & (1 << 2)) != 0  # V≺P bit
    """
    n = len(cvd_state.states)
    mask = np.zeros(n, dtype=np.uint16)

    # Define all 15 event pairs with their bit positions
    # Bit layout matches design doc:
    #   0: V≺F   5: F≺D    9: D≺P   12: P≺X
    #   1: V≺D   6: F≺P   10: D≺X   13: P≺A
    #   2: V≺P   7: F≺X   11: D≺A   14: X≺A
    #   3: V≺X   8: F≺A
    #   4: V≺A
    pairs = [
        # V precedes all others (bits 0-4)
        ("V_timestamps", "F_timestamps", 0),  # V≺F
        ("V_timestamps", "D_timestamps", 1),  # V≺D
        ("V_timestamps", "P_timestamps", 2),  # V≺P
        ("V_timestamps", "X_timestamps", 3),  # V≺X
        ("V_timestamps", "A_timestamps", 4),  # V≺A
        # F precedes D, P, X, A (bits 5-8)
        ("F_timestamps", "D_timestamps", 5),  # F≺D
        ("F_timestamps", "P_timestamps", 6),  # F≺P
        ("F_timestamps", "X_timestamps", 7),  # F≺X
        ("F_timestamps", "A_timestamps", 8),  # F≺A
        # D precedes P, X, A (bits 9-11)
        ("D_timestamps", "P_timestamps", 9),  # D≺P
        ("D_timestamps", "X_timestamps", 10),  # D≺X
        ("D_timestamps", "A_timestamps", 11),  # D≺A
        # P precedes X, A (bits 12-13)
        ("P_timestamps", "X_timestamps", 12),  # P≺X
        ("P_timestamps", "A_timestamps", 13),  # P≺A
        # X precedes A (bit 14)
        ("X_timestamps", "A_timestamps", 14),  # X≺A
    ]

    for earlier_attr, later_attr, bit_pos in pairs:
        earlier_ts = getattr(cvd_state, earlier_attr)
        later_ts = getattr(cvd_state, later_attr)

        # Both events must have occurred
        both_occurred = ~np.isnat(earlier_ts) & ~np.isnat(later_ts)

        # Earlier must be before later
        correct_order = earlier_ts < later_ts

        # Set bit for vulnerabilities where pair is satisfied
        satisfied = both_occurred & correct_order
        mask[satisfied] |= (1 << bit_pos)

    return mask


def compute_history_id(cvd_state: ArrayCVDState) -> np.ndarray:
    """
    Compute history IDs from timestamps (0-69 for complete, 255 for incomplete).

    For vulnerabilities with all 6 events, sorts timestamps to get the
    chronological ordering, then looks up the history in VALID_HISTORIES
    to get the canonical history_id (0-69).

    For incomplete histories (fewer than 6 events), returns 255.

    Args:
        cvd_state: ArrayCVDState with timestamps for all 6 events

    Returns:
        np.ndarray[uint8]: History ID array
            - 0-69: Index into VALID_HISTORIES for complete histories
            - 255: Incomplete history (not all 6 events occurred)

    Example:
        >>> state = ArrayCVDState(...)  # with timestamps for V, F, D, P, X, A
        >>> ids = compute_history_id(state)
        >>> ids[0]  # 69 for VFDPXA (perfect CVD)
    """
    from .constants import HISTORY_TO_ID, INCOMPLETE_HISTORY_ID

    n = len(cvd_state.states)
    result = np.full(n, INCOMPLETE_HISTORY_ID, dtype=np.uint8)

    # Stack all timestamps: shape (N, 6)
    timestamps = np.column_stack([
        cvd_state.V_timestamps,
        cvd_state.F_timestamps,
        cvd_state.D_timestamps,
        cvd_state.P_timestamps,
        cvd_state.X_timestamps,
        cvd_state.A_timestamps,
    ])

    # Event labels for building history strings
    event_labels = ["V", "F", "D", "P", "X", "A"]

    # Check which rows have all 6 events (no NaT values)
    complete_mask = ~np.any(np.isnat(timestamps), axis=1)

    # Process only complete histories
    for i in np.where(complete_mask)[0]:
        # Get timestamps for this vulnerability
        ts = timestamps[i]

        # Sort indices by timestamp to get chronological order
        order = np.argsort(ts)

        # Build history string
        history = "".join(event_labels[j] for j in order)

        # Look up history_id
        if history in HISTORY_TO_ID:
            result[i] = HISTORY_TO_ID[history]
        # else: remains 255 (invalid history - should not happen with valid data)

    return result
