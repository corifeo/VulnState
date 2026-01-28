"""
Data Models - Dataclasses and property descriptors

Provides:
- CVSSScore: Single CVSS score from specific source with vector parsing
- EPSSScore: Single EPSS score snapshot with model version
- CWEEntry: CWE weakness entry with source attribution
- KEVEntry: CISA KEV catalog entry with due dates
- ExploitReference: Reference to known exploit code
- ScoreResult: Computed scores from ScoreExtractor transform
- AnalyticsResult: Computed analytics from CVDStateAnalyzer transform
- ArraySource: Source of truth object arrays for CVDArray
- VulnerabilityIdentity: ID fields (internal_id, cve_id)

- AnalysisResult: Analytics output from DesiderataExtractor
- ArrayState: CVD state bitmask array
- ArrayTimestamps: Event timestamp arrays (V, F, D, P, X, A)
- ArrayIdentifiers: Vulnerability ID arrays (internal_id, cve_id)
- ArrayCVDAnalytics: Precomputed bitmasks for vectorized queries
- ArrayMetadata: Metadata storage
- compute_pair_mask(): Vectorized pair ordering computation

Layer: Core
Dependencies: constants.py
Used by: vulnerability.py, array.py, serialization.py, io.py
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

import numpy as np

from .constants import CVDEvent, FixPath, ThreatState

__all__ = [
    # New API v2 dataclasses
    "CVSSScore",
    "CVSSMetrics",
    "EPSSScore",
    "CWEEntry",
    "CWE",
    "CPE",
    "KEVEntry",
    "KEV",
    "ExploitReference",
    "ScoreResult",
    "AnalyticsResult",
    "ArraySource",
    # Core dataclasses
    "AnalysisResult",
    "CachedAnalyticsProperty",
    "VulnerabilityIdentity",

    "ArrayState",
    "ArrayTimestamps",
    "ArrayIdentifiers",
    "ArrayCVDAnalytics",
    # Helper functions
    "compute_pair_mask",
    "compute_history_id",
]

# ==================== CVSS SCORING ====================


@dataclass(frozen=True)
class CVSSScore:
    """Single CVSS score from a specific source.

    Attributes:
        version: CVSS version (2.0, 3.0, 3.1, 4.0).
        base_score: Base score (0.0-10.0).
        vector: Raw CVSS vector string.
        source: Source identifier (e.g., 'nvd', 'vendor').
        source_status: Status from source (e.g., 'Analyzed', 'Modified').
        reserved_at: When CVE ID was reserved.
        published_at: When vulnerability was published.
        updated_at: When record was last updated.
        temporal_score: Optional temporal score.
        environmental_score: Optional environmental score.
    """

    version: float
    base_score: float
    vector: str
    source: str
    source_status: Optional[str]
    reserved_at: Optional[datetime]
    published_at: Optional[datetime]
    updated_at: Optional[datetime]
    temporal_score: Optional[float]
    environmental_score: Optional[float]

    def parse_vector(self) -> dict[str, str]:
        """Parse CVSS vector string into metrics dict.

        Returns:
            Dict mapping metric abbreviations to values.
            E.g., {'AV': 'N', 'AC': 'L', 'C': 'H', ...}
        """
        result: dict[str, str] = {}
        # Handle v2, v3.0, v3.1, v4.0 prefixes
        vector = self.vector
        for prefix in ("CVSS:4.0/", "CVSS:3.1/", "CVSS:3.0/", "CVSS:2.0/"):
            vector = vector.replace(prefix, "")

        parts = vector.split("/")
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                result[key] = value
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict, omitting None values."""
        d: dict[str, Any] = {
            "version": self.version,
            "base_score": self.base_score,
            "vector": self.vector,
            "source": self.source,
        }
        if self.source_status is not None:
            d["source_status"] = self.source_status
        if self.reserved_at is not None:
            d["reserved_at"] = self.reserved_at.isoformat()
        if self.published_at is not None:
            d["published_at"] = self.published_at.isoformat()
        if self.updated_at is not None:
            d["updated_at"] = self.updated_at.isoformat()
        if self.temporal_score is not None:
            d["temporal_score"] = self.temporal_score
        if self.environmental_score is not None:
            d["environmental_score"] = self.environmental_score
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CVSSScore":
        """Deserialize from dict.

        Args:
            d: Dict with required keys (version, base_score, vector, source)
               and optional keys for other fields.

        Returns:
            CVSSScore instance.
        """
        return cls(
            version=d["version"],
            base_score=d["base_score"],
            vector=d["vector"],
            source=d["source"],
            source_status=d.get("source_status"),
            reserved_at=(
                datetime.fromisoformat(d["reserved_at"]) if d.get("reserved_at") else None
            ),
            published_at=(
                datetime.fromisoformat(d["published_at"]) if d.get("published_at") else None
            ),
            updated_at=(datetime.fromisoformat(d["updated_at"]) if d.get("updated_at") else None),
            temporal_score=d.get("temporal_score"),
            environmental_score=d.get("environmental_score"),
        )

    @classmethod
    def from_raw(
        cls,
        version: float,
        base_score: float,
        vector: str,
        source: str,
        source_status: Optional[str] = None,
        reserved_at: Optional[datetime] = None,
        published_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        temporal_score: Optional[float] = None,
        environmental_score: Optional[float] = None,
    ) -> "CVSSScore":
        """Create from raw data (lazy metrics parsing)."""
        return cls(
            version=version,
            base_score=base_score,
            vector=vector,
            source=source,
            source_status=source_status,
            reserved_at=reserved_at,
            published_at=published_at,
            updated_at=updated_at,
            temporal_score=temporal_score,
            environmental_score=environmental_score,
        )

    @classmethod
    def from_stored(
        cls,
        version: float,
        base_score: float,
        vector: str,
        source: str,
        metrics: Optional[dict[str, str]] = None,
        source_status: Optional[str] = None,
        reserved_at: Optional[datetime] = None,
        published_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        temporal_score: Optional[float] = None,
        environmental_score: Optional[float] = None,
    ) -> "CVSSScore":
        """Create from stored data (pre-parsed metrics available)."""
        # Note: metrics parameter is accepted but not stored (frozen dataclass)
        # Future: could cache parsed metrics in a non-frozen version
        return cls(
            version=version,
            base_score=base_score,
            vector=vector,
            source=source,
            source_status=source_status,
            reserved_at=reserved_at,
            published_at=published_at,
            updated_at=updated_at,
            temporal_score=temporal_score,
            environmental_score=environmental_score,
        )


@dataclass(frozen=True)
class EPSSScore:
    """Single EPSS score snapshot."""

    model: int
    probability: float
    percentile: float
    computed_at: Optional[datetime]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "probability": self.probability,
            "percentile": self.percentile,
        }
        if self.computed_at is not None:
            d["computed_at"] = self.computed_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EPSSScore":
        return cls(
            model=d["model"],
            probability=d["probability"],
            percentile=d["percentile"],
            computed_at=datetime.fromisoformat(d["computed_at"]) if d.get("computed_at") else None,
        )


@dataclass(frozen=True)
class CWEEntry:
    """CWE weakness entry with source attribution (legacy)."""

    id: str
    source: Optional[str]
    primary: bool = True

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "primary": self.primary}
        if self.source is not None:
            d["source"] = self.source
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CWEEntry":
        return cls(id=d["id"], source=d.get("source"), primary=d.get("primary", True))


@dataclass(frozen=True)
class CWE:
    """CWE weakness with full metadata (API v2)."""

    id: str
    name: str = ""
    category_id: str = ""
    category_name: str = ""
    keywords: tuple[str, ...] = ()
    source: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id}
        if self.name:
            d["name"] = self.name
        if self.category_id:
            d["category_id"] = self.category_id
        if self.category_name:
            d["category_name"] = self.category_name
        if self.keywords:
            d["keywords"] = list(self.keywords)
        if self.source:
            d["source"] = self.source
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CWE":
        return cls(
            id=d["id"],
            name=d.get("name", ""),
            category_id=d.get("category_id", ""),
            category_name=d.get("category_name", ""),
            keywords=tuple(d.get("keywords", [])),
            source=d.get("source"),
        )


@dataclass(frozen=True)
class CPE:
    """Parsed CPE 2.3 identifier."""

    raw: str
    part: str  # 'a' (application), 'o' (OS), 'h' (hardware)
    vendor: str
    product: str
    version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "part": self.part,
            "vendor": self.vendor,
            "product": self.product,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CPE":
        return cls(
            raw=d["raw"],
            part=d["part"],
            vendor=d["vendor"],
            product=d["product"],
            version=d["version"],
        )

    @classmethod
    def parse(cls, cpe_string: str) -> "CPE":
        """Parse CPE 2.3 formatted string."""
        # cpe:2.3:a:vendor:product:version:...
        parts = cpe_string.split(":")
        if len(parts) >= 5:
            part_char = parts[2] if len(parts) > 2 else "a"
            part_map = {"a": "application", "o": "os", "h": "hardware"}
            return cls(
                raw=cpe_string,
                part=part_map.get(part_char, part_char),
                vendor=parts[3] if len(parts) > 3 else "",
                product=parts[4] if len(parts) > 4 else "",
                version=parts[5] if len(parts) > 5 else "",
            )
        return cls(raw=cpe_string, part="", vendor="", product="", version="")


@dataclass(frozen=True)
class KEVEntry:
    """CISA Known Exploited Vulnerabilities catalog entry (legacy)."""

    added_at: datetime
    due_date: Optional[datetime]
    required_action: Optional[str]
    ransomware_use: Optional[bool]
    notes: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"added_at": self.added_at.isoformat()}
        if self.due_date is not None:
            d["due_date"] = self.due_date.isoformat()
        if self.required_action is not None:
            d["required_action"] = self.required_action
        if self.ransomware_use is not None:
            d["ransomware_use"] = self.ransomware_use
        if self.notes is not None:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KEVEntry":
        return cls(
            added_at=datetime.fromisoformat(d["added_at"]),
            due_date=datetime.fromisoformat(d["due_date"]) if d.get("due_date") else None,
            required_action=d.get("required_action"),
            ransomware_use=d.get("ransomware_use"),
            notes=d.get("notes"),
        )


@dataclass(frozen=True)
class KEV:
    """CISA Known Exploited Vulnerability with full metadata (API v2)."""

    vendor: str
    product: str
    vulnerability_name: str
    added_at: datetime
    due_date: Optional[datetime] = None
    known_ransomware_use: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "vendor": self.vendor,
            "product": self.product,
            "vulnerability_name": self.vulnerability_name,
            "added_at": self.added_at.isoformat(),
        }
        if self.due_date:
            d["due_date"] = self.due_date.isoformat()
        if self.known_ransomware_use:
            d["known_ransomware_use"] = True
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KEV":
        return cls(
            vendor=d["vendor"],
            product=d["product"],
            vulnerability_name=d["vulnerability_name"],
            added_at=datetime.fromisoformat(d["added_at"]),
            due_date=datetime.fromisoformat(d["due_date"]) if d.get("due_date") else None,
            known_ransomware_use=d.get("known_ransomware_use", False),
            notes=d.get("notes", ""),
        )


@dataclass(frozen=True)
class ExploitReference:
    """Reference to known exploit code."""

    source: str
    reference: str
    metadata: Optional[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"source": self.source, "reference": self.reference}
        if self.metadata is not None:
            d["metadata"] = self.metadata
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExploitReference":
        return cls(source=d["source"], reference=d["reference"], metadata=d.get("metadata"))


@dataclass(frozen=True)
class ScoreResult:
    """Computed scores from ScoreExtractor.apply_single()."""

    cvss_score: Optional[float]
    cvss_max: Optional[float]
    epss_probability: Optional[float]
    epss_percentile: Optional[float]
    kev: bool
    has_exploit: bool
    cwe_count: int
    cpe_count: int
    vendors: set[str]
    products: set[str]


@dataclass(frozen=True)
class AnalyticsResult:
    """Computed analytics from CVDStateAnalyzer.apply_single()."""

    fix_path: FixPath
    threat_state: ThreatState
    is_zero_day: bool
    is_zero_day_exploit: bool
    is_zero_day_attack: bool
    is_coordinated: bool
    is_premature_disclosure: bool
    is_responsible_disclosure: bool
    has_fix_before_exploit: bool
    has_fix_before_attack: bool
    is_private_attack: bool
    is_weaponized: bool
    is_mass_exploitation: bool
    is_fix_available: bool
    is_fix_deployed: bool
    is_under_attack: bool
    fix_lag_days: Optional[float]


@dataclass
class ArraySource:
    """Source of truth - object arrays for CVDArray."""

    cvss_scores: np.ndarray  # object[N], each is list[CVSSScore]
    epss_scores: np.ndarray  # object[N], each is list[EPSSScore]
    cwes: np.ndarray  # object[N], each is list[CWEEntry]
    cpes: np.ndarray  # object[N], each is list[str]
    kev: np.ndarray  # object[N], each is KEVEntry | None
    exploits: np.ndarray  # object[N], each is list[ExploitReference]

    def __getitem__(self, key: Any) -> "ArraySource":
        """Slice all arrays consistently."""
        return ArraySource(
            cvss_scores=self.cvss_scores[key],
            epss_scores=self.epss_scores[key],
            cwes=self.cwes[key],
            cpes=self.cpes[key],
            kev=self.kev[key],
            exploits=self.exploits[key],
        )


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
class VulnerabilityIdentity:
    """Identity information for a vulnerability.

    Attributes:
        internal_id: Internal UUID, auto-generated if not provided.
        cve_id: CVE identifier (e.g., 'CVE-2024-12345').
    """

    internal_id: str = field(default_factory=lambda: str(uuid4()))
    cve_id: Optional[str] = None


# ==================== ARRAY DATACLASSES ====================


@dataclass
class ArrayState:
    """CVD state bitmask array. Bit positions: V=0, F=1, D=2, P=3, X=4, A=5."""

    bitmask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.uint8))

    def __getitem__(self, key: Any) -> "ArrayState":
        """Slice array consistently."""
        return ArrayState(bitmask=self.bitmask[key])

    def __len__(self) -> int:
        """Number of vulnerabilities."""
        return len(self.bitmask)


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
    """Identifier arrays for vulnerabilities.

    Attributes:
        internal_id: Internal UUIDs (object dtype).
        cve_id: CVE identifiers (object dtype).
    """

    internal_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))
    cve_id: np.ndarray = field(default_factory=lambda: np.array([], dtype=object))

    def __getitem__(self, key: Any) -> "ArrayIdentifiers":
        """Slice all arrays consistently."""
        return ArrayIdentifiers(
            internal_id=self.internal_id[key],
            cve_id=self.cve_id[key],
        )


@dataclass
class ArrayCVDAnalytics:
    """
    Precomputed bitmasks for fast vectorized queries.

    Computed once during array construction or sync(), reused for all queries.
    Derived from ArrayState + ArrayTimestamps - not independent source data.

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
class AnalysisResult:
    """All computed analytics from DesiderataExtractor (Groups 1-11)."""

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
class ArrayMetadata:
    """Metadata and backward compatibility fields."""

    raw: dict[str, np.ndarray] = field(default_factory=dict)
    event_timestamps_absolute: dict[CVDEvent, np.ndarray] = field(
        default_factory=dict
    )  # Deprecated


# ==================== HELPER FUNCTIONS ====================


def compute_pair_mask(state: ArrayState, timestamps: ArrayTimestamps) -> np.ndarray:
    """
    Compute pair ordering mask from timestamps (vectorized).

    Computes which of 15 event pairs occurred in the desired chronological order.
    Returns uint16 bitmask where bit i indicates if pair i is satisfied.

    Args:
        state: ArrayState with bitmask array
        timestamps: ArrayTimestamps with V, F, D, P, X, A timestamp arrays

    Returns:
        np.ndarray[uint16]: Bitmask array, one per vulnerability
            Bit 0 = V≺F, Bit 1 = V≺D, ..., Bit 14 = X≺A

    Example:
        >>> state = ArrayState(bitmask=np.array([0b111111], dtype=np.uint8))
        >>> ts = ArrayTimestamps(...)
        >>> mask = compute_pair_mask(state, ts)
        >>> is_coordinated = (mask & (1 << 2)) != 0  # V≺P bit
    """
    n = len(state)
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
        ("V", "F", 0),  # V≺F
        ("V", "D", 1),  # V≺D
        ("V", "P", 2),  # V≺P
        ("V", "X", 3),  # V≺X
        ("V", "A", 4),  # V≺A
        # F precedes D, P, X, A (bits 5-8)
        ("F", "D", 5),  # F≺D
        ("F", "P", 6),  # F≺P
        ("F", "X", 7),  # F≺X
        ("F", "A", 8),  # F≺A
        # D precedes P, X, A (bits 9-11)
        ("D", "P", 9),  # D≺P
        ("D", "X", 10),  # D≺X
        ("D", "A", 11),  # D≺A
        # P precedes X, A (bits 12-13)
        ("P", "X", 12),  # P≺X
        ("P", "A", 13),  # P≺A
        # X precedes A (bit 14)
        ("X", "A", 14),  # X≺A
    ]

    for earlier_attr, later_attr, bit_pos in pairs:
        earlier_ts = getattr(timestamps, earlier_attr)
        later_ts = getattr(timestamps, later_attr)

        # Both events must have occurred
        both_occurred = ~np.isnat(earlier_ts) & ~np.isnat(later_ts)

        # Earlier must be before later
        correct_order = earlier_ts < later_ts

        # Set bit for vulnerabilities where pair is satisfied
        satisfied = both_occurred & correct_order
        mask[satisfied] |= 1 << bit_pos

    return mask


def compute_history_id(state: ArrayState, timestamps: ArrayTimestamps) -> np.ndarray:
    """
    Compute history IDs from timestamps (0-69 for complete, 255 for incomplete).

    For vulnerabilities with all 6 events, sorts timestamps to get the
    chronological ordering, then looks up the history in VALID_HISTORIES
    to get the canonical history_id (0-69).

    For incomplete histories (fewer than 6 events), returns 255.

    Args:
        state: ArrayState with bitmask array
        timestamps: ArrayTimestamps with V, F, D, P, X, A timestamp arrays

    Returns:
        np.ndarray[uint8]: History ID array
            - 0-69: Index into VALID_HISTORIES for complete histories
            - 255: Incomplete history (not all 6 events occurred)

    Example:
        >>> state = ArrayState(bitmask=np.array([0b111111], dtype=np.uint8))
        >>> ts = ArrayTimestamps(...)  # with timestamps for V, F, D, P, X, A
        >>> ids = compute_history_id(state, ts)
        >>> ids[0]  # 69 for VFDPXA (perfect CVD)
    """
    from .constants import HISTORY_TO_ID, INCOMPLETE_HISTORY_ID

    n = len(state)
    result = np.full(n, INCOMPLETE_HISTORY_ID, dtype=np.uint8)

    # Stack all timestamps: shape (N, 6)
    ts_stack = np.column_stack(
        [
            timestamps.V,
            timestamps.F,
            timestamps.D,
            timestamps.P,
            timestamps.X,
            timestamps.A,
        ]
    )

    # Event labels for building history strings
    event_labels = ["V", "F", "D", "P", "X", "A"]

    # Check which rows have all 6 events (no NaT values)
    complete_mask = ~np.any(np.isnat(ts_stack), axis=1)

    # Process only complete histories
    for i in np.where(complete_mask)[0]:
        # Get timestamps for this vulnerability
        ts = ts_stack[i]

        # Sort indices by timestamp to get chronological order
        order = np.argsort(ts)

        # Build history string
        history = "".join(event_labels[j] for j in order)

        # Look up history_id
        if history in HISTORY_TO_ID:
            result[i] = HISTORY_TO_ID[history]
        # else: remains 255 (invalid history - should not happen with valid data)

    return result


@dataclass
class CVSSMetrics:
    """Parsed CVSS vector metrics for batch operations."""

    attack_vector: np.ndarray  # dtype=object, values: "N", "A", "L", "P" or None
    attack_complexity: np.ndarray
    privileges_required: np.ndarray
    user_interaction: np.ndarray
    scope: np.ndarray
    confidentiality_impact: np.ndarray
    integrity_impact: np.ndarray
    availability_impact: np.ndarray

    @classmethod
    def empty(cls, n: int) -> "CVSSMetrics":
        """Create empty metrics arrays of length n."""
        return cls(
            attack_vector=np.empty(n, dtype=object),
            attack_complexity=np.empty(n, dtype=object),
            privileges_required=np.empty(n, dtype=object),
            user_interaction=np.empty(n, dtype=object),
            scope=np.empty(n, dtype=object),
            confidentiality_impact=np.empty(n, dtype=object),
            integrity_impact=np.empty(n, dtype=object),
            availability_impact=np.empty(n, dtype=object),
        )
