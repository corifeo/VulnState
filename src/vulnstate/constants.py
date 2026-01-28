"""
CVD Model Constants - Domain definitions from SEI/CMU 2021 paper

Provides:
- CVDEvent: IntEnum with V, F, D, P, X, A events (0-5)
- STATE_NOTATION: Lookup table for state int -> notation string
- NOTATION_TO_STATE: Reverse lookup for notation string -> state int
- VALID_STATES: Set of 32 valid states satisfying V→F→D constraint
- AntiDesiderataBit: 12 undesirable orderings
- FixPath: Fix path progression (VFD dimension)
- ThreatState: Threat materialization (PXA dimension)
- HistoryValidity: History validation states
- EventPairRelation: Pair relationship types
- DESIDERATA_PAIRS: 12 desired event orderings
- ORDERED_PAIRS_TABLE: Complete 36-pair relationship table
- State conversion functions: string_to_state_int, state_int_to_string, etc.

Layer: Core
Dependencies: None (self-contained)
Used by: analyzer.py, array.py, vulnerability.py, io.py

Based on SEI/CMU 2021 paper:
"A State-Based Model for Multi-Party Coordinated Vulnerability Disclosure (MPCVD)"
by Allen Householder and Jonathan Spring
"""

from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class CVDEvent(IntEnum):
    """CVD lifecycle events as IntEnum for efficient bitmask operations."""

    V = 0  # Vendor awareness
    F = 1  # Fix ready
    D = 2  # Fix deployed
    P = 3  # Public awareness
    X = 4  # Exploit public
    A = 5  # Attacks observed


EVENT_LABELS: dict[int, str] = {
    CVDEvent.V: "Vendor Awareness",
    CVDEvent.F: "Fix Ready",
    CVDEvent.D: "Fix Deployed",
    CVDEvent.P: "Public Awareness",
    CVDEvent.X: "Exploit Public",
    CVDEvent.A: "Attacks Observed",
}

EVENT_DESCRIPTIONS: dict[int, str] = {
    CVDEvent.V: "The vendor becomes aware of the vulnerability",
    CVDEvent.F: "A fix or patch is ready for distribution",
    CVDEvent.D: "The fix has been deployed to affected systems",
    CVDEvent.P: "The vulnerability is publicly disclosed",
    CVDEvent.X: "A working exploit is publicly available",
    CVDEvent.A: "Active exploitation attempts are observed in the wild",
}


def validate_vfd_constraint(event: CVDEvent, current_state_int: int) -> bool:
    """
    Validate V→F→D constraint for event application.

    Checks if an event can be applied to a vulnerability given its current state.
    Enforces the mandatory constraint: V → F → D

    Args:
        event: The event to validate
        current_state_int: Current state as integer (uint8)

    Returns:
        True if event can be applied, False if constraint violated
    """
    if event == CVDEvent.F:
        # F requires V to have occurred
        v_occurred = bool(current_state_int & (1 << CVDEvent.V))
        return v_occurred
    elif event == CVDEvent.D:
        # D requires both V and F to have occurred
        v_occurred = bool(current_state_int & (1 << CVDEvent.V))
        f_occurred = bool(current_state_int & (1 << CVDEvent.F))
        return v_occurred and f_occurred
    else:
        # V, P, X, A have no constraints
        return True


def _build_state_notation(state: int) -> str:
    """Build notation string from state int."""
    chars = []
    for i, name in enumerate("VFDPXA"):
        chars.append(name if state & (1 << i) else name.lower())
    return "".join(chars)


STATE_NOTATION: dict[int, str] = {i: _build_state_notation(i) for i in range(64)}

# Reverse lookup: string -> int
NOTATION_TO_STATE: dict[str, int] = {v: k for k, v in STATE_NOTATION.items()}


def _build_valid_states() -> frozenset[int]:
    """Build set of 32 valid states (satisfy V→F→D constraint)."""
    valid = set()
    for state_int in range(64):
        v = bool(state_int & 1)
        f = bool(state_int & 2)
        d = bool(state_int & 4)
        # V→F→D constraint: (not f or v) and (not d or f)
        if (not f or v) and (not d or f):
            valid.add(state_int)
    return frozenset(valid)


VALID_STATES: frozenset[int] = _build_valid_states()


class AntiDesiderataBit(IntEnum):
    """Bit positions for 12 anti-desiderata (undesirable orderings)."""

    U1_P_V = 0  # P < V: Uncoordinated disclosure
    U2_X_V = 1  # X < V: Zero-day exploit
    U3_A_V = 2  # A < V: Zero-day attack
    U4_P_F = 3  # P < F: Premature disclosure
    U5_X_F = 4  # X < F: Exploit before fix
    U6_A_F = 5  # A < F: Attacks before fix
    U7_P_D = 6  # P < D: Disclosed before remediation
    U8_X_D = 7  # X < D: Exploit before remediation
    U9_A_D = 8  # A < D: Attacks before remediation
    U10_X_P = 9  # X < P: Exploit before disclosure
    U11_A_P = 10  # A < P: Attacks before disclosure
    U12_A_X = 11  # A < X: Private attacks


ANTI_DESIDERATA_LABELS: dict[int, str] = {
    AntiDesiderataBit.U1_P_V: "Uncoordinated Disclosure",
    AntiDesiderataBit.U2_X_V: "Zero-Day Exploit",
    AntiDesiderataBit.U3_A_V: "Zero-Day Attack",
    AntiDesiderataBit.U4_P_F: "Premature Disclosure",
    AntiDesiderataBit.U5_X_F: "Exploit Before Fix",
    AntiDesiderataBit.U6_A_F: "Attacks Before Fix",
    AntiDesiderataBit.U7_P_D: "Disclosed Before Remediation",
    AntiDesiderataBit.U8_X_D: "Exploit Before Remediation",
    AntiDesiderataBit.U9_A_D: "Attacks Before Remediation",
    AntiDesiderataBit.U10_X_P: "Exploit Before Disclosure",
    AntiDesiderataBit.U11_A_P: "Attacks Before Disclosure",
    AntiDesiderataBit.U12_A_X: "Private Attacks",
}

ANTI_DESIDERATA_DESCRIPTIONS: dict[int, str] = {
    AntiDesiderataBit.U1_P_V: "Vulnerability disclosed before vendor was aware",
    AntiDesiderataBit.U2_X_V: "Exploit available before vendor awareness (zero-day)",
    AntiDesiderataBit.U3_A_V: "Attacks observed before vendor awareness (zero-day)",
    AntiDesiderataBit.U4_P_F: "Vulnerability disclosed before fix was ready",
    AntiDesiderataBit.U5_X_F: "Exploit available before fix was ready",
    AntiDesiderataBit.U6_A_F: "Attacks observed before fix was ready",
    AntiDesiderataBit.U7_P_D: "Vulnerability disclosed before fix was deployed",
    AntiDesiderataBit.U8_X_D: "Exploit available before fix was deployed",
    AntiDesiderataBit.U9_A_D: "Attacks observed before fix was deployed",
    AntiDesiderataBit.U10_X_P: "Exploit available before public disclosure",
    AntiDesiderataBit.U11_A_P: "Attacks observed before public disclosure",
    AntiDesiderataBit.U12_A_X: "Attacks without public exploit (APT-style)",
}


class DesiderataBit(IntEnum):
    """Bit positions for 12 desiderata (desired orderings).

    Parallel to AntiDesiderataBit - each D{n} is the complement of U{n}.
    D1 (V<P) is satisfied when U1 (P<V) is violated, and vice versa.
    """

    D1_V_P = 0  # V < P: Coordinated disclosure
    D2_V_X = 1  # V < X: Vendor aware before exploit
    D3_V_A = 2  # V < A: Vendor aware before attacks
    D4_F_P = 3  # F < P: Fix ready before disclosure
    D5_F_X = 4  # F < X: Fix ready before exploit
    D6_F_A = 5  # F < A: Fix ready before attacks
    D7_D_P = 6  # D < P: Deployed before disclosure
    D8_D_X = 7  # D < X: Deployed before exploit
    D9_D_A = 8  # D < A: Deployed before attacks
    D10_P_X = 9  # P < X: Disclosure before exploit
    D11_P_A = 10  # P < A: Disclosure before attacks
    D12_X_A = 11  # X < A: Exploit before attacks


DESIDERATA_LABELS: dict[int, str] = {
    DesiderataBit.D1_V_P: "Coordinated Disclosure",
    DesiderataBit.D2_V_X: "Vendor Aware Before Exploit",
    DesiderataBit.D3_V_A: "Vendor Aware Before Attacks",
    DesiderataBit.D4_F_P: "Fix Ready Before Disclosure",
    DesiderataBit.D5_F_X: "Fix Ready Before Exploit",
    DesiderataBit.D6_F_A: "Fix Ready Before Attacks",
    DesiderataBit.D7_D_P: "Deployed Before Disclosure",
    DesiderataBit.D8_D_X: "Deployed Before Exploit",
    DesiderataBit.D9_D_A: "Deployed Before Attacks",
    DesiderataBit.D10_P_X: "Disclosure Before Exploit",
    DesiderataBit.D11_P_A: "Disclosure Before Attacks",
    DesiderataBit.D12_X_A: "Exploit Before Attacks",
}

DESIDERATA_DESCRIPTIONS: dict[int, str] = {
    DesiderataBit.D1_V_P: "Vendor was aware before public disclosure",
    DesiderataBit.D2_V_X: "Vendor was aware before exploit became available",
    DesiderataBit.D3_V_A: "Vendor was aware before attacks were observed",
    DesiderataBit.D4_F_P: "Fix was ready before public disclosure",
    DesiderataBit.D5_F_X: "Fix was ready before exploit became available",
    DesiderataBit.D6_F_A: "Fix was ready before attacks were observed",
    DesiderataBit.D7_D_P: "Fix was deployed before public disclosure",
    DesiderataBit.D8_D_X: "Fix was deployed before exploit became available",
    DesiderataBit.D9_D_A: "Fix was deployed before attacks were observed",
    DesiderataBit.D10_P_X: "Vulnerability was disclosed before exploit became available",
    DesiderataBit.D11_P_A: "Vulnerability was disclosed before attacks were observed",
    DesiderataBit.D12_X_A: "Exploit was available before attacks (known threat)",
}

# Maps each DesiderataBit to its (earlier, later) event pair
DESIDERATA_PAIR_MAP: dict[int, tuple[CVDEvent, CVDEvent]] = {
    DesiderataBit.D1_V_P: (CVDEvent.V, CVDEvent.P),
    DesiderataBit.D2_V_X: (CVDEvent.V, CVDEvent.X),
    DesiderataBit.D3_V_A: (CVDEvent.V, CVDEvent.A),
    DesiderataBit.D4_F_P: (CVDEvent.F, CVDEvent.P),
    DesiderataBit.D5_F_X: (CVDEvent.F, CVDEvent.X),
    DesiderataBit.D6_F_A: (CVDEvent.F, CVDEvent.A),
    DesiderataBit.D7_D_P: (CVDEvent.D, CVDEvent.P),
    DesiderataBit.D8_D_X: (CVDEvent.D, CVDEvent.X),
    DesiderataBit.D9_D_A: (CVDEvent.D, CVDEvent.A),
    DesiderataBit.D10_P_X: (CVDEvent.P, CVDEvent.X),
    DesiderataBit.D11_P_A: (CVDEvent.P, CVDEvent.A),
    DesiderataBit.D12_X_A: (CVDEvent.X, CVDEvent.A),
}

# Maps each AntiDesiderataBit to its (later, earlier) event pair (violations)
ANTI_DESIDERATA_PAIR_MAP: dict[int, tuple[CVDEvent, CVDEvent]] = {
    AntiDesiderataBit.U1_P_V: (CVDEvent.P, CVDEvent.V),
    AntiDesiderataBit.U2_X_V: (CVDEvent.X, CVDEvent.V),
    AntiDesiderataBit.U3_A_V: (CVDEvent.A, CVDEvent.V),
    AntiDesiderataBit.U4_P_F: (CVDEvent.P, CVDEvent.F),
    AntiDesiderataBit.U5_X_F: (CVDEvent.X, CVDEvent.F),
    AntiDesiderataBit.U6_A_F: (CVDEvent.A, CVDEvent.F),
    AntiDesiderataBit.U7_P_D: (CVDEvent.P, CVDEvent.D),
    AntiDesiderataBit.U8_X_D: (CVDEvent.X, CVDEvent.D),
    AntiDesiderataBit.U9_A_D: (CVDEvent.A, CVDEvent.D),
    AntiDesiderataBit.U10_X_P: (CVDEvent.X, CVDEvent.P),
    AntiDesiderataBit.U11_A_P: (CVDEvent.A, CVDEvent.P),
    AntiDesiderataBit.U12_A_X: (CVDEvent.A, CVDEvent.X),
}

# Severity masks
ZERO_DAY_MASK: int = (1 << AntiDesiderataBit.U2_X_V) | (1 << AntiDesiderataBit.U3_A_V)
CRITICAL_EXPOSURE_MASK: int = (1 << AntiDesiderataBit.U6_A_F) | (1 << AntiDesiderataBit.U9_A_D)
COORDINATION_FAILURE_MASK: int = (1 << AntiDesiderataBit.U1_P_V) | (1 << AntiDesiderataBit.U4_P_F)

# Probability defaults
DEFAULT_XA_SPLIT: float = 0.5  # Equal split between X and A
DEFAULT_THREAT_MULTIPLIER: float = 1.0  # No multiplier


# ==================== DESIDERATA HELPERS ====================


def decode_desiderata_mask(mask: int) -> list[DesiderataBit]:
    """Return list of satisfied desiderata (bits set in mask)."""
    return [bit for bit in DesiderataBit if mask & (1 << bit)]


def decode_anti_desiderata_mask(mask: int) -> list[AntiDesiderataBit]:
    """Return list of violated anti-desiderata (bits set in mask)."""
    return [bit for bit in AntiDesiderataBit if mask & (1 << bit)]


def get_desiderata_labels(mask: int) -> list[str]:
    """Return human-readable labels for satisfied desiderata."""
    return [DESIDERATA_LABELS[bit] for bit in decode_desiderata_mask(mask)]


def get_anti_desiderata_labels(mask: int) -> list[str]:
    """Return human-readable labels for violated anti-desiderata."""
    return [ANTI_DESIDERATA_LABELS[bit] for bit in decode_anti_desiderata_mask(mask)]


def get_desiderata_descriptions(mask: int) -> list[str]:
    """Return full descriptions for satisfied desiderata."""
    return [DESIDERATA_DESCRIPTIONS[bit] for bit in decode_desiderata_mask(mask)]


def get_anti_desiderata_descriptions(mask: int) -> list[str]:
    """Return full descriptions for violated anti-desiderata."""
    return [ANTI_DESIDERATA_DESCRIPTIONS[bit] for bit in decode_anti_desiderata_mask(mask)]


# ==================== STATE DIMENSIONS ====================
#
# The 6-bit CVD state encodes two independent dimensions:
#
# VFD (bits 0-2): Fix Path - Process-driven vendor response
#   Managed through CVD coordination. Progression: vfd → Vfd → VFd → VFD
#   Constraint: V → F → D (must occur in order)
#
# PXA (bits 3-5): Threat Materialization - Probabilistic adversary activity
#   Driven by EPSS probability and real-world exploitation.
#   No strict ordering constraint, but some states are unstable.
#
# Combined state = (vfd_bits) | (pxa_bits << 3)
# Example: VFdPXa (Fix ready, Weaponized) = 0b011_011 = 27
# ============================================================


class FixPath(IntEnum):
    """
    VFD dimension: Fix path progression (bits 0-2).

    Process-driven vendor response to vulnerability. Managed through CVD coordination.
    Values are bitmasks matching the state encoding.

    Progression: NO_AWARENESS → VENDOR_AWARE → FIX_READY → REMEDIATED
    Constraint: Each stage requires previous (V → F → D).

    Attributes:
        NO_AWARENESS (0b000): Vendor unaware, no fix possible (vfd)
        VENDOR_AWARE (0b001): Aware, fix in development (Vfd)
        FIX_READY (0b011): Patch available, not deployed (VFd)
        REMEDIATED (0b111): Fix deployed to systems (VFD)

    Example:
        >>> from vulnstate.constants import FixPath
        >>> state_vfd_bits = 0b011
        >>> stage = FixPath(state_vfd_bits)
        >>> stage.label
        'Fix Ready'
    """

    NO_AWARENESS = 0b000  # vfd - Vendor unaware
    VENDOR_AWARE = 0b001  # Vfd - Aware, fix in development
    FIX_READY = 0b011  # VFd - Patch available
    REMEDIATED = 0b111  # VFD - Fix deployed

    @property
    def label(self) -> str:
        """Human-readable label for display."""
        return FIX_PATH_LABELS[self.value]

    @property
    def state_string(self) -> str:
        """CVD state string representation (vfd, Vfd, VFd, VFD)."""
        return _FIX_PATH_STATE_STRINGS[self.value]


class ThreatState(IntEnum):
    """
    PXA dimension: Threat materialization (bits 3-5).

    Probabilistic adversary activity. Driven by EPSS probability and real-world
    exploitation. No strict ordering constraint, but some states (pX) are unstable
    and auto-transition to PX.

    Attributes:
        LATENT (0b000): No threat activity observed (pxa)
        DISCLOSED (0b001): Public awareness, no exploitation (Pxa)
        WEAPONIZED (0b011): Exploit available, no attacks yet (PXa)
        TARGETED (0b100): Private attacks, APT-style (pxA)
        UNDER_ATTACK (0b101): Attacks without public exploit (PxA)
        ACTIVE_THREAT (0b111): Full threat materialization (PXA)

    Example:
        >>> from vulnstate.constants import ThreatState
        >>> state_pxa_bits = (state >> 3) & 0b111
        >>> threat = ThreatState(state_pxa_bits)
        >>> threat.label
        'Weaponized'
    """

    LATENT = 0b000  # pxa - No threat activity
    DISCLOSED = 0b001  # Pxa - Public awareness only
    WEAPONIZED = 0b011  # PXa - Exploit available
    TARGETED = 0b100  # pxA - Private attacks (APT)
    UNDER_ATTACK = 0b101  # PxA - Attacks without public exploit
    ACTIVE_THREAT = 0b111  # PXA - Full threat

    @property
    def label(self) -> str:
        """Human-readable label for display."""
        return THREAT_LABELS[self.value]

    @property
    def state_string(self) -> str:
        """CVD state string representation (pxa, Pxa, PXa, etc.)."""
        return _THREAT_STATE_STRINGS[self.value]


class HistoryValidity(IntEnum):
    """
    Validity state for vulnerability history.

    Attributes:
        VALID (0): All observed events satisfy V→F→D constraint
        IMPOSSIBLE (1): Observed events violate constraints (e.g., F before V)
    """

    VALID = 0
    IMPOSSIBLE = 1


# Lookup tables for fast vectorized conversion (bitmask → label)
FIX_PATH_LABELS: dict[int, str] = {
    FixPath.NO_AWARENESS: "No Awareness",
    FixPath.VENDOR_AWARE: "Vendor Aware",
    FixPath.FIX_READY: "Fix Ready",
    FixPath.REMEDIATED: "Remediated",
}

THREAT_LABELS: dict[int, str] = {
    ThreatState.LATENT: "Latent",
    ThreatState.DISCLOSED: "Disclosed",
    ThreatState.WEAPONIZED: "Weaponized",
    ThreatState.TARGETED: "Targeted",
    ThreatState.UNDER_ATTACK: "Under Attack",
    ThreatState.ACTIVE_THREAT: "Active Threat",
}

# State string lookup tables
_FIX_PATH_STATE_STRINGS: dict[int, str] = {
    FixPath.NO_AWARENESS: "vfd",
    FixPath.VENDOR_AWARE: "Vfd",
    FixPath.FIX_READY: "VFd",
    FixPath.REMEDIATED: "VFD",
}

_THREAT_STATE_STRINGS: dict[int, str] = {
    ThreatState.LATENT: "pxa",
    ThreatState.DISCLOSED: "Pxa",
    ThreatState.WEAPONIZED: "PXa",
    ThreatState.TARGETED: "pxA",
    ThreatState.UNDER_ATTACK: "PxA",
    ThreatState.ACTIVE_THREAT: "PXA",
}


# ==================== PAIR BITMASK CONSTANTS ====================
#
# The pair_mask is a uint16 bitmask encoding which of 15 event pairs occurred
# in their "desired" chronological order. This enables O(1) vectorized queries
# for CVD analytics like is_zero_day, is_coordinated, etc.
#
# ENCODING:
#   - Bit i = 1 if pair i occurred in desired order (earlier_ts < later_ts)
#   - Bit i = 0 if pair violated, events simultaneous, or one/both events missing
#
# BIT LAYOUT (15 bits, fits in uint16):
#   Bit 0: V≺F    Bit 5: F≺D    Bit 9:  D≺P    Bit 12: P≺X
#   Bit 1: V≺D    Bit 6: F≺P    Bit 10: D≺X    Bit 13: P≺A
#   Bit 2: V≺P    Bit 7: F≺X    Bit 11: D≺A    Bit 14: X≺A
#   Bit 3: V≺X    Bit 8: F≺A
#   Bit 4: V≺A
#
# CLASSIFICATION:
#   - Bits 0, 1, 5 (V≺F, V≺D, F≺D): Constraint pairs - always satisfied in valid states
#   - Bits 2-4, 6-14: Desiderata pairs - the 12 "desired" orderings from CVD model
#
# USAGE EXAMPLES:
#   >>> # Check if coordinated (V before P)
#   >>> is_coordinated = (pair_mask & (1 << 2)) != 0
#
#   >>> # Check if zero-day exploit (X before V = V≺X violated)
#   >>> is_zero_day_exploit = (pair_mask & (1 << 3)) == 0
#
#   >>> # Count satisfied desiderata (12 bits, excluding constraints)
#   >>> DESIDERATA_MASK = 0b0111_1111_1111_1100  # Bits 2-14
#   >>> count = bin(pair_mask & DESIDERATA_MASK).count('1')
#
# See: docs/design/2026-01-20-cvd-state-storage-architecture.md
# See: models.ArrayCVDAnalytics for storage, models.compute_pair_mask() for computation

# All 15 event pairs with bit positions
# Order: V-F, V-D, V-P, V-X, V-A, F-D, F-P, F-X, F-A, D-P, D-X, D-A, P-X, P-A, X-A
_PAIR_ORDER: list[tuple[CVDEvent, CVDEvent]] = [
    (CVDEvent.V, CVDEvent.F),  # 0
    (CVDEvent.V, CVDEvent.D),  # 1
    (CVDEvent.V, CVDEvent.P),  # 2
    (CVDEvent.V, CVDEvent.X),  # 3
    (CVDEvent.V, CVDEvent.A),  # 4
    (CVDEvent.F, CVDEvent.D),  # 5
    (CVDEvent.F, CVDEvent.P),  # 6
    (CVDEvent.F, CVDEvent.X),  # 7
    (CVDEvent.F, CVDEvent.A),  # 8
    (CVDEvent.D, CVDEvent.P),  # 9
    (CVDEvent.D, CVDEvent.X),  # 10
    (CVDEvent.D, CVDEvent.A),  # 11
    (CVDEvent.P, CVDEvent.X),  # 12
    (CVDEvent.P, CVDEvent.A),  # 13
    (CVDEvent.X, CVDEvent.A),  # 14
]

PAIR_BIT_POSITIONS: dict[tuple[CVDEvent, CVDEvent], int] = {
    pair: i for i, pair in enumerate(_PAIR_ORDER)
}

PAIR_NAMES: dict[int, str] = {
    i: f"{pair[0].name}≺{pair[1].name}" for i, pair in enumerate(_PAIR_ORDER)
}

# Required pairs mask: V≺F (0), V≺D (1), F≺D (5) - constraint pairs
REQUIRED_PAIRS_MASK: int = (1 << 0) | (1 << 1) | (1 << 5)  # 0b100011 = 35

# Desiderata mask: all 12 desiderata pairs (bits 2-4, 6-14)
# Use this to extract only desiderata-relevant bits from pair_mask
DESIDERATA_MASK: int = 0b0111_1111_1111_1100  # 0x7FFC = 32764

# Desiderata pairs bit positions (12 desired orderings)
# Maps DESIDERATA_PAIRS to their bit positions in pair_mask
DESIDERATA_PAIRS_BITS: list[int] = [
    2,  # V≺P
    3,  # V≺X
    4,  # V≺A
    6,  # F≺P
    7,  # F≺X
    8,  # F≺A
    9,  # D≺P
    10,  # D≺X
    11,  # D≺A
    12,  # P≺X
    13,  # P≺A
    14,  # X≺A
]


# ==================== VALID HISTORIES ====================
#
# The 70 valid complete disclosure histories (all 6 events in order).
# These are the only possible orderings when the V→F→D constraint is enforced.
# Index 0-69 corresponds to history_id field in ArrayCVDAnalytics.
#
# Ordered from worst (AXPVFD, rank 1) to best (VFDPXA, rank 62, perfect CVD).
# Value 255 is reserved for incomplete histories (fewer than 6 events).
#
# See: docs/ref/cvd-histories.md for detailed ranking and interpretation
VALID_HISTORIES: tuple[str, ...] = (
    # Adverse outcomes (0-8) - rank 1-8, 0-4 desiderata
    "AXPVFD",  # 0: rank 1, worst case - attacks first
    "APVXFD",  # 1: rank 2
    "AVXPFD",  # 2: rank 3
    "XPVAFD",  # 3: rank 4
    "VAXPFD",  # 4: rank 5
    "PVAXFD",  # 5: rank 6
    "AVPXFD",  # 6: rank 7
    "APVFXD",  # 7: rank 7
    "XPVFAD",  # 8: rank 8
    # Poor outcomes (9-24) - rank 9-21, 4-6 desiderata
    "VAPXFD",  # 9: rank 9
    "PVXAFD",  # 10: rank 10
    "VPAXFD",  # 11: rank 11
    "PVAFXD",  # 12: rank 11
    "VXPAFD",  # 13: rank 11
    "AVPFXD",  # 14: rank 12
    "APVFDX",  # 15: rank 13
    "VAPFXD",  # 16: rank 14
    "XPVFDA",  # 17: rank 15
    "PVXFAD",  # 18: rank 16
    "AVFXPD",  # 19: rank 17
    "VPXAFD",  # 20: rank 18
    "PVFAXD",  # 21: rank 19
    "VXPFAD",  # 22: rank 19
    "VPAFXD",  # 23: rank 20
    "VAFXPD",  # 24: rank 21
    # Acceptable outcomes (25-44) - rank 22-39, 5-8 desiderata
    "PVAFDX",  # 25: rank 22
    "AVPFDX",  # 26: rank 23
    "AVFPXD",  # 27: rank 24
    "PVFXAD",  # 28: rank 25
    "VPXFAD",  # 29: rank 25
    "VAPFDX",  # 30: rank 26
    "VAFPXD",  # 31: rank 27
    "PVXFDA",  # 32: rank 28
    "VPFAXD",  # 33: rank 29
    "VFAXPD",  # 34: rank 30
    "VXPFDA",  # 35: rank 31
    "PVFADX",  # 36: rank 32
    "VPAFDX",  # 37: rank 33
    "VPFXAD",  # 38: rank 34
    "AVFPDX",  # 39: rank 35
    "VFAPXD",  # 40: rank 36
    "VPXFDA",  # 41: rank 37
    "PVFXDA",  # 42: rank 37
    "VAFPDX",  # 43: rank 38
    "VPFADX",  # 44: rank 39
    # Good outcomes (45-58) - rank 40-52, 6-10 desiderata
    "VFPAXD",  # 45: rank 40
    "VFXPAD",  # 46: rank 41
    "AVFDXP",  # 47: rank 42
    "PVFDAX",  # 48: rank 43
    "VAFDXP",  # 49: rank 44
    "VPFXDA",  # 50: rank 45
    "VFAPDX",  # 51: rank 46
    "VFPXAD",  # 52: rank 46
    "AVFDPX",  # 53: rank 47
    "PVFDXA",  # 54: rank 48
    "VPFDAX",  # 55: rank 49
    "VFXPDA",  # 56: rank 50
    "VFPADX",  # 57: rank 51
    "VAFDPX",  # 58: rank 52
    # Ideal outcomes (59-69) - rank 53-62, 8-12 desiderata
    "VFADXP",  # 59: rank 53
    "VPFDXA",  # 60: rank 54
    "VFPXDA",  # 61: rank 55
    "VFADPX",  # 62: rank 56
    "VFPDAX",  # 63: rank 57
    "VFDAXP",  # 64: rank 58 - silent fix ideal
    "VFPDXA",  # 65: rank 59
    "VFDAPX",  # 66: rank 60
    "VFDXPA",  # 67: rank 61
    "VFDPAX",  # 68: rank 61
    "VFDPXA",  # 69: rank 62 - perfect CVD
)

# Reverse lookup: history string -> history_id (0-69)
HISTORY_TO_ID: dict[str, int] = {h: i for i, h in enumerate(VALID_HISTORIES)}

# Sentinel value for incomplete histories (fewer than 6 events)
INCOMPLETE_HISTORY_ID: int = 255


class EventPairRelation(Enum):
    """
    Event pair relationship types from Table 3.3 (research tier).

    Based on SEI/CMU 2021 paper Table 3.3 "Ordered Pairs Table".
    Classifies relationships between event pairs (e.g., V→P, F→X).

    This is part of the research tier API - requires knowledge of CVD paper
    concepts and is used for academic analysis of vulnerability disclosure
    histories.

    Attributes:
        IMPOSSIBLE: Violates state machine constraints (symbol: -)
                   Example: F before V violates V→F→D constraint
        REQUIRED: Mandatory ordering enforced by state machine (symbol: r)
                 Example: V must precede F, F must precede D
        DESIRED: Preferred ordering (12 desiderata) (symbol: d)
                Example: V≺P (vendor before public)
        UNDESIRED: Complement of desired orderings (symbol: u)
                  Example: P before V

    Example:
        >>> from vulnstate.constants import EventPairRelation
        >>> rel = EventPairRelation.DESIRED
        >>> rel.value
        'desired'
    """

    IMPOSSIBLE = "impossible"  # Violates constraints (-)
    REQUIRED = "required"  # Mandatory ordering (r)
    DESIRED = "desired"  # Desiderata (d)
    UNDESIRED = "undesired"  # Complement of desired (u)


# Twelve desiderata (desired event orderings) from SEI/CMU 2021 paper
DESIDERATA_PAIRS: list[tuple[CVDEvent, CVDEvent]] = [
    # Vendor before public awareness (basic coordinated disclosure)
    (CVDEvent.V, CVDEvent.P),  # V≺P
    # Fix before exploitation
    (CVDEvent.F, CVDEvent.X),  # F≺X
    # Deployment before attacks
    (CVDEvent.D, CVDEvent.A),  # D≺A
    # Public awareness before exploitation
    (CVDEvent.P, CVDEvent.X),  # P≺X
    # Public awareness before attacks
    (CVDEvent.P, CVDEvent.A),  # P≺A
    # Exploitation before attacks (12th desideratum - was missing)
    (CVDEvent.X, CVDEvent.A),  # X≺A
    # Additional desiderata
    (CVDEvent.V, CVDEvent.X),  # V≺X
    (CVDEvent.V, CVDEvent.A),  # V≺A
    (CVDEvent.F, CVDEvent.P),  # F≺P
    (CVDEvent.F, CVDEvent.A),  # F≺A
    (CVDEvent.D, CVDEvent.X),  # D≺X
    (CVDEvent.D, CVDEvent.P),  # D≺P
]
"""
Twelve desiderata (desired event orderings) from SEI/CMU 2021 paper.

These represent ideal CVD practices where earlier events should
precede later ones. Violations indicate suboptimal disclosure patterns.

Note: X≺A (exploit before attacks) is the 12th desideratum,
previously missing from the implementation.
"""

# Ordered Pairs Table from SEI/CMU 2021 Table 3.3
ORDERED_PAIRS_TABLE: dict[tuple[CVDEvent, CVDEvent], EventPairRelation] = {
    # V→F→D constraint (REQUIRED)
    (CVDEvent.V, CVDEvent.F): EventPairRelation.REQUIRED,
    (CVDEvent.F, CVDEvent.D): EventPairRelation.REQUIRED,
    (CVDEvent.V, CVDEvent.D): EventPairRelation.REQUIRED,  # Transitive
    # Reverse of required = IMPOSSIBLE
    (CVDEvent.F, CVDEvent.V): EventPairRelation.IMPOSSIBLE,
    (CVDEvent.D, CVDEvent.V): EventPairRelation.IMPOSSIBLE,
    (CVDEvent.D, CVDEvent.F): EventPairRelation.IMPOSSIBLE,
    # DESIDERATA (12 pairs marked DESIRED)
    (CVDEvent.V, CVDEvent.P): EventPairRelation.DESIRED,
    (CVDEvent.F, CVDEvent.X): EventPairRelation.DESIRED,
    (CVDEvent.D, CVDEvent.A): EventPairRelation.DESIRED,
    (CVDEvent.P, CVDEvent.X): EventPairRelation.DESIRED,
    (CVDEvent.P, CVDEvent.A): EventPairRelation.DESIRED,
    (CVDEvent.X, CVDEvent.A): EventPairRelation.DESIRED,  # 12th
    (CVDEvent.V, CVDEvent.X): EventPairRelation.DESIRED,
    (CVDEvent.V, CVDEvent.A): EventPairRelation.DESIRED,
    (CVDEvent.F, CVDEvent.P): EventPairRelation.DESIRED,
    (CVDEvent.F, CVDEvent.A): EventPairRelation.DESIRED,
    (CVDEvent.D, CVDEvent.X): EventPairRelation.DESIRED,
    (CVDEvent.D, CVDEvent.P): EventPairRelation.DESIRED,
    # Complements of desiderata = UNDESIRED
    (CVDEvent.P, CVDEvent.V): EventPairRelation.UNDESIRED,
    (CVDEvent.X, CVDEvent.F): EventPairRelation.UNDESIRED,
    (CVDEvent.A, CVDEvent.D): EventPairRelation.UNDESIRED,
    (CVDEvent.X, CVDEvent.P): EventPairRelation.UNDESIRED,
    (CVDEvent.A, CVDEvent.P): EventPairRelation.UNDESIRED,
    (CVDEvent.A, CVDEvent.X): EventPairRelation.UNDESIRED,  # Complement of X≺A
    (CVDEvent.X, CVDEvent.V): EventPairRelation.UNDESIRED,
    (CVDEvent.A, CVDEvent.V): EventPairRelation.UNDESIRED,
    (CVDEvent.P, CVDEvent.F): EventPairRelation.UNDESIRED,
    (CVDEvent.A, CVDEvent.F): EventPairRelation.UNDESIRED,
    (CVDEvent.X, CVDEvent.D): EventPairRelation.UNDESIRED,
    (CVDEvent.P, CVDEvent.D): EventPairRelation.UNDESIRED,
    # Self-pairs (event to itself) - no value in same-event ordering
    (CVDEvent.V, CVDEvent.V): EventPairRelation.UNDESIRED,
    (CVDEvent.F, CVDEvent.F): EventPairRelation.UNDESIRED,
    (CVDEvent.D, CVDEvent.D): EventPairRelation.UNDESIRED,
    (CVDEvent.P, CVDEvent.P): EventPairRelation.UNDESIRED,
    (CVDEvent.X, CVDEvent.X): EventPairRelation.UNDESIRED,
    (CVDEvent.A, CVDEvent.A): EventPairRelation.UNDESIRED,
}
"""
Ordered Pairs Table from SEI/CMU 2021 Table 3.3.

Maps all 36 possible event pairs (6×6) to their relationship type:
- REQUIRED (3): Mandatory V→F→D constraint
- IMPOSSIBLE (3): Constraint violations (F→V, D→F, D→V)
- DESIRED (12): The 12 desiderata
- UNDESIRED (18): Complements of desired + self-pairs

This comprehensive table includes ALL valid pairs, not just desiderata,
providing a complete reference for event relationship analysis that can
be used for both research and operational purposes.

Resolves histories.py:15 TODO.
"""


# ==================== STATE CONVERSION FUNCTIONS ====================
# Replaces CVDStateEncoder class from states.py


def string_to_state_int(state_str: str) -> int:
    """
    Convert state string to integer encoding.

    Args:
        state_str: State string like 'VFdpXa'

    Returns:
        Integer encoding (0-63)

    Raises:
        ValueError: If state_str is invalid or violates V→F→D constraint
    """
    if state_str not in NOTATION_TO_STATE:
        raise ValueError(
            f"Invalid state string: {state_str}. "
            f"Must be 6 characters (V/v, F/f, D/d, P/p, X/x, A/a)."
        )
    state_int = NOTATION_TO_STATE[state_str]
    if state_int not in VALID_STATES:
        raise ValueError(f"Invalid state: {state_str} violates V→F→D constraint.")
    return state_int


def state_int_to_string(state_int: int) -> str:
    """
    Convert integer encoding to state string.

    Args:
        state_int: Integer encoding (0-63)

    Returns:
        State string like 'VFdpXa'

    Raises:
        ValueError: If state_int is invalid
    """
    if state_int not in STATE_NOTATION:
        raise ValueError(f"Invalid state integer: {state_int}. Must be 0-63.")
    if state_int not in VALID_STATES:
        raise ValueError(f"Invalid state integer: {state_int}. Violates V→F→D constraint.")
    return STATE_NOTATION[state_int]


def is_valid_state(state: Union[str, int]) -> bool:
    """
    Check if a state (string or int) is valid.

    Args:
        state: Either state string or integer

    Returns:
        True if valid, False otherwise
    """
    if isinstance(state, str):
        if state not in NOTATION_TO_STATE:
            return False
        return NOTATION_TO_STATE[state] in VALID_STATES
    elif isinstance(state, int):
        return state in VALID_STATES
    return False


def get_all_valid_states() -> list[str]:
    """Get sorted list of all 32 valid state strings."""
    return sorted(STATE_NOTATION[s] for s in VALID_STATES)


def get_all_valid_state_ints() -> list[int]:
    """Get sorted list of all 32 valid state integers."""
    return sorted(VALID_STATES)


# ==================== STATE EXPLANATION FUNCTIONS ====================


STATE_CHAR_DESCRIPTIONS: dict[str, str] = {
    "V": "Vendor Aware",
    "v": "Vendor Unaware",
    "F": "Fix Ready",
    "f": "Fix Not Ready",
    "D": "Fix Deployed",
    "d": "Fix Not Deployed",
    "P": "Public Aware",
    "p": "Public Unaware",
    "X": "Exploit Public",
    "x": "Exploit Unavailable",
    "A": "Attacks Observed",
    "a": "No Attacks",
}


def explain_state_char(char: str) -> str:
    """
    Convert single state character to human-readable description.

    Args:
        char: Single character ('V', 'v', 'F', 'f', etc.)

    Returns:
        Human-readable description like 'Vendor Aware' or 'Fix Not Ready'

    Raises:
        ValueError: If char is not a valid state character
    """
    if char not in STATE_CHAR_DESCRIPTIONS:
        valid = ", ".join(sorted(STATE_CHAR_DESCRIPTIONS.keys()))
        raise ValueError(f"Invalid state character: '{char}'. Valid: {valid}")
    return STATE_CHAR_DESCRIPTIONS[char]


def explain_state(state_string: str) -> str:
    """
    Convert state string to full human-readable description.

    Args:
        state_string: State like 'VFdpXa'

    Returns:
        Full description like 'Vendor Aware, Fix Ready, Fix Not Deployed, ...'

    Raises:
        ValueError: If state_string is not valid
    """
    if len(state_string) != 6:
        raise ValueError(f"State string must be exactly 6 characters, got {len(state_string)}")
    if not is_valid_state(state_string):
        raise ValueError(f"Invalid state string: {state_string}")
    return ", ".join(explain_state_char(c) for c in state_string)


def get_state_label(state_string: str) -> str:
    """
    Get concise label showing which events have occurred (cube name).

    Returns the uppercase letters representing occurred events.

    Args:
        state_string: State like 'VFdpXa'

    Returns:
        Label like 'VFX' (V, F, X occurred) or 'Initial' (no events)

    Raises:
        ValueError: If state_string is not valid
    """
    if len(state_string) != 6:
        raise ValueError(f"State string must be exactly 6 characters, got {len(state_string)}")
    if not is_valid_state(state_string):
        raise ValueError(f"Invalid state string: {state_string}")
    announced = "".join(c for c in state_string if c.isupper())
    return announced if announced else "Initial"


def get_events_from_state(state_int: int) -> list[CVDEvent]:
    """
    Get ordered list of events that have occurred in this state.

    Returns events in order: V, F, D, P, X, A (respects V→F→D constraint).

    Args:
        state_int: State as integer bitmask

    Returns:
        List of CVDEvent in order

    Example:
        >>> get_events_from_state(0b001011)  # VFdPxa
        [CVDEvent.V, CVDEvent.F, CVDEvent.P]
    """
    ordered = [CVDEvent.V, CVDEvent.F, CVDEvent.D, CVDEvent.P, CVDEvent.X, CVDEvent.A]
    return [event for event in ordered if state_int & (1 << event)]


# ==================== EXCEPTION CLASSES ====================


class TransformNotRunError(Exception):
    """Raised when accessing computed properties before transform() is called."""

    def __init__(self, property_name: str):
        self.property_name = property_name
        super().__init__(
            f"Cannot access '{property_name}' before transform() is called. "
            f"Run arr.transform() first."
        )


class ArrayFullError(Exception):
    """Raised when import exceeds fixed array capacity."""

    def __init__(self, capacity: int, attempted: int):
        self.capacity = capacity
        self.attempted = attempted
        super().__init__(
            f"Array has fixed capacity of {capacity}, "
            f"cannot import {attempted} items."
        )


# ==================== VECTORIZED CONVERSION FUNCTIONS ====================


def strings_to_state_ints(state_strs: list[str]) -> "NDArray[np.uint8]":
    """
    Vectorized conversion of state strings to integers.

    Args:
        state_strs: List of state strings like ['VFdpXa', 'vfdpxa']

    Returns:
        numpy array of shape (N,) with dtype=uint8
    """
    import numpy as np

    return np.array([string_to_state_int(s) for s in state_strs], dtype=np.uint8)


def state_ints_to_strings(state_ints: "NDArray[np.uint8]") -> list[str]:
    """
    Vectorized conversion of integer encodings to strings.

    Args:
        state_ints: numpy array of shape (N,) with dtype=uint8

    Returns:
        List of state strings
    """
    return [state_int_to_string(int(i)) for i in state_ints]
