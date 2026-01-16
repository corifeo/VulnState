"""
Tests for CVD Constants

Tests for constants.py module containing CVD paper concepts:
- CVDEvent IntEnum (6 events)
- STATE_NOTATION lookup table
- AntiDesiderataBit enum (12 undesirable orderings)
- FixPath IntEnum (fix path progression)
- ThreatState IntEnum (threat materialization)
- HistoryValidity enum
- EventPairRelation enum
- DESIDERATA_PAIRS constant (12 pairs)
- ORDERED_PAIRS_TABLE (Table 3.3 from SEI/CMU 2021 paper)
"""


def test_event_pair_relation_enum():
    """Test EventPairRelation from Table 3.3 of paper."""
    from vulnstate.constants import EventPairRelation

    assert EventPairRelation.IMPOSSIBLE.value == "impossible"
    assert EventPairRelation.REQUIRED.value == "required"
    assert EventPairRelation.DESIRED.value == "desired"
    assert EventPairRelation.UNDESIRED.value == "undesired"


def test_desiderata_pairs_count():
    """Test DESIDERATA_PAIRS has all 12 desiderata."""
    from vulnstate.constants import DESIDERATA_PAIRS

    assert len(DESIDERATA_PAIRS) == 12


def test_desiderata_pairs_content():
    """Test DESIDERATA_PAIRS includes X≺A (12th desideratum)."""
    from vulnstate import CVDEvent
    from vulnstate.constants import DESIDERATA_PAIRS

    # Check X≺A is present (missing from current implementation)
    assert (CVDEvent.X, CVDEvent.A) in DESIDERATA_PAIRS

    # Check other key desiderata
    assert (CVDEvent.V, CVDEvent.P) in DESIDERATA_PAIRS  # V before P
    assert (CVDEvent.F, CVDEvent.X) in DESIDERATA_PAIRS  # F before X
    assert (CVDEvent.D, CVDEvent.A) in DESIDERATA_PAIRS  # D before A


def test_ordered_pairs_table_structure():
    """Test ORDERED_PAIRS_TABLE has all event pairs."""
    from vulnstate import CVDEvent
    from vulnstate.constants import ORDERED_PAIRS_TABLE, EventPairRelation

    # Should have 6x6 = 36 entries
    assert len(ORDERED_PAIRS_TABLE) == 36

    # Check V→F is REQUIRED (V must precede F)
    assert ORDERED_PAIRS_TABLE[(CVDEvent.V, CVDEvent.F)] == EventPairRelation.REQUIRED

    # Check F→V is IMPOSSIBLE (violates constraint)
    assert ORDERED_PAIRS_TABLE[(CVDEvent.F, CVDEvent.V)] == EventPairRelation.IMPOSSIBLE

    # Check V→P is DESIRED (desideratum)
    assert ORDERED_PAIRS_TABLE[(CVDEvent.V, CVDEvent.P)] == EventPairRelation.DESIRED

    # Check P→V is UNDESIRED (complement of desired)
    assert ORDERED_PAIRS_TABLE[(CVDEvent.P, CVDEvent.V)] == EventPairRelation.UNDESIRED


def test_ordered_pairs_table_12_desiderata():
    """Test all 12 desiderata are marked DESIRED."""
    from vulnstate.constants import (
        DESIDERATA_PAIRS,
        ORDERED_PAIRS_TABLE,
        EventPairRelation,
    )

    for pair in DESIDERATA_PAIRS:
        assert ORDERED_PAIRS_TABLE[pair] == EventPairRelation.DESIRED


def test_domain_exports():
    """Test all domain types are exported in public API."""
    from vulnstate import DESIDERATA_PAIRS, FixPath, ThreatState
    from vulnstate.constants import ORDERED_PAIRS_TABLE, EventPairRelation

    # Verify canonical enums
    assert FixPath.FIX_READY.value == 0b011
    assert ThreatState.WEAPONIZED.value == 0b011

    assert EventPairRelation is not None
    assert len(DESIDERATA_PAIRS) == 12
    assert len(ORDERED_PAIRS_TABLE) == 36


# ==================== STATE DIMENSIONS TESTS ====================


def test_fix_path_enum_values():
    """Test FixPath IntEnum has correct bitmask values."""
    from vulnstate.constants import FixPath

    assert FixPath.NO_AWARENESS.value == 0b000
    assert FixPath.VENDOR_AWARE.value == 0b001
    assert FixPath.FIX_READY.value == 0b011
    assert FixPath.REMEDIATED.value == 0b111


def test_fix_path_enum_labels():
    """Test FixPath.label property returns human-readable labels."""
    from vulnstate.constants import FixPath

    assert FixPath.NO_AWARENESS.label == "No Awareness"
    assert FixPath.VENDOR_AWARE.label == "Vendor Aware"
    assert FixPath.FIX_READY.label == "Fix Ready"
    assert FixPath.REMEDIATED.label == "Remediated"


def test_fix_path_enum_state_strings():
    """Test FixPath.state_string property returns CVD notation."""
    from vulnstate.constants import FixPath

    assert FixPath.NO_AWARENESS.state_string == "vfd"
    assert FixPath.VENDOR_AWARE.state_string == "Vfd"
    assert FixPath.FIX_READY.state_string == "VFd"
    assert FixPath.REMEDIATED.state_string == "VFD"


def test_threat_state_enum_values():
    """Test ThreatState IntEnum has correct bitmask values."""
    from vulnstate.constants import ThreatState

    assert ThreatState.LATENT.value == 0b000
    assert ThreatState.DISCLOSED.value == 0b001
    assert ThreatState.WEAPONIZED.value == 0b011
    assert ThreatState.TARGETED.value == 0b100
    assert ThreatState.UNDER_ATTACK.value == 0b101
    assert ThreatState.ACTIVE_THREAT.value == 0b111


def test_threat_state_enum_labels():
    """Test ThreatState.label property returns human-readable labels."""
    from vulnstate.constants import ThreatState

    assert ThreatState.LATENT.label == "Latent"
    assert ThreatState.DISCLOSED.label == "Disclosed"
    assert ThreatState.WEAPONIZED.label == "Weaponized"
    assert ThreatState.TARGETED.label == "Targeted"
    assert ThreatState.UNDER_ATTACK.label == "Under Attack"
    assert ThreatState.ACTIVE_THREAT.label == "Active Threat"


def test_threat_state_enum_state_strings():
    """Test ThreatState.state_string property returns CVD notation."""
    from vulnstate.constants import ThreatState

    assert ThreatState.LATENT.state_string == "pxa"
    assert ThreatState.DISCLOSED.state_string == "Pxa"
    assert ThreatState.WEAPONIZED.state_string == "PXa"
    assert ThreatState.TARGETED.state_string == "pxA"
    assert ThreatState.UNDER_ATTACK.state_string == "PxA"
    assert ThreatState.ACTIVE_THREAT.state_string == "PXA"


def test_fix_path_labels_dict():
    """Test FIX_PATH_LABELS dict for vectorized lookup."""
    from vulnstate.constants import FIX_PATH_LABELS

    assert FIX_PATH_LABELS[0b000] == "No Awareness"
    assert FIX_PATH_LABELS[0b001] == "Vendor Aware"
    assert FIX_PATH_LABELS[0b011] == "Fix Ready"
    assert FIX_PATH_LABELS[0b111] == "Remediated"


def test_threat_labels_dict():
    """Test THREAT_LABELS dict for vectorized lookup."""
    from vulnstate.constants import THREAT_LABELS

    assert THREAT_LABELS[0b000] == "Latent"
    assert THREAT_LABELS[0b001] == "Disclosed"
    assert THREAT_LABELS[0b011] == "Weaponized"
    assert THREAT_LABELS[0b100] == "Targeted"
    assert THREAT_LABELS[0b101] == "Under Attack"
    assert THREAT_LABELS[0b111] == "Active Threat"


def test_validity_enum():
    """Test HistoryValidity enum for constraint checking."""
    from vulnstate.constants import HistoryValidity

    assert HistoryValidity.VALID.value == 0
    assert HistoryValidity.IMPOSSIBLE.value == 1


def test_state_dimension_extraction():
    """Test extracting both dimensions from combined state."""
    from vulnstate.constants import FixPath, ThreatState

    # Example: VFdPXa (Fix ready, Weaponized) = 0b011_011 = 27
    combined_state = 0b011_011

    vfd_bits = combined_state & 0b000111
    pxa_bits = (combined_state >> 3) & 0b000111

    fix_path = FixPath(vfd_bits)
    threat_state = ThreatState(pxa_bits)

    assert fix_path == FixPath.FIX_READY
    assert fix_path.label == "Fix Ready"
    assert threat_state == ThreatState.WEAPONIZED
    assert threat_state.label == "Weaponized"


def test_pair_bit_positions():
    from vulnstate.constants import PAIR_BIT_POSITIONS, PAIR_NAMES, CVDEvent

    # V≺F should be bit 0
    assert PAIR_BIT_POSITIONS[(CVDEvent.V, CVDEvent.F)] == 0
    # X≺A should be bit 14 (last pair)
    assert PAIR_BIT_POSITIONS[(CVDEvent.X, CVDEvent.A)] == 14
    # 15 pairs total
    assert len(PAIR_BIT_POSITIONS) == 15

    # Reverse lookup
    assert PAIR_NAMES[0] == "V≺F"
    assert PAIR_NAMES[14] == "X≺A"


def test_required_pairs_mask():
    from vulnstate.constants import REQUIRED_PAIRS_MASK

    # Required: V≺F (bit 0), V≺D (bit 1), F≺D (bit 5)
    # Binary: 0b0000_0000_0010_0011 = 35
    assert REQUIRED_PAIRS_MASK == 0b0000_0000_0010_0011


# ==================== PHASE 1: RESTRUCTURE TESTS ====================


def test_cvd_event_is_intenum():
    """CVDEvent should be IntEnum with values 0-5."""
    from vulnstate.constants import CVDEvent

    assert CVDEvent.V == 0
    assert CVDEvent.F == 1
    assert CVDEvent.D == 2
    assert CVDEvent.P == 3
    assert CVDEvent.X == 4
    assert CVDEvent.A == 5
    assert isinstance(CVDEvent.V, int)


def test_cvd_event_count():
    """CVDEvent should define exactly 6 events."""
    from vulnstate import CVDEvent

    assert len(list(CVDEvent)) == 6


def test_cvd_event_bit_positions():
    """CVDEvent IntEnum values should equal bit positions."""
    from vulnstate import CVDEvent

    assert CVDEvent.V == 0  # Bit 0
    assert CVDEvent.F == 1  # Bit 1
    assert CVDEvent.D == 2  # Bit 2
    assert CVDEvent.P == 3  # Bit 3
    assert CVDEvent.X == 4  # Bit 4
    assert CVDEvent.A == 5  # Bit 5


def test_cvd_event_labels():
    """CVDEvent should have labels and descriptions."""
    from vulnstate.constants import (
        EVENT_DESCRIPTIONS,
        EVENT_LABELS,
        CVDEvent,
    )

    assert EVENT_LABELS[CVDEvent.V] == "Vendor Awareness"
    assert "vendor" in EVENT_DESCRIPTIONS[CVDEvent.V].lower()


def test_state_notation_lookup():
    """STATE_NOTATION maps int -> string like 'VFdpxa'."""
    from vulnstate.constants import STATE_NOTATION

    assert STATE_NOTATION[0] == "vfdpxa"  # All lowercase
    assert STATE_NOTATION[0b111111] == "VFDPXA"  # All uppercase
    assert STATE_NOTATION[0b000011] == "VFdpxa"  # V and F set
    assert len(STATE_NOTATION) == 64


def test_anti_desiderata_enum():
    """AntiDesiderataBit should define 12 undesirable orderings."""
    from vulnstate.constants import ANTI_DESIDERATA_LABELS, AntiDesiderataBit

    assert AntiDesiderataBit.U1_P_V == 0
    assert AntiDesiderataBit.U2_X_V == 1  # Zero-day exploit
    assert AntiDesiderataBit.U3_A_V == 2  # Zero-day attack
    assert len(AntiDesiderataBit) == 12
    assert ANTI_DESIDERATA_LABELS[AntiDesiderataBit.U2_X_V] == "Zero-Day Exploit"


def test_probability_defaults():
    """Probability constants should be defined."""
    from vulnstate.constants import (
        DEFAULT_THREAT_MULTIPLIER,
        DEFAULT_XA_SPLIT,
    )

    assert DEFAULT_XA_SPLIT == 0.5
    assert DEFAULT_THREAT_MULTIPLIER == 1.0
