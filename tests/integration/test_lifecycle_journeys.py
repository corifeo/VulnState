"""
Lifecycle Integration Tests - Boolean Analytics Derivation

Tests that computed boolean analytics are correctly derived from event ordering.
Uses all 70 valid histories but tests aggregate correctness, not individual parametrization.
"""

from datetime import datetime, timedelta

from vulnstate import VALID_HISTORIES, CVDArray, CVDEvent, CVDVulnerability


def create_vuln(history: str) -> CVDVulnerability:
    """Create vulnerability with events in history order."""
    vuln = CVDVulnerability(cve_id=f"CVE-TEST-{history}")
    base = datetime(2024, 1, 1)
    for i, char in enumerate(history):
        vuln.apply_event(CVDEvent[char], timestamp=base + timedelta(days=i * 7))
    return vuln


def create_all_histories_array() -> CVDArray:
    """Create CVDArray with all 70 valid histories."""
    return CVDArray([create_vuln(h) for h in VALID_HISTORIES])


class TestBooleanDerivationAggregates:
    """Test that boolean analytics match expected counts across all 70 histories."""

    def test_is_coordinated_count(self):
        """is_coordinated count should match V<P occurrences in histories."""
        arr = create_all_histories_array()
        expected = sum(1 for h in VALID_HISTORIES if h.index("V") < h.index("P"))
        actual = sum(1 for v in arr if v.is_coordinated)
        assert actual == expected, f"Expected {expected} coordinated, got {actual}"

    def test_is_zero_day_exploit_count(self):
        """is_zero_day_exploit count should match X<V occurrences."""
        arr = create_all_histories_array()
        expected = sum(1 for h in VALID_HISTORIES if h.index("X") < h.index("V"))
        actual = sum(1 for v in arr if v.is_zero_day_exploit)
        assert actual == expected

    def test_is_zero_day_attack_count(self):
        """is_zero_day_attack count should match A<V occurrences."""
        arr = create_all_histories_array()
        expected = sum(1 for h in VALID_HISTORIES if h.index("A") < h.index("V"))
        actual = sum(1 for v in arr if v.is_zero_day_attack)
        assert actual == expected

    def test_has_fix_before_exploit_count(self):
        """has_fix_before_exploit count should match F<X occurrences."""
        arr = create_all_histories_array()
        expected = sum(1 for h in VALID_HISTORIES if h.index("F") < h.index("X"))
        actual = sum(1 for v in arr if v.has_fix_before_exploit)
        assert actual == expected

    def test_has_fix_before_attack_count(self):
        """has_fix_before_attack count should match F<A occurrences."""
        arr = create_all_histories_array()
        expected = sum(1 for h in VALID_HISTORIES if h.index("F") < h.index("A"))
        actual = sum(1 for v in arr if v.has_fix_before_attack)
        assert actual == expected

    def test_has_deployment_before_exploit_count(self):
        """has_deployment_before_exploit count should match D<X occurrences."""
        arr = create_all_histories_array()
        expected = sum(1 for h in VALID_HISTORIES if h.index("D") < h.index("X"))
        actual = sum(1 for v in arr if v.has_deployment_before_exploit)
        assert actual == expected

    def test_has_deployment_before_attack_count(self):
        """has_deployment_before_attack count should match D<A occurrences."""
        arr = create_all_histories_array()
        expected = sum(1 for h in VALID_HISTORIES if h.index("D") < h.index("A"))
        actual = sum(1 for v in arr if v.has_deployment_before_attack)
        assert actual == expected

    def test_is_premature_disclosure_count(self):
        """is_premature_disclosure count should match P<F occurrences."""
        arr = create_all_histories_array()
        expected = sum(1 for h in VALID_HISTORIES if h.index("P") < h.index("F"))
        actual = sum(1 for v in arr if v.is_premature_disclosure)
        assert actual == expected


class TestBooleanDerivationSpotChecks:
    """Spot-check specific histories for correct boolean derivation."""

    def test_worst_case_axpvfd(self):
        """AXPVFD (worst): A and X before V should trigger zero-day flags."""
        vuln = create_vuln("AXPVFD")
        assert vuln.is_zero_day_attack, "A before V"
        assert vuln.is_zero_day_exploit, "X before V"
        assert not vuln.is_coordinated, "P before V"
        assert not vuln.has_fix_before_exploit, "X before F"
        assert not vuln.has_fix_before_attack, "A before F"

    def test_best_case_vfdpxa(self):
        """VFDPXA (best): All good orderings should be satisfied."""
        vuln = create_vuln("VFDPXA")
        assert vuln.is_coordinated, "V before P"
        assert not vuln.is_zero_day_exploit, "V before X"
        assert not vuln.is_zero_day_attack, "V before A"
        assert vuln.has_fix_before_exploit, "F before X"
        assert vuln.has_fix_before_attack, "F before A"
        assert vuln.has_deployment_before_exploit, "D before X"
        assert vuln.has_deployment_before_attack, "D before A"
        assert not vuln.is_premature_disclosure, "F before P"

    def test_premature_disclosure_pvaxfd(self):
        """PVAXFD: P before F should trigger premature disclosure."""
        vuln = create_vuln("PVAXFD")
        assert vuln.is_premature_disclosure, "P before F"
        assert not vuln.is_coordinated, "P before V (not coordinated)"


class TestDesiderataDerivation:
    """Test desiderata scoring is correctly derived."""

    def test_best_history_satisfies_all_12_desiderata(self):
        """VFDPXA should satisfy all 12 desiderata."""
        vuln = create_vuln("VFDPXA")
        assert len(vuln.satisfied_desiderata) == 12
        assert len(vuln.violated_desiderata) == 0

    def test_worst_history_has_few_desiderata(self):
        """AXPVFD should satisfy few desiderata."""
        vuln = create_vuln("AXPVFD")
        assert len(vuln.satisfied_desiderata) < 6

    def test_desiderata_ranking_correlation(self):
        """Higher history_id should generally mean more desiderata."""
        worst = create_vuln(VALID_HISTORIES[0])
        best = create_vuln(VALID_HISTORIES[69])
        assert len(best.satisfied_desiderata) > len(worst.satisfied_desiderata)

    def test_all_histories_desiderata_sum_to_12(self):
        """Each history should have satisfied + violated = 12."""
        for h in VALID_HISTORIES:
            vuln = create_vuln(h)
            total = len(vuln.satisfied_desiderata) + len(vuln.violated_desiderata)
            assert total == 12, f"{h}: {total} != 12"


class TestHistoryStringDerivation:
    """Test history_string property matches event application order."""

    def test_history_string_matches_all(self):
        """history_string should match the order events were applied."""
        for h in VALID_HISTORIES:
            vuln = create_vuln(h)
            assert vuln.history_string == h, f"Expected {h}, got {vuln.history_string}"


class TestTerminalStateDerivation:
    """Test all histories reach terminal state."""

    def test_all_histories_terminal(self):
        """All 70 histories should reach terminal state (all events occurred)."""
        for h in VALID_HISTORIES:
            vuln = create_vuln(h)
            assert vuln.is_terminal(), f"{h} should be terminal"
