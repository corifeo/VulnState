"""
Regression tests: desiderata masks must use DesiderataBit layout, not the 15-bit pair layout.

Previously where_desiderata_satisfied(D1_V_P) read pair bit 0 (V≺F), and the
formatter paired anti-desiderata labels with unrelated pair bits.
"""

from datetime import datetime, timedelta

import pytest

from vulnstate import CVDArray, CVDEvent, CVDFormatter, CVDVulnerability
from vulnstate.constants import AntiDesiderataBit, DesiderataBit

pytestmark = pytest.mark.regression

BASE = datetime(2024, 1, 1)


def make(events: str) -> CVDVulnerability:
    vuln = CVDVulnerability("CVE-2024-0001")
    for day, char in enumerate(events):
        vuln.apply_event(CVDEvent[char], timestamp=BASE + timedelta(days=day))
    return vuln


def analyzed(*histories: str) -> CVDArray:
    arr = CVDArray([make(h) for h in histories])
    arr.transform()
    return arr


class TestDesiderataLayout:
    def test_where_satisfied_reads_the_named_desideratum(self):
        # VF (no P): V≺F holds but V≺P is undecided -> D1 not satisfied
        arr = analyzed("VFP", "PVF", "VF")
        assert list(arr.where_desiderata_satisfied(DesiderataBit.D1_V_P)) == [True, False, False]

    def test_where_violated_flags_zero_day_exploit(self):
        # X before V, and X with no V at all, are both zero-day exploits
        arr = analyzed("XV", "X", "VX")
        violated = arr.where_desiderata_violated(AntiDesiderataBit.U2_X_V)
        assert list(violated) == [True, True, False]
        assert list(violated) == list(arr.is_zero_day_exploit)

    def test_premature_disclosure_matches_flag(self):
        arr = analyzed("VPF", "VFP", "VP")
        violated = arr.where_desiderata_violated(AntiDesiderataBit.U4_P_F)
        assert list(violated) == list(arr.is_premature_disclosure)

    def test_score_penalises_unmatched_later_event(self):
        # X with no V used to be credited as "V before X"
        arr = analyzed("VFDP", "X")
        assert arr.desiderata_score[0] == pytest.approx(1.0)
        assert arr.desiderata_score[1] < 1.0

    def test_labels_name_the_right_desiderata(self):
        arr = analyzed("XVP")
        assert "Zero-Day Exploit" in arr.get_violated_desiderata()[0]
        assert "Coordinated Disclosure" in arr.get_satisfied_desiderata()[0]


def table_rows(vuln: CVDVulnerability) -> dict[str, str]:
    from rich.console import Console

    console = Console(record=True, width=120, color_system=None)
    console.print(CVDFormatter.format_desiderata(vuln))
    return {
        line.split("│")[1].strip(): line.split("│")[2]
        for line in console.export_text().splitlines()
        if line.count("│") >= 3
    }


class TestDesiderataTable:
    def test_zero_day_rows_show_violations(self):
        rows = table_rows(make("AXVPF"))
        assert "Violated" in rows["Vendor Aware Before Attacks"]
        assert "Violated" in rows["Vendor Aware Before Exploit"]
        assert "Satisfied" in rows["Coordinated Disclosure"]
        assert "Violated" in rows["Deployed Before Disclosure"]  # P happened, D never did

    def test_unobserved_later_event_is_pending(self):
        rows = table_rows(make("VF"))
        assert "Pending" in rows["Coordinated Disclosure"]
        assert "Pending" in rows["Fix Ready Before Exploit"]
