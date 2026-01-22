"""
DataFrame Export Integration Tests - Cross-API Consistency

Tests that CVDVulnerability, CVDArray, and DataFrame export compute
consistent derived values. Verifies analytics across different access patterns.
"""

from datetime import datetime, timedelta

import numpy as np

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


class TestSingleVsArrayConsistency:
    """Test that single vuln and array compute same analytics."""

    def test_is_coordinated_matches(self):
        """Single vuln is_coordinated should match array.analysis."""
        for h in VALID_HISTORIES[:5] + VALID_HISTORIES[-5:]:  # Sample 10
            single = create_vuln(h)
            arr = CVDArray([single])
            assert single.is_coordinated == arr.analysis.is_coordinated[0]

    def test_is_zero_day_matches(self):
        """Single vuln is_zero_day should match array."""
        for h in VALID_HISTORIES[:5] + VALID_HISTORIES[-5:]:
            single = create_vuln(h)
            arr = CVDArray([single])
            assert single.is_zero_day == arr.is_zero_day[0]

    def test_has_fix_before_exploit_matches(self):
        """Single vuln has_fix_before_exploit should match array.analysis."""
        for h in VALID_HISTORIES[:5] + VALID_HISTORIES[-5:]:
            single = create_vuln(h)
            arr = CVDArray([single])
            assert single.has_fix_before_exploit == arr.analysis.has_fix_before_exploit[0]


class TestArrayVsDataFrameConsistency:
    """Test that array properties match DataFrame columns."""

    def test_is_fix_available_matches(self):
        """Array is_fix_available should match DataFrame column."""
        arr = create_all_histories_array()
        df = arr.to_dataframe()
        for i in range(len(arr)):
            assert bool(arr.is_fix_available[i]) == bool(df.loc[i, "is_fix_available"])

    def test_is_fix_deployed_matches(self):
        """Array is_fix_deployed should match DataFrame column."""
        arr = create_all_histories_array()
        df = arr.to_dataframe()
        for i in range(len(arr)):
            assert bool(arr.is_fix_deployed[i]) == bool(df.loc[i, "is_fix_deployed"])

    def test_is_weaponized_matches(self):
        """Array is_weaponized should match DataFrame column."""
        arr = create_all_histories_array()
        df = arr.to_dataframe()
        for i in range(len(arr)):
            assert bool(arr.is_weaponized[i]) == bool(df.loc[i, "is_weaponized"])

    def test_is_under_attack_matches(self):
        """Array is_under_attack should match DataFrame column."""
        arr = create_all_histories_array()
        df = arr.to_dataframe()
        for i in range(len(arr)):
            assert bool(arr.is_under_attack[i]) == bool(df.loc[i, "is_under_attack"])


class TestAnalysisResultConsistency:
    """Test AnalysisResult matches convenience properties."""

    def test_analyze_matches_properties(self):
        """vuln.analyze() result should match convenience properties."""
        for h in ["AXPVFD", "VFDPXA", "PVAXFD"]:  # Worst, best, middle
            vuln = create_vuln(h)
            result = vuln.analyze()
            assert vuln.is_zero_day == result.is_zero_day[0]
            assert vuln.is_coordinated == result.is_coordinated[0]
            assert vuln.has_fix_before_exploit == result.has_fix_before_exploit[0]
            assert vuln.has_fix_before_attack == result.has_fix_before_attack[0]


class TestAdvancedFiltering:
    """Test filtering on derived values returns correct subsets."""

    def test_filter_zero_days_correctness(self):
        """Filtering zero-days should return vulns where X or A before V."""
        arr = create_all_histories_array()
        zero_days = arr[arr.is_zero_day]

        for vuln in zero_days:
            h = vuln.history_string
            x_before_v = h.index("X") < h.index("V")
            a_before_v = h.index("A") < h.index("V")
            assert x_before_v or a_before_v, f"{h} should be zero-day"

    def test_filter_coordinated_correctness(self):
        """Filtering coordinated should return vulns where V before P."""
        arr = create_all_histories_array()
        coordinated = arr[arr.analysis.is_coordinated]

        for vuln in coordinated:
            h = vuln.history_string
            assert h.index("V") < h.index("P"), f"{h} should be coordinated"

    def test_zero_day_count_matches(self):
        """Zero-day filter count should match expected from histories."""
        arr = create_all_histories_array()
        expected = sum(
            1 for h in VALID_HISTORIES if h.index("X") < h.index("V") or h.index("A") < h.index("V")
        )
        actual = int(np.sum(arr.is_zero_day))
        assert actual == expected


class TestDesiderataViaAnalysis:
    """Test desiderata scores via AnalysisResult."""

    def test_best_history_score_is_1(self):
        """VFDPXA should have desiderata_score = 1.0 (12/12)."""
        arr = create_all_histories_array()
        best_idx = VALID_HISTORIES.index("VFDPXA")
        score = float(arr.analysis.desiderata_score[best_idx])
        assert abs(score - 1.0) < 0.01

    def test_best_history_count_is_12(self):
        """VFDPXA should have desiderata_count = 12."""
        arr = create_all_histories_array()
        best_idx = VALID_HISTORIES.index("VFDPXA")
        count = int(arr.analysis.desiderata_count[best_idx])
        assert count == 12

    def test_worst_history_has_low_score(self):
        """AXPVFD should have low desiderata score."""
        arr = create_all_histories_array()
        score = float(arr.analysis.desiderata_score[0])
        assert score < 0.5


class TestDataFrameGroupOperations:
    """Test DataFrame aggregations on derived columns."""

    def test_all_terminal_have_fix_available(self):
        """All 70 terminal histories should have is_fix_available=True."""
        df = create_all_histories_array().to_dataframe()
        assert df["is_fix_available"].astype(bool).sum() == 70

    def test_all_terminal_have_fix_deployed(self):
        """All 70 terminal histories should have is_fix_deployed=True."""
        df = create_all_histories_array().to_dataframe()
        assert df["is_fix_deployed"].astype(bool).sum() == 70

    def test_groupby_produces_correct_counts(self):
        """GroupBy on boolean column should produce expected counts."""
        arr = create_all_histories_array()
        df = arr.to_dataframe()

        # All terminal, so all should have exploits and attacks
        grouped = df.groupby("is_weaponized").size()
        assert grouped.get(True, 0) == 70
