"""
Performance Tests

Tests for DataFrame export performance and analytics caching.
Converted from examples/06_analytics_caching.py.
"""

import time
from datetime import datetime

import pytest

from vulnstate import CVDArray, CVDEvent, CVDVulnerability


class TestDataFramePerformance:
    """Tests for DataFrame export performance."""

    @pytest.fixture
    def perf_large_array(self):
        """Create array with 10,000 vulnerabilities."""
        vulns = []
        for i in range(10000):
            cvss_score = 9.5 if i % 5 == 0 else 7.5
            vuln = CVDVulnerability(f"CVE-2024-{i:05d}", cvss_score=cvss_score)
            vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))
            vuln.apply_event(CVDEvent.F, timestamp=datetime(2024, 1, 15))
            if i % 10 == 0:
                vuln.apply_event(CVDEvent.X, timestamp=datetime(2023, 12, 15))
            vulns.append(vuln)

        return CVDArray(vulns)

    def test_dataframe_export_performance(self, perf_large_array):
        """DataFrame export for 10K vulns completes in reasonable time."""
        start = time.time()
        df = perf_large_array.to_dataframe()
        elapsed = time.time() - start

        assert len(df) == 10000
        assert elapsed < 5.0  # Should complete in under 5 seconds

    def test_dataframe_has_analytics_columns(self, perf_large_array):
        """DataFrame includes computed analytics columns."""
        df = perf_large_array.to_dataframe()

        # Check for analytics columns
        assert "is_zero_day" in df.columns
        assert "is_fix_available" in df.columns
        assert "is_fix_deployed" in df.columns

    def test_zero_day_detection(self, perf_large_array):
        """Zero-day detection works correctly."""
        df = perf_large_array.to_dataframe()

        zero_day_count = df["is_zero_day"].sum()
        # 10% have X event before V (every 10th item)
        assert zero_day_count == 1000

    def test_severity_classification(self, perf_large_array):
        """Severity classification from CVSS works correctly."""
        df = perf_large_array.to_dataframe()

        # 20% have CVSS 9.5 (critical), 80% have 7.5 (high)
        critical_count = (df["severity"] == "critical").sum()
        high_count = (df["severity"] == "high").sum()

        assert critical_count == 2000
        assert high_count == 8000


class TestCachingBehavior:
    """Tests for analytics caching behavior."""

    @pytest.fixture
    def medium_array(self):
        """Create array with 1,000 vulnerabilities."""
        vulns = []
        for i in range(1000):
            vuln = CVDVulnerability(f"CVE-2024-{i:04d}", cvss_score=7.5)
            vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))
            vulns.append(vuln)

        return CVDArray(vulns)

    def test_repeated_export_consistent(self, medium_array):
        """Repeated DataFrame exports produce consistent results."""
        df1 = medium_array.to_dataframe()
        df2 = medium_array.to_dataframe()

        assert len(df1) == len(df2)
        assert list(df1.columns) == list(df2.columns)

        # Values should match
        assert (df1["state"] == df2["state"]).all()
        assert (df1["cvss_score"] == df2["cvss_score"]).all()

    def test_export_after_modification(self, medium_array):
        """DataFrame export reflects array modifications."""
        df1 = medium_array.to_dataframe()
        initial_f_count = df1["F"].sum()

        # Apply F event to all
        medium_array.apply_event_batch(CVDEvent.F, timestamp=datetime(2024, 2, 1))

        df2 = medium_array.to_dataframe()
        updated_f_count = df2["F"].sum()

        assert updated_f_count > initial_f_count
        assert updated_f_count == 1000
