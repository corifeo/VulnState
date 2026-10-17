"""
Regression Tests for Sync Bugs

Tests that guard against bugs where enrichment data (EPSS, KEV) was set
on vulnerability objects but not synced to array-level properties.

Fixed in commit: 8b1c2c8 (perf: optimize import_nvd with lazy parsing and fix enrichment sync)
"""

import pytest

from vulnstate import CVDArray, CVDVulnerability

pytestmark = pytest.mark.regression


class TestEpssSyncRegression:
    """Guard against EPSS not syncing from vuln objects to array properties."""

    def test_import_epss_syncs_to_array_level(self, tmp_path):
        """EPSS import must update both vuln.epss and arr.epss."""
        # Create array with known CVE
        arr = CVDArray([CVDVulnerability("CVE-2023-0669")])

        # Create minimal EPSS CSV
        epss_file = tmp_path / "epss.csv"
        epss_file.write_text("cve,epss,percentile\nCVE-2023-0669,0.5,0.9\n")

        # Import EPSS data using array convenience method
        arr.import_epss(str(epss_file))
        arr.transform()

        # CRITICAL: Both must be set
        assert arr[0].epss == pytest.approx(0.5)
        assert arr.epss[0] == pytest.approx(0.5), "Array-level EPSS must sync"

    def test_import_kev_syncs_to_array_level(self, tmp_path):
        """KEV import must update both vuln.kev and arr.kev."""
        arr = CVDArray([CVDVulnerability("CVE-2023-0669")])

        kev_file = tmp_path / "kev.csv"
        kev_file.write_text(
            "cveID,vendorProject,product,vulnerabilityName,dateAdded\n"
            "CVE-2023-0669,Fortra,GoAnywhere,Fortra GoAnywhere MFT RCE,2023-02-10\n"
        )

        arr.import_kev(str(kev_file))
        arr.transform()

        assert arr[0].kev == True  # noqa: E712 - numpy bool
        assert arr.kev[0] == True, "Array-level KEV must sync"  # noqa: E712


class TestDataFrameOverwriteRegression:
    """Guard against to_dataframe overwriting enrichment with stale metadata."""

    def test_to_dataframe_preserves_enrichment(self, tmp_path):
        """DataFrame export must not overwrite enrichment columns with metadata."""
        arr = CVDArray([CVDVulnerability("CVE-2023-0669")])

        epss_file = tmp_path / "epss.csv"
        epss_file.write_text("cve,epss,percentile\nCVE-2023-0669,0.5,0.9\n")

        arr.import_epss(str(epss_file))
        arr.transform()

        # Export to DataFrame with metadata explosion
        df = arr.to_dataframe(explode_metadata=True)

        # CRITICAL: Must preserve the enrichment value, not overwrite with NaN
        assert df.loc[0, "epss"] == pytest.approx(0.5), "to_dataframe must not overwrite enrichment"


class TestMixedTimestampTypesRegression:
    """Guard against TypeError when events come from sources with different datetime types."""

    def test_kev_import_after_tz_aware_events(self, tmp_path):
        """NVD gives tz-aware datetimes, KEV gives day-precision dates; both must compare."""
        from datetime import datetime, timezone

        from vulnstate import CVDEvent

        vuln = CVDVulnerability("CVE-2023-0669")
        vuln.apply_event(CVDEvent.V, timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc))
        vuln.apply_event(CVDEvent.P, timestamp=datetime(2023, 1, 5, tzinfo=timezone.utc))
        arr = CVDArray([vuln])

        kev_file = tmp_path / "kev.csv"
        kev_file.write_text("cveID,dateAdded\nCVE-2023-0669,2023-02-10\n")
        arr.import_kev(str(kev_file))
        arr.transform()

        assert arr.is_under_attack[0]
        assert not arr.is_zero_day_attack[0]


class TestEpssPercentileRegression:
    """Guard against import_epss dropping the CSV percentile column."""

    def test_import_epss_csv_keeps_percentile(self, tmp_path):
        arr = CVDArray([CVDVulnerability("CVE-2023-0669")])

        epss_file = tmp_path / "epss.csv"
        epss_file.write_text("cve,epss,percentile\nCVE-2023-0669,0.5,0.9\n")
        arr.import_epss(str(epss_file))
        arr.transform()

        assert arr.epss_percentile[0] == pytest.approx(0.9)


class TestKevExportRegression:
    """Guard against JSON export failing on KEV entries imported from CSV."""

    def test_to_json_batch_after_kev_csv_import(self, tmp_path):
        arr = CVDArray([CVDVulnerability("CVE-2023-0669")])

        kev_file = tmp_path / "kev.csv"
        kev_file.write_text("cveID,dateAdded,dueDate\nCVE-2023-0669,2023-02-10,2023-03-03\n")
        arr.import_kev(str(kev_file))
        arr.transform()

        out = tmp_path / "out.json"
        arr.to_json_batch(str(out), include_computed=True)
        assert "2023-03-03" in out.read_text()
