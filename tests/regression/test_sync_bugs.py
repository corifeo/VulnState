"""
Regression Tests for Sync Bugs

Tests that guard against bugs where enrichment data (EPSS, KEV) was set
on vulnerability objects but not synced to array-level properties.

Fixed in commit: 8b1c2c8 (perf: optimize import_nvd with lazy parsing and fix enrichment sync)
"""

import pytest

from vulnstate import CVDArray, CVDVulnerability


class TestEpssSyncRegression:
    """Guard against EPSS not syncing from vuln objects to array properties."""

    def test_import_epss_syncs_to_array_level(self, tmp_path):
        """EPSS import must update both vuln.epss and arr.enrichment.epss."""
        # Create array with known CVE
        arr = CVDArray([CVDVulnerability("CVE-2023-0669")])

        # Create minimal EPSS CSV
        epss_file = tmp_path / "epss.csv"
        epss_file.write_text("cve,epss,percentile\nCVE-2023-0669,0.5,0.9\n")

        # Import EPSS data using array convenience method
        arr.import_epss(str(epss_file))

        # CRITICAL: Both must be set
        assert arr[0].enrichment.epss == pytest.approx(0.5)
        assert arr.enrichment.epss[0] == pytest.approx(0.5), "Array-level EPSS must sync"

    def test_import_kev_syncs_to_array_level(self, tmp_path):
        """KEV import must update both vuln.kev and arr.enrichment.kev."""
        arr = CVDArray([CVDVulnerability("CVE-2023-0669")])

        kev_file = tmp_path / "kev.csv"
        kev_file.write_text(
            "cveID,vendorProject,product,vulnerabilityName,dateAdded\n"
            "CVE-2023-0669,Fortra,GoAnywhere,Fortra GoAnywhere MFT RCE,2023-02-10\n"
        )

        arr.import_kev(str(kev_file))

        assert arr[0].enrichment.kev == True  # noqa: E712 - numpy bool
        assert arr.enrichment.kev[0] == True, "Array-level KEV must sync"  # noqa: E712


class TestDataFrameOverwriteRegression:
    """Guard against to_dataframe overwriting enrichment with stale metadata."""

    def test_to_dataframe_preserves_enrichment(self, tmp_path):
        """DataFrame export must not overwrite enrichment columns with metadata."""
        arr = CVDArray([CVDVulnerability("CVE-2023-0669")])

        epss_file = tmp_path / "epss.csv"
        epss_file.write_text("cve,epss,percentile\nCVE-2023-0669,0.5,0.9\n")

        arr.import_epss(str(epss_file))

        # Export to DataFrame with metadata explosion
        df = arr.to_dataframe(explode_metadata=True)

        # CRITICAL: Must preserve the enrichment value, not overwrite with NaN
        assert df.loc[0, "epss"] == pytest.approx(0.5), "to_dataframe must not overwrite enrichment"
