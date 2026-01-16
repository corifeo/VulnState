"""
Tests for file-based data import functionality.

Tests CVDIO file-based methods (import_epss_file, import_kev_file,
import_nvd_file) and CVDArray convenience wrappers.
"""

import os

import pytest

from vulnstate import CVDArray, CVDVulnerability
from vulnstate.io import CVDIO

# Test data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestEPSSFileImport:
    """Tests for EPSS CSV file import."""

    def test_import_epss_file_basic(self):
        """Test basic EPSS file import."""
        # Create array with CVEs matching the test data
        vulns = [
            CVDVulnerability("CVE-2024-001"),
            CVDVulnerability("CVE-2024-002"),
            CVDVulnerability("CVE-2024-003"),
        ]
        arr = CVDArray(vulns)

        # Import EPSS scores from file
        epss_file = os.path.join(DATA_DIR, "epss_sample.csv")
        CVDIO.import_epss_file(arr, epss_file)

        # Verify scores were imported
        assert arr[0].epss == pytest.approx(0.85432)
        assert arr[1].epss == pytest.approx(0.42156)
        assert arr[2].epss == pytest.approx(0.12345)

    def test_import_epss_file_missing_cves(self):
        """Test EPSS import with CVEs not in file."""
        vulns = [
            CVDVulnerability("CVE-2024-001"),
            CVDVulnerability("CVE-2024-999"),  # Not in file
        ]
        arr = CVDArray(vulns)

        epss_file = os.path.join(DATA_DIR, "epss_sample.csv")
        CVDIO.import_epss_file(arr, epss_file)

        # First CVE should have score
        assert arr[0].epss == pytest.approx(0.85432)
        # Second CVE should not be updated (epss remains None)
        assert arr[1].epss is None

    def test_import_epss_file_convenience_method(self):
        """Test CVDArray convenience wrapper for EPSS import."""
        vulns = [
            CVDVulnerability("CVE-2024-001"),
            CVDVulnerability("CVE-2024-002"),
        ]
        arr = CVDArray(vulns)

        # Use convenience method
        epss_file = os.path.join(DATA_DIR, "epss_sample.csv")
        arr.import_epss_file(epss_file)

        assert arr[0].epss == pytest.approx(0.85432)
        assert arr[1].epss == pytest.approx(0.42156)


class TestKEVFileImport:
    """Tests for KEV CSV file import."""

    def test_import_kev_file_basic(self):
        """Test basic KEV file import."""
        vulns = [
            CVDVulnerability("CVE-2021-44228"),  # In KEV
            CVDVulnerability("CVE-2024-001"),  # In KEV
            CVDVulnerability("CVE-2024-002"),  # Not in KEV
        ]
        arr = CVDArray(vulns)

        # Import KEV catalog from file
        kev_file = os.path.join(DATA_DIR, "kev_sample.csv")
        CVDIO.import_kev_file(arr, kev_file)

        # Verify KEV flags were set
        assert arr[0].is_kev is True  # CVE-2021-44228
        assert arr[1].is_kev is True  # CVE-2024-001
        assert arr[2].is_kev is False  # CVE-2024-002 not in KEV

    def test_import_kev_file_convenience_method(self):
        """Test CVDArray convenience wrapper for KEV import."""
        vulns = [
            CVDVulnerability("CVE-2021-44228"),
            CVDVulnerability("CVE-2024-001"),
        ]
        arr = CVDArray(vulns)

        # Use convenience method
        kev_file = os.path.join(DATA_DIR, "kev_sample.csv")
        arr.import_kev_file(kev_file)

        assert arr[0].is_kev is True
        assert arr[1].is_kev is True


class TestNVDFileImport:
    """Tests for NVD JSON file import."""

    def test_import_nvd_file_basic(self):
        """Test basic NVD file import."""
        vulns = [
            CVDVulnerability("CVE-2024-001"),  # CVSS v3: 6.1
            CVDVulnerability("CVE-2024-002"),  # CVSS v3: 9.8
            CVDVulnerability("CVE-2024-005"),  # CVSS v2: 7.5
        ]
        arr = CVDArray(vulns)

        # Import NVD data from file
        nvd_file = os.path.join(DATA_DIR, "nvd_sample.json")
        CVDIO.import_nvd_file(arr, nvd_file)

        # Verify CVSS scores were imported
        assert arr[0].cvss_score == 6.1  # CVE-2024-001
        assert arr[1].cvss_score == 9.8  # CVE-2024-002
        assert arr[2].cvss_score == 7.5  # CVE-2024-005 (v2)

    def test_import_nvd_file_cvss_vectors(self):
        """Test NVD import includes CVSS vector strings."""
        vulns = [
            CVDVulnerability("CVE-2024-001"),
            CVDVulnerability("CVE-2024-002"),
        ]
        arr = CVDArray(vulns)

        nvd_file = os.path.join(DATA_DIR, "nvd_sample.json")
        CVDIO.import_nvd_file(arr, nvd_file)

        # Verify vector strings were imported
        assert "CVSS:3.1" in arr[0].cve_vector
        assert "CVSS:3.1" in arr[1].cve_vector

    def test_import_nvd_file_fallback_to_v2(self):
        """Test NVD import falls back to CVSS v2 when v3 unavailable."""
        vulns = [CVDVulnerability("CVE-2024-005")]  # Only has v2
        arr = CVDArray(vulns)

        nvd_file = os.path.join(DATA_DIR, "nvd_sample.json")
        CVDIO.import_nvd_file(arr, nvd_file)

        # Verify v2 score was used
        assert arr[0].cvss_score == 7.5
        assert arr[0].cve_vector == "AV:N/AC:L/Au:N/C:P/I:P/A:P"

    def test_import_nvd_file_convenience_method(self):
        """Test CVDArray convenience wrapper for NVD import."""
        vulns = [
            CVDVulnerability("CVE-2024-001"),
            CVDVulnerability("CVE-2024-002"),
        ]
        arr = CVDArray(vulns)

        # Use convenience method
        nvd_file = os.path.join(DATA_DIR, "nvd_sample.json")
        arr.import_nvd_file(nvd_file)

        assert arr[0].cvss_score == 6.1
        assert arr[1].cvss_score == 9.8


class TestIntegratedWorkflow:
    """Test integrated workflow with multiple file imports."""

    def test_import_all_sources(self):
        """Test importing from all three sources (EPSS, KEV, NVD)."""
        # Create array with overlapping CVEs
        vulns = [
            CVDVulnerability("CVE-2024-001"),  # In all three sources
            CVDVulnerability("CVE-2024-002"),  # In EPSS and NVD
            CVDVulnerability("CVE-2024-003"),  # In EPSS and KEV
        ]
        arr = CVDArray(vulns)

        # Import from all sources
        epss_file = os.path.join(DATA_DIR, "epss_sample.csv")
        kev_file = os.path.join(DATA_DIR, "kev_sample.csv")
        nvd_file = os.path.join(DATA_DIR, "nvd_sample.json")

        arr.import_epss_file(epss_file)
        arr.import_kev_file(kev_file)
        arr.import_nvd_file(nvd_file)

        # Verify CVE-2024-001 has all three enrichments
        assert arr[0].epss == pytest.approx(0.85432)
        assert arr[0].is_kev is True
        assert arr[0].cvss_score == 6.1

        # Verify CVE-2024-002 has EPSS and NVD but not KEV
        assert arr[1].epss == pytest.approx(0.42156)
        assert arr[1].is_kev is False
        assert arr[1].cvss_score == 9.8

        # Verify CVE-2024-003 has EPSS and KEV but not NVD
        assert arr[2].epss == pytest.approx(0.12345)
        assert arr[2].is_kev is True
        assert arr[2].cvss_score is None  # Not in NVD sample


class TestErrorHandling:
    """Test error handling for invalid files."""

    def test_import_nonexistent_file(self):
        """Test import from non-existent file raises error."""
        vulns = [CVDVulnerability("CVE-2024-001")]
        arr = CVDArray(vulns)

        with pytest.raises(FileNotFoundError):
            arr.import_epss_file("nonexistent.csv")

    def test_import_malformed_csv(self, tmp_path):
        """Test import from malformed CSV raises error."""
        # Create malformed CSV (missing required columns)
        malformed = tmp_path / "malformed.csv"
        malformed.write_text("wrong,columns\n1,2\n")

        vulns = [CVDVulnerability("CVE-2024-001")]
        arr = CVDArray(vulns)

        with pytest.raises(KeyError):
            arr.import_epss_file(str(malformed))

    def test_import_malformed_json(self, tmp_path):
        """Test import from malformed JSON raises error."""
        # Create malformed JSON
        malformed = tmp_path / "malformed.json"
        malformed.write_text("not valid json")

        vulns = [CVDVulnerability("CVE-2024-001")]
        arr = CVDArray(vulns)

        with pytest.raises(Exception):  # json.JSONDecodeError
            arr.import_nvd_file(str(malformed))
