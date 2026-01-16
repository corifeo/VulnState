"""Tests for io.py - consolidated I/O operations."""

import tempfile
from pathlib import Path

from vulnstate import CVDArray, CVDVulnerability


class TestCVDIO:
    """Tests for CVDIO class."""

    def test_to_dict_single(self):
        """Convert single vulnerability to dict."""
        from vulnstate.io import CVDIO

        vuln = CVDVulnerability("CVE-2024-1234", cvss_score=9.8)
        data = CVDIO.to_dict(vuln)

        assert data["cve_id"] == "CVE-2024-1234"
        assert data["cvss_score"] == 9.8
        assert "state" in data

    def test_from_dict_single(self):
        """Reconstruct vulnerability from dict."""
        from vulnstate.io import CVDIO

        data = {
            "cve_id": "CVE-2024-5678",
            "vuln_id": "test-id-123",
            "state": "Vfdpxa",
            "cvss_score": 7.5,
        }
        vuln = CVDIO.from_dict(data)

        assert vuln.cve_id == "CVE-2024-5678"
        assert vuln.state == "Vfdpxa"

    def test_to_json_roundtrip(self):
        """JSON roundtrip preserves data."""
        from vulnstate.io import CVDIO

        vuln = CVDVulnerability("CVE-2024-0001", cvss_score=8.5)
        json_str = CVDIO.to_json(vuln)
        restored = CVDIO.from_json(json_str)

        assert restored.cve_id == vuln.cve_id
        assert restored.cvss_score == vuln.cvss_score

    def test_save_load_json_file(self):
        """Save and load JSON file."""
        from vulnstate.io import CVDIO

        vuln = CVDVulnerability("CVE-2024-FILE", cvss_score=6.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "vuln.json"
            CVDIO.save_json(vuln, str(filepath))
            loaded = CVDIO.load_json(str(filepath))

        assert loaded.cve_id == "CVE-2024-FILE"


class TestArrayIO:
    """Tests for array I/O operations."""

    def test_array_to_dict_list(self):
        """Convert array to list of dicts."""
        from vulnstate.io import CVDIO

        arr = CVDArray(
            [
                CVDVulnerability("CVE-2024-001"),
                CVDVulnerability("CVE-2024-002"),
            ]
        )
        data = CVDIO.array_to_dicts(arr)

        assert len(data) == 2
        assert data[0]["cve_id"] == "CVE-2024-001"
        assert data[1]["cve_id"] == "CVE-2024-002"

    def test_array_from_dict_list(self):
        """Create array from list of dicts."""
        from vulnstate.io import CVDIO

        data = [
            {"cve_id": "CVE-2024-A", "state": "vfdpxa"},
            {"cve_id": "CVE-2024-B", "state": "Vfdpxa"},
        ]
        arr = CVDIO.array_from_dicts(data)

        assert len(arr) == 2
        assert arr[0].cve_id == "CVE-2024-A"


class TestEnrichment:
    """Tests for enrichment data import."""

    def test_import_epss(self):
        """Import EPSS scores into array."""
        from vulnstate.io import CVDIO

        arr = CVDArray(
            [
                CVDVulnerability("CVE-2024-001"),
                CVDVulnerability("CVE-2024-002"),
            ]
        )
        epss_data = {"CVE-2024-001": 0.85, "CVE-2024-002": 0.15}

        CVDIO.import_epss(arr, epss_data)

        assert arr[0].epss == 0.85
        assert arr[1].epss == 0.15

    def test_import_kev(self):
        """Import KEV flags into array."""
        from vulnstate.io import CVDIO

        arr = CVDArray(
            [
                CVDVulnerability("CVE-2024-001"),
                CVDVulnerability("CVE-2024-002"),
            ]
        )
        kev_cves = {"CVE-2024-001"}

        CVDIO.import_kev(arr, kev_cves)

        assert arr[0].is_kev == True  # noqa: E712
        assert arr[1].is_kev == False  # noqa: E712


class TestNVDImport:
    """Tests for NVD format import."""

    def test_from_nvd_basic(self):
        """Import basic NVD items."""
        from vulnstate.io import CVDIO

        nvd_items = [
            {
                "cve": {"CVE_data_meta": {"ID": "CVE-2024-001"}},
                "impact": {
                    "baseMetricV3": {"cvssV3": {"baseScore": 9.8, "vectorString": "CVSS:3.1/AV:N"}}
                },
            },
            {
                "cve": {"CVE_data_meta": {"ID": "CVE-2024-002"}},
                "impact": {"baseMetricV2": {"cvssV2": {"baseScore": 5.0}}},
            },
        ]
        arr = CVDIO.from_nvd(nvd_items)

        assert len(arr) == 2
        assert arr[0].cve_id == "CVE-2024-001"
        assert arr[0].cvss_score == 9.8
        assert arr[1].cvss_score == 5.0

    def test_from_nvd_with_published_date(self):
        """NVD published date creates P event."""
        from vulnstate.io import CVDIO

        nvd_items = [
            {
                "cve": {"CVE_data_meta": {"ID": "CVE-2024-001"}},
                "impact": {},
                "publishedDate": "2024-01-15T12:00:00Z",
            },
        ]
        arr = CVDIO.from_nvd(nvd_items)

        assert len(arr) == 1
        # Should have P event from published date
        assert "P" in arr[0].state  # uppercase P means event occurred
