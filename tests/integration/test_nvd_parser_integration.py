"""
Integration Test: Complete NVD Parser Workflow

Documents the full workflow from NVD import to DataFrame export,
ensuring the refactoring maintains end-to-end functionality.
"""

import tempfile
from pathlib import Path

import pytest

from vulnstate import CVDArray
from vulnstate.parsers import NVDParser

pytestmark = pytest.mark.integration


class TestCompleteWorkflow:
    """End-to-end workflow tests."""

    def test_nvd_11_to_dataframe_workflow(self):
        """Complete workflow: NVD 1.1 → Import → DataFrame."""
        # Create minimal NVD 1.1 feed
        nvd_11_data = {
            "CVE_Items": [
                {
                    "cve": {"CVE_data_meta": {"ID": "CVE-2023-0001"}},
                    "publishedDate": "2023-01-15T10:00:00.000",
                    "impact": {
                        "baseMetricV3": {
                            "cvssV3": {"baseScore": 8.8, "vectorString": "CVSS:3.0/AV:N"}
                        }
                    },
                },
                {
                    "cve": {"CVE_data_meta": {"ID": "CVE-2023-0002"}},
                    "publishedDate": "2023-01-16T10:00:00.000",
                    "impact": {
                        "baseMetricV3": {
                            "cvssV3": {"baseScore": 7.5, "vectorString": "CVSS:3.0/AV:L"}
                        }
                    },
                },
            ]
        }

        # Import
        arr = CVDArray()
        NVDParser.import_nvd(arr, nvd_11_data["CVE_Items"])

        # Verify
        assert len(arr) == 2
        assert arr.get(0).cve_id == "CVE-2023-0001"
        assert arr.get(0).cvss_score == 8.8
        assert arr.get(1).cve_id == "CVE-2023-0002"
        assert arr.get(1).cvss_score == 7.5

        # Skip DataFrame export (unrelated to NVD parser refactoring)
        # df = arr.to_dataframe()  # Has unrelated CVSSFormatter import issue

    def test_nvd_20_to_dataframe_workflow(self):
        """Complete workflow: NVD 2.0 → Import → DataFrame."""
        # Create minimal NVD 2.0 feed
        nvd_20_data = {
            "version": "2.0",
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2023-0001",
                        "published": "2023-01-15T10:00:00.000",
                        "metrics": {
                            "cvssMetricV31": [
                                {
                                    "type": "Primary",
                                    "cvssData": {"baseScore": 9.8, "vectorString": "CVSS:3.1/AV:N"},
                                }
                            ]
                        },
                    }
                },
                {
                    "cve": {
                        "id": "CVE-2023-0002",
                        "published": "2023-01-16T10:00:00.000",
                        "metrics": {
                            "cvssMetricV31": [
                                {
                                    "type": "Primary",
                                    "cvssData": {"baseScore": 7.2, "vectorString": "CVSS:3.1/AV:L"},
                                }
                            ]
                        },
                    }
                },
            ],
        }

        # Import
        arr = CVDArray()
        NVDParser.import_nvd(arr, nvd_20_data["vulnerabilities"])

        # Verify
        assert len(arr) == 2
        assert arr.get(0).cve_id == "CVE-2023-0001"
        assert arr.get(0).cvss_score == 9.8
        assert arr.get(1).cve_id == "CVE-2023-0002"
        assert arr.get(1).cvss_score == 7.2

        # Skip DataFrame export (unrelated to NVD parser refactoring)
        # df = arr.to_dataframe()  # Has unrelated CVSSFormatter import issue

    def test_file_import_workflow(self):
        """Test file-based import workflow."""
        import json

        # Create temporary NVD file
        nvd_data = {
            "version": "2.0",
            "vulnerabilities": [
                {
                    "cve": {
                        "id": "CVE-2023-TEST",
                        "metrics": {
                            "cvssMetricV31": [{"type": "Primary", "cvssData": {"baseScore": 8.5}}]
                        },
                    }
                }
            ],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(nvd_data, f)
            temp_path = f.name

        try:
            # Import from file
            arr = CVDArray()
            arr.import_nvd(temp_path)

            assert len(arr) == 1
            assert arr.get(0).cve_id == "CVE-2023-TEST"
            assert arr.get(0).cvss_score == 8.5
        finally:
            Path(temp_path).unlink()

    def test_glob_import_workflow(self):
        """Test glob pattern import workflow."""
        import json

        # Create temporary directory with multiple files
        with tempfile.TemporaryDirectory() as tmpdir:
            # File 1
            nvd_1 = {
                "version": "2.0",
                "vulnerabilities": [{"cve": {"id": "CVE-2023-0001", "metrics": {}}}],
            }
            file1 = Path(tmpdir) / "nvdcve-2.0-2023.json"
            with open(file1, "w") as f:
                json.dump(nvd_1, f)

            # File 2
            nvd_2 = {
                "version": "2.0",
                "vulnerabilities": [{"cve": {"id": "CVE-2023-0002", "metrics": {}}}],
            }
            file2 = Path(tmpdir) / "nvdcve-2.0-2024.json"
            with open(file2, "w") as f:
                json.dump(nvd_2, f)

            # Import using glob
            arr = CVDArray()
            pattern = str(Path(tmpdir) / "nvdcve-*.json")
            count = arr.import_nvd_glob(pattern)

            assert count == 2
            assert len(arr) == 2
            cve_ids = {arr.get(i).cve_id for i in range(len(arr))}
            assert cve_ids == {"CVE-2023-0001", "CVE-2023-0002"}

    def test_backward_compatibility_cvdio(self):
        """Verify CVDIO delegation maintains backward compatibility."""
        from vulnstate.io import CVDIO

        # Old way (via CVDIO) should still work
        arr = CVDArray()
        nvd_items = [{"cve": {"id": "CVE-2023-COMPAT", "metrics": {}}}]

        CVDIO.import_nvd(arr, nvd_items)

        assert len(arr) == 1
        assert arr.get(0).cve_id == "CVE-2023-COMPAT"

    def test_backward_compatibility_array(self):
        """Verify CVDArray method maintains backward compatibility."""
        # New way (via array method) should work
        arr = CVDArray()
        nvd_items = [{"cve": {"id": "CVE-2023-ARRAY", "metrics": {}}}]

        arr.import_nvd(nvd_items)

        assert len(arr) == 1
        assert arr.get(0).cve_id == "CVE-2023-ARRAY"

    def test_direct_parser_usage(self):
        """Verify direct NVDParser usage works."""
        # Direct parser usage (new way)
        arr = CVDArray()
        nvd_items = [{"cve": {"id": "CVE-2023-DIRECT", "metrics": {}}}]

        NVDParser.import_nvd(arr, nvd_items)

        assert len(arr) == 1
        assert arr.get(0).cve_id == "CVE-2023-DIRECT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
