"""
Regression Tests for NVD Parser Refactoring

Ensures that the NVDParser extraction maintains backward compatibility
and correctly handles both NVD 1.1 and 2.0 formats.

Coverage:
- Format detection (1.1 vs 2.0)
- CVE ID extraction from both formats
- CVSS extraction with v3.1/v3.0/v2.0 fallback
- Published date extraction
- Backward compatibility via CVDIO delegation
- Integration with CVDArray
"""

import json
from datetime import datetime

import pytest

from vulnstate import CVDArray
from vulnstate.constants import CVDEvent
from vulnstate.io import CVDIO
from vulnstate.parsers import NVDParser


class TestNVDFormatDetection:
    """Test automatic format detection."""

    def test_detect_nvd_20_with_version_field(self):
        """Detect NVD 2.0 from explicit version field."""
        feed = {"version": "2.0", "vulnerabilities": []}
        assert NVDParser.detect_format(feed) == "2.0"

    def test_detect_nvd_20_with_vulnerabilities_array(self):
        """Detect NVD 2.0 from vulnerabilities array."""
        feed = {"vulnerabilities": []}
        assert NVDParser.detect_format(feed) == "2.0"

    def test_detect_nvd_11_with_cve_items_array(self):
        """Detect NVD 1.1 from CVE_Items array."""
        feed = {"CVE_Items": []}
        assert NVDParser.detect_format(feed) == "1.1"

    def test_detect_defaults_to_11(self):
        """Default to NVD 1.1 for ambiguous feeds."""
        feed = {}
        assert NVDParser.detect_format(feed) == "1.1"


class TestCVEIDExtraction:
    """Test CVE ID extraction from both formats."""

    def test_extract_cve_id_from_nvd_20(self):
        """Extract CVE ID from NVD 2.0 format."""
        item = {"cve": {"id": "CVE-2023-12345"}}
        cve_id = NVDParser.extract_cve_id(item, "2.0")
        assert cve_id == "CVE-2023-12345"

    def test_extract_cve_id_from_nvd_11(self):
        """Extract CVE ID from NVD 1.1 format."""
        item = {"cve": {"CVE_data_meta": {"ID": "CVE-2023-12345"}}}
        cve_id = NVDParser.extract_cve_id(item, "1.1")
        assert cve_id == "CVE-2023-12345"

    def test_extract_cve_id_missing_data(self):
        """Return None for missing CVE ID."""
        item = {"cve": {}}
        assert NVDParser.extract_cve_id(item, "2.0") is None
        assert NVDParser.extract_cve_id(item, "1.1") is None


class TestCVSSExtraction:
    """Test CVSS score extraction with fallback chain."""

    def test_extract_cvss_v31_primary_nvd_20(self):
        """Extract CVSS v3.1 Primary source from NVD 2.0."""
        item = {
            "cve": {
                "metrics": {
                    "cvssMetricV31": [
                        {"type": "Secondary", "cvssData": {"baseScore": 7.0, "vectorString": "CVSS:3.1/AV:N/AC:L"}},
                        {"type": "Primary", "cvssData": {"baseScore": 9.8, "vectorString": "CVSS:3.1/AV:N/AC:H"}},
                    ]
                }
            }
        }
        score, vector = NVDParser.extract_cvss(item, "2.0")
        assert score == 9.8  # Should prefer Primary
        assert vector == "CVSS:3.1/AV:N/AC:H"

    def test_extract_cvss_v31_fallback_nvd_20(self):
        """Fallback to first v3.1 entry if no Primary."""
        item = {
            "cve": {
                "metrics": {
                    "cvssMetricV31": [
                        {"type": "Secondary", "cvssData": {"baseScore": 7.5, "vectorString": "CVSS:3.1/AV:N"}},
                    ]
                }
            }
        }
        score, vector = NVDParser.extract_cvss(item, "2.0")
        assert score == 7.5
        assert vector == "CVSS:3.1/AV:N"

    def test_extract_cvss_v30_fallback_nvd_20(self):
        """Fallback to CVSS v3.0 if v3.1 unavailable."""
        item = {
            "cve": {
                "metrics": {
                    "cvssMetricV30": [
                        {"cvssData": {"baseScore": 6.5, "vectorString": "CVSS:3.0/AV:L"}},
                    ]
                }
            }
        }
        score, vector = NVDParser.extract_cvss(item, "2.0")
        assert score == 6.5
        assert vector == "CVSS:3.0/AV:L"

    def test_extract_cvss_v2_fallback_nvd_20(self):
        """Fallback to CVSS v2.0 if v3.x unavailable."""
        item = {
            "cve": {
                "metrics": {
                    "cvssMetricV2": [
                        {"cvssData": {"baseScore": 5.0, "vectorString": "AV:N/AC:L"}},
                    ]
                }
            }
        }
        score, vector = NVDParser.extract_cvss(item, "2.0")
        assert score == 5.0
        assert vector == "AV:N/AC:L"

    def test_extract_cvss_v3_nvd_11(self):
        """Extract CVSS v3 from NVD 1.1 format."""
        item = {
            "impact": {
                "baseMetricV3": {
                    "cvssV3": {"baseScore": 8.8, "vectorString": "CVSS:3.0/AV:N/AC:L"}
                }
            }
        }
        score, vector = NVDParser.extract_cvss(item, "1.1")
        assert score == 8.8
        assert vector == "CVSS:3.0/AV:N/AC:L"

    def test_extract_cvss_v2_fallback_nvd_11(self):
        """Fallback to CVSS v2 in NVD 1.1 format."""
        item = {
            "impact": {
                "baseMetricV2": {
                    "cvssV2": {"baseScore": 7.5, "vectorString": "AV:N/AC:L"}
                }
            }
        }
        score, vector = NVDParser.extract_cvss(item, "1.1")
        assert score == 7.5
        assert vector == "AV:N/AC:L"

    def test_extract_cvss_no_data(self):
        """Return None when no CVSS data available."""
        item = {"cve": {"metrics": {}}}
        score, vector = NVDParser.extract_cvss(item, "2.0")
        assert score is None
        assert vector is None


class TestPublishedDateExtraction:
    """Test published date extraction from both formats."""

    def test_extract_published_date_nvd_20(self):
        """Extract published date from NVD 2.0 format."""
        item = {"cve": {"published": "2023-01-15T10:00:00.000"}}
        date = NVDParser.extract_published_date(item, "2.0")
        assert date == "2023-01-15T10:00:00.000"

    def test_extract_published_date_nvd_11(self):
        """Extract published date from NVD 1.1 format."""
        item = {"publishedDate": "2023-01-15T10:00:00.000"}
        date = NVDParser.extract_published_date(item, "1.1")
        assert date == "2023-01-15T10:00:00.000"

    def test_extract_published_date_missing(self):
        """Return None for missing published date."""
        item = {"cve": {}}
        assert NVDParser.extract_published_date(item, "2.0") is None


class TestVulnerabilityCreation:
    """Test vulnerability creation from NVD items."""

    def test_create_vuln_from_nvd_20_item(self):
        """Create vulnerability from NVD 2.0 item."""
        item = {
            "cve": {
                "id": "CVE-2023-12345",
                "published": "2023-01-15T10:00:00.000",
                "metrics": {
                    "cvssMetricV31": [
                        {"type": "Primary", "cvssData": {"baseScore": 9.8, "vectorString": "CVSS:3.1/AV:N"}}
                    ]
                },
            }
        }
        vuln = NVDParser.create_vuln_from_item("CVE-2023-12345", item, "2.0")

        assert vuln.cve_id == "CVE-2023-12345"
        assert vuln.cvss_score == 9.8
        assert vuln.cve_vector == "CVSS:3.1/AV:N"
        assert CVDEvent.P in vuln.events

    def test_create_vuln_from_nvd_11_item(self):
        """Create vulnerability from NVD 1.1 item."""
        item = {
            "publishedDate": "2023-01-15T10:00:00.000",
            "impact": {
                "baseMetricV3": {
                    "cvssV3": {"baseScore": 7.5, "vectorString": "CVSS:3.0/AV:N"}
                }
            },
        }
        vuln = NVDParser.create_vuln_from_item("CVE-2023-12345", item, "1.1")

        assert vuln.cve_id == "CVE-2023-12345"
        assert vuln.cvss_score == 7.5
        assert vuln.cve_vector == "CVSS:3.0/AV:N"
        assert CVDEvent.P in vuln.events


class TestBackwardCompatibility:
    """Test backward compatibility via CVDIO delegation."""

    def test_import_nvd_via_cvdio(self):
        """Ensure CVDIO.import_nvd() delegates to NVDParser."""
        arr = CVDArray()
        nvd_items = [
            {
                "cve": {
                    "id": "CVE-2023-0001",
                    "metrics": {
                        "cvssMetricV31": [
                            {"type": "Primary", "cvssData": {"baseScore": 8.8, "vectorString": "CVSS:3.1/AV:N"}}
                        ]
                    },
                }
            }
        ]

        # Should work via CVDIO delegation
        CVDIO.import_nvd(arr, nvd_items)

        assert len(arr) == 1
        assert arr.get(0).cve_id == "CVE-2023-0001"
        assert arr.get(0).cvss_score == 8.8

    def test_import_nvd_via_array(self):
        """Ensure CVDArray.import_nvd() works."""
        arr = CVDArray()
        nvd_items = [
            {
                "cve": {
                    "id": "CVE-2023-0002",
                    "metrics": {
                        "cvssMetricV31": [
                            {"type": "Primary", "cvssData": {"baseScore": 7.5}}
                        ]
                    },
                }
            }
        ]

        # Should work via array method
        arr.import_nvd(nvd_items)

        assert len(arr) == 1
        assert arr.get(0).cve_id == "CVE-2023-0002"

    def test_from_nvd_via_cvdio(self):
        """Ensure CVDIO.from_nvd() delegates to NVDParser."""
        nvd_items = [
            {"cve": {"id": "CVE-2023-0003", "metrics": {}}}
        ]

        arr = CVDIO.from_nvd(nvd_items, format_version="2.0")

        assert len(arr) == 1
        assert arr.get(0).cve_id == "CVE-2023-0003"


class TestIntegration:
    """Integration tests with real-world scenarios."""

    def test_import_mixed_cvss_versions(self):
        """Handle items with different CVSS versions."""
        arr = CVDArray()
        nvd_items = [
            # Has v3.1
            {
                "cve": {
                    "id": "CVE-2023-0001",
                    "metrics": {
                        "cvssMetricV31": [
                            {"type": "Primary", "cvssData": {"baseScore": 9.8}}
                        ]
                    },
                }
            },
            # Has only v3.0
            {
                "cve": {
                    "id": "CVE-2023-0002",
                    "metrics": {
                        "cvssMetricV30": [
                            {"cvssData": {"baseScore": 7.5}}
                        ]
                    },
                }
            },
            # Has only v2
            {
                "cve": {
                    "id": "CVE-2023-0003",
                    "metrics": {
                        "cvssMetricV2": [
                            {"cvssData": {"baseScore": 5.0}}
                        ]
                    },
                }
            },
        ]

        NVDParser.import_nvd(arr, nvd_items)

        assert len(arr) == 3
        assert arr.get(0).cvss_score == 9.8
        assert arr.get(1).cvss_score == 7.5
        assert arr.get(2).cvss_score == 5.0

    def test_import_with_metadata(self):
        """Import NVD data with metadata storage."""
        arr = CVDArray()
        nvd_items = [
            {
                "cve": {
                    "id": "CVE-2023-0001",
                    "metrics": {},
                    "descriptions": [{"lang": "en", "value": "Test description"}],
                }
            }
        ]

        NVDParser.import_nvd(arr, nvd_items, import_metadata=True)

        assert len(arr) == 1
        assert "nvd" in arr.get(0).metadata
        assert "descriptions" in arr.get(0).metadata["nvd"]["cve"]

    def test_update_existing_vulnerability(self):
        """Update existing vulnerability with new NVD data."""
        arr = CVDArray()

        # Initial import without CVSS
        nvd_items_v1 = [{"cve": {"id": "CVE-2023-0001", "metrics": {}}}]
        NVDParser.import_nvd(arr, nvd_items_v1)
        assert arr.get(0).cvss_score is None

        # Update with CVSS data
        nvd_items_v2 = [
            {
                "cve": {
                    "id": "CVE-2023-0001",
                    "metrics": {
                        "cvssMetricV31": [
                            {"type": "Primary", "cvssData": {"baseScore": 8.8}}
                        ]
                    },
                }
            }
        ]
        NVDParser.import_nvd(arr, nvd_items_v2, skip_existing=False)

        # Should update CVSS
        assert len(arr) == 1
        assert arr.get(0).cvss_score == 8.8


class TestRegressionScenarios:
    """Test specific regression scenarios from the refactoring."""

    def test_format_detection_preserves_behavior(self):
        """Ensure format detection matches original behavior."""
        # NVD 2.0 feed structure
        feed_20 = {
            "version": "2.0",
            "vulnerabilities": [{"cve": {"id": "CVE-2023-0001"}}],
        }

        # NVD 1.1 feed structure
        feed_11 = {
            "CVE_Items": [{"cve": {"CVE_data_meta": {"ID": "CVE-2023-0001"}}}],
        }

        assert NVDParser.detect_format(feed_20) == "2.0"
        assert NVDParser.detect_format(feed_11) == "1.1"

    def test_cvss_fallback_chain_preserves_order(self):
        """Ensure CVSS fallback order: v3.1 → v3.0 → v2.0."""
        # Item with all versions (should prefer v3.1)
        item = {
            "cve": {
                "metrics": {
                    "cvssMetricV31": [{"type": "Primary", "cvssData": {"baseScore": 9.8}}],
                    "cvssMetricV30": [{"cvssData": {"baseScore": 7.5}}],
                    "cvssMetricV2": [{"cvssData": {"baseScore": 5.0}}],
                }
            }
        }

        score, _ = NVDParser.extract_cvss(item, "2.0")
        assert score == 9.8  # Should prefer v3.1

    def test_empty_array_optimization(self):
        """Verify empty array optimization path."""
        arr = CVDArray()
        nvd_items = [
            {"cve": {"id": "CVE-2023-0001", "metrics": {}}},
            {"cve": {"id": "CVE-2023-0002", "metrics": {}}},
        ]

        NVDParser.import_nvd(arr, nvd_items)

        # Should use fast path for empty array
        assert len(arr) == 2
        assert arr.get(0).cve_id == "CVE-2023-0001"
        assert arr.get(1).cve_id == "CVE-2023-0002"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
