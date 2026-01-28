"""Tests for enricher classes (EPSS, KEV).

Provides:
- TestEPSSEnricher: Tests for EPSS enrichment functionality
- TestKEVEnricher: Tests for KEV enrichment functionality

These tests verify the import_epss and import_kev methods on CVDArray
which enrich vulnerabilities with external data sources.
"""

import numpy as np
import pytest

from vulnstate import CVDArray, CVDEvent, CVDVulnerability

pytestmark = pytest.mark.unit


class TestEPSSEnricher:
    """Tests for EPSS enrichment functionality."""

    @pytest.fixture
    def array_with_cves(self) -> CVDArray:
        """Array with known CVE IDs."""
        vulns = [
            CVDVulnerability(cve_id="CVE-2024-0001"),
            CVDVulnerability(cve_id="CVE-2024-0002"),
            CVDVulnerability(cve_id="CVE-2024-0003"),
        ]
        return CVDArray(vulns)

    @pytest.fixture
    def epss_data(self) -> dict:
        """Sample EPSS data as dict."""
        return {
            "CVE-2024-0001": {"score": 0.5, "percentile": 0.9},
            "CVE-2024-0002": {"score": 0.1, "percentile": 0.3},
            "CVE-2024-9999": {"score": 0.9, "percentile": 0.99},  # Not in array
        }

    def test_import_epss_matches_by_cve_id(self, array_with_cves, epss_data):
        """EPSS import should match CVEs by ID."""
        array_with_cves.import_epss(epss_data)

        # CVE-2024-0001 should have EPSS 0.5
        assert array_with_cves.get(0).epss == pytest.approx(0.5)
        # CVE-2024-0002 should have EPSS 0.1
        assert array_with_cves.get(1).epss == pytest.approx(0.1)

    def test_import_epss_skips_unmatched(self, array_with_cves, epss_data):
        """Unmatched CVEs in EPSS data should be skipped."""
        array_with_cves.import_epss(epss_data)

        # CVE-2024-0003 not in data, should have no EPSS
        assert array_with_cves.get(2).epss is None

    def test_import_epss_updates_percentile(self, array_with_cves, epss_data):
        """EPSS import should also set percentile."""
        array_with_cves.import_epss(epss_data)

        assert array_with_cves.get(0).epss_percentile == pytest.approx(0.9)
        assert array_with_cves.get(1).epss_percentile == pytest.approx(0.3)

    def test_import_epss_empty_array(self):
        """EPSS import on empty array should not crash."""
        arr = CVDArray([])
        epss_data = {"CVE-2024-0001": {"score": 0.5, "percentile": 0.9}}
        arr.import_epss(epss_data)  # Should not raise
        assert len(arr) == 0

    def test_import_epss_simple_float_values(self):
        """EPSS import accepts simple float values (score only)."""
        vulns = [
            CVDVulnerability(cve_id="CVE-2024-0001"),
            CVDVulnerability(cve_id="CVE-2024-0002"),
        ]
        arr = CVDArray(vulns)

        # Dict with float values (no percentile)
        epss_data = {"CVE-2024-0001": 0.85, "CVE-2024-0002": 0.15}
        arr.import_epss(epss_data)

        assert arr.get(0).epss == pytest.approx(0.85)
        assert arr.get(1).epss == pytest.approx(0.15)

    def test_import_epss_updates_epss_scores_list(self):
        """EPSS import populates the epss_scores list on vulnerability."""
        vulns = [CVDVulnerability(cve_id="CVE-2024-0001")]
        arr = CVDArray(vulns)

        epss_data = {"CVE-2024-0001": {"score": 0.75, "percentile": 0.85}}
        arr.import_epss(epss_data)

        vuln = arr.get(0)
        assert len(vuln.epss_scores) >= 1
        assert vuln.epss_scores[0].probability == pytest.approx(0.75)
        assert vuln.epss_scores[0].percentile == pytest.approx(0.85)

    def test_import_epss_with_metadata(self):
        """EPSS import with import_metadata=True stores full metadata."""
        vulns = [CVDVulnerability(cve_id="CVE-2024-0001")]
        arr = CVDArray(vulns)

        epss_data = {"CVE-2024-0001": {"score": 0.75, "percentile": 0.85}}
        arr.import_epss(epss_data, import_metadata=True)

        vuln = arr.get(0)
        assert "epss" in vuln.metadata
        assert vuln.metadata["epss"]["score"] == 0.75
        assert vuln.metadata["epss"]["percentile"] == 0.85


class TestKEVEnricher:
    """Tests for KEV enrichment functionality."""

    @pytest.fixture
    def array_with_cves(self) -> CVDArray:
        """Array with known CVE IDs."""
        vulns = [
            CVDVulnerability(cve_id="CVE-2024-0001"),
            CVDVulnerability(cve_id="CVE-2024-0002"),
            CVDVulnerability(cve_id="CVE-2024-0003"),
        ]
        return CVDArray(vulns)

    @pytest.fixture
    def kev_data(self) -> dict:
        """Sample KEV data as dict."""
        return {
            "CVE-2024-0001": {
                "dateAdded": "2024-01-15",
                "vendorProject": "TestVendor",
                "product": "TestProduct",
                "knownRansomwareCampaignUse": "Known",
            },
            "CVE-2024-0002": {
                "dateAdded": "2024-02-20",
                "vendorProject": "OtherVendor",
                "product": "OtherProduct",
                "knownRansomwareCampaignUse": "Unknown",
            },
        }

    def test_import_kev_sets_kev_flag(self, array_with_cves, kev_data):
        """KEV import should set kev flag on matching CVEs."""
        array_with_cves.import_kev(kev_data, apply_event=False)

        assert array_with_cves.get(0).kev is True
        assert array_with_cves.get(1).kev is True
        assert array_with_cves.get(2).kev is False  # Not in KEV

    def test_import_kev_sets_entry(self, array_with_cves, kev_data):
        """KEV import should set kev_entry on matching CVEs."""
        array_with_cves.import_kev(kev_data, apply_event=False)

        assert array_with_cves.get(0).kev_entry is not None
        assert array_with_cves.get(1).kev_entry is not None
        assert array_with_cves.get(2).kev_entry is None  # Not in KEV

    def test_import_kev_parses_date(self, array_with_cves, kev_data):
        """KEV import should parse dateAdded."""
        from datetime import datetime

        array_with_cves.import_kev(kev_data, apply_event=False)

        entry = array_with_cves.get(0).kev_entry
        assert entry.added_at == datetime(2024, 1, 15)

    def test_import_kev_extracts_metadata(self, array_with_cves, kev_data):
        """KEV import should extract vendor, product fields to metadata."""
        array_with_cves.import_kev(kev_data, apply_event=False)

        vuln = array_with_cves.get(0)
        # KEV fields are stored in metadata with kev_ prefix
        assert vuln.metadata["kev_vendor_name"] == "TestVendor"
        assert vuln.metadata["kev_product_name"] == "TestProduct"
        assert vuln.metadata["kev_ransomware_use"] == "Known"

    def test_import_kev_empty_array(self):
        """KEV import on empty array should not crash."""
        arr = CVDArray([])
        kev_data = {"CVE-2024-0001": {"dateAdded": "2024-01-15"}}
        arr.import_kev(kev_data)  # Should not raise
        assert len(arr) == 0

    def test_import_kev_applies_event_a(self, array_with_cves, kev_data):
        """KEV import with apply_event=True sets event A."""
        array_with_cves.import_kev(kev_data, apply_event=True)

        # Event A should be applied with dateAdded timestamp
        vuln = array_with_cves.get(0)
        assert vuln.has_event_occurred(CVDEvent.A)
        assert vuln.events[CVDEvent.A] == np.datetime64("2024-01-15")

    def test_import_kev_without_event_application(self, array_with_cves, kev_data):
        """KEV import with apply_event=False only sets kev flag."""
        array_with_cves.import_kev(kev_data, apply_event=False)

        vuln = array_with_cves.get(0)
        # Should NOT have event A
        assert not vuln.has_event_occurred(CVDEvent.A)
        # But should be marked as KEV
        assert vuln.kev is True

    def test_import_kev_missing_date(self):
        """KEV import handles missing dateAdded gracefully."""
        vulns = [CVDVulnerability(cve_id="CVE-2024-0001")]
        arr = CVDArray(vulns)

        # KEV data without dateAdded
        kev_data = {"CVE-2024-0001": {"vendorProject": "TestVendor"}}
        arr.import_kev(kev_data, apply_event=True)

        vuln = arr.get(0)
        # Should not have event A since dateAdded is missing
        assert not vuln.has_event_occurred(CVDEvent.A)
        # But should still be marked as KEV
        assert vuln.kev is True

    def test_import_kev_updates_array_kev_property(self, array_with_cves, kev_data):
        """KEV import updates the array-level kev property."""
        array_with_cves.import_kev(kev_data, apply_event=False)
        array_with_cves.transform()

        # Array-level kev property should reflect individual vuln KEV status
        kev_array = array_with_cves.kev
        assert kev_array[0] is True or kev_array[0] == np.True_
        assert kev_array[1] is True or kev_array[1] == np.True_
        assert kev_array[2] is False or kev_array[2] == np.False_


class TestEnricherIntegration:
    """Integration tests for enrichers working together."""

    def test_import_epss_and_kev_together(self):
        """Both EPSS and KEV can be imported on the same array."""
        vulns = [
            CVDVulnerability(cve_id="CVE-2024-0001"),
            CVDVulnerability(cve_id="CVE-2024-0002"),
        ]
        arr = CVDArray(vulns)

        # Import EPSS
        epss_data = {"CVE-2024-0001": 0.85, "CVE-2024-0002": 0.15}
        arr.import_epss(epss_data)

        # Import KEV
        kev_data = {"CVE-2024-0001": {"dateAdded": "2024-01-15"}}
        arr.import_kev(kev_data, apply_event=False)

        # Verify both enrichments are present
        assert arr.get(0).epss == pytest.approx(0.85)
        assert arr.get(0).kev is True
        assert arr.get(1).epss == pytest.approx(0.15)
        assert arr.get(1).kev is False

    def test_enrichers_sync_to_array(self):
        """Enricher imports properly sync to array-level properties."""
        vulns = [
            CVDVulnerability(cve_id="CVE-2024-0001"),
            CVDVulnerability(cve_id="CVE-2024-0002"),
        ]
        arr = CVDArray(vulns)

        epss_data = {"CVE-2024-0001": 0.85, "CVE-2024-0002": 0.15}
        arr.import_epss(epss_data)
        arr.transform()

        # Check array-level properties
        assert arr.epss[0] == pytest.approx(0.85)
        assert arr.epss[1] == pytest.approx(0.15)
