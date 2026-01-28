"""Tests for infer_events() offset-based and heuristic event inference."""

from datetime import datetime, timedelta

import pytest

from vulnstate import CVDArray, CVDEvent, CVDVulnerability
from vulnstate.transforms.enrichers import infer_events

pytestmark = pytest.mark.unit


def _make_vuln_with_tags(cve_id, tags, published="2023-01-15", last_modified="2023-06-15"):
    """Create a CVDArray with one CVE imported from NVD item with given tags."""
    item = {
        "cve": {
            "CVE_data_meta": {"ID": cve_id},
            "description": {"description_data": [{"lang": "en", "value": "Test"}]},
            "references": {"reference_data": [{"url": "https://example.com", "tags": tags}]},
        },
        "publishedDate": f"{published}T00:00Z",
        "lastModifiedDate": f"{last_modified}T00:00Z",
        "impact": {},
        "configurations": {"nodes": []},
    }
    arr = CVDArray()
    arr.import_nvd([item], infer_vendor=False)  # Don't auto-set V
    return arr


def _make_vuln_with_cpe(
    cve_id, version_end_excluding, published="2023-01-15", last_modified="2023-06-15"
):
    """Create a CVDArray with one CVE that has versionEndExcluding in CPE."""
    item = {
        "cve": {
            "CVE_data_meta": {"ID": cve_id},
            "description": {"description_data": [{"lang": "en", "value": "Test"}]},
            "references": {"reference_data": []},
        },
        "publishedDate": f"{published}T00:00Z",
        "lastModifiedDate": f"{last_modified}T00:00Z",
        "impact": {},
        "configurations": {
            "nodes": [
                {
                    "cpe_match": [
                        {
                            "cpe23Uri": "cpe:2.3:a:vendor:product:*:*:*:*:*:*:*:*",
                            "versionEndExcluding": version_end_excluding,
                            "vulnerable": True,
                        }
                    ]
                }
            ]
        },
    }
    arr = CVDArray()
    arr.import_nvd([item], infer_vendor=False)
    return arr


class TestInferEventsVendor:
    """V inference from advisory tags with configurable offsets."""

    def test_vendor_advisory_offset(self):
        arr = _make_vuln_with_tags("CVE-2023-0001", ["Vendor Advisory"])
        infer_events(arr, vendor_lead=7)
        vuln = arr.get(0)
        p_ts = vuln.events[CVDEvent.P]
        v_ts = vuln.events[CVDEvent.V]
        # V should be P - 7 days
        assert v_ts == p_ts - timedelta(days=7)

    def test_third_party_advisory_offset(self):
        arr = _make_vuln_with_tags("CVE-2023-0001", ["Third Party Advisory"])
        infer_events(arr, thirdparty_lag=14)
        vuln = arr.get(0)
        p_ts = vuln.events[CVDEvent.P]
        v_ts = vuln.events[CVDEvent.V]
        # V should be P + 14 days
        assert v_ts == p_ts + timedelta(days=14)

    def test_patch_tag_implies_V(self):
        arr = _make_vuln_with_tags("CVE-2023-0001", ["Patch"])
        infer_events(arr)
        vuln = arr.get(0)
        # Patch implies vendor was aware
        assert vuln.has_event_occurred(CVDEvent.V)


class TestInferEventsFix:
    """F inference from heuristics (CPE, CVSS temporal)."""

    def test_cpe_version_boundary_infers_F(self):
        arr = _make_vuln_with_cpe("CVE-2023-0001", "4.2.0")
        result = infer_events(arr, heuristics=True)
        vuln = arr.get(0)
        assert vuln.has_event_occurred(CVDEvent.F)
        assert CVDEvent.F in vuln.state.inferred_events
        assert result["F_inferred"] >= 1

    def test_heuristics_false_skips_cpe(self):
        arr = _make_vuln_with_cpe("CVE-2023-0001", "4.2.0")
        infer_events(arr, heuristics=False)
        vuln = arr.get(0)
        assert not vuln.has_event_occurred(CVDEvent.F)


class TestInferEventsDeploy:
    """D inference from F + configurable lag."""

    def test_deploy_true_applies_D(self):
        arr = _make_vuln_with_tags("CVE-2023-0001", ["Patch"])
        result = infer_events(arr, deploy=True, deploy_lag=30)
        vuln = arr.get(0)
        assert vuln.has_event_occurred(CVDEvent.D)
        f_ts = vuln.events[CVDEvent.F]
        d_ts = vuln.events[CVDEvent.D]
        if f_ts is not None and d_ts is not None:
            assert d_ts == f_ts + timedelta(days=30)
        assert result["D_inferred"] >= 1

    def test_deploy_false_skips_D(self):
        arr = _make_vuln_with_tags("CVE-2023-0001", ["Patch"])
        infer_events(arr, deploy=False)
        vuln = arr.get(0)
        assert not vuln.has_event_occurred(CVDEvent.D)

    def test_severity_adjusted_deploy_lag(self):
        # Create CVE with CRITICAL CVSS
        item = {
            "cve": {
                "CVE_data_meta": {"ID": "CVE-2023-0001"},
                "description": {"description_data": [{"lang": "en", "value": "Test"}]},
                "references": {"reference_data": [{"url": "https://x.com", "tags": ["Patch"]}]},
            },
            "publishedDate": "2023-01-15T00:00Z",
            "lastModifiedDate": "2023-06-15T00:00Z",
            "impact": {
                "baseMetricV3": {
                    "cvssV3": {
                        "baseScore": 9.8,
                        "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                    },
                    "exploitabilityScore": 3.9,
                    "impactScore": 5.9,
                }
            },
            "configurations": {"nodes": []},
        }
        arr = CVDArray()
        arr.import_nvd([item], infer_vendor=False)
        infer_events(arr, deploy=True, severity_adjusted=True)
        vuln = arr.get(0)
        assert vuln.has_event_occurred(CVDEvent.D)
        f_ts = vuln.events[CVDEvent.F]
        d_ts = vuln.events[CVDEvent.D]
        if f_ts is not None and d_ts is not None:
            # CRITICAL = 7 day lag
            assert d_ts == f_ts + timedelta(days=7)


class TestInferEventsGeneral:
    """General infer_events() behavior."""

    def test_does_not_overwrite_authoritative(self):
        vuln = CVDVulnerability("CVE-2023-0001")
        real_ts = datetime(2023, 3, 1)
        vuln.apply_event(CVDEvent.P, timestamp=real_ts)
        vuln.apply_event(CVDEvent.V, timestamp=datetime(2023, 2, 15))  # Authoritative V
        arr = CVDArray([vuln])
        infer_events(arr, vendor_lead=30)
        # V should NOT be overwritten
        assert arr.get(0).events[CVDEvent.V] == datetime(2023, 2, 15)

    def test_returns_summary_dict(self):
        arr = _make_vuln_with_tags("CVE-2023-0001", ["Patch", "Vendor Advisory"])
        result = infer_events(arr, deploy=True, deploy_lag=30)
        assert isinstance(result, dict)
        assert "V_inferred" in result
        assert "F_inferred" in result
        assert "D_inferred" in result

    def test_no_events_returns_zeros(self):
        arr = _make_vuln_with_tags("CVE-2023-0001", [])
        result = infer_events(arr, deploy=False, heuristics=False)
        assert result["V_inferred"] == 0
        assert result["F_inferred"] == 0
        assert result["D_inferred"] == 0
