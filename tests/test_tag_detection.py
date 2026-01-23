"""Tests for NVD reference tag detection and event inference."""

from datetime import datetime

from vulnstate import CVDEvent, CVDVulnerability
from vulnstate.parsers import NVDParser


class TestInferredEventFlag:
    """VulnerabilityState tracks which events are inferred."""

    def test_inferred_events_empty_by_default(self):
        vuln = CVDVulnerability("CVE-2023-0001")
        assert vuln.state.inferred_events == set()

    def test_apply_event_not_inferred_by_default(self):
        vuln = CVDVulnerability("CVE-2023-0001")
        vuln.apply_event(CVDEvent.P, timestamp=datetime(2023, 1, 1))
        assert CVDEvent.P not in vuln.state.inferred_events

    def test_apply_event_inferred_flag(self):
        vuln = CVDVulnerability("CVE-2023-0001")
        vuln.apply_event(CVDEvent.V, timestamp=datetime(2023, 1, 1), inferred=True)
        assert CVDEvent.V in vuln.state.inferred_events

    def test_inferred_flag_with_none_timestamp(self):
        vuln = CVDVulnerability("CVE-2023-0001")
        vuln.apply_event(CVDEvent.V, timestamp=datetime(2023, 1, 1))
        vuln.apply_event(CVDEvent.F, timestamp=None, inferred=True)
        assert CVDEvent.F in vuln.state.inferred_events
        assert vuln.has_event_occurred(CVDEvent.F)
        assert not vuln.has_known_timestamp(CVDEvent.F)

    def test_rollback_removes_inferred_flag(self):
        vuln = CVDVulnerability("CVE-2023-0001")
        vuln.apply_event(CVDEvent.V, timestamp=datetime(2023, 1, 1), inferred=True)
        assert CVDEvent.V in vuln.state.inferred_events
        vuln.rollback_event()
        assert CVDEvent.V not in vuln.state.inferred_events


class TestExtractReferenceTags:
    """NVDParser extracts reference tags from NVD items."""

    def test_extract_tags_v11(self):
        item = {
            "cve": {
                "CVE_data_meta": {"ID": "CVE-2023-0001"},
                "references": {
                    "reference_data": [
                        {"url": "https://example.com/patch", "tags": ["Patch", "Vendor Advisory"]},
                        {"url": "https://example.com/exploit", "tags": ["Exploit"]},
                    ]
                },
            }
        }
        tags = NVDParser.extract_reference_tags(item, "1.1")
        assert "Patch" in tags
        assert "Vendor Advisory" in tags
        assert "Exploit" in tags

    def test_extract_tags_v20(self):
        item = {
            "cve": {
                "id": "CVE-2023-0001",
                "references": [
                    {"url": "https://example.com/patch", "tags": ["Patch"]},
                    {"url": "https://example.com/advisory", "tags": ["Third Party Advisory"]},
                ],
            }
        }
        tags = NVDParser.extract_reference_tags(item, "2.0")
        assert "Patch" in tags
        assert "Third Party Advisory" in tags

    def test_extract_tags_empty_references(self):
        item = {"cve": {"CVE_data_meta": {"ID": "CVE-2023-0001"}}}
        tags = NVDParser.extract_reference_tags(item, "1.1")
        assert tags == set()

    def test_extract_tags_no_tags_field(self):
        item = {
            "cve": {
                "CVE_data_meta": {"ID": "CVE-2023-0001"},
                "references": {"reference_data": [{"url": "https://example.com", "tags": []}]},
            }
        }
        tags = NVDParser.extract_reference_tags(item, "1.1")
        assert tags == set()


class TestTagEventDetection:
    """import_nvd applies events from reference tags."""

    def _make_nvd_item(
        self, tags_list, published="2023-01-15T00:00Z", last_modified="2023-06-15T00:00Z"
    ):
        """Helper: create NVD 1.1 item with given reference tags."""
        return {
            "cve": {
                "CVE_data_meta": {"ID": "CVE-2023-9999"},
                "description": {"description_data": [{"lang": "en", "value": "Test"}]},
                "references": {
                    "reference_data": [{"url": "https://example.com", "tags": tags_list}]
                },
            },
            "publishedDate": published,
            "lastModifiedDate": last_modified,
            "impact": {},
            "configurations": {"nodes": []},
        }

    def test_patch_tag_applies_F(self):
        from vulnstate import CVDArray

        arr = CVDArray()
        item = self._make_nvd_item(["Patch"])
        arr.import_nvd([item])
        vuln = arr.get(0)
        assert vuln.has_event_occurred(CVDEvent.F)
        assert CVDEvent.F in vuln.state.inferred_events

    def test_exploit_tag_applies_X(self):
        from vulnstate import CVDArray

        arr = CVDArray()
        item = self._make_nvd_item(["Exploit"])
        arr.import_nvd([item])
        vuln = arr.get(0)
        assert vuln.has_event_occurred(CVDEvent.X)
        assert CVDEvent.X in vuln.state.inferred_events

    def test_vendor_advisory_applies_V_with_published_date(self):
        from vulnstate import CVDArray

        arr = CVDArray()
        item = self._make_nvd_item(["Vendor Advisory"])
        arr.import_nvd([item])
        vuln = arr.get(0)
        assert vuln.has_event_occurred(CVDEvent.V)
        # V timestamp should be publishedDate (V <= P)
        v_ts = vuln.events[CVDEvent.V]
        p_ts = vuln.events[CVDEvent.P]
        assert v_ts is not None
        assert v_ts <= p_ts

    def test_third_party_advisory_applies_V_with_last_modified(self):
        from vulnstate import CVDArray

        arr = CVDArray()
        item = self._make_nvd_item(["Third Party Advisory"])
        arr.import_nvd([item])
        vuln = arr.get(0)
        assert vuln.has_event_occurred(CVDEvent.V)
        # V timestamp should be lastModifiedDate (V > P typically)
        v_ts = vuln.events[CVDEvent.V]
        assert v_ts is not None

    def test_vendor_advisory_wins_over_third_party(self):
        from vulnstate import CVDArray

        arr = CVDArray()
        item = self._make_nvd_item(["Vendor Advisory", "Third Party Advisory"])
        arr.import_nvd([item])
        vuln = arr.get(0)
        # Vendor Advisory should win — V at publishedDate
        v_ts = vuln.events[CVDEvent.V]
        p_ts = vuln.events[CVDEvent.P]
        assert v_ts == p_ts  # Both use publishedDate

    def test_infer_timestamps_false_gives_none(self):
        from vulnstate import CVDArray

        arr = CVDArray()
        item = self._make_nvd_item(["Patch", "Exploit"])
        arr.import_nvd([item], infer_timestamps=False)
        vuln = arr.get(0)
        assert vuln.has_event_occurred(CVDEvent.F)
        assert vuln.has_event_occurred(CVDEvent.X)
        assert vuln.events[CVDEvent.F] is None
        assert vuln.events[CVDEvent.X] is None

    def test_infer_vendor_false_skips_V_from_published(self):
        from vulnstate import CVDArray

        arr = CVDArray()
        # No advisory tags, just a basic CVE
        item = self._make_nvd_item([])
        arr.import_nvd([item], infer_vendor=False)
        vuln = arr.get(0)
        assert not vuln.has_event_occurred(CVDEvent.V)
        # P should still be set
        assert vuln.has_event_occurred(CVDEvent.P)

    def test_infer_vendor_true_applies_V_from_published(self):
        from vulnstate import CVDArray

        arr = CVDArray()
        item = self._make_nvd_item([])
        arr.import_nvd([item], infer_vendor=True)  # default
        vuln = arr.get(0)
        assert vuln.has_event_occurred(CVDEvent.V)
        assert CVDEvent.V in vuln.state.inferred_events

    def test_multiple_tags_all_applied(self):
        from vulnstate import CVDArray

        arr = CVDArray()
        item = self._make_nvd_item(["Patch", "Exploit", "Vendor Advisory"])
        arr.import_nvd([item])
        vuln = arr.get(0)
        assert vuln.has_event_occurred(CVDEvent.V)
        assert vuln.has_event_occurred(CVDEvent.F)
        assert vuln.has_event_occurred(CVDEvent.X)
        assert vuln.has_event_occurred(CVDEvent.P)
