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
                "references": {
                    "reference_data": [{"url": "https://example.com", "tags": []}]
                },
            }
        }
        tags = NVDParser.extract_reference_tags(item, "1.1")
        assert tags == set()
