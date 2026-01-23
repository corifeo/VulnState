"""Tests for NVD reference tag detection and event inference."""

from datetime import datetime

from vulnstate import CVDEvent, CVDVulnerability


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
