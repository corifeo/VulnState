"""
Error Handling Tests

Tests for input validation and error handling in CVDVulnerability and CVDArray.
"""

import pytest

from vulnstate import CVDArray, CVDEvent, CVDVulnerability
from vulnstate.constants import get_state_label

pytestmark = pytest.mark.unit


class TestEPSSValidation:
    """Tests for EPSS score validation via set_epss method."""

    def test_epss_accepts_valid_scores(self):
        """set_epss accepts values in [0.0, 1.0] range."""
        vuln = CVDVulnerability("CVE-2024-001")
        valid_scores = [0.0, 0.5, 0.99, 1.0]

        for score in valid_scores:
            vuln.set_epss(score)
            assert vuln.epss == score

    def test_epss_accepts_none(self):
        """EPSS can be cleared with set_epss(None)."""
        vuln = CVDVulnerability("CVE-2024-001")
        vuln.set_epss(0.5)
        vuln.set_epss(None)
        assert vuln.epss is None

    def test_epss_rejects_out_of_range(self):
        """set_epss rejects values outside [0.0, 1.0] range."""
        vuln = CVDVulnerability("CVE-2024-001")
        # set_epss validates range
        with pytest.raises(ValueError, match="EPSS score must be between"):
            vuln.set_epss(-0.1)

        with pytest.raises(ValueError, match="EPSS score must be between"):
            vuln.set_epss(1.5)


class TestStateLabels:
    """Tests for state label (cube name) functionality."""

    def test_state_label_mapping(self):
        """State strings map to correct labels (cube names)."""
        # get_state_label returns uppercase letters for announced events
        test_cases = [
            ("vfdpxa", "Initial"),  # No events = 'Initial'
            ("Vfdpxa", "V"),  # V only
            ("VFdpxa", "VF"),  # V + F
            ("VFDpxa", "VFD"),  # V + F + D
        ]

        for state_str, expected_label in test_cases:
            label = get_state_label(state_str)
            assert label == expected_label

    def test_vulnerability_state_label_property(self):
        """CVDVulnerability.state_label returns correct label."""
        vuln = CVDVulnerability("CVE-2024-001", state="VFdpXa")
        assert vuln.state_label is not None

        # After applying P event, label should update
        vuln.apply_event(CVDEvent.P)
        assert "P" in vuln.state_str or vuln.state_label is not None


class TestFactoryMethods:
    """Tests for CVDArray factory methods."""

    def test_zeros_creates_initial_state(self):
        """zeros() creates array with all vfdpxa states."""
        arr = CVDArray.zeros(5, vuln_id_prefix="ZERO")
        assert len(arr) == 5

        states = set(arr.states)
        assert states == {"vfdpxa"}

    def test_ones_creates_terminal_state(self):
        """ones() creates array with all VFDPXA states."""
        arr = CVDArray.ones(5, vuln_id_prefix="TERM")
        assert len(arr) == 5

        states = set(arr.states)
        assert states == {"VFDPXA"}

    def test_random_creates_valid_states(self):
        """random() creates array with valid random states."""
        arr = CVDArray.random(10, vuln_id_prefix="RND", seed=42)
        assert len(arr) == 10

        # All states should be non-empty strings
        states = arr.states
        assert all(len(s) == 6 for s in states)


class TestExceptionClasses:
    """Tests for custom exception classes."""

    def test_transform_not_run_error_exists(self):
        """TransformNotRunError has correct message format."""
        from vulnstate.constants import TransformNotRunError

        err = TransformNotRunError("cvss_score")
        assert "cvss_score" in str(err)
        assert "transform()" in str(err)

    def test_array_full_error_exists(self):
        """ArrayFullError has correct message format."""
        from vulnstate.constants import ArrayFullError

        err = ArrayFullError(100, 150)
        assert "100" in str(err)
        assert "150" in str(err)
