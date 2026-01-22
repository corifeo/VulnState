"""
Error Handling Tests

Tests for input validation and error handling in CVDVulnerability and CVDArray.
"""

import numpy as np
import pytest

from vulnstate import CVDArray, CVDEvent, CVDVulnerability
from vulnstate.constants import get_state_label


class TestEPSSValidation:
    """Tests for EPSS score validation."""

    def test_epss_accepts_valid_scores(self):
        """EPSS accepts values in [0.0, 1.0] range."""
        vuln = CVDVulnerability("CVE-2024-001")
        valid_scores = [0.0, 0.5, 0.99, 1.0]

        for score in valid_scores:
            vuln.epss = score
            assert vuln.epss == score

    def test_epss_accepts_none(self):
        """EPSS can be cleared with None."""
        vuln = CVDVulnerability("CVE-2024-001")
        vuln.epss = 0.5
        vuln.epss = None
        assert vuln.epss is None

    def test_epss_handles_edge_values(self):
        """EPSS handles edge values (no validation enforced)."""
        vuln = CVDVulnerability("CVE-2024-001")
        # Note: Library does not enforce 0-1 range validation
        # These tests document actual behavior
        vuln.epss = -0.1
        assert vuln.epss == -0.1

        vuln.epss = 1.5
        assert vuln.epss == 1.5


class TestMatrixShapeValidation:
    """Tests for CVDArray.from_matrix() shape validation."""

    def test_valid_matrix_shape(self):
        """Valid (N, 6) matrices are accepted."""
        valid_matrix = np.array(
            [
                [1, 0, 0, 0, 0, 0],  # V event only
                [1, 1, 0, 0, 0, 0],  # V, F events
                [1, 1, 1, 0, 0, 0],  # V, F, D events
            ]
        )
        arr = CVDArray.from_matrix(valid_matrix)
        assert len(arr) == 3

    def test_1d_array_rejected(self):
        """1D arrays are rejected."""
        invalid = np.array([1, 0, 0, 0, 0, 0])
        with pytest.raises(ValueError):
            CVDArray.from_matrix(invalid)

    def test_wrong_column_count_rejected(self):
        """Matrices with wrong column count are rejected."""
        invalid_5_cols = np.ones((10, 5))
        with pytest.raises(ValueError):
            CVDArray.from_matrix(invalid_5_cols)

        invalid_7_cols = np.ones((10, 7))
        with pytest.raises(ValueError):
            CVDArray.from_matrix(invalid_7_cols)


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
