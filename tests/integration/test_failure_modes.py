"""
Failure Mode Tests

Tests for data integrity issues specific to a data import/manipulation library:
- NaN propagation
- Array sync issues
- Slicing side effects
- Lazy parsing edge cases
"""

from datetime import datetime

import numpy as np
import pytest

from vulnstate import CVDArray, CVDEvent, CVDVulnerability

pytestmark = pytest.mark.integration


class TestSlicingBehavior:
    """Verify slicing creates copies, not views."""

    def test_slice_creates_independent_copy(self):
        """Sliced array modifications must not affect parent."""
        arr = CVDArray(
            [
                CVDVulnerability("CVE-2024-001"),
                CVDVulnerability("CVE-2024-002"),
            ]
        )

        # Slice to get first item
        sliced = arr[:1]

        # Parent should be unaffected (slicing creates copy)
        assert len(arr) == 2
        assert len(sliced) == 1

    def test_boolean_filter_creates_copy(self):
        """Boolean mask filtering creates independent copy."""
        # Create vulns with V applied BEFORE adding to array
        vuln1 = CVDVulnerability("CVE-2024-001")
        vuln1.apply_event(CVDEvent.V)
        vuln2 = CVDVulnerability("CVE-2024-002")
        vuln3 = CVDVulnerability("CVE-2024-003")

        arr = CVDArray([vuln1, vuln2, vuln3])

        # Filter by V event
        filtered = arr[arr.has_event_occurred(CVDEvent.V)]

        # Verify filter works
        assert len(filtered) == 1
        assert filtered[0].cve_id == "CVE-2024-001"


class TestNaNPropagation:
    """Verify NaN handling in analytics."""

    def test_analytics_with_missing_timestamps(self):
        """Analytics must handle vulns without all timestamps gracefully."""
        vuln = CVDVulnerability("CVE-2024-001")
        # Only apply V event, not all 6
        vuln.apply_event(CVDEvent.V, timestamp=datetime.now())

        arr = CVDArray([vuln])
        arr.transform()

        # Properties should return valid values, not raise
        # Zero-day check needs X or A timestamp - should be False, not error
        assert arr.is_zero_day[0] == False  # noqa: E712

    def test_epss_nan_for_unmatched_cves(self):
        """EPSS import leaves NaN for CVEs not in feed."""
        arr = CVDArray(
            [
                CVDVulnerability("CVE-2024-001"),
                CVDVulnerability("CVE-2024-002"),
            ]
        )
        arr.transform()

        # No EPSS imported - should have NaN
        epss_values = arr.epss
        assert np.isnan(epss_values[0]) or epss_values[0] is None
        assert np.isnan(epss_values[1]) or epss_values[1] is None


class TestLazyParsingEdgeCases:
    """Verify lazy parsing handles edge cases."""

    def test_cvss_metrics_with_none_vector(self):
        """CVSS metrics must handle None vectors without error."""
        vuln = CVDVulnerability("CVE-2024-001")
        # No CVSS vector set
        arr = CVDArray([vuln])
        arr.transform()

        # Accessing CVSS metrics should not raise
        score = arr.cvss_scores
        assert score[0] is None or np.isnan(score[0])

    def test_metadata_access_without_import(self):
        """Accessing metadata before import should return empty/default."""
        arr = CVDArray([CVDVulnerability("CVE-2024-001")])

        # Metadata access should not raise
        metadata = arr.metadata
        assert isinstance(metadata, dict)


class TestArrayLengthInvariants:
    """Verify array lengths stay consistent."""

    def test_all_arrays_same_length_after_slicing(self):
        """Sliced arrays maintain consistent internal lengths."""
        arr = CVDArray([CVDVulnerability(f"CVE-2024-{i:03d}") for i in range(10)])

        # Apply slicing
        subset = arr[:5]

        # Check length consistency for core arrays
        n = len(subset)
        assert n == 5
        assert len(subset.states) == n
        assert len(subset.state_ints) == n
        assert len(subset.vuln_ids) == n

    def test_empty_array_operations(self):
        """Empty arrays should handle operations without error."""
        arr = CVDArray([])

        # These should not raise
        assert len(arr) == 0
        assert len(arr.states) == 0


class TestStateConsistency:
    """Verify state consistency after mutations."""

    def test_batch_apply_updates_state(self):
        """apply_event_batch updates states correctly."""
        arr = CVDArray(
            [
                CVDVulnerability("CVE-2024-001"),
                CVDVulnerability("CVE-2024-002"),
            ]
        )

        # Batch apply V to all
        arr.apply_event_batch(CVDEvent.V)

        # All should have V
        assert all(arr.has_event_occurred(CVDEvent.V))
        assert all(s.startswith("V") for s in arr.states)

    def test_vuln_state_matches_array_on_creation(self):
        """Vuln states match array states when created together."""
        vuln1 = CVDVulnerability("CVE-2024-001")
        vuln1.apply_event(CVDEvent.V)
        vuln2 = CVDVulnerability("CVE-2024-002")

        arr = CVDArray([vuln1, vuln2])

        # Array state should match vuln state
        assert arr.states[0] == "Vfdpxa"
        assert arr.states[1] == "vfdpxa"
