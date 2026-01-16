"""
Comprehensive tests for CVDArray batch operations
"""

import numpy as np
import pytest

from vulnstate import CVDEvent
from vulnstate.array import CVDArray
from vulnstate.vulnerability import CVDVulnerability


class TestBatchCreation:
    """Test CVDArray creation."""

    def test_create_from_list(self):
        """Verify creation from vulnerability list."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(5)]
        arr = CVDArray(vulns)

        assert len(arr) == 5

    def test_create_empty(self):
        """Verify creation of empty array."""
        arr = CVDArray()

        assert len(arr) == 0

    def test_create_with_metadata(self):
        """Verify array preserves metadata."""
        vulns = [CVDVulnerability(f"V{i}", cvss_score=5.0 + i) for i in range(3)]
        arr = CVDArray(vulns)

        assert "cvss_score" in arr.metadata

    def test_states_stored_as_uint8(self):
        """Verify states are stored as uint8."""
        vulns = [CVDVulnerability("V1")]
        arr = CVDArray(vulns)

        assert arr.states.dtype == np.uint8

    def test_vuln_ids_stored_as_object(self):
        """Verify IDs are stored as objects."""
        vulns = [CVDVulnerability("CVE-2024-001")]
        arr = CVDArray(vulns)

        assert arr.vuln_ids.dtype == object


class TestBatchIndexing:
    """Test CVDArray indexing."""

    def test_single_item_indexing(self):
        """Verify single item returns CVDVulnerability."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(3)]
        arr = CVDArray(vulns)

        v = arr[1]

        assert isinstance(v, CVDVulnerability)
        # Array stores internal_id, not cve_id
        assert v.vuln_id == vulns[1].vuln_id

    def test_slice_indexing(self):
        """Verify slice indexing returns CVDArray."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(5)]
        arr = CVDArray(vulns)

        subset = arr[1:4]

        assert isinstance(subset, CVDArray)
        assert len(subset) == 3

    def test_boolean_mask_indexing(self):
        """Verify boolean mask indexing."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(5)]
        vulns[0].apply_event(CVDEvent.V)
        vulns[2].apply_event(CVDEvent.V)

        arr = CVDArray(vulns)
        mask = arr.has_event_occurred(CVDEvent.V)

        subset = arr[mask]

        assert len(subset) == 2

    def test_negative_indexing(self):
        """Verify negative indexing."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(5)]
        arr = CVDArray(vulns)

        v = arr[-1]

        # Array stores internal_id, not cve_id
        assert v.vuln_id == vulns[-1].vuln_id

    def test_index_out_of_bounds(self):
        """Verify out of bounds raises IndexError."""
        arr = CVDArray([CVDVulnerability("V0")])

        with pytest.raises(IndexError):
            arr[10]


class TestVectorizedQueries:
    """Test vectorized state queries."""

    def test_has_event_occurred_returns_mask(self):
        """Verify has_event_occurred returns boolean array."""
        vulns = [CVDVulnerability("V1"), CVDVulnerability("V2")]
        vulns[0].apply_event(CVDEvent.V)

        arr = CVDArray(vulns)
        mask = arr.has_event_occurred(CVDEvent.V)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask[0]
        assert not mask[1]

    def test_terminal_mask(self):
        """Verify terminal_mask identification."""
        vulns = [CVDVulnerability("V1"), CVDVulnerability("V2")]

        # Apply all events to first
        for event in CVDEvent:
            vulns[0].apply_event(event)

        arr = CVDArray(vulns)
        mask = arr.terminal_mask

        assert mask[0]
        assert not mask[1]

    def test_states_as_strings(self):
        """Verify state string conversion."""
        vulns = [CVDVulnerability("V1")]
        vulns[0].apply_event(CVDEvent.V)

        arr = CVDArray(vulns)
        states = arr.states_as_strings

        assert states[0] == "Vfdpxa"

    def test_count_by_state(self):
        """Verify state counting."""
        vulns = [CVDVulnerability("V1"), CVDVulnerability("V2"), CVDVulnerability("V3")]
        vulns[0].apply_event(CVDEvent.V)
        vulns[1].apply_event(CVDEvent.V)

        arr = CVDArray(vulns)
        counts = arr.count_by_state()

        assert counts["Vfdpxa"] == 2
        assert counts["vfdpxa"] == 1

    def test_event_occurrence_counts(self):
        """Verify event occurrence counting."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(5)]

        for i in range(3):
            vulns[i].apply_event(CVDEvent.V)

        arr = CVDArray(vulns)
        counts = arr.event_occurrence_counts

        assert counts["V"] == 3
        assert counts["F"] == 0

    def test_event_occurrence_rates(self):
        """Verify event occurrence rate calculation."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(4)]

        for i in range(2):
            vulns[i].apply_event(CVDEvent.V)

        arr = CVDArray(vulns)
        rates = arr.event_occurrence_rates

        assert abs(rates["V"] - 50.0) < 0.1


class TestBatchEventApplication:
    """Test batch event application."""

    def test_apply_event_batch_returns_mask(self):
        """Verify apply_event_batch returns update mask."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(3)]
        arr = CVDArray(vulns)

        mask = arr.apply_event_batch(CVDEvent.V)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool

    def test_apply_event_batch_to_all(self):
        """Verify applying event to all eligible."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(5)]
        arr = CVDArray(vulns)

        mask = arr.apply_event_batch(CVDEvent.V)

        assert mask.sum() == 5
        assert arr.has_event_occurred(CVDEvent.V).all()

    def test_apply_event_batch_respects_constraints(self):
        """Verify batch application respects V→F→D."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(3)]
        arr = CVDArray(vulns)

        # Try to apply F without V
        mask = arr.apply_event_batch(CVDEvent.F)

        # Should not apply to any
        assert mask.sum() == 0

        # Apply V first
        arr.apply_event_batch(CVDEvent.V)

        # Now F should apply
        mask = arr.apply_event_batch(CVDEvent.F)
        assert mask.sum() == 3

    def test_apply_event_batch_with_mask(self):
        """Verify batch application with custom mask."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(5)]
        arr = CVDArray(vulns)

        # Apply to only first 2
        custom_mask = np.array([True, True, False, False, False])
        arr.apply_event_batch(CVDEvent.V, mask=custom_mask)

        assert arr.has_event_occurred(CVDEvent.V)[0]
        assert arr.has_event_occurred(CVDEvent.V)[1]
        assert not arr.has_event_occurred(CVDEvent.V)[2]

    def test_apply_event_batch_sequence(self):
        """Verify applying multiple events in sequence."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(3)]
        arr = CVDArray(vulns)

        arr.apply_event_batch(CVDEvent.V)
        arr.apply_event_batch(CVDEvent.F)
        arr.apply_event_batch(CVDEvent.D)

        assert arr.has_event_occurred(CVDEvent.D).all()


class TestMLIntegration:
    """Test ML matrix conversions."""

    def test_to_matrix_shape(self):
        """Verify to_matrix output shape."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(5)]
        arr = CVDArray(vulns)

        matrix = arr.to_matrix()

        assert matrix.shape == (5, 6)
        assert matrix.dtype == np.uint8

    def test_to_matrix_values(self):
        """Verify to_matrix encodes correctly."""
        vulns = [CVDVulnerability("V1"), CVDVulnerability("V2")]
        vulns[0].apply_event(CVDEvent.V)
        vulns[0].apply_event(CVDEvent.P)
        vulns[1].apply_event(CVDEvent.F)

        arr = CVDArray(vulns)
        matrix = arr.to_matrix()

        # V1 has V and P
        assert matrix[0, 0] == 1  # V
        assert matrix[0, 3] == 1  # P
        # V2 has F but it's invalid (needs V first), so should be 0
        assert matrix[1, 1] == 0  # F not applied

    def test_from_matrix_creation(self):
        """Verify from_matrix creates valid array."""
        matrix = np.array(
            [
                [1, 1, 0, 0, 0, 0],  # VFdpxa
                [1, 0, 0, 1, 0, 0],  # VfdPxa
                [0, 0, 0, 0, 0, 0],  # vfdpxa
            ],
            dtype=np.uint8,
        )

        arr = CVDArray.from_matrix(matrix, ["A", "B", "C"])

        assert len(arr) == 3
        assert arr[0].state == "VFdpxa"
        assert arr[1].state == "VfdPxa"
        assert arr[2].state == "vfdpxa"

    def test_from_matrix_round_trip(self):
        """Verify from_matrix→to_matrix round-trip."""
        vulns = [CVDVulnerability("V1"), CVDVulnerability("V2"), CVDVulnerability("V3")]
        vulns[0].apply_event(CVDEvent.V)
        vulns[0].apply_event(CVDEvent.F)
        vulns[1].apply_event(CVDEvent.P)
        vulns[1].apply_event(CVDEvent.X)

        original = CVDArray(vulns)
        matrix = original.to_matrix()
        recovered = CVDArray.from_matrix(matrix)

        assert (original.to_matrix() == recovered.to_matrix()).all()


class TestSummary:
    """Test summary generation."""

    def test_summary_returns_string(self):
        """Verify summary returns string."""
        arr = CVDArray([CVDVulnerability("V1")])

        summary = arr.summary

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_summary_empty_array(self):
        """Verify summary for empty array."""
        arr = CVDArray()

        summary = arr.summary

        assert "Empty" in summary or "0" in summary

    def test_summary_contains_count(self):
        """Verify summary contains vulnerability count."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(5)]
        arr = CVDArray(vulns)

        summary = arr.summary

        assert "5" in summary


class TestDataPreservation:
    """Test data preservation through operations."""

    def test_indexing_preserves_timestamps(self):
        """Verify timestamps preserved when indexing."""
        from datetime import datetime

        vulns = [CVDVulnerability("V1")]
        ts = datetime(2024, 1, 1, 12, 0, 0)
        vulns[0].apply_event(CVDEvent.V, timestamp=ts)

        arr = CVDArray(vulns)
        v = arr[0]

        assert v.events[CVDEvent.V] == ts

    def test_indexing_preserves_metadata(self):
        """Verify metadata preserved when indexing."""
        vulns = [CVDVulnerability("V1", cvss_score=9.1)]
        arr = CVDArray(vulns)

        v = arr[0]

        assert v.cvss_score == 9.1

    def test_subset_preserves_properties(self):
        """Verify subset preserves array properties."""
        vulns = [CVDVulnerability(f"V{i}", cvss_score=5.0 + i) for i in range(5)]
        arr = CVDArray(vulns)

        subset = arr[1:4]

        assert len(subset) == 3
        assert "cvss_score" in subset.metadata


class TestPerformance:
    """Test performance characteristics (not benchmarks, just sanity checks)."""

    def test_large_array_creation(self):
        """Verify large arrays can be created."""
        vulns = [CVDVulnerability(f"V{i:05d}") for i in range(1000)]
        arr = CVDArray(vulns)

        assert len(arr) == 1000

    def test_large_array_query(self):
        """Verify queries work on large arrays."""
        vulns = [CVDVulnerability(f"V{i:05d}") for i in range(1000)]

        # Apply events to some
        for i in range(500):
            vulns[i].apply_event(CVDEvent.V)

        arr = CVDArray(vulns)
        mask = arr.has_event_occurred(CVDEvent.V)

        assert mask.sum() == 500

    def test_large_array_batch_update(self):
        """Verify batch updates on large arrays."""
        vulns = [CVDVulnerability(f"V{i:05d}") for i in range(1000)]
        arr = CVDArray(vulns)

        mask = arr.apply_event_batch(CVDEvent.V)

        assert mask.sum() == 1000


class TestAnalysisProperty:
    """Test analysis property auto-triggers CVDAnalyzer."""

    def test_analysis_returns_analysis_result(self):
        """Verify analysis property returns AnalysisResult."""
        from vulnstate.models import AnalysisResult

        arr = CVDArray.zeros(5)
        result = arr.analysis

        assert isinstance(result, AnalysisResult)
        assert len(result) == 5

    def test_analysis_is_cached(self):
        """Verify analysis result is cached."""
        arr = CVDArray.zeros(5)

        result1 = arr.analysis
        result2 = arr.analysis

        # Same object (cached)
        assert result1 is result2

    def test_invalidate_analysis_clears_cache(self):
        """Verify invalidate_analysis clears cache."""
        arr = CVDArray.zeros(5)

        result1 = arr.analysis
        arr.invalidate_analysis()
        result2 = arr.analysis

        # Different objects (cache was cleared)
        assert result1 is not result2


class TestExpunge:
    """Test expunge() releases vulnerability objects."""

    def test_expunge_releases_objects(self):
        """Verify expunge releases vulnerability objects."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(5)]
        arr = CVDArray(vulns)

        assert len(arr._vulnerabilities) == 5

        arr.expunge()

        assert len(arr._vulnerabilities) == 0

    def test_expunge_returns_self(self):
        """Verify expunge returns self for chaining."""
        arr = CVDArray.zeros(5)
        result = arr.expunge()

        assert result is arr


# =============================================================================
# Merged from test_data_structures.py - Array internal data structures
# =============================================================================


class TestEventTimestampStorage:
    """Tests for event timestamp storage in arrays."""

    @pytest.fixture
    def mixed_array(self):
        """Create array with vulnerabilities in different states."""
        from datetime import datetime

        vuln1 = CVDVulnerability("CVE-2024-001")
        vuln1.apply_event(CVDEvent.V, datetime(2024, 1, 15))
        vuln1.apply_event(CVDEvent.F, datetime(2024, 2, 10))
        vuln1.apply_event(CVDEvent.D, datetime(2024, 3, 1))
        vuln1.apply_event(CVDEvent.P, datetime(2024, 2, 15))

        vuln2 = CVDVulnerability("CVE-2024-002")
        vuln2.apply_event(CVDEvent.V, datetime(2024, 1, 20))

        vuln3 = CVDVulnerability("CVE-2024-003")

        return CVDArray([vuln1, vuln2, vuln3])

    def test_timestamp_arrays_have_correct_dtype(self, mixed_array):
        """Event timestamp arrays use datetime64[us]."""
        for event in CVDEvent:
            ts_array = mixed_array._event_timestamps_absolute[event]
            assert ts_array.dtype == np.dtype("datetime64[us]")

    def test_nat_for_missing_events(self, mixed_array):
        """NaT used for events that haven't occurred."""
        for event in CVDEvent:
            ts = mixed_array._event_timestamps_absolute[event][2]
            assert np.isnat(ts), f"Event {event.name} should be NaT for vuln3"

    def test_valid_timestamps_for_occurred_events(self, mixed_array):
        """Valid timestamps stored for events that occurred."""
        v_ts = mixed_array._event_timestamps_absolute[CVDEvent.V][0]
        assert not np.isnat(v_ts)
        x_ts = mixed_array._event_timestamps_absolute[CVDEvent.X][0]
        assert np.isnat(x_ts)


class TestMetadataArrays:
    """Tests for metadata array dtypes."""

    @pytest.fixture
    def metadata_array(self):
        """Create array with various metadata."""
        vuln1 = CVDVulnerability("CVE-2024-001")
        vuln1.cvss_score = 9.8
        vuln1.epss = 0.85
        vuln1.is_kev = True

        vuln2 = CVDVulnerability("CVE-2024-002")
        vuln2.cvss_score = 4.3
        vuln2.epss = 0.12
        vuln2.is_kev = False

        return CVDArray([vuln1, vuln2])

    def test_cvss_score_dtype(self, metadata_array):
        """cvss_score stored as float32."""
        cvss_arr = metadata_array._metadata_raw["cvss_score"]
        assert cvss_arr.dtype == np.float32

    def test_epss_dtype(self, metadata_array):
        """epss stored as float32."""
        epss_arr = metadata_array._metadata_raw["epss"]
        assert epss_arr.dtype == np.float32

    def test_is_kev_dtype(self, metadata_array):
        """is_kev stored as bool."""
        kev_arr = metadata_array._metadata_raw["is_kev"]
        assert kev_arr.dtype == np.bool_

    def test_cve_id_dtype(self, metadata_array):
        """CVE IDs stored as object (strings)."""
        cve_arr = metadata_array._metadata_raw["_cve_id"]
        assert cve_arr.dtype == object


# =============================================================================
# Merged from test_dirty_tracking.py - Dirty tracking mechanism
# =============================================================================


class TestDirtyTracking:
    """Tests for dirty tracking mechanism."""

    def test_dirty_indices_initialized_empty(self):
        """Test _dirty_indices is empty set on creation."""
        arr = CVDArray([])
        assert hasattr(arr, "_dirty_indices")
        assert isinstance(arr._dirty_indices, set)
        assert len(arr._dirty_indices) == 0

    def test_apply_event_marks_index_dirty(self):
        """Test apply_event marks target index as dirty."""
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(5)]
        arr = CVDArray(vulns)
        arr.apply_event(2, CVDEvent.V)
        assert 2 in arr._dirty_indices
        assert len(arr._dirty_indices) == 1

    def test_setitem_marks_index_dirty(self):
        """Test arr[idx] = vuln marks index as dirty."""
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(5)]
        arr = CVDArray(vulns)
        new_vuln = CVDVulnerability("CVE-NEW")
        arr[3] = new_vuln
        assert 3 in arr._dirty_indices

    def test_sync_clears_dirty_indices(self):
        """Test sync() clears dirty indices after updating."""
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(5)]
        arr = CVDArray(vulns)
        arr.apply_event(0, CVDEvent.V)
        arr.apply_event(2, CVDEvent.F)
        assert len(arr._dirty_indices) == 2
        arr.sync()
        assert len(arr._dirty_indices) == 0

    def test_sync_updates_cached_arrays_from_objects(self):
        """Test sync() extracts data from objects to cached arrays."""
        from datetime import datetime

        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(3)]
        arr = CVDArray(vulns)
        arr._vulnerabilities[1].apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))
        arr._dirty_indices.add(1)
        arr.sync()
        assert arr.states[1] != 0

    def test_sync_only_updates_dirty_indices(self):
        """Test sync() updates only dirty indices, not entire array."""
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(1000)]
        arr = CVDArray(vulns)
        arr.apply_event(0, CVDEvent.V)
        arr.apply_event(500, CVDEvent.F)
        arr.apply_event(999, CVDEvent.X)
        assert len(arr._dirty_indices) == 3
        assert arr._dirty_indices == {0, 500, 999}
        arr.sync()
        assert len(arr._dirty_indices) == 0


# =============================================================================
# Merged from test_exploded_timestamps.py - Exploded timestamp arrays
# =============================================================================


class TestExplodedTimestamps:
    """Tests for exploded timestamp arrays."""

    def test_timestamp_arrays_created(self):
        """Test 6 separate timestamp arrays exist."""
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(5)]
        arr = CVDArray(vulns)

        assert hasattr(arr, "V_timestamps")
        assert hasattr(arr, "F_timestamps")
        assert hasattr(arr, "D_timestamps")
        assert hasattr(arr, "P_timestamps")
        assert hasattr(arr, "X_timestamps")
        assert hasattr(arr, "A_timestamps")

        for ts_array in [
            arr.V_timestamps,
            arr.F_timestamps,
            arr.D_timestamps,
            arr.P_timestamps,
            arr.X_timestamps,
            arr.A_timestamps,
        ]:
            assert isinstance(ts_array, np.ndarray)
            assert ts_array.dtype.kind == "M"

    def test_timestamps_exploded_from_objects(self):
        """Test sync extracts timestamps to separate arrays."""
        from datetime import datetime

        import pandas as pd

        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(3)]
        arr = CVDArray(vulns)
        arr._vulnerabilities[1].apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))
        arr._vulnerabilities[1].apply_event(CVDEvent.F, timestamp=datetime(2024, 1, 15))
        arr._dirty_indices.add(1)
        arr.sync()

        assert arr.V_timestamps[1] == np.datetime64("2024-01-01")
        assert arr.F_timestamps[1] == np.datetime64("2024-01-15")
        assert pd.isna(arr.D_timestamps[1])
        assert pd.isna(arr.P_timestamps[1])

    def test_timestamp_arrays_handle_nat_correctly(self):
        """Test NaT (Not a Time) for events that haven't occurred."""
        from datetime import datetime

        import pandas as pd

        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(2)]
        arr = CVDArray(vulns)
        arr.apply_event(0, CVDEvent.V, timestamp=datetime(2024, 1, 1), sync=True)

        assert not pd.isna(arr.V_timestamps[0])
        assert pd.isna(arr.F_timestamps[0])
        assert pd.isna(arr.D_timestamps[0])
        assert pd.isna(arr.P_timestamps[0])
        assert pd.isna(arr.X_timestamps[0])
        assert pd.isna(arr.A_timestamps[0])


# =============================================================================
# Merged from test_vectorized_storage.py - Live object storage
# =============================================================================


class TestVectorizedStorage:
    """Tests for CVDArray live object storage."""

    def test_vulnerabilities_array_stores_live_objects(self):
        """Test _vulnerabilities array stores actual CVDVulnerability instances."""
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(5)]
        arr = CVDArray(vulns)

        assert hasattr(arr, "_vulnerabilities")
        assert isinstance(arr._vulnerabilities, np.ndarray)
        assert arr._vulnerabilities.dtype == object
        assert len(arr._vulnerabilities) == 5

        for i in range(5):
            assert arr._vulnerabilities[i] is vulns[i]

    def test_getitem_returns_live_object(self):
        """Test arr[idx] returns live CVDVulnerability object."""
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(3)]
        arr = CVDArray(vulns)

        vuln = arr[0]
        assert isinstance(vuln, CVDVulnerability)
        assert vuln is arr._vulnerabilities[0]


# =============================================================================
# Merged from test_fluent_api.py - Fluent API for arrays
# =============================================================================


class TestFluentAPIArray:
    """Tests for fluent API and method chaining on CVDArray."""

    def test_apply_event_returns_self(self):
        """Test apply_event returns self for chaining."""
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(5)]
        arr = CVDArray(vulns)
        result = arr.apply_event(0, CVDEvent.V)
        assert result is arr

    def test_sync_returns_self(self):
        """Test sync returns self for chaining."""
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(5)]
        arr = CVDArray(vulns)
        arr.apply_event(0, CVDEvent.V)
        result = arr.sync()
        assert result is arr

    def test_method_chaining_apply_and_sync(self):
        """Test fluent chaining: apply_event().sync()."""
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(5)]
        arr = CVDArray(vulns)
        arr.apply_event(0, CVDEvent.V).sync()
        assert len(arr._dirty_indices) == 0
        assert arr.states[0] != 0

    def test_batch_operations_with_single_sync(self):
        """Test multiple operations then single sync (efficient)."""
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(10)]
        arr = CVDArray(vulns)
        arr.apply_event(0, CVDEvent.V)
        arr.apply_event(1, CVDEvent.X)
        arr.apply_event(2, CVDEvent.P)
        assert len(arr._dirty_indices) == 3
        arr.sync()
        assert len(arr._dirty_indices) == 0
        assert arr.states[0] != 0
        assert arr.states[1] != 0
        assert arr.states[2] != 0


# =============================================================================
# Tests for get() method - Task 5.1
# =============================================================================


class TestGetMethod:
    """Tests for CVDArray.get() method for explicit live object access."""

    def test_get_returns_live_object(self):
        """get() returns live CVDVulnerability for mutation."""
        vulns = [CVDVulnerability(f"V{i}") for i in range(3)]
        arr = CVDArray(vulns)

        vuln = arr.get(1)

        assert isinstance(vuln, CVDVulnerability)
        assert vuln is arr._vulnerabilities[1]

    def test_get_with_negative_index(self):
        """get() supports negative indexing."""
        arr = CVDArray([CVDVulnerability("V0"), CVDVulnerability("V1")])

        vuln = arr.get(-1)

        assert vuln.cve_id == "V1"

    def test_get_raises_index_error(self):
        """get() raises IndexError for out of bounds."""
        arr = CVDArray([CVDVulnerability("V0")])

        with pytest.raises(IndexError):
            arr.get(10)

    def test_get_negative_out_of_bounds(self):
        """get() raises IndexError for negative out of bounds."""
        arr = CVDArray([CVDVulnerability("V0")])

        with pytest.raises(IndexError):
            arr.get(-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
