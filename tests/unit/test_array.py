"""
Comprehensive tests for CVDArray batch operations
"""

import numpy as np
import pytest

from vulnstate import CVDEvent
from vulnstate.array import CVDArray
from vulnstate.vulnerability import CVDVulnerability

pytestmark = pytest.mark.unit


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
        """Verify array preserves CVSS scores via API v2 structure."""
        vulns = [CVDVulnerability(f"V{i}", cvss_score=5.0 + i) for i in range(3)]
        arr = CVDArray(vulns)
        arr.transform()  # Required before accessing computed properties

        # CVSS scores are now accessed via computed property (API v2)
        scores = arr.cvss_scores
        assert len(scores) == 3
        assert not np.isnan(scores[0])  # Should have a score

    def test_states_stored_as_uint8(self):
        """Verify states are stored as uint8."""
        vulns = [CVDVulnerability("V1")]
        arr = CVDArray(vulns)

        assert arr.state_ints.dtype == np.uint8

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
        states = arr.states

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
        arr.transform()  # Required before accessing computed properties

        subset = arr[1:4]
        subset.transform()  # Subset also needs transform

        assert len(subset) == 3
        # CVSS scores preserved in API v2 structure (_source)
        scores = subset.cvss_scores
        assert len(scores) == 3


class TestAnalysisProperty:
    """Test analysis property auto-triggers DesiderataExtractor."""

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
        from datetime import datetime

        from vulnstate.models import CVSSScore, EPSSScore, KEVEntry

        vuln1 = CVDVulnerability("CVE-2024-001")
        vuln1.cvss_scores.append(
            CVSSScore(
                version=3.1,
                base_score=9.8,
                vector="",
                source="test",
                source_status=None,
                reserved_at=None,
                published_at=None,
                updated_at=None,
                temporal_score=None,
                environmental_score=None,
            )
        )
        vuln1.epss_scores.append(
            EPSSScore(model=4, probability=0.85, percentile=0.0, computed_at=None)
        )
        vuln1.kev_entry = KEVEntry(
            added_at=datetime.now(),
            due_date=None,
            required_action=None,
            ransomware_use=None,
            notes=None,
        )

        vuln2 = CVDVulnerability("CVE-2024-002")
        vuln2.cvss_scores.append(
            CVSSScore(
                version=3.1,
                base_score=4.3,
                vector="",
                source="test",
                source_status=None,
                reserved_at=None,
                published_at=None,
                updated_at=None,
                temporal_score=None,
                environmental_score=None,
            )
        )
        vuln2.epss_scores.append(
            EPSSScore(model=4, probability=0.12, percentile=0.0, computed_at=None)
        )
        # vuln2.kev is False by default (no kev_entry)

        arr = CVDArray([vuln1, vuln2])
        arr.transform()  # Required before accessing computed properties
        return arr

    def test_cvss_score_dtype(self, metadata_array):
        """cvss_score computed property is float32."""
        cvss_arr = metadata_array.cvss_scores
        assert cvss_arr.dtype == np.float32
        assert len(cvss_arr) == 2

    def test_epss_dtype(self, metadata_array):
        """epss computed property is float32."""
        epss_arr = metadata_array.epss
        assert epss_arr.dtype == np.float32
        assert len(epss_arr) == 2

    def test_kev_dtype(self, metadata_array):
        """kev computed property is bool."""
        kev_arr = metadata_array.kev
        assert kev_arr.dtype == np.bool_
        assert len(kev_arr) == 2

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
        assert arr.state_ints[1] != 0

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
        assert arr.state_ints[0] != 0

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
        assert arr.state_ints[0] != 0
        assert arr.state_ints[1] != 0
        assert arr.state_ints[2] != 0


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


class TestFixPathProperty:
    """Test CVDArray.fix_path property."""

    def test_fix_path_dtype(self):
        """fix_path returns uint8 array."""
        arr = CVDArray.random(100)
        assert arr.fix_path.dtype == np.uint8

    def test_fix_path_valid_values(self):
        """fix_path values are valid FixPath enum values (0, 1, 3, 7)."""
        arr = CVDArray.random(100)
        valid_values = {0, 1, 3, 7}  # NO_AWARENESS, VENDOR_AWARE, FIX_READY, REMEDIATED
        assert all(v in valid_values for v in arr.fix_path)

    def test_fix_path_length_matches_array(self):
        """fix_path array length matches CVDArray length."""
        arr = CVDArray.random(50)
        assert len(arr.fix_path) == len(arr)


class TestThreatStateProperty:
    """Test CVDArray.threat_state property."""

    def test_threat_state_dtype(self):
        """threat_state returns uint8 array."""
        arr = CVDArray.random(100)
        assert arr.threat_state.dtype == np.uint8

    def test_threat_state_valid_values(self):
        """threat_state values are 3-bit values (0-7)."""
        arr = CVDArray.random(100)
        # PXA bits can be any 3-bit value (0-7)
        # ThreatState enum covers common cases (0,1,3,4,5,7)
        # Values 2 (pXa) and 6 (pXA) are also valid states
        assert all(0 <= v <= 7 for v in arr.threat_state)

    def test_threat_state_length_matches_array(self):
        """threat_state array length matches CVDArray length."""
        arr = CVDArray.random(50)
        assert len(arr.threat_state) == len(arr)


class TestDesiderataMaskProperty:
    """Test CVDArray.desiderata_mask property (via analysis)."""

    def test_desiderata_mask_dtype(self):
        """desiderata_mask returns uint16 array."""
        arr = CVDArray.random(100)
        assert arr.analysis.desiderata_mask.dtype == np.uint16

    def test_desiderata_mask_length_matches_array(self):
        """desiderata_mask array length matches CVDArray length."""
        arr = CVDArray.random(50)
        assert len(arr.analysis.desiderata_mask) == len(arr)

    def test_desiderata_mask_max_value(self):
        """desiderata_mask values fit in 15 bits (15 event pairs)."""
        arr = CVDArray.random(100)
        # 15 pairs means max value is 2^15 - 1 = 32767
        assert all(v <= 0x7FFF for v in arr.analysis.desiderata_mask)


class TestFixedSizeSemantics:
    """Test fixed-size vs dynamic array semantics."""

    def test_zeros_creates_fixed_size_array(self):
        """Verify zeros() creates fixed-size array."""
        arr = CVDArray.zeros(100)
        assert arr.is_fixed_size is True
        assert len(arr) == 100

    def test_ones_creates_fixed_size_array(self):
        """Verify ones() creates fixed-size array."""
        arr = CVDArray.ones(50)
        assert arr.is_fixed_size is True
        assert len(arr) == 50

    def test_random_creates_fixed_size_array(self):
        """Verify random() creates fixed-size array."""
        arr = CVDArray.random(75, seed=42)
        assert arr.is_fixed_size is True
        assert len(arr) == 75

    def test_default_constructor_creates_dynamic_array(self):
        """Verify CVDArray() creates dynamic array."""
        arr = CVDArray()
        assert arr.is_fixed_size is False
        assert len(arr) == 0

    def test_constructor_with_list_creates_dynamic_array(self):
        """Verify CVDArray([...]) creates dynamic array."""
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(10)]
        arr = CVDArray(vulns)
        assert arr.is_fixed_size is False
        assert len(arr) == 10

    def test_fixed_size_flag_preserved_during_rebuild(self):
        """Verify _fixed_size flag preserved when _from_list() rebuilds array."""
        arr = CVDArray.zeros(100)
        assert arr.is_fixed_size is True

        # Trigger rebuild by directly calling _from_list
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(50)]
        arr._from_list(vulns)

        # After rebuild, flag should still be True
        assert arr.is_fixed_size is True

    def test_dynamic_flag_preserved_during_rebuild(self):
        """Verify dynamic flag preserved when _from_list() rebuilds array."""
        arr = CVDArray()
        assert arr.is_fixed_size is False

        # Trigger rebuild by directly calling _from_list
        vulns = [CVDVulnerability(f"CVE-{i}") for i in range(50)]
        arr._from_list(vulns)

        # After rebuild, flag should still be False
        assert arr.is_fixed_size is False


class TestNewAPIConsistencyProperties:
    """Test new properties added for API consistency (v0.2.0)."""

    def test_cve_ids_property_returns_array(self):
        """Verify cve_ids property returns CVE ID array."""
        v1 = CVDVulnerability("CVE-2024-001")
        v2 = CVDVulnerability("CVE-2024-002")
        v3 = CVDVulnerability("CVE-2024-003")
        arr = CVDArray([v1, v2, v3])

        cve_ids = arr.cve_ids
        assert isinstance(cve_ids, np.ndarray)
        assert len(cve_ids) == 3
        assert cve_ids[0] == "CVE-2024-001"
        assert cve_ids[1] == "CVE-2024-002"
        assert cve_ids[2] == "CVE-2024-003"

    def test_cve_ids_can_be_used_for_filtering(self):
        """Verify cve_ids can be used to filter vulnerabilities."""
        v1 = CVDVulnerability("CVE-2024-001")
        v2 = CVDVulnerability("CVE-2024-002")
        v3 = CVDVulnerability("CVE-2024-003")
        arr = CVDArray([v1, v2, v3])

        mask = arr.cve_ids == "CVE-2024-002"
        filtered = arr[mask]

        assert len(filtered) == 1
        assert filtered.get(0).cve_id == "CVE-2024-002"

    def test_events_property_returns_dict(self):
        """Verify events property returns dict of timestamp arrays."""
        v1 = CVDVulnerability("CVE-2024-001")
        v1.apply_event(CVDEvent.V)
        v1.apply_event(CVDEvent.F)

        arr = CVDArray([v1])
        arr.sync()

        events = arr.events
        assert isinstance(events, dict)
        assert CVDEvent.V in events
        assert CVDEvent.F in events
        assert isinstance(events[CVDEvent.V], np.ndarray)

    def test_events_dict_matches_exploded_timestamps(self):
        """Verify arr.events matches exploded timestamp properties."""
        v1 = CVDVulnerability("CVE-2024-001")
        v1.apply_event(CVDEvent.V)
        v2 = CVDVulnerability("CVE-2024-002")
        v2.apply_event(CVDEvent.F)

        arr = CVDArray([v1, v2])
        arr.sync()

        # Use 'is' comparison for same array object reference
        assert arr.events[CVDEvent.V] is arr.V_timestamps
        assert arr.events[CVDEvent.F] is arr.F_timestamps
        assert arr.events[CVDEvent.D] is arr.D_timestamps
        assert arr.events[CVDEvent.P] is arr.P_timestamps
        assert arr.events[CVDEvent.X] is arr.X_timestamps
        assert arr.events[CVDEvent.A] is arr.A_timestamps

    def test_is_zero_day_exploit_true_when_x_before_v(self):
        """Verify is_zero_day_exploit true when X before V."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.X, timestamp=base)
        v.apply_event(CVDEvent.V, timestamp=base + timedelta(days=5))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert arr.is_zero_day_exploit[0]

    def test_is_zero_day_attack_true_when_a_before_v(self):
        """Verify is_zero_day_attack true when A before V."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.A, timestamp=base)
        v.apply_event(CVDEvent.V, timestamp=base + timedelta(days=3))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert arr.is_zero_day_attack[0]

    def test_is_coordinated_true_when_v_before_p(self):
        """Verify is_coordinated true when V before P."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.P, timestamp=base + timedelta(days=30))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert arr.is_coordinated[0]

    def test_is_responsible_disclosure_true_when_v_f_p_ordered(self):
        """Verify is_responsible_disclosure true when V→F→P."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
        v.apply_event(CVDEvent.P, timestamp=base + timedelta(days=30))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert arr.is_responsible_disclosure[0]

    def test_has_fix_before_exploit_true_when_f_before_x(self):
        """Verify has_fix_before_exploit true when F before X."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
        v.apply_event(CVDEvent.X, timestamp=base + timedelta(days=20))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert arr.has_fix_before_exploit[0]

    def test_has_fix_before_attack_true_when_f_before_a(self):
        """Verify has_fix_before_attack true when F before A."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
        v.apply_event(CVDEvent.A, timestamp=base + timedelta(days=20))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert arr.has_fix_before_attack[0]

    def test_has_deployment_before_exploit_true_when_d_before_x(self):
        """Verify has_deployment_before_exploit true when D before X."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
        v.apply_event(CVDEvent.D, timestamp=base + timedelta(days=15))
        v.apply_event(CVDEvent.X, timestamp=base + timedelta(days=20))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert arr.has_deployment_before_exploit[0]

    def test_has_deployment_before_attack_true_when_d_before_a(self):
        """Verify has_deployment_before_attack true when D before A."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
        v.apply_event(CVDEvent.D, timestamp=base + timedelta(days=15))
        v.apply_event(CVDEvent.A, timestamp=base + timedelta(days=20))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert arr.has_deployment_before_attack[0]

    def test_is_private_attack_true_when_a_without_x(self):
        """Verify is_private_attack true when A without X."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.A, timestamp=base + timedelta(days=5))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert arr.is_private_attack[0]
        assert not arr.is_weaponized[0]

    def test_is_mass_exploitation_true_when_both_x_and_a(self):
        """Verify is_mass_exploitation true when both X and A."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.X, timestamp=base + timedelta(days=5))
        v.apply_event(CVDEvent.A, timestamp=base + timedelta(days=10))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert arr.is_mass_exploitation[0]
        assert arr.is_weaponized[0]
        assert arr.is_under_attack[0]

    def test_all_new_analytical_properties_return_boolean_arrays(self):
        """Verify all new analytical properties return boolean arrays."""
        v1 = CVDVulnerability("CVE-2024-001")
        v2 = CVDVulnerability("CVE-2024-002")
        arr = CVDArray([v1, v2])
        arr.transform()

        props = [
            "is_zero_day_exploit",
            "is_zero_day_attack",
            "is_coordinated",
            "is_responsible_disclosure",
            "has_fix_before_exploit",
            "has_fix_before_attack",
            "has_deployment_before_exploit",
            "has_deployment_before_attack",
            "is_private_attack",
            "is_mass_exploitation",
        ]

        for prop in props:
            result = getattr(arr, prop)
            assert isinstance(result, np.ndarray), f"{prop} should return ndarray"
            assert result.dtype == bool, f"{prop} should return bool array"
            assert len(result) == 2, f"{prop} should have length 2"


def test_cvss_properties():
    """Test CVSS metric properties."""
    from vulnstate import CVDArray, CVDVulnerability
    from vulnstate.models import CVSSScore

    v = CVDVulnerability("CVE-2024-001")
    # Set CVSSScore first (it contains both base_score and vector)
    v.cvss_scores.append(
        CVSSScore(
            version=3.1,
            base_score=9.8,
            vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            source="test",
            source_status=None,
            reserved_at=None,
            published_at=None,
            updated_at=None,
            temporal_score=None,
            environmental_score=None,
        )
    )

    arr = CVDArray([v])
    arr.transform()  # Required before accessing computed properties

    assert arr.cvss_scores[0] == 9.8
    assert arr.attack_vector[0] == "N"
    assert arr.attack_complexity[0] == "L"
    assert arr.privileges_required[0] == "N"
    assert arr.user_interaction[0] == "N"
    assert arr.scope[0] == "U"
    assert arr.confidentiality_impact[0] == "H"
    assert arr.integrity_impact[0] == "H"
    assert arr.availability_impact[0] == "H"


def test_enrichment_properties():
    """Test enrichment properties (epss, kev)."""
    from datetime import datetime

    from vulnstate import CVDArray, CVDVulnerability
    from vulnstate.models import EPSSScore, KEVEntry

    v = CVDVulnerability("CVE-2024-001")
    v.epss_scores.append(EPSSScore(model=4, probability=0.85, percentile=0.0, computed_at=None))
    v.kev_entry = KEVEntry(
        added_at=datetime.now(),
        due_date=None,
        required_action=None,
        ransomware_use=None,
        notes=None,
    )

    arr = CVDArray([v])
    arr.transform()  # Required before accessing computed properties

    assert arr.epss[0] == 0.85
    assert arr.kev[0]


class TestAnalyticalPropertiesEdgeCases:
    """Test edge cases for new analytical properties (Phase 11)."""

    def test_is_zero_day_exploit_false_when_v_before_x(self):
        """Verify is_zero_day_exploit False when V before X (negative case)."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.X, timestamp=base + timedelta(days=10))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.is_zero_day_exploit[0], "Should be False when V before X"

    def test_is_zero_day_exploit_false_with_nat_x(self):
        """Verify is_zero_day_exploit False when X is NaT (missing)."""
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V)
        # No X event

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.is_zero_day_exploit[0], "Should be False when X is NaT"

    def test_is_zero_day_attack_false_when_v_before_a(self):
        """Verify is_zero_day_attack False when V before A (negative case)."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.A, timestamp=base + timedelta(days=5))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.is_zero_day_attack[0], "Should be False when V before A"

    def test_is_zero_day_attack_false_with_nat_a(self):
        """Verify is_zero_day_attack False when A is NaT (missing)."""
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V)
        # No A event

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.is_zero_day_attack[0], "Should be False when A is NaT"

    def test_is_coordinated_false_when_p_before_v(self):
        """Verify is_coordinated False when P before V (negative case)."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.P, timestamp=base)
        v.apply_event(CVDEvent.V, timestamp=base + timedelta(days=5))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.is_coordinated[0], "Should be False when P before V"

    def test_is_coordinated_false_with_nat_p(self):
        """Verify is_coordinated False when P is NaT (missing)."""
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V)
        # No P event

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.is_coordinated[0], "Should be False when P is NaT"

    def test_is_responsible_disclosure_false_with_wrong_order(self):
        """Verify is_responsible_disclosure False when V→F→P ordering violated."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.P, timestamp=base + timedelta(days=10))
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=20))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.is_responsible_disclosure[0], "Should be False when P before F"

    def test_has_fix_before_exploit_false_when_x_before_f(self):
        """Verify has_fix_before_exploit False when X before F (negative case)."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.X, timestamp=base + timedelta(days=5))
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.has_fix_before_exploit[0], "Should be False when X before F"

    def test_has_fix_before_exploit_false_with_nat_x(self):
        """Verify has_fix_before_exploit False when X is NaT."""
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V)
        v.apply_event(CVDEvent.F)
        # No X event

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.has_fix_before_exploit[0], "Should be False when X is NaT"

    def test_has_fix_before_attack_false_when_a_before_f(self):
        """Verify has_fix_before_attack False when A before F (negative case)."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.A, timestamp=base + timedelta(days=5))
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.has_fix_before_attack[0], "Should be False when A before F"

    def test_has_deployment_before_exploit_false_when_x_before_d(self):
        """Verify has_deployment_before_exploit False when X before D."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=5))
        v.apply_event(CVDEvent.X, timestamp=base + timedelta(days=10))
        v.apply_event(CVDEvent.D, timestamp=base + timedelta(days=15))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.has_deployment_before_exploit[0], "Should be False when X before D"

    def test_has_deployment_before_attack_false_when_a_before_d(self):
        """Verify has_deployment_before_attack False when A before D."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=5))
        v.apply_event(CVDEvent.A, timestamp=base + timedelta(days=10))
        v.apply_event(CVDEvent.D, timestamp=base + timedelta(days=15))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.has_deployment_before_attack[0], "Should be False when A before D"

    def test_is_private_attack_false_when_x_present(self):
        """Verify is_private_attack False when X event present."""
        from datetime import datetime

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.X, timestamp=base)
        v.apply_event(CVDEvent.A, timestamp=base)

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.is_private_attack[0], "Should be False when X present"

    def test_is_mass_exploitation_false_when_only_x(self):
        """Verify is_mass_exploitation False when only X present (needs both X and A)."""
        from datetime import datetime

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.X, timestamp=base)
        # No A event

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.is_mass_exploitation[0], "Should be False when only X (needs both X and A)"

    def test_is_mass_exploitation_false_when_only_a(self):
        """Verify is_mass_exploitation False when only A present (needs both X and A)."""
        from datetime import datetime

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.A, timestamp=base)
        # No X event

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        assert not arr.is_mass_exploitation[0], "Should be False when only A (needs both X and A)"

    def test_all_analytical_properties_with_empty_array(self):
        """Verify all properties return empty arrays for empty CVDArray."""
        arr = CVDArray([])
        arr.transform()

        assert len(arr.is_zero_day_exploit) == 0
        assert len(arr.is_zero_day_attack) == 0
        assert len(arr.is_coordinated) == 0
        assert len(arr.is_responsible_disclosure) == 0
        assert len(arr.has_fix_before_exploit) == 0
        assert len(arr.has_fix_before_attack) == 0
        assert len(arr.has_deployment_before_exploit) == 0
        assert len(arr.has_deployment_before_attack) == 0
        assert len(arr.is_private_attack) == 0
        assert len(arr.is_mass_exploitation) == 0

    def test_all_analytical_properties_with_single_element_array(self):
        """Verify all properties work with single-element array."""
        v = CVDVulnerability("CVE-2024-001")
        arr = CVDArray([v])
        arr.transform()

        # All should return length 1 boolean arrays
        assert len(arr.is_zero_day_exploit) == 1
        assert len(arr.is_zero_day_attack) == 1
        assert len(arr.is_coordinated) == 1
        assert len(arr.is_responsible_disclosure) == 1
        assert len(arr.has_fix_before_exploit) == 1
        assert len(arr.has_fix_before_attack) == 1
        assert len(arr.has_deployment_before_exploit) == 1
        assert len(arr.has_deployment_before_attack) == 1
        assert len(arr.is_private_attack) == 1
        assert len(arr.is_mass_exploitation) == 1

    def test_analytical_properties_with_no_events_all_nat(self):
        """Verify all properties return False when no events have occurred (all NaT)."""
        v1 = CVDVulnerability("CVE-2024-001")
        v2 = CVDVulnerability("CVE-2024-002")
        v3 = CVDVulnerability("CVE-2024-003")
        # No events applied
        arr = CVDArray([v1, v2, v3])
        arr.transform()

        # All should be False when no events occurred
        assert not np.any(arr.is_zero_day_exploit)
        assert not np.any(arr.is_zero_day_attack)
        assert not np.any(arr.is_coordinated)
        assert not np.any(arr.is_responsible_disclosure)
        assert not np.any(arr.has_fix_before_exploit)
        assert not np.any(arr.has_fix_before_attack)
        assert not np.any(arr.has_deployment_before_exploit)
        assert not np.any(arr.has_deployment_before_attack)
        assert not np.any(arr.is_private_attack)
        assert not np.any(arr.is_mass_exploitation)


class TestPropertyConsistency:
    """Test consistency between related properties (Phase 11)."""

    def test_is_zero_day_includes_is_zero_day_exploit(self):
        """Verify is_zero_day is True if is_zero_day_exploit is True."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.X, timestamp=base)
        v.apply_event(CVDEvent.V, timestamp=base + timedelta(days=5))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        # If is_zero_day_exploit is True, is_zero_day should also be True
        if arr.is_zero_day_exploit[0]:
            assert arr.is_zero_day[0], "is_zero_day should be True when is_zero_day_exploit is True"

    def test_is_zero_day_includes_is_zero_day_attack(self):
        """Verify is_zero_day is True if is_zero_day_attack is True."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.A, timestamp=base)
        v.apply_event(CVDEvent.V, timestamp=base + timedelta(days=5))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        # If is_zero_day_attack is True, is_zero_day should also be True
        if arr.is_zero_day_attack[0]:
            assert arr.is_zero_day[0], "is_zero_day should be True when is_zero_day_attack is True"

    def test_is_mass_exploitation_requires_both_weaponized_and_under_attack(self):
        """Verify is_mass_exploitation requires both is_weaponized AND is_under_attack."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.X, timestamp=base + timedelta(days=5))
        v.apply_event(CVDEvent.A, timestamp=base + timedelta(days=10))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        # If is_mass_exploitation is True, both is_weaponized and is_under_attack must be True
        if arr.is_mass_exploitation[0]:
            assert arr.is_weaponized[0], "is_mass_exploitation requires is_weaponized"
            assert arr.is_under_attack[0], "is_mass_exploitation requires is_under_attack"

    def test_events_property_returns_same_objects_as_exploded_properties(self):
        """Verify arr.events[CVDEvent.V] is identical object to arr.V_timestamps."""
        arr = CVDArray.random(10, seed=42)

        # Events dict should return the same array objects (not copies)
        assert arr.events[CVDEvent.V] is arr.V_timestamps
        assert arr.events[CVDEvent.F] is arr.F_timestamps
        assert arr.events[CVDEvent.D] is arr.D_timestamps
        assert arr.events[CVDEvent.P] is arr.P_timestamps
        assert arr.events[CVDEvent.X] is arr.X_timestamps
        assert arr.events[CVDEvent.A] is arr.A_timestamps

    def test_is_private_attack_and_is_mass_exploitation_mutually_exclusive(self):
        """Verify is_private_attack and is_mass_exploitation are mutually exclusive."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        # Case 1: Private attack (A without X)
        v1 = CVDVulnerability("CVE-2024-001")
        v1.apply_event(CVDEvent.V, timestamp=base)
        v1.apply_event(CVDEvent.A, timestamp=base + timedelta(days=5))

        # Case 2: Mass exploitation (both X and A)
        v2 = CVDVulnerability("CVE-2024-002")
        v2.apply_event(CVDEvent.V, timestamp=base)
        v2.apply_event(CVDEvent.X, timestamp=base + timedelta(days=5))
        v2.apply_event(CVDEvent.A, timestamp=base + timedelta(days=10))

        arr = CVDArray([v1, v2])
        arr.sync()
        arr.transform()

        # v1 should be private attack but not mass exploitation
        assert arr.is_private_attack[0]
        assert not arr.is_mass_exploitation[0]

        # v2 should be mass exploitation but not private attack
        assert not arr.is_private_attack[1]
        assert arr.is_mass_exploitation[1]


class TestPairMaskAndHistoryId:
    """Test pair_mask and history_id bitmask analytics."""

    def test_pair_mask_basic(self):
        """Verify pair_mask returns uint16 array."""
        arr = CVDArray.random(10, seed=42)

        mask = arr.pair_mask
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.uint16
        assert len(mask) == 10

    def test_pair_mask_coordinated_disclosure(self):
        """Verify pair_mask bit 2 (V≺P) is set for coordinated disclosure."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        # Coordinated: V before P
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.P, timestamp=base + timedelta(days=10))

        arr = CVDArray([v])
        arr.sync()

        # Bit 2 = V≺P should be set
        assert (arr.pair_mask[0] & (1 << 2)) != 0

    def test_pair_mask_uncoordinated_disclosure(self):
        """Verify pair_mask bit 2 (V≺P) is clear for uncoordinated disclosure."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        # Uncoordinated: P before V
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.P, timestamp=base)
        v.apply_event(CVDEvent.V, timestamp=base + timedelta(days=10))

        arr = CVDArray([v])
        arr.sync()

        # Bit 2 = V≺P should be clear (P came first)
        assert (arr.pair_mask[0] & (1 << 2)) == 0

    def test_pair_mask_simultaneous_events_treated_as_not_satisfied(self):
        """Verify simultaneous events are treated as 'not satisfied' (bit clear)."""
        from datetime import datetime

        base = datetime(2024, 1, 1)

        # V and P at exact same timestamp
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.P, timestamp=base)  # Same timestamp!

        arr = CVDArray([v])
        arr.sync()

        # Bit 2 = V≺P should be clear (not strictly V < P)
        assert (arr.pair_mask[0] & (1 << 2)) == 0

    def test_pair_mask_missing_event_treated_as_not_satisfied(self):
        """Verify missing events result in bit clear."""
        from datetime import datetime

        base = datetime(2024, 1, 1)

        # Only V event, no P event
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)

        arr = CVDArray([v])
        arr.sync()

        # Bit 2 = V≺P should be clear (P not present)
        assert (arr.pair_mask[0] & (1 << 2)) == 0

    def test_pair_mask_zero_day_detection(self):
        """Verify pair_mask can detect zero-day exploit (X before V)."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        # Zero-day: X before V
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.X, timestamp=base)
        v.apply_event(CVDEvent.V, timestamp=base + timedelta(days=5))

        arr = CVDArray([v])
        arr.sync()
        arr.transform()

        # Bit 3 = V≺X should be clear (X came first)
        assert (arr.pair_mask[0] & (1 << 3)) == 0

        # is_zero_day_exploit should be True
        assert arr.is_zero_day_exploit[0]

    def test_pair_mask_caching(self):
        """Verify pair_mask is cached (same object on repeated access)."""
        arr = CVDArray.random(10, seed=42)

        mask1 = arr.pair_mask
        mask2 = arr.pair_mask

        # Should be the same object (cached)
        assert mask1 is mask2

    def test_pair_mask_dirty_after_sync(self):
        """Verify pair_mask is recomputed after sync() modifies timestamps."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)

        arr = CVDArray([v])
        arr.sync()

        # Get initial pair_mask
        mask1 = arr.pair_mask.copy()  # Copy to avoid reference comparison issues
        initial_bit = mask1[0] & (1 << 2)

        # Modify the vulnerability and sync (explicit indices needed since
        # arr[0].apply_event() doesn't automatically mark index dirty)
        arr[0].apply_event(CVDEvent.P, timestamp=base + timedelta(days=10))
        arr.sync(indices=[0])

        # V≺P bit should now be set (was not set before)
        assert initial_bit == 0  # Was not set before P was added
        assert (arr.pair_mask[0] & (1 << 2)) != 0  # Now set after sync

    def test_history_id_incomplete_history(self):
        """Verify history_id returns 255 for incomplete histories."""
        from datetime import datetime

        base = datetime(2024, 1, 1)

        # Only 3 events (incomplete)
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.F, timestamp=base)
        v.apply_event(CVDEvent.P, timestamp=base)

        arr = CVDArray([v])
        arr.sync()

        # Should be 255 (incomplete)
        assert arr.history_id[0] == 255

    def test_history_id_perfect_cvd(self):
        """Verify history_id 69 for perfect VFDPXA ordering."""
        from datetime import datetime, timedelta

        from vulnstate.constants import VALID_HISTORIES

        base = datetime(2024, 1, 1)

        # Perfect CVD: VFDPXA (index 69)
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
        v.apply_event(CVDEvent.D, timestamp=base + timedelta(days=20))
        v.apply_event(CVDEvent.P, timestamp=base + timedelta(days=30))
        v.apply_event(CVDEvent.X, timestamp=base + timedelta(days=40))
        v.apply_event(CVDEvent.A, timestamp=base + timedelta(days=50))

        arr = CVDArray([v])
        arr.sync()

        # Should be 69 (VFDPXA)
        assert arr.history_id[0] == 69
        assert VALID_HISTORIES[69] == "VFDPXA"

    def test_history_id_worst_case(self):
        """Verify history_id 0 for worst case AXPVFD ordering."""
        from datetime import datetime, timedelta

        from vulnstate.constants import VALID_HISTORIES

        base = datetime(2024, 1, 1)

        # Worst case: AXPVFD (index 0)
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.A, timestamp=base)
        v.apply_event(CVDEvent.X, timestamp=base + timedelta(days=10))
        v.apply_event(CVDEvent.P, timestamp=base + timedelta(days=20))
        v.apply_event(CVDEvent.V, timestamp=base + timedelta(days=30))
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=40))
        v.apply_event(CVDEvent.D, timestamp=base + timedelta(days=50))

        arr = CVDArray([v])
        arr.sync()

        # Should be 0 (AXPVFD)
        assert arr.history_id[0] == 0
        assert VALID_HISTORIES[0] == "AXPVFD"

    def test_history_id_caching(self):
        """Verify history_id is cached (same object on repeated access)."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        # Complete history
        v = CVDVulnerability("CVE-2024-001")
        for i, event in enumerate(
            [CVDEvent.V, CVDEvent.F, CVDEvent.D, CVDEvent.P, CVDEvent.X, CVDEvent.A]
        ):
            v.apply_event(event, timestamp=base + timedelta(days=i * 10))

        arr = CVDArray([v])
        arr.sync()

        id1 = arr.history_id
        id2 = arr.history_id

        # Should be the same object (cached)
        assert id1 is id2

    def test_history_id_dirty_after_sync(self):
        """Verify history_id is recomputed after sync() modifies timestamps."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        # Start with 5 events (incomplete)
        v = CVDVulnerability("CVE-2024-001")
        for i, event in enumerate([CVDEvent.V, CVDEvent.F, CVDEvent.D, CVDEvent.P, CVDEvent.X]):
            v.apply_event(event, timestamp=base + timedelta(days=i * 10))

        arr = CVDArray([v])
        arr.sync()

        # Should be 255 (incomplete)
        assert arr.history_id[0] == 255

        # Add the 6th event (explicit indices needed since arr[0].apply_event()
        # doesn't automatically mark index dirty)
        arr[0].apply_event(CVDEvent.A, timestamp=base + timedelta(days=50))
        arr.sync(indices=[0])

        # Now should be 69 (VFDPXA)
        assert arr.history_id[0] == 69

    def test_history_id_empty_array(self):
        """Verify history_id works with empty arrays."""
        arr = CVDArray([])

        assert len(arr.history_id) == 0
        assert arr.history_id.dtype == np.uint8

    def test_history_id_mixed_complete_incomplete(self):
        """Verify history_id handles mix of complete and incomplete histories."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        # v1: Complete (VFDPXA)
        v1 = CVDVulnerability("CVE-2024-001")
        for i, event in enumerate(
            [CVDEvent.V, CVDEvent.F, CVDEvent.D, CVDEvent.P, CVDEvent.X, CVDEvent.A]
        ):
            v1.apply_event(event, timestamp=base + timedelta(days=i * 10))

        # v2: Incomplete (only VFP)
        v2 = CVDVulnerability("CVE-2024-002")
        v2.apply_event(CVDEvent.V, timestamp=base)
        v2.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
        v2.apply_event(CVDEvent.P, timestamp=base + timedelta(days=20))

        arr = CVDArray([v1, v2])
        arr.sync()

        # v1 should be 69, v2 should be 255
        assert arr.history_id[0] == 69
        assert arr.history_id[1] == 255


class TestRecompute:
    """Test _recompute() ETL method."""

    def test_recompute_populates_analysis(self):
        """_recompute() should populate _analysis field."""
        arr = CVDArray.generate(10)
        # Access internal - _analysis should be populated after construction
        assert arr._analysis is not None
        assert len(arr._analysis.is_zero_day) == 10

    def test_recompute_populates_cvss_metrics(self):
        """_recompute() should parse CVSS vectors."""
        arr = CVDArray.generate(10)
        assert arr._cvss_metrics is not None
        assert len(arr._cvss_metrics.attack_vector) == 10

    def test_recompute_called_by_sync(self):
        """sync() should trigger _recompute()."""
        from vulnstate.constants import CVDEvent

        arr = CVDArray.generate(5)
        vuln = arr.get(0)
        vuln.apply_event(CVDEvent.P)
        arr.sync()
        # After sync, analysis should reflect the change
        assert arr._analysis is not None

    def test_analysis_property_uses_etl(self):
        """analysis property should return pre-computed _analysis."""
        arr = CVDArray.generate(10)
        # Should not trigger lazy computation
        analysis = arr.analysis
        assert analysis is arr._analysis  # Same object, not recomputed

    def test_cvss_properties_use_etl(self):
        """CVSS properties should use pre-computed _cvss_metrics."""
        arr = CVDArray.generate(10)
        av = arr.attack_vector
        assert len(av) == 10
        # Should be same array from _cvss_metrics
        assert av is arr._cvss_metrics.attack_vector


class TestApplyEventSignature:
    def test_timestamp_is_third_parameter(self):
        """timestamp should be 3rd param after index and event."""
        from datetime import datetime

        from vulnstate.constants import CVDEvent

        # Use zeros() to ensure no events have been applied
        arr = CVDArray.zeros(5)
        ts = datetime(2024, 1, 15)
        # New signature: apply_event(index, event, timestamp, mask)
        arr.apply_event(0, CVDEvent.P, ts)
        assert arr.get(0).events[CVDEvent.P] == ts

    def test_mask_is_fourth_parameter(self):
        """mask should be 4th param (keyword)."""
        import numpy as np

        from vulnstate.constants import CVDEvent

        # Use zeros() to ensure no events have been applied
        arr = CVDArray.zeros(5)
        mask = np.array([True, True, False, False, False])
        arr.apply_event(0, CVDEvent.P, mask=mask)
        # Only first two should have P applied
        assert arr.get(0).has_event_occurred(CVDEvent.P)
        assert arr.get(1).has_event_occurred(CVDEvent.P)
        assert not arr.get(2).has_event_occurred(CVDEvent.P)


# =============================================================================
# Tests for transform state tracking (API v2 Task 2)
# =============================================================================


class TestTransformStateTracking:
    """Tests for transform state tracking properties (is_transformed, stale, transformed_at)."""

    def test_is_transformed_false_initially(self):
        """is_transformed should be False before transform() is called."""
        arr = CVDArray.zeros(10)
        assert arr.is_transformed is False

    def test_is_transformed_true_after_transform(self):
        """is_transformed should be True after transform() is called."""
        arr = CVDArray.zeros(10)
        arr.transform()
        assert arr.is_transformed is True

    def test_stale_false_after_transform(self):
        """stale should be False immediately after transform()."""
        arr = CVDArray.zeros(10)
        arr.transform()
        assert arr.stale is False

    def test_stale_true_after_mutation(self):
        """stale should be True after mutation following transform()."""
        arr = CVDArray.zeros(10)
        arr.transform()
        arr.lifecycle.apply_event(CVDEvent.V, mask=np.ones(10, dtype=bool))
        assert arr.stale is True

    def test_transformed_at_none_initially(self):
        """transformed_at should be None before transform() is called."""
        arr = CVDArray.zeros(10)
        assert arr.transformed_at is None

    def test_transformed_at_set_after_transform(self):
        """transformed_at should be set to current time after transform()."""
        from datetime import datetime

        arr = CVDArray.zeros(10)
        before = datetime.now()
        arr.transform()
        after = datetime.now()
        assert arr.transformed_at is not None
        assert before <= arr.transformed_at <= after

    def test_transform_returns_self(self):
        """transform() should return self for method chaining."""
        arr = CVDArray.zeros(10)
        result = arr.transform()
        assert result is arr

    def test_transform_skips_if_not_stale_and_not_forced(self):
        """transform() should skip if already transformed, not stale, and force=False."""
        arr = CVDArray.zeros(10)
        arr.transform()
        first_ts = arr.transformed_at

        # Brief pause to ensure time difference if re-transformed
        import time

        time.sleep(0.001)

        # Transform again without force - should skip
        arr.transform()
        assert arr.transformed_at == first_ts

    def test_transform_reruns_if_forced(self):
        """transform() should re-run if force=True."""
        arr = CVDArray.zeros(10)
        arr.transform()
        first_ts = arr.transformed_at

        # Brief pause to ensure time difference
        import time

        time.sleep(0.001)

        # Transform again with force - should re-run
        arr.transform(force=True)
        assert arr.transformed_at > first_ts

    def test_transform_reruns_if_stale(self):
        """transform() should re-run if stale=True."""
        arr = CVDArray.zeros(10)
        arr.transform()
        first_ts = arr.transformed_at

        # Make stale via mutation
        arr.lifecycle.apply_event(CVDEvent.V, mask=np.ones(10, dtype=bool))
        assert arr.stale is True

        # Brief pause to ensure time difference
        import time

        time.sleep(0.001)

        # Transform again - should re-run because stale
        arr.transform()
        assert arr.transformed_at > first_ts
        assert arr.stale is False


# =============================================================================
# Tests for TransformNotRunError guard (API v2 Task 3)
# =============================================================================


class TestTransformNotRunErrorGuard:
    """Tests for TransformNotRunError guard on computed properties."""

    def test_access_cvss_scores_before_transform_raises(self):
        """cvss_scores should raise TransformNotRunError before transform()."""
        from vulnstate.constants import TransformNotRunError

        arr = CVDArray.zeros(10)
        with pytest.raises(TransformNotRunError) as exc_info:
            _ = arr.cvss_scores
        assert "cvss_scores" in str(exc_info.value)

    def test_access_cvss_scores_after_transform_works(self):
        """cvss_scores should work after transform()."""
        arr = CVDArray.zeros(10)
        arr.transform()
        scores = arr.cvss_scores
        assert len(scores) == 10

    def test_access_epss_before_transform_raises(self):
        """epss should raise TransformNotRunError before transform()."""
        from vulnstate.constants import TransformNotRunError

        arr = CVDArray.zeros(10)
        with pytest.raises(TransformNotRunError) as exc_info:
            _ = arr.epss
        assert "epss" in str(exc_info.value)

    def test_access_epss_after_transform_works(self):
        """epss should work after transform()."""
        arr = CVDArray.zeros(10)
        arr.transform()
        scores = arr.epss
        assert len(scores) == 10

    def test_access_kev_before_transform_raises(self):
        """kev should raise TransformNotRunError before transform()."""
        from vulnstate.constants import TransformNotRunError

        arr = CVDArray.zeros(10)
        with pytest.raises(TransformNotRunError) as exc_info:
            _ = arr.kev
        assert "kev" in str(exc_info.value)

    def test_access_kev_after_transform_works(self):
        """kev should work after transform()."""
        arr = CVDArray.zeros(10)
        arr.transform()
        kev = arr.kev
        assert len(kev) == 10

    def test_access_epss_percentile_before_transform_raises(self):
        """epss_percentile should raise TransformNotRunError before transform()."""
        from vulnstate.constants import TransformNotRunError

        arr = CVDArray.zeros(10)
        with pytest.raises(TransformNotRunError) as exc_info:
            _ = arr.epss_percentile
        assert "epss_percentile" in str(exc_info.value)

    def test_access_epss_percentile_after_transform_works(self):
        """epss_percentile should work after transform()."""
        arr = CVDArray.zeros(10)
        arr.transform()
        percentiles = arr.epss_percentile
        assert len(percentiles) == 10

    def test_access_cvss_score_before_transform_raises(self):
        """cvss_score should raise TransformNotRunError before transform()."""
        from vulnstate.constants import TransformNotRunError

        arr = CVDArray.zeros(10)
        with pytest.raises(TransformNotRunError) as exc_info:
            _ = arr.cvss_score
        assert "cvss_score" in str(exc_info.value)

    def test_access_cvss_score_after_transform_works(self):
        """cvss_score should work after transform()."""
        arr = CVDArray.zeros(10)
        arr.transform()
        scores = arr.cvss_score
        assert len(scores) == 10
        assert scores.dtype == np.float32

    def test_access_cvss_max_before_transform_raises(self):
        """cvss_max should raise TransformNotRunError before transform()."""
        from vulnstate.constants import TransformNotRunError

        arr = CVDArray.zeros(10)
        with pytest.raises(TransformNotRunError) as exc_info:
            _ = arr.cvss_max
        assert "cvss_max" in str(exc_info.value)

    def test_access_cvss_max_after_transform_works(self):
        """cvss_max should work after transform()."""
        arr = CVDArray.zeros(10)
        arr.transform()
        scores = arr.cvss_max
        assert len(scores) == 10
        assert scores.dtype == np.float32

    def test_access_has_exploit_before_transform_raises(self):
        """has_exploit should raise TransformNotRunError before transform()."""
        from vulnstate.constants import TransformNotRunError

        arr = CVDArray.zeros(10)
        with pytest.raises(TransformNotRunError) as exc_info:
            _ = arr.has_exploit
        assert "has_exploit" in str(exc_info.value)

    def test_access_has_exploit_after_transform_works(self):
        """has_exploit should work after transform()."""
        arr = CVDArray.zeros(10)
        arr.transform()
        result = arr.has_exploit
        assert len(result) == 10
        assert result.dtype == bool

    def test_access_is_zero_day_before_transform_raises(self):
        """is_zero_day should raise TransformNotRunError before transform()."""
        from vulnstate.constants import TransformNotRunError

        arr = CVDArray.zeros(10)
        with pytest.raises(TransformNotRunError) as exc_info:
            _ = arr.is_zero_day
        assert "is_zero_day" in str(exc_info.value)

    def test_access_is_zero_day_after_transform_works(self):
        """is_zero_day should work after transform()."""
        arr = CVDArray.zeros(10)
        arr.transform()
        result = arr.is_zero_day
        assert len(result) == 10
        assert result.dtype == bool

    def test_all_desiderata_properties_require_transform(self):
        """All 16 desiderata properties should raise TransformNotRunError before transform()."""
        from vulnstate.constants import TransformNotRunError

        arr = CVDArray.zeros(5)

        desiderata_props = [
            "is_zero_day",
            "is_fix_available",
            "is_fix_deployed",
            "is_weaponized",
            "is_under_attack",
            "is_premature_disclosure",
            "is_zero_day_exploit",
            "is_zero_day_attack",
            "is_coordinated",
            "is_responsible_disclosure",
            "has_fix_before_exploit",
            "has_fix_before_attack",
            "has_deployment_before_exploit",
            "has_deployment_before_attack",
            "is_private_attack",
            "is_mass_exploitation",
        ]

        for prop in desiderata_props:
            with pytest.raises(TransformNotRunError) as exc_info:
                _ = getattr(arr, prop)
            assert prop in str(exc_info.value), f"{prop} should mention property name in error"


# =============================================================================
# Tests for data science UX methods (API v2 Task 4)
# =============================================================================


class TestDataScienceUXMethods:
    """Tests for head(), describe(), info() methods."""

    def test_head_returns_first_n(self):
        """head(n) should return first n items as CVDArray."""
        arr = CVDArray.zeros(100)
        head = arr.head(5)
        assert len(head) == 5
        assert isinstance(head, CVDArray)

    def test_head_default_is_5(self):
        """head() with no argument should return first 5 items."""
        arr = CVDArray.zeros(100)
        head = arr.head()
        assert len(head) == 5

    def test_head_small_array(self):
        """head(n) should return all items if array has fewer than n items."""
        arr = CVDArray.zeros(3)
        head = arr.head(10)
        assert len(head) == 3

    def test_describe_returns_dict(self):
        """describe() should return a dict with summary statistics."""
        arr = CVDArray.generate(100, seed=42)
        arr.transform()
        desc = arr.describe()
        assert isinstance(desc, dict)
        assert "count" in desc
        assert "cvss_mean" in desc

    def test_describe_without_transform(self):
        """describe() should work without transform (limited stats)."""
        arr = CVDArray.zeros(50)
        desc = arr.describe()
        assert isinstance(desc, dict)
        assert "count" in desc
        assert desc["count"] == 50

    def test_info_returns_string(self):
        """info() should return a formatted string."""
        arr = CVDArray.zeros(100)
        info = arr.info()
        assert isinstance(info, str)
        assert "100" in info  # length

    def test_info_contains_transform_state(self):
        """info() should include transform state."""
        arr = CVDArray.zeros(100)
        info = arr.info()
        assert "Transformed: False" in info

        arr.transform()
        info = arr.info()
        assert "Transformed: True" in info


# =============================================================================
# Tests for import_dict (API v2 Task 5)
# =============================================================================


class TestImportDict:
    """Tests for import_dict method for dict roundtrip workflow."""

    def test_import_dict_adds_vulnerabilities(self):
        """import_dict should add vulnerabilities from list of dicts."""
        arr = CVDArray()
        arr.import_dict(
            [
                {"cve_id": "CVE-2024-1234", "state": "VFdpxa"},
                {"cve_id": "CVE-2024-5678", "state": "vfdPxa"},
            ]
        )
        assert len(arr) == 2
        assert arr.cve_ids[0] == "CVE-2024-1234"
        assert arr.cve_ids[1] == "CVE-2024-5678"

    def test_import_dict_roundtrip(self):
        """import_dict should enable full roundtrip with to_dict_batch."""
        arr = CVDArray.generate(10, seed=42)
        arr.transform()

        # Export and reimport
        exported = arr.to_dict_batch(include_computed=False)

        arr2 = CVDArray()
        arr2.import_dict(exported)
        arr2.transform()

        assert len(arr2) == len(arr)
        np.testing.assert_array_equal(arr2.cve_ids, arr.cve_ids)

    def test_import_dict_returns_self(self):
        """import_dict should return self for method chaining."""
        arr = CVDArray()
        result = arr.import_dict([{"cve_id": "CVE-2024-001"}])
        assert result is arr

    def test_import_dict_preserves_state(self):
        """import_dict should preserve vulnerability state."""
        arr = CVDArray()
        arr.import_dict(
            [
                {"cve_id": "CVE-2024-001", "state": "VFDpxa"},
            ]
        )
        assert arr.states[0] == "VFDpxa"
        assert arr.has_event_occurred(CVDEvent.V)[0]
        assert arr.has_event_occurred(CVDEvent.F)[0]
        assert arr.has_event_occurred(CVDEvent.D)[0]
        assert not arr.has_event_occurred(CVDEvent.P)[0]

    def test_import_dict_empty_list(self):
        """import_dict with empty list should not change array."""
        arr = CVDArray()
        arr.import_dict([])
        assert len(arr) == 0

    def test_import_dict_appends_to_existing(self):
        """import_dict should append to existing vulnerabilities."""
        arr = CVDArray([CVDVulnerability("CVE-2024-000")])
        assert len(arr) == 1

        arr.import_dict(
            [
                {"cve_id": "CVE-2024-001"},
                {"cve_id": "CVE-2024-002"},
            ]
        )
        assert len(arr) == 3
        assert arr.cve_ids[0] == "CVE-2024-000"
        assert arr.cve_ids[1] == "CVE-2024-001"
        assert arr.cve_ids[2] == "CVE-2024-002"

    def test_import_dict_on_error_skip(self):
        """import_dict with on_error='skip' should skip invalid records."""
        arr = CVDArray()
        # Invalid state should be skipped
        arr.import_dict(
            [
                {"cve_id": "CVE-2024-001", "state": "VFdpxa"},
                {"cve_id": "CVE-2024-002", "state": "INVALID_STATE"},  # Invalid
                {"cve_id": "CVE-2024-003", "state": "vfdpxa"},
            ],
            on_error="skip",
        )
        # Should have at least the valid ones
        assert len(arr) >= 2

    def test_import_dict_on_error_raise(self):
        """import_dict with on_error='raise' should raise on invalid records."""
        arr = CVDArray()
        with pytest.raises((ValueError, KeyError, TypeError)):
            arr.import_dict(
                [
                    {"cve_id": "CVE-2024-001", "state": "INVALID_STATE"},
                ],
                on_error="raise",
            )


class TestFilteredArrayTransformInheritance:
    """Test that filtered arrays inherit transform state."""

    def test_filtered_array_inherits_transform_state(self):
        """Filtered array should inherit is_transformed flag."""
        arr = CVDArray.generate(100, seed=42)
        arr.transform()

        # Filter to subset
        mask = arr.cvss_scores >= 7.0
        filtered = arr[mask]

        # Should inherit transform state
        assert filtered.is_transformed is True
        assert filtered.stale is False

    def test_filtered_array_copies_cache(self):
        """Filtered array should have sliced cache data."""
        arr = CVDArray.generate(100, seed=42)
        arr.transform()

        mask = arr.cvss_scores >= 7.0
        filtered = arr[mask]

        # Should be able to access computed properties without transform()
        _ = filtered.cvss_scores  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
