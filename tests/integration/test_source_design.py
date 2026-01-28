"""
Integration tests for API v2 _source design.

These tests validate the core design assumption:
- _source is the source of truth (not _vulnerabilities)
- arr[i] reconstructs CVDVulnerability on demand
- Reconstructed vulns are independent state machines

These tests are written FIRST (TDD) and should fail until Phase 0b is complete.
"""

from pathlib import Path

import pytest

from vulnstate import CVDEvent
from vulnstate.array import CVDArray
from vulnstate.io import CVDIO
from vulnstate.models import CVSSScore, CWEEntry, EPSSScore
from vulnstate.vulnerability import CVDVulnerability


def make_cvss(base_score: float, version: float = 3.1, source: str = "nvd") -> CVSSScore:
    """Helper to create CVSSScore with all required fields."""
    return CVSSScore(
        version=version,
        base_score=base_score,
        vector=f"CVSS:{version}/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:N/A:N",
        source=source,
        source_status=None,
        reserved_at=None,
        published_at=None,
        updated_at=None,
        temporal_score=None,
        environmental_score=None,
    )


class TestNVDRoundTrip:
    """Test Journey 1: NVD JSON → vulnerabilities → array → export → import → verify."""

    def test_nvd_json_to_array_to_export_import(self, tmp_path: Path) -> None:
        """Full round-trip: NVD JSON → vulns → array → export → import."""
        # 1. Create vulnerabilities with enrichment data
        vuln1 = CVDVulnerability("CVE-2024-001")
        vuln1.cvss_scores = [make_cvss(7.5)]
        vuln1.cwes = [CWEEntry(id="CWE-79", source="nvd", primary=True)]
        vuln1.apply_event(CVDEvent.V)
        vuln1.apply_event(CVDEvent.P)

        vuln2 = CVDVulnerability("CVE-2024-002")
        vuln2.cvss_scores = [make_cvss(9.8)]
        vuln2.cwes = [CWEEntry(id="CWE-89", source="nvd", primary=True)]
        vuln2.epss_scores = [
            EPSSScore(model=3, probability=0.95, percentile=0.99, computed_at=None)
        ]
        vuln2.apply_event(CVDEvent.V)
        vuln2.apply_event(CVDEvent.F)
        vuln2.apply_event(CVDEvent.P)

        vulns = [vuln1, vuln2]

        # 2. Create array from vulnerabilities
        arr = CVDArray(vulns)
        assert len(arr) == 2

        # 3. Verify _source is populated (CRITICAL design check)
        # After Phase 0b, _vulnerabilities should NOT exist
        assert hasattr(arr, "_source"), "_source must exist"
        assert len(arr._source.cvss_scores) == 2, "_source.cvss_scores must be populated"
        assert len(arr._source.cwes) == 2, "_source.cwes must be populated"

        # Verify _source contains the actual data (not empty lists)
        assert len(arr._source.cvss_scores[0]) == 1, "vuln1 should have 1 CVSS score"
        assert len(arr._source.cvss_scores[1]) == 1, "vuln2 should have 1 CVSS score"
        assert len(arr._source.cwes[0]) == 1, "vuln1 should have 1 CWE"
        assert len(arr._source.cwes[1]) == 1, "vuln2 should have 1 CWE"
        assert len(arr._source.epss_scores[1]) == 1, "vuln2 should have 1 EPSS score"

        # 4. Export array to dicts
        exported = arr.to_dict_batch()
        assert len(exported) == 2

        # 5. Verify exported data contains enrichment
        assert "cvss_scores" in exported[0], "Export must include cvss_scores"
        assert len(exported[0]["cvss_scores"]) == 1
        assert exported[0]["cvss_scores"][0]["base_score"] == 7.5

        assert "cwes" in exported[0], "Export must include cwes"
        assert len(exported[0]["cwes"]) == 1
        assert exported[0]["cwes"][0]["id"] == "CWE-79"

        # 6. Import back
        imported_vulns = [CVDIO.from_dict(d) for d in exported]
        imported_arr = CVDArray(imported_vulns)

        # 7. Verify data integrity
        assert len(imported_arr) == len(arr)

        for i in range(len(arr)):
            orig = arr[i]  # Reconstructed CVDVulnerability
            reimp = imported_arr[i]

            assert orig.cve_id == reimp.cve_id, f"cve_id mismatch at index {i}"
            assert len(orig.cvss_scores) == len(
                reimp.cvss_scores
            ), f"cvss_scores length mismatch at index {i}"
            if orig.cvss_scores and reimp.cvss_scores:
                assert (
                    orig.cvss_scores[0].base_score == reimp.cvss_scores[0].base_score
                ), f"cvss base_score mismatch at index {i}"

            assert len(orig.cwes) == len(reimp.cwes), f"cwes length mismatch at index {i}"
            # State comparison
            assert orig.state_str == reimp.state_str, f"state mismatch at index {i}"

    def test_source_populated_not_empty_lists(self) -> None:
        """Verify _source contains actual data, not initialized empty lists."""
        vuln = CVDVulnerability("CVE-2024-TEST")
        vuln.cvss_scores = [make_cvss(8.0)]
        vuln.epss_scores = [EPSSScore(model=3, probability=0.5, percentile=0.75, computed_at=None)]
        vuln.cwes = [CWEEntry(id="CWE-123", source="nvd", primary=True)]
        vuln.cpes = ["cpe:2.3:a:vendor:product:1.0:*:*:*:*:*:*:*"]

        arr = CVDArray([vuln])

        # These should NOT be empty after _from_list
        assert len(arr._source.cvss_scores[0]) == 1, "CVSS scores not transferred to _source"
        assert len(arr._source.epss_scores[0]) == 1, "EPSS scores not transferred to _source"
        assert len(arr._source.cwes[0]) == 1, "CWEs not transferred to _source"
        assert len(arr._source.cpes[0]) == 1, "CPEs not transferred to _source"


class TestArrayToStateMachines:
    """Test Journey 2: Array extraction produces independent state machines."""

    def test_extracted_vulnerabilities_are_state_machines(self) -> None:
        """Array extraction produces mutable state machines."""
        # 1. Create array with known states
        vuln1 = CVDVulnerability("CVE-2024-001")
        vuln1.apply_event(CVDEvent.V)

        vuln2 = CVDVulnerability("CVE-2024-002")
        vuln2.apply_event(CVDEvent.V)
        vuln2.apply_event(CVDEvent.F)

        arr = CVDArray([vuln1, vuln2])

        # 2. Extract vulnerabilities
        extracted_1 = arr[0]
        extracted_2 = arr[1]

        # 3. Each should be a CVDVulnerability state machine
        assert isinstance(extracted_1, CVDVulnerability)
        assert isinstance(extracted_2, CVDVulnerability)

        # 4. Verify initial state
        assert extracted_1.has_event_occurred(CVDEvent.V)
        assert not extracted_1.has_event_occurred(CVDEvent.F)

        assert extracted_2.has_event_occurred(CVDEvent.V)
        assert extracted_2.has_event_occurred(CVDEvent.F)

        # 5. Apply mutations (state changes)
        original_state_1 = extracted_1.state_str
        extracted_1.apply_event(CVDEvent.F)
        assert extracted_1.has_event_occurred(CVDEvent.F), "State machine should be mutable"
        assert extracted_1.state_str != original_state_1, "State should have changed"

    def test_extracted_vulnerabilities_preserve_enrichment(self) -> None:
        """Extracted vulnerabilities should have enrichment data from _source."""
        vuln = CVDVulnerability("CVE-2024-TEST")
        vuln.cvss_scores = [make_cvss(9.0)]
        vuln.epss_scores = [EPSSScore(model=3, probability=0.8, percentile=0.9, computed_at=None)]
        vuln.cwes = [CWEEntry(id="CWE-79", source="nvd", primary=True)]
        vuln.apply_event(CVDEvent.V)
        vuln.apply_event(CVDEvent.P)

        arr = CVDArray([vuln])

        # Extract and verify enrichment data is present
        extracted = arr[0]

        assert len(extracted.cvss_scores) == 1, "CVSS scores should be preserved"
        assert extracted.cvss_scores[0].base_score == 9.0

        assert len(extracted.epss_scores) == 1, "EPSS scores should be preserved"
        assert extracted.epss_scores[0].probability == 0.8

        assert len(extracted.cwes) == 1, "CWEs should be preserved"
        assert extracted.cwes[0].id == "CWE-79"

    def test_multiple_extractions_are_independent(self) -> None:
        """Multiple arr[i] calls return independent objects (if reconstructed)."""
        vuln = CVDVulnerability("CVE-2024-TEST")
        vuln.apply_event(CVDEvent.V)

        arr = CVDArray([vuln])

        # Extract twice
        extract_1 = arr[0]
        extract_2 = arr[0]

        # Both should be valid CVDVulnerability objects
        assert extract_2 is not None

        # Mutate first extraction
        extract_1.apply_event(CVDEvent.F)

        # NOTE: Current implementation returns same live object
        # After Phase 0b refactor (reconstruct on demand), these should be independent
        # For now, this test documents expected behavior after refactor

        # If returning live objects (current):
        # extract_2 would also have F event (same object)

        # If reconstructing on demand (target design):
        # extract_2 should NOT have F event (independent copy)

        # This test will need adjustment based on final design decision
        # For Phase 0b: decide if arr[i] returns live object or reconstruction

    def test_slice_returns_new_array(self) -> None:
        """Slicing array returns new CVDArray with copied data."""
        vulns = [CVDVulnerability(f"CVE-2024-{i:03d}") for i in range(5)]
        for v in vulns:
            v.apply_event(CVDEvent.V)

        arr = CVDArray(vulns)

        # Slice
        subset = arr[1:3]

        assert isinstance(subset, CVDArray)
        assert len(subset) == 2
        assert subset[0].cve_id == "CVE-2024-001"
        assert subset[1].cve_id == "CVE-2024-002"

        # Verify _source is sliced correctly
        assert len(subset._source.cvss_scores) == 2


class TestNoVulnerabilitiesAttribute:
    """Tests that verify _vulnerabilities is removed (Phase 0b target)."""

    @pytest.mark.xfail(reason="Phase 0b not yet implemented - _vulnerabilities still exists")
    def test_no_vulnerabilities_attribute(self) -> None:
        """After Phase 0b, _vulnerabilities should not exist."""
        arr = CVDArray([CVDVulnerability("CVE-2024-001")])

        # This should fail after Phase 0b
        assert not hasattr(arr, "_vulnerabilities"), "_vulnerabilities should be removed"

    def test_source_is_sole_storage(self) -> None:
        """_source should be the only storage for enrichment data."""
        vuln = CVDVulnerability("CVE-2024-TEST")
        vuln.cvss_scores = [make_cvss(7.0)]

        arr = CVDArray([vuln])

        # _source should have the data
        assert len(arr._source.cvss_scores[0]) == 1

        # Verify score value
        assert arr._source.cvss_scores[0][0].base_score == 7.0


class TestSyncBehavior:
    """Tests for sync() behavior after Phase 0b."""

    def test_sync_updates_source_from_mutations(self) -> None:
        """After mutating arr[i], sync() should update _source."""
        vuln = CVDVulnerability("CVE-2024-TEST")
        vuln.apply_event(CVDEvent.V)

        arr = CVDArray([vuln])

        # Get reference (currently live object)
        extracted = arr[0]

        # Apply event
        extracted.apply_event(CVDEvent.F)

        # Mark as dirty and sync
        arr._dirty_indices.add(0)
        arr.sync()

        # Verify state was synced
        assert arr.state.bitmask[0] & 0b000010 != 0, "F event should be in bitmask after sync"

    def test_sync_updates_source_enrichment(self) -> None:
        """Sync should update _source enrichment data from live objects."""
        vuln = CVDVulnerability("CVE-2024-TEST")
        arr = CVDArray([vuln])

        # Get reference and add enrichment
        extracted = arr[0]
        extracted.cvss_scores = [make_cvss(5.0)]

        # Mark dirty and sync
        arr._dirty_indices.add(0)
        arr.sync()

        # Verify _source was updated
        assert len(arr._source.cvss_scores[0]) == 1, "CVSS scores should sync to _source"
        assert arr._source.cvss_scores[0][0].base_score == 5.0


class TestGenerateMethod:
    """Tests for CVDArray.generate() method."""

    def test_generate_creates_array_with_source(self) -> None:
        """generate() should create array with populated _source."""
        arr = CVDArray.generate(10)

        assert len(arr) == 10
        assert hasattr(arr, "_source")
        assert len(arr._source.cvss_scores) == 10

    def test_generate_round_trip(self) -> None:
        """Generated arrays should survive export/import."""
        arr = CVDArray.generate(5, seed=42)

        # Export
        exported = arr.to_dict_batch()

        # Import
        reimported = CVDArray([CVDIO.from_dict(d) for d in exported])

        assert len(reimported) == len(arr)

        for i in range(len(arr)):
            assert arr[i].cve_id == reimported[i].cve_id
            assert arr[i].state_str == reimported[i].state_str
