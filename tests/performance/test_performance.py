"""
Performance Tests

Tests for verifying performance characteristics at scale.

Run with: uv run pytest tests/performance/ -v -m "not slow"  (fast tests only)
Run with: uv run pytest tests/performance/ -v -m slow         (slow tests only)
Run all:  uv run pytest tests/performance/ -v --run-slow
"""

import time

import pytest

from vulnstate import CVDArray, CVDEvent, CVDVulnerability

pytestmark = pytest.mark.performance


# =============================================================================
# Fast Performance Tests (< 5 seconds total)
# =============================================================================


class TestArrayOperations:
    """Performance sanity checks for array operations (should complete quickly)."""

    def test_array_creation_1k(self):
        """Create array with 1,000 vulnerabilities."""
        vulns = [CVDVulnerability(f"CVE-2024-{i:05d}") for i in range(1000)]
        arr = CVDArray(vulns)
        assert len(arr) == 1000

    def test_array_creation_10k(self):
        """Create array with 10,000 vulnerabilities."""
        vulns = [CVDVulnerability(f"CVE-2024-{i:05d}") for i in range(10000)]
        arr = CVDArray(vulns)
        assert len(arr) == 10000

    def test_vectorized_query_10k(self):
        """Query 10,000 vulnerabilities with boolean mask."""
        vulns = [CVDVulnerability(f"CVE-2024-{i:05d}") for i in range(10000)]
        for i in range(5000):
            vulns[i].apply_event(CVDEvent.V)

        arr = CVDArray(vulns)
        mask = arr.has_event_occurred(CVDEvent.V)

        assert mask.sum() == 5000

    def test_batch_event_application_10k(self):
        """Apply event to 10,000 vulnerabilities."""
        vulns = [CVDVulnerability(f"CVE-2024-{i:05d}") for i in range(10000)]
        arr = CVDArray(vulns)

        mask = arr.apply_event_batch(CVDEvent.V)

        assert mask.sum() == 10000

    def test_slicing_10k(self):
        """Slice operations on 10,000 vulnerabilities."""
        vulns = [CVDVulnerability(f"CVE-2024-{i:05d}") for i in range(10000)]
        arr = CVDArray(vulns)

        # Multiple slice operations
        subset1 = arr[:5000]
        subset2 = arr[5000:]
        subset3 = arr[::2]

        assert len(subset1) == 5000
        assert len(subset2) == 5000
        assert len(subset3) == 5000


class TestSerializationPerformance:
    """Performance tests for serialization (should complete quickly)."""

    def test_json_roundtrip_1k(self, tmp_path):
        """JSON roundtrip with 1,000 vulnerabilities."""
        vulns = [
            CVDVulnerability(f"CVE-2024-{i:05d}", cvss_score=float(i % 10))
            for i in range(1000)
        ]
        for v in vulns[:500]:
            v.apply_event(CVDEvent.V)

        arr = CVDArray(vulns)

        # Save and load
        json_file = tmp_path / "test.json"
        arr.to_json_batch(str(json_file))
        arr2 = CVDArray.from_json_batch(str(json_file))

        assert len(arr2) == 1000

    def test_dataframe_export_10k(self):
        """DataFrame export with 10,000 vulnerabilities."""
        vulns = [CVDVulnerability(f"CVE-2024-{i:05d}") for i in range(10000)]
        arr = CVDArray(vulns)

        df = arr.to_dataframe()

        assert len(df) == 10000


# =============================================================================
# Slow Performance Tests (marked with @pytest.mark.slow)
# These test at scale (100k-300k) and are skipped by default
# =============================================================================


@pytest.mark.slow
class TestLargeScaleImport:
    """Large-scale import performance tests (300k vulnerabilities)."""

    def test_array_creation_300k(self):
        """Create array with 300,000 vulnerabilities.

        Target: Complete within reasonable time for CI.
        """
        start = time.time()
        vulns = [CVDVulnerability(f"CVE-2024-{i:06d}") for i in range(300_000)]
        creation_time = time.time() - start

        start = time.time()
        arr = CVDArray(vulns)
        array_time = time.time() - start

        assert len(arr) == 300_000
        print(f"\n  Vuln creation: {creation_time:.2f}s")
        print(f"  Array creation: {array_time:.2f}s")
        print(f"  Total: {creation_time + array_time:.2f}s")

    def test_vectorized_query_300k(self):
        """Query 300,000 vulnerabilities with boolean mask."""
        vulns = [CVDVulnerability(f"CVE-2024-{i:06d}") for i in range(300_000)]
        # Pre-apply events to half
        for i in range(150_000):
            vulns[i].apply_event(CVDEvent.V)

        arr = CVDArray(vulns)

        start = time.time()
        mask = arr.has_event_occurred(CVDEvent.V)
        query_time = time.time() - start

        assert mask.sum() == 150_000
        print(f"\n  Query time: {query_time:.4f}s")

    def test_batch_event_application_300k(self):
        """Apply event to 300,000 vulnerabilities."""
        vulns = [CVDVulnerability(f"CVE-2024-{i:06d}") for i in range(300_000)]
        arr = CVDArray(vulns)

        start = time.time()
        mask = arr.apply_event_batch(CVDEvent.V)
        batch_time = time.time() - start

        assert mask.sum() == 300_000
        print(f"\n  Batch apply time: {batch_time:.2f}s")


@pytest.mark.slow
class TestLargeScaleSerialization:
    """Large-scale serialization performance tests."""

    def test_json_roundtrip_100k(self, tmp_path):
        """JSON roundtrip with 100,000 vulnerabilities."""
        vulns = [
            CVDVulnerability(f"CVE-2024-{i:06d}", cvss_score=float(i % 10))
            for i in range(100_000)
        ]
        for v in vulns[:50_000]:
            v.apply_event(CVDEvent.V)

        arr = CVDArray(vulns)

        json_file = tmp_path / "large.json"

        start = time.time()
        arr.to_json_batch(str(json_file))
        save_time = time.time() - start

        start = time.time()
        arr2 = CVDArray.from_json_batch(str(json_file))
        load_time = time.time() - start

        assert len(arr2) == 100_000
        print(f"\n  Save time: {save_time:.2f}s")
        print(f"  Load time: {load_time:.2f}s")

    def test_dataframe_export_100k(self):
        """DataFrame export with 100,000 vulnerabilities."""
        vulns = [CVDVulnerability(f"CVE-2024-{i:06d}") for i in range(100_000)]
        arr = CVDArray(vulns)

        start = time.time()
        df = arr.to_dataframe()
        export_time = time.time() - start

        assert len(df) == 100_000
        print(f"\n  DataFrame export time: {export_time:.2f}s")
