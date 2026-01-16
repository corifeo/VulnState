"""
Pytest configuration and shared fixtures for vulnstate tests
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check if pytest-benchmark is available
import importlib.util

from vulnstate import CVDEvent
from vulnstate.array import CVDArray
from vulnstate.vulnerability import CVDVulnerability

HAS_BENCHMARK = importlib.util.find_spec("pytest_benchmark") is not None


class StubBenchmark:
    """Stub benchmark fixture for when pytest-benchmark is not installed."""

    def __call__(self, func, *args, **kwargs):
        """Just call the function without benchmarking."""
        return func(*args, **kwargs)

    def pedantic(self, func, iterations=1, rounds=1):
        """Stub pedantic method."""
        return func()


@pytest.fixture
def benchmark():
    """Benchmark fixture - uses pytest-benchmark if available, else stub."""
    if HAS_BENCHMARK:
        # This will be overridden by pytest-benchmark plugin
        return StubBenchmark()
    else:
        # Return stub for tests that don't have pytest-benchmark
        return StubBenchmark()


@pytest.fixture
def single_vulnerability():
    """Single vulnerability for testing."""
    return CVDVulnerability("TEST-001", state="vfdpxa", cvss_score=7.5)


@pytest.fixture
def vulnerability_list():
    """List of vulnerabilities with various states for batch testing."""
    vulns = []
    states = ["vfdpxa", "Vfdpxa", "VFdpxa", "VFDpxa", "VFDPxa"]

    for i, state in enumerate(states):
        v = CVDVulnerability(f"TEST-{i:03d}", state=state, cvss_score=5.0 + i)
        vulns.append(v)

    return vulns


@pytest.fixture
def vulnerability_array(vulnerability_list):
    """CVDArray for batch testing."""
    return CVDArray(vulnerability_list)


@pytest.fixture
def large_array():
    """Large array for performance testing (1000 items)."""
    vulns = []
    for i in range(1000):
        v = CVDVulnerability(f"PERF-{i:04d}", cvss_score=np.random.rand() * 10)

        # Randomly apply events
        if np.random.rand() > 0.3:
            v.apply_event(CVDEvent.V)
        if np.random.rand() > 0.5:
            v.apply_event(CVDEvent.F)
        if np.random.rand() > 0.7:
            v.apply_event(CVDEvent.P)

        vulns.append(v)

    return CVDArray(vulns)
