"""
Pytest configuration and fixtures for vulnstate.

This module defines pytest markers and configuration for running tests,
including options to skip long-running performance tests by default.
"""

import pytest


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "performance: mark test as a performance test (run with '-m performance')"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test"
    )


def pytest_collection_modifyitems(config, items):
    """
    Automatically mark tests based on their location and characteristics.

    By default, skip performance tests unless explicitly requested with:
        pytest -m performance
    """
    for item in items:
        # Mark all tests in test_integration.py as integration tests
        if "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark performance-related tests as slow
        if "performance" in item.nodeid.lower() or "large_scale" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        # Mark batch operation performance tests specifically
        if "test_batch_filtering_performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "test_batch_event_application_performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "test_ml_encoding_large_portfolio" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        if "test_filtering_performance" in item.nodeid:
            item.add_marker(pytest.mark.slow)

    # Skip slow tests by default unless explicitly requested
    if config.option.markexpr is None or "slow" not in config.option.markexpr:
        skip_slow = pytest.mark.skip(reason="use -m slow to run performance tests")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
