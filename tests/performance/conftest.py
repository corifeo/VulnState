"""
Performance test configuration.

Overrides root conftest's auto-slow marking for tests in this folder.
Only tests explicitly marked with @pytest.mark.slow should be slow.
"""

import pytest


@pytest.hookimpl(hookwrapper=True, trylast=True)
def pytest_collection_modifyitems(config, items):
    """
    Remove auto-added slow/skip markers from tests that aren't explicitly marked.

    Uses hookwrapper to run AFTER root conftest has added its markers.
    """
    # Let other hooks run first
    yield

    # Now clean up markers for tests in this folder
    for item in items:
        # Only process items in this folder
        if "tests/performance" not in str(item.fspath) and "tests\\performance" not in str(
            item.fspath
        ):
            continue

        # Check if the test class has explicit slow marker
        has_explicit_slow = False
        if hasattr(item, "cls") and item.cls is not None:
            for marker in getattr(item.cls, "pytestmark", []):
                if marker.name == "slow":
                    has_explicit_slow = True
                    break

        # If no explicit slow marker on class, remove auto-added markers
        if not has_explicit_slow:
            item.own_markers = [m for m in item.own_markers if m.name not in ("slow", "skip")]
