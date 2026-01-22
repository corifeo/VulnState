"""
Tests for formatting.py - Output formatting utilities

Tests CVDFormatter methods for displaying vulnerabilities, arrays, and summaries.
"""

import pytest
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from vulnstate import CVDEvent, CVDVulnerability
from vulnstate.array import CVDArray
from vulnstate.formatting import CVDFormatter


class TestFormatVulnerability:
    """Tests for format_vulnerability method."""

    def test_format_vulnerability_returns_panel(self):
        """format_vulnerability returns a rich Panel."""
        vuln = CVDVulnerability("CVE-2024-001")
        result = CVDFormatter.format_vulnerability(vuln)
        assert isinstance(result, Panel)

    def test_format_vulnerability_with_events(self):
        """format_vulnerability works with events applied."""
        vuln = CVDVulnerability("CVE-2024-001")
        vuln.apply_event(CVDEvent.V)
        result = CVDFormatter.format_vulnerability(vuln)
        assert isinstance(result, Panel)


class TestFormatHistory:
    """Tests for format_history method."""

    def test_format_history_returns_tree(self):
        """format_history returns a rich Tree."""
        vuln = CVDVulnerability("CVE-2024-001")
        vuln.apply_event(CVDEvent.V)
        result = CVDFormatter.format_history(vuln)
        assert isinstance(result, Tree)

    def test_format_history_empty(self):
        """format_history works with no events (just initial state)."""
        vuln = CVDVulnerability("CVE-2024-001")
        result = CVDFormatter.format_history(vuln)
        assert isinstance(result, Tree)


class TestFormatArraySummary:
    """Tests for format_array_summary method."""

    def test_format_array_summary_returns_table(self):
        """format_array_summary returns a rich Table."""
        arr = CVDArray.zeros(5)
        result = CVDFormatter.format_array_summary(arr)
        assert isinstance(result, Table)

    def test_format_array_summary_with_events(self):
        """format_array_summary works with events applied."""
        arr = CVDArray.zeros(10)
        arr.apply_event_batch(CVDEvent.V)
        result = CVDFormatter.format_array_summary(arr)
        assert isinstance(result, Table)


class TestFormatDesiderata:
    """Tests for format_desiderata method."""

    def test_format_desiderata_returns_table(self):
        """format_desiderata returns a rich Table."""
        vuln = CVDVulnerability("CVE-2024-001")
        vuln.apply_event(CVDEvent.V)
        result = CVDFormatter.format_desiderata(vuln)
        assert isinstance(result, Table)


class TestFormatDimensions:
    """Tests for format_dimensions method."""

    def test_format_dimensions_returns_table(self):
        """format_dimensions returns a rich Table."""
        vuln = CVDVulnerability("CVE-2024-001")
        vuln.apply_event(CVDEvent.V)
        result = CVDFormatter.format_dimensions(vuln)
        assert isinstance(result, Table)

    def test_format_dimensions_initial_state(self):
        """format_dimensions works with initial state."""
        vuln = CVDVulnerability("CVE-2024-001")
        result = CVDFormatter.format_dimensions(vuln)
        assert isinstance(result, Table)


class TestFormatStateLegend:
    """Tests for format_state_legend method."""

    def test_format_state_legend_returns_panel(self):
        """format_state_legend returns a rich Panel."""
        result = CVDFormatter.format_state_legend()
        assert isinstance(result, Panel)


class TestPrintMethods:
    """Tests for print_* methods (output to stdout)."""

    def test_print_vulnerability_no_error(self, capsys):
        """print_vulnerability executes without error."""
        vuln = CVDVulnerability("CVE-2024-001")
        CVDFormatter.print_vulnerability(vuln)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_array_summary_no_error(self, capsys):
        """print_array_summary executes without error."""
        arr = CVDArray.zeros(3)
        CVDFormatter.print_array_summary(arr)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_desiderata_no_error(self, capsys):
        """print_desiderata executes without error."""
        vuln = CVDVulnerability("CVE-2024-001")
        vuln.apply_event(CVDEvent.V)
        CVDFormatter.print_desiderata(vuln)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_dimensions_no_error(self, capsys):
        """print_dimensions executes without error."""
        vuln = CVDVulnerability("CVE-2024-001")
        vuln.apply_event(CVDEvent.V)
        CVDFormatter.print_dimensions(vuln)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_print_state_legend_no_error(self, capsys):
        """print_state_legend executes without error."""
        CVDFormatter.print_state_legend()
        captured = capsys.readouterr()
        assert len(captured.out) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
