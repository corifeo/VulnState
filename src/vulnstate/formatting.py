"""
Rich Formatting - Console output formatting for CVD objects

Provides:
- CVDFormatter: Rich console output for vulnerabilities and arrays
  - format_vulnerability: Single vulnerability display
  - format_array: Array summary display
  - format_state_distribution: State breakdown

Layer: I/O
Dependencies: vulnerability.py, array.py, constants.py, states.py
Used by: External consumers
"""

from typing import TYPE_CHECKING, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree

from .constants import EVENT_LABELS

if TYPE_CHECKING:
    from .array import CVDArray
    from .vulnerability import CVDVulnerability


class CVDFormatter:
    """Rich text formatting for CVD vulnerability objects."""

    # Color scheme
    COLOR_SUCCESS = "green"
    COLOR_FAILURE = "red"
    COLOR_WARNING = "yellow"
    COLOR_INFO = "blue"
    COLOR_MUTED = "dim cyan"
    COLOR_TIMESTAMP = "cyan"

    @staticmethod
    def format_vulnerability(vuln: "CVDVulnerability", include_metadata: bool = True) -> Panel:
        """
        Format vulnerability as rich panel with colors.

        Args:
            vuln: CVDVulnerability object to format
            include_metadata: Include metadata fields in output

        Returns:
            rich.Panel with formatted vulnerability information
        """
        lines = []

        # Header with ID and state
        lines.append(f"[bold blue]Vulnerability:[/bold blue] {vuln.cve_id or vuln.vuln_id}")
        lines.append(f"[bold blue]State:[/bold blue] {vuln.state}")
        lines.append(f"[bold blue]Description:[/bold blue] {vuln.state_description}")

        # Cube (VFD portion of state) and history
        cube_text = vuln.state[:3]  # First 3 chars are VFD (fix path)
        lines.append(f"[bold blue]Cube:[/bold blue] {cube_text}")
        lines.append(f"[bold blue]History:[/bold blue] {vuln.history_string}")

        # Event timestamps
        if vuln.events:
            lines.append("")
            lines.append("[bold]Event Timestamps:[/bold]")
            # Filter out events with None (unknown) timestamps, sort by timestamp
            known_events = [(e, t) for e, t in vuln.events.items() if t is not None]
            for event, ts in sorted(known_events, key=lambda x: x[1]):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                lines.append(
                    f"  [{CVDFormatter.COLOR_TIMESTAMP}]{EVENT_LABELS[event]}[/{CVDFormatter.COLOR_TIMESTAMP}]: {ts_str}"
                )

        # Metadata
        if include_metadata and vuln.metadata:
            lines.append("")
            lines.append("[bold]Metadata:[/bold]")
            for key, value in vuln.metadata.items():
                lines.append(f"  {key}: {value}")

        content = "\n".join(lines)
        return Panel(
            content,
            title=f"[bold cyan]{vuln.cve_id or vuln.vuln_id}[/bold cyan]",
            border_style="cyan",
        )

    @staticmethod
    def format_history(vuln: "CVDVulnerability") -> Tree:
        """
        Format event history as tree with timestamps.

        Args:
            vuln: CVDVulnerability object

        Returns:
            rich.Tree with formatted history
        """
        tree = Tree(f"[bold cyan]History for {vuln.cve_id or vuln.vuln_id}[/bold cyan]")

        if not vuln.history:
            tree.add("[dim]No history recorded[/dim]")
            return tree

        for _i, entry in enumerate(vuln.history):
            event = entry.get("event")
            ts = entry.get("timestamp")
            from_state = entry.get("from_state", "?")
            to_state = entry.get("to_state", "?")
            actor = entry.get("actor", "system")
            notes = entry.get("notes", "")

            if event is None:
                # Initial state
                event_text = "[dim]Initial State[/dim]"
            else:
                event_text = f"[bold green]{event.name}[/bold green] ({EVENT_LABELS[event]})"

            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if ts else "N/A"

            node = tree.add(f"{event_text}")
            node.add(
                f"[{CVDFormatter.COLOR_TIMESTAMP}]Time:[/{CVDFormatter.COLOR_TIMESTAMP}] {ts_str}"
            )
            node.add(
                f"[{CVDFormatter.COLOR_INFO}]Transition:[/{CVDFormatter.COLOR_INFO}] {from_state} → {to_state}"
            )
            node.add(f"[{CVDFormatter.COLOR_MUTED}]Actor:[/{CVDFormatter.COLOR_MUTED}] {actor}")
            if notes:
                node.add(f"[{CVDFormatter.COLOR_MUTED}]Notes:[/{CVDFormatter.COLOR_MUTED}] {notes}")

        return tree

    @staticmethod
    def format_array_summary(arr: "CVDArray") -> Table:
        """
        Format CVDArray statistics as table.

        Args:
            arr: CVDArray object

        Returns:
            rich.Table with summary statistics
        """
        table = Table(title="[bold cyan]Vulnerability Array Summary[/bold cyan]", box=box.ROUNDED)

        table.add_column("Metric", style=CVDFormatter.COLOR_INFO)
        table.add_column("Value", style=CVDFormatter.COLOR_SUCCESS)

        # Basic stats
        table.add_row("Total Vulnerabilities", str(len(arr)))

        # State distribution
        state_counts = arr.count_by_state()
        for state, count in sorted(state_counts.items()):
            pct = (count / len(arr) * 100) if len(arr) > 0 else 0
            table.add_row(f"  State: {state}", f"{count} ({pct:.1f}%)")

        # Event occurrence
        table.add_row("", "")  # Blank line
        event_rates = arr.event_occurrence_rates
        for event_name, pct in sorted(event_rates.items()):
            table.add_row(f"  Event {event_name}", f"{pct:.1f}%")

        # Terminal states
        terminal_mask = arr.terminal_mask
        terminal_count = int(terminal_mask.sum())
        terminal_pct = (terminal_count / len(arr) * 100) if len(arr) > 0 else 0
        table.add_row("", "")  # Blank line
        table.add_row("Complete (VFDPXA)", f"{terminal_count} ({terminal_pct:.1f}%)")

        return table

    @staticmethod
    def format_desiderata(vuln: "CVDVulnerability") -> Table:
        """
        Format desiderata with color coding (green=satisfied, red=violated).

        Args:
            vuln: CVDVulnerability object

        Returns:
            rich.Table with desiderata satisfaction
        """
        from .constants import ANTI_DESIDERATA_LABELS, AntiDesiderataBit

        table = Table(title="[bold cyan]Desiderata Satisfaction[/bold cyan]", box=box.ROUNDED)

        table.add_column("Desideratum", style=CVDFormatter.COLOR_INFO)
        table.add_column("Status", style=CVDFormatter.COLOR_INFO)

        # Use analyze() to get desiderata masks
        analysis = vuln.analyze()
        anti_mask = analysis.anti_desiderata_mask[0]
        observed_mask = analysis.pair_observed_mask[0]

        satisfied_count = 0
        total_count = 0

        for bit in AntiDesiderataBit:
            label = ANTI_DESIDERATA_LABELS[bit]
            bit_mask = 1 << bit

            # Check if this pair was observed (both events occurred)
            pair_observed = (observed_mask & bit_mask) != 0
            # Check if anti-desideratum is violated
            violated = (anti_mask & bit_mask) != 0

            if not pair_observed:
                status_text = "[yellow]○ Pending[/yellow]"
            elif violated:
                status_text = "[red bold]✗ Violated[/red bold]"
            else:
                status_text = "[green bold]✓ Satisfied[/green bold]"
                satisfied_count += 1

            if pair_observed:
                total_count += 1

            table.add_row(label, status_text)

        # Summary
        table.add_row("", "")
        table.add_row(
            "[bold]Summary[/bold]", f"[green]{satisfied_count}/{total_count}[/green] observed"
        )

        return table

    @staticmethod
    def format_state_legend() -> Panel:
        """
        Format legend explaining state abbreviations and meanings.

        Returns:
            rich.Panel with formatted legend
        """
        from .constants import explain_state_char

        lines = []
        lines.append("[bold cyan]State Legend[/bold cyan]")
        lines.append("")
        lines.append("[bold]Character Codes:[/bold]")
        lines.append("  Uppercase = Event occurred")
        lines.append("  Lowercase = Event not occurred")
        lines.append("")
        lines.append("[bold]Events:[/bold]")

        for char in ["V", "F", "D", "P", "X", "A"]:
            upper_desc = explain_state_char(char)
            lower_desc = explain_state_char(char.lower())
            lines.append(f"  {char}/{char.lower()}: {upper_desc} / {lower_desc}")

        lines.append("")
        lines.append("[bold]Cube Stages:[/bold]")
        lines.append("  vfd: Vendor Unaware, No Fix, Not Deployed")
        lines.append("  Vfd: Vendor Aware, Fix Not Ready, Not Deployed")
        lines.append("  VFd: Fix Ready, Not Deployed")
        lines.append("  VFD: Fix Deployed (terminal)")

        content = "\n".join(lines)
        return Panel(content, border_style="blue")

    @staticmethod
    def _get_cube_color(cube: str) -> str:
        """Get color based on cube progression."""
        if cube == "VFD":
            return "green"
        elif cube == "VFd":
            return "yellow"
        elif cube == "Vfd":
            return "blue"
        else:  # vfd
            return "red"

    @staticmethod
    def print_vulnerability(vuln: "CVDVulnerability", console: Optional[Console] = None) -> None:
        """
        Print formatted vulnerability to console.

        Args:
            vuln: CVDVulnerability object
            console: rich.Console instance (creates new one if None)
        """
        if console is None:
            console = Console()

        console.print(CVDFormatter.format_vulnerability(vuln))

    @staticmethod
    def print_array_summary(arr: "CVDArray", console: Optional[Console] = None) -> None:
        """
        Print formatted array summary to console.

        Args:
            arr: CVDArray object
            console: rich.Console instance (creates new one if None)
        """
        if console is None:
            console = Console()

        console.print(CVDFormatter.format_array_summary(arr))

    @staticmethod
    def print_desiderata(vuln: "CVDVulnerability", console: Optional[Console] = None) -> None:
        """
        Print formatted desiderata to console.

        Args:
            vuln: CVDVulnerability object
            console: rich.Console instance (creates new one if None)
        """
        if console is None:
            console = Console()

        console.print(CVDFormatter.format_desiderata(vuln))

    @staticmethod
    def print_state_legend(console: Optional[Console] = None) -> None:
        """
        Print state legend to console.

        Args:
            console: rich.Console instance (creates new one if None)
        """
        if console is None:
            console = Console()

        console.print(CVDFormatter.format_state_legend())
