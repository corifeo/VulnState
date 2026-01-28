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
        lines.append(f"[bold blue]State:[/bold blue] {vuln.state_str}")
        lines.append(f"[bold blue]Description:[/bold blue] {vuln.state_description}")

        # Fix Path (VFD portion of state) and history
        fix_path_text = vuln.state_str[:3]  # First 3 chars are VFD (fix path)
        lines.append(f"[bold blue]Fix Path:[/bold blue] {fix_path_text}")
        lines.append(f"[bold blue]History:[/bold blue] {vuln.history_string}")

        # Event timestamps
        if vuln.events:
            lines.append("")
            lines.append("[bold]Event Timestamps:[/bold]")
            # Filter out events with None (unknown) timestamps, sort by timestamp
            known_events = [(e, t) for e, t in vuln.events.items() if t is not None]
            for event, ts in sorted(known_events, key=lambda x: x[1]):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                label = EVENT_LABELS[event]
                lines.append(
                    f"  [{CVDFormatter.COLOR_TIMESTAMP}]{label}"
                    f"[/{CVDFormatter.COLOR_TIMESTAMP}]: {ts_str}"
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
            transition = f"{from_state} → {to_state}"
            node.add(
                f"[{CVDFormatter.COLOR_INFO}]Transition:"
                f"[/{CVDFormatter.COLOR_INFO}] {transition}"
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
    def format_dimensions(vuln: "CVDVulnerability") -> Table:
        """
        Format fix_path and threat_state as separate dimensions.

        Args:
            vuln: CVDVulnerability object

        Returns:
            rich.Table with dimension breakdown
        """
        table = Table(title="[bold cyan]Vulnerability Dimensions[/bold cyan]", box=box.ROUNDED)

        table.add_column("Dimension", style=CVDFormatter.COLOR_INFO)
        table.add_column("State", style=CVDFormatter.COLOR_SUCCESS)
        table.add_column("Description", style=CVDFormatter.COLOR_MUTED)

        # Fix Path (VFD)
        fix_path = vuln.state_str[:3]
        fix_path_color = CVDFormatter._get_fix_path_color(fix_path)
        fix_path_desc = {
            "vfd": "NO_AWARENESS - Vendor unaware",
            "Vfd": "VENDOR_AWARE - No fix ready",
            "VFd": "FIX_READY - Fix available",
            "VFD": "REMEDIATED - Fix deployed",
        }.get(fix_path, "Unknown")

        table.add_row(
            "Fix Path (VFD)", f"[{fix_path_color}]{fix_path}[/{fix_path_color}]", fix_path_desc
        )

        # Threat State (PXA)
        threat_state = vuln.state_str[3:]
        threat_state_desc = {
            "pxa": "LATENT - No disclosure",
            "Pxa": "DISCLOSED - Publicly known",
            "PXa": "WEAPONIZED - Exploit exists",
            "PXA": "ACTIVE_THREAT - Under attack",
        }.get(threat_state, "Unknown")

        table.add_row("Threat State (PXA)", threat_state, threat_state_desc)

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
        lines.append("[bold]Fix Path (VFD Dimension):[/bold]")
        lines.append("  vfd (000): NO_AWARENESS - Vulnerability exists but vendor unaware")
        lines.append("  Vfd (001): VENDOR_AWARE - Vendor knows but no fix ready")
        lines.append("  VFd (011): FIX_READY - Fix available but not deployed")
        lines.append("  VFD (111): REMEDIATED - Fix deployed")
        lines.append("")
        lines.append("[bold]Threat State (PXA Dimension):[/bold]")
        lines.append("  pxa (000): LATENT - No public disclosure or exploitation")
        lines.append("  Pxa (001): DISCLOSED - Publicly known but no exploit")
        lines.append("  PXa (011): WEAPONIZED - Public exploit exists")
        lines.append("  PXA (111): ACTIVE_THREAT - Under active attack")

        content = "\n".join(lines)
        return Panel(content, border_style="blue")

    @staticmethod
    def _get_fix_path_color(fix_path: str) -> str:
        """Get color based on fix path progression."""
        fix_path_colors = {
            "VFD": "green",  # REMEDIATED
            "VFd": "yellow",  # FIX_READY
            "Vfd": "blue",  # VENDOR_AWARE
        }
        return fix_path_colors.get(fix_path, "red")  # NO_AWARENESS or unknown

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
    def print_dimensions(vuln: "CVDVulnerability", console: Optional[Console] = None) -> None:
        """
        Print formatted dimensions to console.

        Args:
            vuln: CVDVulnerability object
            console: rich.Console instance (creates new one if None)
        """
        if console is None:
            console = Console()

        console.print(CVDFormatter.format_dimensions(vuln))

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

    @staticmethod
    def array_summary_text(arr: "CVDArray") -> str:
        """Plain text summary of CVDArray statistics.

        Args:
            arr: CVDArray to summarize

        Returns:
            Multi-line string with statistics
        """
        n = len(arr)
        if n == 0:
            return "Empty CVDArray (0 vulnerabilities)"

        lines = ["CVDArray Summary", "=" * 50, f"Total Vulnerabilities: {n}", ""]

        # State distribution
        state_counts = arr.count_by_state()
        lines.append("State Distribution:")
        for state, count in sorted(state_counts.items()):
            pct = count / n * 100
            lines.append(f"  {state}: {count} ({pct:.1f}%)")
        lines.append("")

        # Event occurrence rates
        lines.append("Event Occurrence Rates:")
        for event_name, pct in arr.event_occurrence_rates.items():
            lines.append(f"  {event_name}: {pct:.1f}%")

        # Terminal states
        terminal_count = int(arr.terminal_mask.sum())
        lines.append(f"\nComplete (VFDPXA): {terminal_count}/{n} ({terminal_count / n * 100:.1f}%)")

        return "\n".join(lines)
