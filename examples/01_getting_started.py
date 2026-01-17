"""
Getting Started: CVD State Machine Fundamentals

Learn the basics of the Coordinated Vulnerability Disclosure (CVD) state machine.
Based on the SEI/CMU 2021-SR-021 CVD model.

Topics:
- CVDVulnerability: single vulnerability with state and metadata
- CVDEvent: the 6 CVD events (V, F, D, P, X, A)
- CVDArray: vectorized batch container
- State transitions and constraints
- Rich formatting output
"""

import random
from datetime import datetime, timedelta

from rich.console import Console
from rich.panel import Panel

from vulnstate import CVDArray, CVDEvent, CVDVulnerability
from vulnstate.formatting import CVDFormatter

console = Console()


def section_1_introduction():
    """Section 1: CVD model overview."""
    console.print(
        Panel.fit(
            "[bold cyan]Getting Started with CVD State Machine[/bold cyan]", border_style="cyan"
        )
    )
    console.print()

    console.print("[bold magenta]Key Classes:[/bold magenta]")
    console.print(
        "  [magenta]CVDVulnerability[/magenta] - Single vulnerability with state and metadata"
    )
    console.print(
        "  [magenta]CVDArray[/magenta]         - Vectorized container for batch operations"
    )
    console.print("  [magenta]CVDEvent[/magenta]         - Enum of 6 CVD events (V, F, D, P, X, A)")
    console.print()

    console.print("[bold]The 6 CVD Events:[/bold]")
    console.print("  [blue]V[/blue] - Vendor Awareness    Vendor knows vulnerability exists")
    console.print("  [blue]F[/blue] - Fix Ready           Vendor has created deployable fix")
    console.print("  [blue]D[/blue] - Fix Deployed        Fix installed on vulnerable systems")
    console.print("  [blue]P[/blue] - Public Awareness    Vulnerability publicly known")
    console.print("  [blue]X[/blue] - Exploit Public      Exploit code publicly available")
    console.print("  [blue]A[/blue] - Attacks Observed    Exploitation observed in the wild")
    console.print()
    console.print("[bold]Constraint:[/bold] V -> F -> D (vendor must know before fixing)")
    console.print()


def section_2_single_vulnerability():
    """Section 2: Working with a single vulnerability."""
    console.print(
        Panel.fit("[bold cyan]Section 2: Single Vulnerability[/bold cyan]", border_style="cyan")
    )

    # Create
    console.print("[yellow]Creating a vulnerability...[/yellow]")
    console.print("[dim]>>> vuln = CVDVulnerability('CVE-2024-12345', cvss_score=8.9)[/dim]")
    vuln = CVDVulnerability(
        "CVE-2024-12345", vendor="Example Corp", cvss_score=8.9, severity="high"
    )
    console.print(f"  State: {vuln.state} (initial - no events)")
    console.print()

    # Apply events
    console.print("[yellow]Applying events...[/yellow]")
    console.print("[magenta]Method: vuln.apply_event(CVDEvent, timestamp) -> bool[/magenta]")
    base = datetime(2024, 1, 1)

    vuln.apply_event(CVDEvent.V, timestamp=base)
    console.print(f"  V: {vuln.state}")

    vuln.apply_event(CVDEvent.F, timestamp=base + timedelta(days=14))
    console.print(f"  F: {vuln.state}")

    vuln.apply_event(CVDEvent.D, timestamp=base + timedelta(days=30))
    console.print(f"  D: {vuln.state}")
    console.print()

    # Show formatted output
    console.print("[magenta]Helper: CVDFormatter.format_vulnerability(vuln) -> Panel[/magenta]")
    CVDFormatter.print_vulnerability(vuln)
    console.print()

    # Second example: incomplete fix cube (VFd - fix ready but not deployed)
    console.print("[yellow]Creating vulnerability with incomplete fix...[/yellow]")
    vuln2 = CVDVulnerability("CVE-2024-99999", vendor="Acme Inc", cvss_score=7.2, severity="high")
    vuln2.apply_event(CVDEvent.V, timestamp=base)
    vuln2.apply_event(CVDEvent.F, timestamp=base + timedelta(days=21))
    vuln2.apply_event(CVDEvent.P, timestamp=base + timedelta(days=7))
    console.print(f"  State: {vuln2.state} (fix ready, not deployed, publicly known)")
    CVDFormatter.print_vulnerability(vuln2)
    console.print()

    # Third example: zero-day scenario (attacks before fix exists)
    console.print("[yellow]Creating zero-day vulnerability...[/yellow]")
    vuln3 = CVDVulnerability(
        "CVE-2024-00001", vendor="BigCorp", cvss_score=9.8, severity="critical"
    )
    # Exploit discovered in the wild, attacks observed, then vendor notified
    vuln3.apply_event(CVDEvent.X, timestamp=base)  # Exploit goes public
    vuln3.apply_event(CVDEvent.A, timestamp=base + timedelta(days=1))  # Attacks begin
    vuln3.apply_event(CVDEvent.P, timestamp=base + timedelta(days=2))  # Public learns
    vuln3.apply_event(CVDEvent.V, timestamp=base + timedelta(days=3))  # Vendor notified
    console.print(f"  State: {vuln3.state} (zero-day: exploited with no fix)")
    CVDFormatter.print_vulnerability(vuln3)
    console.print()


def section_3_state_properties():
    """Section 3: State properties and checks."""
    console.print(
        Panel.fit("[bold cyan]Section 3: State Properties[/bold cyan]", border_style="cyan")
    )

    # Create vulnerability with some events
    vuln = CVDVulnerability("CVE-2024-TEST", cvss_score=7.5)
    vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))
    vuln.apply_event(CVDEvent.F, timestamp=datetime(2024, 1, 15))
    vuln.apply_event(CVDEvent.P, timestamp=datetime(2024, 1, 10))

    console.print("[bold]State Information:[/bold]")
    console.print("[magenta]Properties: .state, .state_description, .state_label[/magenta]")
    console.print(f"  State:       {vuln.state}")
    console.print(f"  Description: {vuln.state_description}")
    console.print(f"  Label:       {vuln.state_label}")
    console.print()

    console.print("[bold]Event Checks:[/bold]")
    console.print("[magenta]Method: vuln.has_event_occurred(CVDEvent) -> bool[/magenta]")
    for event in CVDEvent:
        occurred = vuln.has_event_occurred(event)
        status = "[green]Yes[/green]" if occurred else "[dim]No[/dim]"
        console.print(f"  {event.name}: {status}")
    console.print()

    console.print("[bold]Analytics (via analyze()):[/bold]")
    console.print("[magenta]Method: vuln.analyze() -> AnalysisResult[/magenta]")
    result = vuln.analyze()
    console.print(f"  Is Zero-Day:      {result.is_zero_day[0]}")
    console.print(f"  Is Coordinated:   {result.is_coordinated[0]}")
    console.print(f"  Desiderata Score: {result.desiderata_score[0]}")
    console.print()


def section_3b_polished_concepts():
    """Section 3b: Polished Concepts - FixPath and ThreatState."""
    console.print(
        Panel.fit("[bold cyan]Section 3b: Polished Concepts[/bold cyan]", border_style="cyan")
    )

    from vulnstate.constants import FixPath, ThreatState

    console.print("[bold]FixPath & ThreatState Dimensions:[/bold]")
    console.print("  [magenta]FixPath[/magenta] - Progress toward remediation (V→F→D)")
    console.print("  [magenta]ThreatState[/magenta] - Threat evolution (P→X→A)")
    console.print()

    # Create vulnerability with specific state
    vuln = CVDVulnerability("CVE-2024-DEMO", cvss_score=8.2)
    vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))
    vuln.apply_event(CVDEvent.F, timestamp=datetime(2024, 1, 15))
    vuln.apply_event(CVDEvent.P, timestamp=datetime(2024, 1, 10))
    vuln.apply_event(CVDEvent.X, timestamp=datetime(2024, 1, 12))

    console.print("[yellow]Example vulnerability (VFdPXa):[/yellow]")
    console.print(f"  State:        {vuln.state}")
    console.print(f"  Fix Path:     {vuln.fix_path} ({vuln.fix_path.name})")
    console.print(f"  Threat State: {vuln.threat_state} ({vuln.threat_state.name})")
    console.print()

    console.print("[bold]FixPath Values:[/bold]")
    console.print("  NO_FIX (0)      - Vendor unaware (v)")
    console.print("  FIX_READY (1)   - Fix exists but not deployed (VFd)")
    console.print("  REMEDIATED (2)  - Fix deployed (VFD)")
    console.print()

    console.print("[bold]ThreatState Values:[/bold]")
    console.print("  PRIVATE (0)     - Not publicly known (p)")
    console.print("  PUBLIC (1)      - Public awareness (P)")
    console.print("  WEAPONIZED (2)  - Exploit available (PX)")
    console.print("  ATTACKED (3)    - Active exploitation (PXA)")
    console.print()


def section_4_batch_operations():
    """Section 4: Working with multiple vulnerabilities."""
    console.print(
        Panel.fit("[bold cyan]Section 4: Batch Operations[/bold cyan]", border_style="cyan")
    )

    # Create dataset
    console.print("[yellow]Creating 100 vulnerabilities...[/yellow]")
    console.print("[magenta]Class: CVDArray - vectorized container[/magenta]")
    vulns = []
    for i in range(100):
        v = CVDVulnerability(
            f"CVE-2024-{i:05d}",
            vendor=["Apache", "Microsoft", "Linux"][i % 3],
            severity=["low", "medium", "high", "critical"][i % 4],
            cvss_score=round(random.uniform(3.0, 10.0), 1),
        )
        v.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))
        if random.random() < 0.7:
            v.apply_event(CVDEvent.F, timestamp=datetime(2024, 1, 15))
        vulns.append(v)

    console.print("[dim]>>> arr = CVDArray(vulns)[/dim]")
    arr = CVDArray(vulns)
    console.print(f"  Created array with {len(arr)} items")
    console.print()

    # Quick summary
    console.print("[magenta]Helper: CVDFormatter.format_array_summary(arr) -> Table[/magenta]")
    CVDFormatter.print_array_summary(arr)
    console.print()

    # State distribution
    console.print("[bold]State Distribution:[/bold]")
    console.print("[magenta]Method: arr.count_by_state() -> Dict[str, int][/magenta]")
    state_counts = arr.count_by_state()
    for state, count in sorted(state_counts.items(), key=lambda x: -x[1])[:5]:
        console.print(f"  {state}: {count}")
    console.print()


def section_5_serialization():
    """Section 5: Serialization and export."""
    console.print(Panel.fit("[bold cyan]Section 5: Serialization[/bold cyan]", border_style="cyan"))

    # Create sample data
    vulns = []
    for i in range(10):
        v = CVDVulnerability(
            f"CVE-2024-{i:05d}",
            vendor=["Apache", "Microsoft"][i % 2],
            cvss_score=round(random.uniform(4.0, 9.0), 1),
        )
        v.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))
        if i % 2 == 0:
            v.apply_event(CVDEvent.F, timestamp=datetime(2024, 1, 15))
        vulns.append(v)

    arr = CVDArray(vulns)

    console.print("[yellow]Export to dict batch...[/yellow]")
    console.print("[magenta]Method: arr.to_dict_batch() -> List[Dict][/magenta]")
    dicts = arr.to_dict_batch()
    console.print(f"  Records: {len(dicts)}")
    console.print()

    console.print("[bold]Sample Record:[/bold]")
    sample = dicts[0]
    for key in ["vuln_id", "state", "vendor", "cvss_score"]:
        if key in sample:
            console.print(f"  {key}: {sample[key]}")
    console.print()

    console.print("[bold]Other Export Methods:[/bold]")
    console.print("  arr.to_json_batch()     -> JSON string")
    console.print("  arr.save_pickle_batch() -> Binary file")
    console.print("  vuln.to_dict()          -> Single record dict")
    console.print()


def section_6_next_steps():
    """Section 6: What to learn next."""
    console.print(Panel.fit("[bold cyan]Next Steps[/bold cyan]", border_style="cyan"))
    console.print()
    console.print("  02_batch_operations.py - Vectorized filtering and analysis")
    console.print("  03_risk_analysis.py    - Risk assessment workflow")
    console.print("  cvd_workflow.ipynb     - Interactive Jupyter notebook")
    console.print()


def main():
    """Run the getting started tutorial."""
    section_1_introduction()
    section_2_single_vulnerability()
    section_3_state_properties()
    section_3b_polished_concepts()
    section_4_batch_operations()
    section_5_serialization()
    section_6_next_steps()

    console.print(
        Panel.fit("[bold green]Getting Started Complete![/bold green]", border_style="green")
    )


if __name__ == "__main__":
    main()
