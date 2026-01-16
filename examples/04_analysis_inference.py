"""
Analysis and Inference: CVDAnalyzer Deep Dive

Learn how the analysis engine computes metrics and infers missing data.

Topics:
- CVDAnalyzer.analyze() - Compute all analytics
- Inference rules - How missing events are inferred
- Provenance tracking - Distinguish observed vs inferred data
- AnalysisResult - Access computed metrics
"""

from rich.console import Console
from rich.panel import Panel

from vulnstate import CVDArray, CVDEvent, CVDVulnerability

console = Console()


def section_1_basic_analysis():
    """Section 1: Run analyze() and explore AnalysisResult."""
    console.print(
        Panel.fit("[bold cyan]Section 1: Basic Analysis[/bold cyan]", border_style="cyan")
    )
    console.print()

    # Create array with some vulnerabilities
    vulns = [
        CVDVulnerability("CVE-2024-001", state="VFDpxa"),  # Fix deployed, no threat
        CVDVulnerability("CVE-2024-002", state="VfdPXa"),  # Zero-day exploit
        CVDVulnerability("CVE-2024-003", state="VFDPXA"),  # Complete/terminal
        CVDVulnerability("CVE-2024-004", state="vfdPxa"),  # Public before vendor aware
    ]
    arr = CVDArray(vulns)

    console.print("[bold]Created array with 4 vulnerabilities[/bold]")
    console.print()

    # Access arr.analysis (lazy, cached)
    console.print("[magenta]arr.analysis[/magenta] - Lazy computation, cached after first access")
    result = arr.analysis

    console.print()
    console.print("[bold]Group 1: Validity & Completeness[/bold]")
    console.print(f"  event_count: {result.event_count}")
    console.print(f"  is_complete: {result.is_complete}")
    console.print()

    console.print("[bold]Group 2: Desiderata[/bold]")
    console.print(f"  desiderata_score: {result.desiderata_score}")
    console.print()

    console.print("[bold]Group 3: Zero-Day Detection[/bold]")
    console.print(f"  is_zero_day: {result.is_zero_day}")
    console.print(f"  is_zero_day_exploit: {result.is_zero_day_exploit}")
    console.print()

    console.print("[bold]Group 4: Coordination Quality[/bold]")
    console.print(f"  is_coordinated: {result.is_coordinated}")
    console.print(f"  is_premature_disclosure: {result.is_premature_disclosure}")
    console.print()


def section_2_inference_rules():
    """Section 2: Demonstrate inference rules."""
    console.print(
        Panel.fit("[bold cyan]Section 2: Inference Rules[/bold cyan]", border_style="cyan")
    )
    console.print()

    console.print("[bold]Inference Rules:[/bold]")
    console.print("  P without V -> infer V at P timestamp (vendor learns from public)")
    console.print("  X without P -> infer P at X timestamp (exploit = disclosure)")
    console.print("  F without V -> infer V at F timestamp (fix implies awareness)")
    console.print("  D without F -> infer F at D timestamp (deploy implies fix)")
    console.print()

    # Create vuln with only P (public disclosure, vendor not aware)
    vuln = CVDVulnerability("CVE-2024-005")
    vuln.apply_event(CVDEvent.P)  # Only P occurred

    console.print("[yellow]Created vulnerability with only P event[/yellow]")
    console.print(f"  State before analysis: {vuln.state}")
    console.print(f"  V timestamp: {vuln.events.get(CVDEvent.V, 'Not set')}")
    console.print()

    arr = CVDArray([vuln])

    # Run analyze with inference (default)
    console.print("[magenta]analyze(infer=True)[/magenta] - V inferred from P")
    result_with_infer = arr.reanalyze(infer=True)
    console.print(f"  State after inference: {arr.get(0).state}")
    console.print(f"  inferred_V: {result_with_infer.inferred_V}")
    console.print()

    # Create fresh vuln to show infer=False
    vuln2 = CVDVulnerability("CVE-2024-006")
    vuln2.apply_event(CVDEvent.P)
    arr2 = CVDArray([vuln2])

    console.print("[magenta]analyze(infer=False)[/magenta] - No inference applied")
    result_no_infer = arr2.reanalyze(infer=False)
    console.print(f"  State (unchanged): {arr2.get(0).state}")
    console.print(f"  inferred_V: {result_no_infer.inferred_V}")
    console.print()


def section_3_provenance_tracking():
    """Section 3: Show how to distinguish observed vs inferred."""
    console.print(
        Panel.fit("[bold cyan]Section 3: Provenance Tracking[/bold cyan]", border_style="cyan")
    )
    console.print()

    # Create mixed dataset - some observed, some will be inferred
    vulns = [
        CVDVulnerability("CVE-2024-010", state="VFdpxa"),  # V observed
        CVDVulnerability("CVE-2024-011", state="vfdPxa"),  # V will be inferred from P
        CVDVulnerability("CVE-2024-012", state="VFDPxa"),  # V observed
        CVDVulnerability("CVE-2024-013", state="vfdPXa"),  # V inferred from P, P inferred from X
    ]
    arr = CVDArray(vulns)

    console.print("[bold]Dataset with 4 vulnerabilities[/bold]")
    console.print("  CVE-2024-010: V observed (state=VFdpxa)")
    console.print("  CVE-2024-011: V will be inferred from P")
    console.print("  CVE-2024-012: V observed (state=VFDPxa)")
    console.print("  CVE-2024-013: V inferred from P, P inferred from X")
    console.print()

    result = arr.analysis

    console.print("[bold]Provenance masks:[/bold]")
    console.print(f"  inferred_mask (raw): {result.inferred_mask}")
    console.print(f"  inferred_V: {result.inferred_V}")
    console.print(f"  inferred_P: {result.inferred_P}")
    console.print()

    # Filter to only observed V events
    observed_v_mask = ~result.inferred_V
    console.print("[yellow]Filter: Only observed V events[/yellow]")
    console.print(f"  Mask: {observed_v_mask}")
    console.print(f"  Count: {observed_v_mask.sum()} of {len(arr)}")
    console.print()

    console.print("[bold]Use case: Report on data quality[/bold]")
    pct_inferred = result.inferred_V.sum() / len(arr) * 100
    console.print(f"  {pct_inferred:.0f}% of V timestamps were inferred (not observed)")
    console.print()


def section_4_reanalysis():
    """Section 4: Demonstrate reanalyze() after data changes."""
    console.print(
        Panel.fit("[bold cyan]Section 4: Reanalysis After Changes[/bold cyan]", border_style="cyan")
    )
    console.print()

    # Load initial data
    vulns = [CVDVulnerability(f"CVE-2024-{i:03d}", state="Vfdpxa") for i in range(5)]
    arr = CVDArray(vulns)

    console.print("[bold]Initial dataset: 5 vulnerabilities in state 'Vfdpxa'[/bold]")
    result1 = arr.analysis
    console.print(f"  is_coordinated: {result1.is_coordinated}")
    console.print()

    # Simulate importing vendor data - add F events
    console.print("[yellow]Simulating vendor CSV import - adding F events...[/yellow]")
    for i in range(3):
        arr.get(i).apply_event(CVDEvent.F)
    arr.sync()

    console.print("[magenta]arr.reanalyze()[/magenta] - Force re-computation")
    result2 = arr.reanalyze()

    console.print()
    console.print("[bold]After import:[/bold]")
    console.print(f"  States: {list(arr.states_as_strings)}")
    console.print(f"  has_fix_before_exploit: {result2.has_fix_before_exploit}")
    console.print()


def main():
    """Run all sections."""
    console.print(
        Panel.fit(
            "[bold magenta]Analysis and Inference Demo[/bold magenta]\n"
            "CVDAnalyzer deep dive with inference and provenance",
            border_style="magenta",
        )
    )
    console.print()

    section_1_basic_analysis()
    section_2_inference_rules()
    section_3_provenance_tracking()
    section_4_reanalysis()

    console.print("[bold green]Demo complete![/bold green]")


if __name__ == "__main__":
    main()
