"""
Batch Operations: Vectorized Processing at Scale

Learn to efficiently process thousands of vulnerabilities using
CVDArray's vectorized operations.

Topics:
- Creating and populating CVDArray
- Boolean masking and filtering
- State distribution analysis
- Batch event application
- DataFrame conversion for analysis
"""

from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vulnstate import CVDArray, CVDEvent

console = Console()


def create_sample_dataset(size: int) -> CVDArray:
    """Create a sample vulnerability dataset using CVDArray.generate()."""
    return CVDArray.generate(
        size,
        vendors=["Apache", "Microsoft", "Apple", "Linux", "OpenSSL", "Nginx"],
    )


def section_1_creating_arrays():
    """Section 1: Creating and working with CVDArray."""
    console.print(
        Panel.fit("[bold cyan]Section 1: Creating Arrays[/bold cyan]", border_style="cyan")
    )

    console.print("[bold magenta]Key Classes:[/bold magenta]")
    console.print("  [magenta]CVDArray[/magenta] - Vectorized container for batch operations")
    console.print("  [magenta]CVDArray(vulns)[/magenta] - Create from list of CVDVulnerability")
    console.print()

    # Create from list
    console.print("[yellow]Creating array from vulnerability list...[/yellow]")
    console.print("[dim]>>> vulns = [CVDVulnerability(...) for i in range(1000)][/dim]")
    console.print("[dim]>>> arr = CVDArray(vulns)[/dim]")
    arr = create_sample_dataset(1000)
    console.print(f"  Created array with {len(arr)} items")
    console.print()

    # Factory methods
    console.print("[bold]Factory Methods:[/bold]")
    console.print(
        "[magenta]CVDArray.zeros(n) -> CVDArray[/magenta] - "
        "create n items in initial state (vfdpxa)"
    )
    console.print(
        "[magenta]CVDArray.ones(n) -> CVDArray[/magenta] - "
        "create n items in terminal state (VFDPXA)"
    )
    console.print(
        "[magenta]CVDArray.random(n) -> CVDArray[/magenta] - create n items with random states"
    )
    console.print()

    zeros = CVDArray.zeros(5, vuln_id_prefix="INIT")
    ones = CVDArray.ones(5, vuln_id_prefix="TERM")
    randoms = CVDArray.random(5, vuln_id_prefix="RND", seed=42)

    console.print(f"  zeros(5):  {set(zeros.states)}")
    console.print(f"  ones(5):   {set(ones.states)}")
    console.print(f"  random(5): {set(randoms.states)}")
    console.print()


def section_2_boolean_masking():
    """Section 2: Vectorized filtering with boolean masks."""
    console.print(
        Panel.fit("[bold cyan]Section 2: Boolean Masking[/bold cyan]", border_style="cyan")
    )

    arr = create_sample_dataset(5000)
    console.print(f"[yellow]Working with {len(arr)} vulnerabilities...[/yellow]\n")

    # Boolean mask from event check
    console.print("[bold]Event-Based Masks:[/bold]")
    console.print("[magenta].has_event_occurred(CVDEvent) -> np.ndarray[/magenta]")
    console.print()

    has_fix = arr.has_event_occurred(CVDEvent.F)
    has_deploy = arr.has_event_occurred(CVDEvent.D)
    has_public = arr.has_event_occurred(CVDEvent.P)

    console.print(f"  Has fix (F):     {has_fix.sum():,} ({has_fix.sum()/len(arr)*100:.1f}%)")
    console.print(f"  Has deploy (D):  {has_deploy.sum():,} ({has_deploy.sum()/len(arr)*100:.1f}%)")
    console.print(f"  Is public (P):   {has_public.sum():,} ({has_public.sum()/len(arr)*100:.1f}%)")
    console.print()

    # Combine masks
    console.print("[bold]Combining Masks:[/bold]")
    console.print("[dim]Use & (and), | (or), ~ (not) operators[/dim]")
    console.print()

    has_f_not_d = has_fix & ~has_deploy
    public_no_fix = has_public & ~has_fix
    console.print(f"  Fix ready but NOT deployed: {has_f_not_d.sum():,}")
    console.print(f"  Public but NO fix:          {public_no_fix.sum():,}")
    console.print()


def section_3_state_distribution():
    """Section 3: Analyzing state distribution."""
    console.print(
        Panel.fit("[bold cyan]Section 3: State Distribution[/bold cyan]", border_style="cyan")
    )

    arr = create_sample_dataset(5000)

    console.print("[bold]State Distribution:[/bold]")
    console.print("[magenta].count_by_state() -> Dict[str, int][/magenta]")
    console.print()

    state_counts = arr.count_by_state()
    table = Table(title="Top 10 States")
    table.add_column("State", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Percent", style="yellow")

    for state, count in sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = count / len(arr) * 100
        table.add_row(state, f"{count:,}", f"{pct:.1f}%")

    console.print(table)
    console.print()

    # Labeled distribution
    console.print("[bold]Cube (Label) Distribution:[/bold]")
    console.print("[magenta].state_distribution_labeled -> Dict[str, int][/magenta]")
    console.print()

    label_dist = arr.state_distribution_labeled
    for label, count in sorted(label_dist.items(), key=lambda x: -x[1])[:5]:
        pct = count / len(arr) * 100
        console.print(f"  {label:.<15} {count:>5,} ({pct:>5.1f}%)")
    console.print()


def section_3b_dimension_analysis():
    """Section 3b: Dimension-based analysis using FixPath and ThreatState."""
    console.print(
        Panel.fit("[bold cyan]Section 3b: Dimension Analysis[/bold cyan]", border_style="cyan")
    )

    from vulnstate.constants import FixPath, ThreatState

    arr = create_sample_dataset(5000)

    console.print("[bold]FixPath Distribution:[/bold]")
    console.print("[magenta].count_by_fix_path() -> Dict[FixPath, int][/magenta]")
    console.print()

    # Count by fix path using convenience method
    fix_counts = arr.count_by_fix_path()
    for fp in [FixPath.NO_AWARENESS, FixPath.VENDOR_AWARE, FixPath.FIX_READY, FixPath.REMEDIATED]:
        count = fix_counts.get(fp, 0)
        pct = count / len(arr) * 100
        console.print(f"  {fp.label:.<20} {count:>5,} ({pct:>5.1f}%)")
    console.print()

    console.print("[bold]ThreatState Distribution:[/bold]")
    console.print("[magenta].count_by_threat_state() -> Dict[ThreatState, int][/magenta]")
    console.print()

    # Count by threat state using convenience method
    threat_counts = arr.count_by_threat_state()
    for ts in ThreatState:
        count = threat_counts.get(ts, 0)
        if count > 0:
            pct = count / len(arr) * 100
            console.print(f"  {ts.label:.<20} {count:>5,} ({pct:>5.1f}%)")
    console.print()

    console.print("[bold]Risk Filtering by Dimension:[/bold]")
    console.print("[dim]Combine fix_path and threat_state for risk segmentation[/dim]")
    console.print()

    # High-risk: weaponized threats without fixes
    high_risk = arr[
        (arr.fix_path < FixPath.FIX_READY) & (arr.threat_state >= ThreatState.WEAPONIZED)
    ]
    console.print(f"  High risk (no fix + weaponized): {len(high_risk):,}")

    # Medium-risk: public but fix ready
    med_risk = arr[
        (arr.fix_path == FixPath.FIX_READY) & (arr.threat_state >= ThreatState.DISCLOSED)
    ]
    console.print(f"  Medium risk (fix ready + public): {len(med_risk):,}")

    # Low-risk: remediated
    low_risk = arr[arr.fix_path == FixPath.REMEDIATED]
    console.print(f"  Low risk (remediated): {len(low_risk):,}")
    console.print()


def section_4_batch_event_application():
    """Section 4: Applying events to many items at once."""
    console.print(
        Panel.fit("[bold cyan]Section 4: Batch Event Application[/bold cyan]", border_style="cyan")
    )

    arr = create_sample_dataset(5000)
    console.print("[yellow]Applying events to 5,000 vulnerabilities...[/yellow]\n")

    # Get baseline
    before_p = arr.has_event_occurred(CVDEvent.P).sum()
    console.print(f"[bold]Before:[/bold] {before_p:,} items with P event\n")

    # Apply P to all that don't have it
    console.print("[bold]Applying P event (Public Awareness) to all...[/bold]")
    console.print("[magenta].apply_event_batch(CVDEvent, timestamp) -> np.ndarray[/magenta]")
    console.print()

    mask = arr.apply_event_batch(CVDEvent.P, timestamp=datetime(2024, 6, 15))

    after_p = arr.has_event_occurred(CVDEvent.P).sum()
    changed = mask.sum()
    console.print(f"  Changed: {changed:,} items")
    console.print(f"  Total with P: {after_p:,}")
    console.print()

    # Show state distribution
    console.print("[bold]State distribution after P event:[/bold]")
    state_counts = arr.count_by_state()
    top_states = sorted(state_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for state, count in top_states:
        pct = count / len(arr) * 100
        console.print(f"    {state}: {count:,} ({pct:.1f}%)")
    console.print()


def section_5_serialization():
    """Section 5: Serialization and export."""
    console.print(Panel.fit("[bold cyan]Section 5: Serialization[/bold cyan]", border_style="cyan"))

    arr = create_sample_dataset(100)
    console.print("[bold magenta]Export Methods:[/bold magenta]")
    console.print("[magenta].to_dict_batch() -> List[Dict][/magenta]")
    console.print("[magenta].to_json_batch() -> str[/magenta]")
    console.print("[magenta].save_pickle_batch(path) -> None[/magenta]")
    console.print()

    console.print("[yellow]Exporting to dict batch...[/yellow]")
    dicts = arr.to_dict_batch()
    console.print(f"  Records: {len(dicts)}")
    console.print()

    console.print("[bold]Sample Record Fields:[/bold]")
    sample = dicts[0]
    for key in list(sample.keys())[:8]:
        value = sample[key]
        if isinstance(value, str) and len(value) > 40:
            value = value[:40] + "..."
        console.print(f"  {key}: {value}")
    console.print()

    console.print("[bold]Single Vuln JSON:[/bold]")
    json_str = arr.get(0).to_json()
    console.print(f"  Length: {len(json_str)} characters")
    console.print(f"  Preview: {json_str[:60]}...")
    console.print()

    console.print("[bold]Filtering with masks then export:[/bold]")
    has_fix = arr.has_event_occurred(CVDEvent.F)
    fixed_arr = arr[has_fix]
    console.print(f"  Fixed vulns: {len(fixed_arr)} of {len(arr)}")
    fixed_dicts = fixed_arr.to_dict_batch()
    console.print(f"  Exported records: {len(fixed_dicts)}")
    console.print()


def main():
    """Run all batch operations demos."""
    console.print(
        Panel.fit(
            "[bold cyan]Batch Operations: Vectorized Processing[/bold cyan]", border_style="cyan"
        )
    )
    console.print()

    section_1_creating_arrays()
    section_2_boolean_masking()
    section_3_state_distribution()
    section_3b_dimension_analysis()
    section_4_batch_event_application()
    section_5_serialization()

    console.print(
        Panel.fit("[bold green]Batch Operations Complete![/bold green]", border_style="green")
    )
    console.print()
    console.print("  03_risk_analysis.py  - Risk assessment workflow")
    console.print("  cvd_workflow.ipynb   - Interactive Jupyter notebook")
    console.print()


if __name__ == "__main__":
    main()
