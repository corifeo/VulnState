"""
Risk Analysis: Portfolio Assessment Workflow

Learn to analyze vulnerability portfolios for risk prioritization.
Covers statistical analysis, risk tiers, and export for reporting.

Topics:
- Portfolio statistics and vendor breakdown
- Risk tier classification (CVSS + exploit status)
- Vendor analysis and comparison
- Export for reporting

Note: This example uses pandas for DataFrame operations.
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vulnstate import CVDArray, CVDEvent

console = Console()


def create_sample_dataset(size: int = 500) -> CVDArray:
    """Create sample vulnerability dataset using CVDArray.generate()."""
    return CVDArray.generate(
        size,
        event_probs={
            CVDEvent.V: 1.0,
            CVDEvent.F: 0.7,
            CVDEvent.D: 0.4,
            CVDEvent.P: 0.5,
            CVDEvent.X: 0.1,
            CVDEvent.A: 0.05,
        },
        vendors=["Apache", "Microsoft", "Apple", "Linux", "OpenSSL", "Nginx"],
    )


def section_1_portfolio_overview():
    """Section 1: Portfolio statistics and severity breakdown."""
    console.print(
        Panel.fit("[bold cyan]Section 1: Portfolio Overview[/bold cyan]", border_style="cyan")
    )

    arr = create_sample_dataset(500)
    df = arr.to_dataframe(include_events=True)
    console.print(f"[yellow]Analyzing portfolio of {len(arr)} vulnerabilities...[/yellow]\n")

    # CVSS Statistics
    console.print("[bold]CVSS Score Statistics:[/bold]")
    console.print("[magenta]DataFrame column: cvss_score[/magenta]")
    cvss_stats = df["cvss_score"].describe()
    console.print(f"  Mean:   {cvss_stats['mean']:.2f}")
    console.print(f"  Median: {df['cvss_score'].median():.2f}")
    console.print(f"  Std:    {cvss_stats['std']:.2f}")
    console.print(f"  Range:  {cvss_stats['min']:.1f} - {cvss_stats['max']:.1f}")
    console.print()

    # Severity distribution (derived from CVSS score)
    import pandas as pd

    console.print("[bold]Severity Distribution (from CVSS):[/bold]")
    df["severity"] = pd.cut(
        df["cvss_score"],
        bins=[0, 4.0, 7.0, 9.0, 10.0],
        labels=["low", "medium", "high", "critical"],
    )
    severity_counts = df["severity"].value_counts()
    for severity in ["critical", "high", "medium", "low"]:
        count = severity_counts.get(severity, 0)
        pct = count / len(df) * 100
        bar = "#" * int(pct / 2)
        console.print(f"  {severity:.<12} {count:>4} ({pct:>5.1f}%) {bar}")
    console.print()

    # Event occurrence
    console.print("[bold]Event Occurrence Rates:[/bold]")
    console.print("[magenta]DataFrame columns: V, F, D, P, X, A (0/1 flags)[/magenta]")
    event_info = [
        ("V", "Vendor Aware"),
        ("F", "Fix Ready"),
        ("D", "Deployed"),
        ("P", "Public"),
        ("X", "Exploit"),
        ("A", "Attacks"),
    ]
    for event, description in event_info:
        count = int(df[event].sum())
        pct = count / len(df) * 100
        console.print(f"  {event} ({description:.<15}) {count:>4} ({pct:>5.1f}%)")
    console.print()


def section_1b_dimension_based_risk():
    """Section 1b: Risk analysis using FixPath and ThreatState."""
    console.print(
        Panel.fit("[bold cyan]Section 1b: Dimension-Based Risk[/bold cyan]", border_style="cyan")
    )

    from vulnstate.constants import FixPath, ThreatState

    arr = create_sample_dataset(500)
    console.print("[yellow]Analyzing risk using FixPath and ThreatState dimensions...[/yellow]\n")

    console.print("[bold]Risk Matrix (FixPath × ThreatState):[/bold]")
    console.print("[magenta].fix_path, .threat_state -> dimension arrays[/magenta]")
    console.print()

    # Create risk matrix
    table = Table(title="Vulnerability Distribution by Dimension")
    table.add_column("Fix Path \\ Threat", style="cyan")
    table.add_column("Private", style="green")
    table.add_column("Public", style="yellow")
    table.add_column("Weaponized", style="red")
    table.add_column("Attacked", style="red bold")

    for fix_path in [FixPath.NO_AWARENESS, FixPath.FIX_READY, FixPath.REMEDIATED]:
        row = [fix_path.name]
        for threat_state in [
            ThreatState.LATENT,
            ThreatState.DISCLOSED,
            ThreatState.WEAPONIZED,
            ThreatState.ACTIVE_THREAT,
        ]:
            count = ((arr.fix_path == fix_path) & (arr.threat_state == threat_state)).sum()
            row.append(str(count))
        table.add_row(*row)

    console.print(table)
    console.print()

    # Priority segments
    console.print("[bold]Risk Priority Segments:[/bold]")
    critical_risk = arr[
        (arr.fix_path == FixPath.NO_AWARENESS) & (arr.threat_state >= ThreatState.WEAPONIZED)
    ]
    high_risk = arr[
        (arr.fix_path == FixPath.FIX_READY) & (arr.threat_state >= ThreatState.WEAPONIZED)
    ]
    watch_list = arr[
        (arr.fix_path < FixPath.REMEDIATED) & (arr.threat_state >= ThreatState.DISCLOSED)
    ]

    console.print(f"  [red]Critical:[/red] No fix + weaponized/attacked: {len(critical_risk)}")
    console.print(f"  [yellow]High:[/yellow]     Fix ready + weaponized/attacked: {len(high_risk)}")
    console.print(f"  [blue]Watch:[/blue]    Unpatched + public: {len(watch_list)}")
    console.print()


def section_2_risk_tiers():
    """Section 2: Risk tier classification."""
    console.print(Panel.fit("[bold cyan]Section 2: Risk Tiers[/bold cyan]", border_style="cyan"))

    arr = create_sample_dataset(500)
    df = arr.to_dataframe(include_events=True)
    console.print("[yellow]Classifying vulnerabilities into risk tiers...[/yellow]\n")

    console.print("[bold]Risk Classification Logic:[/bold]")
    console.print("  [red]Critical:[/red] CVSS >= 9.0 AND (exploit available OR under attack)")
    console.print("  [yellow]High:[/yellow]     CVSS >= 7.0 AND no fix available")
    console.print("  [blue]Medium:[/blue]   CVSS >= 5.0 AND publicly known")
    console.print("  [green]Low:[/green]      CVSS < 5.0")
    console.print()

    # Calculate risk tiers
    critical_mask = (df["cvss_score"] >= 9.0) & ((df["X"] == 1) | (df["A"] == 1))
    high_mask = (df["cvss_score"] >= 7.0) & (df["F"] == 0) & ~critical_mask
    medium_mask = (df["cvss_score"] >= 5.0) & (df["P"] == 1) & ~critical_mask & ~high_mask
    low_mask = df["cvss_score"] < 5.0

    tiers = [
        ("Critical", critical_mask, "red"),
        ("High", high_mask, "yellow"),
        ("Medium", medium_mask, "blue"),
        ("Low", low_mask, "green"),
    ]

    table = Table(title="Risk Assessment")
    table.add_column("Tier", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Percent", style="yellow")
    table.add_column("Avg CVSS")

    for tier_name, mask, color in tiers:
        tier_df = df[mask]
        count = len(tier_df)
        pct = count / len(df) * 100
        avg_cvss = tier_df["cvss_score"].mean() if count > 0 else 0
        table.add_row(
            f"[{color}]{tier_name}[/{color}]", str(count), f"{pct:.1f}%", f"{avg_cvss:.1f}"
        )

    console.print(table)
    console.print()

    # Priority actions
    console.print("[bold]Recommended Actions:[/bold]")
    critical_count = critical_mask.sum()
    unpatched_critical = len(df[(df["cvss_score"] >= 9.0) & (df["F"] == 0)])
    public_unpatched = len(df[(df["P"] == 1) & (df["F"] == 0)])

    console.print(f"  1. [red]Emergency response:[/red] {critical_count} critical exploited vulns")
    console.print(f"  2. [yellow]Urgent patching:[/yellow] {unpatched_critical} unpatched critical")
    console.print(f"  3. [blue]Fast-track fixes:[/blue] {public_unpatched} public unpatched vulns")
    console.print()


def section_3_vendor_analysis():
    """Section 3: Per-vendor vulnerability analysis."""
    console.print(
        Panel.fit("[bold cyan]Section 3: Vendor Analysis[/bold cyan]", border_style="cyan")
    )

    arr = create_sample_dataset(500)
    df = arr.to_dataframe(include_events=True)
    console.print("[yellow]Analyzing vulnerabilities by vendor...[/yellow]\n")

    console.print("[bold]Vendor Portfolio:[/bold]")
    console.print("[magenta]DataFrame column: vendor[/magenta]")
    console.print()

    table = Table(title="Vendor Vulnerability Metrics")
    table.add_column("Vendor", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Avg CVSS", style="yellow")
    table.add_column("Critical %", style="red")
    table.add_column("Patched %", style="green")

    for vendor in sorted(df["vendor"].unique()):
        vendor_data = df[df["vendor"] == vendor]
        count = len(vendor_data)
        avg_cvss = vendor_data["cvss_score"].mean()
        critical_pct = (vendor_data["cvss_score"] >= 9.0).sum() / count * 100
        patched_pct = vendor_data["F"].sum() / count * 100

        table.add_row(
            vendor, str(count), f"{avg_cvss:.1f}", f"{critical_pct:.0f}%", f"{patched_pct:.0f}%"
        )

    console.print(table)
    console.print()

    # Worst vendors
    console.print("[bold]Vendors with Most Unpatched Critical:[/bold]")
    for vendor in df["vendor"].unique():
        vendor_data = df[df["vendor"] == vendor]
        unpatched_critical = len(
            vendor_data[(vendor_data["cvss_score"] >= 9.0) & (vendor_data["F"] == 0)]
        )
        if unpatched_critical > 0:
            console.print(f"  {vendor}: {unpatched_critical} unpatched critical")
    console.print()


def section_4_export_for_reporting():
    """Section 4: Export data for reporting."""
    console.print(
        Panel.fit("[bold cyan]Section 4: Export for Reporting[/bold cyan]", border_style="cyan")
    )

    arr = create_sample_dataset(100)
    console.print("[bold magenta]Export Methods:[/bold magenta]")
    console.print("[magenta].to_dict_batch() -> List[Dict][/magenta]")
    console.print("[magenta].to_json_batch(path) -> None[/magenta]")
    console.print("[magenta]arr.to_dataframe(include_events=True) -> pd.DataFrame[/magenta]")
    console.print()

    # Dict batch export
    console.print("[yellow]Exporting to dict batch...[/yellow]")
    dicts = arr.to_dict_batch()
    console.print(f"  Records: {len(dicts)}")
    console.print()

    # DataFrame for analysis
    console.print("[bold]DataFrame Export Sample:[/bold]")
    df = arr.to_dataframe(include_events=True)
    sample_cols = ["vuln_id", "state", "severity", "cvss_score", "vendor"]
    available = [c for c in sample_cols if c in df.columns]
    console.print(df[available].head(5).to_string())
    console.print()

    # CSV via pandas
    console.print("[bold]Export to CSV via pandas:[/bold]")
    console.print("[magenta]df.to_csv('report.csv', index=False)[/magenta]")
    console.print("[dim]Tip: Use df.to_excel() or df.to_json() for other formats[/dim]")
    console.print()


def main():
    """Run risk analysis workflow."""
    console.print(
        Panel.fit("[bold cyan]Risk Analysis: Portfolio Assessment[/bold cyan]", border_style="cyan")
    )
    console.print()

    section_1_portfolio_overview()
    section_1b_dimension_based_risk()
    section_2_risk_tiers()
    section_3_vendor_analysis()
    section_4_export_for_reporting()

    console.print(
        Panel.fit("[bold green]Risk Analysis Complete![/bold green]", border_style="green")
    )
    console.print()
    console.print("  cvd_workflow.ipynb - Interactive Jupyter notebook")
    console.print()


if __name__ == "__main__":
    main()
