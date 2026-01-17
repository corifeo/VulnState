"""
Data Import: Building a Vulnerability Dataset

Load data from multiple sources and build enriched CVDArray.

Topics:
- Creating vulnerabilities from scratch
- EPSS enrichment (exploit prediction scores)
- KEV enrichment (known exploited vulnerabilities)
- Export options
"""

import random
from datetime import datetime, timedelta

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vulnstate import CVDArray, CVDEvent, CVDVulnerability

console = Console()


def create_base_dataset(size: int = 50) -> CVDArray:
    """Create a base vulnerability dataset for enrichment demos."""
    vulns = []
    vendors = ["Apache", "Microsoft", "Linux", "OpenSSL"]

    for i in range(size):
        cve_id = f"CVE-2024-{i:05d}"
        vuln = CVDVulnerability(
            cve_id,
            vendor=vendors[i % len(vendors)],
            cvss_score=round(random.uniform(4.0, 10.0), 1),
            severity=["low", "medium", "high", "critical"][i % 4],
        )

        # Apply some events
        base = datetime(2024, 1, 1) + timedelta(days=i)
        vuln.apply_event(CVDEvent.V, timestamp=base)

        if random.random() < 0.6:
            vuln.apply_event(CVDEvent.F, timestamp=base + timedelta(days=random.randint(7, 30)))

        if random.random() < 0.4:
            vuln.apply_event(CVDEvent.P, timestamp=base + timedelta(days=random.randint(1, 20)))

        vulns.append(vuln)

    return CVDArray(vulns)


def section_1_creating_vulnerabilities():
    """Section 1: Creating vulnerabilities from scratch."""
    console.print(
        Panel.fit("[bold cyan]Section 1: Creating Vulnerabilities[/bold cyan]", border_style="cyan")
    )
    console.print()

    console.print("[bold]Single Vulnerability:[/bold]")
    console.print("[magenta]CVDVulnerability(cve_id, vendor=..., cvss_score=...)[/magenta]")
    console.print()

    vuln = CVDVulnerability(
        "CVE-2024-12345",
        vendor="Apache",
        cvss_score=8.5,
        severity="high",
    )
    vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 3, 15))
    vuln.apply_event(CVDEvent.P, timestamp=datetime(2024, 3, 20))

    console.print(f"  CVE ID: {vuln.cve_id}")
    console.print(f"  State: {vuln.state}")
    console.print(f"  CVSS: {vuln.cvss_score}")
    console.print()

    console.print("[bold]Batch Creation:[/bold]")
    console.print("[magenta]CVDArray([vuln1, vuln2, ...]) or CVDArray.zeros(n)[/magenta]")
    console.print()

    arr = create_base_dataset(20)
    console.print(f"  Created array with {len(arr)} vulnerabilities")
    console.print(f"  States: {len(arr.count_by_state())} unique states")
    console.print()


def section_2_epss_enrichment():
    """Section 2: Add EPSS scores to existing array."""
    console.print(
        Panel.fit("[bold cyan]Section 2: EPSS Enrichment[/bold cyan]", border_style="cyan")
    )
    console.print()

    arr = create_base_dataset(30)

    console.print("[bold]EPSS (Exploit Prediction Scoring System):[/bold]")
    console.print("  Probability that a CVE will be exploited in the next 30 days")
    console.print("  Range: 0.0 (unlikely) to 1.0 (highly likely)")
    console.print()

    console.print(
        "[bold magenta]Method: arr.import_epss(epss_data: Dict[str, float])[/bold magenta]"
    )
    console.print()

    # Simulate EPSS data (in real usage, load from FIRST.org API or CSV)
    epss_data = {}
    for i in range(30):
        cve_id = f"CVE-2024-{i:05d}"
        # High CVSS vulns tend to have higher EPSS (simulated)
        base_score = random.uniform(0.01, 0.5)
        if i % 4 == 3:  # Critical severity
            base_score += 0.3
        epss_data[cve_id] = min(round(base_score, 4), 1.0)

    console.print("[yellow]Importing EPSS scores...[/yellow]")
    arr.import_epss(epss_data)

    # Show some results
    console.print()
    console.print("[bold]Sample EPSS Scores:[/bold]")
    table = Table()
    table.add_column("CVE ID", style="cyan")
    table.add_column("CVSS", style="yellow")
    table.add_column("EPSS", style="green")
    table.add_column("Risk Level")

    for i in range(5):
        vuln = arr.get(i)
        epss = vuln.epss or 0.0
        risk = (
            "[red]HIGH[/red]"
            if epss > 0.5
            else "[yellow]MEDIUM[/yellow]" if epss > 0.1 else "[green]LOW[/green]"
        )
        table.add_row(vuln.cve_id, f"{vuln.cvss_score:.1f}", f"{epss:.4f}", risk)

    console.print(table)
    console.print()

    # Filter by EPSS
    console.print("[bold]Filtering by EPSS threshold:[/bold]")
    high_epss_count = sum(1 for i in range(len(arr)) if (arr.get(i).epss or 0) > 0.3)
    console.print(f"  Vulns with EPSS > 0.3: {high_epss_count} of {len(arr)}")
    console.print()


def section_3_kev_enrichment():
    """Section 3: Mark Known Exploited Vulnerabilities."""
    console.print(
        Panel.fit("[bold cyan]Section 3: KEV Enrichment[/bold cyan]", border_style="cyan")
    )
    console.print()

    arr = create_base_dataset(30)

    console.print("[bold]CISA KEV (Known Exploited Vulnerabilities):[/bold]")
    console.print("  Catalog of CVEs actively exploited in the wild")
    console.print("  Federal agencies must remediate within specified timelines")
    console.print()

    console.print("[bold magenta]Method: arr.import_kev(kev_cves: Set[str])[/bold magenta]")
    console.print()

    # Simulate KEV catalog (in real usage, download from CISA)
    kev_cves = {f"CVE-2024-{i:05d}" for i in [0, 5, 10, 15, 20, 25]}

    console.print("[yellow]Importing KEV catalog...[/yellow]")
    console.print(f"  KEV entries to match: {len(kev_cves)}")
    arr.import_kev(kev_cves)

    # Show results
    console.print()
    console.print("[bold]KEV Status:[/bold]")
    kev_count = sum(1 for i in range(len(arr)) if arr.get(i).is_kev)
    console.print(f"  Marked as KEV: {kev_count} of {len(arr)}")
    console.print()

    console.print("[bold]KEV Vulnerabilities:[/bold]")
    table = Table()
    table.add_column("CVE ID", style="cyan")
    table.add_column("State", style="yellow")
    table.add_column("CVSS", style="green")
    table.add_column("KEV", style="red")

    shown = 0
    for i in range(len(arr)):
        vuln = arr.get(i)
        if vuln.is_kev and shown < 5:
            table.add_row(
                vuln.cve_id,
                vuln.state,
                f"{vuln.cvss_score:.1f}",
                "[red]YES[/red]",
            )
            shown += 1

    console.print(table)
    console.print()


def section_4_combined_enrichment():
    """Section 4: Combined enrichment workflow."""
    console.print(
        Panel.fit(
            "[bold cyan]Section 4: Combined Enrichment Workflow[/bold cyan]", border_style="cyan"
        )
    )
    console.print()

    console.print("[bold]Typical Data Pipeline:[/bold]")
    console.print("  1. Create/load base vulnerability data")
    console.print("  2. Enrich with EPSS scores")
    console.print("  3. Mark KEV entries")
    console.print("  4. Run analysis")
    console.print("  5. Export for reporting")
    console.print()

    # Step 1: Create base data
    console.print("[yellow]Step 1: Create base dataset...[/yellow]")
    arr = create_base_dataset(50)
    console.print(f"  Loaded {len(arr)} vulnerabilities")

    # Step 2: EPSS
    console.print("[yellow]Step 2: Enrich with EPSS...[/yellow]")
    epss_data = {f"CVE-2024-{i:05d}": round(random.uniform(0.01, 0.8), 4) for i in range(50)}
    arr.import_epss(epss_data)
    console.print("  EPSS scores imported")

    # Step 3: KEV
    console.print("[yellow]Step 3: Mark KEV entries...[/yellow]")
    kev_cves = {f"CVE-2024-{i:05d}" for i in range(0, 50, 7)}
    arr.import_kev(kev_cves)
    kev_count = sum(1 for i in range(len(arr)) if arr.get(i).is_kev)
    console.print(f"  Marked {kev_count} KEV entries")

    # Step 4: Analysis
    console.print("[yellow]Step 4: Run analysis...[/yellow]")
    result = arr.analysis
    zero_day_count = result.is_zero_day.sum()
    console.print(f"  Zero-day vulns: {zero_day_count}")

    # Step 5: Export
    console.print("[yellow]Step 5: Export...[/yellow]")
    dicts = arr.to_dict_batch()
    console.print(f"  Exported {len(dicts)} records")
    console.print()

    # Priority report
    console.print("[bold]Priority Report (KEV + High EPSS):[/bold]")
    table = Table()
    table.add_column("CVE ID", style="cyan")
    table.add_column("CVSS", style="yellow")
    table.add_column("EPSS", style="green")
    table.add_column("KEV", style="red")
    table.add_column("State")

    priority_vulns = []
    for i in range(len(arr)):
        vuln = arr.get(i)
        epss = vuln.epss or 0.0
        if vuln.is_kev or epss > 0.5:
            priority_vulns.append((vuln, epss))

    for vuln, epss in sorted(priority_vulns, key=lambda x: -x[1])[:8]:
        kev_status = "[red]YES[/red]" if vuln.is_kev else "[dim]no[/dim]"
        table.add_row(
            vuln.cve_id,
            f"{vuln.cvss_score:.1f}",
            f"{epss:.4f}",
            kev_status,
            vuln.state,
        )

    console.print(table)
    console.print()


def section_4b_dimension_based_filtering():
    """Section 4b: Risk segmentation using dimensions."""
    console.print(
        Panel.fit(
            "[bold cyan]Section 4b: Risk Segmentation by Dimension[/bold cyan]", border_style="cyan"
        )
    )
    console.print()

    from vulnstate.constants import FixPath, ThreatState

    # Create enriched dataset
    arr = create_base_dataset(100)
    epss_data = {f"CVE-2024-{i:05d}": round(random.uniform(0.01, 0.8), 4) for i in range(100)}
    arr.import_epss(epss_data)
    kev_cves = {f"CVE-2024-{i:05d}" for i in range(0, 100, 10)}
    arr.import_kev(kev_cves)

    console.print("[bold]Dimension-Based Risk Segmentation:[/bold]")
    console.print("[magenta].fix_path, .threat_state -> filter by remediation and threat[/magenta]")
    console.print()

    # High risk: no fix + weaponized
    high_risk = arr[
        (arr.fix_path < FixPath.FIX_READY) &
        (arr.threat_state >= ThreatState.WEAPONIZED)
    ]
    console.print(f"  [red]High risk[/red] (no fix + weaponized): {len(high_risk)}")

    # Medium risk: fix ready but public
    med_risk = arr[
        (arr.fix_path == FixPath.FIX_READY) &
        (arr.threat_state >= ThreatState.PUBLIC)
    ]
    console.print(f"  [yellow]Medium risk[/yellow] (fix ready + public): {len(med_risk)}")

    # Low risk: remediated
    low_risk = arr[arr.fix_path == FixPath.REMEDIATED]
    console.print(f"  [green]Low risk[/green] (remediated): {len(low_risk)}")
    console.print()

    console.print("[bold]Priority Focus: KEV + High Threat State:[/bold]")
    priority = []
    for i in range(len(arr)):
        vuln = arr.get(i)
        if vuln.is_kev and vuln.threat_state >= ThreatState.WEAPONIZED:
            priority.append(vuln)

    console.print(f"  KEV entries with weaponized threat: {len(priority)}")
    if priority:
        console.print("\n  [bold]Top Priority Items:[/bold]")
        for vuln in priority[:3]:
            console.print(f"    {vuln.cve_id}: {vuln.state} (CVSS {vuln.cvss_score})")
    console.print()


def section_5_export_options():
    """Section 5: Export options."""
    console.print(
        Panel.fit("[bold cyan]Section 5: Export Options[/bold cyan]", border_style="cyan")
    )
    console.print()

    arr = create_base_dataset(20)

    console.print("[bold]Available Export Methods:[/bold]")
    console.print()

    console.print("[magenta]arr.to_dict_batch()[/magenta] -> List[Dict]")
    dicts = arr.to_dict_batch()
    console.print(f"  Records: {len(dicts)}")
    console.print(f"  Fields: {list(dicts[0].keys())[:5]}...")
    console.print()

    console.print("[magenta]arr.to_json_batch(filepath)[/magenta] -> None (writes file)")
    console.print("  Saves all vulnerabilities to JSON file")
    console.print()

    console.print("[magenta]arr.save_pickle_batch(filepath)[/magenta] -> None (writes file)")
    console.print("  Saves binary format (fastest for large datasets)")
    console.print()

    console.print("[magenta]vuln.to_dict() / vuln.to_json()[/magenta] -> single record")
    vuln = arr.get(0)
    console.print(f"  Sample: {vuln.to_json()[:60]}...")
    console.print()

    console.print("[bold]Roundtrip Example:[/bold]")
    console.print("  # Save")
    console.print("  arr.to_json_batch('vulnerabilities.json')")
    console.print("  # Load")
    console.print("  arr2 = CVDArray.from_json_batch('vulnerabilities.json')")
    console.print()


def main():
    """Run data import workflow demo."""
    console.print(
        Panel.fit(
            "[bold magenta]Data Import: Building a Vulnerability Dataset[/bold magenta]\n"
            "Loading, enriching, and exporting vulnerability data",
            border_style="magenta",
        )
    )
    console.print()

    section_1_creating_vulnerabilities()
    section_2_epss_enrichment()
    section_3_kev_enrichment()
    section_4_combined_enrichment()
    section_4b_dimension_based_filtering()
    section_5_export_options()

    console.print(
        Panel.fit("[bold green]Data Import Demo Complete![/bold green]", border_style="green")
    )
    console.print()
    console.print("  data_science_workflow.ipynb - Statistical analysis notebook")
    console.print("  security_analysis.ipynb     - Risk assessment notebook")
    console.print()


if __name__ == "__main__":
    main()
