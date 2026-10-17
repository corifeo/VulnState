"""Regenerate the README screenshots and charts in .github/assets/.

Everything is computed by vulnstate from the sample data in examples/
(770 real 2023 NVD CVEs + EPSS + CISA KEV).

    uv run --extra viz python scripts/make_readme_assets.py
"""

from __future__ import annotations

import io
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from rich import box
from rich.console import Console
from rich.measure import Measurement
from rich.table import Table
from rich.text import Text

from vulnstate import CVDArray, CVDEvent, CVDVulnerability, FixPath, ThreatState

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
ASSETS = ROOT / ".github" / "assets"


def recording_console(width: int = 104) -> Console:
    return Console(
        record=True, width=width, file=io.StringIO(), force_terminal=True, color_system="truecolor"
    )


def yes_no(flag: bool, bad_when_true: bool) -> Text:
    if flag:
        return Text("yes", style="bold red" if bad_when_true else "bold green")
    return Text("no", style="green" if bad_when_true else "red")


def load_real_dataset() -> CVDArray:
    arr = CVDArray()
    arr.import_nvd(str(EXAMPLES / "sample_nvdcve_2023.json"))
    arr.import_epss(str(EXAMPLES / "sample_epss.csv"))
    arr.import_kev(str(EXAMPLES / "sample_kev.csv"))
    arr.import_csv(
        source=str(EXAMPLES / "sample_deployed.csv"),
        cve_column="cve_id",
        event=CVDEvent.D,
        timestamp_column="deployed_date",
    )
    arr.transform()
    return arr


# ---------------------------------------------------------------- lifecycle.svg


def make_lifecycle_svg() -> None:
    """Three disclosure stories -> what vulnstate infers from event ordering alone."""
    base = datetime(2024, 1, 1)
    stories = [
        ("Coordinated", [("V", 0), ("F", 30), ("D", 40), ("P", 45)]),
        ("Premature disclosure", [("V", 0), ("P", 5), ("X", 7), ("F", 20), ("A", 25)]),
        ("Zero-day in the wild", [("A", 0), ("X", 3), ("V", 4), ("P", 5), ("F", 12)]),
    ]
    vulns = []
    for i, (_, events) in enumerate(stories, start=1):
        v = CVDVulnerability(f"CVE-2024-000{i}")
        for name, day in events:
            v.apply_event(CVDEvent[name], timestamp=base + timedelta(days=day))
        vulns.append(v)

    arr = CVDArray(vulns)
    arr.transform()

    table = Table(box=box.ROUNDED, header_style="bold cyan", pad_edge=False)
    for col in [
        "Story",
        "Event order",
        "State",
        "Fix path",
        "Threat",
        "Zero-day",
        "Coordinated",
        "Premature",
        "Fix < exploit",
        "Desiderata",
    ]:
        table.add_column(
            col, justify="left" if col in ("Story", "Event order") else "center", no_wrap=True
        )

    for i, (story, events) in enumerate(stories):
        score = float(arr.desiderata_score[i])
        score_style = "green" if score >= 0.75 else "yellow" if score >= 0.4 else "red"
        table.add_row(
            Text(story, style="bold"),
            Text(" → ".join(name for name, _ in events), style="magenta"),
            Text(str(arr.states[i]), style="bold"),
            FixPath(int(arr.fix_path[i])).name,
            ThreatState(int(arr.threat_state[i])).name,
            yes_no(bool(arr.is_zero_day[i]), bad_when_true=True),
            yes_no(bool(arr.is_coordinated[i]), bad_when_true=False),
            yes_no(bool(arr.is_premature_disclosure[i]), bad_when_true=True),
            yes_no(bool(arr.has_fix_before_exploit[i]), bad_when_true=False),
            Text(f"{score:.2f}", style=f"bold {score_style}"),
        )
    # Size the terminal to the table so nothing wraps and there's no dead space
    wide = recording_console(300)
    width = Measurement.get(wide, wide.options, table).maximum
    console = recording_console(width)
    console.print(
        "[dim]>>>[/dim] arr = CVDArray(vulns); arr.transform()   "
        "[dim]# analytics derived from event order only[/dim]\n"
    )
    console.print(table)
    console.save_svg(str(ASSETS / "lifecycle.svg"), title="vulnstate · lifecycle analytics")


# ------------------------------------------------------------------- triage.svg


def make_triage_svg(arr: CVDArray) -> None:
    """Real 2023 data: NVD + EPSS + KEV -> patch-priority tiers."""
    console = recording_console()
    console.print("[dim]>>>[/dim] arr.import_nvd(...); arr.import_epss(...); arr.import_kev(...)")

    tier1 = arr.kev | arr.is_under_attack
    tier2 = ~tier1 & ((arr.epss_percentile > 0.95) | (arr.is_weaponized & (arr.cvss_scores >= 9.0)))
    tier3 = ~tier1 & ~tier2 & (arr.event_age_days(CVDEvent.P) > 90) & (arr.epss_percentile > 0.5)
    rest = ~tier1 & ~tier2 & ~tier3

    console.print(
        f"[bold]{len(arr)}[/bold] CVEs (NVD 2023)  ·  "
        f"[bold]{int(arr.kev.sum())}[/bold] in CISA KEV  ·  "
        f"[bold]{int(arr.is_weaponized.sum())}[/bold] with public exploit  ·  "
        f"[bold red]{int(arr.is_zero_day.sum())}[/bold red] zero-days\n"
    )

    tiers = Table(box=box.SIMPLE_HEAVY, header_style="bold cyan", title="Patch priority")
    tiers.add_column("Tier")
    tiers.add_column("Rule")
    tiers.add_column("CVEs", justify="right")
    for name, style, rule, mask in [
        ("1 · IMMEDIATE", "bold red", "KEV or attacks observed", tier1),
        ("2 · URGENT", "bold yellow", "EPSS pct > 95% or weaponized & CVSS ≥ 9", tier2),
        ("3 · AGING", "bold blue", "public > 90 days & EPSS pct > 50%", tier3),
        ("4 · BACKLOG", "dim", "everything else", rest),
    ]:
        tiers.add_row(Text(name, style=style), rule, str(int(mask.sum())))
    console.print(tiers)

    idx = np.where(tier1)[0]
    idx = idx[np.argsort(-arr.epss[idx])][:8]
    top = Table(box=box.ROUNDED, header_style="bold cyan", title="Tier 1, ranked by EPSS")
    for col in ["CVE", "CVSS", "EPSS", "State", "Zero-day", "Public → KEV"]:
        top.add_column(col, justify="left" if col == "CVE" else "center")
    for i in idx:
        lag = (arr.A_timestamps[i] - arr.P_timestamps[i]).astype("timedelta64[h]").astype(int) / 24
        top.add_row(
            Text(str(arr.cve_ids[i]), style="bold"),
            f"{arr.cvss_scores[i]:.1f}",
            f"{arr.epss[i]:.2f}",
            Text(str(arr.states[i]), style="magenta"),
            yes_no(bool(arr.is_zero_day[i]), bad_when_true=True),
            f"{lag:+.0f} d",
        )
    console.print(top)
    console.save_svg(
        str(ASSETS / "triage.svg"), title="vulnstate · triage on real NVD/EPSS/KEV data"
    )


# ---------------------------------------------------------- fix_threat_*.png

# Reference palette: sequential blue ramp + light/dark chart chrome
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "empty": "#f0efec",
        "ramp": BLUE_RAMP,
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "empty": "#262624",
        "ramp": BLUE_RAMP[::-1],
    },
}


def make_heatmap(arr: CVDArray) -> None:
    """Where the 770 CVEs sit on the two CVD axes: remediation progress vs threat."""
    fix_paths = [FixPath.VENDOR_AWARE, FixPath.FIX_READY, FixPath.REMEDIATED]
    threats = [
        ThreatState.DISCLOSED,
        ThreatState.WEAPONIZED,
        ThreatState.UNDER_ATTACK,
        ThreatState.ACTIVE_THREAT,
    ]
    counts = np.array(
        [
            [int(((arr.fix_path == f) & (arr.threat_state == t)).sum()) for t in threats]
            for f in fix_paths
        ]
    )
    row_labels = ["Vendor aware\n(no fix)", "Fix ready\n(not deployed)", "Fix deployed"]
    col_labels = ["Disclosed", "Exploit public", "Attacked\n(private)", "Exploit +\nattacks"]

    for mode, t in THEMES.items():
        cmap = LinearSegmentedColormap.from_list("seq", t["ramp"])
        fig, ax = plt.subplots(figsize=(8.4, 4.2), dpi=200)
        fig.patch.set_facecolor(t["surface"])
        ax.set_facecolor(t["surface"])
        vmax = np.log1p(counts.max())
        for r in range(counts.shape[0]):
            for c in range(counts.shape[1]):
                n = counts[r, c]
                frac = np.log1p(n) / vmax
                color = t["empty"] if n == 0 else cmap(0.15 + 0.85 * frac)
                ax.add_patch(
                    plt.Rectangle((c + 0.03, r + 0.04), 0.94, 0.92, color=color, linewidth=0)
                )
                if n == 0:
                    text_color = t["muted"]
                else:
                    red, green, blue = to_rgb(color)
                    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                    text_color = "#ffffff" if luminance < 0.45 else "#0b0b0b"
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    f"{n}",
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color=text_color,
                )
        ax.set_xlim(0, counts.shape[1])
        ax.set_ylim(counts.shape[0], 0)
        ax.set_xticks(np.arange(counts.shape[1]) + 0.5, col_labels, color=t["ink2"], fontsize=9)
        ax.set_yticks(np.arange(counts.shape[0]) + 0.5, row_labels, color=t["ink2"], fontsize=9)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Threat state (P → X → A)", color=t["muted"], fontsize=9, labelpad=8)
        ax.set_ylabel("Fix path (V → F → D)", color=t["muted"], fontsize=9, labelpad=8)
        fig.suptitle(
            f"{len(arr)} real 2023 CVEs by CVD state",
            x=0.02,
            ha="left",
            color=t["ink"],
            fontsize=12,
            fontweight="bold",
        )
        fig.text(
            0.02,
            0.015,
            "Source: NVD 2023 sample + EPSS + CISA KEV · computed by vulnstate",
            color=t["muted"],
            fontsize=7.5,
        )
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
        fig.savefig(ASSETS / f"fix_threat_{mode}.png", facecolor=t["surface"])
        plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_lifecycle_svg()
    arr = load_real_dataset()
    make_triage_svg(arr)
    make_heatmap(arr)
    print(f"wrote assets to {ASSETS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
