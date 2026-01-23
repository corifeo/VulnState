"""Generate sample_deployed.csv from sample NVD data.

Selects ~65% of CVEs with "Patch" tag and adds a realistic deployment
offset (7-90 days from lastModifiedDate). Deterministic with seed=42.
"""

import csv
import random
from datetime import datetime, timedelta

import sys
sys.path.insert(0, "src")

from vulnstate.parsers import NVDParser, _load_json


def main():
    random.seed(42)

    nvd_feed = _load_json("examples/sample_nvdcve_2023.json")
    format_version = NVDParser.detect_format(nvd_feed)
    items = nvd_feed.get("CVE_Items", [])

    deployable = []
    for item in items:
        cve_id = NVDParser.extract_cve_id(item, format_version)
        if not cve_id:
            continue

        ref_tags = NVDParser.extract_reference_tags(item, format_version)
        if "Patch" not in ref_tags:
            continue

        last_modified = NVDParser.extract_last_modified(item, format_version)
        if not last_modified:
            continue

        try:
            lm_dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            continue

        deployable.append((cve_id, lm_dt))

    # Select ~65%
    selected = random.sample(deployable, int(len(deployable) * 0.65))

    # Generate deployment dates
    rows = []
    for cve_id, lm_dt in sorted(selected):
        offset_days = random.randint(7, 90)
        deploy_date = lm_dt + timedelta(days=offset_days)
        rows.append({"cve_id": cve_id, "deployed_date": deploy_date.strftime("%Y-%m-%d")})

    # Write CSV
    with open("examples/sample_deployed.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cve_id", "deployed_date"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} deployment records from {len(deployable)} patched CVEs")


if __name__ == "__main__":
    main()
