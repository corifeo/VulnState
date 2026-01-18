# Sample Datasets

Example datasets for vulnerability analysis notebooks and demonstrations.

## Files

### `sample_nvdcve_2023.json` (3.4 MB)
NVD CVE data with intelligent filtering to ensure good KEV coverage.

**Contents:** 800 CVEs
- **37 KEV vulnerabilities** (from all years in KEV catalog)
- **763 CVE-2023-*** vulnerabilities (to fill remaining slots)

**Strategy:** Includes ALL CVEs from the KEV catalog that exist in the NVD 2023 feed, then fills remaining slots with 2023 CVEs. This ensures realistic KEV coverage for security analysis.

### `sample_kev.csv` (336 KB)
Full CISA Known Exploited Vulnerabilities (KEV) catalog.

**Contents:** 952 entries (complete catalog, not filtered by year)

Contains all actively exploited vulnerabilities that federal agencies must remediate per CISA directives.

### `sample_epss.csv` (23 KB)
EPSS (Exploit Prediction Scoring System) scores for the NVD sample.

**Contents:** 740 scores (92.5% match rate with NVD sample)

Probability scores (0.0-1.0) that a CVE will be exploited in the next 30 days.

## Coverage Statistics

- **800 total vulnerabilities** in NVD sample
- **37 KEV entries** (4.6% of sample) - good for triage analysis
- **740 EPSS scores** (92.5% coverage) - good for risk modeling
- **Diverse year range** - KEV entries span multiple years while maintaining 2023 focus

## Regenerating

To regenerate with different parameters:

```bash
python create_example_datasets.py
```

Edit `target_size` and `base_year` parameters in the script to adjust:
- Total number of CVEs to include
- Base year for non-KEV CVEs
