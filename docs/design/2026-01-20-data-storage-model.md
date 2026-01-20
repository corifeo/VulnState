# Data Storage Model

**Date:** 2026-01-20
**Status:** Authoritative
**Purpose:** Document what data is stored vs calculated, and the metadata structure

## Design Principle

**Store only what you need. Calculate the rest.**

- **Dataclass properties**: Indexed, queryable, first-class citizens
- **metadata dict**: Parsed, understood fields from imports (no raw JSON blobs)
- **Computed properties**: Derived on-access from stored data

## CVDVulnerability Storage

### Dataclass Properties (Stored)

| Dataclass | Field | Type | Description |
|-----------|-------|------|-------------|
| `VulnerabilityIdentity` | `vuln_id` | `str` | Internal UUID |
| | `cve_id` | `Optional[str]` | CVE identifier |
| `VulnerabilityEventData` | `state_encoded` | `uint8` | 6-bit state bitmask |
| | `events` | `dict[CVDEvent, datetime]` | Event timestamps |
| | `history` | `list[dict]` | Transition audit log |
| `VulnerabilityScoringData` | `cvss_base_score` | `Optional[float]` | CVSS score |
| | `cve_vector` | `Optional[str]` | CVSS vector string |
| | `attack_vector`, etc. | `Optional[str]` | Parsed CVSS metrics |
| `VulnerabilityEnrichmentData` | `epss` | `Optional[float]` | EPSS score |
| | `is_kev` | `bool` | KEV flag |

### metadata Dict (Parsed Fields)

Fields stored in `vuln.metadata` from imports:

**From NVD (no prefix):**
| Key | Type | Source | Notes |
|-----|------|--------|-------|
| `cpe_strings` | `list[str]` | `configurations[].nodes[].cpeMatch[].criteria` | All CPE 2.3 strings |
| `vendors` | `list[str]` | Derived from CPE | Unique, sorted vendor IDs |
| `products` | `list[str]` | Derived from CPE | Unique, sorted product IDs |
| `description` | `str` | `descriptions[].value` | English description |
| `cwe_ids` | `list[str]` | `weaknesses[].cweId` | CWE identifiers |
| `severity` | `str` | `cvssMetricV31[].baseSeverity` | CRITICAL/HIGH/MEDIUM/LOW |
| `published_date` | `str` | `published` | ISO 8601 timestamp |
| `last_modified` | `str` | `lastModified` | ISO 8601 timestamp |

**From KEV (kev_ prefix):**
| Key | Type | Source |
|-----|------|--------|
| `kev_vendor_name` | `str` | `vendorProject` |
| `kev_product_name` | `str` | `product` |
| `kev_description` | `str` | `shortDescription` |
| `kev_ransomware_use` | `str` | `knownRansomwareCampaignUse` |
| `kev_required_action` | `str` | `requiredAction` |
| `kev_due_date` | `str` | `dueDate` |
| `kev_notes` | `str` | `notes` |
| `kev_date_added` | `str` | `dateAdded` |

**From EPSS (epss_ prefix):**
| Key | Type | Source |
|-----|------|--------|
| `epss_percentile` | `float` | `percentile` |

### Computed Properties (On-Access)

| Property | Source | Computation |
|----------|--------|-------------|
| `state` | `state_encoded` | `state_int_to_string()` lookup |
| `state_label` | `state` | Extract uppercase chars |
| `fix_path` | `state_encoded` | Bits 0-2 |
| `threat_state` | `state_encoded` | Bits 3-5 |
| `event_order` | `history` | Extract events in order |
| `is_zero_day`, etc. | `_analytics` | Cached from CVDAnalyzer |

## CVDArray Storage

### Stored Arrays

| Category | Array | dtype | Description |
|----------|-------|-------|-------------|
| **Core** | `states` | `uint8` | State bitmasks |
| | `vulnerabilities` | `object` | Live CVDVulnerability objects |
| **Timestamps** | `V_timestamps` | `datetime64[us]` | V event times |
| | `F_timestamps` | `datetime64[us]` | F event times |
| | `D_timestamps` | `datetime64[us]` | D event times |
| | `P_timestamps` | `datetime64[us]` | P event times |
| | `X_timestamps` | `datetime64[us]` | X event times |
| | `A_timestamps` | `datetime64[us]` | A event times |
| **Identity** | `vuln_ids` | `object` | Internal UUIDs |
| | `cve_ids` | `object` | CVE identifiers |
| **Scoring** | `cvss_scores` | `float32` | CVSS scores |
| | `epss` | `float32` | EPSS scores |
| | `kev` | `bool` | KEV flags |

### Cached Arrays (Computed on First Access)

| Array | dtype | Description | Invalidated |
|-------|-------|-------------|-------------|
| `pair_mask` | `uint16` | 15-bit pair ordering | On `sync()` |
| `history_id` | `uint8` | 0-69 or 255 | On `sync()` |
| `_analysis_cache` | `AnalysisResult` | Full analytics | On state change |

### Computed Properties (From Cached)

| Property | Source | Computation |
|----------|--------|-------------|
| `states` (strings) | `state_ints` | `state_int_to_string()` |
| `state_labels` | `state_ints` | `get_state_label()` |
| `is_zero_day_exploit` | `pair_mask` | Bit 3 check |
| `is_coordinated` | `pair_mask` | Bit 2 check |
| `desiderata_scores` | `pair_mask` | Bit count |

## Memory Profile

### Per Vulnerability (Core CVD Data)

| Component | Bytes | Notes |
|-----------|-------|-------|
| `state_encoded` | 1 | 6-bit bitmask |
| 6 timestamps | 48 | datetime64[us] |
| `pair_mask` | 2 | Cached |
| `history_id` | 1 | Cached |
| **Total core** | **~52** | Without metadata |

### With Typical Imports

| Scenario | Approx. Size | Notes |
|----------|--------------|-------|
| Core CVD only | 52 bytes | Minimal |
| + Identity | +200 bytes | Strings |
| + NVD metadata | +500 bytes | Parsed fields |
| + KEV metadata | +300 bytes | If in KEV |
| **Typical total** | ~1-2 KB | Per vulnerability |

## CPE-Based Vendor/Product Lists

A CVE can affect multiple vendors and products. All CPE strings are extracted and stored in metadata:

```
cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*
         │ │      │
         │ │      └── product = "log4j"
         │ └── vendor = "apache"
         └── part (a=app, o=OS, h=hardware)
```

Access via metadata:
```python
# Get all affected vendors/products for a vulnerability
vuln.metadata["vendors"]    # ['apache', 'microsoft']
vuln.metadata["products"]   # ['log4j', 'struts', 'exchange']
vuln.metadata["cpe_strings"]  # Full CPE strings

# Filter array by vendor (check if any vendor matches)
apache_mask = np.array([
    'apache' in (v.metadata.get('vendors', []))
    for v in arr._vulnerabilities
])
apache_vulns = arr[apache_mask]
```

## Key Takeaways

1. **No raw JSON blobs** - Only parsed, understood fields in metadata
2. **Lists for multi-value fields** - `vendors`, `products`, `cwe_ids` are lists (CVE can affect multiple)
3. **Namespaced metadata** - `kev_*` prefix for KEV fields, `epss_*` for EPSS
4. **Lazy computation** - Analytics cached, computed on first access
5. **Dirty tracking** - Cache invalidated on state changes via `sync()`
