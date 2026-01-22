<p align="center">
  <h1 align="center">vulnstate</h1>
  <p align="center">
    Track vulnerability disclosure lifecycles at scale.
  </p>
</p>

<p align="center">
  <a href="#installation">Installation</a> &middot;
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#batch-operations">Batch Operations</a> &middot;
  <a href="#analytics">Analytics</a> &middot;
  <a href="#data-import">Data Import</a> &middot;
  <a href="#api-reference">API Reference</a>
</p>

---

When a vulnerability is discovered, a lot happens before it's actually fixed: a vendor gets notified, a patch gets written, the public finds out, maybe an exploit drops, maybe attackers show up. The ordering of those events matters. Did the vendor know before the exploit went public? Was a fix available before attacks started? These questions determine whether a disclosure was coordinated, chaotic, or somewhere in between.

**vulnstate** models this as a state machine. Each vulnerability tracks six events (Vendor aware, Fix ready, Deployed, Public, eXploit, Attack) and the library computes what the ordering means: zero-day status, coordination quality, fix effectiveness, and 12 desiderata metrics from the CVD literature.

It's built for security researchers, PSIRT teams, and anyone analyzing vulnerability disclosure data at scale. You can track a single CVE through its lifecycle, or load 100k+ records from NVD/EPSS/KEV and run vectorized analytics across the whole dataset.

Based on the SEI/CMU Coordinated Vulnerability Disclosure model by Householder & Spring (CMU/SEI-2021-SR-021).

## The Model

At its core, vulnstate is a state machine. Every vulnerability sits in one of 32 valid states, defined by which of six events have occurred:

| Event | Meaning | Constraint |
|-------|---------|------------|
| **V** | Vendor becomes aware | Must precede F |
| **F** | Fix is ready | Must precede D |
| **D** | Fix is deployed | Requires F |
| **P** | Public becomes aware | Independent |
| **X** | Exploit becomes public | Independent |
| **A** | Attacks are observed | Independent |

The state is encoded as a 6-bit integer where each bit represents whether an event has occurred. The key constraint is **V->F->D**: you can't have a fix without vendor awareness, and you can't deploy what doesn't exist. The other three events (P, X, A) can happen in any order relative to everything else, which is what makes real-world disclosure messy and interesting to study.

A state like `VFdPxa` reads as: vendor is aware (V), fix is ready (F), not yet deployed (d), public knows (P), no exploit yet (x), no attacks (a). Uppercase = occurred, lowercase = not yet.

Transitions happen by applying events. The library enforces the V->F->D constraint (trying to apply F before V raises an error), but otherwise events can arrive in any order. This matches reality: an exploit can drop before the vendor even knows, or attacks can start before anyone goes public.

The 32 valid states (out of 64 possible) form the space of all reachable disclosure situations. Analytics are then computed from the *ordering* of event timestamps: did V happen before X? Did F arrive before A? The 12 desiderata from the CVD literature define the ideal orderings, and the library scores each vulnerability against them.

## Installation

```bash
pip install vulnstate
```

## Quick Start

```python
from vulnstate import CVDVulnerability, CVDEvent
from datetime import datetime

vuln = CVDVulnerability('CVE-2024-001', cvss_score=9.1)

vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))   # Vendor aware
vuln.apply_event(CVDEvent.F, timestamp=datetime(2024, 1, 8))   # Fix ready
vuln.apply_event(CVDEvent.P, timestamp=datetime(2024, 2, 1))   # Public aware

vuln.state_str       # 'VFdPxa'
vuln.state_label     # 'VFP'
vuln.fix_path        # FixPath.FIX_READY
vuln.threat_state    # ThreatState.DISCLOSED

# Analytical properties (lazy, cached)
vuln.is_zero_day              # False
vuln.is_coordinated           # True (V before P)
vuln.is_responsible_disclosure  # True (V->F->P maintained)
```

### Serialization

```python
# Dict / JSON round-trip
data = vuln.to_dict(include_computed=True)
vuln = CVDVulnerability.from_dict(data)

json_str = vuln.to_json()
vuln = CVDVulnerability.from_json(json_str)

# File I/O
vuln.save_json('vuln.json')
vuln = CVDVulnerability.load_json('vuln.json')
```

## Batch Operations

```python
from vulnstate import CVDArray, CVDEvent
import numpy as np

# Factory methods
arr = CVDArray.zeros(1000)              # All initial state
arr = CVDArray.ones(1000)               # All terminal state
arr = CVDArray.random(1000, seed=42)    # Random valid states
arr = CVDArray.generate(1000, seed=42)  # Realistic timelines + CVSS

# From existing vulnerabilities
arr = CVDArray([vuln1, vuln2, vuln3])

# Numpy-style interface
arr.shape       # (1000,)
arr.size        # 1000
arr.state_ints  # uint8 bitmask array

# Filtering
mask = arr.cve_ids == 'CVE-2024-001'
subset = arr[mask]

target_cves = ['CVE-2024-001', 'CVE-2024-002']
subset = arr[np.isin(arr.cve_ids, target_cves)]

# Vectorized operations
arr.apply_event_batch(CVDEvent.V)              # Apply to all eligible
applied = arr.apply_event_batch(CVDEvent.F)    # Returns bool mask of updated

# State queries
arr.states              # String array: ['VFdpxa', 'VFDPXA', ...]
arr.state_labels        # Announced events: ['VF', 'VFDPXA', ...]
arr.count_by_state()    # {'VFdpxa': 42, 'VFDPXA': 15, ...}
arr.count_by_fix_path() # {FixPath.VENDOR_AWARE: 30, ...}
```

### Timestamps

```python
# Per-event timestamp arrays (datetime64[us], NaT if not occurred)
arr.V_timestamps    # Vendor awareness
arr.F_timestamps    # Fix ready
arr.D_timestamps    # Deployment
arr.P_timestamps    # Public disclosure
arr.X_timestamps    # Exploit public
arr.A_timestamps    # Attack observed

# Dict-style access (matches CVDVulnerability.events pattern)
arr.events[CVDEvent.V]  # Same as arr.V_timestamps
```

### Scoring & Enrichment

```python
# CVSS (float32 arrays, lazy-parsed metrics)
arr.cvss_scores           # 0.0-10.0
arr.attack_vector         # N/A/L/P
arr.attack_complexity     # L/H
arr.privileges_required   # N/L/H
arr.confidentiality_impact  # N/L/H

# EPSS & KEV
arr.epss              # Exploitation probability (0.0-1.0)
arr.epss_percentile   # Percentile rank (0.0-1.0)
arr.kev               # CISA KEV membership (bool)
arr.kev_dates         # Date added to KEV catalog
```

## Analytics

Once you have events tracked, vulnstate can answer the important questions: was this a zero-day? Did the vendor get a head start? Was the fix out before attackers showed up? All of this is derived from event ordering, no manual labeling needed.

All properties work on both `CVDVulnerability` (returns a scalar) and `CVDArray` (returns bool/float numpy arrays for vectorized filtering):

```python
# Zero-day indicators
arr.is_zero_day              # X or A before V
arr.is_zero_day_exploit      # X before V
arr.is_zero_day_attack       # A before V

# Coordination quality
arr.is_coordinated           # V before P
arr.is_responsible_disclosure  # V->F->P ordering
arr.is_premature_disclosure  # P before F

# Fix effectiveness
arr.has_fix_before_exploit
arr.has_fix_before_attack
arr.has_deployment_before_exploit
arr.has_deployment_before_attack

# Threat characteristics
arr.is_weaponized            # X occurred
arr.is_under_attack          # A occurred
arr.is_private_attack        # A without X (targeted)
arr.is_mass_exploitation     # X and A both (widespread)
```

### Metrics

```python
# Temporal metrics (float32, NaN if events missing)
arr.disclosure_window_days   # days between V and P
arr.fix_lag_days             # days between V and F
arr.deployment_lag_days      # days between F and D

# Desiderata scoring
arr.desiderata_score         # 0.0-1.0 (fraction of 12 ideal orderings satisfied)
arr.skill_score              # 0.0-1.0 (weighted stakeholder skill)
arr.violated_orderings_count # 0-12

# Dimensions
arr.fix_path      # uint8: NO_AWARENESS / VENDOR_AWARE / FIX_READY / REMEDIATED
arr.threat_state  # uint8: LATENT / DISCLOSED / WEAPONIZED / ACTIVE_ATTACK
arr.severities    # CRITICAL / HIGH / MEDIUM / LOW / NONE
```

### Composing Filters

```python
# Complex prioritization query
high_priority = arr[
    arr.is_zero_day_exploit |
    arr.is_mass_exploitation |
    (arr.kev & ~arr.has_fix_before_exploit)
]

# Desiderata-based filtering
from vulnstate.constants import DesiderataBit, AntiDesiderataBit

coordinated = arr[arr.where_desiderata_satisfied(DesiderataBit.D1_V_P)]
zero_days = arr[arr.where_desiderata_violated(AntiDesiderataBit.U2_X_V)]
```

## Data Import

Real-world CVD analysis usually means stitching together data from multiple sources: NVD for disclosure dates, vendor advisories for patch timelines, EPSS for exploitation likelihood, KEV for known-exploited status. vulnstate handles the merging. Import each source and it matches by CVE ID, applying the right events with their timestamps.

```python
arr = CVDArray()

# NVD (JSON 1.1/2.0 format)
arr.import_nvd('nvdcve-1.1-2024.json')
arr.import_nvd_glob('nvdcve-*.json')

# EPSS & KEV enrichment
arr.import_epss('epss_scores.csv')
arr.import_kev('known_exploited.csv')   # Also applies event A

# Generic CSV/JSON with any CVD event
arr.import_csv(
    source='vendor_patches.csv',
    cve_column='cve_id',
    event=CVDEvent.F,
    timestamp_column='patch_date',
)

arr.import_json(
    source='threat_intel.json',
    cve_field='vulnerability.cve_id',   # Dot notation for nested fields
    event=CVDEvent.A,
    timestamp_field='threat_intel.first_observed',
)

# Export
df = arr.to_dataframe(include_analytics=True)
arr.to_json_batch('output.json', include_computed=True)
```

## API Reference

### State Model

```
Constraint: V -> F -> D (mandatory ordering)
Independent: P, X, A (no ordering constraints)

Events:
  V  Vendor Awareness     (must precede F)
  F  Fix Ready            (must precede D)
  D  Fix Deployed
  P  Public Awareness
  X  Exploit Public
  A  Attacks Observed

32 valid states (out of 64 possible 6-bit combinations)
```

### Module Layout

| Module | Purpose |
|--------|---------|
| `vulnerability.py` | `CVDVulnerability` - single instance state machine |
| `array.py` | `CVDArray` - vectorized batch container |
| `analyzer.py` | `CVDAnalyzer` - computed analytics engine |
| `constants.py` | `CVDEvent`, state encoding, desiderata definitions |
| `io.py` | `CVDIO` - NVD/EPSS/KEV import, serialization |
| `parsers.py` | `NVDParser` - NVD JSON parsing (1.1 and 2.0) |
| `models.py` | `AnalysisResult`, array dataclasses |

## Examples

```bash
uv run python examples/01_getting_started.py
uv run python examples/02_batch_operations.py
uv run python examples/03_risk_analysis.py
uv run python examples/04_analysis_inference.py
uv run python examples/05_data_import.py
```

## Development

```bash
uv sync                              # Install dependencies
uv run pytest tests/ -v              # Run tests (493 passing)
uv run mypy src/ --strict            # Type checking
uv run ruff check src/ tests/        # Linting
```

## License

MIT
