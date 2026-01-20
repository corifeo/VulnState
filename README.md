# vulnstate

Python library implementing the SEI/CMU Coordinated Vulnerability Disclosure (CVD) state machine with numpy optimization for batch operations.

Based on: *"A State-Based Model for Multi-Party Coordinated Vulnerability Disclosure (MPCVD)"* by Allen Householder and Jonathan Spring, CMU/SEI-2021-SR-021

## Features

- **Faithful Implementation**: 32-state CVD model with V→F→D constraint enforcement
- **Performance**: Vectorized batch operations via numpy
- **Analytics**: Zero-day detection, coordination quality, fix effectiveness metrics
- **Type Safe**: Full type hints with mypy --strict validation
- **Well Tested**: 369 tests with comprehensive coverage

## Installation

```bash
pip install vulnstate
```

## Quick Start

### Single Vulnerability

```python
from vulnstate import CVDVulnerability, CVDEvent
from datetime import datetime

# Create vulnerability
vuln = CVDVulnerability('CVE-2024-001', cvss_score=9.1)

# Track CVD lifecycle
vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))   # Vendor aware
vuln.apply_event(CVDEvent.F, timestamp=datetime(2024, 1, 8))   # Fix ready
vuln.apply_event(CVDEvent.P, timestamp=datetime(2024, 2, 1))   # Public aware

print(vuln.state)       # 'VFdPxa'
print(vuln.state_label) # 'VFP'

# Convenience analytics (computed lazily, cached)
print(vuln.is_zero_day)       # False - V came before X/A
print(vuln.is_coordinated)    # True - V before P
```

### Batch Operations

```python
from vulnstate import CVDArray, CVDEvent
import numpy as np

# Create from list
vulns = [CVDVulnerability(f'CVE-2024-{i:03d}') for i in range(1000)]
arr = CVDArray(vulns)

# Or use factory methods
arr = CVDArray.zeros(1000)   # All at initial state
arr = CVDArray.random(1000, seed=42)  # Random states

# Filter by CVE ID (primary identifier)
mask = arr.cve_ids == 'CVE-2024-001'
critical = arr[mask]

# Filter multiple CVEs
target_cves = ['CVE-2024-001', 'CVE-2024-002', 'CVE-2024-003']
mask = np.isin(arr.cve_ids, target_cves)
selected = arr[mask]

# Vectorized queries
v_mask = arr.has_event_occurred(CVDEvent.V)
print(f"Vendor aware: {v_mask.sum()}")

# Batch operations
arr.apply_event_batch(CVDEvent.V)

# State distribution
print(arr.count_by_state())
```

### Analytics

```python
from vulnstate import CVDAnalyzer, CVDEvent

# Analyze batch with CVDAnalyzer
result = CVDAnalyzer.analyze(arr)
print(result.desiderata_score)   # float array (0-1)
print(result.fix_lag_days)       # float array

# Or use direct analytical properties on CVDArray
high_priority = arr[
    arr.is_zero_day_exploit |           # Exploit before vendor aware
    arr.is_mass_exploitation |          # Both exploit and attacks
    (arr.kev & ~arr.has_fix_before_exploit)  # KEV without fix
]

# Timestamp access (dict-style, matches CVDVulnerability.events)
v_times = arr.events[CVDEvent.V]  # Same as arr.V_timestamps
p_times = arr.events[CVDEvent.P]  # Same as arr.P_timestamps

# Advanced analytics (all available on both CVDVulnerability and CVDArray)
print(arr.is_zero_day_exploit)    # Exploit before vendor aware
print(arr.is_coordinated)         # Vendor aware before public
print(arr.is_responsible_disclosure)  # V→F→P ordering maintained
print(arr.has_fix_before_exploit) # Fix ready before exploit
print(arr.is_private_attack)      # Attacks without public exploit

# Single vulnerability - same properties
vuln = CVDVulnerability('CVE-2024-001')
vuln.apply_event(CVDEvent.X)  # Exploit first = zero-day
print(vuln.is_zero_day)       # True
print(vuln.is_zero_day_exploit)  # True
```

### Data Import

**Dynamic arrays** (grow as you import):
```python
from vulnstate import CVDArray

# Start with empty array
arr = CVDArray()

# Import NVD data - array grows to fit
arr.import_nvd('nvdcve-1.1-2024.json')
print(len(arr))  # Size depends on items in file

# Import more - continues growing
count = arr.import_nvd_glob('nvdcve-*.json')
print(f"Loaded from {count} files")
```

**Fixed-size arrays** (pre-allocated, strict capacity):
```python
# Pre-allocate 1000 slots
arr = CVDArray.zeros(1000)

# Import up to 1000 items (updates existing slots)
arr.import_nvd('nvdcve-1.1-2024.json')
print(len(arr))  # Still 1000

# Attempting to import more than capacity raises ValueError
# arr.import_nvd('huge_file_with_2000_items.json')  # ❌ ValueError
```

**Enrichment** (EPSS, KEV - only updates existing):
```python
# Works on both fixed and dynamic arrays
arr.import_epss('epss_scores.csv')  # Updates matching CVE IDs
arr.import_kev('known_exploited.csv')  # Applies event A automatically
```

**Create directly from data**:
```python
# Best for large datasets - auto-sizes to fit data
arr = CVDArray.from_nvd('nvdcve-1.1-2024.json')
```

## Architecture

### Modules

| Module | Purpose |
|--------|---------|
| `constants.py` | CVDEvent enum, state encoding, desiderata pairs |
| `vulnerability.py` | CVDVulnerability - single instance state machine |
| `array.py` | CVDArray - vectorized batch container |
| `analyzer.py` | CVDAnalyzer - computed analytics |
| `io.py` | CVDIO - generic I/O operations and serialization |
| `parsers.py` | NVDParser - NVD JSON parsing (1.1 and 2.0 formats) |
| `formatting.py` | CVDFormatter - rich console output |
| `models.py` | Data models and AnalysisResult |

### State Model

```
Mandatory constraint: V → F → D
Optional events: P, X, A (independent)

Events:
  V - Vendor Awareness    (must precede F)
  F - Fix Ready           (must precede D)
  D - Fix Deployed
  P - Public Awareness
  X - Exploit Public
  A - Attacks Observed

32 valid states (out of 64 possible)
```

### Analytical Properties

Both CVDVulnerability and CVDArray provide analytical properties for vulnerability assessment:

```python
# Zero-Day indicators
vuln.is_zero_day            # X or A before V
vuln.is_zero_day_exploit    # X before V
vuln.is_zero_day_attack     # A before V

# Coordination quality
vuln.is_coordinated         # V before P
vuln.is_responsible_disclosure  # V→F→P ordering
vuln.is_premature_disclosure    # Public before vendor aware

# Fix effectiveness
vuln.has_fix_before_exploit
vuln.has_fix_before_attack
vuln.has_deployment_before_exploit
vuln.has_deployment_before_attack

# Threat characteristics
vuln.is_private_attack      # A without X (targeted attacks)
vuln.is_weaponized          # X occurred
vuln.is_mass_exploitation   # X and A both (widespread)

# Same properties available on CVDArray (returns boolean arrays)
arr.is_zero_day_exploit     # np.ndarray[bool]
arr.is_coordinated          # np.ndarray[bool]
arr.has_fix_before_exploit  # np.ndarray[bool]
# ... all properties above
```

## Examples

```bash
uv run python examples/01_getting_started.py
uv run python examples/02_batch_operations.py
uv run python examples/03_risk_analysis.py
uv run python examples/04_analysis_inference.py
uv run python examples/05_data_import.py
```

## Testing

```bash
uv run pytest tests/ -v
```

## Citation

```bibtex
@techreport{householder2021state,
  title={A State-Based Model for Multi-Party Coordinated Vulnerability Disclosure},
  author={Householder, Allen D and Spring, Jonathan M},
  year={2021},
  institution={Software Engineering Institute, Carnegie Mellon University},
  number={CMU/SEI-2021-SR-021}
}
```

## License

MIT License
