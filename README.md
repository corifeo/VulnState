# vulnstate

Python library implementing the SEI/CMU Coordinated Vulnerability Disclosure (CVD) state machine with numpy optimization for batch operations.

Based on: *"A State-Based Model for Multi-Party Coordinated Vulnerability Disclosure (MPCVD)"* by Allen Householder and Jonathan Spring, CMU/SEI-2021-SR-021

## Features

- **Faithful Implementation**: 32-state CVD model with V→F→D constraint enforcement
- **Performance**: Vectorized batch operations via numpy
- **Analytics**: Zero-day detection, coordination quality, fix effectiveness metrics
- **Type Safe**: Full type hints with mypy --strict validation
- **Well Tested**: 350+ tests with comprehensive coverage

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

# Create from list
vulns = [CVDVulnerability(f'CVE-{i}') for i in range(1000)]
arr = CVDArray(vulns)

# Or use factory methods
arr = CVDArray.zeros(1000)   # All at initial state
arr = CVDArray.random(1000, seed=42)  # Random states

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
from vulnstate import CVDAnalyzer

# Analyze batch
result = CVDAnalyzer.analyze(arr)

# Access computed metrics
print(result.is_zero_day)        # bool array
print(result.is_coordinated)     # bool array
print(result.desiderata_score)   # float array (0-1)
print(result.fix_lag_days)       # float array

# Single vulnerability
vuln = CVDVulnerability('CVE-2024-001')
vuln.apply_event(CVDEvent.X)  # Exploit first = zero-day
print(vuln.is_zero_day)  # True
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

### Convenience Properties

CVDVulnerability provides 13 cached boolean properties:

```python
# Zero-Day indicators
vuln.is_zero_day            # X or A before V
vuln.is_zero_day_exploit    # X before V
vuln.is_zero_day_attack     # A before V

# Coordination quality
vuln.is_coordinated         # V before P
vuln.is_responsible_disclosure
vuln.is_premature_disclosure

# Fix effectiveness
vuln.has_fix_before_exploit
vuln.has_fix_before_attack
vuln.has_deployment_before_exploit
vuln.has_deployment_before_attack

# Threat characteristics
vuln.is_private_attack      # A without X
vuln.is_weaponized          # X occurred
vuln.is_mass_exploitation   # X and A both
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
