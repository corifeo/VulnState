# vulnstate

A modern Python library implementing the SEI/CMU Coordinated Vulnerability Disclosure (CVD) state machine with numpy optimization for batch operations and ML integration.

Based on: *"A State-Based Model for Multi-Party Coordinated Vulnerability Disclosure (MPCVD)"* by Allen Householder and Jonathan Spring, CMU/SEI-2021-SR-021

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-343%2B%20passing-brightgreen.svg)]()

## Features

- **Faithful Implementation**: 32-state CVD model from SEI/CMU paper with constraint enforcement
- **Performance**: 50-120x faster batch operations via numpy vectorization
- **ML Ready**: DataFrame export for pandas/scikit-learn integration
- **Advanced Filtering**: Pattern matching, ordering violations, time-based queries, metadata filtering
- **Backward Compatible**: Single-instance API for existing code
- **Type Safe**: Full type hints and mypy validation
- **Well Tested**: 343+ tests covering all functionality
- **Production Ready**: Professional code quality and documentation

## Installation

```bash
pip install vulnstate

# With ML examples
pip install vulnstate[demo]
```

## Quick Start

### Single Vulnerability

```python
from vulnstate import CVDEvent, CVDVulnerability
from datetime import datetime

# Create vulnerability with optional CVE ID
vuln = CVDVulnerability('CVE-2024-001', cvss=9.1)

# Track CVD process
vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))    # Vendor aware
vuln.apply_event(CVDEvent.F, timestamp=datetime(2024, 1, 8))    # Fix ready
vuln.apply_event(CVDEvent.D, timestamp=datetime(2024, 1, 15))   # Fix deployed
vuln.apply_event(CVDEvent.P, timestamp=datetime(2024, 2, 1))    # Public aware

print(vuln.state)        # 'VFDpxa'
print(vuln.state_label)  # 'VFDP' (announced events)
print(vuln.is_terminal()) # False (X and A not occurred)

# Add exploitation data
vuln.set_epss(0.85)  # Exploit Prediction Scoring System (0.0-1.0)
vuln.cve_vector = 'CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H'
vuln.is_kev = True   # CISA Known Exploited Vulnerabilities
```

### Dual ID System

```python
# New dual ID system: internal_id (auto) + cve_id (user-provided)
vuln1 = CVDVulnerability('CVE-2024-001')  # With CVE ID
vuln2 = CVDVulnerability()                 # Auto internal_id only

print(vuln1.cve_id)      # 'CVE-2024-001'
print(vuln1.internal_id) # UUID (never changes)
print(vuln1.vuln_id)     # 'CVE-2024-001' (property that returns cve_id or internal_id)

# Event control
vuln_auto_v = CVDVulnerability('CVE-2024-002', create_events=True)   # Auto-applies V
vuln_vendor = CVDVulnerability('CVE-2024-003', vendor_aware=True)    # Manually set V
```

### Batch Operations

```python
from vulnstate import CVDArray

# Create portfolio - old way
vulns = [CVDVulnerability(f'CVE-{i}') for i in range(10000)]
arr = CVDArray(vulns)

# Create portfolio - new array helpers
arr_init = CVDArray.zeros(1000, vuln_id_prefix='INIT')      # Initial state
arr_term = CVDArray.ones(1000, vuln_id_prefix='TERM')       # Terminal state
arr_rand = CVDArray.random(1000, vuln_id_prefix='RND', seed=42)  # Random states

# Vectorized queries (50x faster than loops)
v_mask = arr.has_event_occurred(CVDEvent.V)
print(f"Vendor aware: {v_mask.sum()}")

# Batch event application
arr.apply_event_batch(CVDEvent.V)

# State distribution - traditional and labeled
state_counts = arr.count_by_state()
label_dist = arr.get_state_distribution_labeled()  # New: group by announced events
print(f"Label distribution: {label_dist}")  # e.g., {'VF': 50, 'VFDPXA': 20}

# Import external data
epss_scores = {'CVE-0001': 0.85, 'CVE-0002': 0.42}
arr.import_epss(epss_scores)

kev_cves = {'CVE-0001', 'CVE-0003'}
arr.import_kev(kev_cves)
```

### Advanced Filtering

```python
from vulnstate.filtering import CVDFilter

# Complex multi-criteria queries
result = (CVDFilter.query(arr)
    .cvss_above(8.5)
    .has_events([CVDEvent.V, CVDEvent.P])
    .event_pattern('VF?P??')
    .execute())
```

### ML Integration (Deferred)

ML feature encoding has been deferred to a future release. See `docs/design/ml-encoding.md` for the planned approach using pandas DataFrames with scikit-learn pipelines.

For now, export to DataFrame and use standard ML tools:
```python
from vulnstate import CVDArray
import pandas as pd

arr = CVDArray(vulnerabilities)
df = arr.to_dataframe()  # Use with pandas, scikit-learn, etc.
```

## Architecture

### Core Modules

| Module | Purpose |
|--------|---------|
| `events.py` | 6 CVD events (V, F, D, P, X, A) |
| `states.py` | 32 valid states with V→F→D constraint |
| `core.py` | Single vulnerability API |
| `batch.py` | Vectorized numpy operations (50-120x faster) |
| `filtering.py` | Advanced filtering and QueryBuilder |
| `analysis.py` | Portfolio analysis and metrics |

### State Machine Model

The library enforces the SEI/CMU CVD state constraints:

```
Mandatory constraint: V → F → D
Optional events: P, X, A (can occur in any order)

Events:
  V - Vendor Awareness       (must be first)
  F - Fix Ready              (requires V)
  D - Fix Deployed           (requires F)
  P - Public Awareness       (independent)
  X - Exploit Public         (independent)
  A - Attacks Observed       (independent)
```

This produces exactly **32 valid states** (out of 64 possible):

```
Initial:  'vfdpxa' (no events)
Final:    'VFDPXA' (all events)
Examples: 'VFdpxa', 'VfdPXa', 'vfdPXA'
```

## Class Naming Convention

The library uses a consistent naming convention for core classes:

**Classes with `CVD` Prefix** (Core state machine):
- `CVDEvent` - 6 vulnerability disclosure events
- `CVDVulnerability` - Single vulnerability with state tracking
- `CVDArray` - Batch operations (numpy-optimized)
- `CVDAnalysis` - Portfolio-level analytics
- `CVDFilter` - Advanced filtering and queries
- `CVDFormatter` - Rich terminal output

**Classes without Prefix** (Utilities and encoding):
- `CVDState` - State string ↔ integer conversion
- `CVDTransitionProbabilities` - Transition probabilities
- `CVDPossibleHistories` - History data and frequencies
- `CVDPandasConverter` - Pandas integration

**Rationale**: The `CVD` prefix identifies the core state machine classes that directly implement the coordinated vulnerability disclosure model. Other classes provide supporting functionality and utilities.

## Performance

Vectorized operations provide dramatic speedup:

| Operation | 1K Items | 10K Items | Speedup |
|-----------|----------|-----------|---------|
| Event check | 0.5ms | 5ms | 50x |
| Apply event | 1ms | 12ms | 60x |
| Filtering | 2ms | 25ms | 80x |
| DataFrame export | 3ms | 30ms | 90x |

## Live Interactive Demo

Visualize the vulnstate in real-time with our interactive terminal demo:

```bash
# Default: 10 vulnerabilities with 2 events/second
python examples/live_demo.py

# Custom configuration
python examples/live_demo.py --num-vulns 50 --rate 3.0
```

**Demo Features:**
- Real-time state transitions in a beautiful table display
- Random event simulation respecting CVD constraints
- Live event feed with timestamps and event descriptions
- Progress tracking and completion statistics
- Color-coded states for easy visualization
- Press CTRL+C to exit and view summary statistics

This demo visually demonstrates the library's power for tracking and visualizing vulnerability portfolios!

## Examples

Run comprehensive examples:

```bash
# Basic single-instance usage
python examples/01_getting_started.py

# Batch operations and portfolio analysis
python examples/02_batch_processing.py

# Error handling and edge cases
python examples/05_error_handling.py
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=vulnstate

# 274+ tests covering all functionality
```

## API Overview

### CVDEvent
```python
CVDEvent.V  # Vendor Awareness
CVDEvent.F  # Fix Ready
CVDEvent.D  # Fix Deployed
CVDEvent.P  # Public Awareness
CVDEvent.X  # Exploit Public
CVDEvent.A  # Attacks Observed
```

### CVDVulnerability
```python
v = CVDVulnerability('CVE-2024-001', cvss=9.1, **metadata)

v.apply_event(event, timestamp=None)  # Apply event
v.has_event_occurred(event)           # Check if occurred
v.get_possible_events()               # Next possible events
v.is_terminal()                       # All events occurred?
v.to_vector()                         # Basic 6-bit state vector [V,F,D,P,X,A]
```

### CVDArray (Vectorized)
```python
arr = CVDArray(vulnerabilities)

arr[i]                              # Single item
arr[mask]                           # Boolean masking
arr.has_event_occurred(event)       # Vectorized query
arr.apply_event_batch(event)        # Batch operation
arr.count_by_state()                # State distribution
arr.to_matrix()                     # Basic state matrix (N, 6)
```

### CVDFilter
```python
# Pattern matching: uppercase=must, lowercase=must-not, ?=don't-care
CVDFilter.by_event_pattern(arr, 'VF????')

# Ordering violations
CVDFilter.by_desirable_ordering(arr, CVDEvent.V, CVDEvent.P)

# Time-based filtering
CVDFilter.by_time_delta(arr, CVDEvent.V, CVDEvent.P, min_days=30)

# Metadata filtering
CVDFilter.by_metadata(arr, 'cvss', '>', 8.0)

# Fluent query builder
result = (CVDFilter.query(arr)
    .cvss_above(8.5)
    .event_pattern('VF????')
    .execute())
```

### CVDAnalysis
```python
# Identify ordering violations
violations = CVDAnalysis.identify_ordering_violations(arr)

# Time metrics
disclosure = CVDAnalysis.calculate_disclosure_window(arr)
fix_lag = CVDAnalysis.calculate_fix_lag(arr)

# Portfolio categorization
categories = CVDAnalysis.categorize_portfolio(arr)

# Generate report
report = CVDAnalysis.generate_summary_report(arr)
```

## ML Integration (Deferred)

ML feature encoding has been deferred to a future release. See `docs/design/ml-encoding.md` for the planned approach using pandas DataFrames with scikit-learn pipelines.

For now, export to DataFrame and use standard ML tools:
```python
from vulnstate import CVDArray
import pandas as pd

arr = CVDArray(vulnerabilities)
df = arr.to_dataframe()  # Use with pandas, scikit-learn, etc.
```

## Real-World Use Cases

### Vulnerability Remediation Prioritization
```python
# Find unpatched critical vulnerabilities
critical_unpatched = (CVDFilter.query(arr)
    .cvss_above(9.0)
    .event_pattern('Vf????')  # V but not F
    .execute())
```

### Incident Response Triage
```python
# Find actively exploited vulnerabilities
active = (CVDFilter.query(arr)
    .has_events([CVDEvent.X, CVDEvent.A], all_of=True)
    .execute())
```

### Disclosure Window Analysis
```python
# Find long-delayed disclosures
long_delay = (CVDFilter.query(arr)
    .disclosure_window(min_days=90)
    .execute())
```

### Ordering Violation Detection
```python
# Find zero-days (exploitation before vendor awareness)
zero_days = CVDAnalysis.identify_zero_days(arr)
```

## Documentation

- **[API Reference](docs/api.md)** - Complete API documentation
- **[Examples](examples/)** - 4 comprehensive example files
- **[Original Paper](https://resources.sei.cmu.edu/library/report.cfm?id=20743)** - SEI/CMU CVD paper

## Design Highlights

### Constraint Enforcement
- V→F→D constraint enforced at state level
- Only 32 valid states (out of 64 possible)
- Cannot create invalid states

### Performance Optimization
- Numpy vectorization for batch operations
- Boolean indexing instead of loops
- O(1) state encoding/decoding with lookup tables
- Memory-efficient uint8 internal representation

### Type Safety
- Full type hints throughout
- Mypy validation
- Clear API contracts

## Contributing

Contributions welcome! Please ensure:
- All tests pass (`pytest tests/`)
- Type checking passes (`mypy src/`)
- Code formatted with black
- New tests for new features

## Citation

If you use this library in research, please cite the original paper:

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

MIT License - see [LICENSE](LICENSE) file for details

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/vulnstate/issues)
- **Documentation**: [Read the Docs](https://vulnstate.readthedocs.io/)

## Acknowledgments

- Original CVD model: Allen Householder and Jonathan Spring, CMU/SEI
- Built with [numpy](https://numpy.org/) for performance
- Tested with [pytest](https://pytest.org/)

---

**Ready to get started?** See [Quick Start](#quick-start) or explore [examples/](examples/).
