# CLAUDE.md

This document provides guidance for AI assistants working with the vulnstate codebase.

## Project Overview

**vulnstate** is a Python library for tracking and analyzing vulnerability disclosure lifecycles at scale. It implements the SEI/CMU Coordinated Vulnerability Disclosure (CVD) state machine model, enabling analysis of how vulnerabilities move through their lifecycle from discovery to remediation.

The library is evolving toward an **ETL pipeline architecture** for vulnerability data processing: Extract from multiple sources (NVD, EPSS, KEV), Transform via analytics and enrichment, Load into various output formats.

**Version:** 0.2.0 (Alpha)
**Python:** 3.9 - 3.13
**License:** MIT

### Core Problem

Vulnerability disclosure involves six key events whose ordering determines disclosure quality:
- **V** - Vendor becomes aware
- **F** - Fix is ready
- **D** - Fix is deployed
- **P** - Public becomes aware
- **X** - Exploit becomes public
- **A** - Attacks are observed

The library computes what these orderings mean: Was it a zero-day? Was disclosure coordinated? Was a fix available before attacks started?

## Directory Structure

```
vulnstate/
├── src/vulnstate/           # Main source code
│   ├── __init__.py          # Public API exports
│   ├── vulnerability.py     # CVDVulnerability - single instance
│   ├── array.py             # CVDArray - vectorized batch container
│   ├── lifecycle.py         # CVDLifecycle - state machine (single source of truth)
│   ├── constants.py         # CVDEvent enum, state encoding, desiderata
│   ├── models.py            # Data models (CVSSScore, EPSSScore, etc.)
│   ├── io.py                # NVD/EPSS/KEV import, serialization
│   ├── parsers.py           # NVD JSON parsing (1.1 and 2.0 formats)
│   ├── formatting.py        # Rich console output formatting
│   ├── factories.py         # Factory methods for array creation
│   └── transforms/          # ETL transform stage
│       ├── __init__.py      # Transform protocol definition
│       ├── desiderata.py    # DesiderataExtractor - CVD analytics
│       ├── enrichers.py     # EPSS/KEV enrichment (external data)
│       └── extractors.py    # CVSS/CPE extraction
├── tests/                   # Test suite (~688 tests)
│   ├── conftest.py          # Shared fixtures
│   ├── unit/                # Component-level tests
│   ├── integration/         # Multi-component workflows
│   ├── regression/          # Bug regression tests
│   └── performance/         # Benchmarks (marked slow)
├── examples/                # Executable examples (01-05)
├── pyproject.toml           # Dependencies & tool configuration
├── README.md                # User documentation
└── CHANGELOG.md             # Version history
```

## The State Machine Model

### State Encoding

States are 6-bit integers where each bit represents an event:
```
Bit 0: V (Vendor aware)
Bit 1: F (Fix ready)
Bit 2: D (Deployed)
Bit 3: P (Public aware)
Bit 4: X (Exploit public)
Bit 5: A (Attacks observed)
```

**Critical Constraint:** V→F→D is mandatory. You cannot have a fix without vendor awareness, and cannot deploy without a fix.

**32 valid states** exist (out of 64 possible combinations).

### State String Format

States display as 6-character strings: `VFdPxa`
- **Uppercase** = event has occurred
- **Lowercase** = event has not occurred

Example: `VFdPxa` means vendor aware (V), fix ready (F), not deployed (d), public knows (P), no exploit (x), no attacks (a).

### Two-Dimensional Decomposition

States decompose into two independent dimensions:

**FixPath (VFD)** - Vendor response progression:
- `NO_AWARENESS` → `VENDOR_AWARE` → `FIX_READY` → `REMEDIATED`

**ThreatState (PXA)** - Threat materialization:
- `LATENT` → `DISCLOSED` → `WEAPONIZED` → `ACTIVE_THREAT`

## Key Classes

### CVDVulnerability (`vulnerability.py`)
Single vulnerability instance with state machine. Delegates all state to `ScalarLifecycle`.

```python
from vulnstate import CVDVulnerability, CVDEvent
from datetime import datetime

vuln = CVDVulnerability("CVE-2024-1234", cvss_score=9.1)
vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))
vuln.state_str       # 'Vfdpxa'
```

### CVDArray (`array.py`)
Vectorized batch container using NumPy columnar storage. The primary interface for ETL operations at scale.

```python
from vulnstate import CVDArray

# Extract: Load from sources
arr = CVDArray()
arr.import_nvd('nvdcve-2024.json')
arr.import_epss('epss_scores.csv')
arr.import_kev('known_exploited.csv')

# Transform: Compute analytics
# (transforms run explicitly, not lazily)

# Load: Export results
df = arr.to_dataframe(include_analytics=True)
```

### CVDLifecycle (`lifecycle.py`)
Abstract state machine with two implementations:
- `ScalarLifecycle`: Single vulnerability (pure Python)
- `VectorLifecycle`: Batch operations (NumPy arrays)

**Single Source of Truth Pattern:** All state lives in the lifecycle, not duplicated.

### Transforms (`transforms/`)
ETL transform stage implementations:
- `DesiderataExtractor`: CVD analytics (fix_path, threat_state, desiderata scores)
- `ScoreExtractor`: CVSS vector parsing
- `EPSSEnricher`/`KEVEnricher`: External data enrichment

## Development Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest tests/ -v

# Run specific test categories
uv run pytest tests/ -m unit           # Unit tests only
uv run pytest tests/ -m integration    # Integration tests
uv run pytest tests/ -m "not slow"     # Skip performance tests

# Type checking (strict mode)
uv run mypy src/ --strict

# Linting
uv run ruff check src/ tests/

# Formatting
uv run black src/ tests/

# Run examples
uv run python examples/01_getting_started.py
```

## Code Conventions

### Type Annotations
All functions require type hints (strict mypy enforcement):
```python
def apply_event(self, event: CVDEvent, timestamp: datetime | None = None) -> bool:
```

### Line Length
100 characters maximum (configured in pyproject.toml).

### Imports
Use absolute imports. Import order enforced by ruff (I rules):
1. Standard library
2. Third-party packages
3. Local modules

### Docstrings
NumPy-style docstrings for public APIs:
```python
def method(self, param: str) -> bool:
    """Short description.

    Parameters
    ----------
    param : str
        Parameter description.

    Returns
    -------
    bool
        Return value description.
    """
```

### Error Handling
- Raise `ValueError` for invalid state transitions (e.g., applying F before V)
- Use custom exceptions from `constants.py`: `TransformNotRunError`, `ArrayFullError`

## Architecture: ETL Pipeline

The library is evolving toward a proper ETL (Extract-Transform-Load) pipeline for vulnerability data:

### Extract Stage
- **Sources:** NVD JSON (1.1/2.0), EPSS CSV, KEV CSV, generic CSV/JSON
- **Entry points:** `io.py` import methods, `parsers.py` for complex formats
- **Pattern:** Each source maps to CVD events with timestamps

### Transform Stage
- **Location:** `transforms/` directory
- **Protocol:** `Transform` base class in `transforms/__init__.py`
- **Implementations:**
  - `DesiderataExtractor`: CVD analytics (fix_path, threat_state, desiderata scores)
  - `ScoreExtractor`: CVSS vector parsing
  - `EPSSEnricher`/`KEVEnricher`: External data enrichment

### Load Stage
- **Outputs:** JSON, DataFrame, pickle, dict batches
- **Methods:** `to_json()`, `to_dataframe()`, `to_dict()`

### Core Design Principles

**Single Source of Truth:** `CVDLifecycle` owns all state. `CVDVulnerability` and `CVDArray` delegate, never duplicate.

**Delegation Pattern:** `CVDVulnerability` delegates to `ScalarLifecycle`, `CVDArray` delegates to `VectorLifecycle`.

**Explicit over Implicit:** Prefer explicit transform/compute calls over magic property access. The codebase is moving away from lazy evaluation patterns.

## Testing Patterns

### Test Organization
- `unit/`: Single component tests
- `integration/`: Multi-component workflows
- `regression/`: Specific bug fixes
- `performance/`: Benchmarks (marked `@pytest.mark.slow`)

### Fixtures
Common fixtures in `tests/conftest.py`:
- Sample vulnerabilities
- Pre-built arrays
- NVD test data

### Parametrized Tests
Use `@pytest.mark.parametrize` for testing multiple states/scenarios:
```python
@pytest.mark.parametrize("state,expected", [
    ("Vfdpxa", True),
    ("vfdPxa", False),
])
def test_is_coordinated(state, expected):
    ...
```

## Common Tasks

### Adding a New Data Source (Extract)
1. Add parser in `parsers.py` if format is complex
2. Add import method in `io.py`
3. Map source fields to CVD events with timestamps
4. Add convenience wrapper in `CVDArray`
5. Add tests in `tests/integration/test_io.py`

### Adding a New Transform
1. Create new class implementing `Transform` protocol in `transforms/`
2. Define `extract()` method that computes derived data
3. Register in `transforms/__init__.py` exports
4. Add tests in `tests/unit/test_transforms.py`

### Adding a New Analytics Metric
1. Add computation in `transforms/desiderata.py`
2. Expose via `DesiderataExtractor`
3. Add accessor in `CVDVulnerability` and `CVDArray`
4. Add tests in `tests/unit/test_transforms.py`

### Adding a New Output Format (Load)
1. Add serialization method in `io.py`
2. Add convenience wrapper in `CVDArray`
3. Add tests in `tests/integration/test_io.py`

## Important Notes

### State Transition Validation
Attempting invalid transitions raises `ValueError`:
```python
vuln.apply_event(CVDEvent.F)  # Raises: V must precede F
```

### Timestamp Semantics
- Events without timestamps use `None` or `NaT` (numpy)
- Analytics like `disclosure_window_days` require timestamps to compute
- Missing timestamps return `NaN` for float metrics

### Desiderata Scoring
12 ideal orderings from CVD literature define disclosure quality:
- Score 0-12: count of satisfied orderings
- Normalized score 0.0-1.0: fraction satisfied

### Performance Considerations
- Vectorized operations (~O(1) bitmask ops) for batch queries
- Avoid loops over individual vulnerabilities when possible
- Use boolean masks for filtering, not list comprehensions

## Architectural Direction

The codebase is evolving from a state-machine-with-properties model toward a proper ETL pipeline:

**Being phased out:**
- Lazy evaluation via `@property` with caching
- Magic property access that triggers computation
- Dirty tracking for automatic sync

**Being adopted:**
- Explicit transform/compute calls
- Clear ETL stage separation (Extract → Transform → Load)
- Batch-first operations with NumPy

When adding new features, prefer explicit methods over lazy properties.

## Key Files for Understanding the Codebase

1. `constants.py`: Start here for CVDEvent enum and state encoding
2. `lifecycle.py`: Core state machine implementation
3. `vulnerability.py`: High-level API for single vulnerabilities
4. `array.py`: Batch operations API
5. `transforms/desiderata.py`: Analytics computation logic
6. `io.py`: ETL extract/load operations

## Public API Exports

All public API is exported from `__init__.py`:
```python
from vulnstate import (
    CVDVulnerability,    # Single vulnerability
    CVDArray,            # Batch container
    CVDEvent,            # Event enum (V, F, D, P, X, A)
    FixPath,             # Fix progression states
    ThreatState,         # Threat states
    DesiderataBit,       # Ideal orderings
    AnalysisResult,      # Analytics result container
)
```
