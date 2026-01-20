"""
Vulnstate - CVD lifecycle tracking library

Based on SEI/CMU Coordinated Vulnerability Disclosure (CVD) model.

Core Classes:
    CVDVulnerability: Single vulnerability with state machine
    CVDArray: Batch container for vectorized operations
    CVDEvent: IntEnum of 6 CVD events (V, F, D, P, X, A)

Analytics:
    CVDAnalyzer: Compute fix_path, threat_state, validity, desiderata
    AnalysisResult: All computed analytics from analyze()

I/O:
    CVDIO: Dict, JSON, file import/export
    CVDFormatter: Rich console formatting

Enums:
    HistoryValidity: History validation states
    FixPath: V/F/D progression states
    ThreatState: P/X/A threat states
    DesiderataBit: 12 ideal event orderings

Example:
    >>> from vulnstate import CVDVulnerability, CVDEvent
    >>> vuln = CVDVulnerability("CVE-2024-1234")
    >>> vuln.apply_event(CVDEvent.V)
    >>> print(vuln.state)  # 'Vfdpxa'

Advanced Usage:
    For low-level state conversion functions, import from constants:
    >>> from vulnstate.constants import string_to_state_int
"""

from .analyzer import CVDAnalyzer
from .array import CVDArray
from .constants import (
    DESIDERATA_PAIRS,
    AntiDesiderataBit,
    CVDEvent,
    DesiderataBit,
    FixPath,
    HistoryValidity,
    ThreatState,
)
from .formatting import CVDFormatter
from .models import AnalysisResult
from .vulnerability import TIMESTAMP_UNKNOWN, CVDVulnerability

__version__ = "0.2.0"

__all__ = [
    # Core classes
    "CVDVulnerability",
    "CVDArray",
    "CVDEvent",
    # Analytics
    "CVDAnalyzer",
    "AnalysisResult",
    # I/O
    "CVDFormatter",
    # Enums
    "HistoryValidity",
    "FixPath",
    "ThreatState",
    "DesiderataBit",
    "AntiDesiderataBit",
    # Constants
    "DESIDERATA_PAIRS",
    "TIMESTAMP_UNKNOWN",
    # Version
    "__version__",
]
