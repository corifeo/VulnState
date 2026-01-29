"""
Vulnstate - CVD lifecycle tracking library

Based on SEI/CMU Coordinated Vulnerability Disclosure (CVD) model.

Core Classes:
    CVDVulnerability: Single vulnerability with state machine
    CVDArray: Batch container for vectorized operations
    CVDEvent: IntEnum of 6 CVD events (V, F, D, P, X, A)

Analytics:
    DesiderataExtractor: Compute fix_path, threat_state, validity, desiderata
    AnalysisResult: All computed analytics from analyze()

I/O:
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

from .array import CVDArray
from .constants import (
    DESIDERATA_PAIRS,
    VALID_HISTORIES,
    AntiDesiderataBit,
    ArrayFullError,
    CVDEvent,
    DesiderataBit,
    FixPath,
    HistoryValidity,
    ThreatState,
    TransformNotRunError,
)
from .formatting import CVDFormatter
from .lifecycle import (
    LifecycleNamespace,
    ScalarLifecycle,
    VectorLifecycle,
)
from .models import (
    AnalysisResult,
    ArraySource,
    CVSSScore,
    CWEEntry,
    EPSSScore,
    ExploitReference,
    KEVEntry,
    ScoreResult,
)
from .transforms import (
    DesiderataExtractor,
    ScoreExtractor,
    Transform,
)
from .vulnerability import TIMESTAMP_UNKNOWN, CVDVulnerability

__version__ = "0.2.0"

__all__ = [
    # Core classes
    "CVDVulnerability",
    "CVDArray",
    "CVDEvent",
    "ScalarLifecycle",
    "VectorLifecycle",
    "LifecycleNamespace",
    # Analytics
    "DesiderataExtractor",
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
    "VALID_HISTORIES",
    "TIMESTAMP_UNKNOWN",
    # Exceptions
    "TransformNotRunError",
    "ArrayFullError",
    # Model classes
    "CVSSScore",
    "EPSSScore",
    "CWEEntry",
    "KEVEntry",
    "ExploitReference",
    "ScoreResult",
    "ArraySource",
    # Transform classes
    "Transform",
    "ScoreExtractor",
    "DesiderataExtractor",
    # Version
    "__version__",
]
