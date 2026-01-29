"""Transform protocol and built-in transforms.

Provides:
- Transform: Protocol for pluggable computation units
- ScoreExtractor: Extract CVSS/EPSS/enrichment data into computed slots
- DesiderataExtractor: Compute CVD analytics (desiderata) from state

Layer: Analytics
Dependencies: models.py, constants.py
Used by: array.py, vulnerability.py
"""

from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from vulnstate.array import CVDArray
    from vulnstate.vulnerability import CVDVulnerability

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Transform(Protocol[T_co]):
    """Pluggable computation unit for CVDArray and CVDVulnerability.

    Transforms MUST:
    - Operate on entire arrays (vectorized) in apply()
    - Return numpy arrays of length N (matching array size)
    - Be stateless and idempotent

    Transforms SHOULD:
    - Compute related slots together (one pass)
    - Use numpy operations, avoid Python loops
    - Use appropriate dtypes (float32, uint8, bool)
    """

    name: str

    def apply(self, array: "CVDArray") -> dict[str, np.ndarray]:
        """Compute slots for entire array."""
        ...

    def apply_single(self, vuln: "CVDVulnerability") -> T_co:
        """Compute result for single vulnerability."""
        ...


from vulnstate.transforms.desiderata import DesiderataExtractor  # noqa: E402
from vulnstate.transforms.enrichers import (  # noqa: E402
    EPSSEnricher,
    EventInferenceTransform,
    KEVEnricher,
    infer_events,
)
from vulnstate.transforms.extractors import ScoreExtractor, parse_cpe  # noqa: E402

__all__ = [
    "Transform",
    "ScoreExtractor",
    "parse_cpe",
    "DesiderataExtractor",
    "EPSSEnricher",
    "EventInferenceTransform",
    "KEVEnricher",
    "infer_events",
]
