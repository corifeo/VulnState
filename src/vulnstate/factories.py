"""Factory methods for CVDArray.

Provides:
- zeros: Create fixed-size array of vulnerabilities in initial state
- ones: Create fixed-size array of vulnerabilities in terminal state
- random: Create fixed-size array with random valid states
- generate: Generate realistic sample dataset

Layer: Factory
Dependencies: constants.py, vulnerability.py
Used by: array.py
"""

from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .array import CVDArray

from .constants import (
    CVDEvent,
    get_all_valid_states,
    string_to_state_int,
)
from .vulnerability import CVDVulnerability


def create_zeros(n: int, vuln_id_prefix: Optional[str] = None) -> "CVDArray":
    """Create fixed-size array of n vulnerabilities in initial state (vfdpxa).

    All vulnerabilities start in the initial 'vfdpxa' state with no events applied.

    Args:
        n: Number of vulnerabilities to create (fixed capacity)
        vuln_id_prefix: Optional prefix for auto-generated CVE IDs

    Returns:
        Fixed-size CVDArray with n vulnerabilities in vfdpxa state
    """
    from .array import CVDArray

    vulns = []
    for i in range(n):
        cve_id = f"{vuln_id_prefix}-{i:05d}" if vuln_id_prefix else None
        vuln = CVDVulnerability(cve_id=cve_id)
        vulns.append(vuln)
    return CVDArray(vulns, fixed_size=True)


def create_ones(n: int, vuln_id_prefix: Optional[str] = None) -> "CVDArray":
    """Create fixed-size array of n vulnerabilities in terminal state (VFDPXA).

    All vulnerabilities start with all events announced (VFDPXA state).

    Args:
        n: Number of vulnerabilities to create (fixed capacity)
        vuln_id_prefix: Optional prefix for auto-generated CVE IDs

    Returns:
        Fixed-size CVDArray with n vulnerabilities in VFDPXA state
    """
    from .array import CVDArray

    vulns = []
    for i in range(n):
        cve_id = f"{vuln_id_prefix}-{i:05d}" if vuln_id_prefix else None
        vuln = CVDVulnerability(
            cve_id=cve_id,
            vendor_aware=True,
            fix_aware=True,
            deployed_aware=True,
            public_aware=True,
            exploit_aware=True,
            attack_aware=True,
            create_timestamps=False,
        )
        vulns.append(vuln)
    return CVDArray(vulns, fixed_size=True)


def create_random(
    n: int, vuln_id_prefix: Optional[str] = None, seed: Optional[int] = None
) -> "CVDArray":
    """Create fixed-size array of n vulnerabilities with random valid states.

    Each vulnerability is assigned a random valid state from the 32 possible CVD states.

    Args:
        n: Number of vulnerabilities to create (fixed capacity)
        vuln_id_prefix: Optional prefix for auto-generated CVE IDs
        seed: Optional random seed for reproducibility

    Returns:
        Fixed-size CVDArray with n vulnerabilities in random valid states
    """
    from .array import CVDArray

    if seed is not None:
        np.random.seed(seed)

    valid_states = get_all_valid_states()
    vulns = []
    for i in range(n):
        cve_id = f"{vuln_id_prefix}-{i:05d}" if vuln_id_prefix else None
        random_state_str = np.random.choice(valid_states)
        state_int = string_to_state_int(random_state_str)
        vuln = CVDVulnerability(
            cve_id=cve_id,
            vendor_aware=bool(state_int & (1 << CVDEvent.V)),
            fix_aware=bool(state_int & (1 << CVDEvent.F)),
            deployed_aware=bool(state_int & (1 << CVDEvent.D)),
            public_aware=bool(state_int & (1 << CVDEvent.P)),
            exploit_aware=bool(state_int & (1 << CVDEvent.X)),
            attack_aware=bool(state_int & (1 << CVDEvent.A)),
            create_timestamps=False,
        )
        vulns.append(vuln)
    return CVDArray(vulns, fixed_size=True)


def generate(
    size: int,
    event_probs: Optional[dict[CVDEvent, float]] = None,
    cvss_range: tuple[float, float] = (3.0, 10.0),
    vendors: Optional[list[str]] = None,
    seed: Optional[int] = None,
) -> "CVDArray":
    """Generate a realistic sample dataset with configurable distributions.

    Creates vulnerabilities with randomized events, CVSS scores, and vendor
    assignments based on the provided probability distributions.

    Args:
        size: Number of vulnerabilities to generate
        event_probs: Probability of each event occurring. Defaults:
            V=1.0, F=0.7, D=0.4, P=0.5, X=0.2, A=0.1
        cvss_range: Min/max CVSS base score range (default: 3.0-10.0)
        vendors: List of vendor names to assign
        seed: Optional random seed for reproducibility

    Returns:
        CVDArray with generated vulnerabilities (not fixed-size)
    """
    import random as rand_mod

    from .array import CVDArray

    if seed is not None:
        np.random.seed(seed)
        rand_mod.seed(seed)

    default_probs: dict[CVDEvent, float] = {
        CVDEvent.V: 1.0,
        CVDEvent.F: 0.7,
        CVDEvent.D: 0.4,
        CVDEvent.P: 0.5,
        CVDEvent.X: 0.2,
        CVDEvent.A: 0.1,
    }
    if event_probs:
        default_probs.update(event_probs)
    probs = default_probs

    if vendors is None:
        vendors = ["VendorA", "VendorB", "VendorC", "VendorD", "VendorE"]

    cvss_min, cvss_max = cvss_range

    vulns: list[CVDVulnerability] = []
    for i in range(size):
        cve_id = f"CVE-2024-{i:05d}"
        vendor = vendors[i % len(vendors)]
        cvss = round(rand_mod.uniform(cvss_min, cvss_max), 1)

        vuln = CVDVulnerability(
            cve_id=cve_id,
            vendor=vendor,
            cvss_score=cvss,
        )

        base = datetime(2024, 1, 1) + timedelta(days=i % 365)

        if rand_mod.random() < probs[CVDEvent.V]:
            vuln.apply_event(CVDEvent.V, timestamp=base)

            if rand_mod.random() < probs[CVDEvent.F]:
                vuln.apply_event(
                    CVDEvent.F,
                    timestamp=base + timedelta(days=rand_mod.randint(5, 30)),
                )

                if rand_mod.random() < probs[CVDEvent.D]:
                    vuln.apply_event(
                        CVDEvent.D,
                        timestamp=base + timedelta(days=rand_mod.randint(20, 60)),
                    )

        if rand_mod.random() < probs[CVDEvent.P]:
            vuln.apply_event(
                CVDEvent.P,
                timestamp=base + timedelta(days=rand_mod.randint(1, 45)),
            )

        if rand_mod.random() < probs[CVDEvent.X]:
            vuln.apply_event(
                CVDEvent.X,
                timestamp=base + timedelta(days=rand_mod.randint(0, 30)),
            )

        if rand_mod.random() < probs[CVDEvent.A]:
            vuln.apply_event(
                CVDEvent.A,
                timestamp=base + timedelta(days=rand_mod.randint(5, 45)),
            )

        vulns.append(vuln)

    return CVDArray(vulns)
