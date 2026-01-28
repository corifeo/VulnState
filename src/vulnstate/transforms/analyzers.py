"""CVD state analysis transforms.

Provides:
- CVDStateAnalyzer: Compute CVD analytics (desiderata) from state

Layer: Analytics
Dependencies: models.py, constants.py, vulnerability.py
Used by: array.py
"""

from datetime import datetime
from typing import TYPE_CHECKING, Optional

import numpy as np

from vulnstate.constants import CVDEvent
from vulnstate.models import AnalyticsResult

if TYPE_CHECKING:
    from vulnstate.array import CVDArray
    from vulnstate.vulnerability import CVDVulnerability


class CVDStateAnalyzer:
    """Compute CVD analytics from vulnerability state.

    This transform computes the CVD desiderata (is_zero_day, is_coordinated, etc.)
    from vulnerability state. It implements the Transform protocol.

    Example:
        >>> from vulnstate.transforms import CVDStateAnalyzer
        >>> from vulnstate.vulnerability import CVDVulnerability
        >>> vuln = CVDVulnerability(cve_id="CVE-2024-1234")
        >>> analyzer = CVDStateAnalyzer()
        >>> result = analyzer.apply_single(vuln)
        >>> print(result.is_coordinated)
    """

    name: str = "analytics"

    def _get_timestamp(self, vuln: "CVDVulnerability", event: CVDEvent) -> Optional[datetime]:
        """Get timestamp for an event, or None if not occurred."""
        return vuln.events.get(event)

    def _compute_fix_lag(self, vuln: "CVDVulnerability") -> Optional[float]:
        """Compute days between fix and public awareness.

        Returns the number of days between Fix_Ready and Public_Aware.
        Positive means fix was ready before public awareness (good).
        Negative means fix came after public awareness (bad).
        """
        fix_time = self._get_timestamp(vuln, CVDEvent.F)
        public_time = self._get_timestamp(vuln, CVDEvent.P)
        if fix_time and public_time:
            delta = (public_time - fix_time).total_seconds() / 86400
            return delta
        return None

    def apply_single(self, vuln: "CVDVulnerability") -> AnalyticsResult:
        """Compute analytics for single vulnerability.

        Args:
            vuln: CVDVulnerability instance to analyze

        Returns:
            AnalyticsResult with all computed desiderata booleans
        """
        # Get timestamps
        v = self._get_timestamp(vuln, CVDEvent.V)
        f = self._get_timestamp(vuln, CVDEvent.F)
        d = self._get_timestamp(vuln, CVDEvent.D)
        p = self._get_timestamp(vuln, CVDEvent.P)
        x = self._get_timestamp(vuln, CVDEvent.X)
        a = self._get_timestamp(vuln, CVDEvent.A)

        # Compute desiderata booleans
        # Zero-day indicators: threat activity before vendor awareness
        is_zero_day = p is not None and v is not None and p < v
        is_zero_day_exploit = x is not None and v is not None and x < v
        is_zero_day_attack = a is not None and v is not None and a < v

        # Coordination quality: vendor awareness before public disclosure
        is_coordinated = v is not None and p is not None and v <= p

        # Premature disclosure: public awareness before fix ready
        is_premature_disclosure = p is not None and (f is None or p < f)

        # Responsible disclosure: V <= P and F <= P (vendor aware and fix ready before public)
        is_responsible_disclosure = (
            f is not None and p is not None and f <= p and v is not None and v <= p
        )

        # Fix effectiveness
        has_fix_before_exploit = f is not None and (x is None or f <= x)
        has_fix_before_attack = f is not None and (a is None or f <= a)

        # Threat characteristics
        is_private_attack = a is not None and p is None
        is_weaponized = x is not None
        is_mass_exploitation = x is not None and a is not None

        # State indicators
        is_fix_available = f is not None
        is_fix_deployed = d is not None
        is_under_attack = a is not None

        # Get fix path and threat state from existing vulnerability properties
        fix_path = vuln.fix_path
        threat_state = vuln.threat_state

        return AnalyticsResult(
            fix_path=fix_path,
            threat_state=threat_state,
            is_zero_day=is_zero_day,
            is_zero_day_exploit=is_zero_day_exploit,
            is_zero_day_attack=is_zero_day_attack,
            is_coordinated=is_coordinated,
            is_premature_disclosure=is_premature_disclosure,
            is_responsible_disclosure=is_responsible_disclosure,
            has_fix_before_exploit=has_fix_before_exploit,
            has_fix_before_attack=has_fix_before_attack,
            is_private_attack=is_private_attack,
            is_weaponized=is_weaponized,
            is_mass_exploitation=is_mass_exploitation,
            is_fix_available=is_fix_available,
            is_fix_deployed=is_fix_deployed,
            is_under_attack=is_under_attack,
            fix_lag_days=self._compute_fix_lag(vuln),
        )

    def apply(self, array: "CVDArray") -> dict[str, np.ndarray]:
        """Compute analytics slots for entire array (vectorized).

        Uses numpy vectorized operations on timestamp arrays for performance.
        This is 10-100x faster than the Python loop approach for large arrays.

        Args:
            array: CVDArray to analyze

        Returns:
            Dict mapping slot names to numpy arrays of computed values
        """
        n = len(array)
        if n == 0:
            return self._empty_result()

        # Get timestamp arrays directly (numpy datetime64)
        v = array.timestamps.V
        f = array.timestamps.F
        d = array.timestamps.D
        p = array.timestamps.P
        x = array.timestamps.X
        a = array.timestamps.A

        # Build "has event" masks (not NaT)
        has_v = ~np.isnat(v)
        has_f = ~np.isnat(f)
        has_d = ~np.isnat(d)
        has_p = ~np.isnat(p)
        has_x = ~np.isnat(x)
        has_a = ~np.isnat(a)

        # Zero-day indicators: threat activity before vendor awareness
        # is_zero_day: P < V (public before vendor aware)
        is_zero_day = has_p & has_v & (p < v)

        # is_zero_day_exploit: X < V (exploit before vendor aware)
        is_zero_day_exploit = has_x & has_v & (x < v)

        # is_zero_day_attack: A < V (attack before vendor aware)
        is_zero_day_attack = has_a & has_v & (a < v)

        # Coordination quality: V <= P (vendor aware before or at public disclosure)
        is_coordinated = has_v & has_p & (v <= p)

        # Premature disclosure: P and (no F or P < F)
        is_premature_disclosure = has_p & (~has_f | (p < f))

        # Responsible disclosure: V <= P and F <= P
        is_responsible_disclosure = has_f & has_p & has_v & (f <= p) & (v <= p)

        # Fix effectiveness
        # has_fix_before_exploit: F <= X (both must exist for True)
        has_fix_before_exploit = has_f & has_x & (f <= x)

        # has_fix_before_attack: F <= A (both must exist for True)
        has_fix_before_attack = has_f & has_a & (f <= a)

        # Threat characteristics
        # is_private_attack: A without X (targeted/private attack without public exploit)
        is_private_attack = has_a & ~has_x

        # is_weaponized: X occurred
        is_weaponized = has_x

        # is_mass_exploitation: X and A both occurred
        is_mass_exploitation = has_x & has_a

        # State indicators
        is_fix_available = has_f
        is_fix_deployed = has_d
        is_under_attack = has_a

        # Fix lag: days between P and F (positive = fix before public)
        fix_lag_days = np.full(n, np.nan, dtype=np.float32)
        both_have = has_f & has_p
        if np.any(both_have):
            # Convert to float64 days
            delta = p[both_have].astype("datetime64[us]").astype(np.int64) - f[both_have].astype(
                "datetime64[us]"
            ).astype(np.int64)
            fix_lag_days[both_have] = delta / (86400 * 1_000_000)  # microseconds to days

        # Fix path and threat state from state bitmask
        states = array.state.bitmask
        fix_path_arr = (states & 0b000111).astype(np.uint8)
        threat_state_arr = ((states >> 3) & 0b000111).astype(np.uint8)

        return {
            "is_zero_day": is_zero_day,
            "is_zero_day_exploit": is_zero_day_exploit,
            "is_zero_day_attack": is_zero_day_attack,
            "is_coordinated": is_coordinated,
            "is_premature_disclosure": is_premature_disclosure,
            "is_responsible_disclosure": is_responsible_disclosure,
            "has_fix_before_exploit": has_fix_before_exploit,
            "has_fix_before_attack": has_fix_before_attack,
            "is_private_attack": is_private_attack,
            "is_weaponized": is_weaponized,
            "is_mass_exploitation": is_mass_exploitation,
            "is_fix_available": is_fix_available,
            "is_fix_deployed": is_fix_deployed,
            "is_under_attack": is_under_attack,
            "fix_lag_days": fix_lag_days,
            "fix_path": fix_path_arr,
            "threat_state": threat_state_arr,
        }

    def _empty_result(self) -> dict[str, np.ndarray]:
        """Return empty result dict for zero-length arrays."""
        return {
            "is_zero_day": np.array([], dtype=bool),
            "is_zero_day_exploit": np.array([], dtype=bool),
            "is_zero_day_attack": np.array([], dtype=bool),
            "is_coordinated": np.array([], dtype=bool),
            "is_premature_disclosure": np.array([], dtype=bool),
            "is_responsible_disclosure": np.array([], dtype=bool),
            "has_fix_before_exploit": np.array([], dtype=bool),
            "has_fix_before_attack": np.array([], dtype=bool),
            "is_private_attack": np.array([], dtype=bool),
            "is_weaponized": np.array([], dtype=bool),
            "is_mass_exploitation": np.array([], dtype=bool),
            "is_fix_available": np.array([], dtype=bool),
            "is_fix_deployed": np.array([], dtype=bool),
            "is_under_attack": np.array([], dtype=bool),
            "fix_lag_days": np.array([], dtype=np.float32),
            "fix_path": np.array([], dtype=np.uint8),
            "threat_state": np.array([], dtype=np.uint8),
        }
