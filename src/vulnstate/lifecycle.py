# src/vulnstate/lifecycle.py
"""CVD Lifecycle State Machine.

Provides:
- LifecycleState: Abstract base for CVD lifecycle
- ScalarLifecycle: Single-item lifecycle (pure Python)
- VectorLifecycle: Vectorized lifecycle (numpy arrays)
- TimestampsNamespace: Namespace for arr.lifecycle.timestamps.V/F/D/P/X/A
- LifecycleNamespace: Public API namespace for arr.lifecycle.*

Layer: Core
Dependencies: constants.py
Used by: vulnerability.py, array.py
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray

from vulnstate.constants import (
    FIX_PATH_LABELS,
    PAIR_BIT_POSITIONS,
    THREAT_LABELS,
    CVDEvent,
    FixPath,
    ThreatState,
    explain_state,
    state_int_to_string,
    validate_vfd_constraint,
)

if TYPE_CHECKING:
    from vulnstate.array import CVDArray
    from vulnstate.transforms.desiderata import DesiderataExtractor


class LifecycleState(ABC):
    """Abstract base for CVD lifecycle state machine."""

    @property
    @abstractmethod
    def bitmask(self) -> Union[int, NDArray[np.uint8]]:
        """State bitmask (6 bits: VFDPXA)."""
        ...

    @property
    @abstractmethod
    def pair_mask(self) -> Union[int, NDArray[np.uint16]]:
        """Pair ordering mask (15 bits)."""
        ...

    @abstractmethod
    def apply_event(
        self,
        event: CVDEvent,
        timestamp: Optional[datetime] = None,
        mask: Optional[NDArray[np.bool_]] = None,
    ) -> None:
        """Apply event to lifecycle state."""
        ...

    @abstractmethod
    def has_event(self, event: CVDEvent) -> Union[bool, NDArray[np.bool_]]:
        """Check if event has occurred."""
        ...


class ScalarLifecycle(LifecycleState):
    """Single-item lifecycle. Pure Python, no numpy overhead."""

    def __init__(self) -> None:
        self._bitmask: int = 0
        self._pair_mask: int = 0
        self._timestamps: dict[CVDEvent, Optional[datetime]] = {}

    @property
    def bitmask(self) -> int:
        return self._bitmask

    @property
    def pair_mask(self) -> int:
        return self._pair_mask

    @property
    def timestamps(self) -> dict[CVDEvent, Optional[datetime]]:
        return self._timestamps

    @property
    def state(self) -> str:
        """State string (e.g., 'VFdpxa')."""
        return state_int_to_string(self._bitmask)

    @property
    def summary(self) -> str:
        """Human-readable summary."""
        return explain_state(self.state)

    @property
    def fix_path(self) -> FixPath:
        """Fix path classification."""
        vfd_bits = self._bitmask & 0b000111
        return FixPath(vfd_bits)

    @property
    def threat_state(self) -> ThreatState:
        """Threat state classification."""
        pxa_bits = (self._bitmask >> 3) & 0b000111
        valid_values = {e.value for e in ThreatState}
        if pxa_bits not in valid_values:
            if pxa_bits & 0b100:
                return ThreatState.ACTIVE_THREAT if pxa_bits & 0b010 else ThreatState.UNDER_ATTACK
            return ThreatState.LATENT
        return ThreatState(pxa_bits)

    def apply_event(
        self,
        event: CVDEvent,
        timestamp: Optional[datetime] = None,
        mask: Optional[NDArray[np.bool_]] = None,
    ) -> None:
        if not validate_vfd_constraint(event, self._bitmask):
            if event == CVDEvent.F:
                raise ValueError("V must occur before F")
            elif event == CVDEvent.D:
                raise ValueError("F must occur before D")

        self._bitmask |= 1 << event
        self._timestamps[event] = timestamp
        self._recompute_pair_mask()

    def has_event(self, event: CVDEvent) -> bool:
        return bool(self._bitmask & (1 << event))

    def _recompute_pair_mask(self) -> None:
        self._pair_mask = 0
        for (earlier, later), bit_pos in PAIR_BIT_POSITIONS.items():
            earlier_ts = self._timestamps.get(earlier)
            later_ts = self._timestamps.get(later)
            if earlier_ts is not None and later_ts is not None and earlier_ts < later_ts:
                self._pair_mask |= 1 << bit_pos


class VectorLifecycle(LifecycleState):
    """Vectorized lifecycle. Fixed or dynamic sizing."""

    def __init__(self, n: int = 0, *, fixed: bool = False) -> None:
        self._fixed = fixed
        self._size = n
        self._capacity = n if fixed else max(n, 16)

        self._bitmask_arr = np.zeros(self._capacity, dtype=np.uint8)
        self._pair_mask_arr = np.zeros(self._capacity, dtype=np.uint16)
        self._timestamps_arr: dict[CVDEvent, NDArray[np.datetime64]] = {
            event: np.full(self._capacity, np.datetime64("NaT"), dtype="datetime64[us]")
            for event in CVDEvent
        }

    def __len__(self) -> int:
        return self._size

    @property
    def bitmask(self) -> NDArray[np.uint8]:
        return self._bitmask_arr[: self._size]

    @property
    def pair_mask(self) -> NDArray[np.uint16]:
        return self._pair_mask_arr[: self._size]

    @property
    def timestamps(self) -> dict[CVDEvent, NDArray[np.datetime64]]:
        return {k: v[: self._size] for k, v in self._timestamps_arr.items()}

    @property
    def states(self) -> list[str]:
        """State strings for all items."""
        return [state_int_to_string(int(b)) for b in self.bitmask]

    def extend(self, count: int) -> slice:
        """Grow by count slots. Returns slice for new items."""
        if self._fixed:
            raise TypeError("Cannot extend fixed-size lifecycle")

        new_size = self._size + count
        if new_size > self._capacity:
            new_capacity = max(new_size, self._capacity * 2)
            self._grow(new_capacity)

        result = slice(self._size, new_size)
        self._size = new_size
        return result

    def _grow(self, new_capacity: int) -> None:
        new_bitmask = np.zeros(new_capacity, dtype=np.uint8)
        new_bitmask[: self._capacity] = self._bitmask_arr
        self._bitmask_arr = new_bitmask

        new_pair_mask = np.zeros(new_capacity, dtype=np.uint16)
        new_pair_mask[: self._capacity] = self._pair_mask_arr
        self._pair_mask_arr = new_pair_mask

        for event in CVDEvent:
            new_ts: NDArray[np.datetime64] = np.full(
                new_capacity, np.datetime64("NaT"), dtype="datetime64[us]"
            )
            new_ts[: self._capacity] = self._timestamps_arr[event]
            self._timestamps_arr[event] = new_ts

        self._capacity = new_capacity

    def apply_event(
        self,
        event: CVDEvent,
        timestamp: Optional[datetime] = None,
        mask: Optional[NDArray[np.bool_]] = None,
    ) -> None:
        if mask is None:
            mask = np.ones(self._size, dtype=bool)

        ts_value = np.datetime64(timestamp) if timestamp else np.datetime64("NaT")
        self._bitmask_arr[: self._size][mask] |= 1 << event
        self._timestamps_arr[event][: self._size][mask] = ts_value
        self._recompute_pair_mask(mask)

    def has_event(self, event: CVDEvent) -> NDArray[np.bool_]:
        return (self._bitmask_arr[: self._size] & (1 << event)) != 0

    def _recompute_pair_mask(self, mask: NDArray[np.bool_]) -> None:
        indices = np.where(mask)[0]
        for idx in indices:
            new_pair_mask = 0
            for (earlier, later), bit_pos in PAIR_BIT_POSITIONS.items():
                earlier_ts = self._timestamps_arr[earlier][idx]
                later_ts = self._timestamps_arr[later][idx]
                if not np.isnat(earlier_ts) and not np.isnat(later_ts) and earlier_ts < later_ts:
                    new_pair_mask |= 1 << bit_pos
            self._pair_mask_arr[idx] = new_pair_mask


class TimestampsNamespace:
    """Namespace for arr.lifecycle.timestamps.V/F/D/P/X/A access.

    Provides attribute-style access to event timestamp arrays.

    Example:
        >>> arr.lifecycle.timestamps.V  # When vendor became aware
        >>> arr.lifecycle.timestamps.P  # When public became aware
    """

    def __init__(self, lifecycle: VectorLifecycle) -> None:
        self._lifecycle = lifecycle

    @property
    def V(self) -> NDArray[np.datetime64]:
        """Vendor awareness timestamps."""
        return self._lifecycle._timestamps_arr[CVDEvent.V][: self._lifecycle._size]

    @property
    def F(self) -> NDArray[np.datetime64]:
        """Fix ready timestamps."""
        return self._lifecycle._timestamps_arr[CVDEvent.F][: self._lifecycle._size]

    @property
    def D(self) -> NDArray[np.datetime64]:
        """Fix deployed timestamps."""
        return self._lifecycle._timestamps_arr[CVDEvent.D][: self._lifecycle._size]

    @property
    def P(self) -> NDArray[np.datetime64]:
        """Public awareness timestamps."""
        return self._lifecycle._timestamps_arr[CVDEvent.P][: self._lifecycle._size]

    @property
    def X(self) -> NDArray[np.datetime64]:
        """Exploit public timestamps."""
        return self._lifecycle._timestamps_arr[CVDEvent.X][: self._lifecycle._size]

    @property
    def A(self) -> NDArray[np.datetime64]:
        """Attacks observed timestamps."""
        return self._lifecycle._timestamps_arr[CVDEvent.A][: self._lifecycle._size]


class ArrayTimestampsNamespace:
    """Namespace for arr.lifecycle.timestamps.V/F/D/P/X/A access.

    Provides attribute-style access to event timestamp arrays from the array's
    actual timestamps storage.

    Example:
        >>> arr.lifecycle.timestamps.V  # When vendor became aware
        >>> arr.lifecycle.timestamps.P  # When public became aware
    """

    def __init__(self, array: "CVDArray") -> None:
        self._array = array

    @property
    def V(self) -> NDArray[np.datetime64]:
        """Vendor awareness timestamps."""
        return self._array.timestamps.V

    @property
    def F(self) -> NDArray[np.datetime64]:
        """Fix ready timestamps."""
        return self._array.timestamps.F

    @property
    def D(self) -> NDArray[np.datetime64]:
        """Fix deployed timestamps."""
        return self._array.timestamps.D

    @property
    def P(self) -> NDArray[np.datetime64]:
        """Public awareness timestamps."""
        return self._array.timestamps.P

    @property
    def X(self) -> NDArray[np.datetime64]:
        """Exploit public timestamps."""
        return self._array.timestamps.X

    @property
    def A(self) -> NDArray[np.datetime64]:
        """Attacks observed timestamps."""
        return self._array.timestamps.A

    def _get_event_array(self, event: CVDEvent) -> NDArray[np.datetime64]:
        """Get timestamp array for event."""
        return getattr(self._array.timestamps, event.name)

    def _set_event_array(self, event: CVDEvent, value: NDArray[np.datetime64]) -> None:
        """Set timestamp array for event."""
        setattr(self._array.timestamps, event.name, value)


class LifecycleNamespace:
    """Public API namespace for arr.lifecycle.* access.

    Provides the public API for CVD lifecycle state access:
    - arr.lifecycle.state - State strings (e.g., "VFdpXa")
    - arr.lifecycle.summary - Human-readable summaries
    - arr.lifecycle.fix_path - FixPath enum values
    - arr.lifecycle.fix_path_labels - Human-readable fix path labels
    - arr.lifecycle.threat_state - ThreatState enum values
    - arr.lifecycle.threat_labels - Human-readable threat labels
    - arr.lifecycle.timestamps.V/F/D/P/X/A - Event timestamps
    - arr.lifecycle.desiderata - DesiderataExtractor for analysis

    Example:
        >>> arr = CVDArray.generate(100)
        >>> arr.lifecycle.state  # ["VFdpxa", "VFDPxa", ...]
        >>> arr.lifecycle.timestamps.V  # Vendor awareness timestamps
        >>> arr.lifecycle.desiderata.is_zero_day  # Zero-day flags
    """

    def __init__(self, array: "CVDArray") -> None:
        self._array = array
        self._timestamps_ns: Optional[ArrayTimestampsNamespace] = None
        self._desiderata: Optional[DesiderataExtractor] = None

    @property
    def state(self) -> NDArray[np.object_]:
        """State strings array (e.g., ['VFdpxa', 'vfdPxa'])."""
        return np.array(
            [state_int_to_string(int(b)) for b in self._array.state_ints],
            dtype=object,
        )

    @property
    def summary(self) -> NDArray[np.object_]:
        """Human-readable state summaries."""
        return np.array(
            [explain_state(state_int_to_string(int(b))) for b in self._array.state_ints],
            dtype=object,
        )

    @property
    def fix_path(self) -> NDArray[np.uint8]:
        """VFD dimension as uint8 array (FixPath enum values)."""
        return (self._array.state_ints & 0b000111).astype(np.uint8)

    @property
    def fix_path_labels(self) -> NDArray[np.object_]:
        """Human-readable fix path labels."""
        return np.array(
            [FIX_PATH_LABELS.get(int(fp), "Unknown") for fp in self.fix_path],
            dtype=object,
        )

    @property
    def threat_state(self) -> NDArray[np.uint8]:
        """PXA dimension as uint8 array (ThreatState enum values)."""
        return ((self._array.state_ints >> 3) & 0b000111).astype(np.uint8)

    @property
    def threat_labels(self) -> NDArray[np.object_]:
        """Human-readable threat state labels."""
        return np.array(
            [THREAT_LABELS.get(int(ts), "Unknown") for ts in self.threat_state],
            dtype=object,
        )

    @property
    def timestamps(self) -> ArrayTimestampsNamespace:
        """Namespace for event timestamps (V, F, D, P, X, A)."""
        if self._timestamps_ns is None:
            self._timestamps_ns = ArrayTimestampsNamespace(self._array)
        return self._timestamps_ns

    @property
    def desiderata(self) -> "DesiderataExtractor":
        """Desiderata analysis (lazy, cached)."""
        if self._desiderata is None:
            from vulnstate.transforms.desiderata import DesiderataExtractor

            self._desiderata = DesiderataExtractor(self._array)
        return self._desiderata

    @property
    def bitmask(self) -> NDArray[np.uint8]:
        """Raw state bitmask (for advanced use)."""
        return self._array.state_ints

    @property
    def pair_mask(self) -> NDArray[np.uint16]:
        """Raw pair ordering mask (for advanced use)."""
        return self.desiderata.pair_mask

    # ==================== Lifecycle Operations ====================

    def has_event(self, event: CVDEvent) -> NDArray[np.bool_]:
        """Check which vulnerabilities have event occurred (vectorized).

        Args:
            event: Event to check

        Returns:
            Boolean array where True = event occurred
        """
        return (self._array.state_ints & (1 << event)) != 0

    def apply_event(
        self,
        event: CVDEvent,
        mask: Optional[NDArray[np.bool_]] = None,
        timestamp: Optional[datetime] = None,
    ) -> NDArray[np.bool_]:
        """Apply event to multiple vulnerabilities (vectorized).

        Validates V→F→D constraints before applying.

        Args:
            event: Event to apply
            mask: Boolean mask indicating which to update.
                  If None, applies to all that satisfy constraints.
            timestamp: Timestamp for the event (default: now)

        Returns:
            Boolean mask indicating which vulnerabilities were updated
        """
        if mask is None:
            # Create mask: vulnerabilities that don't have event
            mask = ~self.has_event(event)

            # Apply V→F→D constraints
            if event == CVDEvent.F:
                mask &= self.has_event(CVDEvent.V)
            elif event == CVDEvent.D:
                mask &= self.has_event(CVDEvent.F)

        # Apply event by setting bit
        self._array.state_ints[mask] |= 1 << event

        # Update timestamps
        ts = timestamp or datetime.now()
        ts_dt64 = np.datetime64(ts, "us")

        # Only set where not already set
        event_ts = self.timestamps._get_event_array(event)
        new_ts = np.where(np.isnat(event_ts) & mask, ts_dt64, event_ts)
        self.timestamps._set_event_array(event, new_ts)

        # Invalidate desiderata cache (pair_mask needs recompute)
        self._desiderata = None

        # Mark array as stale if it was transformed (API v2)
        if self._array._is_transformed:
            self._array._stale = True

        return mask
