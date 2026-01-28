# tests/unit/test_lifecycle.py
"""Tests for lifecycle module."""
from datetime import datetime

import numpy as np
import pytest

from vulnstate.constants import CVDEvent, FixPath
from vulnstate.lifecycle import ScalarLifecycle, VectorLifecycle


class TestScalarLifecycle:
    def test_initial_state_is_zero(self):
        lc = ScalarLifecycle()
        assert lc.bitmask == 0
        assert lc.state == "vfdpxa"

    def test_apply_event_sets_bit(self):
        lc = ScalarLifecycle()
        lc.apply_event(CVDEvent.V)
        assert lc.has_event(CVDEvent.V)
        assert lc.bitmask == 0b000001

    def test_apply_event_with_timestamp(self):
        lc = ScalarLifecycle()
        ts = datetime(2024, 1, 15)
        lc.apply_event(CVDEvent.V, timestamp=ts)
        assert lc.timestamps[CVDEvent.V] == ts

    def test_vfd_constraint_enforced(self):
        lc = ScalarLifecycle()
        with pytest.raises(ValueError):
            lc.apply_event(CVDEvent.F)

    def test_pair_mask_updated(self):
        lc = ScalarLifecycle()
        lc.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))
        lc.apply_event(CVDEvent.P, timestamp=datetime(2024, 1, 2))
        assert (lc.pair_mask & (1 << 2)) != 0  # V<P

    def test_fix_path_property(self):
        lc = ScalarLifecycle()
        lc.apply_event(CVDEvent.V)
        lc.apply_event(CVDEvent.F)
        assert lc.fix_path == FixPath.FIX_READY


class TestVectorLifecycle:
    def test_fixed_size_initialization(self):
        lc = VectorLifecycle(n=100, fixed=True)
        assert len(lc) == 100

    def test_dynamic_extend(self):
        lc = VectorLifecycle(n=5, fixed=False)
        idx = lc.extend(3)
        assert len(lc) == 8
        assert idx == slice(5, 8)

    def test_extend_fixed_raises(self):
        lc = VectorLifecycle(n=5, fixed=True)
        with pytest.raises(TypeError):
            lc.extend(1)

    def test_apply_event_with_mask(self):
        lc = VectorLifecycle(n=5, fixed=True)
        mask = np.array([True, False, True, False, True])
        lc.apply_event(CVDEvent.V, mask=mask)
        assert lc.has_event(CVDEvent.V).tolist() == [True, False, True, False, True]

    def test_states_property(self):
        lc = VectorLifecycle(n=2, fixed=True)
        lc.apply_event(CVDEvent.V, mask=np.array([True, False]))
        assert lc.states[0] == "Vfdpxa"
        assert lc.states[1] == "vfdpxa"
