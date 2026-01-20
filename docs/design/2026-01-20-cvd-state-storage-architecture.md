# CVD State Storage Architecture

**Date:** 2026-01-20
**Status:** Authoritative Design
**Purpose:** Document the memory/performance trade-offs for CVD state storage to prevent repeated architectural mistakes

## Problem Statement

When storing CVD state for batch operations (CVDArray), we need to balance:
1. **Memory efficiency** - minimize bytes per vulnerability
2. **Query performance** - enable fast vectorized analytics
3. **API clarity** - intuitive access patterns for users

The naive approach of storing all derived data (history, event_order, pairs) leads to data duplication and confusion about the source of truth.

## Core Principle: Source of Truth

**Stored Data (Source of Truth):**
```python
state_encoded: np.uint8           # Which events occurred (6 bits: VFDPXA)
timestamps: dict[event, np.ndarray[datetime64]]  # When each event occurred
```

**Derived Data (Computed or Cached):**
- `event_order` - Sort timestamps by time → list of events
- `history` - Replay events on state machine → transitions
- `pair_mask` - Check timestamp ordering → uint16 bitmask
- `history_id` - Map complete ordering → 0-69 (if all 6 events present)

## Storage Options Analysis

### Option A: Store Everything (Naive - AVOID)
```python
# Source data
states: np.ndarray[uint8]
V_timestamps, F_timestamps, ... : np.ndarray[datetime64]

# Duplicate/derived data (WASTEFUL)
ArrayHistories: events, from_states, to_states, timestamps, offsets
ArrayEventOrders: events, offsets
```

**Memory:** ~150-200 bytes/vuln
**Performance:** Fast access, but wastes memory on redundant data
**Problem:** Multiple sources of truth, potential inconsistency
**Verdict:** ❌ **DO NOT USE** - This is what we mistakenly implemented

### Option B: Compute On-Demand (Pure Lazy)
```python
# Source data only
states: np.ndarray[uint8]
V_timestamps, F_timestamps, ... : np.ndarray[datetime64]

# Everything computed on access
@property
def event_order(self) -> list[list[CVDEvent]]:
    """Sort timestamps for each vuln."""
    return [self._compute_order(i) for i in range(len(self))]
```

**Memory:** 49 bytes/vuln (optimal)
**Performance:** O(n log n) per access, recomputation overhead
**Problem:** Slow for repeated queries (analytics, filtering)
**Verdict:** ⚠️ **Use only for single-vulnerability access**

### Option C: Precomputed Bitmasks (Balanced - RECOMMENDED)
```python
# Source data
states: np.ndarray[uint8]                      # 1 byte/vuln
V_timestamps, F_timestamps, ... : np.ndarray[datetime64]  # 48 bytes/vuln

# Precomputed analytics cache (computed once, reused)
pair_mask: np.ndarray[uint16]                  # 2 bytes/vuln
history_id: np.ndarray[uint8]                  # 1 byte/vuln (optional)
```

**Memory:** 52 bytes/vuln (49 source + 3 cache)
**Performance:** O(1) vectorized queries on pair_mask
**Use Case:** Enables fast analytics without recomputation
**Verdict:** ✅ **RECOMMENDED for CVDArray**

## Detailed Design: Option C (Recommended)

### 1. Source of Truth Storage

```python
@dataclass
class ArrayCVDState:
    """
    CVD state machine data - SINGLE SOURCE OF TRUTH.

    All other CVD state information (history, event_order, pairs)
    can be derived from these two fields.
    """
    # Which events occurred (bitmask: bit 0=V, 1=F, 2=D, 3=P, 4=X, 5=A)
    states: np.ndarray  # uint8[N]

    # When each event occurred (NaT if not occurred)
    V_timestamps: np.ndarray  # datetime64[us][N]
    F_timestamps: np.ndarray  # datetime64[us][N]
    D_timestamps: np.ndarray  # datetime64[us][N]
    P_timestamps: np.ndarray  # datetime64[us][N]
    X_timestamps: np.ndarray  # datetime64[us][N]
    A_timestamps: np.ndarray  # datetime64[us][N]
```

### 2. Precomputed Analytics Cache

```python
@dataclass
class ArrayCVDAnalytics:
    """
    Precomputed bitmasks for fast vectorized queries.

    Computed once during array construction or sync(), reused for all queries.
    These are DERIVED from ArrayCVDState - not independent data.
    """
    # Pair ordering mask (15 bits for 15 event pairs)
    # Bit i = 1 if pair i occurred in desired order
    # Example: bit 2 (V≺P) = 1 if V timestamp < P timestamp
    pair_mask: np.ndarray  # uint16[N]

    # History ID (0-69 for complete histories, 255 for incomplete)
    # Maps to one of 70 valid complete orderings (see cvd-histories.md)
    # Only valid if all 6 events occurred
    history_id: np.ndarray  # uint8[N] (optional, can derive from pair_mask)
```

### 3. Pair Mask Encoding

**Bit Layout (15 bits for 15 pairs):**
```
Bit  0: V≺F    Bit  5: F≺D    Bit  9: D≺P    Bit 12: P≺X
Bit  1: V≺D    Bit  6: F≺P    Bit 10: D≺X    Bit 13: P≺A
Bit  2: V≺P    Bit  7: F≺X    Bit 11: D≺A    Bit 14: X≺A
Bit  3: V≺X    Bit  8: F≺A
Bit  4: V≺A
```

**Computation (vectorized):**
```python
def compute_pair_mask(arr: CVDArray) -> np.ndarray:
    """Compute pair_mask from timestamps (vectorized)."""
    mask = np.zeros(len(arr), dtype=np.uint16)

    # For each of 15 pairs, check if earlier < later
    pairs = [
        (CVDEvent.V, CVDEvent.F, 0),
        (CVDEvent.V, CVDEvent.D, 1),
        # ... all 15 pairs
    ]

    for earlier, later, bit_pos in pairs:
        earlier_ts = arr.timestamps[earlier]
        later_ts = arr.timestamps[later]

        # Both occurred and earlier < later
        both_occurred = ~np.isnat(earlier_ts) & ~np.isnat(later_ts)
        correct_order = earlier_ts < later_ts

        # Set bit if pair satisfied
        mask[both_occurred & correct_order] |= (1 << bit_pos)

    return mask
```

**Usage (O(1) vectorized queries):**
```python
# Check desiderata satisfaction
DESIDERATA_MASK = 0b0111_1111_1111_1100  # Bits 2-14
desiderata_count = np.array([bin(m & DESIDERATA_MASK).count('1')
                               for m in arr.pair_mask], dtype=np.uint8)

# Find violations of required pairs (V≺F, V≺D, F≺D)
REQUIRED_MASK = 0b0000_0000_0010_0011  # Bits 0, 1, 5
violations = ~arr.pair_mask & REQUIRED_MASK
has_violation = violations != 0

# Find zero-day exploits (X before V = violation of V≺X at bit 3)
is_zero_day_exploit = (arr.pair_mask & (1 << 3)) == 0
```

### 4. API Design

**User-facing properties (compute on-demand for single vuln):**
```python
# CVDVulnerability
@property
def events(self) -> dict[CVDEvent, Optional[datetime]]:
    """Dict of event timestamps (derived from state + timestamps)."""
    return {e: self.timestamps.get(e) for e in CVDEvent
            if self.state_encoded & (1 << e)}

@property
def event_order(self) -> list[CVDEvent]:
    """Chronological order of events (derived from timestamps)."""
    occurred = [(e, ts) for e, ts in self.timestamps.items()
                if ts is not None]
    return [e for e, ts in sorted(occurred, key=lambda x: x[1])]

@property
def history(self) -> list[dict[str, Any]]:
    """Full state transitions (derived from event_order + state replay)."""
    transitions = []
    state = 0
    for event in self.event_order:
        from_state = state
        state |= (1 << event)
        transitions.append({
            'event': event,
            'from_state': from_state,
            'to_state': state,
            'timestamp': self.timestamps[event]
        })
    return transitions
```

**Array-level properties (use cached pair_mask):**
```python
# CVDArray
@property
def is_zero_day_exploit(self) -> np.ndarray:
    """X before V (vectorized from pair_mask)."""
    return (self.analytics.pair_mask & (1 << 3)) == 0  # V≺X bit clear

@property
def is_coordinated(self) -> np.ndarray:
    """V before P (vectorized from pair_mask)."""
    return (self.analytics.pair_mask & (1 << 2)) != 0  # V≺P bit set

@property
def desiderata_scores(self) -> np.ndarray:
    """Fraction of 12 desiderata satisfied (vectorized)."""
    DESIDERATA_MASK = 0b0111_1111_1111_1100
    satisfied_bits = self.analytics.pair_mask & DESIDERATA_MASK
    counts = np.array([bin(m).count('1') for m in satisfied_bits], dtype=np.uint8)
    return counts / 12.0
```

### 5. When to Compute pair_mask

**Trigger points:**
```python
# 1. During array construction from vulnerabilities
arr = CVDArray([vuln1, vuln2, ...])
# → compute pair_mask once for all vulns

# 2. After sync() when live objects modified
arr[0].apply_event(CVDEvent.V)
arr.sync()
# → recompute pair_mask only for modified indices

# 3. After batch event application
arr.apply_event_batch(CVDEvent.V, mask=some_mask)
# → recompute pair_mask only for affected vulns
```

**Caching strategy:**
```python
class CVDArray:
    def __init__(self, ...):
        self._pair_mask_dirty = True

    @property
    def pair_mask(self) -> np.ndarray:
        """Lazy computation with caching."""
        if self._pair_mask_dirty:
            self.analytics.pair_mask = compute_pair_mask(self)
            self._pair_mask_dirty = False
        return self.analytics.pair_mask

    def sync(self, indices=None):
        """Mark pair_mask dirty after state changes."""
        # ... sync states and timestamps ...
        self._pair_mask_dirty = True
```

## Memory Comparison

For 100,000 vulnerabilities:

| Approach | Memory | Access Speed | Query Speed |
|----------|--------|--------------|-------------|
| **Option A (store all)** | 15-20 MB | O(1) | O(1) |
| **Option B (pure lazy)** | 4.9 MB | O(n log n) | O(n log n) |
| **Option C (bitmask cache)** | 5.2 MB | O(1) | O(1) |

**Verdict:** Option C provides O(1) query performance with only 6% more memory than pure lazy.

## Implementation Checklist

- [ ] Remove ArrayHistories dataclass (redundant with timestamps)
- [ ] Remove ArrayEventOrders dataclass (redundant with timestamps)
- [ ] Add ArrayCVDAnalytics with pair_mask field
- [ ] Implement compute_pair_mask() vectorized function
- [ ] Add lazy computation pattern with dirty tracking
- [ ] Make CVDVulnerability.history a computed property
- [ ] Make CVDVulnerability.event_order a computed property
- [ ] Update CVDArray to use pair_mask for analytical properties
- [ ] Update CVDAnalyzer to use pair_mask instead of recomputing
- [ ] Add tests for pair_mask computation accuracy
- [ ] Document this architecture in main docs

## References

- **Bitmask storage design:** `docs/design/2026-01-15-bitmask-storage.md`
- **Memory optimization:** `docs/archive/memory_optimization.md`
- **70 valid histories:** `docs/ref/cvd-histories.md`
- **Pair relationships:** `src/vulnstate/constants.py` (ORDERED_PAIRS_TABLE)

## Key Takeaway

**Store only the source of truth (state + timestamps). Cache precomputed bitmasks for fast vectorized queries. Never store redundant derived data like history or event_order - compute on-demand or derive from pair_mask.**

This design achieves the best balance of memory efficiency and query performance for batch operations.
