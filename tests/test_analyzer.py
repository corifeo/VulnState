# tests/test_analyzer.py
"""
Tests for CVDAnalyzer - Vectorized analytics computation.
"""

import numpy as np

from vulnstate import CVDArray, CVDVulnerability
from vulnstate.constants import CVDEvent


def test_analyzer_init():
    """Analyzer initializes with array context."""
    from vulnstate.analyzer import CVDAnalyzer

    arr = CVDArray.zeros(10)
    analyzer = CVDAnalyzer(arr)

    assert analyzer.n == 10
    assert analyzer.states is arr.states


def test_analyzer_fix_path():
    """Fix path computed from VFD bits."""
    from vulnstate.analyzer import CVDAnalyzer

    vulns = [
        CVDVulnerability(state="vfdpxa"),  # 0b000 = No Awareness
        CVDVulnerability(state="Vfdpxa"),  # 0b001 = Vendor Aware
        CVDVulnerability(state="VFdpxa"),  # 0b011 = Fix Ready
        CVDVulnerability(state="VFDpxa"),  # 0b111 = Remediated
    ]
    arr = CVDArray(vulns)
    analyzer = CVDAnalyzer(arr)

    fix_paths = analyzer.fix_path
    assert fix_paths[0] == 0b000
    assert fix_paths[1] == 0b001
    assert fix_paths[2] == 0b011
    assert fix_paths[3] == 0b111


def test_analyzer_threat_state():
    """Threat state computed from PXA bits."""
    from vulnstate.analyzer import CVDAnalyzer

    vulns = [
        CVDVulnerability(state="VFDpxa"),  # 0b000 = Latent
        CVDVulnerability(state="VFDPxa"),  # 0b001 = Disclosed
        CVDVulnerability(state="VFDPXa"),  # 0b011 = Weaponized
        CVDVulnerability(state="VFDPXA"),  # 0b111 = Active Threat
    ]
    arr = CVDArray(vulns)
    analyzer = CVDAnalyzer(arr)

    threat_states = analyzer.threat_state
    assert threat_states[0] == 0b000
    assert threat_states[1] == 0b001
    assert threat_states[2] == 0b011
    assert threat_states[3] == 0b111


def test_analyzer_pair_mask_valid():
    """Pair mask computed for valid history."""
    from datetime import datetime, timedelta

    from vulnstate.analyzer import CVDAnalyzer

    base = datetime(2024, 1, 1)
    vuln = CVDVulnerability()
    # Apply in valid order: V, F, D, P
    vuln.apply_event(CVDEvent.V, timestamp=base)
    vuln.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
    vuln.apply_event(CVDEvent.D, timestamp=base + timedelta(days=20))
    vuln.apply_event(CVDEvent.P, timestamp=base + timedelta(days=30))

    arr = CVDArray([vuln])
    analyzer = CVDAnalyzer(arr)

    # Check pair mask has V≺F, V≺D, F≺D bits set
    pair_mask = analyzer.pair_mask
    assert pair_mask[0] & (1 << 0)  # V≺F
    assert pair_mask[0] & (1 << 1)  # V≺D
    assert pair_mask[0] & (1 << 5)  # F≺D


def test_analyzer_validity_valid():
    """Valid history passes validation."""
    from datetime import datetime, timedelta

    from vulnstate.analyzer import CVDAnalyzer
    from vulnstate.constants import HistoryValidity

    base = datetime(2024, 1, 1)
    vuln = CVDVulnerability()
    vuln.apply_event(CVDEvent.V, timestamp=base)
    vuln.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))

    arr = CVDArray([vuln])
    analyzer = CVDAnalyzer(arr)

    assert analyzer.validity[0] == HistoryValidity.VALID


def test_analyzer_validity_impossible():
    """Impossible history (F before V) detected."""
    from datetime import datetime, timedelta

    from vulnstate.analyzer import CVDAnalyzer
    from vulnstate.constants import HistoryValidity

    # Manually create impossible state (F timestamp before V)
    base = datetime(2024, 1, 1)
    vuln = CVDVulnerability()
    vuln.apply_event(CVDEvent.V, timestamp=base + timedelta(days=10))  # V later
    # Can't apply F before V normally, so modify timestamps directly
    vuln.events[CVDEvent.F] = base  # F earlier (impossible)
    vuln.event_data.state_encoded |= 1 << 1  # Set F bit

    arr = CVDArray([vuln])
    analyzer = CVDAnalyzer(arr)

    assert analyzer.validity[0] == HistoryValidity.IMPOSSIBLE


def test_analyzer_infer_v_from_p():
    """V inferred from P (vendor learns from public)."""
    from datetime import datetime

    from vulnstate.analyzer import CVDAnalyzer

    vuln = CVDVulnerability()
    vuln.apply_event(CVDEvent.P, timestamp=datetime(2024, 2, 1))
    # V not applied, but should be inferred

    arr = CVDArray([vuln])
    analyzer = CVDAnalyzer(arr)

    # Before inference
    assert np.isnat(arr.timestamps.V[0])

    # Run inference
    analyzer.apply_inferences()

    # After inference: V timestamp filled, inferred_mask set
    assert not np.isnat(arr.timestamps.V[0])
    assert arr.timestamps.V[0] <= arr.timestamps.P[0]
    assert analyzer.inferred_mask[0] & (1 << 0)  # V bit set in inferred_mask


def test_analyzer_infer_v_from_f():
    """V inferred from F (causality constraint)."""
    from datetime import datetime

    from vulnstate.analyzer import CVDAnalyzer

    # Create vuln with F but no V (impossible to do normally, simulate import)
    vuln = CVDVulnerability()
    vuln.events[CVDEvent.F] = datetime(2024, 1, 15)
    vuln.event_data.state_encoded |= 1 << 1  # Set F bit

    arr = CVDArray([vuln])
    arr.timestamps.F[0] = np.datetime64(datetime(2024, 1, 15), "us")

    analyzer = CVDAnalyzer(arr)
    analyzer.apply_inferences()

    # V should be inferred before F
    assert not np.isnat(arr.timestamps.V[0])
    assert arr.timestamps.V[0] <= arr.timestamps.F[0]


def test_analyzer_inferred_mask():
    """Inferred mask tracks which events were inferred."""
    from datetime import datetime

    from vulnstate.analyzer import CVDAnalyzer

    vuln = CVDVulnerability()
    vuln.apply_event(CVDEvent.P, timestamp=datetime(2024, 2, 1))

    arr = CVDArray([vuln])
    analyzer = CVDAnalyzer(arr)
    analyzer.apply_inferences()

    # V was inferred (bit 0), P was observed (bit 3 not set)
    mask = analyzer.inferred_mask[0]
    assert mask & (1 << 0)  # V inferred
    assert not (mask & (1 << 3))  # P observed


def test_analyzer_desiderata_score():
    """Desiderata score computed from satisfied pairs."""
    from datetime import datetime, timedelta

    from vulnstate.analyzer import CVDAnalyzer

    base = datetime(2024, 1, 1)

    # Perfect history: V, F, D, P, X, A in ideal order
    vuln = CVDVulnerability()
    vuln.apply_event(CVDEvent.V, timestamp=base)
    vuln.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
    vuln.apply_event(CVDEvent.D, timestamp=base + timedelta(days=20))
    vuln.apply_event(CVDEvent.P, timestamp=base + timedelta(days=30))
    vuln.apply_event(CVDEvent.X, timestamp=base + timedelta(days=40))
    vuln.apply_event(CVDEvent.A, timestamp=base + timedelta(days=50))

    arr = CVDArray([vuln])
    analyzer = CVDAnalyzer(arr)

    # All 12 desiderata should be satisfied
    assert analyzer.desiderata_score[0] == 12


def test_analyzer_desiderata_score_partial():
    """Partial desiderata score for incomplete history."""
    from datetime import datetime, timedelta

    from vulnstate.analyzer import CVDAnalyzer

    base = datetime(2024, 1, 1)

    # History with P before F (violates F≺P desideratum)
    vuln = CVDVulnerability()
    vuln.apply_event(CVDEvent.V, timestamp=base)
    vuln.apply_event(CVDEvent.P, timestamp=base + timedelta(days=5))  # P before F
    vuln.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))

    arr = CVDArray([vuln])
    analyzer = CVDAnalyzer(arr)

    # F≺P violated, score < 12
    assert analyzer.desiderata_score[0] < 12


def test_analyzer_fix_path_labels():
    """Fix path labels from conversion table."""
    from vulnstate.analyzer import CVDAnalyzer

    vulns = [
        CVDVulnerability(state="vfdpxa"),
        CVDVulnerability(state="Vfdpxa"),
        CVDVulnerability(state="VFdpxa"),
        CVDVulnerability(state="VFDpxa"),
    ]
    arr = CVDArray(vulns)
    analyzer = CVDAnalyzer(arr)

    labels = analyzer.fix_path_labels
    assert labels[0] == "No Awareness"
    assert labels[1] == "Vendor Aware"
    assert labels[2] == "Fix Ready"
    assert labels[3] == "Remediated"


def test_analyzer_threat_labels():
    """Threat state labels from conversion table."""
    from vulnstate.analyzer import CVDAnalyzer

    vulns = [
        CVDVulnerability(state="VFDpxa"),
        CVDVulnerability(state="VFDPxa"),
        CVDVulnerability(state="VFDPXa"),
        CVDVulnerability(state="VFDPXA"),
    ]
    arr = CVDArray(vulns)
    analyzer = CVDAnalyzer(arr)

    labels = analyzer.threat_labels
    assert labels[0] == "Latent"
    assert labels[1] == "Disclosed"
    assert labels[2] == "Weaponized"
    assert labels[3] == "Active Threat"


def test_analyzer_importable_from_package():
    """CVDAnalyzer importable from main package."""
    from vulnstate import CVDAnalyzer, HistoryValidity

    assert CVDAnalyzer is not None
    assert HistoryValidity.VALID == 0


def test_analyzer_full_workflow():
    """
    Integration test: Import → Analyze → Enrich workflow.

    Simulates: NVD import (P only) → Inference → Validation → Labels
    """
    from datetime import datetime

    from vulnstate import CVDAnalyzer
    from vulnstate.constants import HistoryValidity

    # Simulate NVD import: only P timestamps
    vulns = []
    for i in range(100):
        vuln = CVDVulnerability(cve_id=f"CVE-2024-{i:04d}")
        vuln.apply_event(CVDEvent.P, timestamp=datetime(2024, 1, 1 + i % 28))
        vulns.append(vuln)

    arr = CVDArray(vulns)

    # Before inference: V timestamps missing
    assert np.isnat(arr.timestamps.V).all()

    # Run analyzer
    analyzer = CVDAnalyzer(arr)
    analyzer.apply_inferences()

    # After inference: V timestamps filled
    assert not np.isnat(arr.timestamps.V).any()

    # Check validity
    assert (analyzer.validity == HistoryValidity.VALID).all()

    # Check labels
    assert (analyzer.fix_path_labels == "Vendor Aware").all()  # All have V inferred
    assert (analyzer.threat_labels == "Disclosed").all()  # All have P

    # Check provenance
    assert (analyzer.inferred_mask & (1 << 0)).all()  # All V inferred


def test_explain_pair_violations_single():
    """explain_pair_violations returns violated pair names."""
    from vulnstate.analyzer import CVDAnalyzer

    # All required pairs violated (none satisfied) - bits 0, 1, 5 NOT set
    pair_mask = 0b0000_0000_0000_0000  # Nothing satisfied
    violations = CVDAnalyzer.explain_pair_violations(pair_mask)

    assert "V≺F" in violations  # bit 0 not set
    assert "V≺D" in violations  # bit 1 not set
    assert "F≺D" in violations  # bit 5 not set
    assert len(violations) == 3


def test_explain_pair_violations_valid():
    """No violations when all required pairs satisfied."""
    from vulnstate.analyzer import CVDAnalyzer

    # Required pairs satisfied: V≺F (0), V≺D (1), F≺D (5)
    pair_mask = 0b0000_0000_0010_0011  # bits 0, 1, 5 set
    violations = CVDAnalyzer.explain_pair_violations(pair_mask)

    assert violations == []


def test_explain_pair_violations_partial():
    """Partial violations: some required pairs satisfied, some violated."""
    from vulnstate.analyzer import CVDAnalyzer

    # V≺F satisfied (bit 0), but V≺D and F≺D violated
    pair_mask = 0b0000_0000_0000_0001  # Only bit 0 set
    violations = CVDAnalyzer.explain_pair_violations(pair_mask)

    assert "V≺D" in violations  # bit 1 not set
    assert "F≺D" in violations  # bit 5 not set
    assert "V≺F" not in violations  # bit 0 IS set
    assert len(violations) == 2


def test_explain_pair_violations_with_analyzer():
    """Integration: explain violations from analyzer.pair_mask."""
    from datetime import datetime, timedelta

    from vulnstate.analyzer import CVDAnalyzer

    # Create impossible history: F before V
    base = datetime(2024, 1, 1)
    vuln = CVDVulnerability()
    vuln.apply_event(CVDEvent.V, timestamp=base + timedelta(days=10))
    vuln.events[CVDEvent.F] = base  # F before V (impossible)
    vuln.event_data.state_encoded |= 1 << 1

    arr = CVDArray([vuln])
    analyzer = CVDAnalyzer(arr)

    violations = CVDAnalyzer.explain_pair_violations(int(analyzer.pair_mask[0]))
    assert "V≺F" in violations  # F happened before V


def test_analyze_static_method():
    """CVDAnalyzer.analyze() returns complete AnalysisResult."""
    from datetime import datetime, timedelta

    from vulnstate.analyzer import CVDAnalyzer
    from vulnstate.models import AnalysisResult

    base = datetime(2024, 1, 1)

    # Create vulnerability with ideal history
    vuln = CVDVulnerability()
    vuln.apply_event(CVDEvent.V, timestamp=base)
    vuln.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
    vuln.apply_event(CVDEvent.D, timestamp=base + timedelta(days=20))
    vuln.apply_event(CVDEvent.P, timestamp=base + timedelta(days=30))
    vuln.apply_event(CVDEvent.X, timestamp=base + timedelta(days=40))
    vuln.apply_event(CVDEvent.A, timestamp=base + timedelta(days=50))

    arr = CVDArray([vuln])
    result = CVDAnalyzer.analyze(arr)

    # Check return type
    assert isinstance(result, AnalysisResult)
    assert len(result) == 1

    # Group 1: Validity & Completeness
    assert result.event_count[0] == 6
    assert result.is_complete[0] == True  # noqa: E712

    # Group 3: Zero-Day Indicators
    assert result.is_zero_day[0] == False  # noqa: E712 (V before X and A)

    # Group 4: Coordination Quality
    assert result.is_coordinated[0] == True  # noqa: E712 (V before P)

    # Group 5: Fix Effectiveness
    assert result.has_fix_before_exploit[0] == True  # noqa: E712

    # Group 8: Scores
    assert result.desiderata_count[0] == 12  # Perfect history


def test_analyze_zero_day_detection():
    """CVDAnalyzer.analyze() detects zero-day scenarios."""
    from datetime import datetime, timedelta

    from vulnstate.analyzer import CVDAnalyzer

    base = datetime(2024, 1, 1)

    # Zero-day exploit: X before V
    vuln = CVDVulnerability()
    vuln.apply_event(CVDEvent.X, timestamp=base)
    vuln.apply_event(CVDEvent.V, timestamp=base + timedelta(days=10))

    arr = CVDArray([vuln])
    result = CVDAnalyzer.analyze(arr)

    assert result.is_zero_day[0] == True  # noqa: E712
    assert result.is_zero_day_exploit[0] == True  # noqa: E712
    assert result.is_zero_day_attack[0] == False  # noqa: E712


def test_analyze_slicing():
    """AnalysisResult supports slicing."""
    from vulnstate.analyzer import CVDAnalyzer

    arr = CVDArray.zeros(10)
    result = CVDAnalyzer.analyze(arr)

    # Slice first 5
    sliced = result[:5]
    assert len(sliced) == 5
    assert len(sliced.validity_int) == 5
