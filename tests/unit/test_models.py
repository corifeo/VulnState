"""Tests for models.py dataclasses."""

from dataclasses import FrozenInstanceError
from datetime import datetime

import numpy as np
import pytest

pytestmark = pytest.mark.unit


class TestCVSSScore:
    """Tests for CVSSScore dataclass."""

    def test_cvss_score_creation(self):
        """CVSSScore holds all CVSS scoring data."""
        from vulnstate.models import CVSSScore

        score = CVSSScore(
            version=3.1,
            base_score=9.8,
            vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            source="nvd",
            source_status="Analyzed",
            reserved_at=None,
            published_at=datetime(2024, 1, 15),
            updated_at=datetime(2024, 1, 20),
            temporal_score=None,
            environmental_score=None,
        )
        assert score.version == 3.1
        assert score.base_score == 9.8
        assert score.source == "nvd"

    def test_cvss_score_frozen(self):
        """CVSSScore is immutable (frozen dataclass)."""
        from vulnstate.models import CVSSScore

        score = CVSSScore(
            version=3.1,
            base_score=9.8,
            vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            source="nvd",
            source_status=None,
            reserved_at=None,
            published_at=None,
            updated_at=None,
            temporal_score=None,
            environmental_score=None,
        )
        with pytest.raises(FrozenInstanceError):
            score.base_score = 5.0

    def test_cvss_score_to_dict(self):
        """CVSSScore.to_dict() serializes, omitting None values."""
        from vulnstate.models import CVSSScore

        score = CVSSScore(
            version=3.1,
            base_score=9.8,
            vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            source="nvd",
            source_status="Analyzed",
            reserved_at=None,
            published_at=datetime(2024, 1, 15),
            updated_at=None,
            temporal_score=None,
            environmental_score=None,
        )
        d = score.to_dict()
        assert d["version"] == 3.1
        assert d["source"] == "nvd"
        assert "reserved_at" not in d  # None fields omitted
        assert "updated_at" not in d  # None fields omitted
        assert "published_at" in d  # Non-None field included

    def test_cvss_score_from_dict(self):
        """CVSSScore.from_dict() deserializes, defaulting missing to None."""
        from vulnstate.models import CVSSScore

        d = {
            "version": 3.1,
            "base_score": 9.8,
            "vector": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            "source": "nvd",
        }
        score = CVSSScore.from_dict(d)
        assert score.version == 3.1
        assert score.source_status is None

    def test_cvss_score_parse_vector_v31(self):
        """CVSSScore.parse_vector() parses CVSS 3.1 vector strings."""
        from vulnstate.models import CVSSScore

        score = CVSSScore(
            version=3.1,
            base_score=9.8,
            vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
            source="nvd",
            source_status=None,
            reserved_at=None,
            published_at=None,
            updated_at=None,
            temporal_score=None,
            environmental_score=None,
        )
        metrics = score.parse_vector()
        assert metrics["AV"] == "N"
        assert metrics["AC"] == "L"
        assert metrics["C"] == "H"


class TestVulnerabilityIdentity:
    """Tests for VulnerabilityIdentity dataclass."""

    def test_vulnerability_identity_auto_generates_internal_id(self):
        """VulnerabilityIdentity auto-generates internal_id."""
        from vulnstate.models import VulnerabilityIdentity

        identity = VulnerabilityIdentity(cve_id="CVE-2024-1234")
        assert identity.internal_id is not None
        assert len(identity.internal_id) == 36  # UUID format
        assert identity.cve_id == "CVE-2024-1234"


class TestArrayTimestamps:
    """Tests for ArrayTimestamps dataclass."""

    def test_array_timestamps_creation(self):
        """ArrayTimestamps holds 6 timestamp arrays."""
        from vulnstate.models import ArrayTimestamps

        n = 10
        ts = ArrayTimestamps(
            V=np.empty(n, dtype="datetime64[s]"),
            F=np.empty(n, dtype="datetime64[s]"),
            D=np.empty(n, dtype="datetime64[s]"),
            P=np.empty(n, dtype="datetime64[s]"),
            X=np.empty(n, dtype="datetime64[s]"),
            A=np.empty(n, dtype="datetime64[s]"),
        )
        assert len(ts.V) == n
        assert ts.V.dtype == np.dtype("datetime64[s]")

    def test_array_timestamps_slicing(self):
        """ArrayTimestamps supports slicing."""
        from vulnstate.models import ArrayTimestamps

        n = 10
        base = np.datetime64("2024-01-01", "s")
        ts = ArrayTimestamps(
            V=base + np.arange(n).astype("timedelta64[s]"),
            F=base + np.arange(n).astype("timedelta64[s]"),
            D=base + np.arange(n).astype("timedelta64[s]"),
            P=base + np.arange(n).astype("timedelta64[s]"),
            X=base + np.arange(n).astype("timedelta64[s]"),
            A=base + np.arange(n).astype("timedelta64[s]"),
        )
        sliced = ts[:5]
        assert len(sliced.V) == 5


class TestArrayIdentifiers:
    """Tests for ArrayIdentifiers dataclass."""

    def test_array_identifiers_creation(self):
        """ArrayIdentifiers holds string arrays."""
        from vulnstate.models import ArrayIdentifiers

        ids = ArrayIdentifiers(
            internal_id=np.array(["v1", "v2"], dtype=object),
            cve_id=np.array(["CVE-2024-1", "CVE-2024-2"], dtype=object),
        )
        assert ids.cve_id[0] == "CVE-2024-1"
        assert ids.internal_id[1] == "v2"


class TestArrayState:
    """Tests for ArrayState dataclass."""

    def test_array_state_creation(self):
        """ArrayState holds bitmask array."""
        from vulnstate.models import ArrayState

        state = ArrayState(bitmask=np.array([0, 1, 3, 7, 63], dtype=np.uint8))
        assert len(state) == 5
        assert state.bitmask[0] == 0  # vfdpxa (no events)
        assert state.bitmask[4] == 63  # VFDPXA (all events)

    def test_array_state_slicing(self):
        """ArrayState supports slicing."""
        from vulnstate.models import ArrayState

        state = ArrayState(bitmask=np.array([0, 1, 3, 7, 15], dtype=np.uint8))
        sliced = state[1:4]
        assert len(sliced) == 3
        assert sliced.bitmask[0] == 1

    def test_array_state_boolean_indexing(self):
        """ArrayState supports boolean mask indexing."""
        from vulnstate.models import ArrayState

        state = ArrayState(bitmask=np.array([0, 1, 3, 7, 15], dtype=np.uint8))
        mask = np.array([True, False, True, False, True])
        filtered = state[mask]
        assert len(filtered) == 3
        assert list(filtered.bitmask) == [0, 3, 15]

    def test_array_state_default_empty(self):
        """ArrayState defaults to empty array."""
        from vulnstate.models import ArrayState

        state = ArrayState()
        assert len(state) == 0
        assert state.bitmask.dtype == np.uint8


class TestAnalysisResult:
    """Tests for AnalysisResult dataclass."""

    def test_analysis_result_basic(self):
        """AnalysisResult holds computed analytics."""
        from vulnstate.models import AnalysisResult

        n = 2
        result = AnalysisResult(
            # Group 1: Validity
            validity_int=np.array([0, 1], dtype=np.uint8),
            event_count=np.array([3, 6], dtype=np.uint8),
            is_complete=np.array([False, True]),
            # Group 2: Pairs
            desiderata_mask=np.array([0b111, 0b000], dtype=np.uint16),
            anti_desiderata_mask=np.array([0b000, 0b111], dtype=np.uint16),
            pair_observed_mask=np.array([0b111111, 0b111111111111111], dtype=np.uint16),
            # Group 3: Zero-day
            is_zero_day=np.array([True, False]),
            is_zero_day_exploit=np.array([True, False]),
            is_zero_day_attack=np.array([False, False]),
            # Group 4: Coordination
            is_coordinated=np.array([True, False]),
            is_premature_disclosure=np.array([False, True]),
            is_responsible_disclosure=np.array([True, False]),
            # Group 5: Fix Effectiveness
            has_fix_before_exploit=np.array([True, False]),
            has_fix_before_attack=np.array([True, False]),
            has_deployment_before_exploit=np.array([True, False]),
            has_deployment_before_attack=np.array([True, False]),
            # Group 6: Threat Characteristics
            is_private_attack=np.array([False, False]),
            is_weaponized=np.array([True, False]),
            is_mass_exploitation=np.array([False, True]),
        )
        assert result.validity_int[0] == 0
        assert result.is_zero_day[0] == True  # noqa: E712
        assert len(result) == n

    def test_analysis_result_slicing(self):
        """AnalysisResult supports slicing."""
        from vulnstate.models import AnalysisResult

        n = 3
        result = AnalysisResult(
            validity_int=np.arange(n, dtype=np.uint8),
            event_count=np.arange(n, dtype=np.uint8),
            is_complete=np.array([False, True, False]),
            desiderata_mask=np.arange(n, dtype=np.uint16),
            anti_desiderata_mask=np.arange(n, dtype=np.uint16),
            pair_observed_mask=np.arange(n, dtype=np.uint16),
            is_zero_day=np.array([True, False, True]),
            is_zero_day_exploit=np.array([True, False, False]),
            is_zero_day_attack=np.array([False, False, True]),
            is_coordinated=np.array([True, False, True]),
            is_premature_disclosure=np.array([False, True, False]),
            is_responsible_disclosure=np.array([True, False, True]),
            has_fix_before_exploit=np.array([True, False, True]),
            has_fix_before_attack=np.array([True, False, True]),
            has_deployment_before_exploit=np.array([True, False, True]),
            has_deployment_before_attack=np.array([True, False, True]),
            is_private_attack=np.array([False, False, False]),
            is_weaponized=np.array([True, False, True]),
            is_mass_exploitation=np.array([False, True, False]),
        )
        sliced = result[:2]
        assert len(sliced.validity_int) == 2

        # Boolean mask
        mask = np.array([True, False, True])
        filtered = result[mask]
        assert len(filtered.validity_int) == 2

    def test_analysis_result_groups_7_11(self):
        """AnalysisResult Groups 7-11 have defaults and support slicing."""
        from vulnstate.models import AnalysisResult

        n = 3
        result = AnalysisResult(
            # Groups 1-6 required
            validity_int=np.arange(n, dtype=np.uint8),
            event_count=np.arange(n, dtype=np.uint8),
            is_complete=np.array([False, True, False]),
            desiderata_mask=np.arange(n, dtype=np.uint16),
            anti_desiderata_mask=np.arange(n, dtype=np.uint16),
            pair_observed_mask=np.arange(n, dtype=np.uint16),
            is_zero_day=np.array([True, False, True]),
            is_zero_day_exploit=np.array([True, False, False]),
            is_zero_day_attack=np.array([False, False, True]),
            is_coordinated=np.array([True, False, True]),
            is_premature_disclosure=np.array([False, True, False]),
            is_responsible_disclosure=np.array([True, False, True]),
            has_fix_before_exploit=np.array([True, False, True]),
            has_fix_before_attack=np.array([True, False, True]),
            has_deployment_before_exploit=np.array([True, False, True]),
            has_deployment_before_attack=np.array([True, False, True]),
            is_private_attack=np.array([False, False, False]),
            is_weaponized=np.array([True, False, True]),
            is_mass_exploitation=np.array([False, True, False]),
            # Groups 7-11 optional with explicit values
            fix_path_int=np.array([0, 1, 3], dtype=np.uint8),
            threat_state_int=np.array([0, 4, 7], dtype=np.uint8),
            desiderata_score=np.array([1.0, 0.5, 0.8], dtype=np.float32),
            fix_lag_days=np.array([10.0, 20.0, np.nan], dtype=np.float32),
            can_transition_mask=np.array([0b111111, 0b000000, 0b010101], dtype=np.uint8),
            prob_X=np.array([0.5, 0.3, 0.1], dtype=np.float32),
        )

        # Test Groups 7-11 values
        assert result.fix_path_int[1] == 1
        assert result.desiderata_score[0] == 1.0
        assert np.isnan(result.fix_lag_days[2])
        assert result.prob_X[0] == 0.5

        # Test slicing with Groups 7-11
        sliced = result[:2]
        assert len(sliced.fix_path_int) == 2
        assert sliced.fix_path_int[0] == 0

    def test_analysis_result_transition_helpers(self):
        """AnalysisResult provides transition check helpers."""
        from vulnstate.models import AnalysisResult

        n = 2
        result = AnalysisResult(
            # Groups 1-6 required (minimal)
            validity_int=np.zeros(n, dtype=np.uint8),
            event_count=np.zeros(n, dtype=np.uint8),
            is_complete=np.zeros(n, dtype=bool),
            desiderata_mask=np.zeros(n, dtype=np.uint16),
            anti_desiderata_mask=np.zeros(n, dtype=np.uint16),
            pair_observed_mask=np.zeros(n, dtype=np.uint16),
            is_zero_day=np.zeros(n, dtype=bool),
            is_zero_day_exploit=np.zeros(n, dtype=bool),
            is_zero_day_attack=np.zeros(n, dtype=bool),
            is_coordinated=np.zeros(n, dtype=bool),
            is_premature_disclosure=np.zeros(n, dtype=bool),
            is_responsible_disclosure=np.zeros(n, dtype=bool),
            has_fix_before_exploit=np.zeros(n, dtype=bool),
            has_fix_before_attack=np.zeros(n, dtype=bool),
            has_deployment_before_exploit=np.zeros(n, dtype=bool),
            has_deployment_before_attack=np.zeros(n, dtype=bool),
            is_private_attack=np.zeros(n, dtype=bool),
            is_weaponized=np.zeros(n, dtype=bool),
            is_mass_exploitation=np.zeros(n, dtype=bool),
            # Group 10: Transitions
            can_transition_mask=np.array([0b111111, 0b000001], dtype=np.uint8),
        )

        # All transitions possible for first
        assert result.can_transition_V()[0] == True  # noqa: E712
        assert result.can_transition_F()[0] == True  # noqa: E712
        assert result.can_transition_A()[0] == True  # noqa: E712

        # Only V possible for second
        assert result.can_transition_V()[1] == True  # noqa: E712
        assert result.can_transition_F()[1] == False  # noqa: E712

        # Generic helper
        assert result.can_transition(0)[0] == True  # V  # noqa: E712
        assert result.can_transition(5)[1] == False  # A  # noqa: E712

    def test_analysis_result_where_desiderata(self):
        """AnalysisResult provides desiderata filtering."""
        from vulnstate.models import AnalysisResult

        n = 3
        result = AnalysisResult(
            # Groups 1-6 required (minimal)
            validity_int=np.zeros(n, dtype=np.uint8),
            event_count=np.zeros(n, dtype=np.uint8),
            is_complete=np.zeros(n, dtype=bool),
            desiderata_mask=np.array([0b111, 0b101, 0b001], dtype=np.uint16),
            anti_desiderata_mask=np.zeros(n, dtype=np.uint16),
            pair_observed_mask=np.zeros(n, dtype=np.uint16),
            is_zero_day=np.zeros(n, dtype=bool),
            is_zero_day_exploit=np.zeros(n, dtype=bool),
            is_zero_day_attack=np.zeros(n, dtype=bool),
            is_coordinated=np.zeros(n, dtype=bool),
            is_premature_disclosure=np.zeros(n, dtype=bool),
            is_responsible_disclosure=np.zeros(n, dtype=bool),
            has_fix_before_exploit=np.zeros(n, dtype=bool),
            has_fix_before_attack=np.zeros(n, dtype=bool),
            has_deployment_before_exploit=np.zeros(n, dtype=bool),
            has_deployment_before_attack=np.zeros(n, dtype=bool),
            is_private_attack=np.zeros(n, dtype=bool),
            is_weaponized=np.zeros(n, dtype=bool),
            is_mass_exploitation=np.zeros(n, dtype=bool),
        )

        # Require bit 0 and 2 (0b101)
        mask = result.where_desiderata(required=0b101)
        assert mask[0] == True  # 0b111 has both  # noqa: E712
        assert mask[1] == True  # 0b101 has both  # noqa: E712
        assert mask[2] == False  # 0b001 missing bit 2  # noqa: E712

        # Forbid bit 1
        mask = result.where_desiderata(forbidden=0b010)
        assert mask[0] == False  # 0b111 has bit 1  # noqa: E712
        assert mask[1] == True  # 0b101 no bit 1  # noqa: E712
        assert mask[2] == True  # 0b001 no bit 1  # noqa: E712


class TestCachedAnalyticsProperty:
    """Tests for CachedAnalyticsProperty descriptor."""

    def test_cached_analytics_property_returns_bool(self):
        """CachedAnalyticsProperty returns bool for single vulnerability."""
        from vulnstate import CVDEvent, CVDVulnerability

        vuln = CVDVulnerability()
        vuln.apply_event(CVDEvent.X)
        result = vuln.is_zero_day
        assert type(result) is bool
        assert result is True

    def test_cached_analytics_invalidation_on_apply_event(self):
        """CachedAnalyticsProperty cache invalidates on apply_event."""
        from vulnstate import CVDEvent, CVDVulnerability

        vuln = CVDVulnerability()
        vuln.apply_event(CVDEvent.X)
        _ = vuln.is_zero_day  # Populate cache
        assert vuln._analytics is not None

        vuln.apply_event(CVDEvent.V)  # State change
        assert vuln._analytics is None  # Cache invalidated

    def test_cached_analytics_invalidation_on_rollback(self):
        """CachedAnalyticsProperty cache invalidates on rollback_event."""
        from vulnstate import CVDEvent, CVDVulnerability

        vuln = CVDVulnerability()
        vuln.apply_event(CVDEvent.V)
        vuln.apply_event(CVDEvent.X)
        _ = vuln.is_zero_day  # Populate cache
        assert vuln._analytics is not None

        vuln.rollback_event()  # Rollback X
        assert vuln._analytics is None  # Cache invalidated

    def test_all_analytics_properties_return_bool(self):
        """All 13 CachedAnalyticsProperty descriptors return bool."""
        from vulnstate import CVDEvent, CVDVulnerability

        vuln = CVDVulnerability()
        vuln.apply_event(CVDEvent.V)
        vuln.apply_event(CVDEvent.F)
        vuln.apply_event(CVDEvent.X)

        # All 13 properties should return bool
        properties = [
            "is_zero_day",
            "is_zero_day_exploit",
            "is_zero_day_attack",
            "is_coordinated",
            "is_responsible_disclosure",
            "is_premature_disclosure",
            "has_fix_before_exploit",
            "has_fix_before_attack",
            "has_deployment_before_exploit",
            "has_deployment_before_attack",
            "is_private_attack",
            "is_weaponized",
            "is_mass_exploitation",
        ]

        for prop in properties:
            result = getattr(vuln, prop)
            assert type(result) is bool, f"{prop} returned {type(result)}, expected bool"

    def test_cached_analytics_property_class_access(self):
        """Accessing property on class returns descriptor."""
        from vulnstate import CVDVulnerability
        from vulnstate.models import CachedAnalyticsProperty

        prop = CVDVulnerability.is_zero_day
        assert isinstance(prop, CachedAnalyticsProperty)

    def test_cached_analytics_property_is_read_only(self):
        """CachedAnalyticsProperty raises AttributeError on assignment."""
        import pytest

        from vulnstate import CVDEvent, CVDVulnerability

        vuln = CVDVulnerability()
        vuln.apply_event(CVDEvent.V)

        with pytest.raises(AttributeError, match="is_zero_day is read-only"):
            vuln.is_zero_day = True


class TestEPSSScore:
    def test_epss_score_creation(self):
        from vulnstate.models import EPSSScore

        score = EPSSScore(
            model=4, probability=0.73, percentile=0.89, computed_at=datetime(2024, 1, 15)
        )
        assert score.model == 4
        assert score.probability == 0.73

    def test_epss_score_roundtrip(self):
        from vulnstate.models import EPSSScore

        score = EPSSScore(model=4, probability=0.73, percentile=0.89, computed_at=None)
        d = score.to_dict()
        restored = EPSSScore.from_dict(d)
        assert restored == score


class TestCWEEntry:
    def test_cwe_entry_creation(self):
        from vulnstate.models import CWEEntry

        cwe = CWEEntry(id="CWE-79", source="nvd@nist.gov", primary=True)
        assert cwe.id == "CWE-79"

    def test_cwe_entry_roundtrip(self):
        from vulnstate.models import CWEEntry

        cwe = CWEEntry(id="CWE-79", source=None, primary=False)
        d = cwe.to_dict()
        restored = CWEEntry.from_dict(d)
        assert restored == cwe


class TestKEVEntry:
    def test_kev_entry_creation(self):
        from vulnstate.models import KEVEntry

        kev = KEVEntry(
            added_at=datetime(2024, 1, 15),
            due_date=datetime(2024, 2, 15),
            required_action="Apply updates",
            ransomware_use=True,
            notes="Critical vulnerability",
        )
        assert kev.ransomware_use is True

    def test_kev_entry_roundtrip(self):
        from vulnstate.models import KEVEntry

        kev = KEVEntry(
            added_at=datetime(2024, 1, 15),
            due_date=None,
            required_action=None,
            ransomware_use=None,
            notes=None,
        )
        d = kev.to_dict()
        restored = KEVEntry.from_dict(d)
        assert restored == kev


class TestExploitReference:
    def test_exploit_reference_creation(self):
        from vulnstate.models import ExploitReference

        ref = ExploitReference(
            source="metasploit", reference="exploit/multi/http/log4j", metadata={"verified": True}
        )
        assert ref.source == "metasploit"

    def test_exploit_reference_roundtrip(self):
        from vulnstate.models import ExploitReference

        ref = ExploitReference(source="exploitdb", reference="12345", metadata=None)
        d = ref.to_dict()
        restored = ExploitReference.from_dict(d)
        assert restored == ref


class TestScoreResult:
    def test_score_result_creation(self):
        from vulnstate.models import ScoreResult

        result = ScoreResult(
            cvss_score=9.8,
            cvss_max=9.8,
            epss_probability=0.73,
            epss_percentile=0.89,
            kev=True,
            has_exploit=True,
            cwe_count=2,
            cpe_count=5,
            vendors={"apache"},
            products={"log4j"},
        )
        assert result.cvss_score == 9.8
        assert result.kev is True
        assert "apache" in result.vendors


class TestArraySource:
    def test_array_source_creation(self):
        import numpy as np

        from vulnstate.models import ArraySource

        source = ArraySource(
            cvss_scores=np.array([[], []], dtype=object),
            epss_scores=np.array([[], []], dtype=object),
            cwes=np.array([[], []], dtype=object),
            cpes=np.array([[], []], dtype=object),
            kev=np.array([None, None], dtype=object),
            exploits=np.array([[], []], dtype=object),
        )
        assert len(source.cvss_scores) == 2

    def test_array_source_slicing(self):
        import numpy as np

        from vulnstate.models import ArraySource, CVSSScore

        score1 = CVSSScore(
            version=3.1,
            base_score=9.8,
            vector="v",
            source="nvd",
            source_status=None,
            reserved_at=None,
            published_at=None,
            updated_at=None,
            temporal_score=None,
            environmental_score=None,
        )
        source = ArraySource(
            cvss_scores=np.array([[score1], []], dtype=object),
            epss_scores=np.array([[], []], dtype=object),
            cwes=np.array([[], []], dtype=object),
            cpes=np.array([[], []], dtype=object),
            kev=np.array([None, None], dtype=object),
            exploits=np.array([[], []], dtype=object),
        )
        sliced = source[0:1]
        assert len(sliced.cvss_scores) == 1
        assert sliced.cvss_scores[0][0].base_score == 9.8


class TestAnalyticsResult:
    def test_analytics_result_creation(self):
        from vulnstate.constants import FixPath, ThreatState
        from vulnstate.models import AnalyticsResult

        result = AnalyticsResult(
            fix_path=FixPath.REMEDIATED,
            threat_state=ThreatState.LATENT,
            is_zero_day=False,
            is_zero_day_exploit=False,
            is_zero_day_attack=False,
            is_coordinated=True,
            is_premature_disclosure=False,
            is_responsible_disclosure=True,
            has_fix_before_exploit=True,
            has_fix_before_attack=True,
            is_private_attack=False,
            is_weaponized=True,
            is_mass_exploitation=False,
            is_fix_available=True,
            is_fix_deployed=True,
            is_under_attack=True,
            fix_lag_days=30.0,
        )
        assert result.is_coordinated is True
        assert result.fix_lag_days == 30.0


class TestCVSSMetrics:
    def test_create_empty(self):
        from vulnstate.models import CVSSMetrics

        metrics = CVSSMetrics.empty(5)
        assert len(metrics.attack_vector) == 5
        assert metrics.attack_vector.dtype == object

    def test_all_fields_same_length(self):
        import numpy as np

        from vulnstate.models import CVSSMetrics

        metrics = CVSSMetrics(
            attack_vector=np.array(["N", "L"], dtype=object),
            attack_complexity=np.array(["L", "H"], dtype=object),
            privileges_required=np.array(["N", "L"], dtype=object),
            user_interaction=np.array(["N", "R"], dtype=object),
            scope=np.array(["U", "C"], dtype=object),
            confidentiality_impact=np.array(["H", "L"], dtype=object),
            integrity_impact=np.array(["H", "L"], dtype=object),
            availability_impact=np.array(["H", "N"], dtype=object),
        )
        assert len(metrics.attack_vector) == 2
