"""Tests for models.py dataclasses."""

import numpy as np


class TestEventMetadata:
    """Tests for EventMetadata dataclass."""

    def test_event_metadata_creation(self):
        """EventMetadata holds actor and notes."""
        from vulnstate.models import EventMetadata

        meta = EventMetadata(actor="security-team", notes="Found via fuzzing")
        assert meta.actor == "security-team"
        assert meta.notes == "Found via fuzzing"

    def test_event_metadata_defaults(self):
        """EventMetadata fields default to None."""
        from vulnstate.models import EventMetadata

        meta = EventMetadata()
        assert meta.actor is None
        assert meta.notes is None


class TestVulnerabilityIdentity:
    """Tests for VulnerabilityIdentity dataclass."""

    def test_vulnerability_identity_auto_generates_vuln_id(self):
        """VulnerabilityIdentity auto-generates vuln_id."""
        from vulnstate.models import VulnerabilityIdentity

        identity = VulnerabilityIdentity(cve_id="CVE-2024-1234")
        assert identity.vuln_id is not None
        assert len(identity.vuln_id) == 36  # UUID format
        assert identity.cve_id == "CVE-2024-1234"

    def test_vulnerability_identity_vendor_id(self):
        """VulnerabilityIdentity accepts vendor_id."""
        from vulnstate.models import VulnerabilityIdentity

        identity = VulnerabilityIdentity(cve_id="CVE-2024-1234", vendor_id="VENDOR-001")
        assert identity.vendor_id == "VENDOR-001"


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


class TestArrayCoreData:
    """Tests for ArrayCoreData dataclass."""

    def test_array_core_data_creation(self):
        """ArrayCoreData holds vulnerabilities, states, and vuln_ids."""
        from vulnstate.models import ArrayCoreData

        n = 5
        core = ArrayCoreData(
            vulnerabilities=np.array([None] * n, dtype=object),
            states=np.zeros(n, dtype=np.uint8),
            vuln_ids=np.array(["v1", "v2", "v3", "v4", "v5"], dtype=object),
        )
        assert len(core.states) == n
        assert core.states.dtype == np.uint8
        assert len(core.vuln_ids) == n


class TestArrayIdentifiers:
    """Tests for ArrayIdentifiers dataclass."""

    def test_array_identifiers_creation(self):
        """ArrayIdentifiers holds string arrays."""
        from vulnstate.models import ArrayIdentifiers

        ids = ArrayIdentifiers(
            vuln_id=np.array(["v1", "v2"], dtype=object),
            cve_id=np.array(["CVE-2024-1", "CVE-2024-2"], dtype=object),
            vendor_id=np.array([None, "Acme"], dtype=object),
        )
        assert ids.cve_id[0] == "CVE-2024-1"
        assert ids.vendor_id[1] == "Acme"


class TestArrayScoring:
    """Tests for ArrayScoring dataclass."""

    def test_array_scoring_creation(self):
        """ArrayScoring holds CVSS scoring data."""
        from vulnstate.models import ArrayScoring

        n = 3
        scoring = ArrayScoring(
            cvss_score=np.array([9.8, 7.5, 4.0], dtype=np.float32),
            cvss_exploitability=np.array([3.9, 3.0, 2.0], dtype=np.float32),
            cvss_impact=np.array([5.9, 5.2, 3.6], dtype=np.float32),
            cvss_vector_int=np.array([0x0FFF, 0x0AAA, 0x0555], dtype=np.uint16),
        )
        assert scoring.cvss_score[0] == 9.8
        assert len(scoring.cvss_impact) == n


class TestArrayEnrichment:
    """Tests for ArrayEnrichment dataclass."""

    def test_array_enrichment_creation(self):
        """ArrayEnrichment holds EPSS and KEV data."""
        from vulnstate.models import ArrayEnrichment

        enrichment = ArrayEnrichment(
            epss=np.array([0.95, 0.15], dtype=np.float32),
            epss_percentile=np.array([0.99, 0.80], dtype=np.float32),
            kev=np.array([True, False], dtype=bool),
            kev_date=np.array(["2024-01-15", "NaT"], dtype="datetime64[s]"),
        )
        assert enrichment.epss[0] == 0.95
        assert enrichment.kev[0] == True  # noqa: E712


class TestArrayProbabilities:
    """Tests for ArrayProbabilities dataclass."""

    def test_array_probabilities_creation(self):
        """ArrayProbabilities holds flexible probability inputs."""
        from vulnstate.models import ArrayProbabilities

        n = 3
        probs = ArrayProbabilities(
            base_threat=np.array([0.5, 0.7, np.nan], dtype=np.float32),
            xa_split_ratio=np.full(n, 0.5, dtype=np.float32),
            threat_multiplier=np.ones(n, dtype=np.float32),
            override_X=np.full(n, np.nan, dtype=np.float32),
            override_A=np.full(n, np.nan, dtype=np.float32),
        )
        assert probs.base_threat[0] == 0.5
        assert np.isnan(probs.override_X[0])

    def test_array_probabilities_slicing(self):
        """ArrayProbabilities supports slicing."""
        from vulnstate.models import ArrayProbabilities

        n = 5
        probs = ArrayProbabilities(
            base_threat=np.arange(n, dtype=np.float32) / 10,
            xa_split_ratio=np.full(n, 0.5, dtype=np.float32),
            threat_multiplier=np.ones(n, dtype=np.float32),
            override_X=np.full(n, np.nan, dtype=np.float32),
            override_A=np.full(n, np.nan, dtype=np.float32),
        )
        sliced = probs[:3]
        assert len(sliced.base_threat) == 3


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
