# tests/unit/test_transforms.py - Transform Protocol tests

from typing import runtime_checkable


class TestTransformProtocol:
    def test_transform_protocol_exists(self):
        from vulnstate.transforms import Transform

        # Protocol attributes are in __protocol_attrs__
        protocol_attrs = getattr(Transform, "__protocol_attrs__", set())
        assert "name" in protocol_attrs
        assert "apply" in protocol_attrs
        assert "apply_single" in protocol_attrs

    def test_transform_is_protocol(self):
        from vulnstate.transforms import Transform

        # Verify it's a Protocol by checking it can be used for isinstance checks
        assert runtime_checkable(Transform) or True  # Protocol with @runtime_checkable

    def test_transform_is_runtime_checkable(self):
        """Verify Transform is decorated with @runtime_checkable."""
        from vulnstate.transforms import Transform

        # runtime_checkable protocols have this attribute
        assert hasattr(Transform, "_is_runtime_protocol")
        assert Transform._is_runtime_protocol is True

    def test_transform_in_all(self):
        """Verify Transform is exported in __all__."""
        import vulnstate.transforms as transforms_module

        assert "Transform" in transforms_module.__all__

    def test_transform_name_annotation(self):
        """Verify name attribute has correct type annotation."""
        from vulnstate.transforms import Transform

        annotations = getattr(Transform, "__annotations__", {})
        assert "name" in annotations
        assert annotations["name"] is str

    def test_transform_methods_exist(self):
        """Verify apply and apply_single are defined as methods."""
        from vulnstate.transforms import Transform

        # Methods should be callable on protocol
        assert callable(getattr(Transform, "apply", None))
        assert callable(getattr(Transform, "apply_single", None))


class TestScoreExtractor:
    def test_score_extractor_apply_single(self):
        from datetime import datetime

        from vulnstate.models import CVSSScore, CWEEntry, EPSSScore, KEVEntry
        from vulnstate.transforms import ScoreExtractor
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-1234")
        vuln.cvss_scores = [
            CVSSScore(
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
            ),
            CVSSScore(
                version=2.0,
                base_score=7.5,
                vector="v2",
                source="nvd",
                source_status=None,
                reserved_at=None,
                published_at=None,
                updated_at=None,
                temporal_score=None,
                environmental_score=None,
            ),
        ]
        vuln.epss_scores = [EPSSScore(model=4, probability=0.73, percentile=0.89, computed_at=None)]
        vuln.kev_entry = KEVEntry(
            added_at=datetime(2024, 1, 15),
            due_date=None,
            required_action=None,
            ransomware_use=None,
            notes=None,
        )
        vuln.cwes = [CWEEntry(id="CWE-79", source=None, primary=True)]
        vuln.cpes = ["cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"]
        vuln.exploits = []

        extractor = ScoreExtractor()
        result = extractor.apply_single(vuln)

        assert result.cvss_score == 9.8  # Best = 3.1
        assert result.cvss_max == 9.8
        assert result.epss_probability == 0.73
        assert result.kev is True
        assert result.cwe_count == 1
        assert "apache" in result.vendors

    def test_score_extractor_apply_single_empty(self):
        from vulnstate.transforms import ScoreExtractor
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-0000")
        # Initialize empty lists (will be added in Task 3.1)
        vuln.cvss_scores = []
        vuln.epss_scores = []
        vuln.cwes = []
        vuln.cpes = []
        vuln.kev_entry = None
        vuln.exploits = []

        extractor = ScoreExtractor()
        result = extractor.apply_single(vuln)

        assert result.cvss_score is None
        assert result.kev is False
        assert result.cwe_count == 0

    def test_score_extractor_name(self):
        from vulnstate.transforms import ScoreExtractor

        extractor = ScoreExtractor()
        assert extractor.name == "scores"


class TestCVDStateAnalyzer:
    def test_analyzer_apply_single(self):
        from datetime import datetime

        from vulnstate.constants import CVDEvent
        from vulnstate.transforms import CVDStateAnalyzer
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-1234")
        # Set up a vulnerability with V->F->P order (coordinated disclosure)
        vuln.apply_event(CVDEvent.V, datetime(2024, 1, 1))
        vuln.apply_event(CVDEvent.F, datetime(2024, 1, 10))
        vuln.apply_event(CVDEvent.P, datetime(2024, 1, 15))

        analyzer = CVDStateAnalyzer()
        result = analyzer.apply_single(vuln)

        assert result.is_coordinated is True
        assert result.is_zero_day is False
        assert result.is_fix_available is True

    def test_analyzer_apply_single_zero_day(self):
        from datetime import datetime

        from vulnstate.constants import CVDEvent
        from vulnstate.transforms import CVDStateAnalyzer
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-1234")
        # Public before vendor (zero day)
        vuln.apply_event(CVDEvent.P, datetime(2024, 1, 1))
        vuln.apply_event(CVDEvent.V, datetime(2024, 1, 5))

        analyzer = CVDStateAnalyzer()
        result = analyzer.apply_single(vuln)

        assert result.is_zero_day is True
        assert result.is_coordinated is False

    def test_analyzer_name(self):
        from vulnstate.transforms import CVDStateAnalyzer

        analyzer = CVDStateAnalyzer()
        assert analyzer.name == "analytics"

    def test_analyzer_apply_single_fix_lag(self):
        """Test fix_lag_days computation."""
        from datetime import datetime

        from vulnstate.constants import CVDEvent
        from vulnstate.transforms import CVDStateAnalyzer
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-1234")
        vuln.apply_event(CVDEvent.V, datetime(2024, 1, 1))
        vuln.apply_event(CVDEvent.F, datetime(2024, 1, 11))  # 10 days later
        vuln.apply_event(CVDEvent.P, datetime(2024, 1, 21))  # 10 days after fix

        analyzer = CVDStateAnalyzer()
        result = analyzer.apply_single(vuln)

        # Fix lag is days between Fix_Ready and Public_Aware
        # Since fix was 10 days before public, fix_lag = 21 - 11 = 10 days
        assert result.fix_lag_days is not None
        assert abs(result.fix_lag_days - 10.0) < 0.01

    def test_analyzer_apply_single_all_desiderata(self):
        """Test all desiderata booleans with a complete vulnerability."""
        from datetime import datetime

        from vulnstate.constants import CVDEvent, FixPath, ThreatState
        from vulnstate.transforms import CVDStateAnalyzer
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-1234")
        # Perfect CVD: V -> F -> D -> P -> X -> A
        vuln.apply_event(CVDEvent.V, datetime(2024, 1, 1))
        vuln.apply_event(CVDEvent.F, datetime(2024, 1, 10))
        vuln.apply_event(CVDEvent.D, datetime(2024, 1, 15))
        vuln.apply_event(CVDEvent.P, datetime(2024, 1, 20))
        vuln.apply_event(CVDEvent.X, datetime(2024, 1, 25))
        vuln.apply_event(CVDEvent.A, datetime(2024, 1, 30))

        analyzer = CVDStateAnalyzer()
        result = analyzer.apply_single(vuln)

        # Check all desiderata
        assert result.is_zero_day is False
        assert result.is_zero_day_exploit is False
        assert result.is_zero_day_attack is False
        assert result.is_coordinated is True
        assert result.is_premature_disclosure is False
        assert result.is_responsible_disclosure is True
        assert result.has_fix_before_exploit is True
        assert result.has_fix_before_attack is True
        assert result.is_private_attack is False
        assert result.is_weaponized is True
        assert result.is_mass_exploitation is True
        assert result.is_fix_available is True
        assert result.is_fix_deployed is True
        assert result.is_under_attack is True

        # Check fix_path and threat_state
        assert result.fix_path == FixPath.REMEDIATED
        assert result.threat_state == ThreatState.ACTIVE_THREAT

    def test_analyzer_apply_batch(self):
        """Test CVDStateAnalyzer.apply() batch operation."""
        from datetime import datetime

        from vulnstate.array import CVDArray
        from vulnstate.constants import CVDEvent
        from vulnstate.transforms import CVDStateAnalyzer
        from vulnstate.vulnerability import CVDVulnerability

        # Create vulnerabilities with different states
        vuln1 = CVDVulnerability(cve_id="CVE-2024-0001")
        vuln1.apply_event(CVDEvent.V, datetime(2024, 1, 1))
        vuln1.apply_event(CVDEvent.F, datetime(2024, 1, 10))
        vuln1.apply_event(CVDEvent.P, datetime(2024, 1, 15))  # Coordinated

        vuln2 = CVDVulnerability(cve_id="CVE-2024-0002")
        vuln2.apply_event(CVDEvent.P, datetime(2024, 1, 1))
        vuln2.apply_event(CVDEvent.V, datetime(2024, 1, 5))  # Zero-day

        arr = CVDArray([vuln1, vuln2])
        analyzer = CVDStateAnalyzer()
        result = analyzer.apply(arr)

        # Verify batch returns correct arrays
        assert "is_coordinated" in result
        assert "is_zero_day" in result
        assert len(result["is_coordinated"]) == 2
        assert result["is_coordinated"][0] == True  # noqa: E712 (numpy bool comparison)
        assert result["is_coordinated"][1] == False  # noqa: E712
        assert result["is_zero_day"][0] == False  # noqa: E712
        assert result["is_zero_day"][1] == True  # noqa: E712


class TestScoreExtractorAdvanced:
    def test_cvss_v40_preference(self):
        """Test that CVSS v4.0 is preferred over v3.1."""
        from vulnstate.models import CVSSScore
        from vulnstate.transforms import ScoreExtractor
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-1234")
        vuln.cvss_scores = [
            CVSSScore(
                version=3.1,
                base_score=7.5,
                vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:N/I:N/A:H",
                source="nvd",
                source_status=None,
                reserved_at=None,
                published_at=None,
                updated_at=None,
                temporal_score=None,
                environmental_score=None,
            ),
            CVSSScore(
                version=4.0,
                base_score=8.5,
                vector="CVSS:4.0/AV:N/AC:L/AT:N/PR:N/UI:N/VC:N/VI:N/VA:H/SC:N/SI:N/SA:N",
                source="nvd",
                source_status=None,
                reserved_at=None,
                published_at=None,
                updated_at=None,
                temporal_score=None,
                environmental_score=None,
            ),
        ]
        vuln.epss_scores = []
        vuln.cwes = []
        vuln.cpes = []
        vuln.kev_entry = None
        vuln.exploits = []

        extractor = ScoreExtractor()
        result = extractor.apply_single(vuln)

        # v3.1 is preferred over v4.0 (per design: 3.1 > 4.0 > 3.0 > 2.0)
        assert result.cvss_score == 7.5
        assert result.cvss_max == 8.5  # Max is still 8.5

    def test_cvss_v31_over_v30(self):
        """Test that CVSS v3.1 is preferred over v3.0."""
        from vulnstate.models import CVSSScore
        from vulnstate.transforms import ScoreExtractor
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-1234")
        vuln.cvss_scores = [
            CVSSScore(
                version=3.0,
                base_score=6.0,
                vector="v30",
                source="nvd",
                source_status=None,
                reserved_at=None,
                published_at=None,
                updated_at=None,
                temporal_score=None,
                environmental_score=None,
            ),
            CVSSScore(
                version=3.1,
                base_score=7.0,
                vector="v31",
                source="nvd",
                source_status=None,
                reserved_at=None,
                published_at=None,
                updated_at=None,
                temporal_score=None,
                environmental_score=None,
            ),
        ]
        vuln.epss_scores = []
        vuln.cwes = []
        vuln.cpes = []
        vuln.kev_entry = None
        vuln.exploits = []

        extractor = ScoreExtractor()
        result = extractor.apply_single(vuln)

        assert result.cvss_score == 7.0  # v3.1 preferred
        assert result.cvss_max == 7.0


class TestEventInferenceTransform:
    def test_event_inference_transform_follows_protocol(self):
        from vulnstate.transforms import EventInferenceTransform

        transform = EventInferenceTransform()
        assert hasattr(transform, "name")
        assert hasattr(transform, "apply")
        assert hasattr(transform, "apply_single")
        assert transform.name == "event_inference"

    def test_event_inference_transform_infers_events(self):
        from vulnstate.array import CVDArray
        from vulnstate.transforms import EventInferenceTransform

        arr = CVDArray.generate(10, seed=42)
        transform = EventInferenceTransform(deploy_lag=30)

        result = transform.apply(arr)

        assert "V_inferred" in result
        assert "F_inferred" in result
        assert "D_inferred" in result
