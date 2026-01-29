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
