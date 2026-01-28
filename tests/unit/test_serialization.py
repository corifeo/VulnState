"""
Serialization Tests - CVDIO serialization and deserialization

Tests:
- TestVulnerabilitySerializationV2: Tests for new API v2 enrichment fields (CVDIO)
- TestCVDVulnerabilityV2Serialization: Tests for CVDVulnerability.to_dict/from_dict API

Layer: Test
"""

from datetime import datetime


class TestCVDVulnerabilityV2Serialization:
    """Tests for CVDVulnerability.to_dict() and from_dict() API with new v2 fields."""

    def test_to_dict_includes_new_fields(self):
        """Test that vuln.to_dict() includes new enrichment fields."""
        from vulnstate.models import CVSSScore, CWEEntry
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-1234")
        vuln.cvss_scores = [
            CVSSScore(
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
        ]
        vuln.cwes = [CWEEntry(id="CWE-79", source="nvd@nist.gov", primary=True)]
        vuln.cpes = ["cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"]

        d = vuln.to_dict()

        assert "cvss_scores" in d
        assert len(d["cvss_scores"]) == 1
        assert d["cvss_scores"][0]["base_score"] == 9.8
        assert "cwes" in d
        assert "cpes" in d

    def test_from_dict_restores_new_fields(self):
        """Test that CVDVulnerability.from_dict() restores new enrichment fields."""
        from vulnstate.vulnerability import CVDVulnerability

        d = {
            "cve_id": "CVE-2024-1234",
            "cvss_scores": [{"version": 3.1, "base_score": 9.8, "vector": "v", "source": "nvd"}],
            "epss_scores": [{"model": 4, "probability": 0.73, "percentile": 0.89}],
            "cwes": [{"id": "CWE-79", "primary": True}],
            "cpes": ["cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"],
            "kev_entry": {"added_at": "2024-01-15T00:00:00"},
            "exploits": [],
        }

        vuln = CVDVulnerability.from_dict(d)

        assert len(vuln.cvss_scores) == 1
        assert vuln.cvss_scores[0].base_score == 9.8
        assert len(vuln.epss_scores) == 1
        assert vuln.epss_scores[0].probability == 0.73
        assert len(vuln.cwes) == 1
        assert vuln.kev_entry is not None
        assert len(vuln.cpes) == 1

    def test_roundtrip_new_fields(self):
        """Test roundtrip serialization via CVDVulnerability API."""
        from vulnstate.models import CVSSScore, EPSSScore
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
            )
        ]
        vuln.epss_scores = [EPSSScore(model=4, probability=0.73, percentile=0.89, computed_at=None)]

        d = vuln.to_dict()
        restored = CVDVulnerability.from_dict(d)

        assert len(restored.cvss_scores) == 1
        assert restored.cvss_scores[0].base_score == 9.8
        assert len(restored.epss_scores) == 1


class TestVulnerabilitySerializationV2:
    """Tests for serialization of new API v2 enrichment fields."""

    def test_to_dict_with_new_fields(self):
        """Test that to_dict includes new enrichment fields."""
        from vulnstate.io import CVDIO
        from vulnstate.models import CVSSScore
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-1234")
        vuln.cvss_scores = [
            CVSSScore(
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
        ]

        d = CVDIO.to_dict(vuln)
        assert "cvss_scores" in d
        assert len(d["cvss_scores"]) == 1
        assert d["cvss_scores"][0]["base_score"] == 9.8

    def test_from_dict_with_new_fields(self):
        """Test that from_dict restores new enrichment fields."""
        from vulnstate.io import CVDIO

        d = {
            "cve_id": "CVE-2024-1234",
            "cvss_scores": [{"version": 3.1, "base_score": 9.8, "vector": "v", "source": "nvd"}],
            "cwes": [{"id": "CWE-79", "primary": True}],
        }

        vuln = CVDIO.from_dict(d)
        assert len(vuln.cvss_scores) == 1
        assert vuln.cvss_scores[0].source == "nvd"

    def test_roundtrip_with_new_fields(self):
        """Test roundtrip serialization of new enrichment fields."""
        from vulnstate.io import CVDIO
        from vulnstate.models import CVSSScore, EPSSScore, KEVEntry
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-1234")
        vuln.cvss_scores = [
            CVSSScore(
                version=3.1,
                base_score=9.8,
                vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                source="nvd",
                source_status="Analyzed",
                reserved_at=None,
                published_at=datetime(2024, 1, 1),
                updated_at=None,
                temporal_score=None,
                environmental_score=None,
            )
        ]
        vuln.epss_scores = [EPSSScore(model=4, probability=0.5, percentile=0.7, computed_at=None)]
        vuln.kev_entry = KEVEntry(
            added_at=datetime(2024, 1, 15),
            due_date=None,
            required_action=None,
            ransomware_use=True,
            notes=None,
        )

        io = CVDIO()
        d = io.to_dict(vuln)
        restored = io.from_dict(d)

        assert len(restored.cvss_scores) == 1
        assert restored.cvss_scores[0].source_status == "Analyzed"
        assert restored.kev_entry.ransomware_use is True

    def test_to_dict_with_cwes(self):
        """Test that to_dict includes CWE entries."""
        from vulnstate.io import CVDIO
        from vulnstate.models import CWEEntry
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-5678")
        vuln.cwes = [
            CWEEntry(id="CWE-79", source="nvd", primary=True),
            CWEEntry(id="CWE-89", source="nvd", primary=False),
        ]

        d = CVDIO.to_dict(vuln)
        assert "cwes" in d
        assert len(d["cwes"]) == 2
        assert d["cwes"][0]["id"] == "CWE-79"
        assert d["cwes"][1]["primary"] is False

    def test_to_dict_with_cpes(self):
        """Test that to_dict includes CPE identifiers."""
        from vulnstate.io import CVDIO
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-5678")
        vuln.cpes = [
            "cpe:2.3:a:vendor:product:1.0:*:*:*:*:*:*:*",
            "cpe:2.3:a:vendor:product:2.0:*:*:*:*:*:*:*",
        ]

        d = CVDIO.to_dict(vuln)
        assert "cpes" in d
        assert len(d["cpes"]) == 2

    def test_to_dict_with_exploits(self):
        """Test that to_dict includes exploit references."""
        from vulnstate.io import CVDIO
        from vulnstate.models import ExploitReference
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-5678")
        vuln.exploits = [
            ExploitReference(
                source="exploit-db",
                reference="https://exploit-db.com/exploits/12345",
                metadata={"verified": True},
            )
        ]

        d = CVDIO.to_dict(vuln)
        assert "exploits" in d
        assert len(d["exploits"]) == 1
        assert d["exploits"][0]["source"] == "exploit-db"

    def test_from_dict_with_all_new_fields(self):
        """Test that from_dict restores all new enrichment fields."""
        from vulnstate.io import CVDIO

        d = {
            "cve_id": "CVE-2024-9999",
            "cvss_scores": [{"version": 3.1, "base_score": 7.5, "vector": "v", "source": "vendor"}],
            "epss_scores": [{"model": 4, "probability": 0.3, "percentile": 0.5}],
            "cwes": [{"id": "CWE-89", "primary": True}],
            "cpes": ["cpe:2.3:a:vendor:product:1.0:*:*:*:*:*:*:*"],
            "kev_entry": {"added_at": "2024-02-01T00:00:00", "ransomware_use": False},
            "exploits": [{"source": "github", "reference": "https://github.com/exploit/poc"}],
        }

        vuln = CVDIO.from_dict(d)

        assert len(vuln.cvss_scores) == 1
        assert vuln.cvss_scores[0].base_score == 7.5

        assert len(vuln.epss_scores) == 1
        assert vuln.epss_scores[0].probability == 0.3

        assert len(vuln.cwes) == 1
        assert vuln.cwes[0].id == "CWE-89"

        assert len(vuln.cpes) == 1
        assert "vendor:product" in vuln.cpes[0]

        assert vuln.kev_entry is not None
        assert vuln.kev_entry.ransomware_use is False

        assert len(vuln.exploits) == 1
        assert vuln.exploits[0].source == "github"

    def test_roundtrip_empty_new_fields(self):
        """Test roundtrip with empty new fields does not add keys."""
        from vulnstate.io import CVDIO
        from vulnstate.vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id="CVE-2024-0000")
        # Leave all new fields empty/default

        d = CVDIO.to_dict(vuln)

        # Empty lists should not be serialized
        assert "cvss_scores" not in d or d["cvss_scores"] == []
        assert "epss_scores" not in d or d["epss_scores"] == []
        assert "cwes" not in d or d["cwes"] == []
        assert "cpes" not in d or d["cpes"] == []
        assert "exploits" not in d or d["exploits"] == []
        assert "kev_entry" not in d or d["kev_entry"] is None

        # Should roundtrip successfully
        restored = CVDIO.from_dict(d)
        assert restored.cve_id == "CVE-2024-0000"
        assert len(restored.cvss_scores) == 0
        assert len(restored.epss_scores) == 0
        assert len(restored.cwes) == 0
        assert len(restored.cpes) == 0
        assert len(restored.exploits) == 0
        assert restored.kev_entry is None
