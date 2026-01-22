"""
Tests for io.py - consolidated I/O operations.

Covers serialization (dict, JSON, pickle), file imports (EPSS, KEV, NVD),
and enrichment operations for both single vulnerabilities and batch arrays.
"""

import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from vulnstate import CVDArray, CVDEvent, CVDVulnerability
from vulnstate.io import CVDIO


class TestCVDIO:
    """Tests for CVDIO class."""

    def test_to_dict_single(self):
        """Convert single vulnerability to dict."""

        vuln = CVDVulnerability("CVE-2024-1234", cvss_score=9.8)
        data = CVDIO.to_dict(vuln)

        assert data["cve_id"] == "CVE-2024-1234"
        assert data["cvss_score"] == 9.8
        assert "state" in data

    def test_from_dict_single(self):
        """Reconstruct vulnerability from dict."""

        data = {
            "cve_id": "CVE-2024-5678",
            "vuln_id": "test-id-123",
            "state": "Vfdpxa",
            "cvss_score": 7.5,
        }
        vuln = CVDIO.from_dict(data)

        assert vuln.cve_id == "CVE-2024-5678"
        assert vuln.state_str == "Vfdpxa"

    def test_to_json_roundtrip(self):
        """JSON roundtrip preserves data."""

        vuln = CVDVulnerability("CVE-2024-0001", cvss_score=8.5)
        json_str = CVDIO.to_json(vuln)
        restored = CVDIO.from_json(json_str)

        assert restored.cve_id == vuln.cve_id
        assert restored.cvss_score == vuln.cvss_score

    def test_save_load_json_file(self):
        """Save and load JSON file."""

        vuln = CVDVulnerability("CVE-2024-FILE", cvss_score=6.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "vuln.json"
            CVDIO.save_json(vuln, str(filepath))
            loaded = CVDIO.load_json(str(filepath))

        assert loaded.cve_id == "CVE-2024-FILE"


class TestArrayIO:
    """Tests for array I/O operations."""

    def test_array_to_dict_list(self):
        """Convert array to list of dicts."""

        arr = CVDArray(
            [
                CVDVulnerability("CVE-2024-001"),
                CVDVulnerability("CVE-2024-002"),
            ]
        )
        data = CVDIO.array_to_dicts(arr)

        assert len(data) == 2
        assert data[0]["cve_id"] == "CVE-2024-001"
        assert data[1]["cve_id"] == "CVE-2024-002"

    def test_array_from_dict_list(self):
        """Create array from list of dicts."""

        data = [
            {"cve_id": "CVE-2024-A", "state": "vfdpxa"},
            {"cve_id": "CVE-2024-B", "state": "Vfdpxa"},
        ]
        arr = CVDIO.array_from_dicts(data)

        assert len(arr) == 2
        assert arr[0].cve_id == "CVE-2024-A"

    def test_to_dataframe(self):
        """Convert array to pandas DataFrame."""
        import pandas as pd

        arr = CVDArray(
            [
                CVDVulnerability("CVE-2024-001", cvss_score=9.8),
                CVDVulnerability("CVE-2024-002", cvss_score=7.5),
            ]
        )
        arr[0].apply_event(CVDEvent.V)
        arr[1].apply_event(CVDEvent.V)

        df = arr.to_dataframe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "cve_id" in df.columns
        assert "cvss_score" in df.columns
        assert df.iloc[0]["cve_id"] == "CVE-2024-001"
        assert df.iloc[0]["cvss_score"] == 9.8

    def test_to_dataframe_with_computed(self):
        """Convert array to DataFrame with computed properties."""
        arr = CVDArray([CVDVulnerability("CVE-2024-001")])
        arr[0].apply_event(CVDEvent.V)

        df_basic = arr.to_dataframe(include_analytics=False)
        df_computed = arr.to_dataframe(include_analytics=True)

        # For now, both should have same columns since analytics is TODO
        # This test will be updated when analytics are added
        assert len(df_computed.columns) >= len(df_basic.columns)


class TestEnrichment:
    """Tests for enrichment data import."""

    def test_import_epss(self):
        """Import EPSS scores into array."""

        arr = CVDArray(
            [
                CVDVulnerability("CVE-2024-001"),
                CVDVulnerability("CVE-2024-002"),
            ]
        )
        epss_data = {"CVE-2024-001": 0.85, "CVE-2024-002": 0.15}

        CVDIO.import_epss(arr, epss_data)

        assert arr[0].epss == 0.85
        assert arr[1].epss == 0.15

    def test_import_kev(self):
        """Import KEV flags into array."""

        arr = CVDArray(
            [
                CVDVulnerability("CVE-2024-001"),
                CVDVulnerability("CVE-2024-002"),
            ]
        )
        kev_data = {"CVE-2024-001": {"dateAdded": "2021-11-03"}}

        CVDIO.import_kev(arr, kev_data, apply_event=False)

        assert arr[0].kev == True  # noqa: E712
        assert arr[1].kev == False  # noqa: E712

    def test_import_epss_syncs_to_array_level(self):
        """import_epss updates both vuln objects AND array-level arr.epss property.

        This test catches the bug where import_epss would update vuln.epss but
        not sync to arr.enrichment.epss, causing arr.epss to return all NaN.
        """
        arr = CVDArray(
            [
                CVDVulnerability("CVE-2024-001"),
                CVDVulnerability("CVE-2024-002"),
            ]
        )
        epss_data = {"CVE-2024-001": 0.85, "CVE-2024-002": 0.15}

        CVDIO.import_epss(arr, epss_data)

        # Check both object-level AND array-level access
        assert arr[0].epss == 0.85  # Object access
        assert arr[1].epss == 0.15
        assert arr.epss[0] == 0.85  # Array-level property access
        assert arr.epss[1] == 0.15

    def test_import_kev_syncs_to_array_level(self):
        """import_kev updates both vuln objects AND array-level arr.kev property.

        This test catches the bug where import_kev would update vuln.kev but
        not sync to arr.enrichment.kev, causing arr.kev to return all False.
        """
        arr = CVDArray(
            [
                CVDVulnerability("CVE-2024-001"),
                CVDVulnerability("CVE-2024-002"),
            ]
        )
        kev_data = {"CVE-2024-001": {"dateAdded": "2021-11-03"}}

        CVDIO.import_kev(arr, kev_data, apply_event=False)

        # Check both object-level AND array-level access
        assert arr[0].kev == True  # noqa: E712 - Object access
        assert arr[1].kev == False  # noqa: E712
        assert arr.kev[0] == True  # noqa: E712 - Array-level property access
        assert arr.kev[1] == False  # noqa: E712


class TestNVDImport:
    """Tests for NVD format import."""

    def test_from_nvd_basic(self):
        """Import basic NVD items."""

        nvd_items = [
            {
                "cve": {"CVE_data_meta": {"ID": "CVE-2024-001"}},
                "impact": {
                    "baseMetricV3": {"cvssV3": {"baseScore": 9.8, "vectorString": "CVSS:3.1/AV:N"}}
                },
            },
            {
                "cve": {"CVE_data_meta": {"ID": "CVE-2024-002"}},
                "impact": {"baseMetricV2": {"cvssV2": {"baseScore": 5.0}}},
            },
        ]
        arr = CVDIO.from_nvd(nvd_items)

        assert len(arr) == 2
        assert arr[0].cve_id == "CVE-2024-001"
        assert arr[0].cvss_score == 9.8
        assert arr[1].cvss_score == 5.0

    def test_from_nvd_with_published_date(self):
        """NVD published date creates P event."""

        nvd_items = [
            {
                "cve": {"CVE_data_meta": {"ID": "CVE-2024-001"}},
                "impact": {},
                "publishedDate": "2024-01-15T12:00:00Z",
            },
        ]
        arr = CVDIO.from_nvd(nvd_items)

        assert len(arr) == 1
        # Should have P event from published date
        assert "P" in arr[0].state_str  # uppercase P means event occurred


# ============================================================================
# SERIALIZATION TESTS (from test_serialization.py)
# ============================================================================


class TestDictSerialization:
    """Test core dictionary serialization."""

    def test_to_dict_basic(self):
        """Test basic to_dict conversion."""
        vuln = CVDVulnerability("TEST-001")
        data = vuln.to_dict()

        assert data["cve_id"] == "TEST-001"
        assert data["state"] == "vfdpxa"
        assert "metadata" in data
        assert "event_timestamps" in data
        assert "history" in data

    def test_to_dict_with_metadata(self):
        """Test to_dict preserves metadata."""
        vuln = CVDVulnerability("TEST-002", vendor="Apache", cvss_score=8.5)
        data = vuln.to_dict()

        assert data["metadata"]["vendor"] == "Apache"
        assert data["cvss_score"] == 8.5

    def test_to_dict_with_events(self):
        """Test to_dict with applied events."""
        vuln = CVDVulnerability("TEST-003")
        vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1, 12, 0))
        vuln.apply_event(CVDEvent.F, timestamp=datetime(2024, 1, 2, 14, 30))

        data = vuln.to_dict()

        assert data["state"] == "VFdpxa"  # V and F applied
        assert len(data["history"]) > 0
        assert len(data["event_timestamps"]) == 2

    def test_to_dict_with_computed(self):
        """Test to_dict include_computed option."""
        vuln = CVDVulnerability("TEST-004")
        vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))
        vuln.apply_event(CVDEvent.F, timestamp=datetime(2024, 1, 2))
        vuln.apply_event(CVDEvent.D, timestamp=datetime(2024, 1, 3))
        vuln.apply_event(CVDEvent.P, timestamp=datetime(2024, 1, 4))

        # Without computed
        data_basic = vuln.to_dict(include_computed=False)
        assert "state_label" not in data_basic
        assert "history_string" not in data_basic

        # With computed
        data_computed = vuln.to_dict(include_computed=True)
        assert data_computed["state_label"] == "VFDP"
        assert data_computed["history_string"] == "VFDP"

    def test_from_dict_basic(self):
        """Test basic from_dict reconstruction."""
        vuln = CVDVulnerability("TEST-005")
        data = vuln.to_dict()

        vuln2 = CVDVulnerability.from_dict(data)

        assert vuln2.cve_id == vuln.cve_id
        assert vuln2.state_str == vuln.state_str
        assert vuln2.state.state_encoded == vuln.state.state_encoded

    def test_from_dict_with_metadata(self):
        """Test from_dict preserves metadata."""
        vuln = CVDVulnerability("TEST-006", severity="high", cvss_score=7.5)
        data = vuln.to_dict()

        vuln2 = CVDVulnerability.from_dict(data)

        assert vuln2.metadata["severity"] == "high"
        assert vuln2.cvss_score == 7.5

    def test_from_dict_with_timestamps(self):
        """Test from_dict reconstructs datetime objects."""
        vuln = CVDVulnerability("TEST-007")
        ts = datetime(2024, 1, 15, 10, 30, 45)
        vuln.apply_event(CVDEvent.V, timestamp=ts)

        data = vuln.to_dict()
        vuln2 = CVDVulnerability.from_dict(data)

        # Verify datetime reconstruction
        assert CVDEvent.V in vuln2.events
        assert vuln2.events[CVDEvent.V] == ts


class TestJSONSerialization:
    """Test JSON serialization."""

    def test_to_json(self):
        """Test to_json produces valid JSON."""
        vuln = CVDVulnerability("TEST-008", vendor="Microsoft")
        json_str = vuln.to_json()

        # Should be valid JSON
        data = json.loads(json_str)
        assert data["cve_id"] == "TEST-008"

    def test_from_json(self):
        """Test from_json reconstruction."""
        vuln = CVDVulnerability("TEST-009")
        vuln.apply_event(CVDEvent.V)
        vuln.apply_event(CVDEvent.F)

        json_str = vuln.to_json()
        vuln2 = CVDVulnerability.from_json(json_str)

        assert vuln2.cve_id == vuln.cve_id
        assert vuln2.state_str == vuln.state_str
        assert vuln2.state.state_encoded == vuln.state.state_encoded
        assert vuln2.history_string == vuln.history_string

    def test_save_load_json_file(self, tmp_path):
        """Test saving and loading JSON files."""
        vuln = CVDVulnerability("TEST-010", severity="critical")
        vuln.apply_event(CVDEvent.V)

        # Save
        filepath = tmp_path / "vuln.json"
        vuln.save_json(str(filepath))
        assert filepath.exists()

        # Load
        vuln2 = CVDVulnerability.load_json(str(filepath))

        assert vuln2.cve_id == vuln.cve_id
        assert vuln2.metadata["severity"] == "critical"

    def test_json_with_computed_properties(self):
        """Test JSON includes computed properties when requested."""
        vuln = CVDVulnerability("TEST-011")
        vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1))
        vuln.apply_event(CVDEvent.F, timestamp=datetime(2024, 1, 2))

        json_str = vuln.to_json(include_computed=True)
        data = json.loads(json_str)

        assert "state_label" in data
        assert "history_string" in data

    def test_json_file_not_found(self, tmp_path):
        """Test error handling for missing files."""
        filepath = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            CVDVulnerability.load_json(str(filepath))

    def test_json_corrupted_file(self, tmp_path):
        """Test error handling for corrupted JSON."""
        filepath = tmp_path / "corrupted.json"
        with open(filepath, "w") as f:
            f.write("{invalid json content")

        with pytest.raises(json.JSONDecodeError):
            CVDVulnerability.load_json(str(filepath))


class TestPickleSerialization:
    """Test Pickle serialization."""

    def test_save_load_pickle_file(self, tmp_path):
        """Test saving and loading pickle files."""
        vuln = CVDVulnerability("TEST-012", cvss_score=9.8)
        vuln.apply_event(CVDEvent.V, timestamp=datetime(2024, 1, 1, 12, 0))
        vuln.apply_event(CVDEvent.F)

        # Save
        filepath = tmp_path / "vuln.pkl"
        vuln.save_pickle(str(filepath))
        assert filepath.exists()

        # Load
        vuln2 = CVDVulnerability.load_pickle(str(filepath))

        # Verify complete reconstruction
        assert vuln2.cve_id == vuln.cve_id
        assert vuln2.state_str == vuln.state_str
        assert vuln2.state.state_encoded == vuln.state.state_encoded
        assert vuln2.cvss_score == 9.8

    def test_pickle_preserves_datetime_exactly(self, tmp_path):
        """Test that pickle preserves datetime objects exactly."""
        vuln = CVDVulnerability("TEST-013")
        ts = datetime(2024, 1, 15, 10, 30, 45, 123456)
        vuln.apply_event(CVDEvent.V, timestamp=ts)

        filepath = tmp_path / "vuln.pkl"
        vuln.save_pickle(str(filepath))
        vuln2 = CVDVulnerability.load_pickle(str(filepath))

        # Exact preservation including microseconds
        assert vuln2.events[CVDEvent.V] == ts

    def test_pickle_vs_json_sizes(self, tmp_path):
        """Compare file sizes between pickle and JSON."""
        vuln = CVDVulnerability("TEST-014")
        for event in [CVDEvent.V, CVDEvent.F, CVDEvent.D]:
            vuln.apply_event(event)

        # JSON file
        json_file = tmp_path / "vuln.json"
        vuln.save_json(str(json_file))
        json_size = json_file.stat().st_size

        # Pickle file
        pkl_file = tmp_path / "vuln.pkl"
        vuln.save_pickle(str(pkl_file))
        pickle_size = pkl_file.stat().st_size

        # Pickle should not be excessively larger than JSON
        # With dataclasses, pickle may have more metadata overhead
        assert pickle_size <= json_size * 2.0  # Allow up to 2x size

    def test_pickle_file_not_found(self, tmp_path):
        """Test error handling for missing pickle files."""
        filepath = tmp_path / "nonexistent.pkl"

        with pytest.raises(FileNotFoundError):
            CVDVulnerability.load_pickle(str(filepath))


class TestBatchSerialization:
    """Test batch serialization with CVDArray."""

    def test_batch_to_dict_list(self):
        """Test converting batch to list of dicts."""
        vulns = [CVDVulnerability(f"V-{i:03d}") for i in range(10)]
        for v in vulns:
            v.apply_event(CVDEvent.V)

        arr = CVDArray(vulns)
        data_list = arr.to_dict_batch()

        assert len(data_list) == 10
        assert all("cve_id" in d for d in data_list)
        assert all(d["state"] == "Vfdpxa" for d in data_list)

    def test_batch_json_save_load(self, tmp_path):
        """Test batch JSON save and load."""
        vulns = [
            CVDVulnerability(f"BATCH-{i:04d}", severity=["low", "medium", "high"][i % 3])
            for i in range(50)
        ]
        arr = CVDArray(vulns)

        # Save
        filepath = tmp_path / "batch.json"
        arr.to_json_batch(str(filepath))
        assert filepath.exists()

        # Verify file is valid JSON
        with open(filepath) as f:
            data = json.load(f)
        assert len(data) == 50

        # Load
        arr2 = CVDArray.from_json_batch(str(filepath))

        assert len(arr2) == 50
        assert arr2[0].cve_id == "BATCH-0000"

    def test_batch_pickle_save_load(self, tmp_path):
        """Test batch pickle save and load."""
        vulns = [CVDVulnerability(f"PICKLE-{i:04d}") for i in range(50)]
        arr = CVDArray(vulns)

        # Save
        filepath = tmp_path / "batch.pkl"
        arr.save_pickle_batch(str(filepath))
        assert filepath.exists()

        # Load
        arr2 = CVDArray.load_pickle_batch(str(filepath))

        assert len(arr2) == 50
        assert arr2[0].cve_id == "PICKLE-0000"

    def test_batch_performance_large_scale(self, tmp_path):
        """Test batch serialization with 1000 vulnerabilities."""
        vulns = [
            CVDVulnerability(f"PERF-{i:05d}", severity="high" if i % 2 else "low")
            for i in range(1000)
        ]
        for v in vulns:
            v.apply_event(CVDEvent.V)
            if v.cve_id.endswith(("0", "2", "4")):
                v.apply_event(CVDEvent.F)

        arr = CVDArray(vulns)

        # Benchmark JSON
        json_file = tmp_path / "perf.json"
        start = time.time()
        arr.to_json_batch(str(json_file))
        json_save_time = time.time() - start

        start = time.time()
        arr_json = CVDArray.from_json_batch(str(json_file))
        json_load_time = time.time() - start

        # Benchmark Pickle
        pkl_file = tmp_path / "perf.pkl"
        start = time.time()
        arr.save_pickle_batch(str(pkl_file))
        pkl_save_time = time.time() - start

        start = time.time()
        arr_pkl = CVDArray.load_pickle_batch(str(pkl_file))
        pkl_load_time = time.time() - start

        # Print results
        print("\n1000 vulnerabilities:")
        print(f"  JSON save: {json_save_time:.3f}s ({1000 / json_save_time:.0f} vulns/sec)")
        print(f"  JSON load: {json_load_time:.3f}s ({1000 / json_load_time:.0f} vulns/sec)")
        print(f"  Pickle save: {pkl_save_time:.3f}s ({1000 / pkl_save_time:.0f} vulns/sec)")
        print(f"  Pickle load: {pkl_load_time:.3f}s ({1000 / pkl_load_time:.0f} vulns/sec)")
        print(f"  Pickle speedup: {json_save_time / pkl_save_time:.1f}x faster (save)")

        # Verify data integrity
        assert len(arr_json) == 1000
        assert len(arr_pkl) == 1000


class TestRoundtripIntegrity:
    """Test that roundtrip serialization preserves all data."""

    def test_json_roundtrip_full_lifecycle(self):
        """Test JSON roundtrip with complex vulnerability state."""
        vuln = CVDVulnerability("ROUNDTRIP-001", vendor="Linux", cvss_score=9.1)
        ts_v = datetime(2024, 1, 1, 8, 0)
        ts_f = datetime(2024, 1, 2, 10, 30)
        ts_d = datetime(2024, 1, 5, 14, 0)

        vuln.apply_event(CVDEvent.V, timestamp=ts_v)
        vuln.apply_event(CVDEvent.F, timestamp=ts_f)
        vuln.apply_event(CVDEvent.D, timestamp=ts_d)

        # Serialize and deserialize
        json_str = vuln.to_json(include_computed=True)
        vuln2 = CVDVulnerability.from_json(json_str)

        # Verify all data
        assert vuln2.cve_id == "ROUNDTRIP-001"
        assert vuln2.state_str == "VFDpxa"  # V, F, and D applied
        assert vuln2.state_label == "VFD"  # Fix deployed
        assert vuln2.history_string == "VFD"
        assert vuln2.metadata["vendor"] == "Linux"
        assert vuln2.cvss_score == 9.1
        assert vuln2.events[CVDEvent.V] == ts_v
        assert vuln2.events[CVDEvent.F] == ts_f
        assert vuln2.events[CVDEvent.D] == ts_d

    def test_pickle_roundtrip_preserves_types(self, tmp_path):
        """Test pickle roundtrip preserves Python types exactly."""
        vuln = CVDVulnerability(
            "PICKLE-RT-001",
            cvss_score=7.5,
            severity="high",
            affected=[1, 2, 3],
            tags={"critical", "rce"},
        )

        filepath = tmp_path / "rt.pkl"
        vuln.save_pickle(str(filepath))
        vuln2 = CVDVulnerability.load_pickle(str(filepath))

        # Verify types preserved
        assert isinstance(vuln2.cvss_score, float)
        assert isinstance(vuln2.metadata["severity"], str)
        assert isinstance(vuln2.metadata["affected"], list)
        assert isinstance(vuln2.metadata["tags"], set)


class TestSerializationEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_vulnerability_serialization(self):
        """Test serialization of new vulnerability with no events."""
        vuln = CVDVulnerability("EMPTY-001")

        data = vuln.to_dict()
        json_str = vuln.to_json()
        vuln2 = CVDVulnerability.from_json(json_str)

        assert data["state"] == "vfdpxa"
        assert vuln2.state_str == "vfdpxa"
        assert vuln2.history_string == ""

    def test_special_characters_in_metadata(self):
        """Test serialization with special characters."""
        vuln = CVDVulnerability(
            "SPECIAL-001",
            description="Contains 'quotes' and \"double quotes\" and\nnewlines",
            tags=["foo:bar", "test/path", "item@domain"],
        )

        json_str = vuln.to_json()
        vuln2 = CVDVulnerability.from_json(json_str)

        assert vuln2.metadata["description"] == vuln.metadata["description"]
        assert vuln2.metadata["tags"] == vuln.metadata["tags"]

    def test_unicode_in_metadata(self):
        """Test serialization with unicode characters."""
        vuln = CVDVulnerability(
            "UNICODE-001", vendor="中文测试", component="日本語", notes="Emoji test: 🔒🔓🚨"
        )

        json_str = vuln.to_json()
        vuln2 = CVDVulnerability.from_json(json_str)

        assert vuln2.metadata["vendor"] == "中文测试"
        assert vuln2.metadata["component"] == "日本語"
        assert vuln2.metadata["notes"] == "Emoji test: 🔒🔓🚨"

    def test_none_values_in_metadata(self):
        """Test serialization with None values."""
        vuln = CVDVulnerability("NONE-001", field1=None, field2="value", field3=None)

        data = vuln.to_dict()
        vuln2 = CVDVulnerability.from_dict(data)

        assert vuln2.metadata["field1"] is None
        assert vuln2.metadata["field2"] == "value"
        assert vuln2.metadata["field3"] is None


# ============================================================================
# FILE IMPORT TESTS (from test_file_import.py)
# ============================================================================

# Test data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class TestEPSSFileImport:
    """Tests for EPSS CSV file import."""

    def test_import_epss_file_basic(self):
        """Test basic EPSS file import."""
        # Create array with CVEs matching the test data
        vulns = [
            CVDVulnerability("CVE-2024-001"),
            CVDVulnerability("CVE-2024-002"),
            CVDVulnerability("CVE-2024-003"),
        ]
        arr = CVDArray(vulns)

        # Import EPSS scores from file
        epss_file = os.path.join(DATA_DIR, "epss_sample.csv")
        CVDIO.import_epss(arr, epss_file)

        # Verify scores were imported
        assert arr[0].epss == pytest.approx(0.85432)
        assert arr[1].epss == pytest.approx(0.42156)
        assert arr[2].epss == pytest.approx(0.12345)

    def test_import_epss_file_missing_cves(self):
        """Test EPSS import with CVEs not in file."""
        vulns = [
            CVDVulnerability("CVE-2024-001"),
            CVDVulnerability("CVE-2024-999"),  # Not in file
        ]
        arr = CVDArray(vulns)

        epss_file = os.path.join(DATA_DIR, "epss_sample.csv")
        CVDIO.import_epss(arr, epss_file)

        # First CVE should have score
        assert arr[0].epss == pytest.approx(0.85432)
        # Second CVE should not be updated (epss remains None)
        assert arr[1].epss is None

    def test_import_epss_file_convenience_method(self):
        """Test CVDArray convenience wrapper for EPSS import."""
        vulns = [
            CVDVulnerability("CVE-2024-001"),
            CVDVulnerability("CVE-2024-002"),
        ]
        arr = CVDArray(vulns)

        # Use convenience method
        epss_file = os.path.join(DATA_DIR, "epss_sample.csv")
        arr.import_epss(epss_file)

        assert arr[0].epss == pytest.approx(0.85432)
        assert arr[1].epss == pytest.approx(0.42156)


class TestKEVFileImport:
    """Tests for KEV CSV file import."""

    def test_import_kev_file_basic(self):
        """Test basic KEV file import."""
        vulns = [
            CVDVulnerability("CVE-2021-44228"),  # In KEV
            CVDVulnerability("CVE-2024-001"),  # In KEV
            CVDVulnerability("CVE-2024-002"),  # Not in KEV
        ]
        arr = CVDArray(vulns)

        # Import KEV catalog from file
        kev_file = os.path.join(DATA_DIR, "kev_sample.csv")
        CVDIO.import_kev(arr, kev_file)

        # Verify KEV flags were set
        assert arr[0].kev is True  # CVE-2021-44228
        assert arr[1].kev is True  # CVE-2024-001
        assert arr[2].kev is False  # CVE-2024-002 not in KEV

    def test_import_kev_file_convenience_method(self):
        """Test CVDArray convenience wrapper for KEV import."""
        vulns = [
            CVDVulnerability("CVE-2021-44228"),
            CVDVulnerability("CVE-2024-001"),
        ]
        arr = CVDArray(vulns)

        # Use convenience method
        kev_file = os.path.join(DATA_DIR, "kev_sample.csv")
        arr.import_kev(kev_file)

        assert arr[0].kev is True
        assert arr[1].kev is True


class TestNVDFileImport:
    """Tests for NVD JSON file import."""

    def test_import_nvd_file_basic(self):
        """Test basic NVD file import."""
        vulns = [
            CVDVulnerability("CVE-2024-001"),  # CVSS v3: 6.1
            CVDVulnerability("CVE-2024-002"),  # CVSS v3: 9.8
            CVDVulnerability("CVE-2024-005"),  # CVSS v2: 7.5
        ]
        arr = CVDArray(vulns)

        # Import NVD data from file
        nvd_file = os.path.join(DATA_DIR, "nvd_sample.json")
        arr.import_nvd(nvd_file)

        # Verify CVSS scores were imported
        assert arr[0].cvss_score == 6.1  # CVE-2024-001
        assert arr[1].cvss_score == 9.8  # CVE-2024-002
        assert arr[2].cvss_score == 7.5  # CVE-2024-005 (v2)

    def test_import_nvd_file_cvss_vectors(self):
        """Test NVD import includes CVSS vector strings."""
        vulns = [
            CVDVulnerability("CVE-2024-001"),
            CVDVulnerability("CVE-2024-002"),
        ]
        arr = CVDArray(vulns)

        nvd_file = os.path.join(DATA_DIR, "nvd_sample.json")
        arr.import_nvd(nvd_file)

        # Verify vector strings were imported
        assert "CVSS:3.1" in arr[0].cve_vector
        assert "CVSS:3.1" in arr[1].cve_vector

    def test_import_nvd_file_fallback_to_v2(self):
        """Test NVD import falls back to CVSS v2 when v3 unavailable."""
        vulns = [CVDVulnerability("CVE-2024-005")]  # Only has v2
        arr = CVDArray(vulns)

        nvd_file = os.path.join(DATA_DIR, "nvd_sample.json")
        arr.import_nvd(nvd_file)

        # Verify v2 score was used
        assert arr[0].cvss_score == 7.5
        assert arr[0].cve_vector == "AV:N/AC:L/Au:N/C:P/I:P/A:P"

    def test_import_nvd_file_convenience_method(self):
        """Test CVDArray convenience wrapper for NVD import."""
        vulns = [
            CVDVulnerability("CVE-2024-001"),
            CVDVulnerability("CVE-2024-002"),
        ]
        arr = CVDArray(vulns)

        # Use convenience method
        nvd_file = os.path.join(DATA_DIR, "nvd_sample.json")
        arr.import_nvd(nvd_file)

        assert arr[0].cvss_score == 6.1
        assert arr[1].cvss_score == 9.8


class TestIntegratedWorkflow:
    """Test integrated workflow with multiple file imports."""

    def test_import_all_sources(self):
        """Test importing from all three sources (EPSS, KEV, NVD)."""
        # Create array with overlapping CVEs
        vulns = [
            CVDVulnerability("CVE-2024-001"),  # In all three sources
            CVDVulnerability("CVE-2024-002"),  # In EPSS and NVD
            CVDVulnerability("CVE-2024-003"),  # In EPSS and KEV
        ]
        arr = CVDArray(vulns)

        # Import from all sources
        epss_file = os.path.join(DATA_DIR, "epss_sample.csv")
        kev_file = os.path.join(DATA_DIR, "kev_sample.csv")
        nvd_file = os.path.join(DATA_DIR, "nvd_sample.json")

        arr.import_epss(epss_file)
        arr.import_kev(kev_file)
        arr.import_nvd(nvd_file)

        # Verify CVE-2024-001 has all three enrichments
        assert arr[0].epss == pytest.approx(0.85432)
        assert arr[0].kev is True
        assert arr[0].cvss_score == 6.1

        # Verify CVE-2024-002 has EPSS and NVD but not KEV
        assert arr[1].epss == pytest.approx(0.42156)
        assert arr[1].kev is False
        assert arr[1].cvss_score == 9.8

        # Verify CVE-2024-003 has EPSS and KEV but not NVD
        assert arr[2].epss == pytest.approx(0.12345)
        assert arr[2].kev is True
        assert arr[2].cvss_score is None  # Not in NVD sample


class TestFileImportErrorHandling:
    """Test error handling for invalid files."""

    def test_import_nonexistent_file(self):
        """Test import from non-existent file raises error."""
        vulns = [CVDVulnerability("CVE-2024-001")]
        arr = CVDArray(vulns)

        with pytest.raises(FileNotFoundError):
            arr.import_epss("nonexistent.csv")

    def test_import_malformed_csv(self, tmp_path):
        """Test import from malformed CSV raises error."""
        # Create malformed CSV (missing required columns)
        malformed = tmp_path / "malformed.csv"
        malformed.write_text("wrong,columns\n1,2\n")

        vulns = [CVDVulnerability("CVE-2024-001")]
        arr = CVDArray(vulns)

        with pytest.raises(KeyError):
            arr.import_epss(str(malformed))

    def test_import_malformed_json(self, tmp_path):
        """Test import from malformed JSON raises error."""
        # Create malformed JSON
        malformed = tmp_path / "malformed.json"
        malformed.write_text("not valid json")

        vulns = [CVDVulnerability("CVE-2024-001")]
        arr = CVDArray(vulns)

        with pytest.raises(json.JSONDecodeError):
            arr.import_nvd(str(malformed))


class TestMetadataPluck:
    """Tests for metadata extraction with pluck()."""

    def test_pluck_metadata(self):
        """Extract nested metadata values as numpy array."""
        # Create array with metadata
        arr = CVDArray.zeros(3)
        arr.get(0).metadata = {"kev": {"dateAdded": "2021-11-03", "vendor": "Adobe"}}
        arr.get(1).metadata = {"kev": {"dateAdded": "2022-01-15", "vendor": "Microsoft"}}
        arr.get(2).metadata = {}  # No KEV data

        # Extract KEV dates
        dates = arr.pluck("kev.dateAdded")
        assert dates[0] == "2021-11-03"
        assert dates[1] == "2022-01-15"
        assert dates[2] is None

        # Extract vendors
        vendors = arr.pluck("kev.vendor")
        assert vendors[0] == "Adobe"
        assert vendors[1] == "Microsoft"
        assert vendors[2] is None

    def test_pluck_single_level(self):
        """Extract top-level metadata values."""
        arr = CVDArray.zeros(2)
        arr.get(0).metadata = {"vendor": "Adobe", "severity": "high"}
        arr.get(1).metadata = {"vendor": "Microsoft"}

        vendors = arr.pluck("vendor")
        assert vendors[0] == "Adobe"
        assert vendors[1] == "Microsoft"

        # Missing field
        severity = arr.pluck("severity")
        assert severity[0] == "high"
        assert severity[1] is None

    def test_pluck_deeply_nested(self):
        """Extract deeply nested metadata values."""
        arr = CVDArray.zeros(2)
        arr.get(0).metadata = {"a": {"b": {"c": {"d": "value1"}}}}
        arr.get(1).metadata = {"a": {"b": {"c": {"d": "value2"}}}}

        values = arr.pluck("a.b.c.d")
        assert values[0] == "value1"
        assert values[1] == "value2"

    def test_pluck_partial_path(self):
        """Extract values when path exists partially."""
        arr = CVDArray.zeros(2)
        arr.get(0).metadata = {"kev": {"dateAdded": "2021-11-03"}}
        arr.get(1).metadata = {"kev": {}}  # kev exists but no dateAdded

        dates = arr.pluck("kev.dateAdded")
        assert dates[0] == "2021-11-03"
        assert dates[1] is None

    def test_pluck_empty_array(self):
        """Extract from empty array."""
        import numpy as np

        arr = CVDArray.zeros(0)
        values = arr.pluck("kev.dateAdded")

        assert isinstance(values, np.ndarray)
        assert len(values) == 0

    def test_pluck_numeric_values(self):
        """Extract numeric metadata values."""
        arr = CVDArray.zeros(3)
        arr.get(0).metadata = {"epss": {"score": 0.85}}
        arr.get(1).metadata = {"epss": {"score": 0.42}}
        arr.get(2).metadata = {}

        scores = arr.pluck("epss.score")
        assert scores[0] == 0.85
        assert scores[1] == 0.42
        # Numeric arrays convert None to np.nan for filtering compatibility
        assert np.isnan(scores[2])

    def test_pluck_numeric_filtering(self):
        """Boolean filtering works with numeric values and NaN."""
        arr = CVDArray.zeros(4)
        arr.get(0).metadata = {"epss": {"score": 0.95}}
        arr.get(1).metadata = {"epss": {"score": 0.42}}
        arr.get(2).metadata = {"epss": {"score": 0.88}}
        arr.get(3).metadata = {}  # Missing -> NaN

        # Filter using comparison operators (NaN comparisons return False)
        high_epss = arr[arr.pluck("epss.score") > 0.8]
        assert len(high_epss) == 2
        assert high_epss.vuln_ids[0] == arr.vuln_ids[0]  # 0.95
        assert high_epss.vuln_ids[1] == arr.vuln_ids[2]  # 0.88


class TestKEVEventApplication:
    """Tests for KEV import with event A application."""

    def test_import_kev_applies_event_a(self):
        """KEV import applies event A with dateAdded timestamp."""
        # Create array with CVEs
        vulns = [
            CVDVulnerability("CVE-2021-27104"),
            CVDVulnerability("CVE-2021-27102"),
        ]
        arr = CVDArray(vulns)

        # Import KEV with event application
        kev_data = {
            "CVE-2021-27104": {
                "dateAdded": "2021-11-03",
                "vendorProject": "Accellion",
                "product": "FTA",
            }
        }

        arr.import_kev(kev_data, apply_event=True)

        # Check event A was applied
        vuln = arr.get(0)
        assert vuln.has_event_occurred(CVDEvent.A)
        assert vuln.events[CVDEvent.A] == np.datetime64("2021-11-03")

        # Second vuln should not have event A
        assert not arr.get(1).has_event_occurred(CVDEvent.A)

    def test_import_kev_without_event_application(self):
        """KEV import with apply_event=False only sets kev flag."""
        arr = CVDArray([CVDVulnerability("CVE-2021-27104")])

        kev_data = {"CVE-2021-27104": {"dateAdded": "2021-11-03"}}

        arr.import_kev(kev_data, apply_event=False)

        vuln = arr.get(0)
        # Should not have event A
        assert not vuln.has_event_occurred(CVDEvent.A)
        # But should be marked as KEV
        assert vuln.kev

    def test_import_kev_stores_parsed_metadata(self):
        """KEV import stores parsed fields with kev_ prefix."""
        arr = CVDArray([CVDVulnerability("CVE-2021-27104")])

        kev_data = {
            "CVE-2021-27104": {
                "dateAdded": "2021-11-03",
                "vendorProject": "Accellion",
                "product": "FTA",
            }
        }

        arr.import_kev(kev_data, apply_event=True)

        vuln = arr.get(0)
        # KEV fields stored with kev_ prefix
        assert vuln.metadata["kev_date_added"] == "2021-11-03"
        assert vuln.metadata["kev_vendor_name"] == "Accellion"
        assert vuln.metadata["kev_product_name"] == "FTA"

    def test_import_kev_stores_all_available_fields(self):
        """KEV import extracts all available KEV fields."""
        arr = CVDArray([CVDVulnerability("CVE-2021-27104")])

        kev_data = {
            "CVE-2021-27104": {
                "dateAdded": "2021-11-03",
                "vendorProject": "Accellion",
                "product": "FTA",
                "shortDescription": "Test description",
                "knownRansomwareCampaignUse": "Known",
                "requiredAction": "Apply update",
                "dueDate": "2021-12-01",
                "notes": "Test notes",
            }
        }

        arr.import_kev(kev_data, apply_event=True)

        vuln = arr.get(0)
        assert vuln.metadata["kev_vendor_name"] == "Accellion"
        assert vuln.metadata["kev_product_name"] == "FTA"
        assert vuln.metadata["kev_description"] == "Test description"
        assert vuln.metadata["kev_ransomware_use"] == "Known"
        assert vuln.metadata["kev_required_action"] == "Apply update"
        assert vuln.metadata["kev_due_date"] == "2021-12-01"
        assert vuln.metadata["kev_notes"] == "Test notes"
        assert vuln.metadata["kev_date_added"] == "2021-11-03"

    def test_import_kev_missing_date(self):
        """KEV import handles missing dateAdded gracefully."""
        arr = CVDArray([CVDVulnerability("CVE-2021-27104")])

        # KEV data without dateAdded
        kev_data = {"CVE-2021-27104": {"vendorProject": "Accellion"}}

        arr.import_kev(kev_data, apply_event=True)

        vuln = arr.get(0)
        # Should not have event A since dateAdded is missing
        assert not vuln.has_event_occurred(CVDEvent.A)
        # But should still be marked as KEV
        assert vuln.kev

    def test_import_epss_stores_metadata(self):
        """EPSS import stores full metadata when import_metadata=True."""
        arr = CVDArray([CVDVulnerability("CVE-2023-0001")])

        epss_data = {"CVE-2023-0001": {"score": 0.85432, "percentile": 0.95123}}

        arr.import_epss(epss_data, import_metadata=True)

        vuln = arr.get(0)
        assert "epss" in vuln.metadata
        assert vuln.metadata["epss"]["score"] == 0.85432
        assert vuln.metadata["epss"]["percentile"] == 0.95123

    def test_import_epss_without_metadata(self):
        """EPSS import stores score in enrichment when import_metadata=False."""
        arr = CVDArray([CVDVulnerability("CVE-2023-0002")])

        epss_data = {"CVE-2023-0002": {"score": 0.75, "percentile": 0.85}}

        arr.import_epss(epss_data, import_metadata=False)

        vuln = arr.get(0)
        # Score should be in enrichment
        assert vuln.epss == 0.75
        # But not in metadata
        assert "epss" not in vuln.metadata

    def test_import_epss_filtering(self):
        """EPSS import with include/exclude filters."""
        arr = CVDArray([CVDVulnerability("CVE-2023-0003")])

        epss_data = {"CVE-2023-0003": {"score": 0.85432, "percentile": 0.95123}}

        # Include only score
        arr.import_epss(epss_data, import_metadata=True, include=["score"])

        vuln = arr.get(0)
        assert "score" in vuln.metadata["epss"]
        assert "percentile" not in vuln.metadata["epss"]


class TestGenericCSVImport:
    """Tests for generic CSV import method."""

    def test_import_csv_applies_event(self):
        """Generic CSV import applies specified event."""
        import csv
        import tempfile

        # Create test CSV
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["cve_id", "patch_date", "advisory_id"])
            writer.writeheader()
            writer.writerow(
                {
                    "cve_id": "CVE-2024-0001",
                    "patch_date": "2024-02-01",
                    "advisory_id": "VENDOR-2024-001",
                }
            )
            csv_path = f.name

        try:
            arr = CVDArray.zeros(1)
            arr.get(0).identity.cve_id = "CVE-2024-0001"
            # Update the metadata array to reflect the CVE ID
            arr._metadata_raw["_cve_id"][0] = "CVE-2024-0001"
            # Apply event V first to satisfy V→F→D constraint
            arr.get(0).apply_event(CVDEvent.V, np.datetime64("2024-01-01"))
            arr.sync()

            arr.import_csv(
                source=csv_path,
                cve_column="cve_id",
                event=CVDEvent.F,
                timestamp_column="patch_date",
                apply_event=True,
            )

            vuln = arr.get(0)
            assert vuln.has_event_occurred(CVDEvent.F)
            assert vuln.events[CVDEvent.F] == np.datetime64("2024-02-01")
        finally:
            os.unlink(csv_path)


def test_import_json_nested_fields():
    """JSON import handles nested field paths."""
    import json
    import tempfile

    import numpy as np

    from vulnstate import CVDArray, CVDEvent

    # Create test JSON
    data = [
        {
            "vulnerability": {"cve_id": "CVE-2024-0001"},
            "threat_intel": {"first_observed": "2024-03-01"},
        }
    ]

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(data, f)
        json_path = f.name

    try:
        arr = CVDArray.zeros(1)
        arr.get(0).identity.cve_id = "CVE-2024-0001"
        arr._metadata_raw["_cve_id"][0] = "CVE-2024-0001"
        arr.sync()

        arr.import_json(
            source=json_path,
            cve_field="vulnerability.cve_id",
            event=CVDEvent.A,
            timestamp_field="threat_intel.first_observed",
            apply_event=True,
        )

        vuln = arr.get(0)
        assert vuln.has_event_occurred(CVDEvent.A)
        assert vuln.events[CVDEvent.A] == np.datetime64("2024-03-01")
    finally:
        os.unlink(json_path)


def test_to_dataframe_basic():
    """to_dataframe exports base columns."""
    import numpy as np

    from vulnstate import CVDArray, CVDEvent

    arr = CVDArray.zeros(2)
    arr.get(0).identity.cve_id = "CVE-2024-0001"
    arr.get(0).apply_event(CVDEvent.V, np.datetime64("2024-01-01"))
    arr.get(1).identity.cve_id = "CVE-2024-0002"
    arr.sync()

    df = arr.to_dataframe(include_analytics=False, explode_cvss=False, explode_metadata=False)

    assert "cve_id" in df.columns
    assert "state" in df.columns
    assert "V_timestamp" in df.columns
    assert len(df) == 2
    assert df.loc[0, "cve_id"] == "CVE-2024-0001"


class TestDataFrameIntegration:
    """Integration tests for DataFrame export (Phase 11)."""

    def test_to_dataframe_includes_all_new_analytical_properties(self):
        """Verify DataFrame export includes all 10 new analytical properties from Phase 5."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        # Create vulnerability with various events to trigger analytical properties
        v1 = CVDVulnerability("CVE-2024-001")
        v1.apply_event(CVDEvent.X, timestamp=base)  # Exploit before vendor awareness
        v1.apply_event(CVDEvent.V, timestamp=base + timedelta(days=5))
        v1.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))

        v2 = CVDVulnerability("CVE-2024-002")
        v2.apply_event(CVDEvent.V, timestamp=base)
        v2.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
        v2.apply_event(CVDEvent.P, timestamp=base + timedelta(days=30))

        arr = CVDArray([v1, v2])
        df = arr.to_dataframe(include_analytics=True)

        # Verify all 10 new analytical properties are present
        new_properties = [
            "is_zero_day_exploit",
            "is_zero_day_attack",
            "is_coordinated",
            "is_responsible_disclosure",
            "has_fix_before_exploit",
            "has_fix_before_attack",
            "has_deployment_before_exploit",
            "has_deployment_before_attack",
            "is_private_attack",
            "is_mass_exploitation",
        ]

        for prop in new_properties:
            assert prop in df.columns, f"Missing analytical property: {prop}"

        # Verify values are correct for v1 (zero-day exploit case)
        assert df.loc[0, "is_zero_day_exploit"], "v1 should have is_zero_day_exploit=True"

        # Verify values are correct for v2 (coordinated disclosure case)
        assert df.loc[1, "is_coordinated"], "v2 should have is_coordinated=True"
        assert df.loc[
            1, "is_responsible_disclosure"
        ], "v2 should have is_responsible_disclosure=True"

    def test_to_dataframe_includes_original_analytical_properties(self):
        """Verify DataFrame includes original analytical properties from before Phase 5."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.V, timestamp=base)
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
        v.apply_event(CVDEvent.X, timestamp=base + timedelta(days=20))

        arr = CVDArray([v])
        df = arr.to_dataframe(include_analytics=True)

        # Original analytical properties (before Phase 5)
        original_properties = [
            "is_fix_available",
            "is_fix_deployed",
            "is_weaponized",
            "is_under_attack",
            "is_premature_disclosure",
            "disclosure_window_days",
            "fix_lag_days",
            "deployment_lag_days",
        ]

        for prop in original_properties:
            assert prop in df.columns, f"Missing original analytical property: {prop}"

    def test_to_dataframe_column_count_with_analytics(self):
        """Verify DataFrame has expected number of columns with analytics enabled."""
        arr = CVDArray.random(5, seed=42)
        df = arr.to_dataframe(include_analytics=True, explode_cvss=False, explode_metadata=False)

        # Expected columns:
        # - cve_id, vuln_id, state, state_label (4 identity/state)
        # - cvss_score, epss, kev (3 scoring/enrichment)
        # - V_timestamp, F_timestamp, D_timestamp, P_timestamp, X_timestamp, A_timestamp (6 timestamps)
        # - attack_vector, attack_complexity, privileges_required, user_interaction, scope,
        #   confidentiality_impact, integrity_impact, availability_impact (8 CVSS metrics)
        # - is_fix_available, is_fix_deployed, is_weaponized, is_under_attack,
        #   is_premature_disclosure, disclosure_window_days, fix_lag_days, deployment_lag_days (8 original analytics)
        # - is_zero_day_exploit, is_zero_day_attack, is_coordinated, is_responsible_disclosure,
        #   has_fix_before_exploit, has_fix_before_attack, has_deployment_before_exploit,
        #   has_deployment_before_attack, is_private_attack, is_mass_exploitation (10 new analytics)
        # Total: 4 + 3 + 6 + 8 + 8 + 10 = 39 columns (plus metadata if present)

        # At minimum, should have all the core analytical columns
        assert len(df.columns) >= 35, f"Expected at least 35 columns, got {len(df.columns)}"

        # Verify key columns are present
        assert "cve_id" in df.columns
        assert "state_label" in df.columns
        assert "is_zero_day_exploit" in df.columns
        assert "is_mass_exploitation" in df.columns

    def test_to_dataframe_analytics_false_excludes_analytical_properties(self):
        """Verify include_analytics=False excludes all analytical properties."""
        arr = CVDArray.random(5, seed=42)
        df = arr.to_dataframe(include_analytics=False, explode_cvss=False, explode_metadata=False)

        # Should NOT have analytical properties
        analytical_properties = [
            "is_fix_available",
            "is_zero_day_exploit",
            "is_coordinated",
            "has_fix_before_exploit",
            "disclosure_window_days",
        ]

        for prop in analytical_properties:
            assert prop not in df.columns, f"Should not have {prop} when include_analytics=False"

        # Should still have identity and timestamp columns
        assert "cve_id" in df.columns
        assert "state" in df.columns
        assert "V_timestamp" in df.columns

    def test_to_dataframe_with_empty_array(self):
        """Verify DataFrame export works with empty array."""
        arr = CVDArray([])
        df = arr.to_dataframe(include_analytics=True)

        assert len(df) == 0, "Empty array should produce empty DataFrame"
        # Empty array should still have column schema for downstream compatibility
        assert "cve_id" in df.columns
        assert "state" in df.columns

    def test_to_dataframe_preserves_values_across_roundtrip(self):
        """Verify serialization round-trip preserves new analytical property values."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        # Create vulnerability with specific analytical property values
        v = CVDVulnerability("CVE-2024-001")
        v.apply_event(CVDEvent.X, timestamp=base)
        v.apply_event(CVDEvent.V, timestamp=base + timedelta(days=5))
        v.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
        v.apply_event(CVDEvent.A, timestamp=base + timedelta(days=15))

        arr = CVDArray([v])
        arr.sync()

        # Export to DataFrame
        df = arr.to_dataframe(include_analytics=True)

        # Verify analytical properties in DataFrame match array properties
        assert df.loc[0, "is_zero_day_exploit"] == arr.is_zero_day_exploit[0]
        assert df.loc[0, "is_zero_day_attack"] == arr.is_zero_day_attack[0]
        assert df.loc[0, "has_fix_before_exploit"] == arr.has_fix_before_exploit[0]
        assert df.loc[0, "has_fix_before_attack"] == arr.has_fix_before_attack[0]
        assert df.loc[0, "is_mass_exploitation"] == arr.is_mass_exploitation[0]

    def test_to_dataframe_enrichment_not_overwritten_by_metadata(self):
        """Verify enrichment data (epss, kev) is not overwritten by metadata during export.

        This test catches the bug where explode_metadata=True would overwrite the
        correct arr.epss/arr.kev values with stale metadata arrays.
        """
        # Create array with metadata (simulates import_nvd which initializes metadata)
        v1 = CVDVulnerability("CVE-2024-001")
        v2 = CVDVulnerability("CVE-2024-002")
        arr = CVDArray([v1, v2])

        # Simulate the metadata structure that import_nvd creates (all NaN initially)
        import numpy as np

        arr._metadata_raw["epss"] = np.array([np.nan, np.nan], dtype=np.float32)
        arr._metadata_raw["kev"] = np.array([False, False], dtype=bool)

        # Now import enrichment data (this updates arr.enrichment, not metadata)
        CVDIO.import_epss(arr, {"CVE-2024-001": 0.85, "CVE-2024-002": 0.15})
        CVDIO.import_kev(arr, {"CVE-2024-001": {"dateAdded": "2021-11-03"}}, apply_event=False)

        # Export with explode_metadata=True - this should NOT overwrite enrichment
        df = arr.to_dataframe(include_analytics=False, explode_metadata=True)

        # Verify enrichment data is correctly exported (not overwritten by stale metadata)
        assert df.loc[0, "epss"] == 0.85, "EPSS should come from enrichment, not metadata"
        assert df.loc[1, "epss"] == 0.15
        assert df.loc[0, "kev"] == True, "KEV should come from enrichment"  # noqa: E712
        assert df.loc[1, "kev"] == False  # noqa: E712


class TestSerializationRoundTrip:
    """Test serialization round-trip preserves new properties (Phase 11)."""

    def test_dict_roundtrip_preserves_analytical_property_inputs(self):
        """Verify dict serialization preserves events that enable analytical properties."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        # Create vulnerability with events that trigger analytical properties
        v1 = CVDVulnerability("CVE-2024-001")
        v1.apply_event(CVDEvent.X, timestamp=base)
        v1.apply_event(CVDEvent.V, timestamp=base + timedelta(days=5))
        v1.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))

        # Serialize to dict
        data = v1.to_dict()

        # Reconstruct from dict
        v2 = CVDVulnerability.from_dict(data)

        # Verify analytical properties are correctly recomputed
        arr1 = CVDArray([v1])
        arr2 = CVDArray([v2])

        assert arr1.is_zero_day_exploit[0] == arr2.is_zero_day_exploit[0]
        assert arr1.has_fix_before_exploit[0] == arr2.has_fix_before_exploit[0]

    def test_json_roundtrip_preserves_analytical_property_inputs(self):
        """Verify JSON serialization preserves events that enable analytical properties."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v1 = CVDVulnerability("CVE-2024-001")
        v1.apply_event(CVDEvent.V, timestamp=base)
        v1.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
        v1.apply_event(CVDEvent.P, timestamp=base + timedelta(days=30))

        # Serialize to JSON and back
        json_str = v1.to_json()
        v2 = CVDVulnerability.from_json(json_str)

        # Verify analytical properties match
        arr1 = CVDArray([v1])
        arr2 = CVDArray([v2])

        assert arr1.is_coordinated[0] == arr2.is_coordinated[0]
        assert arr1.is_responsible_disclosure[0] == arr2.is_responsible_disclosure[0]

    def test_array_dict_roundtrip_preserves_all_properties(self):
        """Verify array serialization round-trip preserves all data."""
        from datetime import datetime, timedelta

        base = datetime(2024, 1, 1)

        v1 = CVDVulnerability("CVE-2024-001")
        v1.apply_event(CVDEvent.X, timestamp=base)
        v1.apply_event(CVDEvent.V, timestamp=base + timedelta(days=5))
        v1.apply_event(CVDEvent.A, timestamp=base + timedelta(days=10))

        v2 = CVDVulnerability("CVE-2024-002")
        v2.apply_event(CVDEvent.V, timestamp=base)
        v2.apply_event(CVDEvent.F, timestamp=base + timedelta(days=10))
        v2.apply_event(CVDEvent.X, timestamp=base + timedelta(days=20))

        arr1 = CVDArray([v1, v2])

        # Serialize and deserialize
        dicts = CVDIO.array_to_dicts(arr1)
        arr2 = CVDIO.array_from_dicts(dicts)

        # Verify analytical properties match
        assert np.array_equal(arr1.is_zero_day_exploit, arr2.is_zero_day_exploit)
        assert np.array_equal(arr1.is_zero_day_attack, arr2.is_zero_day_attack)
        assert np.array_equal(arr1.has_fix_before_exploit, arr2.has_fix_before_exploit)
        assert np.array_equal(arr1.is_mass_exploitation, arr2.is_mass_exploitation)


class TestCPEParser:
    """Tests for CPE parsing functionality."""

    def test_parse_cpe_extracts_vendor_and_product(self):
        """CPE parser extracts vendor_id and product_id."""
        from vulnstate.parsers import parse_cpe

        result = parse_cpe("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*")
        assert result["vendor_id"] == "apache"
        assert result["product_id"] == "log4j"

    def test_parse_cpe_handles_os_type(self):
        """CPE parser handles OS-type CPE strings."""
        from vulnstate.parsers import parse_cpe

        result = parse_cpe("cpe:2.3:o:microsoft:windows_10:*:*:*:*:*:*:*:*")
        assert result["vendor_id"] == "microsoft"
        assert result["product_id"] == "windows_10"

    def test_parse_cpe_handles_none(self):
        """CPE parser returns None values for None input."""
        from vulnstate.parsers import parse_cpe

        result = parse_cpe(None)
        assert result["vendor_id"] is None
        assert result["product_id"] is None

    def test_parse_cpe_handles_invalid_format(self):
        """CPE parser returns None values for invalid format."""
        from vulnstate.parsers import parse_cpe

        result = parse_cpe("not-a-cpe-string")
        assert result["vendor_id"] is None
        assert result["product_id"] is None

    def test_parse_cpe_handles_wildcard_vendor(self):
        """CPE parser treats wildcards as None."""
        from vulnstate.parsers import parse_cpe

        result = parse_cpe("cpe:2.3:a:*:someproduct:1.0:*:*:*:*:*:*:*")
        assert result["vendor_id"] is None
        assert result["product_id"] == "someproduct"


class TestNVDMetadataExtraction:
    """Tests for enhanced NVD metadata extraction."""

    def test_nvd_parser_extracts_description(self):
        """NVD parser extracts English description."""
        from vulnstate.parsers import NVDParser

        item = {
            "cve": {"descriptions": [{"lang": "en", "value": "Test vulnerability description"}]}
        }

        desc = NVDParser.extract_description(item, "2.0")
        assert desc == "Test vulnerability description"

    def test_nvd_parser_extracts_cwe_ids(self):
        """NVD parser extracts CWE IDs."""
        from vulnstate.parsers import NVDParser

        item = {
            "cve": {"weaknesses": [{"description": [{"value": "CWE-79"}, {"value": "CWE-89"}]}]}
        }

        cwe_ids = NVDParser.extract_cwe_ids(item, "2.0")
        assert cwe_ids == ["CWE-79", "CWE-89"]

    def test_nvd_parser_extracts_severity(self):
        """NVD parser extracts severity level."""
        from vulnstate.parsers import NVDParser

        item = {"cve": {"metrics": {"cvssMetricV31": [{"cvssData": {"baseSeverity": "HIGH"}}]}}}

        severity = NVDParser.extract_severity(item, "2.0")
        assert severity == "HIGH"

    def test_nvd_parser_extracts_all_cpes(self):
        """NVD parser extracts all CPE strings from configurations."""
        from vulnstate.parsers import NVDParser

        item = {
            "cve": {
                "configurations": [
                    {
                        "nodes": [
                            {
                                "cpeMatch": [
                                    {"criteria": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"},
                                    {"criteria": "cpe:2.3:a:apache:struts:2.5.0:*:*:*:*:*:*:*"},
                                ]
                            }
                        ]
                    }
                ]
            }
        }

        cpes = NVDParser.extract_all_cpes(item, "2.0")
        assert len(cpes) == 2
        assert "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*" in cpes
        assert "cpe:2.3:a:apache:struts:2.5.0:*:*:*:*:*:*:*" in cpes

    def test_nvd_parser_extracts_vendors_products_from_cpes(self):
        """NVD parser extracts unique vendors and products from CPE strings."""
        from vulnstate.parsers import NVDParser

        cpes = [
            "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*",
            "cpe:2.3:a:apache:struts:2.5.0:*:*:*:*:*:*:*",
            "cpe:2.3:a:microsoft:exchange:2019:*:*:*:*:*:*:*",
        ]

        vendors, products = NVDParser.extract_vendors_products(cpes)
        assert vendors == ["apache", "microsoft"]  # Sorted, unique
        assert products == ["exchange", "log4j", "struts"]  # Sorted, unique

    def test_create_vuln_from_item_stores_cpes_in_metadata(self):
        """create_vuln_from_item stores CPE strings in metadata (vendors/products parsed lazily)."""
        from vulnstate.parsers import NVDParser

        item = {
            "cve": {
                "id": "CVE-2024-12345",
                "published": "2024-01-15T00:00:00Z",
                "configurations": [
                    {
                        "nodes": [
                            {
                                "cpeMatch": [
                                    {"criteria": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"},
                                    {"criteria": "cpe:2.3:a:apache:struts:2.5.0:*:*:*:*:*:*:*"},
                                ]
                            }
                        ]
                    }
                ],
            }
        }

        vuln = NVDParser.create_vuln_from_item("CVE-2024-12345", item, "2.0")

        # CPE strings stored as list
        assert "cpe_strings" in vuln.metadata
        assert len(vuln.metadata["cpe_strings"]) == 2

        # Vendors and products are parsed lazily during export (not stored in metadata)
        # Verify they can still be extracted via extract_vendors_products
        vendors, products = NVDParser.extract_vendors_products(vuln.metadata["cpe_strings"])
        assert vendors == ["apache"]  # Unique, sorted
        assert products == ["log4j", "struts"]  # Unique, sorted

    def test_create_vuln_from_item_stores_metadata(self):
        """create_vuln_from_item stores parsed metadata fields (vendors/products parsed lazily)."""
        from vulnstate.parsers import NVDParser

        item = {
            "cve": {
                "id": "CVE-2024-12345",
                "published": "2024-01-15T00:00:00Z",
                "lastModified": "2024-01-20T00:00:00Z",
                "descriptions": [{"lang": "en", "value": "Test description"}],
                "weaknesses": [{"description": [{"value": "CWE-79"}]}],
                "metrics": {"cvssMetricV31": [{"cvssData": {"baseSeverity": "CRITICAL"}}]},
                "configurations": [
                    {
                        "nodes": [
                            {
                                "cpeMatch": [
                                    {"criteria": "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*"}
                                ]
                            }
                        ]
                    }
                ],
            }
        }

        vuln = NVDParser.create_vuln_from_item("CVE-2024-12345", item, "2.0")

        assert vuln.metadata["description"] == "Test description"
        assert vuln.metadata["cwe_ids"] == ["CWE-79"]
        assert vuln.metadata["severity"] == "CRITICAL"
        # Vendors/products are parsed lazily, verify cpe_strings is stored
        assert "cpe_strings" in vuln.metadata
        vendors, products = NVDParser.extract_vendors_products(vuln.metadata["cpe_strings"])
        assert vendors == ["apache"]  # List of vendors
        assert products == ["log4j"]  # List of products
        assert vuln.metadata["published_date"] == "2024-01-15T00:00:00Z"
        assert vuln.metadata["last_modified"] == "2024-01-20T00:00:00Z"
