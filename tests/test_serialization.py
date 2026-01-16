"""
Comprehensive serialization tests for CVDVulnerability and CVDArray.

Tests JSON and Pickle serialization with roundtrip verification,
computed properties, and batch operations.
"""

import json
import time
from datetime import datetime

import pytest

from vulnstate import CVDArray, CVDEvent, CVDVulnerability


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
        assert vuln2.state == vuln.state

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
        assert vuln2.state == vuln.state
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
        assert vuln2.state == vuln.state
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
        assert all("cve_id" in d for d in data_list)  # Changed from vuln_id to cve_id
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
        assert arr2[0].cve_id == "BATCH-0000"  # Changed from vuln_id to cve_id

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
        assert arr2[0].cve_id == "PICKLE-0000"  # Changed from vuln_id to cve_id

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
        assert vuln2.state == "VFDpxa"  # V, F, and D applied
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


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_vulnerability_serialization(self):
        """Test serialization of new vulnerability with no events."""
        vuln = CVDVulnerability("EMPTY-001")

        data = vuln.to_dict()
        json_str = vuln.to_json()
        vuln2 = CVDVulnerability.from_json(json_str)

        assert data["state"] == "vfdpxa"
        assert vuln2.state == "vfdpxa"
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
