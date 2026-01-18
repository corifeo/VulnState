"""
I/O Operations - Unified import/export for CVD objects

Provides:
- CVDIO: Unified I/O class for vulnerabilities and arrays
  - to_dict/from_dict: Dictionary conversion
  - to_json/from_json: JSON string conversion
  - save_json/load_json: JSON file I/O
  - import_epss/import_kev/import_nvdcve: Dict-based enrichment import
  - import_csv: Generic CSV import for any CVD event with timestamps
  - import_json: Generic JSON import with nested field support (dot notation)
  - import_epss_file/import_kev_file/import_nvd_file: File-based import
  - from_nvd: Create array from NVD JSON items

Layer: I/O
Dependencies: vulnerability.py, array.py, models.py, constants.py, states.py
Used by: array.py (convenience wrappers)
"""

import json
import warnings
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union

from .constants import CVDEvent

if TYPE_CHECKING:
    from .array import CVDArray
    from .vulnerability import CVDVulnerability


class CVDIO:
    """Unified I/O operations for CVD vulnerabilities and arrays.

    Consolidates functionality from CVDSerializer and CVDDataImporter
    into a single, consistent interface.
    """

    # ==================== SINGLE VULNERABILITY ====================

    @staticmethod
    def to_dict(vuln: "CVDVulnerability", include_computed: bool = False) -> dict[str, Any]:
        """Convert vulnerability to dictionary.

        Args:
            vuln: CVDVulnerability instance
            include_computed: If True, include computed properties

        Returns:
            Dictionary with all vulnerability data
        """
        data: dict[str, Any] = {
            "cve_id": vuln.cve_id,
            "vuln_id": vuln.vuln_id,
            "state": vuln.state,
            "metadata": dict(vuln.metadata) if vuln.metadata else {},
            "cvss_score": vuln.cvss_score,
            "epss": vuln.epss,
            "cve_vector": vuln.cve_vector,
            "is_kev": vuln.is_kev,
            "event_timestamps": {
                event.name: (None if timestamp is None else timestamp.isoformat())
                for event, timestamp in vuln.events.items()
            },
            "history": [
                {
                    "event": entry["event"].name if entry["event"] is not None else None,
                    "from_state": entry["from_state"],
                    "to_state": entry["to_state"],
                    "timestamp": entry["timestamp"].isoformat() if entry["timestamp"] else None,
                    "actor": entry.get("actor"),
                    "notes": entry.get("notes"),
                }
                for entry in vuln.history
            ],
        }

        if include_computed:
            data["state_label"] = vuln.state_label
            data["history_string"] = vuln.history_string

        return data

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "CVDVulnerability":
        """Reconstruct vulnerability from dictionary.

        Args:
            data: Dictionary from to_dict()

        Returns:
            CVDVulnerability instance
        """
        from .constants import string_to_state_int
        from .models import (
            VulnerabilityEnrichmentData,
            VulnerabilityEventData,
            VulnerabilityIdentity,
            VulnerabilityScoringData,
        )
        from .vulnerability import CVDVulnerability

        vuln_id = data.get("vuln_id") or data.get("internal_id")

        def parse_timestamp(ts_str: Optional[str]) -> Optional[datetime]:
            # Handle None, "TIMESTAMP_UNKNOWN" (legacy), or missing timestamps
            if ts_str is None or ts_str == "TIMESTAMP_UNKNOWN":
                return None  # TIMESTAMP_UNKNOWN
            return datetime.fromisoformat(ts_str)

        events = {
            CVDEvent[event_name]: parse_timestamp(timestamp_str)
            for event_name, timestamp_str in data.get("event_timestamps", {}).items()
        }

        history = [
            {
                "event": CVDEvent[entry["event"]] if entry["event"] is not None else None,
                "from_state": entry["from_state"],
                "to_state": entry["to_state"],
                "timestamp": (
                    datetime.fromisoformat(entry["timestamp"])
                    if entry["timestamp"]
                    else None  # TIMESTAMP_UNKNOWN
                ),
                "actor": entry.get("actor"),
                "notes": entry.get("notes"),
            }
            for entry in data.get("history", [])
        ]

        vuln = CVDVulnerability.__new__(CVDVulnerability)
        # Ensure vuln_id is a string (use cve_id as fallback)
        final_vuln_id = vuln_id or data.get("cve_id") or ""
        vuln.identity = VulnerabilityIdentity(vuln_id=str(final_vuln_id), cve_id=data.get("cve_id"))
        vuln.scoring = VulnerabilityScoringData(
            cvss_base_score=data.get("cvss_score"), cve_vector=data.get("cve_vector")
        )
        vuln.enrichment = VulnerabilityEnrichmentData(
            epss=data.get("epss"), is_kev=data.get("is_kev", False)
        )
        vuln.event_data = VulnerabilityEventData(
            state_encoded=string_to_state_int(data.get("state", "vfdpxa")),
            events=events,
            history=history,
        )
        vuln.metadata = data.get("metadata", {})

        return vuln

    @staticmethod
    def to_json(vuln: "CVDVulnerability", indent: int = 2, include_computed: bool = False) -> str:
        """Convert to JSON string.

        Args:
            vuln: CVDVulnerability instance
            indent: JSON indentation level
            include_computed: Include computed properties

        Returns:
            JSON string
        """
        data = CVDIO.to_dict(vuln, include_computed=include_computed)
        return json.dumps(data, indent=indent, default=str)

    @staticmethod
    def from_json(json_str: str) -> "CVDVulnerability":
        """Reconstruct from JSON string.

        Args:
            json_str: JSON string from to_json()

        Returns:
            CVDVulnerability instance
        """
        return CVDIO.from_dict(json.loads(json_str))

    @staticmethod
    def save_json(vuln: "CVDVulnerability", filepath: str, include_computed: bool = False) -> None:
        """Save vulnerability to JSON file.

        Args:
            vuln: CVDVulnerability instance
            filepath: Path to .json file
            include_computed: Include computed properties
        """
        with open(filepath, "w") as f:
            f.write(CVDIO.to_json(vuln, include_computed=include_computed))

    @staticmethod
    def load_json(filepath: str) -> "CVDVulnerability":
        """Load vulnerability from JSON file.

        Args:
            filepath: Path to .json file

        Returns:
            CVDVulnerability instance
        """
        with open(filepath) as f:
            return CVDIO.from_json(f.read())

    # ==================== ARRAY OPERATIONS ====================

    @staticmethod
    def array_to_dicts(arr: "CVDArray", include_computed: bool = False) -> list[dict[str, Any]]:
        """Convert array to list of dictionaries.

        Args:
            arr: CVDArray instance
            include_computed: Include computed properties

        Returns:
            List of vulnerability dictionaries
        """
        return [CVDIO.to_dict(vuln, include_computed=include_computed) for vuln in arr]

    @staticmethod
    def array_from_dicts(data: list[dict[str, Any]]) -> "CVDArray":
        """Create array from list of dictionaries.

        Args:
            data: List of vulnerability dictionaries

        Returns:
            CVDArray instance
        """
        from .array import CVDArray

        vulns = [CVDIO.from_dict(d) for d in data]
        return CVDArray(vulns)

    # ==================== ENRICHMENT IMPORT ====================

    @staticmethod
    def import_epss(
        arr: "CVDArray",
        source: Union[str, dict[str, Union[float, dict[str, Any]]]],
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """
        Import EPSS (Exploit Prediction Scoring System) data.

        Args:
            arr: CVDArray instance to update
            source: Filepath to EPSS CSV or dict mapping CVE IDs to EPSS data
            import_metadata: Store full EPSS data in vuln.metadata['epss'] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)

        Note:
            Score is always stored in enrichment.epss for performance.
            Metadata storage is optional for preserving percentile and other fields.
        """
        import csv

        import numpy as np

        # Parse source
        if isinstance(source, str):
            # Read from CSV file
            epss_data: dict[str, dict[str, Any]] = {}
            with open(source) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cve_id = row["cve"]
                    epss_data[cve_id] = {
                        "score": float(row["epss"]),
                        "percentile": float(row["percentile"]),
                    }
        else:
            # Handle dict input - normalize to dict[str, dict] format
            epss_data = {}
            for cve_id, value in source.items():
                if isinstance(value, dict):
                    epss_data[cve_id] = value
                else:
                    # Legacy float format
                    epss_data[cve_id] = {"score": float(value)}

        # Initialize epss metadata if needed
        if "epss" not in arr._metadata_raw:
            arr._metadata_raw["epss"] = np.full(len(arr), np.nan, dtype=np.float32)

        # Get CVE IDs from array
        if "_cve_id" not in arr._metadata_raw:
            return

        cve_ids = arr._metadata_raw["_cve_id"]

        # Process each vulnerability
        if arr._vulnerabilities is not None and len(arr._vulnerabilities) > 0:
            for i in range(len(arr)):
                cve_id = str(cve_ids[i])
                if not cve_id or cve_id not in epss_data:
                    continue

                vuln = arr.get(i)
                epss_row = epss_data[cve_id]

                # Extract score
                if "score" in epss_row:
                    try:
                        score = float(epss_row["score"])
                        if 0.0 <= score <= 1.0:
                            arr._metadata_raw["epss"][i] = score
                            vuln.epss = score
                        else:
                            warnings.warn(f"Invalid EPSS score for {cve_id}: {score}", stacklevel=2)
                    except (ValueError, TypeError):
                        warnings.warn(
                            f"Invalid EPSS score for {cve_id}: {epss_row['score']}", stacklevel=2
                        )

                # Store metadata
                if import_metadata:
                    metadata_dict = dict(epss_row)

                    # Apply include/exclude filters
                    if include is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k in include}
                    if exclude is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k not in exclude}

                    if "epss" not in vuln.metadata:
                        vuln.metadata["epss"] = {}
                    vuln.metadata["epss"].update(metadata_dict)
        else:
            # TODO: Handle expunged _vulnerabilities
            pass

    @staticmethod
    def import_kev(
        arr: "CVDArray",
        source: Union[str, dict[str, dict[str, Any]]],
        apply_event: bool = True,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """
        Import KEV catalog data.

        Args:
            arr: CVDArray instance to update
            source: Filepath to KEV CSV or dict mapping CVE IDs to KEV data
            apply_event: Apply event A with dateAdded timestamp (default True)
            import_metadata: Store full KEV data in vuln.metadata['kev'] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)
        """
        import csv

        import numpy as np

        # Parse source
        if isinstance(source, str):
            # Read from CSV file
            kev_data = {}
            with open(source) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cve_id = row["cveID"]
                    kev_data[cve_id] = dict(row)
        else:
            kev_data = source

        # Get CVE IDs from array
        if "_cve_id" not in arr._metadata_raw:
            return

        cve_ids = arr._metadata_raw["_cve_id"]

        # Initialize is_kev metadata if needed
        if "is_kev" not in arr._metadata_raw:
            arr._metadata_raw["is_kev"] = np.full(len(arr), False, dtype=bool)

        # Process each vulnerability
        if arr._vulnerabilities is not None and len(arr._vulnerabilities) > 0:
            for i in range(len(arr)):
                cve_id = str(cve_ids[i])
                if not cve_id or cve_id not in kev_data:
                    continue

                vuln = arr.get(i)
                kev_row = kev_data[cve_id]

                # Set is_kev flag
                arr._metadata_raw["is_kev"][i] = True
                vuln.is_kev = True

                # Apply event A
                if apply_event and "dateAdded" in kev_row:
                    date_added = np.datetime64(kev_row["dateAdded"])
                    try:
                        vuln.apply_event(CVDEvent.A, date_added)
                    except ValueError:
                        # Event constraints violated, skip
                        pass

                # Store metadata
                if import_metadata:
                    metadata_dict = dict(kev_row)

                    # Apply include/exclude filters
                    if include is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k in include}
                    if exclude is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k not in exclude}

                    if "kev" not in vuln.metadata:
                        vuln.metadata["kev"] = {}
                    vuln.metadata["kev"].update(metadata_dict)
        else:
            # TODO: Handle expunged _vulnerabilities
            pass

    @staticmethod
    def import_nvdcve(arr: "CVDArray", nvd_data: dict[str, dict[str, Any]]) -> None:
        """Import NVD data into array.

        Args:
            arr: CVDArray instance to update
            nvd_data: Dictionary mapping CVE IDs to NVD data dicts
        """
        import numpy as np

        if "cvss_score" not in arr._metadata_raw:
            arr._metadata_raw["cvss_score"] = np.full(len(arr), np.nan, dtype=np.float32)
        if "cve_vector" not in arr._metadata_raw:
            arr._metadata_raw["cve_vector"] = np.full(len(arr), None, dtype=object)

        if "_cve_id" not in arr._metadata_raw:
            return

        cve_ids = arr._metadata_raw["_cve_id"]

        for i in range(len(arr)):
            cve_id = str(cve_ids[i])
            if cve_id and cve_id in nvd_data:
                nvd_info = nvd_data[cve_id]
                vuln = arr.get(i)

                if "cvss_score" in nvd_info:
                    try:
                        score = float(nvd_info["cvss_score"])
                        if 0.0 <= score <= 10.0:
                            arr._metadata_raw["cvss_score"][i] = score
                            vuln.cvss_score = score
                        else:
                            warnings.warn(f"Invalid CVSS score for {cve_id}: {score}", stacklevel=2)
                    except (ValueError, TypeError):
                        warnings.warn(f"Invalid CVSS score for {cve_id}", stacklevel=2)

                if "cve_vector" in nvd_info:
                    arr._metadata_raw["cve_vector"][i] = nvd_info["cve_vector"]
                    vuln.cve_vector = nvd_info["cve_vector"]

    # ==================== FILE-BASED IMPORT ====================

    @staticmethod
    def import_csv(
        arr: "CVDArray",
        source: Union[str, list[dict[str, Any]]],
        cve_column: str,
        event: CVDEvent,
        timestamp_column: str,
        apply_event: bool = True,
        metadata_namespace: Optional[str] = None,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """
        Generic CSV import that applies any CVD event with timestamps.

        This method enables importing vendor patch advisories (event F),
        threat intel (event A), or any custom timeline data. It can match
        CVEs and apply the specified event with a timestamp, optionally
        storing the full row data in metadata.

        Args:
            arr: CVDArray instance to update
            source: Filepath to CSV or list of row dicts
            cve_column: Column name containing CVE IDs
            event: CVD event to apply (V, F, D, P, X, or A)
            timestamp_column: Column name containing event timestamps
            apply_event: Apply event with timestamp (default True)
            metadata_namespace: Namespace for storing row metadata (default: None)
            import_metadata: Store full row in vuln.metadata[namespace] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)

        Example:
            # Import vendor patch dates (event F)
            >>> arr.import_csv(
            ...     source='vendor_patches.csv',
            ...     cve_column='cve_id',
            ...     event=CVDEvent.F,
            ...     timestamp_column='patch_date',
            ...     apply_event=True
            ... )

            # Import threat intel with metadata
            >>> arr.import_csv(
            ...     source='threat_intel.csv',
            ...     cve_column='cve',
            ...     event=CVDEvent.A,
            ...     timestamp_column='attack_date',
            ...     apply_event=True,
            ...     import_metadata=True,
            ...     metadata_namespace='threat_intel'
            ... )
        """
        import csv

        import numpy as np

        # Parse source
        if isinstance(source, str):
            # Read from CSV file
            csv_data: dict[str, dict[str, Any]] = {}
            with open(source) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cve_id = row[cve_column]
                    csv_data[cve_id] = dict(row)
        else:
            # List of dicts
            csv_data = {row[cve_column]: row for row in source}

        # Get CVE IDs from array
        if "_cve_id" not in arr._metadata_raw:
            return

        cve_ids = arr._metadata_raw["_cve_id"]

        # Process each vulnerability
        if arr._vulnerabilities is not None and len(arr._vulnerabilities) > 0:
            for i in range(len(arr)):
                cve_id = str(cve_ids[i])
                if not cve_id or cve_id not in csv_data:
                    continue

                vuln = arr.get(i)
                csv_row = csv_data[cve_id]

                # Apply event with timestamp
                if apply_event and timestamp_column in csv_row:
                    timestamp = np.datetime64(csv_row[timestamp_column])
                    try:
                        vuln.apply_event(event, timestamp)
                    except ValueError:
                        # Event constraints violated, skip
                        pass

                # Store metadata
                if import_metadata and metadata_namespace:
                    metadata_dict = dict(csv_row)

                    # Apply include/exclude filters
                    if include is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k in include}
                    if exclude is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k not in exclude}

                    if metadata_namespace not in vuln.metadata:
                        vuln.metadata[metadata_namespace] = {}
                    vuln.metadata[metadata_namespace].update(metadata_dict)
        else:
            # TODO: Handle expunged _vulnerabilities
            pass

    @staticmethod
    def import_json(
        arr: "CVDArray",
        source: Union[str, list[dict[str, Any]]],
        cve_field: str,
        event: CVDEvent,
        timestamp_field: str,
        apply_event: bool = True,
        metadata_namespace: Optional[str] = None,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """
        Generic JSON import that applies any CVD event with timestamps.

        This method enables importing data from JSON files or lists of dicts
        with nested field access using dot notation (e.g., "vulnerability.cve_id").
        Supports both JSON arrays and newline-delimited JSON.

        Args:
            arr: CVDArray instance to update
            source: Filepath to JSON or list of dicts
            cve_field: Field path to CVE IDs (supports dot notation for nested fields)
            event: CVD event to apply (V, F, D, P, X, or A)
            timestamp_field: Field path to event timestamps (supports dot notation)
            apply_event: Apply event with timestamp (default True)
            metadata_namespace: Namespace for storing row metadata (default: None)
            import_metadata: Store full row in vuln.metadata[namespace] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)

        Example:
            # Import from nested JSON structure
            >>> arr.import_json(
            ...     source='threat_intel.json',
            ...     cve_field='vulnerability.cve_id',
            ...     event=CVDEvent.A,
            ...     timestamp_field='threat_intel.first_observed',
            ...     apply_event=True
            ... )

            # Import with metadata storage
            >>> arr.import_json(
            ...     source='vendor_data.json',
            ...     cve_field='cve.id',
            ...     event=CVDEvent.F,
            ...     timestamp_field='patch.release_date',
            ...     apply_event=True,
            ...     import_metadata=True,
            ...     metadata_namespace='vendor'
            ... )
        """
        import numpy as np

        def get_nested(data: dict[str, Any], path: str) -> Any:
            """Extract nested value using dot notation."""
            keys = path.split(".")
            value = data
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value

        # Parse source
        if isinstance(source, str):
            # Read from JSON file
            with open(source) as f:
                content = f.read().strip()
                # Try to parse as JSON array first
                try:
                    json_data = json.loads(content)
                    if not isinstance(json_data, list):
                        # Single object -> wrap in list
                        json_data = [json_data]
                except json.JSONDecodeError:
                    # Try newline-delimited JSON
                    json_data = [json.loads(line) for line in content.split("\n") if line.strip()]
        else:
            # List of dicts
            json_data = source

        # Build lookup dict by CVE ID
        cve_lookup: dict[str, dict[str, Any]] = {}
        for row in json_data:
            cve_id = get_nested(row, cve_field)
            if cve_id:
                cve_lookup[str(cve_id)] = row

        # Get CVE IDs from array
        if "_cve_id" not in arr._metadata_raw:
            return

        cve_ids = arr._metadata_raw["_cve_id"]

        # Process each vulnerability
        if arr._vulnerabilities is not None and len(arr._vulnerabilities) > 0:
            for i in range(len(arr)):
                cve_id = str(cve_ids[i])
                if not cve_id or cve_id not in cve_lookup:
                    continue

                vuln = arr.get(i)
                json_row = cve_lookup[cve_id]

                # Apply event with timestamp
                if apply_event and timestamp_field:
                    timestamp_value = get_nested(json_row, timestamp_field)
                    if timestamp_value:
                        timestamp = np.datetime64(timestamp_value)
                        try:
                            vuln.apply_event(event, timestamp)
                        except ValueError:
                            # Event constraints violated, skip
                            pass

                # Store metadata
                if import_metadata and metadata_namespace:
                    metadata_dict = dict(json_row)

                    # Apply include/exclude filters
                    if include is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k in include}
                    if exclude is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k not in exclude}

                    if metadata_namespace not in vuln.metadata:
                        vuln.metadata[metadata_namespace] = {}
                    vuln.metadata[metadata_namespace].update(metadata_dict)
        else:
            # TODO: Handle expunged _vulnerabilities
            pass

    @staticmethod
    def import_epss_file(
        arr: "CVDArray",
        filepath: str,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """
        Import EPSS scores from CSV file (convenience wrapper).

        Expected CSV format:
            cve,epss,percentile
            CVE-2024-001,0.85432,0.95123

        Args:
            arr: CVDArray instance to update
            filepath: Path to EPSS CSV file
            import_metadata: Store full EPSS data in vuln.metadata['epss'] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)
        """
        CVDIO.import_epss(arr, filepath, import_metadata=import_metadata, include=include, exclude=exclude)

    @staticmethod
    def import_kev_file(
        arr: "CVDArray",
        filepath: str,
        apply_event: bool = True,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """Import KEV catalog from CSV file.

        Expected CSV format (CISA KEV catalog):
            cveID,vendorProject,product,vulnerabilityName,dateAdded,...

        Args:
            arr: CVDArray instance to update
            filepath: Path to KEV CSV file
            apply_event: Apply event A with dateAdded timestamp (default True)
            import_metadata: Store full KEV data in metadata['kev'] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)
        """
        CVDIO.import_kev(arr, filepath, apply_event, import_metadata, include, exclude)

    @staticmethod
    def import_nvd_file(arr: "CVDArray", filepath: str) -> None:
        """Import NVD data from JSON file (NVD Feed 1.1 format).

        Falls back to CVSS v2 if v3 is not available.

        Args:
            arr: CVDArray instance to update
            filepath: Path to NVD JSON file
        """
        with open(filepath) as f:
            nvd_feed = json.load(f)

        nvd_data: dict[str, dict[str, Any]] = {}
        for item in nvd_feed.get("CVE_Items", []):
            cve_id = item["cve"]["CVE_data_meta"]["ID"]
            nvd_info: dict[str, Any] = {}

            impact = item.get("impact", {})
            if "baseMetricV3" in impact:
                cvss_v3 = impact["baseMetricV3"]["cvssV3"]
                nvd_info["cvss_score"] = cvss_v3["baseScore"]
                nvd_info["cve_vector"] = cvss_v3.get("vectorString", "")
            elif "baseMetricV2" in impact:
                cvss_v2 = impact["baseMetricV2"]["cvssV2"]
                nvd_info["cvss_score"] = cvss_v2["baseScore"]
                nvd_info["cve_vector"] = cvss_v2.get("vectorString", "")

            if nvd_info:
                nvd_data[cve_id] = nvd_info

        CVDIO.import_nvdcve(arr, nvd_data)

    # ==================== NVD IMPORT ====================

    @staticmethod
    def from_nvd(nvd_items: list[dict[str, Any]]) -> "CVDArray":
        """Create CVDArray from NVD CVE items (NVD JSON 2.0 format).

        Args:
            nvd_items: List of CVE items from NVD JSON feed

        Returns:
            CVDArray with vulnerabilities populated from NVD data

        Example:
            >>> with open("nvdcve-1.1-2024.json") as f:
            ...     data = json.load(f)
            >>> arr = CVDIO.from_nvd(data["CVE_Items"])
        """
        from .array import CVDArray

        vulns = []
        for item in nvd_items:
            # Extract CVE ID
            cve_id = item.get("cve", {}).get("CVE_data_meta", {}).get("ID")
            if not cve_id:
                continue

            # Create vulnerability
            vuln = CVDIO._create_vuln_from_nvd_item(cve_id, item)
            vulns.append(vuln)

        return CVDArray(vulns)

    @staticmethod
    def _create_vuln_from_nvd_item(cve_id: str, item: dict[str, Any]) -> "CVDVulnerability":
        """Create a CVDVulnerability from an NVD item."""
        from .vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id)

        # Extract CVSS score
        impact = item.get("impact", {})
        if "baseMetricV3" in impact:
            cvss_v3 = impact["baseMetricV3"].get("cvssV3", {})
            vuln.cvss_score = cvss_v3.get("baseScore")
            vuln.cve_vector = cvss_v3.get("vectorString")
        elif "baseMetricV2" in impact:
            cvss_v2 = impact["baseMetricV2"].get("cvssV2", {})
            vuln.cvss_score = cvss_v2.get("baseScore")
            vuln.cve_vector = cvss_v2.get("vectorString")

        # Extract published date as P event
        published_date = item.get("publishedDate")
        if published_date:
            try:
                # NVD format: "2024-01-15T12:00:00Z"
                dt = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
                vuln.apply_event(CVDEvent.P, timestamp=dt)
            except (ValueError, TypeError):
                pass

        return vuln
