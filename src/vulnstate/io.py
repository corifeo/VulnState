"""
CVD I/O - Data import/export and serialization

Provides:
- CVDIO: Internal implementation of all I/O operations
  - NVD, KEV, EPSS imports
  - CSV/JSON generic imports
  - DataFrame export with analytics
  - Serialization (JSON, pickle)

Layer: I/O
Dependencies: array.py, vulnerability.py, constants.py, formatting.py
Used by: array.py (delegation)
"""

import contextlib
import csv
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from .constants import CVDEvent

if TYPE_CHECKING:
    import pandas as pd

    from .array import CVDArray
    from .vulnerability import CVDVulnerability


class CVDIO:
    """Internal I/O implementation - users interact via CVDArray methods."""

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
                    with contextlib.suppress(ValueError):
                        # Event constraints violated, skip if raised
                        vuln.apply_event(CVDEvent.A, date_added)

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
            nvd_data: Dict mapping CVE IDs to NVD data dicts

        Note: This method is internal. Use arr.import_nvd() instead.
        """
        # Enrich existing vulnerabilities with CVSS data
        if arr._vulnerabilities is not None and len(arr._vulnerabilities) > 0:
            for i in range(len(arr)):
                vuln = arr.get(i)
                cve_id = vuln.identity.vuln_id

                if cve_id in nvd_data:
                    item = nvd_data[cve_id]

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

    @staticmethod
    def import_epss(
        arr: "CVDArray",
        source: Union[str, dict[str, Union[float, dict[str, Any]]]],
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """
        Import EPSS scores.

        Args:
            arr: CVDArray instance to update
            source: Filepath to EPSS CSV or dict mapping CVE IDs to scores/metadata
            import_metadata: Store full EPSS data in vuln.metadata['epss'] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)
        """
        import csv

        # Parse source
        if isinstance(source, str):
            # Read from CSV file
            epss_data = {}
            with open(source) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cve_id = row["cve"]
                    if import_metadata:
                        # Store full row as dict
                        epss_data[cve_id] = dict(row)
                    else:
                        # Just store the score as float
                        epss_data[cve_id] = float(row["epss"])
        else:
            epss_data = source

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
                epss_value = epss_data[cve_id]

                if import_metadata:
                    # Store full metadata
                    metadata_dict = epss_value if isinstance(epss_value, dict) else {"epss": epss_value}

                    # Apply include/exclude filters
                    if include is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k in include}
                    if exclude is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k not in exclude}

                    if "epss" not in vuln.metadata:
                        vuln.metadata["epss"] = {}
                    vuln.metadata["epss"].update(metadata_dict)

                    # Also set epss score attribute for backward compatibility
                    if "epss" in metadata_dict:
                        vuln.epss = float(metadata_dict["epss"])
                else:
                    # Just set score
                    score = epss_value if isinstance(epss_value, float) else float(epss_value.get("epss", 0))
                    vuln.epss = score
        else:
            # TODO: Handle expunged _vulnerabilities
            pass

    @staticmethod
    def import_csv(
        arr: "CVDArray",
        source: str,
        cve_field: str,
        event: Optional["CVDEvent"] = None,
        timestamp_field: Optional[str] = None,
        apply_event: bool = True,
        metadata_namespace: Optional[str] = None,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """
        Import generic CSV file.

        Args:
            arr: CVDArray instance to update
            source: Path to CSV file
            cve_field: Column name containing CVE IDs
            event: Event to apply (if None, no event applied)
            timestamp_field: Column name containing event timestamp
            apply_event: Apply specified event (default True)
            metadata_namespace: Namespace for storing metadata (e.g., 'vendor_advisory')
            import_metadata: Store row data in vuln.metadata[namespace] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)
        """
        import csv

        import numpy as np

        # Read CSV
        csv_data = {}
        with open(source) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if cve_field not in row:
                    continue
                cve_id = row[cve_field]
                csv_data[cve_id] = dict(row)

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
                row = csv_data[cve_id]

                # Apply event
                if apply_event and event is not None and timestamp_field and timestamp_field in row:
                    timestamp = np.datetime64(row[timestamp_field])
                    with contextlib.suppress(ValueError):
                        vuln.apply_event(event, timestamp)

                # Store metadata
                if import_metadata and metadata_namespace:
                    metadata_dict = dict(row)

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
        event: Optional["CVDEvent"] = None,
        timestamp_field: Optional[str] = None,
        apply_event: bool = True,
        metadata_namespace: Optional[str] = None,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """
        Import generic JSON file.

        Args:
            arr: CVDArray instance to update
            source: Path to JSON file or list of dicts
            cve_field: Field path for CVE ID (supports dot notation like 'cve.id')
            event: Event to apply (if None, no event applied)
            timestamp_field: Field path for event timestamp (supports dot notation)
            apply_event: Apply specified event (default True)
            metadata_namespace: Namespace for storing metadata (e.g., 'vendor_advisory')
            import_metadata: Store record in vuln.metadata[namespace] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)
        """
        import numpy as np

        def get_nested(data: dict[str, Any], path: str) -> Any:
            """Extract value from nested dict using dot notation."""
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
            with open(source) as f:
                json_data_list = json.load(f)
        else:
            json_data_list = source

        # Build dict mapping CVE ID -> record
        json_data = {}
        for record in json_data_list:
            cve_id = get_nested(record, cve_field)
            if cve_id:
                json_data[cve_id] = record

        # Get CVE IDs from array
        if "_cve_id" not in arr._metadata_raw:
            return

        cve_ids = arr._metadata_raw["_cve_id"]

        # Process each vulnerability
        if arr._vulnerabilities is not None and len(arr._vulnerabilities) > 0:
            for i in range(len(arr)):
                cve_id = str(cve_ids[i])
                if not cve_id or cve_id not in json_data:
                    continue

                vuln = arr.get(i)
                record = json_data[cve_id]

                # Apply event
                if apply_event and event is not None and timestamp_field:
                    timestamp_value = get_nested(record, timestamp_field)
                    if timestamp_value:
                        timestamp = np.datetime64(timestamp_value)
                        with contextlib.suppress(ValueError):
                            vuln.apply_event(event, timestamp)

                # Store metadata
                if import_metadata and metadata_namespace:
                    metadata_dict = dict(record)

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
    def import_nvd(
        arr: "CVDArray",
        source: Union[str, list[dict[str, Any]]],
        apply_event: bool = True,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        skip_existing: bool = False,
    ) -> None:
        """
        Import NVD vulnerability data.

        Args:
            arr: CVDArray instance to update
            source: Path to NVD JSON file or list of CVE items
            apply_event: Apply event P (Public) from publishedDate (default True)
            import_metadata: Store full NVD record in metadata['nvd'] (default False)
            include: Only import these metadata fields
            exclude: Skip these metadata fields
            skip_existing: Skip CVEs already in array (default False)
        """
        from .array import CVDArray
        from .vulnerability import CVDVulnerability

        # Parse source
        if isinstance(source, str):
            with open(source) as f:
                nvd_feed = json.load(f)
            nvd_items = nvd_feed.get("CVE_Items", nvd_feed.get("vulnerabilities", []))
        else:
            nvd_items = source

        # If array is empty, treat all items as new
        if len(arr) == 0:
            new_vulns = []
            for item in nvd_items:
                cve_id = item.get("cve", {}).get("CVE_data_meta", {}).get("ID")
                if not cve_id:
                    continue

                vuln = CVDIO._create_vuln_from_nvd_item(cve_id, item)

                # Store metadata if requested
                if import_metadata:
                    metadata_dict = dict(item)
                    if include is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k in include}
                    if exclude is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k not in exclude}
                    if "nvd" not in vuln.metadata:
                        vuln.metadata["nvd"] = {}
                    vuln.metadata["nvd"].update(metadata_dict)

                new_vulns.append(vuln)

            # Initialize array with new vulns using _from_list
            arr._from_list(new_vulns)
            return

        # Build set of existing CVE IDs for O(1) lookup
        existing_cves = set()
        if "_cve_id" in arr._metadata_raw:
            existing_cves = {str(cve_id) for cve_id in arr._metadata_raw["_cve_id"]}

        # Build index of existing vulnerabilities by CVE ID
        cve_to_index = {}
        if arr._vulnerabilities is not None and len(arr._vulnerabilities) > 0:
            for i in range(len(arr)):
                vuln = arr.get(i)
                cve_to_index[vuln.cve_id] = i

        new_vulns = []

        # Process NVD items
        for item in nvd_items:
            # Extract CVE ID
            cve_id = item.get("cve", {}).get("CVE_data_meta", {}).get("ID")
            if not cve_id:
                continue

            if cve_id in existing_cves:
                if skip_existing:
                    continue

                # Update existing vulnerability
                idx = cve_to_index.get(cve_id)
                if idx is not None:
                    vuln = arr.get(idx)

                    # Update CVSS score and vector
                    impact = item.get("impact", {})
                    if "baseMetricV3" in impact:
                        cvss_v3 = impact["baseMetricV3"].get("cvssV3", {})
                        cvss_score = cvss_v3.get("baseScore")
                        cvss_vector = cvss_v3.get("vectorString")
                        vuln.cvss_score = cvss_score
                        vuln.cve_vector = cvss_vector
                        # Update metadata arrays
                        if "cvss_score" in arr._metadata_raw:
                            arr._metadata_raw["cvss_score"][idx] = cvss_score
                        if "cve_vector" in arr._metadata_raw:
                            arr._metadata_raw["cve_vector"][idx] = cvss_vector
                    elif "baseMetricV2" in impact:
                        cvss_v2 = impact["baseMetricV2"].get("cvssV2", {})
                        cvss_score = cvss_v2.get("baseScore")
                        cvss_vector = cvss_v2.get("vectorString")
                        vuln.cvss_score = cvss_score
                        vuln.cve_vector = cvss_vector
                        # Update metadata arrays
                        if "cvss_score" in arr._metadata_raw:
                            arr._metadata_raw["cvss_score"][idx] = cvss_score
                        if "cve_vector" in arr._metadata_raw:
                            arr._metadata_raw["cve_vector"][idx] = cvss_vector

                    # Apply event P
                    if apply_event:
                        published_date = item.get("publishedDate")
                        if published_date:
                            try:
                                dt = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
                                with contextlib.suppress(ValueError):
                                    vuln.apply_event(CVDEvent.P, timestamp=dt)
                            except (ValueError, TypeError):
                                pass

                    # Store metadata
                    if import_metadata:
                        metadata_dict = dict(item)

                        # Apply include/exclude filters
                        if include is not None:
                            metadata_dict = {k: v for k, v in metadata_dict.items() if k in include}
                        if exclude is not None:
                            metadata_dict = {k: v for k, v in metadata_dict.items() if k not in exclude}

                        if "nvd" not in vuln.metadata:
                            vuln.metadata["nvd"] = {}
                        vuln.metadata["nvd"].update(metadata_dict)
            else:
                # Create new vulnerability
                vuln = CVDIO._create_vuln_from_nvd_item(cve_id, item)

                # Store metadata if requested
                if import_metadata:
                    metadata_dict = dict(item)

                    # Apply include/exclude filters
                    if include is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k in include}
                    if exclude is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k not in exclude}

                    if "nvd" not in vuln.metadata:
                        vuln.metadata["nvd"] = {}
                    vuln.metadata["nvd"].update(metadata_dict)

                new_vulns.append(vuln)

        # Add new vulnerabilities to array or sync if only updates
        if new_vulns:
            # Rebuild array with existing + new vulnerabilities
            all_vulns = list(arr._vulnerabilities) + new_vulns
            arr._from_list(all_vulns)
        else:
            # No new vulns, but existing ones may have been updated
            # Sync to update metadata arrays
            arr.sync()

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

    @staticmethod
    def to_dataframe(
        arr: "CVDArray",
        include_analytics: bool = True,
        explode_cvss: bool = True,
        explode_metadata: bool = True,
    ) -> "pd.DataFrame":
        """
        Convert CVDArray to pandas DataFrame with comprehensive data.

        Args:
            arr: CVDArray instance
            include_analytics: Include computed analytics fields (default True)
            explode_cvss: Explode CVSS vector into separate columns (default True)
            explode_metadata: Flatten metadata dicts into columns (default True)

        Returns:
            DataFrame with vulnerability data
        """
        import pandas as pd

        # Get base dicts
        dicts = CVDIO.array_to_dicts(arr)

        # Enhance with analytics
        if include_analytics:
            for i, vuln_dict in enumerate(dicts):
                vuln_dict["is_fix_available"] = arr.is_fix_available[i]
                vuln_dict["is_fix_deployed"] = arr.is_fix_deployed[i]
                vuln_dict["has_public_exploit"] = arr.has_public_exploit[i]
                vuln_dict["is_under_attack"] = arr.is_under_attack[i]
                vuln_dict["premature_disclosure"] = arr.premature_disclosure[i]
                vuln_dict["disclosure_window_days"] = arr.disclosure_window_days[i]
                vuln_dict["fix_lag_days"] = arr.fix_lag_days[i]
                vuln_dict["deployment_lag_days"] = arr.deployment_lag_days[i]

        # Explode metadata
        if explode_metadata:
            for i, vuln_dict in enumerate(dicts):
                vuln = arr.get(i)
                for namespace, meta_dict in vuln.metadata.items():
                    if isinstance(meta_dict, dict):
                        for key, value in meta_dict.items():
                            vuln_dict[f"{namespace}_{key}"] = value

        # Explode CVSS vector
        if explode_cvss:
            from .formatting import CVSSFormatter

            for vuln_dict in dicts:
                if vuln_dict.get("cve_vector"):
                    cvss_dict = CVSSFormatter.parse_vector(vuln_dict["cve_vector"])
                    for key, value in cvss_dict.items():
                        vuln_dict[f"cvss_{key}"] = value

        return pd.DataFrame(dicts)

    @staticmethod
    def array_to_dicts(arr: "CVDArray") -> list[dict[str, Any]]:
        """Convert array to list of dicts."""
        dicts = []

        if arr._vulnerabilities is not None and len(arr._vulnerabilities) > 0:
            for i in range(len(arr)):
                vuln = arr.get(i)

                # Basic fields
                vuln_dict = {
                    "vuln_id": vuln.identity.vuln_id,
                    "state": vuln.event_data.state,
                    "cvss_score": vuln.cvss_score,
                    "cve_vector": vuln.cve_vector,
                    "vendor": vuln.vendor,
                    "product": vuln.product,
                    "severity": vuln.severity,
                    "epss": vuln.epss,
                    "is_kev": vuln.is_kev,
                }

                # Event timestamps
                for event in CVDEvent:
                    ts = vuln.event_data.events.get(event)
                    vuln_dict[event.name] = ts

                dicts.append(vuln_dict)

        return dicts

    @staticmethod
    def to_json_file(arr: "CVDArray", filepath: str, include_computed: bool = False) -> None:
        """Save array to JSON file."""
        # Use to_json() for each vulnerability to handle type conversions
        json_list = [json.loads(vuln.to_json(include_computed=include_computed)) for vuln in arr]
        with open(filepath, "w") as f:
            json.dump(json_list, f, indent=2)

    @staticmethod
    def from_json_file(filepath: str) -> "CVDArray":
        """Load array from JSON file."""
        from .array import CVDArray
        from .vulnerability import CVDVulnerability

        with open(filepath) as f:
            data_list = json.load(f)
        vulns = [CVDVulnerability.from_dict(data) for data in data_list]
        return CVDArray(vulns)

    @staticmethod
    def to_pickle_file(arr: "CVDArray", filepath: str) -> None:
        """Save array to pickle file (fast)."""
        with open(filepath, "wb") as f:
            pickle.dump(arr, f)

    @staticmethod
    def from_pickle_file(filepath: str) -> "CVDArray":
        """Load array from pickle file (fast)."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
