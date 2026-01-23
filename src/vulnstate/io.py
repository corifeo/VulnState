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
    def _resolve_source(source: Union[str, Path]) -> str:
        """
        Resolve source to a local file path, downloading from URL if needed.

        Args:
            source: Local file path or HTTP(S) URL

        Returns:
            Local file path (downloaded to temp file if URL)

        Note:
            Requires 'requests' package for URL downloads. Falls back gracefully
            if requests is not available.
        """
        source_str = str(source)

        # Check if it's a URL
        if source_str.startswith(("http://", "https://")):
            try:
                import tempfile

                # TODO: replace with httpx for async support
                # TODO: add test case for URL import
                import requests

                # Download to temp file
                response = requests.get(source_str, timeout=30)
                response.raise_for_status()

                # Create temp file with appropriate extension
                suffix = ".csv" if source_str.endswith(".csv") else ".json"
                with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
                    f.write(response.text)
                    return f.name

            except ImportError:
                raise ImportError(
                    "URL support requires 'requests' package. " "Install with: pip install requests"
                ) from None
            except Exception as e:
                raise ValueError(f"Failed to download from URL: {e}") from e

        # Return local path as-is
        return source_str

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

        import numpy as np

        # Parse source
        if isinstance(source, str):
            # Resolve URL or local path
            local_path = CVDIO._resolve_source(source)
            # Read from CSV file
            kev_data = {}
            with open(local_path, encoding="utf-8") as f:
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

        # Initialize kev metadata if needed
        if "kev" not in arr._metadata_raw:
            arr._metadata_raw["kev"] = np.full(len(arr), False, dtype=bool)

        # Process each vulnerability
        modified_indices: set[int] = set()
        if arr._vulnerabilities is not None and len(arr._vulnerabilities) > 0:
            for i in range(len(arr)):
                cve_id = str(cve_ids[i])
                if not cve_id or cve_id not in kev_data:
                    continue

                vuln = arr.get(i)
                kev_row = kev_data[cve_id]

                # Set kev flag
                arr._metadata_raw["kev"][i] = True
                vuln.kev = True

                # Extract parsed KEV fields with kev_ prefix
                if "vendorProject" in kev_row:
                    vuln.metadata["kev_vendor_name"] = kev_row["vendorProject"]
                if "product" in kev_row:
                    vuln.metadata["kev_product_name"] = kev_row["product"]
                if "shortDescription" in kev_row:
                    vuln.metadata["kev_description"] = kev_row["shortDescription"]
                if "knownRansomwareCampaignUse" in kev_row:
                    vuln.metadata["kev_ransomware_use"] = kev_row["knownRansomwareCampaignUse"]
                if "requiredAction" in kev_row:
                    vuln.metadata["kev_required_action"] = kev_row["requiredAction"]
                if "dueDate" in kev_row:
                    vuln.metadata["kev_due_date"] = kev_row["dueDate"]
                if "notes" in kev_row:
                    vuln.metadata["kev_notes"] = kev_row["notes"]
                if "dateAdded" in kev_row:
                    vuln.metadata["kev_date_added"] = kev_row["dateAdded"]
                    # Store in enrichment as datetime
                    with contextlib.suppress(ValueError, TypeError):
                        vuln.enrichment.kev_date = datetime.fromisoformat(kev_row["dateAdded"])

                # Apply event A
                if apply_event and "dateAdded" in kev_row:
                    date_added = np.datetime64(kev_row["dateAdded"])
                    with contextlib.suppress(ValueError):
                        # Event constraints violated, skip if raised
                        vuln.apply_event(CVDEvent.A, date_added)

                modified_indices.add(i)

            # Sync modified indices to update array-level enrichment arrays
            if modified_indices:
                arr._dirty_indices.update(modified_indices)
                arr.sync()
        else:
            # TODO: Handle expunged _vulnerabilities
            pass

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

        # Parse source
        if isinstance(source, str):
            # Resolve URL or local path
            local_path = CVDIO._resolve_source(source)
            # Read from CSV file
            epss_data = {}
            with open(local_path, encoding="utf-8") as f:
                # Skip comment lines (EPSS files start with #model_version...)
                lines = [line for line in f if not line.startswith("#")]
                reader = csv.DictReader(lines)
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
        modified_indices: set[int] = set()
        if arr._vulnerabilities is not None and len(arr._vulnerabilities) > 0:
            for i in range(len(arr)):
                cve_id = str(cve_ids[i])
                if not cve_id or cve_id not in epss_data:
                    continue

                vuln = arr.get(i)
                epss_value = epss_data[cve_id]

                if import_metadata:
                    # Store full metadata
                    metadata_dict = (
                        epss_value if isinstance(epss_value, dict) else {"epss": epss_value}
                    )

                    # Apply include/exclude filters
                    if include is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k in include}
                    if exclude is not None:
                        metadata_dict = {k: v for k, v in metadata_dict.items() if k not in exclude}

                    if "epss" not in vuln.metadata:
                        vuln.metadata["epss"] = {}
                    vuln.metadata["epss"].update(metadata_dict)

                    # Also set epss score attribute for backward compatibility (handle both "epss" and "score" keys)
                    if "epss" in metadata_dict:
                        vuln.epss = float(metadata_dict["epss"])
                    elif "score" in metadata_dict:
                        vuln.epss = float(metadata_dict["score"])

                    # Extract percentile to enrichment
                    if "percentile" in metadata_dict:
                        vuln.enrichment.epss_percentile = float(metadata_dict["percentile"])
                else:
                    # Just set score (handle both "epss" and "score" keys for backward compatibility)
                    if isinstance(epss_value, float):
                        score = epss_value
                    elif isinstance(epss_value, dict):
                        # Try "score" first (test format), then "epss" (CSV format)
                        score = float(epss_value.get("score", epss_value.get("epss", 0)))
                    else:
                        score = float(epss_value)
                    vuln.epss = score

                    # Also extract percentile if it's a dict
                    if isinstance(epss_value, dict) and "percentile" in epss_value:
                        vuln.enrichment.epss_percentile = float(epss_value["percentile"])

                modified_indices.add(i)

            # Sync modified indices to update array-level enrichment arrays
            if modified_indices:
                arr._dirty_indices.update(modified_indices)
                arr.sync()
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

        import numpy as np

        # Resolve URL or local path
        local_path = CVDIO._resolve_source(source)

        # Read CSV
        csv_data = {}
        with open(local_path, encoding="utf-8") as f:
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
            # Resolve URL or local path
            local_path = CVDIO._resolve_source(source)
            with open(local_path, encoding="utf-8") as f:
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
        infer_vendor: bool = True,
        infer_timestamps: bool = True,
    ) -> None:
        """Import NVD data (delegates to NVDParser)."""
        from .parsers import NVDParser

        return NVDParser.import_nvd(
            arr,
            source,
            apply_event,
            import_metadata,
            include,
            exclude,
            skip_existing,
            infer_vendor=infer_vendor,
            infer_timestamps=infer_timestamps,
        )

    @staticmethod
    def import_nvd_glob(
        arr: "CVDArray",
        pattern: str,
        apply_event: bool = True,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        skip_existing: bool = True,
    ) -> int:
        """Import NVD data from glob pattern (delegates to NVDParser)."""
        from .parsers import NVDParser

        return NVDParser.import_nvd_glob(
            arr, pattern, apply_event, import_metadata, include, exclude, skip_existing
        )

    @staticmethod
    def from_nvd(nvd_items: list[dict[str, Any]], format_version: str = "1.1") -> "CVDArray":
        """Create array from NVD file (delegates to NVDParser)."""
        from .parsers import NVDParser

        return NVDParser.from_nvd(nvd_items, format_version)

    @staticmethod
    def to_dataframe(
        arr: "CVDArray",
        include_analytics: bool = True,
        include_events: bool = False,
        explode_cvss: bool = True,
        explode_metadata: bool = True,
    ) -> "pd.DataFrame":
        """
        Convert CVDArray to pandas DataFrame with comprehensive data.

        Args:
            arr: CVDArray instance
            include_analytics: Include computed analytics fields (default True)
            include_events: Add V, F, D, P, X, A columns as 0/1 flags (default False)
            explode_cvss: Explode CVSS vector into separate columns (default True)
            explode_metadata: Flatten metadata dicts into columns (default True)

        Returns:
            DataFrame with vulnerability data
        """
        import pandas as pd

        # Build DataFrame directly from arrays (vectorized, fast)
        data: dict[str, Any] = {
            "cve_id": arr.cve_ids,
            "vuln_id": arr.vuln_ids,
            "state": arr.states,
            "cvss_score": arr.cvss_scores,
            "epss": arr.epss,
            "kev": arr.kev,
            "state_label": arr.state_labels,
        }

        # Add event flags as 0/1 columns (vectorized)
        if include_events:
            for event in CVDEvent:
                data[event.name] = arr.has_event_occurred(event).astype(np.int8)

        # Add CVSS metrics (vectorized)
        data["attack_vector"] = arr.attack_vector
        data["attack_complexity"] = arr.attack_complexity
        data["privileges_required"] = arr.privileges_required
        data["user_interaction"] = arr.user_interaction
        data["scope"] = arr.scope
        data["confidentiality_impact"] = arr.confidentiality_impact
        data["integrity_impact"] = arr.integrity_impact
        data["availability_impact"] = arr.availability_impact

        # Add analytics if requested (vectorized)
        if include_analytics:
            data["is_fix_available"] = arr.is_fix_available
            data["is_fix_deployed"] = arr.is_fix_deployed
            data["is_weaponized"] = arr.is_weaponized
            data["is_under_attack"] = arr.is_under_attack
            data["is_premature_disclosure"] = arr.is_premature_disclosure
            data["disclosure_window_days"] = arr.disclosure_window_days
            data["fix_lag_days"] = arr.fix_lag_days
            data["deployment_lag_days"] = arr.deployment_lag_days
            data["is_zero_day_exploit"] = arr.is_zero_day_exploit
            data["is_zero_day_attack"] = arr.is_zero_day_attack
            data["is_coordinated"] = arr.is_coordinated
            data["is_responsible_disclosure"] = arr.is_responsible_disclosure
            data["has_fix_before_exploit"] = arr.has_fix_before_exploit
            data["has_fix_before_attack"] = arr.has_fix_before_attack
            data["has_deployment_before_exploit"] = arr.has_deployment_before_exploit
            data["has_deployment_before_attack"] = arr.has_deployment_before_attack
            data["is_private_attack"] = arr.is_private_attack
            data["is_mass_exploitation"] = arr.is_mass_exploitation

        # Add event timestamps (vectorized)
        data["V_timestamp"] = arr.V_timestamps
        data["F_timestamp"] = arr.F_timestamps
        data["D_timestamp"] = arr.D_timestamps
        data["P_timestamp"] = arr.P_timestamps
        data["X_timestamp"] = arr.X_timestamps
        data["A_timestamp"] = arr.A_timestamps

        df = pd.DataFrame(data)

        # Explode metadata if requested
        if explode_metadata:
            # Get metadata dict and flatten it
            metadata_raw = arr.metadata
            for key, values in metadata_raw.items():
                # Skip internal keys and keys already populated from array properties
                # (epss, kev come from arr.enrichment, not metadata)
                if not key.startswith("_") and key not in df.columns:
                    df[key] = values

            # Lazy parse vendors/products from cpe_strings if not already present
            if "cpe_strings" in df.columns and "vendors" not in df.columns:
                from .parsers import NVDParser

                def _parse_vendors(cpes: Any) -> Any:
                    if cpes is None or (isinstance(cpes, float) and pd.isna(cpes)):
                        return None
                    return NVDParser.extract_vendors_products(cpes)[0] if cpes else None

                def _parse_products(cpes: Any) -> Any:
                    if cpes is None or (isinstance(cpes, float) and pd.isna(cpes)):
                        return None
                    return NVDParser.extract_vendors_products(cpes)[1] if cpes else None

                df["vendors"] = df["cpe_strings"].apply(_parse_vendors)
                df["products"] = df["cpe_strings"].apply(_parse_products)

        return df

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

    # Single vulnerability serialization
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
            "state": vuln.state_str,
            "metadata": dict(vuln.metadata) if vuln.metadata else {},
            "cvss_score": vuln.cvss_score,
            "epss": vuln.epss,
            "cve_vector": vuln.cve_vector,
            "kev": vuln.kev,
            "event_timestamps": {
                event.name: (
                    None
                    if timestamp is None
                    else (
                        timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
                    )
                )
                for event, timestamp in vuln.events.items()
            },
            "history": [
                {
                    "event": entry["event"].name if entry["event"] is not None else None,
                    "from_state": entry["from_state"],
                    "to_state": entry["to_state"],
                    "timestamp": (
                        entry["timestamp"].isoformat()
                        if entry["timestamp"] and hasattr(entry["timestamp"], "isoformat")
                        else str(entry["timestamp"]) if entry["timestamp"] else None
                    ),
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
        from datetime import datetime

        from .constants import CVDEvent, string_to_state_int
        from .models import (
            VulnerabilityEnrichment,
            VulnerabilityIdentity,
            VulnerabilityScoring,
            VulnerabilityState,
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
        vuln.scoring = VulnerabilityScoring(
            cvss_base_score=data.get("cvss_score"), cve_vector=data.get("cve_vector")
        )
        vuln.enrichment = VulnerabilityEnrichment(epss=data.get("epss"), kev=data.get("kev", False))
        vuln.state = VulnerabilityState(
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

    @staticmethod
    def array_from_dicts(data: list[dict[str, Any]], include_computed: bool = False) -> "CVDArray":
        """Convert array to list of dictionaries.

        Args:
            data: List of dictionaries from to_dict()
            include_computed: Include computed properties

        Returns:
            CVDArray instance
        """
        from .array import CVDArray

        vulns = [CVDIO.from_dict(d) for d in data]
        return CVDArray(vulns)
