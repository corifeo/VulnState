"""
Parsers - NVD Data Import and Processing

Provides:
- NVDParser: Parser for NVD (National Vulnerability Database) JSON feeds

Layer: I/O
Dependencies: constants.py, vulnerability.py, array.py
Used by: io.py
"""

import contextlib
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union

if TYPE_CHECKING:
    from .array import CVDArray
    from .vulnerability import CVDVulnerability

from .constants import CVDEvent


class NVDParser:
    """
    Parser for NVD (National Vulnerability Database) JSON feeds.

    Supports both NVD 1.1 and 2.0 formats with automatic detection.
    Provides methods for importing NVD data into CVDArray instances.

    Format Detection:
        - NVD 2.0: Checks for "version": "2.0" or "vulnerabilities" array
        - NVD 1.1: Checks for "CVE_Items" array or defaults to 1.1

    CVSS Extraction:
        - Supports CVSS v3.1, v3.0, and v2.0 with automatic fallback
        - Prefers "Primary" source in NVD 2.0 format

    Usage:
        >>> from vulnstate.parsers import NVDParser
        >>> from vulnstate import CVDArray
        >>> arr = CVDArray.zeros(0)
        >>> NVDParser.import_nvd(arr, "nvdcve-2.0-2023.json")
        >>> len(arr)
        31122
    """

    @staticmethod
    def detect_format(nvd_feed: dict[str, Any]) -> str:
        """
        Detect NVD JSON format version (1.1 vs 2.0).

        Args:
            nvd_feed: Loaded NVD JSON feed

        Returns:
            "1.1" or "2.0"
        """
        # Check for explicit version field (NVD 2.0)
        if "version" in nvd_feed:
            version = nvd_feed["version"]
            if isinstance(version, str) and version.startswith("2"):
                return "2.0"

        # Check for vulnerabilities array (NVD 2.0)
        if "vulnerabilities" in nvd_feed:
            return "2.0"

        # Check for CVE_Items array (NVD 1.1)
        if "CVE_Items" in nvd_feed:
            return "1.1"

        # Default to 1.1 for backwards compatibility
        return "1.1"

    @staticmethod
    def extract_cve_id(item: dict[str, Any], format_version: str) -> Optional[str]:
        """
        Extract CVE ID from NVD item based on format version.

        Args:
            item: NVD vulnerability item
            format_version: "1.1" or "2.0"

        Returns:
            CVE ID string or None
        """
        if format_version == "2.0":
            # NVD 2.0: item['cve']['id']
            cve_data = item.get("cve", {})
            cve_id = cve_data.get("id") if isinstance(cve_data, dict) else None
            return str(cve_id) if cve_id is not None else None
        else:
            # NVD 1.1: item['cve']['CVE_data_meta']['ID']
            cve_data = item.get("cve", {})
            if isinstance(cve_data, dict):
                meta_data = cve_data.get("CVE_data_meta", {})
                if isinstance(meta_data, dict):
                    cve_id = meta_data.get("ID")
                    return str(cve_id) if cve_id is not None else None
            return None

    @staticmethod
    def extract_cvss(item: dict[str, Any], format_version: str) -> tuple[Optional[float], Optional[str]]:
        """
        Extract CVSS score and vector from NVD item based on format version.

        Args:
            item: NVD vulnerability item
            format_version: "1.1" or "2.0"

        Returns:
            Tuple of (cvss_score, cvss_vector), both Optional
        """
        if format_version == "2.0":
            # NVD 2.0: item['cve']['metrics']['cvssMetricV31'][0]['cvssData']
            metrics = item.get("cve", {}).get("metrics", {})

            # Try CVSS v3.1 first (cvssMetricV31)
            cvss_v31_list = metrics.get("cvssMetricV31", [])
            if cvss_v31_list:
                # Prefer primary source, fallback to first available
                for metric in cvss_v31_list:
                    if metric.get("type") == "Primary":
                        cvss_data = metric.get("cvssData", {})
                        return cvss_data.get("baseScore"), cvss_data.get("vectorString")
                # No primary, use first one
                cvss_data = cvss_v31_list[0].get("cvssData", {})
                return cvss_data.get("baseScore"), cvss_data.get("vectorString")

            # Fallback to CVSS v3.0 (cvssMetricV30)
            cvss_v30_list = metrics.get("cvssMetricV30", [])
            if cvss_v30_list:
                cvss_data = cvss_v30_list[0].get("cvssData", {})
                return cvss_data.get("baseScore"), cvss_data.get("vectorString")

            # Fallback to CVSS v2.0
            cvss_v2_list = metrics.get("cvssMetricV2", [])
            if cvss_v2_list:
                cvss_data = cvss_v2_list[0].get("cvssData", {})
                return cvss_data.get("baseScore"), cvss_data.get("vectorString")

        else:
            # NVD 1.1: item['impact']['baseMetricV3']['cvssV3']
            impact = item.get("impact", {})

            # Try CVSS v3
            if "baseMetricV3" in impact:
                cvss_v3 = impact["baseMetricV3"].get("cvssV3", {})
                return cvss_v3.get("baseScore"), cvss_v3.get("vectorString")

            # Fallback to CVSS v2
            if "baseMetricV2" in impact:
                cvss_v2 = impact["baseMetricV2"].get("cvssV2", {})
                return cvss_v2.get("baseScore"), cvss_v2.get("vectorString")

        # No CVSS data found
        return None, None

    @staticmethod
    def extract_published_date(item: dict[str, Any], format_version: str) -> Optional[str]:
        """
        Extract published date from NVD item based on format version.

        Args:
            item: NVD vulnerability item
            format_version: "1.1" or "2.0"

        Returns:
            Published date string or None
        """
        if format_version == "2.0":
            # NVD 2.0: item['cve']['published']
            cve_data = item.get("cve", {})
            if isinstance(cve_data, dict):
                pub_date = cve_data.get("published")
                return str(pub_date) if pub_date is not None else None
            return None
        else:
            # NVD 1.1: item['publishedDate']
            pub_date = item.get("publishedDate")
            return str(pub_date) if pub_date is not None else None

    @staticmethod
    def create_vuln_from_item(cve_id: str, item: dict[str, Any], format_version: str = "1.1") -> "CVDVulnerability":
        """
        Create a CVDVulnerability from an NVD item.

        Args:
            cve_id: CVE identifier
            item: NVD vulnerability item
            format_version: "1.1" or "2.0"

        Returns:
            CVDVulnerability instance
        """
        from .vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id)

        # Extract CVSS score and vector using format-aware helper
        cvss_score, cvss_vector = NVDParser.extract_cvss(item, format_version)
        vuln.cvss_score = cvss_score
        vuln.cve_vector = cvss_vector

        # Extract published date as P event
        published_date = NVDParser.extract_published_date(item, format_version)
        if published_date:
            try:
                # NVD format: "2024-01-15T12:00:00Z" or "2024-01-15T12:00:00.627"
                dt = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
                vuln.apply_event(CVDEvent.P, timestamp=dt)
            except (ValueError, TypeError):
                pass

        return vuln

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
        Import NVD vulnerability data (supports both 1.1 and 2.0 formats).

        Args:
            arr: CVDArray instance to update
            source: Path to NVD JSON file or list of CVE items
            apply_event: Apply event P (Public) from publishedDate (default True)
            import_metadata: Store full NVD record in metadata['nvd'] (default False)
            include: Only import these metadata fields
            exclude: Skip these metadata fields
            skip_existing: Skip CVEs already in array (default False)

        Note:
            Automatically detects NVD format version (1.1 or 2.0) and parses accordingly.
        """
        from .io import CVDIO

        # Parse source and detect format
        if isinstance(source, str):
            # Resolve URL or local path
            local_path = CVDIO._resolve_source(source)
            with open(local_path, encoding="utf-8") as f:
                nvd_feed = json.load(f)

            # Detect format version
            format_version = NVDParser.detect_format(nvd_feed)

            # Extract items based on format
            if format_version == "2.0":
                nvd_items = nvd_feed.get("vulnerabilities", [])
            else:
                nvd_items = nvd_feed.get("CVE_Items", [])
        else:
            # When passed as list, assume format will be detected from item structure
            nvd_items = source
            # Detect format from first item if available
            if nvd_items:
                # Try to detect from item structure
                first_item = nvd_items[0]
                if "cve" in first_item and "id" in first_item.get("cve", {}):
                    format_version = "2.0"
                else:
                    format_version = "1.1"
            else:
                format_version = "1.1"

        # For fixed-size arrays, check capacity before processing
        if hasattr(arr, '_fixed_size') and arr._fixed_size:
            capacity = len(arr)

            # Count new items (not already in array)
            existing_cve_ids: set[str] = set()
            if "_cve_id" in arr._metadata_raw:
                existing_cve_ids = {str(cve_id) for cve_id in arr._metadata_raw["_cve_id"]}

            new_items = []
            for item in nvd_items:
                cve_id = NVDParser.extract_cve_id(item, format_version)
                if cve_id and cve_id not in existing_cve_ids:
                    new_items.append(item)

            if len(new_items) > capacity:
                raise ValueError(
                    f"Import exceeds fixed-size array capacity: "
                    f"array capacity={capacity}, new items={len(new_items)}. "
                    f"Use CVDArray() for dynamic arrays or CVDArray.from_nvd() to create from data."
                )

        # If array is empty, treat all items as new
        if len(arr) == 0:
            new_vulns = []
            for item in nvd_items:
                cve_id = NVDParser.extract_cve_id(item, format_version)
                if not cve_id:
                    continue

                vuln = NVDParser.create_vuln_from_item(cve_id, item, format_version)

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
            cve_id = NVDParser.extract_cve_id(item, format_version)
            if not cve_id:
                continue

            if cve_id in existing_cves:
                if skip_existing:
                    continue

                # Update existing vulnerability
                idx = cve_to_index.get(cve_id)
                if idx is not None:
                    vuln = arr.get(idx)

                    # Update CVSS score and vector using format-aware helper
                    cvss_score, cvss_vector = NVDParser.extract_cvss(item, format_version)
                    if cvss_score is not None:
                        vuln.cvss_score = cvss_score
                        vuln.cve_vector = cvss_vector
                        # Update metadata arrays
                        if "cvss_score" in arr._metadata_raw:
                            arr._metadata_raw["cvss_score"][idx] = cvss_score
                        if "cve_vector" in arr._metadata_raw:
                            arr._metadata_raw["cve_vector"][idx] = cvss_vector

                    # Apply event P
                    if apply_event:
                        published_date = NVDParser.extract_published_date(item, format_version)
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
                vuln = NVDParser.create_vuln_from_item(cve_id, item, format_version)

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
    def import_nvd_glob(
        arr: "CVDArray",
        pattern: str,
        apply_event: bool = True,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
        skip_existing: bool = True,
    ) -> int:
        """
        Import NVD data from multiple files matching a glob pattern.

        Args:
            arr: CVDArray instance to update
            pattern: Glob pattern for NVD JSON files (e.g., 'nvdcve-*.json')
            apply_event: Apply event P (Public) from publishedDate (default True)
            import_metadata: Store full NVD record in metadata['nvd'] (default False)
            include: Only import these metadata fields
            exclude: Skip these metadata fields
            skip_existing: Skip CVEs already in array (default True)

        Returns:
            Number of files processed

        Example:
            >>> arr = CVDArray.zeros(0)
            >>> count = NVDParser.import_nvd_glob(arr, 'nvdcve-2.0-*.json')
            >>> print(f"Loaded from {count} files")
        """
        import glob as glob_module

        # Find all matching files
        matched_files = sorted(glob_module.glob(pattern))

        if not matched_files:
            return 0

        # Import each file
        for filepath in matched_files:
            NVDParser.import_nvd(
                arr,
                source=filepath,
                apply_event=apply_event,
                import_metadata=import_metadata,
                include=include,
                exclude=exclude,
                skip_existing=skip_existing,
            )

        return len(matched_files)

    @classmethod
    def from_nvd(cls, nvd_items: list[dict[str, Any]], format_version: str = "1.1") -> "CVDArray":
        """
        Create CVDArray from NVD CVE items (supports both 1.1 and 2.0 formats).

        Args:
            nvd_items: List of CVE items from NVD JSON feed
            format_version: NVD format version ("1.1" or "2.0")

        Returns:
            CVDArray with vulnerabilities populated from NVD data

        Example:
            >>> with open("nvdcve-1.1-2024.json") as f:
            ...     data = json.load(f)
            >>> arr = NVDParser.from_nvd(data["CVE_Items"], format_version="1.1")
        """
        from .array import CVDArray

        vulns = []
        for item in nvd_items:
            # Extract CVE ID
            cve_id = cls.extract_cve_id(item, format_version)
            if not cve_id:
                continue

            # Create vulnerability
            vuln = cls.create_vuln_from_item(cve_id, item, format_version)
            vulns.append(vuln)

        return CVDArray(vulns)
