"""
Parsers - NVD Data Import and Processing

Provides:
- NVDParser: Parser for NVD (National Vulnerability Database) JSON feeds

Layer: I/O
Dependencies: constants.py, vulnerability.py, array.py
Used by: io.py
"""

import contextlib
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Union

# Use orjson for faster JSON parsing if available
try:
    import orjson

    def _load_json(filepath: str) -> Any:
        with open(filepath, "rb") as f:
            return orjson.loads(f.read())

except ImportError:
    import json

    def _load_json(filepath: str) -> Any:
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)


if TYPE_CHECKING:
    from .array import CVDArray
    from .vulnerability import CVDVulnerability

from .constants import CVDEvent
from .models import CVSSScore, CWEEntry
from .transforms import parse_cpe


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
    def extract_cvss(
        item: dict[str, Any], format_version: str
    ) -> tuple[Optional[float], Optional[str], Optional[float], Optional[float]]:
        """
        Extract CVSS score, vector, and sub-scores from NVD item.

        Args:
            item: NVD vulnerability item
            format_version: "1.1" or "2.0"

        Returns:
            Tuple of (base_score, vector_string, exploitability_score, impact_score)
        """
        if format_version == "2.0":
            # NVD 2.0: item['cve']['metrics']['cvssMetricV31'][0]
            metrics = item.get("cve", {}).get("metrics", {})

            # Try CVSS v3.1 first (cvssMetricV31)
            cvss_v31_list = metrics.get("cvssMetricV31", [])
            if cvss_v31_list:
                # Prefer primary source, fallback to first available
                for metric in cvss_v31_list:
                    if metric.get("type") == "Primary":
                        cvss_data = metric.get("cvssData", {})
                        return (
                            cvss_data.get("baseScore"),
                            cvss_data.get("vectorString"),
                            metric.get("exploitabilityScore"),
                            metric.get("impactScore"),
                        )
                # No primary, use first one
                metric = cvss_v31_list[0]
                cvss_data = metric.get("cvssData", {})
                return (
                    cvss_data.get("baseScore"),
                    cvss_data.get("vectorString"),
                    metric.get("exploitabilityScore"),
                    metric.get("impactScore"),
                )

            # Fallback to CVSS v3.0 (cvssMetricV30)
            cvss_v30_list = metrics.get("cvssMetricV30", [])
            if cvss_v30_list:
                metric = cvss_v30_list[0]
                cvss_data = metric.get("cvssData", {})
                return (
                    cvss_data.get("baseScore"),
                    cvss_data.get("vectorString"),
                    metric.get("exploitabilityScore"),
                    metric.get("impactScore"),
                )

            # Fallback to CVSS v2.0
            cvss_v2_list = metrics.get("cvssMetricV2", [])
            if cvss_v2_list:
                metric = cvss_v2_list[0]
                cvss_data = metric.get("cvssData", {})
                return (
                    cvss_data.get("baseScore"),
                    cvss_data.get("vectorString"),
                    metric.get("exploitabilityScore"),
                    metric.get("impactScore"),
                )

        else:
            # NVD 1.1: item['impact']['baseMetricV3']
            impact = item.get("impact", {})

            # Try CVSS v3
            if "baseMetricV3" in impact:
                base_metric = impact["baseMetricV3"]
                cvss_v3 = base_metric.get("cvssV3", {})
                return (
                    cvss_v3.get("baseScore"),
                    cvss_v3.get("vectorString"),
                    base_metric.get("exploitabilityScore"),
                    base_metric.get("impactScore"),
                )

            # Fallback to CVSS v2
            if "baseMetricV2" in impact:
                base_metric = impact["baseMetricV2"]
                cvss_v2 = base_metric.get("cvssV2", {})
                return (
                    cvss_v2.get("baseScore"),
                    cvss_v2.get("vectorString"),
                    base_metric.get("exploitabilityScore"),
                    base_metric.get("impactScore"),
                )

        # No CVSS data found
        return None, None, None, None

    @staticmethod
    def parse_cvss_vector(vector: Optional[str]) -> dict[str, Optional[str]]:
        """
        Parse CVSS 3.1 vector string to extract base metrics.

        Simple parser for CVSS 3.1 format: "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"

        Note: This is a minimal implementation for CVSS 3.1 base metrics only.
        For full CVSS 2.0/3.0/3.1/4.0 support with temporal and environmental metrics,
        consider using a dedicated library like 'cvsslib'.

        Args:
            vector: CVSS vector string or None

        Returns:
            Dict with keys: AV, AC, PR, UI, S, C, I, A
            Values are single-letter codes or None if not present/parseable
        """
        result = {
            "AV": None,
            "AC": None,
            "PR": None,
            "UI": None,
            "S": None,
            "C": None,
            "I": None,
            "A": None,
        }

        if not vector or not isinstance(vector, str):
            return result

        # Parse format: "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
        try:
            parts = vector.split("/")
            for part in parts[1:]:  # Skip "CVSS:3.1"
                if ":" in part:
                    key, value = part.split(":", 1)
                    if key in result:
                        result[key] = value
        except Exception:
            # Invalid format, return all None
            pass

        return result

    @staticmethod
    def detect_cve_status(item: dict[str, Any], format_version: str) -> str:
        """
        Detect CVE lifecycle status from NVD item.

        Checks vulnStatus field (NVD 2.0) and description markers (both formats)
        for rejection, dispute, or reservation indicators.

        Args:
            item: NVD vulnerability item
            format_version: "1.1" or "2.0"

        Returns:
            One of: "rejected", "disputed", "reserved", "active"
        """
        # NVD 2.0: check vulnStatus field
        if format_version == "2.0":
            vuln_status = item.get("cve", {}).get("vulnStatus", "")
            if vuln_status == "Rejected":
                return "rejected"

        # Check description markers (both formats)
        description = NVDParser.extract_description(item, format_version)
        if not description:
            return "reserved"

        if "** REJECT **" in description or description.startswith("Rejected reason:"):
            return "rejected"
        if "** DISPUTED **" in description:
            return "disputed"
        if "** RESERVED **" in description:
            return "reserved"

        return "active"

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
    def extract_description(item: dict[str, Any], format_version: str) -> Optional[str]:
        """Extract English description from NVD item."""
        if format_version == "2.0":
            descriptions = item.get("cve", {}).get("descriptions", [])
        else:
            descriptions = item.get("cve", {}).get("description", {}).get("description_data", [])

        for desc in descriptions:
            lang = desc.get("lang", "")
            if lang == "en":
                return desc.get("value")
        # Fallback to first description
        if descriptions:
            return descriptions[0].get("value")
        return None

    @staticmethod
    def extract_cwe_ids(item: dict[str, Any], format_version: str) -> list[str]:
        """Extract CWE IDs from NVD item."""
        cwe_ids = []

        if format_version == "2.0":
            weaknesses = item.get("cve", {}).get("weaknesses", [])
            for weakness in weaknesses:
                for desc in weakness.get("description", []):
                    cwe_id = desc.get("value")
                    if cwe_id and cwe_id.startswith("CWE-"):
                        cwe_ids.append(cwe_id)
        else:
            problem_types = item.get("cve", {}).get("problemtype", {}).get("problemtype_data", [])
            for pt in problem_types:
                for desc in pt.get("description", []):
                    cwe_id = desc.get("value")
                    if cwe_id and cwe_id.startswith("CWE-"):
                        cwe_ids.append(cwe_id)

        return cwe_ids

    @staticmethod
    def extract_all_cpes(item: dict[str, Any], format_version: str) -> list[str]:
        """
        Extract ALL CPE strings from NVD configurations.

        A CVE can affect multiple vendors/products, so we extract all CPEs.

        Args:
            item: NVD vulnerability item
            format_version: "1.1" or "2.0"

        Returns:
            List of CPE 2.3 strings (may be empty)
        """
        cpe_strings: list[str] = []

        if format_version == "2.0":
            configurations = item.get("cve", {}).get("configurations", [])
            for config in configurations:
                for node in config.get("nodes", []):
                    for match in node.get("cpeMatch", []):
                        criteria = match.get("criteria", "")
                        if criteria.startswith("cpe:2.3:"):
                            cpe_strings.append(criteria)
        else:
            configurations = item.get("configurations", {})
            for node in configurations.get("nodes", []):
                for match in node.get("cpe_match", []):
                    cpe23 = match.get("cpe23Uri", "")
                    if cpe23.startswith("cpe:2.3:"):
                        cpe_strings.append(cpe23)

        return cpe_strings

    @staticmethod
    def extract_vendors_products(cpe_strings: list[str]) -> tuple[list[str], list[str]]:
        """
        Extract unique vendors and products from CPE strings.

        Args:
            cpe_strings: List of CPE 2.3 strings

        Returns:
            Tuple of (vendors list, products list) - both unique, sorted
        """
        vendors: set[str] = set()
        products: set[str] = set()

        for cpe in cpe_strings:
            parsed = parse_cpe(cpe)
            if parsed["vendor"]:
                vendors.add(parsed["vendor"])
            if parsed["product"]:
                products.add(parsed["product"])

        return sorted(vendors), sorted(products)

    @staticmethod
    def extract_cvss_scores(item: dict[str, Any], format_version: str) -> list[CVSSScore]:
        """
        Extract all CVSS scores as CVSSScore objects from NVD item.

        Supports CVSS v4.0, v3.1, v3.0, and v2.0 metrics.

        Args:
            item: NVD vulnerability item
            format_version: "1.1" or "2.0"

        Returns:
            List of CVSSScore objects (may be empty)
        """
        cvss_scores: list[CVSSScore] = []

        if format_version == "2.0":
            metrics = item.get("cve", {}).get("metrics", {})

            # Process all CVSS metric versions
            metric_versions = [
                ("cvssMetricV40", 4.0),
                ("cvssMetricV31", 3.1),
                ("cvssMetricV30", 3.0),
                ("cvssMetricV2", 2.0),
            ]

            for metric_key, default_version in metric_versions:
                for metric in metrics.get(metric_key, []):
                    cvss_data = metric.get("cvssData", {})

                    # Parse version from data or use default
                    version_str = cvss_data.get("version", str(default_version))
                    try:
                        version = float(version_str)
                    except (ValueError, TypeError):
                        version = default_version

                    cvss_scores.append(
                        CVSSScore(
                            version=version,
                            base_score=cvss_data.get("baseScore", 0.0),
                            vector=cvss_data.get("vectorString", ""),
                            source=metric.get("source", "unknown"),
                            source_status=None,
                            reserved_at=None,
                            published_at=None,
                            updated_at=None,
                            temporal_score=None,
                            environmental_score=None,
                        )
                    )

        else:
            # NVD 1.1 format
            impact = item.get("impact", {})

            # CVSS v3
            if "baseMetricV3" in impact:
                base_metric = impact["baseMetricV3"]
                cvss_v3 = base_metric.get("cvssV3", {})
                cvss_scores.append(
                    CVSSScore(
                        version=3.0,
                        base_score=cvss_v3.get("baseScore", 0.0),
                        vector=cvss_v3.get("vectorString", ""),
                        source="nvd@nist.gov",
                        source_status=None,
                        reserved_at=None,
                        published_at=None,
                        updated_at=None,
                        temporal_score=None,
                        environmental_score=None,
                    )
                )

            # CVSS v2
            if "baseMetricV2" in impact:
                base_metric = impact["baseMetricV2"]
                cvss_v2 = base_metric.get("cvssV2", {})
                cvss_scores.append(
                    CVSSScore(
                        version=2.0,
                        base_score=cvss_v2.get("baseScore", 0.0),
                        vector=cvss_v2.get("vectorString", ""),
                        source="nvd@nist.gov",
                        source_status=None,
                        reserved_at=None,
                        published_at=None,
                        updated_at=None,
                        temporal_score=None,
                        environmental_score=None,
                    )
                )

        return cvss_scores

    @staticmethod
    def extract_cwe_entries(item: dict[str, Any], format_version: str) -> list[CWEEntry]:
        """
        Extract CWE entries as CWEEntry objects from NVD item.

        Skips NVD-CWE-Other and NVD-CWE-noinfo entries as they don't provide
        actionable weakness information.

        Args:
            item: NVD vulnerability item
            format_version: "1.1" or "2.0"

        Returns:
            List of CWEEntry objects (may be empty)
        """
        cwe_entries: list[CWEEntry] = []

        if format_version == "2.0":
            weaknesses = item.get("cve", {}).get("weaknesses", [])
            for weakness in weaknesses:
                source = weakness.get("source")
                is_primary = weakness.get("type") == "Primary"
                for desc in weakness.get("description", []):
                    if desc.get("lang") != "en":
                        continue
                    cwe_id = desc.get("value", "")
                    # Skip NVD placeholder CWEs
                    if cwe_id.startswith("CWE-") and not cwe_id.startswith("CWE-noinfo"):
                        cwe_entries.append(CWEEntry(id=cwe_id, source=source, primary=is_primary))
        else:
            # NVD 1.1 format
            problem_types = item.get("cve", {}).get("problemtype", {}).get("problemtype_data", [])
            for pt in problem_types:
                for desc in pt.get("description", []):
                    cwe_id = desc.get("value", "")
                    # Skip NVD placeholder CWEs
                    if cwe_id.startswith("CWE-") and not cwe_id.startswith("CWE-noinfo"):
                        cwe_entries.append(CWEEntry(id=cwe_id, source="nvd@nist.gov", primary=True))

        return cwe_entries

    @staticmethod
    def extract_severity(item: dict[str, Any], format_version: str) -> Optional[str]:
        """Extract severity level from NVD item."""
        if format_version == "2.0":
            metrics = item.get("cve", {}).get("metrics", {})
            for metric_key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
                metric_list = metrics.get(metric_key, [])
                if metric_list:
                    cvss_data = metric_list[0].get("cvssData", {})
                    severity = cvss_data.get("baseSeverity")
                    if severity:
                        return severity.upper()
        else:
            impact = item.get("impact", {})
            if "baseMetricV3" in impact:
                return impact["baseMetricV3"].get("cvssV3", {}).get("baseSeverity", "").upper()
            if "baseMetricV2" in impact:
                return impact["baseMetricV2"].get("severity", "").upper()
        return None

    @staticmethod
    def extract_last_modified(item: dict[str, Any], format_version: str) -> Optional[str]:
        """Extract last modified date from NVD item."""
        if format_version == "2.0":
            return item.get("cve", {}).get("lastModified")
        else:
            return item.get("lastModifiedDate")

    @staticmethod
    def extract_reference_tags(item: dict[str, Any], format_version: str) -> set[str]:
        """
        Extract all unique reference tags from NVD item.

        Scans all references and collects their tags into a flat set.
        Common tags: "Patch", "Exploit", "Vendor Advisory", "Third Party Advisory",
        "Mitigation", "US Government Resource".

        Args:
            item: NVD vulnerability item
            format_version: "1.1" or "2.0"

        Returns:
            Set of tag strings (may be empty)
        """
        tags: set[str] = set()

        if format_version == "2.0":
            references = item.get("cve", {}).get("references", [])
            for ref in references:
                tags.update(ref.get("tags", []))
        else:
            ref_data = item.get("cve", {}).get("references", {}).get("reference_data", [])
            for ref in ref_data:
                tags.update(ref.get("tags", []))

        return tags

    @staticmethod
    def create_vuln_from_item(
        cve_id: str,
        item: dict[str, Any],
        format_version: str = "1.1",
        infer_vendor: bool = True,
        infer_timestamps: bool = True,
    ) -> "CVDVulnerability":
        """
        Create a CVDVulnerability from an NVD item.

        Extracts all parseable fields from the NVD item:
        - cvss_score, cve_vector: CVSS data (stored in scoring)
        - cpe_strings, vendors, products: From CPE (stored in metadata as lists)
        - description, cwe_ids, severity, published_date, last_modified: Parsed metadata

        Infers events from reference tags:
        - "Patch" tag -> F event (fix available)
        - "Exploit" tag -> X event (exploit public)
        - "Vendor Advisory" tag -> V event at publishedDate
        - "Third Party Advisory" tag -> V event at lastModifiedDate

        Note: A CVE can affect multiple vendors/products, so these are stored as
        lists in metadata rather than single values.

        Args:
            cve_id: CVE identifier
            item: NVD vulnerability item
            format_version: "1.1" or "2.0"
            infer_vendor: If True, infer V event from publishedDate (default True)
            infer_timestamps: If True, use proxy timestamps for inferred events;
                if False, set timestamp to None (default True)

        Returns:
            CVDVulnerability instance with all extracted data
        """
        from .vulnerability import CVDVulnerability

        vuln = CVDVulnerability(cve_id)

        # Extract CVSS sub-scores for metadata (exploitability/impact)
        _, _, exploitability, impact = NVDParser.extract_cvss(item, format_version)
        if exploitability is not None:
            vuln.metadata["cvss_exploitability_score"] = exploitability
        if impact is not None:
            vuln.metadata["cvss_impact_score"] = impact

        # Extract API v2 enrichment fields
        # CVSS scores (list of CVSSScore objects)
        vuln.cvss_scores = NVDParser.extract_cvss_scores(item, format_version)

        # CWE entries (list of CWEEntry objects)
        vuln.cwes = NVDParser.extract_cwe_entries(item, format_version)

        # CPE entries (parsed from CPE 2.3 URIs)
        cpe_strings = NVDParser.extract_all_cpes(item, format_version)
        from .models import CPE

        vuln.cpes = [CPE.parse(cpe_str) for cpe_str in cpe_strings]

        # Also store raw strings in metadata for backward compatibility
        if cpe_strings:
            vuln.metadata["cpe_strings"] = cpe_strings

        # Detect and store CVE status
        cve_status = NVDParser.detect_cve_status(item, format_version)
        vuln.metadata["cve_status"] = cve_status

        # Store raw vulnStatus from NVD 2.0 (for analysis tracking)
        if format_version == "2.0":
            raw_vuln_status = item.get("cve", {}).get("vulnStatus")
            if raw_vuln_status:
                vuln.metadata["nvd_status"] = raw_vuln_status

        # Extract description
        description = NVDParser.extract_description(item, format_version)
        if description:
            vuln.metadata["description"] = description

        # Extract CWE IDs
        cwe_ids = NVDParser.extract_cwe_ids(item, format_version)
        if cwe_ids:
            vuln.metadata["cwe_ids"] = cwe_ids

        # Extract severity
        severity = NVDParser.extract_severity(item, format_version)
        if severity:
            vuln.metadata["severity"] = severity

        # Extract published date and apply P event
        published_date = NVDParser.extract_published_date(item, format_version)
        if published_date:
            vuln.metadata["published_date"] = published_date
            try:
                dt = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
                vuln.apply_event(CVDEvent.P, timestamp=dt)
            except (ValueError, TypeError):
                pass

        # Extract last modified date
        last_modified = NVDParser.extract_last_modified(item, format_version)
        if last_modified:
            vuln.metadata["last_modified"] = last_modified

        # Parse proxy timestamps for inferred events (reuse already-extracted strings)
        published_dt: Optional[datetime] = None
        last_modified_dt: Optional[datetime] = None
        try:
            if published_date:
                published_dt = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
            if last_modified:
                last_modified_dt = datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

        # Apply V from publishedDate if infer_vendor=True (default behavior)
        if infer_vendor and published_dt:
            v_ts = published_dt if infer_timestamps else None
            with contextlib.suppress(ValueError):
                vuln.apply_event(CVDEvent.V, timestamp=v_ts)

        # Detect events from reference tags
        ref_tags = NVDParser.extract_reference_tags(item, format_version)

        # Store tags for later use by infer_events()
        if ref_tags:
            vuln.metadata["ref_tags"] = list(ref_tags)

        # V from advisory tags (more specific signal than V-from-publishedDate)
        if "Vendor Advisory" in ref_tags:
            v_ts = published_dt if infer_timestamps else None
            with contextlib.suppress(ValueError):
                if vuln.has_event_occurred(CVDEvent.V):
                    vuln.events[CVDEvent.V] = v_ts
                else:
                    vuln.apply_event(CVDEvent.V, timestamp=v_ts)
        elif "Third Party Advisory" in ref_tags:
            v_ts = last_modified_dt if infer_timestamps else None
            with contextlib.suppress(ValueError):
                if vuln.has_event_occurred(CVDEvent.V):
                    vuln.events[CVDEvent.V] = v_ts
                else:
                    vuln.apply_event(CVDEvent.V, timestamp=v_ts)

        # F from Patch tag (100% reliable - patch exists)
        if "Patch" in ref_tags:
            f_ts = last_modified_dt if infer_timestamps else None
            with contextlib.suppress(ValueError):
                vuln.apply_event(CVDEvent.F, timestamp=f_ts)

        # X from Exploit tag
        if "Exploit" in ref_tags:
            x_ts = last_modified_dt if infer_timestamps else None
            with contextlib.suppress(ValueError):
                vuln.apply_event(CVDEvent.X, timestamp=x_ts)

        # Check for versionEndExcluding in configurations (fix version boundary)
        has_version_end = False
        configurations = item.get("configurations", {})
        if isinstance(configurations, dict):
            for node in configurations.get("nodes", []):
                for match in node.get("cpe_match", []):
                    if "versionEndExcluding" in match:
                        has_version_end = True
                        break
                if has_version_end:
                    break
                # Also check children nodes
                for child in node.get("children", []):
                    for match in child.get("cpe_match", []):
                        if "versionEndExcluding" in match:
                            has_version_end = True
                            break
                    if has_version_end:
                        break
        elif isinstance(configurations, list):
            # NVD 2.0 format
            for config in configurations:
                for node in config.get("nodes", []):
                    for match in node.get("cpeMatch", []):
                        if "versionEndExcluding" in match:
                            has_version_end = True
                            break
                    if has_version_end:
                        break
        if has_version_end:
            vuln.metadata["has_version_end_excluding"] = True

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
        infer_vendor: bool = True,
        infer_timestamps: bool = True,
        include_rejected: bool = False,
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
            infer_vendor: Infer V event from publishedDate (default True)
            infer_timestamps: Use proxy timestamps for inferred events (default True)
            include_rejected: Include rejected CVEs in import (default False)

        Note:
            Automatically detects NVD format version (1.1 or 2.0) and parses accordingly.
        """
        from .io import CVDIO

        # Parse source and detect format
        if isinstance(source, str):
            # Resolve URL or local path
            local_path = CVDIO._resolve_source(source)
            nvd_feed = _load_json(local_path)

            # Handle case where file contains a plain JSON array (e.g., nvd.handsonhacking.org)
            if isinstance(nvd_feed, list):
                nvd_items = nvd_feed
                # Detect format from first item
                if nvd_items:
                    first_item = nvd_items[0]
                    if "cve" in first_item and "id" in first_item.get("cve", {}):
                        format_version = "2.0"
                    else:
                        format_version = "1.1"
                else:
                    format_version = "1.1"
            else:
                # Standard NVD feed with wrapper object
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
        if hasattr(arr, "_fixed_size") and arr._fixed_size:
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

                # Skip rejected CVEs unless explicitly included
                if not include_rejected:
                    cve_status = NVDParser.detect_cve_status(item, format_version)
                    if cve_status == "rejected":
                        continue

                vuln = NVDParser.create_vuln_from_item(
                    cve_id,
                    item,
                    format_version,
                    infer_vendor=infer_vendor,
                    infer_timestamps=infer_timestamps,
                )

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

            # Skip rejected CVEs unless explicitly included
            if not include_rejected:
                cve_status = NVDParser.detect_cve_status(item, format_version)
                if cve_status == "rejected":
                    continue

            if cve_id in existing_cves:
                if skip_existing:
                    continue

                # Update existing vulnerability
                idx = cve_to_index.get(cve_id)
                if idx is not None:
                    vuln = arr.get(idx)

                    # Update CVSS score, vector, and sub-scores
                    cvss_score, cvss_vector, exploitability, impact = NVDParser.extract_cvss(
                        item, format_version
                    )
                    if cvss_score is not None:
                        # Use list-based approach for cvss_score
                        from .models import CVSSScore

                        if vuln.cvss_scores:
                            # Replace first score
                            vuln.cvss_scores[0] = CVSSScore(
                                version=3.1,
                                base_score=cvss_score,
                                vector=cvss_vector or "",
                                source="nvd",
                                source_status=None,
                                reserved_at=None,
                                published_at=None,
                                updated_at=None,
                                temporal_score=None,
                                environmental_score=None,
                            )
                        else:
                            vuln.cvss_scores.append(
                                CVSSScore(
                                    version=3.1,
                                    base_score=cvss_score,
                                    vector=cvss_vector or "",
                                    source="nvd",
                                    source_status=None,
                                    reserved_at=None,
                                    published_at=None,
                                    updated_at=None,
                                    temporal_score=None,
                                    environmental_score=None,
                                )
                            )
                        # Store sub-scores in metadata
                        if exploitability is not None:
                            vuln.metadata["cvss_exploitability_score"] = exploitability
                        if impact is not None:
                            vuln.metadata["cvss_impact_score"] = impact
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
                            metadata_dict = {
                                k: v for k, v in metadata_dict.items() if k not in exclude
                            }

                        if "nvd" not in vuln.metadata:
                            vuln.metadata["nvd"] = {}
                        vuln.metadata["nvd"].update(metadata_dict)
            else:
                # Create new vulnerability
                vuln = NVDParser.create_vuln_from_item(
                    cve_id,
                    item,
                    format_version,
                    infer_vendor=infer_vendor,
                    infer_timestamps=infer_timestamps,
                )

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


def extract_cpe_from_configurations(configurations: list[dict[str, Any]]) -> Optional[str]:
    """
    Extract first CPE string from NVD configurations.

    NVD 2.0 format nests CPE strings in configurations[].nodes[].cpeMatch[].criteria

    Args:
        configurations: NVD configurations array

    Returns:
        First CPE string found, or None
    """
    if not configurations:
        return None

    for config in configurations:
        nodes = config.get("nodes", [])
        for node in nodes:
            cpe_matches = node.get("cpeMatch", [])
            for match in cpe_matches:
                criteria = match.get("criteria")
                if criteria and criteria.startswith("cpe:2.3:"):
                    return criteria

    return None
