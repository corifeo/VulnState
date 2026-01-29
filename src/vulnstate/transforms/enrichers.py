"""Enrichment transforms (fetch external data).

These transforms fetch data from external sources (CSV files, APIs)
and populate vulnerability data. They follow the Transform protocol
and delegate to CVDIO methods for actual implementation.

Provides:
- infer_events: Infer V/F/D events from metadata tags and heuristics
- EventInferenceTransform: Class wrapper for infer_events following Transform protocol
- EPSSEnricher: Load EPSS scores (delegates to CVDIO.import_epss)
- KEVEnricher: Load KEV entries (delegates to CVDIO.import_kev)

Layer: I/O
Dependencies: models.py, array.py, io.py
Used by: array.py, User code
"""

import contextlib
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from vulnstate.array import CVDArray

from vulnstate.constants import CVDEvent


def infer_events(
    arr: "CVDArray",
    vendor_lead: int = 0,
    thirdparty_lag: int = 7,
    deploy: bool = True,
    deploy_lag: int = 30,
    severity_adjusted: bool = False,
    heuristics: bool = True,
) -> dict[str, int]:
    """Infer V/F/D events from metadata tags and heuristics.

    Tier 1 (always): Refines V timestamps from advisory tags using offsets,
    infers V from Patch tag, applies D from F + lag.
    Tier 2 (heuristics=True): CPE/CVSS-based F inference, age-based V.

    All inferred events are added via apply_event(). Never overwrites
    events that have already occurred.

    Args:
        arr: CVDArray to enrich
        vendor_lead: Days V precedes P for Vendor Advisory tag (default 0)
        thirdparty_lag: Days V follows P for Third Party Advisory tag (default 7)
        deploy: Whether to infer D from F + lag (default True)
        deploy_lag: Flat days between F and D (default 30)
        severity_adjusted: Use severity-based D lag instead of flat (default False)
        heuristics: Enable CPE/CVSS/age-based guesses (default True)

    Returns:
        Summary dict: {"V_inferred": N, "F_inferred": N, "D_inferred": N}
    """
    summary: dict[str, int] = {"V_inferred": 0, "F_inferred": 0, "D_inferred": 0}

    if arr._vulnerabilities is None or len(arr._vulnerabilities) == 0:
        return summary

    # Severity-based deploy lag mapping
    severity_lag_map = {
        "CRITICAL": 7,
        "HIGH": 14,
        "MEDIUM": 30,
        "LOW": 60,
    }

    for i in range(len(arr)):
        vuln = arr.get(i)
        metadata = vuln.metadata

        # Get reference tags from stored metadata
        ref_tags = set(metadata.get("ref_tags", []))

        p_ts = vuln.events.get(CVDEvent.P)

        # --- V inference ---
        # Only infer if V not set
        if not vuln.has_event_occurred(CVDEvent.V):
            v_ts: Optional[datetime] = None

            if "Vendor Advisory" in ref_tags and p_ts is not None:
                v_ts = p_ts - timedelta(days=vendor_lead)
            elif "Third Party Advisory" in ref_tags and p_ts is not None:
                v_ts = p_ts + timedelta(days=thirdparty_lag)
            elif "Patch" in ref_tags and p_ts is not None:
                # Patch implies vendor awareness (at or before P)
                v_ts = p_ts - timedelta(days=vendor_lead)
            elif heuristics and p_ts is not None and metadata.get("has_version_end_excluding"):
                # Heuristic: CPE boundary implies vendor awareness
                v_ts = p_ts

            if v_ts is not None:
                if vuln.has_event_occurred(CVDEvent.V):
                    # Update timestamp on already-inferred V
                    vuln.events[CVDEvent.V] = v_ts
                else:
                    vuln.apply_event(CVDEvent.V, timestamp=v_ts)
                summary["V_inferred"] += 1

        # --- F inference ---
        if not vuln.has_event_occurred(CVDEvent.F):
            f_ts: Optional[datetime] = None

            # Patch tag implies fix exists (use last_modified as proxy)
            if "Patch" in ref_tags:
                last_mod_str = metadata.get("last_modified")
                if last_mod_str:
                    with contextlib.suppress(ValueError, TypeError):
                        f_ts = datetime.fromisoformat(last_mod_str.replace("Z", "+00:00"))

            # CPE versionEndExcluding implies fix version exists (heuristic)
            if f_ts is None and heuristics and metadata.get("has_version_end_excluding"):
                last_mod_str = metadata.get("last_modified")
                if last_mod_str:
                    with contextlib.suppress(ValueError, TypeError):
                        f_ts = datetime.fromisoformat(last_mod_str.replace("Z", "+00:00"))

            if f_ts is not None:
                # F requires V first - ensure V is set
                if not vuln.has_event_occurred(CVDEvent.V) and p_ts is not None:
                    vuln.apply_event(CVDEvent.V, timestamp=p_ts)

                with contextlib.suppress(ValueError):
                    vuln.apply_event(CVDEvent.F, timestamp=f_ts)
                    summary["F_inferred"] += 1

        # --- D inference ---
        if (
            deploy
            and vuln.has_event_occurred(CVDEvent.F)
            and not vuln.has_event_occurred(CVDEvent.D)
        ):
            f_ts_val = vuln.events.get(CVDEvent.F)
            if f_ts_val is not None and isinstance(f_ts_val, datetime):
                if severity_adjusted:
                    # Use CVSS score to determine severity tier
                    cvss = vuln.cvss_score
                    if cvss is not None:
                        if cvss >= 9.0:
                            lag = severity_lag_map["CRITICAL"]
                        elif cvss >= 7.0:
                            lag = severity_lag_map["HIGH"]
                        elif cvss >= 4.0:
                            lag = severity_lag_map["MEDIUM"]
                        else:
                            lag = severity_lag_map["LOW"]
                    else:
                        lag = deploy_lag  # Fallback to flat lag
                else:
                    lag = deploy_lag

                d_ts = f_ts_val + timedelta(days=lag)
                with contextlib.suppress(ValueError):
                    vuln.apply_event(CVDEvent.D, timestamp=d_ts)
                    summary["D_inferred"] += 1

    # Sync array state
    arr.sync()
    return summary


class EPSSEnricher:
    """Enrich array with EPSS scores via Transform protocol.

    Delegates to CVDIO.import_epss() for actual implementation.
    Two access patterns supported:
    - Imperative: arr.import_epss("file.csv")
    - Transform: arr.register_transform(EPSSEnricher("file.csv")); arr.transform()

    Attributes:
        name: Transform identifier ("epss_enricher")
        source: Path to EPSS CSV file or dict mapping CVE IDs to scores

    Example:
        >>> enricher = EPSSEnricher("epss_scores.csv")
        >>> enricher.apply(array)  # Returns {"epss_enriched": N}
    """

    name = "epss_enricher"

    def __init__(
        self,
        source: "str | dict[str, float | dict[str, Any]]",
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """Initialize EPSS enricher.

        Args:
            source: Path to EPSS CSV file or dict mapping CVE IDs to scores/metadata
            import_metadata: Store full EPSS data in vuln.metadata['epss'] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)
        """
        self.source = source
        self.import_metadata = import_metadata
        self.include = include
        self.exclude = exclude

    def apply(self, array: "CVDArray") -> dict[str, int]:
        """Populate EPSS scores from external source.

        Delegates to CVDIO.import_epss() for actual implementation.

        Args:
            array: CVDArray to enrich

        Returns:
            Summary dict: {"epss_enriched": N}
        """
        from vulnstate.io import CVDIO

        # Count initial EPSS scores
        initial_count = sum(
            1 for i in range(len(array)) if array.get(i).epss is not None
        )

        # Delegate to CVDIO
        CVDIO.import_epss(
            array,
            self.source,
            self.import_metadata,
            self.include,
            self.exclude,
        )

        # Count final EPSS scores
        final_count = sum(
            1 for i in range(len(array)) if array.get(i).epss is not None
        )

        return {"epss_enriched": final_count - initial_count}

    def apply_single(self, vuln: Any) -> None:
        """Single-item enrichment not supported for EPSS.

        EPSS data is loaded from files that contain many CVEs at once.
        Use apply() on an array instead.

        Args:
            vuln: CVDVulnerability to enrich

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "EPSSEnricher.apply_single() not supported. "
            "Use EPSSEnricher.apply() on a CVDArray instead."
        )


class EventInferenceTransform:
    """Transform that infers V/F/D events from metadata and heuristics.

    Migrated from infer_events() function to follow Transform protocol.
    """

    name = "event_inference"

    def __init__(
        self,
        vendor_lead: int = 0,
        thirdparty_lag: int = 7,
        deploy: bool = True,
        deploy_lag: int = 30,
        severity_adjusted: bool = False,
        heuristics: bool = True,
    ):
        self.vendor_lead = vendor_lead
        self.thirdparty_lag = thirdparty_lag
        self.deploy = deploy
        self.deploy_lag = deploy_lag
        self.severity_adjusted = severity_adjusted
        self.heuristics = heuristics

    def apply(self, array: "CVDArray") -> dict[str, int]:
        """Apply inference to array."""
        return infer_events(
            array,
            vendor_lead=self.vendor_lead,
            thirdparty_lag=self.thirdparty_lag,
            deploy=self.deploy,
            deploy_lag=self.deploy_lag,
            severity_adjusted=self.severity_adjusted,
            heuristics=self.heuristics,
        )

    def apply_single(self, vuln: Any) -> None:
        """Single-item inference not supported."""
        raise NotImplementedError(
            "EventInferenceTransform operates on arrays only"
        )


class KEVEnricher:
    """Enrich array with KEV data via Transform protocol.

    Delegates to CVDIO.import_kev() for actual implementation.
    Two access patterns supported:
    - Imperative: arr.import_kev("file.csv")
    - Transform: arr.register_transform(KEVEnricher("file.csv")); arr.transform()

    Attributes:
        name: Transform identifier ("kev_enricher")
        source: Path to KEV CSV file or dict mapping CVE IDs to KEV data

    Example:
        >>> enricher = KEVEnricher("known_exploited_vulnerabilities.csv")
        >>> enricher.apply(array)  # Returns {"kev_enriched": N}
    """

    name = "kev_enricher"

    def __init__(
        self,
        source: "str | dict[str, dict[str, Any]]",
        apply_event: bool = True,
        import_metadata: bool = False,
        include: Optional[list[str]] = None,
        exclude: Optional[list[str]] = None,
    ) -> None:
        """Initialize KEV enricher.

        Args:
            source: Path to KEV CSV file or dict mapping CVE IDs to KEV data
            apply_event: Apply event A with dateAdded timestamp (default True)
            import_metadata: Store full KEV data in vuln.metadata['kev'] (default False)
            include: Only store these fields (if import_metadata=True)
            exclude: Skip these fields (if import_metadata=True)
        """
        self.source = source
        self.apply_event = apply_event
        self.import_metadata = import_metadata
        self.include = include
        self.exclude = exclude

    def apply(self, array: "CVDArray") -> dict[str, int]:
        """Populate KEV entries from external source.

        Delegates to CVDIO.import_kev() for actual implementation.

        Args:
            array: CVDArray to enrich

        Returns:
            Summary dict: {"kev_enriched": N}
        """
        from vulnstate.io import CVDIO

        # Count initial KEV entries
        initial_count = sum(1 for i in range(len(array)) if array.get(i).kev)

        # Delegate to CVDIO
        CVDIO.import_kev(
            array,
            self.source,
            self.apply_event,
            self.import_metadata,
            self.include,
            self.exclude,
        )

        # Count final KEV entries
        final_count = sum(1 for i in range(len(array)) if array.get(i).kev)

        return {"kev_enriched": final_count - initial_count}

    def apply_single(self, vuln: Any) -> None:
        """Single-item enrichment not supported for KEV.

        KEV data is loaded from files that contain many CVEs at once.
        Use apply() on an array instead.

        Args:
            vuln: CVDVulnerability to enrich

        Raises:
            NotImplementedError: Always
        """
        raise NotImplementedError(
            "KEVEnricher.apply_single() not supported. "
            "Use KEVEnricher.apply() on a CVDArray instead."
        )
