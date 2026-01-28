"""Score and enrichment extraction transforms.

Provides:
- ScoreExtractor: Extract CVSS/EPSS/enrichment data into computed slots
- parse_cpe: Parse CPE 2.3 strings into components

Layer: Analytics
Dependencies: models.py
Used by: array.py
"""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from numpy.typing import NDArray

from vulnstate.models import ScoreResult

if TYPE_CHECKING:
    from vulnstate.array import CVDArray
    from vulnstate.vulnerability import CVDVulnerability


def parse_cpe(cpe: Optional[str]) -> dict[str, str]:
    """Parse CPE 2.3 string into components.

    CPE 2.3 format: cpe:2.3:part:vendor:product:version:update:edition:...

    Args:
        cpe: CPE 2.3 string (e.g., "cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*")
             or None

    Returns:
        Dict with type, vendor, product, version keys.
        Empty strings for missing/invalid components or None input.
        Wildcards (*) and placeholders (-) are treated as empty.

    Example:
        >>> parse_cpe("cpe:2.3:a:apache:log4j:2.14.1:*:*:*:*:*:*:*")
        {'type': 'a', 'vendor': 'apache', 'product': 'log4j', 'version': '2.14.1'}
        >>> parse_cpe(None)
        {'type': '', 'vendor': '', 'product': '', 'version': ''}
    """
    empty_result = {"type": "", "vendor": "", "product": "", "version": ""}

    if not cpe or not isinstance(cpe, str):
        return empty_result

    parts = cpe.split(":")

    # Validate CPE 2.3 format
    if len(parts) < 5 or parts[0] != "cpe" or parts[1] != "2.3":
        return empty_result

    def clean_value(val: str) -> str:
        """Filter out wildcards and placeholders."""
        return "" if val in ("*", "-", "") else val

    return {
        "type": clean_value(parts[2]) if len(parts) > 2 else "",
        "vendor": clean_value(parts[3]) if len(parts) > 3 else "",
        "product": clean_value(parts[4]) if len(parts) > 4 else "",
        "version": clean_value(parts[5]) if len(parts) > 5 else "",
    }


class ScoreExtractor:
    """Extract score and enrichment data into computed slots.

    Implements the Transform protocol for extracting CVSS, EPSS, KEV,
    CWE, CPE, and exploit data from vulnerabilities.

    Attributes:
        name: Transform identifier ("scores")

    Example:
        >>> extractor = ScoreExtractor()
        >>> result = extractor.apply_single(vuln)
        >>> print(f"CVSS: {result.cvss_score}, KEV: {result.kev}")
    """

    name = "scores"

    def _best_cvss(self, cvss_scores: list[Any]) -> Optional[float]:
        """Get best available CVSS score (prefer 3.1 > 4.0 > 3.0 > 2.0).

        Args:
            cvss_scores: List of CVSSScore objects

        Returns:
            Best CVSS base score, or None if no scores available
        """
        for version in [3.1, 4.0, 3.0, 2.0]:
            for score in cvss_scores:
                if score.version == version:
                    return float(score.base_score)
        return None

    def _max_cvss(self, cvss_scores: list[Any]) -> Optional[float]:
        """Get maximum CVSS score across all versions.

        Args:
            cvss_scores: List of CVSSScore objects

        Returns:
            Maximum base score, or None if no scores available
        """
        if not cvss_scores:
            return None
        return float(max(s.base_score for s in cvss_scores))

    def _best_epss(self, epss_scores: list[Any]) -> tuple[Optional[float], Optional[float]]:
        """Get best EPSS (prefer v4 > v3).

        Args:
            epss_scores: List of EPSSScore objects

        Returns:
            Tuple of (probability, percentile), both None if no scores
        """
        for model in [4, 3]:
            for score in epss_scores:
                if score.model == model:
                    return score.probability, score.percentile
        return None, None

    def _extract_vendors_products(self, cpes: list[str]) -> tuple[set[str], set[str]]:
        """Extract unique vendors and products from CPE strings.

        Args:
            cpes: List of CPE 2.3 strings

        Returns:
            Tuple of (vendors set, products set)
        """
        vendors: set[str] = set()
        products: set[str] = set()
        for cpe in cpes:
            parsed = parse_cpe(cpe)
            if parsed["vendor"]:
                vendors.add(parsed["vendor"])
            if parsed["product"]:
                products.add(parsed["product"])
        return vendors, products

    def apply_single(self, vuln: "CVDVulnerability") -> ScoreResult:
        """Compute scores for single vulnerability.

        Args:
            vuln: CVDVulnerability with cvss_scores, epss_scores, cwes,
                  cpes, kev, and exploits attributes

        Returns:
            ScoreResult with computed scores and enrichment data
        """
        # Access enrichment fields (added in Task 3.1)
        cvss_scores: list[Any] = getattr(vuln, "cvss_scores", [])
        epss_scores: list[Any] = getattr(vuln, "epss_scores", [])
        cwes: list[Any] = getattr(vuln, "cwes", [])
        cpes: list[str] = getattr(vuln, "cpes", [])
        kev_entry: Any = getattr(vuln, "kev_entry", None)
        exploits: list[Any] = getattr(vuln, "exploits", [])

        prob, pctl = self._best_epss(epss_scores)
        vendors, products = self._extract_vendors_products(cpes)

        return ScoreResult(
            cvss_score=self._best_cvss(cvss_scores),
            cvss_max=self._max_cvss(cvss_scores),
            epss_probability=prob,
            epss_percentile=pctl,
            kev=kev_entry is not None,
            has_exploit=len(exploits) > 0,
            cwe_count=len(cwes),
            cpe_count=len(cpes),
            vendors=vendors,
            products=products,
        )

    def apply(self, array: "CVDArray") -> dict[str, NDArray[Any]]:
        """Compute score slots for entire array (vectorized).

        Args:
            array: CVDArray with _source containing enrichment data

        Returns:
            Dict mapping slot names to numpy arrays of length N
        """
        n = len(array)

        cvss_score: NDArray[np.float32] = np.full(n, np.nan, dtype=np.float32)
        cvss_max: NDArray[np.float32] = np.full(n, np.nan, dtype=np.float32)
        epss_probability: NDArray[np.float32] = np.full(n, np.nan, dtype=np.float32)
        epss_percentile: NDArray[np.float32] = np.full(n, np.nan, dtype=np.float32)
        kev: NDArray[np.bool_] = np.zeros(n, dtype=bool)
        has_exploit: NDArray[np.bool_] = np.zeros(n, dtype=bool)
        cwe_count: NDArray[np.uint8] = np.zeros(n, dtype=np.uint8)
        cpe_count: NDArray[np.uint16] = np.zeros(n, dtype=np.uint16)
        vendors: NDArray[np.object_] = np.empty(n, dtype=object)
        products: NDArray[np.object_] = np.empty(n, dtype=object)

        # Access _source (added in Task 3.2)
        source: Any = getattr(array, "_source", None)
        if source is None:
            # Return empty arrays if no source data
            return {
                "cvss_score": cvss_score,
                "cvss_max": cvss_max,
                "epss_probability": epss_probability,
                "epss_percentile": epss_percentile,
                "kev": kev,
                "has_exploit": has_exploit,
                "cwe_count": cwe_count,
                "cpe_count": cpe_count,
                "vendors": vendors,
                "products": products,
            }

        for i in range(n):
            # Defensive access - handle None or missing entries gracefully
            cvss_list = source.cvss_scores[i] if source.cvss_scores[i] else []
            epss_list = source.epss_scores[i] if source.epss_scores[i] else []
            cpe_list = source.cpes[i] if source.cpes[i] else []
            cwe_list = source.cwes[i] if source.cwes[i] else []
            exploit_list = source.exploits[i] if source.exploits[i] else []
            kev_entry = source.kev[i]

            if cvss_list:
                best = self._best_cvss(cvss_list)
                if best is not None:
                    cvss_score[i] = best
                max_score = self._max_cvss(cvss_list)
                if max_score is not None:
                    cvss_max[i] = max_score

            prob, pctl = self._best_epss(epss_list)
            if prob is not None:
                epss_probability[i] = prob
            if pctl is not None:
                epss_percentile[i] = pctl

            kev[i] = kev_entry is not None
            has_exploit[i] = len(exploit_list) > 0
            cwe_count[i] = len(cwe_list)
            cpe_count[i] = len(cpe_list)

            v, p = self._extract_vendors_products(cpe_list)
            vendors[i] = v
            products[i] = p

        return {
            "cvss_score": cvss_score,
            "cvss_max": cvss_max,
            "epss_probability": epss_probability,
            "epss_percentile": epss_percentile,
            "kev": kev,
            "has_exploit": has_exploit,
            "cwe_count": cwe_count,
            "cpe_count": cpe_count,
            "vendors": vendors,
            "products": products,
        }
