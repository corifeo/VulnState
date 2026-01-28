# tests/test_array_transforms.py
"""Tests for CVDArray transform infrastructure.

Tests for Task 4.1: CVDArray Transform Infrastructure
- _source: ArraySource holding object arrays
- _cache: dict[str, np.ndarray] holding computed slots
- _transforms: list[Transform] holding registered transforms
"""

import numpy as np


class TestCVDArrayTransformInfrastructure:
    """Tests for transform infrastructure on CVDArray."""

    def test_array_has_source(self):
        """Test that CVDArray has _source attribute of type ArraySource."""
        from vulnstate.array import CVDArray
        from vulnstate.models import ArraySource

        arr = CVDArray.generate(5)
        assert hasattr(arr, "_source")
        assert isinstance(arr._source, ArraySource)

    def test_array_has_cache(self):
        """Test that CVDArray has _cache attribute as dict."""
        from vulnstate.array import CVDArray

        arr = CVDArray.generate(5)
        assert hasattr(arr, "_cache")
        assert isinstance(arr._cache, dict)

    def test_array_has_transforms(self):
        """Test that CVDArray has _transforms attribute as list."""
        from vulnstate.array import CVDArray

        arr = CVDArray.generate(5)
        assert hasattr(arr, "_transforms")
        assert isinstance(arr._transforms, list)

    def test_register_transform(self):
        """Test registering a transform adds it to _transforms list."""
        from vulnstate.array import CVDArray
        from vulnstate.transforms import ScoreExtractor

        arr = CVDArray.generate(5)
        extractor = ScoreExtractor()
        arr.register_transform(extractor)
        assert extractor in arr._transforms

    def test_run_transforms_populates_cache(self):
        """Test running transforms populates the _cache dict."""
        from vulnstate.array import CVDArray
        from vulnstate.transforms import ScoreExtractor

        arr = CVDArray.generate(5)
        extractor = ScoreExtractor()
        arr.register_transform(extractor)
        arr.run_transforms()

        assert "cvss_score" in arr._cache
        assert len(arr._cache["cvss_score"]) == 5

    def test_slicing_preserves_source_and_cache(self):
        """Test slicing an array preserves _source and _cache with correct lengths."""
        from vulnstate.array import CVDArray
        from vulnstate.transforms import ScoreExtractor

        arr = CVDArray.generate(10)
        arr.register_transform(ScoreExtractor())
        arr.run_transforms()

        sliced = arr[0:5]
        assert len(sliced._source.cvss_scores) == 5
        assert "cvss_score" in sliced._cache
        assert len(sliced._cache["cvss_score"]) == 5

    def test_source_arrays_initialized_to_correct_size(self):
        """Test that _source arrays are initialized to match array size."""
        from vulnstate.array import CVDArray

        arr = CVDArray.generate(7)
        assert len(arr._source.cvss_scores) == 7
        assert len(arr._source.epss_scores) == 7
        assert len(arr._source.cwes) == 7
        assert len(arr._source.cpes) == 7
        assert len(arr._source.kev) == 7
        assert len(arr._source.exploits) == 7

    def test_source_arrays_contain_empty_lists(self):
        """Test that _source arrays are initialized with empty lists/None.

        Note: generate() creates vulns with cvss_score, so cvss_scores won't be empty.
        For truly empty source arrays, use zeros() which has no enrichment data.
        """
        from vulnstate.array import CVDArray

        # Use zeros() for empty enrichment data
        arr = CVDArray.zeros(3)
        # Each slot should contain empty list or None
        for i in range(3):
            assert arr._source.cvss_scores[i] == []
            assert arr._source.epss_scores[i] == []
            assert arr._source.cwes[i] == []
            assert arr._source.cpes[i] == []
            assert arr._source.kev[i] is None
            assert arr._source.exploits[i] == []

    def test_cache_starts_empty(self):
        """Test that _cache starts as empty dict."""
        from vulnstate.array import CVDArray

        arr = CVDArray.generate(5)
        assert arr._cache == {}

    def test_transforms_starts_empty(self):
        """Test that _transforms starts as empty list."""
        from vulnstate.array import CVDArray

        arr = CVDArray.generate(5)
        assert arr._transforms == []

    def test_multiple_transforms_run_in_order(self):
        """Test multiple transforms run and populate cache."""
        from vulnstate.array import CVDArray
        from vulnstate.transforms import ScoreExtractor

        arr = CVDArray.generate(5)
        # Register same transform twice (both should run)
        arr.register_transform(ScoreExtractor())
        arr.register_transform(ScoreExtractor())
        arr.run_transforms()

        # Cache should have slots from both runs (second overwrites first)
        assert "cvss_score" in arr._cache
        assert len(arr._transforms) == 2

    def test_slicing_copies_transforms_list(self):
        """Test slicing copies _transforms list (not shared reference)."""
        from vulnstate.array import CVDArray
        from vulnstate.transforms import ScoreExtractor

        arr = CVDArray.generate(10)
        extractor = ScoreExtractor()
        arr.register_transform(extractor)

        sliced = arr[0:5]
        assert extractor in sliced._transforms
        # Modifying sliced._transforms should not affect original
        sliced._transforms.clear()
        assert extractor in arr._transforms

    def test_boolean_mask_slicing_preserves_source(self):
        """Test boolean mask slicing preserves _source correctly."""
        from vulnstate.array import CVDArray

        arr = CVDArray.generate(10)
        mask = np.array([True, False, True, False, True, False, True, False, True, False])
        sliced = arr[mask]

        assert len(sliced._source.cvss_scores) == 5

    def test_empty_array_has_empty_source(self):
        """Test empty array has empty _source arrays."""
        from vulnstate.array import CVDArray

        arr = CVDArray()
        assert len(arr._source.cvss_scores) == 0
        assert len(arr._source.epss_scores) == 0
