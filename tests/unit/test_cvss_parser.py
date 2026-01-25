"""Tests for CVSS vector parsing."""

import pytest

from vulnstate.parsers import NVDParser

pytestmark = pytest.mark.unit


def test_parse_cvss_vector_valid():
    """Test parsing valid CVSS 3.1 vector."""
    vector = "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H"
    result = NVDParser.parse_cvss_vector(vector)

    assert result["AV"] == "N"
    assert result["AC"] == "L"
    assert result["PR"] == "N"
    assert result["UI"] == "N"
    assert result["S"] == "U"
    assert result["C"] == "H"
    assert result["I"] == "H"
    assert result["A"] == "H"


def test_parse_cvss_vector_partial():
    """Test parsing partial CVSS vector."""
    vector = "CVSS:3.1/AV:N/AC:L"
    result = NVDParser.parse_cvss_vector(vector)

    assert result["AV"] == "N"
    assert result["AC"] == "L"
    assert result["PR"] is None
    assert result["UI"] is None


def test_parse_cvss_vector_none():
    """Test parsing None vector."""
    result = NVDParser.parse_cvss_vector(None)

    assert all(v is None for v in result.values())


def test_parse_cvss_vector_invalid():
    """Test parsing invalid vector."""
    result = NVDParser.parse_cvss_vector("invalid")

    assert all(v is None for v in result.values())
