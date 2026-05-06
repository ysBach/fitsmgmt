"""Tests for WCS helper utilities."""

import fitsmgmt as fm
from fitsmgmt import wcstools


class TestWcsTools:
    """Tests for WCS helper exports."""

    def test_wcstools_exported_from_package_root(self):
        """WCS helpers live in wcstools and package root."""
        assert fm.wcsremove is wcstools.wcsremove

    def test_wcsremove_header(self, sample_header):
        """Test WCS keyword removal from an in-memory header."""
        hdr = sample_header.copy()
        hdr["CRVAL1"] = 1.0
        hdr["CRVAL2"] = 2.0
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"

        out = wcstools.wcsremove(hdr, verbose=False)

        assert "CRVAL1" not in out
        assert "CRVAL2" not in out
        assert "CTYPE1" not in out
        assert "CTYPE2" not in out
        assert "OBJECT" in out
