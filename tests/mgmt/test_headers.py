"""Tests for FITS header editing and accessor helpers."""

import pytest
from astropy import units as u
from astropy.io import fits

import astroimred as air
from astroimred import headers, misc


class TestHeaderExports:
    """Tests for header helper exports."""

    def test_header_helpers_live_in_headers(self):
        """Header edit/accessor helpers live in headers."""
        assert air.hedit is headers.hedit
        assert air.key_remover is headers.key_remover
        assert air.hdrval is headers.hdrval
        assert not hasattr(headers, "valinhdr")
        assert not hasattr(headers, "get_from_header")
        assert not hasattr(headers, "get_if_none")


class TestHdrval:
    """Tests for hdrval."""

    def test_priority_units_and_source(self):
        """hdrval replaces the old header accessor helpers."""
        hdr = fits.Header()
        hdr["EXPTIME"] = 20

        assert headers.hdrval(None, hdr, "EXPTIME", default=0) == 20
        assert headers.hdrval(3, hdr, "EXPTIME", default=0) == 3
        assert headers.hdrval(None, hdr, "MISSING", default=0) == 0
        assert headers.hdrval(3 * u.s, hdr, "EXPTIME", unit=u.s) == 3 * u.s

        value, source = headers.hdrval(
            None, hdr, "EXPTIME", unit=u.s, return_source=True
        )
        assert value == 20 * u.s
        assert source == "EXPTIME in header"

        value, source = headers.hdrval(
            None, hdr, "MISSING", default=1, unit=u.s, return_source=True
        )
        assert value == 1 * u.s
        assert source == "default"


class TestCmt2Hdr:
    """Tests for cmt2hdr function (adding comments/history to header)."""

    def test_headers_owns_helper(self):
        """Moved header helpers are not re-exported from misc."""
        assert not hasattr(misc, "cmt2hdr")
        assert not hasattr(misc, "update_tlm")
        assert not hasattr(misc, "update_process")

    def test_add_history(self, sample_header):
        """Test adding HISTORY to header."""
        hdr = sample_header.copy()
        headers.cmt2hdr(hdr, "h", "Test history entry", time_fmt=None)
        # Check that HISTORY was added
        assert "HISTORY" in hdr
        assert "Test history entry" in str(hdr["HISTORY"])

    def test_add_comment(self, sample_header):
        """Test adding COMMENT to header."""
        hdr = sample_header.copy()
        headers.cmt2hdr(hdr, "c", "Test comment entry", time_fmt=None)
        # Check that COMMENT was added
        assert "COMMENT" in hdr
        assert "Test comment entry" in str(hdr["COMMENT"])

    @pytest.mark.parametrize("histcomm", ["h", "hist", "history", "HISTORY"])
    def test_history_aliases(self, sample_header, histcomm):
        """Test various aliases for HISTORY."""
        hdr = sample_header.copy()
        headers.cmt2hdr(hdr, histcomm, "Test", time_fmt=None)
        assert "HISTORY" in hdr

    @pytest.mark.parametrize("histcomm", ["c", "com", "comm", "comment", "COMMENT"])
    def test_comment_aliases(self, sample_header, histcomm):
        """Test various aliases for COMMENT."""
        hdr = sample_header.copy()
        headers.cmt2hdr(hdr, histcomm, "Test", time_fmt=None)
        assert "COMMENT" in hdr

    def test_invalid_histcomm_raises(self, sample_header):
        """Test that invalid histcomm raises ValueError."""
        hdr = sample_header.copy()
        with pytest.raises(ValueError):
            headers.cmt2hdr(hdr, "invalid", "Test", time_fmt=None)


class TestUpdateProcess:
    """Tests for update_process function."""

    def test_add_process(self, sample_header):
        """Test adding process key to header."""
        hdr = sample_header.copy()
        headers.update_process(hdr, process="B")
        assert "PROCESS" in hdr
        assert "B" in hdr["PROCESS"]

    def test_append_process(self, sample_header):
        """Test appending to existing process key."""
        hdr = sample_header.copy()
        hdr["PROCESS"] = "B"
        headers.update_process(hdr, process="D")
        # Should contain both B and D
        assert "B" in hdr["PROCESS"]
        assert "D" in hdr["PROCESS"]
