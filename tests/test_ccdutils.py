"""Tests for CCDData manipulation helpers."""

import numpy as np
from astropy.nddata import CCDData

import fitsmgmt as fm
from fitsmgmt import ccdutils


def _header_value(header, key):
    value = header[key]
    if isinstance(value, tuple):
        return value[0]
    return value


class TestCcdUtils:
    """Tests for CCD helper module exports."""

    def test_ccd_helpers_have_canonical_modules(self):
        """CCD operations are exposed from ccdutils."""
        assert fm.imslice is ccdutils.imslice
        assert fm.cut_ccd is ccdutils.cut_ccd
        assert fm.bin_ccd is ccdutils.bin_ccd
        assert fm.set_ccd_attribute is ccdutils.set_ccd_attribute

    def test_bin_ccd_uses_xyz_header_keys_for_2d(self):
        """2-D binning should write X/Y binning header cards."""
        ccd = CCDData(np.arange(6 * 8).reshape(6, 8), unit="adu")

        out = ccdutils.bin_ccd(ccd, factors=(2, 3), binfunc=np.sum)

        assert out.shape == (2, 4)
        assert _header_value(out.header, "XBINNING") == 2
        assert _header_value(out.header, "YBINNING") == 3
        assert "BINNING1" not in out.header

    def test_bin_ccd_uses_xyz_header_keys_for_3d(self):
        """3-D binning should write X/Y/Z binning header cards."""
        ccd = CCDData(np.arange(4 * 6 * 8).reshape(4, 6, 8), unit="adu")

        out = ccdutils.bin_ccd(ccd, factors=(2, 3, 2), binfunc=np.sum)

        assert out.shape == (2, 2, 4)
        assert _header_value(out.header, "XBINNING") == 2
        assert _header_value(out.header, "YBINNING") == 3
        assert _header_value(out.header, "ZBINNING") == 2
        assert "BINNING1" not in out.header

    def test_bin_ccd_uses_numbered_header_keys_for_4d(self):
        """Higher-dimensional binning should write generic numbered cards."""
        ccd = CCDData(np.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6), unit="adu")

        out = ccdutils.bin_ccd(ccd, factors=(2, 2, 3, 1), binfunc=np.sum)

        assert out.shape == (2, 1, 2, 3)
        assert _header_value(out.header, "BINNING1") == 2
        assert _header_value(out.header, "BINNING2") == 2
        assert _header_value(out.header, "BINNING3") == 3
        assert _header_value(out.header, "BINNING4") == 1
        assert "XBINNING" not in out.header

    def test_bin_ccd_default_is_noop_for_nd_data(self):
        """Default factors should be a no-op for any data dimensionality."""
        ccd = CCDData(np.arange(2 * 3 * 4).reshape(2, 3, 4), unit="adu")

        out = ccdutils.bin_ccd(ccd)

        assert out is ccd

    def test_bin_ccd_header_records_effective_none_factor(self):
        """None factors should be recorded as their effective collapse factor."""
        ccd = CCDData(np.arange(4 * 6 * 8).reshape(4, 6, 8), unit="adu")

        out = ccdutils.bin_ccd(ccd, factors=(2, 3, None), binfunc=np.sum)

        assert out.shape == (1, 2, 4)
        assert _header_value(out.header, "XBINNING") == 2
        assert _header_value(out.header, "YBINNING") == 3
        assert _header_value(out.header, "ZBINNING") == 4
