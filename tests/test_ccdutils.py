"""Tests for CCDData manipulation helpers."""

import fitsmgmt as fm
from fitsmgmt import ccdutils


class TestCcdUtils:
    """Tests for CCD helper module exports."""

    def test_ccd_helpers_have_canonical_modules(self):
        """CCD operations are exposed from ccdutils."""
        assert fm.imslice is ccdutils.imslice
        assert fm.cut_ccd is ccdutils.cut_ccd
        assert fm.bin_ccd is ccdutils.bin_ccd
        assert fm.set_ccd_attribute is ccdutils.set_ccd_attribute
