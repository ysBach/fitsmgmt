"""Tests for pixel mask and saturation helpers."""

import fitsmgmt as fm
from fitsmgmt import pixels


class TestPixelTools:
    """Tests for pixel helper module exports."""

    def test_pixel_helpers_have_canonical_modules(self):
        """Pixel operations are exposed from pixels."""
        assert fm.fixpix is pixels.fixpix
        assert fm.find_extpix is pixels.find_extpix
        assert fm.find_satpix is pixels.find_satpix
