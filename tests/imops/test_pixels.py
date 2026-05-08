"""Tests for pixel mask and saturation helpers."""

import astroimred as air
from astroimred import pixels


class TestPixelTools:
    """Tests for pixel helper module exports."""

    def test_pixel_helpers_have_canonical_modules(self):
        """Pixel operations are exposed from pixels."""
        assert air.fixpix is pixels.fixpix
        assert air.find_extpix is pixels.find_extpix
        assert air.find_satpix is pixels.find_satpix
