"""Tests for standalone misc utilities."""

import numpy as np
import pytest

from astroimred import misc

# Strict tolerance for numerical comparisons
RTOL = 1e-6
ATOL = 1e-8


class TestMovedHelpers:
    """Tests for helpers moved out of misc."""

    def test_io_owns_helper(self):
        """Moved I/O helpers are not re-exported from misc."""
        assert not hasattr(misc, "get_size")

    def test_mathutils_owns_helpers(self):
        """Moved math helpers are not re-exported from misc."""
        assert not hasattr(misc, "weighted_avg")
        assert not hasattr(misc, "min_max_med_1d")
        assert not hasattr(misc, "mean_std_1d")
        assert not hasattr(misc, "quantile_lh")
        assert not hasattr(misc, "quantile_sigma")
        assert not hasattr(misc, "binning")
        assert not hasattr(misc, "dB2epadu")
        assert not hasattr(misc, "epadu2dB")

    def test_headers_own_helpers(self):
        """Moved header helpers are not re-exported from misc."""
        assert not hasattr(misc, "cmt2hdr")
        assert not hasattr(misc, "update_tlm")
        assert not hasattr(misc, "update_process")


class TestCircularMask:
    """Tests for circular_mask function."""

    def test_basic_2d(self):
        """Test basic 2D circular mask."""
        mask = misc.circular_mask(shape=(10, 10), center=(5, 5), radius=3)
        assert mask.shape == (10, 10)
        assert mask.dtype == bool
        # Center should be inside the circle
        assert mask[5, 5] == True
        # Corners should be outside
        assert mask[0, 0] == False
        assert mask[9, 9] == False

    def test_mask_sum_known(self):
        """Test that mask sum matches expected count."""
        # For a 21x21 grid centered at (10,10) with radius=5
        # The number of pixels inside should be approximately pi*r^2 = 78.5
        mask = misc.circular_mask(shape=(21, 21), center=(10, 10), radius=5)
        # Allow some tolerance for discretization
        assert 70 <= np.sum(mask) <= 90

    def test_default_center(self):
        """Test that default center is image center."""
        mask = misc.circular_mask(shape=(10, 10), radius=2)
        # Default center should be (5, 5) for a 10x10 image
        assert mask[5, 5] == True


class TestCircularMask2D:
    """Tests for circular_mask_2d function (photutils-based)."""

    def test_basic(self):
        """Test basic 2D circular mask using photutils."""
        mask = misc.circular_mask_2d(shape=(100, 100), center=(50, 50), radius=10)
        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        # Center should be inside
        assert mask[50, 50] == True

    @pytest.mark.parametrize("radius,expected_sum", [
        (1.0, 1),
        (5.0, 69),
        (10.0, 305),
    ])
    def test_mask_sum_by_radius(self, radius, expected_sum):
        """Test mask pixel count for various radii."""
        mask = misc.circular_mask_2d(
            shape=(100, 100),
            center=(50, 50),
            radius=radius,
            method="center"
        )
        assert np.sum(mask) == expected_sum


class TestStrNow:
    """Tests for str_now function."""

    def test_returns_string(self):
        """Test that str_now returns a string."""
        result = misc.str_now()
        assert isinstance(result, str)

    def test_precision(self):
        """Test precision parameter affects output."""
        result_low = misc.str_now(precision=0)
        result_high = misc.str_now(precision=6)
        # Higher precision should result in longer string
        # (more decimal places in seconds)
        # Both should be valid ISO format times
        assert "T" in result_low
        assert "T" in result_high


class TestChangeToQuantity:
    """Tests for change_to_quantity function."""

    def test_float_to_quantity(self):
        """Test converting `float` to `~astropy.units.Quantity`."""
        from astropy import units as u
        result = misc.change_to_quantity(5.0, u.m, to_value=False)
        assert hasattr(result, "unit")
        assert result.value == 5.0

    def test_quantity_passthrough(self):
        """Test that `~astropy.units.Quantity` is passed through."""
        from astropy import units as u
        q = 5.0 * u.m
        result = misc.change_to_quantity(q, u.m, to_value=False)
        assert result.value == 5.0
        assert result.unit == u.m

    def test_to_value_true(self):
        """Test extracting value from `~astropy.units.Quantity`."""
        from astropy import units as u
        result = misc.change_to_quantity(5.0 * u.km, u.m, to_value=True)
        np.testing.assert_allclose(result, 5000.0, rtol=RTOL, atol=ATOL)
