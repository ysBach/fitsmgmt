"""Tests for standalone numeric helpers."""

import numpy as np

from fitsmgmt import mathutils, misc

RTOL = 1e-6
ATOL = 1e-8


class TestWeightedAvg:
    """Tests for weighted_avg function."""

    def test_mathutils_owns_helper(self):
        """Moved math helpers are not re-exported from misc."""
        assert not hasattr(misc, "weighted_avg")

    def test_known_values(self):
        """Test weighted average with known values."""
        val = np.array([1.0, 2.0, 3.0])
        err = np.array([0.1, 0.2, 0.1])  # weights = 1/err^2

        # Manual calculation:
        # w = 1/err^2 = [100, 25, 100]
        # weighted_avg = (1*100 + 2*25 + 3*100) / (100+25+100)
        #              = (100 + 50 + 300) / 225 = 450/225 = 2.0
        result = mathutils.weighted_avg(val, err)
        # Result is (weighted_avg, weighted_std_err)
        np.testing.assert_allclose(result[0], 2.0, rtol=RTOL, atol=ATOL)

    def test_equal_weights(self):
        """Test that equal weights give simple mean."""
        val = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        err = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = mathutils.weighted_avg(val, err)
        np.testing.assert_allclose(result[0], 3.0, rtol=RTOL, atol=ATOL)


class TestMinMaxMed1D:
    """Tests for min_max_med_1d function."""

    def test_mathutils_owns_helpers(self):
        """Moved math helpers are not re-exported from misc."""
        assert not hasattr(misc, "min_max_med_1d")
        assert not hasattr(misc, "mean_std_1d")
        assert not hasattr(misc, "quantile_lh")
        assert not hasattr(misc, "quantile_sigma")
        assert not hasattr(misc, "binning")

    def test_odd_length(self):
        """Odd-length arrays should return the central sorted value."""
        arr = np.array([3, 1, 2])

        assert mathutils.min_max_med_1d(arr) == (1, 3, 2)

    def test_even_length(self):
        """Even-length arrays should average the two central sorted values."""
        arr = np.array([4, 1, 2, 3])

        assert mathutils.min_max_med_1d(arr) == (1, 4, 2.5)


class TestGainConversion:
    """Tests for gain conversion helpers."""

    def test_mathutils_owns_helpers(self):
        """Moved gain helpers are not re-exported from misc."""
        assert not hasattr(misc, "dB2epadu")
        assert not hasattr(misc, "epadu2dB")

    def test_roundtrip(self):
        """dB and electron/ADU conversions should round-trip."""
        gain = 2.5
        np.testing.assert_allclose(
            mathutils.dB2epadu(mathutils.epadu2dB(gain)),
            gain,
            rtol=RTOL,
            atol=ATOL,
        )
