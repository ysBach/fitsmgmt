import numpy as np
from astropy.io import fits

from fitsmgmt import summary


class TestSummary:
    """Tests for summary module."""

    def test_make_summary(self, tmp_path):
        """Test creating summary table from FITS files."""
        # Create files with headers
        keys = ["OBJECT", "FILTER", "EXPTIME"]
        data = [
            ("M1", "V", 10.0),
            ("M1", "B", 20.0),
            ("M2", "V", 10.0)
        ]

        paths = []
        for i, (obj, filt, exp) in enumerate(data):
            p = tmp_path / f"img{i}.fits"
            hdr = fits.Header()
            hdr["OBJECT"] = obj
            hdr["FILTER"] = filt
            hdr["EXPTIME"] = exp
            fits.writeto(p, np.zeros((10, 10)), header=hdr)
            paths.append(str(p))

        # Run make_summary
        df = summary.make_summary(
            paths,
            keywords=keys,
            verbose=False
        )

        assert len(df) == 3
        assert "file" in df.columns
        assert list(df["OBJECT"]) == ["M1", "M1", "M2"]
        assert list(df["FILTER"]) == ["V", "B", "V"]
        np.testing.assert_allclose(df["EXPTIME"], [10.0, 20.0, 10.0])
