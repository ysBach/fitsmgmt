import numpy as np
from astropy.io import fits

from astroimred import summary


class TestSummary:
    """Tests for summary module."""

    def test_fits_summary(self, tmp_path):
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

        # Run fits_summary
        df = summary.fits_summary(
            paths,
            keywords=keys,
            verbose=False
        )

        assert len(df) == 3
        assert "file" in df.columns
        assert list(df["OBJECT"]) == ["M1", "M1", "M2"]
        assert list(df["FILTER"]) == ["V", "B", "V"]
        np.testing.assert_allclose(df["EXPTIME"], [10.0, 20.0, 10.0])

    def test_fits_summary_parq_output(self, tmp_path, monkeypatch):
        """Test parquet output selection for .parq summary files."""
        p = tmp_path / "img.fits"
        hdr = fits.Header()
        hdr["OBJECT"] = "M1"
        fits.writeto(p, np.zeros((10, 10)), header=hdr)

        calls = []

        def fake_to_parquet(self, output, index=False):
            calls.append((output, index, list(self.columns)))

        monkeypatch.setattr("pandas.DataFrame.to_parquet", fake_to_parquet)

        output = tmp_path / "summary.parq"
        df = summary.fits_summary(
            [p],
            keywords=["OBJECT"],
            output=output,
            verbose=False,
        )

        assert list(df["OBJECT"]) == ["M1"]
        assert calls == [(output, False, ["file", "filesize", "OBJECT"])]
