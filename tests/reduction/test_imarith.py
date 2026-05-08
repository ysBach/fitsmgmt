import numpy as np
from astropy.io import fits
from astropy.nddata import CCDData

import astroimred.reduction as imred


class TestImarith:
    """Tests for `~imred.imarith`."""

    def test_header_params_and_replace_are_preserved(self):
        data = np.array([[1.0, np.nan], [3.0, np.inf]])
        ccd = CCDData(data, unit="adu")
        ccd.header["OBJECT"] = "raw"

        result = imred.imarith(
            ccd,
            "+",
            1,
            replace=0,
            header_params={"OBJECT": "processed", "TESTKEY": 3},
            verbose=False,
        )

        assert result.header["OBJECT"] == "processed"
        assert result.header["TESTKEY"] == 3
        assert any("IMARITH" in item for item in result.header.get("HISTORY", []))
        np.testing.assert_allclose(result.data, [[2.0, 0.0], [4.0, 0.0]])

    def test_default_extension_uses_first_image_hdu(self, tmp_path):
        data = np.arange(6, dtype="float32").reshape(2, 3)
        header = fits.Header({"OBJECT": "science"})
        path = tmp_path / "mef.fits"
        fits.HDUList(
            [fits.PrimaryHDU(), fits.ImageHDU(data=data, header=header, name="SCI")]
        ).writeto(path)

        result = imred.imarith(path, "+", 1, verbose=False)

        np.testing.assert_allclose(result.data, data + 1)
        assert result.header["OBJECT"] == "science"

    def test_default_extension_output_keeps_valid_primary_header(self, tmp_path):
        data = np.arange(6, dtype="float32").reshape(2, 3)
        path = tmp_path / "mef.fits"
        output = tmp_path / "result.fits"
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(
                    data=data, header=fits.Header({"OBJECT": "science"}), name="SCI"
                ),
            ]
        ).writeto(path)

        imred.imarith(path, "+", 1, output=output, overwrite=True, verbose=False)

        with fits.open(output) as hdul:
            hdul.verify("exception")
            np.testing.assert_allclose(hdul[0].data, data + 1)
            assert hdul[0].header["OBJECT"] == "science"
