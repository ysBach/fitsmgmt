import numpy as np
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
