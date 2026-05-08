import pytest

pytest.importorskip("ccdproc")
pytest.importorskip("scipy")

import astroimred.reduction as imred


def pytest_configure(config):
    # The legacy regression fixtures were generated against the non-numba path.
    imred.imutil_config.IMUTIL_USE_NUMBA = False
