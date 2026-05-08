import pytest

pytest.importorskip("ccdproc")
pytest.importorskip("scipy")

import astroimred.reduction as fir


def pytest_configure(config):
    # The legacy regression fixtures were generated against the non-numba path.
    fir.imutil_config.IMUTIL_USE_NUMBA = False
