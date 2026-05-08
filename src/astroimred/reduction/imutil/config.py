# Central configuration for imutil
# Users can set via: imred.IMUTIL_USE_NUMBA = False
IMUTIL_USE_NUMBA = True


def _get_use_numba():
    """Returns the current value of IMUTIL_USE_NUMBA."""
    return IMUTIL_USE_NUMBA
