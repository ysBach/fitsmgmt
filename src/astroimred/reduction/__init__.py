import sys
import types

from .combutil import *
from .crrej import *
from .imutil import *
from .imutil import IMUTIL_USE_NUMBA
from .imutil import config as imutil_config
from .preproc import *


def set_imutil_use_numba(value):
    """Set IMUTIL_USE_NUMBA flag.

    Usage:
        import astroimred.reduction as imred
        imred.set_imutil_use_numba(True)
        # Or directly: imred.IMUTIL_USE_NUMBA = True
    """
    imutil_config.IMUTIL_USE_NUMBA = bool(value)


class _NumbaModule(types.ModuleType):
    """Custom module that intercepts assignment to IMUTIL_USE_NUMBA"""

    def __setattr__(self, name, value):
        if name == "IMUTIL_USE_NUMBA":
            # Update the underlying value in imutil.config module
            imutil_config.IMUTIL_USE_NUMBA = bool(value)
        else:
            super().__setattr__(name, value)

    def __getattribute__(self, name):
        if name == "IMUTIL_USE_NUMBA":
            # Return the current value from imutil.config module
            return imutil_config.IMUTIL_USE_NUMBA
        return super().__getattribute__(name)


# Replace current module with custom module class
_current_module = sys.modules[__name__]
_new_module = _NumbaModule(__name__)
# Copy __dict__ directly (preserves all attributes)
_new_module.__dict__.update(_current_module.__dict__)
# Replace in sys.modules
sys.modules[__name__] = _new_module
