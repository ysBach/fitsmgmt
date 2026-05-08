"""Utilities that resemble IRAF's IMUTIL package.

The Python versions keep the familiar names (``imred.imcombine``,
``imred.imarith``, etc.). The CLI uses one ``imred`` command with subcommands
(``imred comb``, ``imred copy``, ``imred arith``, etc.) to avoid colliding with
IRAF and other astronomy command-line tools.
"""

from .config import IMUTIL_USE_NUMBA
from .imarith import *
from .imcombine import *
from .imcopy import *
from .imsmooth import *
