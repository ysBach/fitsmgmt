"""fitsmgmt: FITS file management and lightweight inspection utilities."""

from .logging import logger, set_log_level, enable_console_logging
from .airmass import *
from .filemgmt import *
from .hduutil import *
from .io import *
from .mathutils import *
from .misc import *
from .wcstools import *

from . import airmass, filemgmt, hduutil, io, logging, mathutils, misc, wcstools

__all__ = [
    "airmass",
    "filemgmt",
    "hduutil",
    "io",
    "logging",
    "mathutils",
    "misc",
    "wcstools",
    "logger",
    "set_log_level",
    "enable_console_logging",
]
for _module in (airmass, filemgmt, hduutil, io, mathutils, misc, wcstools):
    __all__.extend(getattr(_module, "__all__", []))
