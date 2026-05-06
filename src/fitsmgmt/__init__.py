"""fitsmgmt: FITS file management and lightweight inspection utilities."""

from .logging import logger, set_log_level, enable_console_logging
from .airmass import *
from .filemgmt import *
from .hduutil import *
from .misc import *

from . import airmass, filemgmt, files, hduutil, images, logging, misc, utils, wcstools

__all__ = [
    "airmass",
    "filemgmt",
    "files",
    "hduutil",
    "images",
    "logging",
    "misc",
    "utils",
    "wcstools",
    "logger",
    "set_log_level",
    "enable_console_logging",
]
for _module in (airmass, filemgmt, hduutil, misc):
    __all__.extend(getattr(_module, "__all__", []))
