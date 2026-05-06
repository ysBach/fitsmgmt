"""fitsmgmt: FITS file management and lightweight inspection utilities."""

from .logging import logger, set_log_level, enable_console_logging
from .airmass import *
from .ccdutils import *
from .filemgmt import *
from .headers import *
from .imstat import *
from .io import *
from .mathutils import *
from .misc import *
from .pixels import *
from .wcstools import *

from . import (
    airmass,
    ccdutils,
    filemgmt,
    headers,
    imstat,
    io,
    logging,
    mathutils,
    misc,
    pixels,
    wcstools,
)

__all__ = [
    "airmass",
    "ccdutils",
    "filemgmt",
    "headers",
    "imstat",
    "io",
    "logging",
    "mathutils",
    "misc",
    "pixels",
    "wcstools",
    "logger",
    "set_log_level",
    "enable_console_logging",
]
for _module in (
    airmass,
    ccdutils,
    filemgmt,
    headers,
    imstat,
    io,
    mathutils,
    misc,
    pixels,
    wcstools,
):
    __all__.extend(getattr(_module, "__all__", []))
