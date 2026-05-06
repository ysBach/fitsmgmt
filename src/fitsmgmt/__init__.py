"""fitsmgmt: FITS file management and lightweight inspection utilities."""

from .logging import logger, set_log_level, enable_console_logging
from .airmass import *
from .ccdutils import *
from .headers import *
from .imstat import *
from .io import *
from .mathutils import *
from .misc import *
from .paths import *
from .pixels import *
from .summary import *
from .wcstools import *

from . import (
    airmass,
    ccdutils,
    headers,
    imstat,
    io,
    logging,
    mathutils,
    misc,
    paths,
    pixels,
    summary,
    wcstools,
)

__all__ = [
    "airmass",
    "ccdutils",
    "headers",
    "imstat",
    "io",
    "logging",
    "mathutils",
    "misc",
    "paths",
    "pixels",
    "summary",
    "wcstools",
    "logger",
    "set_log_level",
    "enable_console_logging",
]
for _module in (
    airmass,
    ccdutils,
    headers,
    imstat,
    io,
    mathutils,
    misc,
    paths,
    pixels,
    summary,
    wcstools,
):
    __all__.extend(getattr(_module, "__all__", []))
