"""FITS file, header, WCS, path, and metadata management helpers."""

from .logging import logger, set_log_level, enable_console_logging
from .airmass import *
from .headers import *
from .io import *
from .misc import *
from .paths import *
from .summary import *
from .wcstools import *

from . import airmass, headers, io, logging, misc, paths, summary, wcstools

__all__ = [
    "airmass",
    "headers",
    "io",
    "logging",
    "misc",
    "paths",
    "summary",
    "wcstools",
    "logger",
    "set_log_level",
    "enable_console_logging",
]

for _module in (airmass, headers, io, logging, misc, paths, summary, wcstools):
    __all__.extend(getattr(_module, "__all__", []))

__all__ = list(dict.fromkeys(__all__))
