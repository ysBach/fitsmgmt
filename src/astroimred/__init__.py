"""astroimred: astronomical image reduction and FITS utilities."""

import sys as _sys

from . import imops, mgmt
from .imops import *
from .imops import ccdutils, imstat, mathutils, pixels, viz
from .imops.ccdutils import *
from .imops.imstat import *
from .imops.mathutils import *
from .imops.pixels import *
from .logging import enable_console_logging, logger, set_log_level
from .mgmt import *
from .mgmt import airmass, headers, io, logging, misc, paths, summary, wcstools
from .mgmt.airmass import *
from .mgmt.headers import *
from .mgmt.io import *
from .mgmt.misc import *
from .mgmt.paths import *
from .mgmt.summary import *
from .mgmt.wcstools import *

_COMPAT_MODULES = {
    "airmass": airmass,
    "headers": headers,
    "io": io,
    "logging": logging,
    "misc": misc,
    "paths": paths,
    "summary": summary,
    "wcstools": wcstools,
    "ccdutils": ccdutils,
    "imstat": imstat,
    "mathutils": mathutils,
    "pixels": pixels,
    "viz": viz,
}
for _name, _module in _COMPAT_MODULES.items():
    _sys.modules[f"{__name__}.{_name}"] = _module

__all__ = [
    "mgmt",
    "imops",
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
    "viz",
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
    viz,
    summary,
    wcstools,
    mgmt,
    imops,
):
    __all__.extend(getattr(_module, "__all__", []))

__all__ = list(dict.fromkeys(__all__))
