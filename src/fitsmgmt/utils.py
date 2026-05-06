"""Compatibility alias for the old fitsmgmt.utils skeleton module."""

from .misc import *
from astro_ndslice import listify

__all__ = list(globals().get("__all__", [])) + ["listify"]
