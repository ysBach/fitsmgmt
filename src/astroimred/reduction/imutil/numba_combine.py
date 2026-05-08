"""
Per-pixel Numba JIT kernels for ndcombine (combine along axis=0, NaN ignored).

Designed so that chunking can later pass slices arr[:, y0:y1, x0:x1]
without changing the kernel signature (same 3D array, possibly a view).

When has_nan is False, fast-path kernels (no isfinite checks) are used for speed.
IMUTIL_USE_NUMBA is imported from ``imred.imutil`` (can be set via
``imred.IMUTIL_USE_NUMBA``).
"""

import numpy as np
from numba import njit, prange

# Import config to access IMUTIL_USE_NUMBA dynamically
from . import config

# --- Fast-path 1d kernels (no NaN/inf handling); use when has_nan is False ---


@njit(cache=True, fastmath=True)
def _mean_1d(vals):
    """Mean of all values (no isfinite check)."""
    s = 0.0
    for i in range(len(vals)):
        s += vals[i]
    return s / len(vals)


@njit(cache=True, fastmath=True)
def _sum_1d(vals):
    """Sum of all values (no isfinite check)."""
    s = 0.0
    for i in range(len(vals)):
        s += vals[i]
    return s


@njit(cache=True, fastmath=True)
def _min_1d(vals):
    """Min of all values (no isfinite check)."""
    out = vals[0]
    for i in range(1, len(vals)):
        if vals[i] < out:
            out = vals[i]
    return out


@njit(cache=True, fastmath=True)
def _max_1d(vals):
    """Max of all values (no isfinite check)."""
    out = vals[0]
    for i in range(1, len(vals)):
        if vals[i] > out:
            out = vals[i]
    return out


@njit(cache=True, fastmath=True)
def _median_1d(vals):
    """Median (mean of two middle for even n); no isfinite check."""
    buf = np.copy(vals)
    buf.sort()
    n = len(buf)
    mid = n // 2
    if n % 2 == 1:
        return buf[mid]
    return (buf[mid - 1] + buf[mid]) / 2.0


@njit(cache=True, fastmath=True)
def _median_lower_1d(vals):
    """Lower median (for even n take lower of two middle); no isfinite check."""
    buf = np.copy(vals)
    buf.sort()
    n = len(buf)
    mid = n // 2
    if n % 2 == 1:
        return buf[mid]
    return buf[mid - 1]


# --- NaN-ignoring 1d kernels (use when has_nan is True) ---


@njit(cache=True)
def _nanmean_1d(vals):
    """Mean of finite values; NaN if none."""
    s = 0.0
    n = 0
    for i in range(len(vals)):
        x = vals[i]
        if np.isfinite(x):
            s += x
            n += 1
    if n == 0:
        return np.nan
    return s / n


@njit(cache=True)
def _nansum_1d(vals):
    """Sum of finite values; NaN if none."""
    s = 0.0
    n = 0
    for i in range(len(vals)):
        x = vals[i]
        if np.isfinite(x):
            s += x
            n += 1
    if n == 0:
        return np.nan
    return s


@njit(cache=True)
def _nanmin_1d(vals):
    """Min of finite values; NaN if none."""
    out = np.nan
    for i in range(len(vals)):
        x = vals[i]
        if np.isfinite(x):
            if np.isnan(out) or x < out:
                out = x
    return out


@njit(cache=True)
def _nanmax_1d(vals):
    """Max of finite values; NaN if none."""
    out = np.nan
    for i in range(len(vals)):
        x = vals[i]
        if np.isfinite(x):
            if np.isnan(out) or x > out:
                out = x
    return out


@njit(cache=True)
def _nanmedian_1d(vals):
    """Median of finite values (mean of two middle for even); NaN if none."""
    buf = np.empty(len(vals))
    n = 0
    for i in range(len(vals)):
        x = vals[i]
        if np.isfinite(x):
            buf[n] = x
            n += 1
    if n == 0:
        return np.nan
    buf = buf[:n]
    buf.sort()
    mid = n // 2
    if n % 2 == 1:
        return buf[mid]
    return (buf[mid - 1] + buf[mid]) / 2.0


@njit(cache=True)
def _nanmedian_lower_1d(vals):
    """Lower median of finite values (for even n take lower of two middle); NaN if none."""
    buf = np.empty(len(vals))
    n = 0
    for i in range(len(vals)):
        x = vals[i]
        if np.isfinite(x):
            buf[n] = x
            n += 1
    if n == 0:
        return np.nan
    buf = buf[:n]
    buf.sort()
    mid = n // 2
    if n % 2 == 1:
        return buf[mid]
    return buf[mid - 1]


@njit(cache=True, parallel=True)
def combine_nanmean(arr):
    """(N, H, W) -> (H, W); mean along axis=0 ignoring NaN."""
    n, h, w = arr.shape
    out = np.empty((h, w), dtype=arr.dtype)
    for i in prange(h):
        for j in range(w):
            col = np.copy(arr[:, i, j])
            out[i, j] = _nanmean_1d(col)
    return out


@njit(cache=True, parallel=True)
def combine_nansum(arr):
    """(N, H, W) -> (H, W); sum along axis=0 ignoring NaN."""
    n, h, w = arr.shape
    out = np.empty((h, w), dtype=arr.dtype)
    for i in prange(h):
        for j in range(w):
            out[i, j] = _nansum_1d(arr[:, i, j])
    return out


@njit(cache=True, parallel=True)
def combine_nanmin(arr):
    """(N, H, W) -> (H, W); min along axis=0 ignoring NaN."""
    n, h, w = arr.shape
    out = np.empty((h, w), dtype=arr.dtype)
    for i in prange(h):
        for j in range(w):
            out[i, j] = _nanmin_1d(arr[:, i, j])
    return out


@njit(cache=True, parallel=True)
def combine_nanmax(arr):
    """(N, H, W) -> (H, W); max along axis=0 ignoring NaN."""
    n, h, w = arr.shape
    out = np.empty((h, w), dtype=arr.dtype)
    for i in prange(h):
        for j in range(w):
            out[i, j] = _nanmax_1d(arr[:, i, j])
    return out


@njit(cache=True, parallel=True)
def combine_nanmedian(arr):
    """(N, H, W) -> (H, W); median along axis=0 ignoring NaN (mean of two for even)."""
    n, h, w = arr.shape
    out = np.empty((h, w), dtype=arr.dtype)
    for i in prange(h):
        for j in range(w):
            out[i, j] = _nanmedian_1d(arr[:, i, j])
    return out


@njit(cache=True, parallel=True)
def combine_nanlmedian(arr):
    """(N, H, W) -> (H, W); lower median along axis=0 ignoring NaN."""
    n, h, w = arr.shape
    out = np.empty((h, w), dtype=arr.dtype)
    for i in prange(h):
        for j in range(w):
            out[i, j] = _nanmedian_lower_1d(arr[:, i, j])
    return out


# --- Fast-path combine (no NaN handling) ---


@njit(cache=True, fastmath=True, parallel=True)
def combine_mean(arr):
    """(N, H, W) -> (H, W); mean along axis=0 (no NaN check)."""
    n, h, w = arr.shape
    out = np.empty((h, w), dtype=arr.dtype)
    for i in prange(h):
        for j in range(w):
            out[i, j] = _mean_1d(arr[:, i, j])
    return out


@njit(cache=True, fastmath=True, parallel=True)
def combine_sum(arr):
    """(N, H, W) -> (H, W); sum along axis=0 (no NaN check)."""
    n, h, w = arr.shape
    out = np.empty((h, w), dtype=arr.dtype)
    for i in prange(h):
        for j in range(w):
            out[i, j] = _sum_1d(arr[:, i, j])
    return out


@njit(cache=True, fastmath=True, parallel=True)
def combine_min(arr):
    """(N, H, W) -> (H, W); min along axis=0 (no NaN check)."""
    n, h, w = arr.shape
    out = np.empty((h, w), dtype=arr.dtype)
    for i in prange(h):
        for j in range(w):
            out[i, j] = _min_1d(arr[:, i, j])
    return out


@njit(cache=True, fastmath=True, parallel=True)
def combine_max(arr):
    """(N, H, W) -> (H, W); max along axis=0 (no NaN check)."""
    n, h, w = arr.shape
    out = np.empty((h, w), dtype=arr.dtype)
    for i in prange(h):
        for j in range(w):
            out[i, j] = _max_1d(arr[:, i, j])
    return out


@njit(cache=True, fastmath=True, parallel=True)
def combine_median(arr):
    """(N, H, W) -> (H, W); median along axis=0 (no NaN check)."""
    n, h, w = arr.shape
    out = np.empty((h, w), dtype=arr.dtype)
    for i in prange(h):
        for j in range(w):
            out[i, j] = _median_1d(arr[:, i, j])
    return out


@njit(cache=True, fastmath=True, parallel=True)
def combine_median_lower(arr):
    """(N, H, W) -> (H, W); lower median along axis=0 (no NaN check)."""
    n, h, w = arr.shape
    out = np.empty((h, w), dtype=arr.dtype)
    for i in prange(h):
        for j in range(w):
            out[i, j] = _median_lower_1d(arr[:, i, j])
    return out


def combine_along_axis0_numba(arr, combine, has_nan=True):
    """
    Combine arr (N, H, W) along axis=0 using Numba kernels.

    Parameters
    ----------
    arr : ndarray
        (N, H, W) array.
    combine : str
        One of 'average','mean','median','lmedian','sum','min','max'.
    has_nan : bool, optional
        If True, use NaN-ignoring kernels (slower). If False, use fast path
        (no isfinite checks).

    Returns
    -------
    (H, W) array, or None if Numba disabled or unsupported combine.
    """
    if not config._get_use_numba():
        return None
    arr = np.ascontiguousarray(arr)
    if arr.ndim != 3:
        return None
    combine = combine.lower() if isinstance(combine, str) else combine

    if combine in ("average", "mean"):
        return combine_nanmean(arr) if has_nan else combine_mean(arr)
    if combine in ("med", "medi", "median"):
        return combine_nanmedian(arr) if has_nan else combine_median(arr)
    if combine in ("lmed", "lmd", "lmedian"):
        return combine_nanlmedian(arr) if has_nan else combine_median_lower(arr)
    if combine == "sum":
        return combine_nansum(arr) if has_nan else combine_sum(arr)
    if combine in ("min", "mini", "minimum"):
        return combine_nanmin(arr) if has_nan else combine_min(arr)
    if combine in ("max", "maxi", "maximum"):
        return combine_nanmax(arr) if has_nan else combine_max(arr)
    return None
