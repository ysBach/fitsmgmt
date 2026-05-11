"""
Per-pixel Numba JIT kernels for rejection algorithms (sigclip, ccdclip, minmax).

Designed so that chunking can later pass slices arr[:, y0:y1, x0:x1]
without changing the kernel signature (same 3D array, possibly a view).

IMUTIL_USE_NUMBA is imported from ``imred.imutil`` (can be set via
``imred.IMUTIL_USE_NUMBA``).
"""

import numpy as np
from numba import njit, prange

# Import config to access IMUTIL_USE_NUMBA dynamically
from . import config


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
def _nanstd_1d(vals, ddof=0):
    """Std of finite values; NaN if none."""
    mean = _nanmean_1d(vals)
    if np.isnan(mean):
        return np.nan
    s = 0.0
    n = 0
    for i in range(len(vals)):
        x = vals[i]
        if np.isfinite(x):
            diff = x - mean
            s += diff * diff
            n += 1
    if n <= ddof:
        return np.nan
    return np.sqrt(s / (n - ddof))


@njit(cache=True)
def _reject_sigclip_pixel(
    col,
    mask_in,
    sigma_lower,
    sigma_upper,
    maxiters,
    ddof,
    nkeep,
    maxrej,
    use_median,
    irafmode,
    ccdclip,
    rdnoise_ref,
    snoise_ref,
    scale_ref,
    zero_ref,
    dtype_max,
):
    """
    Per-pixel sigclip rejection.

    Parameters
    ----------
    col : 1D array (N,)
            Column of values for this pixel
    mask_in : 1D bool array (N,)
            Input mask (True = already masked)
    sigma_lower, sigma_upper : float
            Sigma thresholds
    maxiters : int
            Maximum iterations
    ddof : int
            Degrees of freedom for std
    nkeep : int
            Minimum number to keep
    maxrej : int
            Maximum number to reject
    use_median : bool
            True for median, False for mean
    irafmode : bool
            If True, restore pixels based on residual

    Returns
    -------
    mask_out : 1D bool array (N,)
            Output mask (True = rejected)
    low : float
            Lower bound
    upp : float
            Upper bound
    nit : int
            Number of iterations
    code : int
            Rejection code
    """
    n = len(col)
    # Copy column and apply input mask
    vals = np.empty(n)
    mask = np.zeros(n, dtype=np.bool_)
    n_finite = 0
    for i in range(n):
        if mask_in[i] or not np.isfinite(col[i]):
            mask[i] = True
            vals[i] = np.nan
        else:
            vals[i] = col[i]
            n_finite += 1

    n_finite_old = n_finite
    nkeep_val = nkeep if nkeep > 0 else 0
    maxrej_val = maxrej if maxrej is not None else n
    mask_nkeep = n_finite < nkeep_val
    mask_maxrej = (n - n_finite) > maxrej_val
    mask_pix = mask_nkeep | mask_maxrej

    # Initial bounds
    low = np.nan
    upp = np.nan
    for i in range(n):
        if not mask[i]:
            if np.isnan(low) or vals[i] < low:
                low = vals[i]
            if np.isnan(upp) or vals[i] > upp:
                upp = vals[i]
    low_new = low
    upp_new = upp

    nit = 1
    cen = np.nan
    std = np.nan

    # Iterate
    if (nkeep_val == 0) and (maxrej_val == n):
        # No nkeep/maxrej checks
        for _k in range(maxiters):
            # Compute center and std
            cen = _nanmedian_1d(vals) if use_median else _nanmean_1d(vals)
            if ccdclip:
                std = np.sqrt(
                    (1.0 + snoise_ref) * abs(cen + zero_ref) * scale_ref
                    + rdnoise_ref * rdnoise_ref
                )
            else:
                std = _nanstd_1d(vals, ddof)
            if np.isnan(cen) or np.isnan(std):
                break

            low_new = cen - sigma_lower * std
            upp_new = cen + sigma_upper * std

            # Mask out of bounds
            n_finite_new = 0
            for i in range(n):
                if not mask[i]:
                    if vals[i] < low_new or vals[i] > upp_new:
                        mask[i] = True
                        vals[i] = np.nan
                    else:
                        n_finite_new += 1

            if n_finite_new == n_finite_old:
                break
            n_finite_old = n_finite_new
            nit += 1
    else:
        # With nkeep/maxrej checks
        for _k in range(maxiters):
            # Compute center and std
            cen = _nanmedian_1d(vals) if use_median else _nanmean_1d(vals)
            if ccdclip:
                std = np.sqrt(
                    (1.0 + snoise_ref) * abs(cen + zero_ref) * scale_ref
                    + rdnoise_ref * rdnoise_ref
                )
            else:
                std = _nanstd_1d(vals, ddof)
            if np.isnan(cen) or np.isnan(std):
                break

            low_new = cen - sigma_lower * std
            upp_new = cen + sigma_upper * std

            # Mask out of bounds
            n_finite_new = 0
            for i in range(n):
                if not mask[i]:
                    if vals[i] < low_new or vals[i] > upp_new:
                        mask[i] = True
                        vals[i] = np.nan
                    else:
                        n_finite_new += 1

            nrej = n - n_finite_new
            mask_nkeep = n_finite_new < nkeep_val
            mask_maxrej = nrej > maxrej_val
            mask_pix = mask_nkeep | mask_maxrej

            # Revert bounds for masked pixels
            if mask_pix:
                low_new = low
                upp_new = upp

            if n_finite_new == n_finite_old:
                break
            n_finite_old = n_finite_new
            nit += 1

    # Final mask: input mask OR out of bounds in original array (matching original _iter_rej logic)
    mask_final = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        if (
            mask_in[i]
            or not np.isfinite(col[i])
            or col[i] < low_new
            or col[i] > upp_new
        ):
            mask_final[i] = True

    # IRAF mode: restore based on residual (after final mask computed, matching original)
    if irafmode and not np.isnan(cen):
        n_minimum = max(nkeep_val, n - maxrej_val) if maxrej_val < n else nkeep_val
        if n_minimum > 0:
            # Compute residuals from original array (matching original: resid = abs(_arr - cen))
            resid = np.empty(n)
            for i in range(n):
                if mask_final[i]:
                    resid[i] = dtype_max  # Set to large value so it won't be selected
                else:
                    resid[i] = abs(col[i] - cen)

            # Sort and find cutoff (matching original: max of partition[:n_minimum])
            resid_sorted = np.sort(resid)
            if n_minimum > 0 and n_minimum <= len(resid_sorted):
                resid_cut = resid_sorted[n_minimum - 1]
                # Restore pixels with resid <= cutoff
                for i in range(n):
                    if mask_final[i] and resid[i] <= resid_cut:
                        mask_final[i] = False

    # Compute code
    code = 0
    if maxiters == 0:
        code = 1
    else:
        mask_nochange = n_finite_new == n_finite_old
        nrej = n - n_finite_new
        mask_nkeep = n_finite_new < nkeep_val
        mask_maxrej = nrej > maxrej_val
        code = 2 * mask_nochange + 4 * mask_nkeep + 8 * mask_maxrej

    return mask_final, low_new, upp_new, nit, code


@njit(cache=True, parallel=True)
def reject_sigclip_3d(
    arr,
    mask_in,
    sigma_lower,
    sigma_upper,
    maxiters,
    ddof,
    nkeep,
    maxrej,
    use_median,
    irafmode,
    ccdclip,
    rdnoise_ref,
    snoise_ref,
    scale_ref,
    zero_ref,
    dtype_max,
):
    """
    (N, H, W) -> (H, W) masks, bounds, nit, code.
    Per-pixel sigclip rejection.
    """
    n, h, w = arr.shape
    mask_out = np.zeros((n, h, w), dtype=np.bool_)
    low_out = np.empty((h, w), dtype=arr.dtype)
    upp_out = np.empty((h, w), dtype=arr.dtype)
    nit_out = np.empty((h, w), dtype=np.uint8)
    code_out = np.empty((h, w), dtype=np.uint8)

    for i in prange(h):
        for j in range(w):
            m, low, upp, nit_val, code_val = _reject_sigclip_pixel(
                arr[:, i, j],
                mask_in[:, i, j],
                sigma_lower,
                sigma_upper,
                maxiters,
                ddof,
                nkeep,
                maxrej,
                use_median,
                irafmode,
                ccdclip,
                rdnoise_ref,
                snoise_ref,
                scale_ref,
                zero_ref,
                dtype_max,
            )
            mask_out[:, i, j] = m
            low_out[i, j] = low
            upp_out[i, j] = upp
            nit_out[i, j] = nit_val
            code_out[i, j] = code_val

    return mask_out, low_out, upp_out, nit_out, code_out


@njit(cache=True)
def _reject_minmax_pixel(col, mask_in, q_low, q_upp, calc_low, calc_upp):
    """
    Per-pixel minmax rejection.

    Returns: mask, low, upp, code
    """
    n = len(col)
    vals = np.empty(n)
    mask = np.zeros(n, dtype=np.bool_)
    n_finite = 0

    # Apply input mask
    for i in range(n):
        if mask_in[i] or not np.isfinite(col[i]):
            mask[i] = True
            vals[i] = np.nan
        else:
            vals[i] = col[i]
            n_finite += 1

    # Compute rejection counts (IRAF: add 0.001)
    n_rej_low = int(n_finite * q_low + 0.001)
    n_rej_upp = int(n_finite * q_upp + 0.001)

    low = np.nan
    upp = np.nan
    for i in range(n):
        if not mask[i]:
            if np.isnan(low) or vals[i] < low:
                low = vals[i]
            if np.isnan(upp) or vals[i] > upp:
                upp = vals[i]

    # Reject lowest
    if n_rej_low > 0:
        # Find n_rej_low smallest indices
        for _ in range(n_rej_low):
            min_idx = -1
            min_val = np.inf
            for i in range(n):
                if not mask[i] and vals[i] < min_val:
                    min_val = vals[i]
                    min_idx = i
            if min_idx >= 0:
                mask[min_idx] = True
                vals[min_idx] = np.nan

    # Reject highest
    if n_rej_upp > 0:
        for _ in range(n_rej_upp):
            max_idx = -1
            max_val = -np.inf
            for i in range(n):
                if not mask[i] and vals[i] > max_val:
                    max_val = vals[i]
                    max_idx = i
            if max_idx >= 0:
                mask[max_idx] = True
                vals[max_idx] = np.nan

    code = 1 if (n_rej_low == 0 or n_rej_upp == 0) else 0
    return mask, low, upp, code


@njit(cache=True, parallel=True)
def reject_minmax_3d(arr, mask_in, q_low, q_upp, calc_low, calc_upp):
    """
    (N, H, W) -> (H, W) masks, bounds, code.
    Per-pixel minmax rejection.
    """
    n, h, w = arr.shape
    mask_out = np.zeros((n, h, w), dtype=np.bool_)
    low_out = np.empty((h, w), dtype=arr.dtype)
    upp_out = np.empty((h, w), dtype=arr.dtype)
    code_out = np.empty((h, w), dtype=np.uint8)

    for i in prange(h):
        for j in range(w):
            m, low, upp, code_val = _reject_minmax_pixel(
                arr[:, i, j], mask_in[:, i, j], q_low, q_upp, calc_low, calc_upp
            )
            mask_out[:, i, j] = m
            low_out[i, j] = low
            upp_out[i, j] = upp
            code_out[i, j] = code_val

    return mask_out, low_out, upp_out, code_out


def reject_sigclip_numba(
    arr,
    mask,
    sigma_lower,
    sigma_upper,
    maxiters,
    ddof,
    nkeep,
    maxrej,
    cenfunc,
    irafmode,
    ccdclip=False,
    rdnoise_ref=0.0,
    snoise_ref=0.0,
    scale_ref=1.0,
    zero_ref=0.0,
):
    """
    Wrapper for Numba sigclip/ccdclip rejection.
    Returns (mask, low, upp, nit, code) or None if disabled/unsupported.
    """
    if not config._get_use_numba():
        return None
    if arr.ndim != 3:
        return None
    arr = np.ascontiguousarray(arr)
    mask = (
        np.ascontiguousarray(mask)
        if mask is not None
        else np.zeros(arr.shape, dtype=np.bool_)
    )
    use_median = "median" in str(cenfunc).lower() if cenfunc is not None else True
    dtype_max = float(np.finfo(arr.dtype).max)
    return reject_sigclip_3d(
        arr,
        mask,
        sigma_lower,
        sigma_upper,
        maxiters,
        ddof,
        nkeep,
        maxrej,
        use_median,
        irafmode,
        ccdclip,
        rdnoise_ref,
        snoise_ref,
        scale_ref,
        zero_ref,
        dtype_max,
    )


def reject_minmax_numba(arr, mask, q_low, q_upp, calc_low, calc_upp):
    """
    Wrapper for Numba minmax rejection.
    Returns (mask, low, upp, code) or None if disabled/unsupported.
    """
    if not config._get_use_numba():
        return None
    if arr.ndim != 3:
        return None
    arr = np.ascontiguousarray(arr)
    mask = (
        np.ascontiguousarray(mask)
        if mask is not None
        else np.zeros(arr.shape, dtype=np.bool_)
    )
    return reject_minmax_3d(arr, mask, q_low, q_upp, calc_low, calc_upp)
