"""Standalone array math helpers."""

import numpy as np

__all__ = [
    "weighted_avg",
    "quantile_lh",
    "quantile_sigma",
    "min_max_med_1d",
    "mean_std_1d",
    "binning",
    "dB2epadu",
    "epadu2dB",
]


def weighted_avg(val, err):
    """Return the inverse-variance weighted mean and standard error.

    Parameters
    ----------
    val, err : array-like
        Values and 1-sigma uncertainties. Weights are calculated as
        ``1 / err**2``.

    Returns
    -------
    mean, stderr : float
        Weighted mean and its standard error.
    """
    val = np.asarray(val)
    err = np.asarray(err)
    w = 1 / (err**2)
    wsum = np.sum(w)
    wvg = np.sum(w * val) / wsum
    wse = 1 / np.sqrt(wsum)
    return wvg, wse


def quantile_lh(
    a,
    lq,
    hq,
    axis=None,
    nanfunc=False,
    interpolation="linear",
    linterp=None,
    hinterp=None,
):
    """Return lower and upper quantiles.

    Parameters
    ----------
    a : array-like
        Input data.

    lq, hq : array_like of `float`
        Quantile or sequence of quantiles to compute, which must be between 0
        and 1 inclusive.

    axis : {`int`, `tuple` of `int`, `None`}, optional
        Axis or axes along which the quantiles are computed. The default is to
        compute the quantile(s) along a flattened version of the array.

    nanfunc : `bool`, optional
        Whether to use `~numpy.nanquantile` instead of `~numpy.quantile`.
        Default: `False`.

    interpolation, linterp, hinterp : ``{'linear', 'lower', 'higher', 'midpoint', 'nearest'}``, optional.
        This optional parameter specifies the interpolation method to use when
        the desired quantile lies between two data points ``i < j``:
        * 'linear': ``i + (j - i) * fraction``, where ``fraction`` is the
          fractional part of the index surrounded by ``i`` and ``j``.
        * 'lower': ``i``.
        * 'higher': ``j``.
        * 'nearest': ``i`` or ``j``, whichever is nearest.
        * 'midpoint': ``(i + j) / 2``.
        To tune the interpolation method for lower and higher quantiles
        individually, set `linterp` and `hinterp` separately. An idea is to use
        ``linterp='higher', hinterp='lower'`` to estimate the robust standard
        deviation estimate.
    """
    a = np.asarray(a)
    linterp = interpolation if linterp is None else linterp
    hinterp = interpolation if hinterp is None else hinterp

    qfunc = np.nanquantile if nanfunc else np.quantile

    try:
        lq = float(lq)
        hq = float(hq)
    except TypeError:
        raise TypeError("lq and hq must be floats, not array-like.")

    if linterp == hinterp:
        out = qfunc(a, (lq, hq), axis=axis, interpolation=linterp)
    else:
        out_l = qfunc(a, lq, axis=axis, interpolation=linterp)
        out_h = qfunc(a, hq, axis=axis, interpolation=hinterp)
        out = [out_l, out_h]

    return out


def quantile_sigma(
    a, axis=None, nanfunc=False, interpolation="linear", linterp=None, hinterp=None
):
    """Estimate sigma from the 15.87 and 84.13 percent quantiles."""
    low, upp = quantile_lh(
        a,
        0.1587,
        0.8413,
        axis=axis,
        nanfunc=nanfunc,
        interpolation=interpolation,
        linterp=linterp,
        hinterp=hinterp,
    )
    return np.abs(upp - low) / 2


def min_max_med_1d(arr):
    """Return the minimum, maximum, and median of a 1-D array."""
    arr = np.asarray(arr)
    if arr.size < 1000:
        _a = np.sort(arr)
        mid = _a.size // 2
        if _a.size % 2:
            med = _a[mid]
        else:
            med = 0.5 * (_a[mid] + _a[mid - 1])
        return _a[0], _a[-1], med
    else:
        return np.min(arr), np.max(arr), np.median(arr)


def mean_std_1d(arr, ddof=0, std=True, var=False):
    """Return mean with standard deviation and/or variance.

    Parameters
    ----------
    arr : array-like
        Input values.
    ddof : int, optional
        Delta degrees of freedom for variance normalization.
    std, var : bool, optional
        Select whether to include standard deviation and/or variance.
    """
    arr = np.asarray(arr)
    sum_a = np.sum(arr)
    sqsum = np.sum(arr**2)
    inv_n = 1.0 / arr.size
    inv_d = 1.0 / (arr.size - ddof) if ddof > 0 else inv_n
    mean = sum_a * inv_n
    var_value = sqsum * inv_d - mean * sum_a * inv_d
    if var:
        if std:
            return mean, np.sqrt(var_value), var_value
        return mean, var_value
    if std:
        return mean, np.sqrt(var_value)
    raise ValueError("At least one of `std` or `var` must be True.")


def _validate_binning_factor(factor, axis):
    if factor is None:
        return None
    if isinstance(factor, (bool, np.bool_)):
        raise ValueError(f"factor for axis {axis} must be a positive integer.")
    if not isinstance(factor, (int, np.integer)):
        raise ValueError(f"factor for axis {axis} must be a positive integer.")
    factor = int(factor)
    if factor < 1:
        raise ValueError(f"factor for axis {axis} must be a positive integer.")
    return factor


def _normalize_binning_factors(arr_shape, factors, order_xyz):
    ndim = len(arr_shape)
    if factors is None:
        return np.ones(ndim, dtype=np.intp)

    raw_factors = list(np.asarray(factors, dtype=object).ravel())
    if len(raw_factors) != ndim:
        raise ValueError(
            f"factors must have the same length as arr.ndim ({ndim}); "
            f"got {len(raw_factors)}."
        )
    if order_xyz:
        raw_factors = raw_factors[::-1]

    normalized = []
    for axis, factor in enumerate(raw_factors):
        factor = arr_shape[axis] if factor is None else factor
        normalized.append(_validate_binning_factor(factor, axis))
    return np.asarray(normalized, dtype=np.intp)


def binning(
    arr,
    factors=None,
    order_xyz=True,
    binfunc=np.mean,
    trim_end=False,
):
    """Bin an array by integer factors.

    Parameters
    ---------
    arr : array-like
        Input array.

    factors : `list`-like of `int`, optional.
        The factors in pythonic axis order (``order_xyz=False``) or in xyz-style
        order (``order_xyz=True``), which is reversed into NumPy axis order. The
        number of factors must match ``arr.ndim``. If any factor is ``None``,
        that factor is replaced by the size of the array along that axis, i.e.,
        collapse along that axis.
        Default: `None`.

    binfunc : callable, optional
        The function to be applied for binning, such as ``np.sum``,
        ``np.mean``, and ``np.median``.
        Default: ``np.mean``.

    trim_end : `bool`, optional.
        Whether to trim the end of each axis so that the trimmed shape is
        divisible by the binning factors.
        Default: `False`.

    Notes
    -----
    This kind of binning is ~ 20-30 to upto 10^5 times faster than
    astropy.nddata's block_reduce:


    >>> from astropy.nddata.blocks import block_reduce
    >>> import astroimred as air
    >>> from astropy.nddata import CCDData
    >>> import numpy as np
    >>> ccd = CCDData(data=np.arange(1000).reshape(20, 50), unit='adu')
    >>> bin_kw = dict(factors=(5, 5), binfunc=np.sum, trim_end=True)
    >>> ccd_kw = dict(factors=(5, 5), binfunc=np.sum, trim_end=True)
    >>> %timeit air.binning(ccd.data, **bin_kw)
    >>> # 10.9 +- 0.216 us (7 runs, 100000 loops each)
    >>> %timeit air.bin_ccd(ccd, **ccd_kw, update_header=False)
    >>> # 32.9 µs +- 878 ns per loop (7 runs, 10000 loops each)
    >>> %timeit -r 1 -n 1 block_reduce(ccd, block_size=5)
    >>> # 518 ms, 2.13 ms, 250 us, 252 us, 257 us, 267 us
    >>> # 5.e+5   ...      ...     ...     ...     27  -- times slower
    >>> # some strange caching happens?
    Tested on MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16
    GB (2400MHz DDR4), Radeon Pro 560X (4GB)]
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError("arr must not be empty.")

    factors = _normalize_binning_factors(
        arr.shape,
        factors,
        order_xyz,
    )
    shape = np.asarray(arr.shape, dtype=np.intp)

    if np.any(factors > shape):
        axis = int(np.flatnonzero(factors > shape)[0])
        raise ValueError(
            f"factor for axis {axis} ({factors[axis]}) is larger than "
            f"the axis length ({shape[axis]})."
        )

    remainder = shape % factors
    if trim_end:
        trim_shape = shape - remainder
        if np.any(trim_shape == 0):
            axis = int(np.flatnonzero(trim_shape == 0)[0])
            raise ValueError(
                f"factor for axis {axis} ({factors[axis]}) trims the axis "
                "to length 0."
            )
        slices = tuple(slice(None, int(size)) for size in trim_shape)
        arr = arr[slices]
        shape = trim_shape
    elif np.any(remainder):
        axis = int(np.flatnonzero(remainder)[0])
        raise ValueError(
            f"array shape along axis {axis} ({shape[axis]}) is not divisible "
            f"by factor {factors[axis]}; use trim_end=True to trim trailing "
            "elements."
        )

    nbin = shape // factors
    newshape = tuple(
        int(item) for pair in zip(nbin, factors, strict=True) for item in pair
    )
    reshaped = arr.reshape(newshape)
    return binfunc(reshaped, axis=tuple(range(1, reshaped.ndim, 2)))


# FIXME: I am not sure whether these gain conversions are universal or just
# for ASI cameras...
def dB2epadu(gain_dB: float) -> float:
    """Convert gain from decibels to electron/ADU."""
    return 5 / 10 ** (gain_dB / 20)


def epadu2dB(gain_epadu: float) -> float:
    """Convert gain from electron/ADU to decibels."""
    return 20 * np.log10(5 / gain_epadu)
