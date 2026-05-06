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
    # Weighted mean and standard error
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
    """Find quantiles for lower and higher values

    Parameters
    ----------
    a : `~numpy.ndarray`

    lq, hq : array_like of `float`
        Quantile or sequence of quantiles to compute, which must be between 0
        and 1 inclusive.

    axis : {`int`, `tuple` of `int`, `None`}, optional
        Axis or axes along which the quantiles are computed. The default is to
        compute the quantile(s) along a flattened version of the array.

    nanfunc : `bool`, optional.
        Whether to use `~np.nanquantile` instead of `~np.qualtile`.
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
    """Extract "sigma" (std. dev.) from quantile to avoid bad values."""
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
    """Return minimum, maximum and median of array."""
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
    """Return mean and standard deviation of array."""
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


def binning(
    arr,
    factor_x=None,
    factor_y=None,
    factors=None,
    order_xyz=True,
    binfunc=np.mean,
    trim_end=False,
):
    """Bins the given arr frame.

    Parameters
    ---------
    arr: 2d array
        The array to be binned

    factor_x, factor_y: `int` or `None`, optional.
        The binning factors in x, y direction. This is left as legacy and for
        clarity, because mostly this function is used for 2-D CCD data. If any
        of these is given, `order_xyz` is overridden as `True`.

    factors : `list`-like of `int`, optional.
        The factors in pythonic axis order (``order_xyz=False``) or in the xyz
        order (``order_xyz=True``). If any of the `tuple` is `None`, that will be
        replaced by the size of the array along that axis, i.e., collapse along
        that axis.
        Default: `None`.

    binfunc : funciton object, optional.
        The function to be applied for binning, such as ``np.sum``,
        ``np.mean``, and ``np.median``.
        Default: ``np.mean``.

    trim_end : `bool`, optional.
        Whether to trim the end of x, y axes such that binning is done without
        error.
        Default: `False`.

    Notes
    -----
    This kind of binning is ~ 20-30 to upto 10^5 times faster than
    astropy.nddata's block_reduce:


    >>> from astropy.nddata.blocks import block_reduce
    >>> import fitsmgmt as fm
    >>> from astropy.nddata import CCDData
    >>> import numpy as np
    >>> ccd = CCDData(data=np.arange(1000).reshape(20, 50), unit='adu')
    >>> kw = dict(factor_x=5, factor_y=5, binfunc=np.sum, trim_end=True)
    >>> %timeit fm.binning(ccd.data, **kw)
    >>> # 10.9 +- 0.216 us (7 runs, 100000 loops each)
    >>> %timeit fm.bin_ccd(ccd, **kw, update_header=False)
    >>> # 32.9 µs +- 878 ns per loop (7 runs, 10000 loops each)
    >>> %timeit -r 1 -n 1 block_reduce(ccd, block_size=5)
    >>> # 518 ms, 2.13 ms, 250 us, 252 us, 257 us, 267 us
    >>> # 5.e+5   ...      ...     ...     ...     27  -- times slower
    >>> # some strange chaching happens?
    Tested on MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16
    GB (2400MHz DDR4), Radeon Pro 560X (4GB)]
    """
    # def binning(arr, factor_x=1, factor_y=1, binfunc=np.mean, trim_end=False):
    #     binned = arr.copy()
    #     if trim_end:
    #         ny_orig, nx_orig = binned.shape
    #         iy_max = ny_orig - (ny_orig % factor_y)
    #         ix_max = nx_orig - (nx_orig % factor_x)
    #         binned = binned[:iy_max, :ix_max]
    #     ny, nx = binned.shape
    #     nby = ny // factor_y
    #     nbx = nx // factor_x
    #     binned = binned.reshape(nby, factor_y, nbx, factor_x)
    #     binned = binfunc(binned, axis=(-1, 1))
    #     return binned

    binned = arr.copy()

    if factor_x is not None or factor_y is not None:
        factors = (factor_x, factor_y)
        order_xyz = True

    if factors is None:
        factors = np.ones(arr.ndim)
    else:
        factors = np.array(factors).ravel()
        for i, f in enumerate(factors):
            if f is None:
                factors[i] = arr.shape[i]

    if order_xyz:
        factors = factors[::-1]  # convert back to python order

    if trim_end:
        n_orig = binned.shape
        i_max = n_orig - (n_orig % factors)
        slices = tuple(slice(None, im, None) for im in i_max)
        binned = binned[slices]

    npix = binned.shape
    nbin = npix // factors
    nbin[nbin == 0] = 1
    newshape = []
    for nbin_i, factor_i in zip(nbin, factors):
        newshape.append(nbin_i)
        newshape.append(factor_i)

    binned = binned.reshape(newshape)
    funcaxis = np.arange(1, binned.ndim + 1, 2).astype(int)
    binned = binfunc(binned, axis=tuple(funcaxis))
    return binned


# FIXME: I am not sure whether these gain conversions are universal or just
# for ASI cameras...
def dB2epadu(gain_dB: float) -> float:
    return 5 / 10 ** (gain_dB / 20)


def epadu2dB(gain_epadu: float) -> float:
    return 20 * np.log10(5 / gain_epadu)
