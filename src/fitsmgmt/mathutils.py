"""Standalone array statistics and uncertainty helpers."""

from os import PathLike

import bottleneck as bn
import numpy as np
from astro_ndslice import slicefy
from astropy import units as u
from astropy.io import fits
from astropy.stats import mad_std
from astropy.visualization import ZScaleInterval

from .logging import logger

try:
    import numexpr as ne

    HAS_NE = True
except ImportError:
    HAS_NE = False

__all__ = [
    "weighted_avg",
    "quantile_lh",
    "quantile_sigma",
    "min_max_med_1d",
    "mean_std_1d",
    "binning",
    "dB2epadu",
    "epadu2dB",
    "errormap",
    "give_stats",
]


def _data_header_from_array_or_path(item, extension=None):
    if isinstance(item, np.ndarray):
        return item, None
    if isinstance(item, (str, PathLike)):
        with fits.open(item) as hdul:
            data = hdul[extension if extension is not None else 0].data.copy()
            hdr = hdul[extension if extension is not None else 0].header.copy()
        return data, hdr
    raise TypeError(
        "mathutils helpers accept numpy.ndarray or path-like FITS inputs. "
        f"Received {type(item)}."
    )


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


def errormap(
    ccd_biassub,
    gain_epadu=1,
    rdnoise_electron=0,
    subtracted_dark=0.0,
    flat=1.0,
    dark_std=0.0,
    flat_err=0.0,
    dark_std_min="rdnoise",
    return_variance=False,
):
    """Calculate the detailed pixel-wise error map in ADU unit.

    ``ccd_biassub`` is now intentionally accepted as either `~numpy.ndarray` or
    path-like FITS input. For CCDData/HDU inputs, pass their `.data` explicitly.
    """
    data, _ = _data_header_from_array_or_path(ccd_biassub)
    data = np.array(data, copy=True)
    data[data < 0] = 0  # make all negative pixel to 0

    if isinstance(gain_epadu, u.Quantity):
        gain_epadu = gain_epadu.to(u.electron / u.adu).value
    elif isinstance(gain_epadu, str):
        gain_epadu = float(gain_epadu)

    if isinstance(rdnoise_electron, u.Quantity):
        rdnoise_electron = rdnoise_electron.to(u.electron).value
    elif isinstance(rdnoise_electron, str):
        rdnoise_electron = float(rdnoise_electron)

    if dark_std_min == "rdnoise":
        dark_std_min = rdnoise_electron / gain_epadu
    if isinstance(dark_std, np.ndarray):
        dark_std[dark_std < dark_std_min] = dark_std_min

    # Calculate the full variance map
    # restore dark for Poisson term calculation
    if HAS_NE:
        eval_str = (
            "(data + subtracted_dark)/(gain_epadu*flat**2)"
            "+ (dark_std/flat)**2"
            "+ data**2*(flat_err/flat)**2"
            "+ (rdnoise_electron/(gain_epadu*flat))**2"
        )
        if return_variance:
            return ne.evaluate(eval_str)
        else:  # Sqrt is the most time-consuming part...
            return ne.evaluate(f"sqrt({eval_str})")
    else:
        variance = (
            (data + subtracted_dark) / (gain_epadu * flat**2)
            + (dark_std / flat) ** 2
            + data**2 * (flat_err / flat) ** 2
            + (rdnoise_electron / (gain_epadu * flat)) ** 2
        )
        if return_variance:
            return variance
        else:
            return np.sqrt(variance)


# TODO: add sigma-clipped statistics option (hdr key can be using "SIGC", e.g., SIGCAVG.)
def give_stats(
    item,
    mask=None,
    extension=None,
    statsecs=None,
    percentiles=[1, 99],
    N_extrema=None,
    return_header=False,
):
    """Calculates simple statistics.

    ``item`` is now intentionally accepted as either `~numpy.ndarray` or
    path-like FITS input. For CCDData/HDU inputs, pass their `.data` explicitly.
    """
    data, hdr = _data_header_from_array_or_path(item, extension=extension)
    data = np.array(data, copy=True)
    if mask is not None:
        data[mask] = np.nan

    if statsecs is not None:
        statsecs = [statsecs] if isinstance(statsecs, str) else list(statsecs)
        data = np.array([data[slicefy(sec)] for sec in statsecs])

    data = data.ravel()
    data = data[np.isfinite(data)]

    minf = np.min
    maxf = np.max
    avgf = np.mean
    medf = bn.median  # Still median from bn seems faster!
    stdf = np.std
    pctf = np.percentile

    result = dict(
        num=np.size(data),
        min=minf(data),
        max=maxf(data),
        avg=avgf(data),
        med=medf(data),
        std=stdf(data, ddof=1),
        madstd=mad_std(data),
        percentiles=percentiles,
        pct=pctf(data, percentiles),
        slices=statsecs,
    )
    # d_pct = np.percentile(data, percentiles)
    # for i, pct in enumerate(percentiles):
    #     result[f"percentile_{round(pct, 4)}"] = d_pct[i]

    d_zmin, d_zmax = ZScaleInterval().get_limits(data)
    result["zmin"] = d_zmin
    result["zmax"] = d_zmax

    if N_extrema is not None:
        if 2 * N_extrema > result["num"]:
            logger.warning(
                "Extrema overlaps (2*N_extrema (%s) > N_pix (%s))",
                2 * N_extrema,
                result["num"],
            )
        data_flatten = np.sort(data, axis=None)  # axis=None will do flatten.
        d_los = data_flatten[:N_extrema]
        d_his = data_flatten[-1 * N_extrema :]
        result["ext_lo"] = d_los
        result["ext_hi"] = d_his

    if return_header and hdr is not None:
        hdr["STATNPIX"] = (result["num"], "Number of pixels used in statistics below")
        hdr["STATMIN"] = (result["min"], "Minimum value of the pixels")
        hdr["STATMAX"] = (result["max"], "Maximum value of the pixels")
        hdr["STATAVG"] = (result["avg"], "Average value of the pixels")
        hdr["STATMED"] = (result["med"], "Median value of the pixels")
        hdr["STATSTD"] = (
            result["std"],
            "Sample standard deviation value of the pixels",
        )
        hdr["STATMED"] = (result["zmin"], "Median value of the pixels")
        hdr["STATZMIN"] = (result["zmin"], "zscale minimum value of the pixels")
        hdr["STATZMAX"] = (result["zmax"], "zscale minimum value of the pixels")
        for i, p in enumerate(percentiles):
            hdr[f"PERCTS{i+1:02d}"] = (p, "The percentile used in STATPCii")
            hdr[f"STATPC{i+1:02d}"] = (result["pct"][i], "Percentile value at PERCTSii")

        if statsecs is not None:
            for i, sec in enumerate(statsecs):
                hdr[f"STATSEC{i+1:01d}"] = (sec, "Sections used for statistics")

        if N_extrema is not None:
            if N_extrema > 99:
                logger.warning("N_extrema > 99 may not work properly in header.")
            for i in range(N_extrema):
                hdr[f"STATLO{i+1:02d}"] = (
                    result["ext_lo"][i],
                    f"Lower extreme values (N_extrema={N_extrema})",
                )
                hdr[f"STATHI{i+1:02d}"] = (
                    result["ext_hi"][i],
                    f"Upper extreme values (N_extrema={N_extrema})",
                )
        return result, hdr
    return result
