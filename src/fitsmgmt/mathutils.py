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
