import numpy as np
import pandas as pd
from astro_ndslice import listify, slicefy
from astropy import units as u
from astropy.nddata import CCDData
from astropy.time import Time

from astroimred.imops.ccdutils import (
    CCDData_astype,
    imslice,
)
from astroimred.mgmt.io import (
    load_ccd,
    _parse_image,
)
from astroimred.imops.pixels import fixpix
from astroimred.mgmt.headers import cmt2hdr, hdrval, update_process, update_tlm
from astroimred.mgmt.misc import (
    change_to_quantity,
)
from astroimred.mgmt.logging import logger

from .crrej import LACOSMIC_CRREJ, crrej, medfilt_bpm

__all__ = [
    "crrej",
    "medfilt_bpm",
    "biascor",
    "darkcor",
    "flatcor",
    "frincor",
    "ccdred",
    "run_reduc_plan",
]


def _addfrm(ccd, name, path):
    ccd.header[f"{name.upper()[:4]}FRM"] = (path, f"Applied {name.upper()} frame")


def scancor(
    ccd,
    overscan=None,
    scansec=None,
    scanax=0,
    fit_func="legendre",
    fit_order=1,
    fit_kw=None,
):
    """Do overscan correction

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The `~astropy.nddata.CCDData` to be corrected.

    overscan : `~numpy.ndarray`, optional.
        The overscan region in `~numpy.ndarray`, e.g., ``ccd.data[:, :overscan]``. One
        and only one of `overscan` or `scansec` should be given.
        Default: `None`.

    scansec : `str`, optional.
        The section of the overscan region to be used for correction, e.g.,
        "[1:10, :]" in FITS section format. One and only one of `overscan` or
        `scansec` should be given.
        Default: `None`.

    scanax : `int`, `None`, optional.
        Axis along which overscan should combined with mean or median. Axis
        numbering follows the *python* convention for ordering, so 0 is the
        first axis and 1 is the second axis.

        If overscan_axis is explicitly set to `None`, the axis is set to
        the shortest dimension of the overscan section (or 1 in case
        of a square overscan).
        Default is ``0``.

    fit_func : `str`, optional.
        The fitting function to use for overscan fitting.
        Default is ``"legendre"``.

    fit_order : `int`, optional.
        The order of the fitting polynomial.
        Default is ``1``.

    fit_kw : `dict`, optional.
        Keyword arguments passed to the fitting function.
        Default is ``dict(sigma=(3, 3), maxiters=1, grow=0)``.
    """
    raise NotImplementedError("scancor is not implemented yet.")
    # if fit_kw is None:
    #     fit_kw = dict(sigma=(3, 3), maxiters=1, grow=0)
    # pass


def biascor(ccd, mbias=None, mbiaspath=None, copy=True, verbose=1):
    """Do bias correction (purpose: helper function of `~astroimred.reduction.preproc.ccdred`)

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The `~astropy.nddata.CCDData` to be corrected.
    mbias : `~astropy.nddata.CCDData`, `~numpy.ndarray`, optional
        The master calibration (bias) frame.
        Default: `None`.
    mbiaspath : path-like, optional
        The path to the master calibration (bias) frame.
        Default: `None`.
    copy : `bool`, optional
        Whether to return a copy of the data (`True`) or a reference to the
        original data (`False`). Using `False` will be slightly faster (few ms
        order) and memory efficient, but the original data may be modified
        unintentionally.
        Default is `True`.
    verbose : `int`, optional
        Verbosity level for header logging.

    Returns
    -------
    `~astropy.nddata.CCDData`
        Bias-corrected CCD.
    """
    if mbias is None and mbiaspath is None:
        return ccd.copy() if copy else ccd

    _t = Time.now()
    nccd = ccd.copy() if copy else ccd
    mbias, mbiasname, _ = _parse_image(mbias or mbiaspath, name=mbiaspath)
    # For BIAS, header information is not needed at all... I guess?
    nccd.data = nccd.data - mbias
    _addfrm(nccd, "BIAS", mbiasname)
    cmt2hdr(
        nccd.header,
        "h",
        verbose=verbose >= 1,
        t_ref=_t,
        s=f"[biascor] Bias subtracted (BIASFRM = {mbiasname})",
    )
    update_process(nccd.header, "B")
    return nccd


def darkcor(
    ccd,
    mdark=None,
    mdarkpath=None,
    exptime_key="EXPTIME",
    exptime_data=None,
    exptime_dark=None,
    dark_scale=False,
    copy=True,
    verbose=1,
):
    """Do dark correction (purpose: helper function of `~astroimred.reduction.preproc.ccdred`)

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The `~astropy.nddata.CCDData` to be corrected.
    mdark : `~astropy.nddata.CCDData`, `~numpy.ndarray`, optional
        The master calibration (dark) frame.
        Default: `None`.
    mdarkpath : path-like, optional
        The path to the master calibration (dark) frame.
        Default: `None`.
    exptime_key : `str`, optional
        Header keyword for exposure time. Used only if `dark_scale` is `True`.
    exptime_data, exptime_dark : numeric, optional
        The exposure time of the data and the dark frame in the same unit. If
        `None`, ``exptime = header.get(exptime_key, 1)`` is used for data and
        dark, respectively. Otherwise, header information is ignored.
        Ignored if `dark_scale` is `False`.
        Default: `None`
    dark_scale : `bool`, optional
        Whether to scale dark frame. If `True`,
        ``scale = exptime_data/exptime_dark`` is multiplied to dark frame.
        Default: `False`
    copy : `bool`, optional
        Whether to return a copy of the data (`True`) or a reference to the
        original data (`False`). Using `False` will be slightly faster (few ms
        order) and memory efficient, but the original data may be modified
        unintentionally.
        Default is `True`.
    verbose : `int`, optional
        Verbosity level for header logging.

    Returns
    -------
    `~astropy.nddata.CCDData`
        Dark-corrected CCD.
    """

    if mdark is None and mdarkpath is None:
        return ccd.copy() if copy else ccd

    _t = Time.now()
    nccd = ccd.copy() if copy else ccd
    use_ccddata = dark_scale and exptime_dark is None
    mdark, mdarkname, _ = _parse_image(mdark or mdarkpath, name=mdarkpath, force_ccddata=use_ccddata)

    if dark_scale:
        exptime_data = exptime_data or ccd.header.get(exptime_key, None)
        exptime_dark = exptime_dark or mdark.header.get(exptime_key, None)

        msg = "[darkcor] Dark scaled by exptime: "
        if exptime_data is None or exptime_dark is None:
            logger.warning(
                "exptime_data=%s, exptime_dark=%s. Fix scale=1.",
                exptime_data,
                exptime_dark,
            )
            scale = 1
            msg += "Fixed scale=1 (metadata missing)."
        else:
            scale = exptime_data / exptime_dark
            msg += f"({exptime_data:.3f}/{exptime_dark:.3f}) = {scale:.3f}"

        mdark = mdark.data * scale if use_ccddata else mdark * scale
        # ^ mdark is now ndarray regardless of use_ccddata
        cmt2hdr(
            ccd.header,
            "h",
            verbose=verbose >= 1,
            s=msg,
        )
    nccd.data = nccd.data - mdark
    _addfrm(nccd, "DARK", mdarkname)
    cmt2hdr(
        nccd.header,
        "h",
        verbose=verbose >= 1,
        t_ref=_t,
        s=f"[darkcor] Dark subtracted (DARKFRM = {mdarkname})",
    )
    update_process(nccd.header, "D")
    return nccd


# add flat_norm_value
def flatcor(
    ccd,
    mflat=None,
    mflatpath=None,
    flat_mask=0,
    flat_fill=1,
    copy=True,
    flat_norm_value=1,
    verbose=1,
):
    """Do flat correction (purpose: helper function of `~astroimred.reduction.preproc.ccdred`)

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The `~astropy.nddata.CCDData` to be corrected.
    mflat : `~astropy.nddata.CCDData`, `~numpy.ndarray`, optional
        The master calibration (flat) frame.
        Default: `None`.
    mflatpath : path-like, optional
        The path to the master calibration (flat) frame.
        Default: `None`.
    flat_mask : numeric, `~numpy.ndarray`, `None`, optional
        Mask to replace bad flat pixels by ``mflat[flat_mask] = flat_fill``. If
        numeric, ``mflat[mflat < flat_mask] = flat_fill``. Skipped if `None`.
        Default: ``0``
    flat_fill : numeric, optional
        The value to fill the masked pixels.
        Default: ``1``.
    copy : `bool`, optional
        Whether to return a copy of the data (`True`) or a reference to the
        original data (`False`). Using `False` will be slightly faster (few ms
        order) and memory efficient, but the original data may be modified
        unintentionally.
        Default is `True`.
    flat_norm_value : numeric, optional
        If `None`, normalize the flat by its mean. Otherwise divide the flat by
        this value before correction.
        Default: ``1``.
    verbose : `int`, optional
        Verbosity level for header logging.

    Returns
    -------
    `~astropy.nddata.CCDData`
        Flat-corrected CCD.
    """
    if mflat is None and mflatpath is None:
        return ccd.copy() if copy else ccd

    _t = Time.now()
    nccd = ccd.copy() if copy else ccd
    mflat, mflatname, _ = _parse_image(mflat or mflatpath, name=mflatpath, force_ccddata=False)
    # For FLAT, header information is not needed at all... I guess?
    if flat_mask is not None:
        if isinstance(flat_mask, np.ndarray):
            maskstr = "Flat pixels with `value < flat_mask (User-provided ndarray)`"
        else:
            maskstr = f"Flat pixels with `value < {flat_mask = }`"
            flat_mask = mflat < flat_mask
        mflat[flat_mask] = flat_fill
        cmt2hdr(
            nccd.header,
            "h",
            verbose=verbose >= 1,
            s=(f"[flatcor] {maskstr} are replaced by `{flat_fill = }`."),
        )

    if flat_norm_value is None:
        mflat /= np.mean(mflat)
        cmt2hdr(
            nccd.header,
            "h",
            verbose=verbose >= 1,
            s=("[flatcor] Flat normalized by its mean."),
        )
    elif float(flat_norm_value) != 1.0:
        mflat /= float(flat_norm_value)
        cmt2hdr(
            nccd.header,
            "h",
            verbose=verbose >= 1,
            s=(f"[flatcor] Flat divided by {flat_norm_value = }."),
        )

    nccd.data = nccd.data / mflat
    _addfrm(nccd, "FLAT", mflatname)
    cmt2hdr(
        nccd.header,
        "h",
        verbose=verbose >= 1,
        t_ref=_t,
        s=f"[flatcor] Flat corrected (FLATFRM = {mflatname})",
    )
    update_process(nccd.header, "F")

    return nccd


def frincor(
    ccd,
    mfrin,
    mfrinpath=None,
    fringe_scale=None,
    fringe_scale_region=None,
    fringe_scale_kw=None,
    exptime_key="EXPTIME",
    exptime_data=None,
    exptime_frin=None,
    copy=True,
    verbose=1,
):
    """Subtract fringe frame

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The ccd to processed.

    mfringe : `~astropy.nddata.CCDData`
        The fringe frame.

    fringe_scale : `int`, `float`, ndarray, function object, ``{"exp", "exposure", "exptime"}``, optional.
        The scale to be applied to the fringe frame. If numeric or `~numpy.ndarray`, it
        will directly be multiplied to the fringe before fringe subtraction. If
        function object, it will be applied to the fringe before fringe
        subtraction (using `fringe_scale_section`). If "exp", "exposure", or
        "exptime", the exposure time of the fringe frame will be used. (using
        either `frin_exposure` or `exptime_key`). If `None`, the fringe
        will be subtracted without modification.
        Default: `None`.

    fringe_scale_region : `~numpy.ndarray`(`bool`), `str`, [`list` of] `int`, [`list` of] slice, optional.
        The mask or FITS-convention section of the fringe and object (science)
        frames to match the fringe pattern before the subtraction. If `~numpy.ndarray`,
        it will be forced to be changed into `bool` array. The scale will be
        ``fringe_scale(object_frame[fringe_scale_region]) /
        fringe_scale(frin_frame[fringe_scale_region])``.
        Default: `None`.

    fringe_scale_kw : `dict`, optional.
        The kwargs that can be passed to `fringe_scale` if it is a function.
        Default: ``{}``.

    exptime_key : `str`, optional.
        The header keyword for exposure time. Used only if `fringe_scale` is in
        ``{"exp", "exposure", "exptime"}``.

    exptime_data, exptime_frin : `None`, numeric, optional.
        The exposure time of the data and the fringe frame in the same unit. If
        `None`, ``exptime = header.get(exptime_key, 1)`` is used for data and
        fringe, respectively. Otherwise, header information is ignored.
        Used only if when `fringe_scale` is exposure time mode.
        Default: `None`

    copy : `bool`, optional
        Whether to return a copy of the data (`True`) or a reference to the
        original data (`False`). Using `False` will be slightly faster (few ms
        order) and memory efficient, but the original data may be modified
        unintentionally.
        Default is `True`.
    """
    if mfrin is None and mfrinpath is None:
        return ccd.copy() if copy else ccd
    if fringe_scale_kw is None:
        fringe_scale_kw = {}

    def _str(_ccd, frm, sec=None, fun=None, scal=None):
        str1 = f"[frincor] Fringe subtracted (FRINFRM = {frm})"
        _ccd.header["FRINFRM"] = (frm, "Fringe frame")
        noscal = scal is None
        nosec = sec is None
        nofun = fun is None
        if noscal and nofun and nosec:
            return str1

        str2 = "[frincor] IMAGE - FRINSCAL*FRINFRM "
        elems = []
        if not noscal:  # scal is not None
            _ccd.header["FRINSCAL"] = (scal, "Scale FRINFUNC(FRINFRM[FRINSECT])")
            elems.append(f"`FRINSCAL = {scal}`")
        if not nofun:
            _ccd.header["FRINFUNC"] = (fun, "Function used to get FRINSCAL")
            elems.append(f"`FRINFUNC = {fun}`")
        if not nosec:
            _ccd.header["FRINSECT"] = (sec, "The region used to get FRINSCAL")
            elems.append(f"`FRINSECT = {sec}`")
        return [str1, str2, ",".join(elems)]

    _t = Time.now()
    nccd = ccd.copy() if copy else ccd

    mfrin, mfrinname, _ = _parse_image(mfrin, name=mfrinpath, force_ccddata=True)
    #                                                         ^^^^^^^^^^^^^^^^^^
    # Converting an ndarray to CCDData (or vice versa) is very quick, so just
    # force CCDData for the sake of simplicity.

    if fringe_scale is None:
        nccd.data -= mfrin.data
        infostr = _str(nccd, mfrinname)
    elif isinstance(fringe_scale, (int, float)):
        nccd.data -= fringe_scale * mfrin.data
        infostr = _str(
            nccd, mfrinname, fun=type(fringe_scale).__name__, scal=fringe_scale
        )
    elif isinstance(fringe_scale, np.ndarray):
        nccd.data -= fringe_scale * mfrin.data
        infostr = _str(nccd, mfrinname, fun=f"{type(fringe_scale).__name__}")
    elif isinstance(fringe_scale, str):
        if fringe_scale.lower() in ["exp", "exposure", "exptime"]:
            exptime_data = exptime_data or nccd.header.get(exptime_key, 1)
            exptime_frin = exptime_frin or mfrin.header.get(exptime_key, 1)
            scale = exptime_data / exptime_frin
            nccd.data -= scale * mfrin.data
            infostr = _str(nccd, mfrinname, fun="EXPTIME", scal=scale)
        else:
            raise ValueError(
                f'`{fringe_scale=}` not in {{"exp", "exposure", "exptime"}}.'
            )
    else:  # Function
        if isinstance(fringe_scale_region, str):
            reg = slicefy(fringe_scale_region)
            sec = fringe_scale_region
        elif isinstance(fringe_scale_region, np.ndarray):
            reg = fringe_scale_region.astype(bool)
            sec = "User-provided mask"
        else:
            reg = None  # All
            sec = None
        scale = fringe_scale(ccd.data[reg] / mfrin.data[reg], **fringe_scale_kw)
        nccd.data -= scale * mfrin.data
        infostr = _str(
            nccd,
            mfrinname,
            fun=f"{fringe_scale.__name__} with {fringe_scale_kw}",
            sec=sec,
            scal=scale,
        )

    # FRINSCAL=FRINFUNC(FRINFRM[FRINSECT])
    _addfrm(nccd, "FRIN", mfrinname)
    cmt2hdr(ccd.header, "h", verbose=verbose, t_ref=_t, s=infostr)
    update_process(nccd.header, "R")

    return nccd


def illumcor(
    ccd,
):
    """Apply illumination correction.

    This placeholder is reserved for a future implementation.
    """
    raise NotImplementedError("illumcor is not implemented yet.")


# TODO: add overscan
# TODO: add normalization (e.g., `normalize` = {"mean", "median", "mode",
# "sum", "exptime", })
def ccdred(
    ccd,
    output: str | None = None,
    extension: int | str | None = None,
    trimsec: str | None = None,
    mbiaspath: str | None = None,
    mdarkpath: str | None = None,
    mflatpath: str | None = None,
    mfrinpath: str | None = None,
    mbias: CCDData | None = None,
    mdark: CCDData | None = None,
    mflat: CCDData | None = None,
    mfrin: CCDData | None = None,
    fringe_flat_fielded: bool = True,
    fringe_scale=None,
    fringe_scale_region: str | None = None,
    fringe_scale_kw: dict | None = None,
    gain: float | u.Unit = 1,
    gain_key: str = "GAIN",
    gain_unit: u.Unit = u.electron / u.adu,
    rdnoise: float | u.Unit = 0,
    rdnoise_key: str = "RDNOISE",
    rdnoise_unit: u.Unit = u.electron,
    exptime_key: str = "EXPTIME",
    exptime_frin: float | None = None,
    exptime_dark: float | None = None,
    exptime_data: float | None = None,
    dark_scale: bool = False,
    pixel_min: float | None = None,
    pixel_min_fill: float = 0,
    pixel_max: float | None = None,
    pixel_max_fill: float = 65535,
    flat_mask: float | int = 0,
    flat_fill: float = 1,
    flat_norm_value: float = 1,
    do_crrej: bool = False,
    crrej_kw: dict | None = None,
    propagate_crmask: bool = False,
    verbose_crrej: bool = False,
    verbose_bdf: int = 1,
    output_verify: str = "fix",
    overwrite: bool = True,
    dtype: str = "float32",
):
    """Do basic CCD reduction.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, or number-like
        The ccd to be processed.

    output : path-like or `None`, optional.
        The path if you want to save the resulting `ccd` object.
        Default: `None`.

    extension : `int`, `str`, or `None`, optional.
        The extension of the FITS file to be processed. If `None`, the first
        extension is used. Default: `None`.

    trimsec: `str`, optional.
        Region of `ccd` to be trimmed; see `~ccdproc.subtract_overscan` for
        details. Default is `None`.

    mbiaspath, mdarkpath, mflatpath, mfrinpath : path-like, optional.
        The path to master bias, dark, flat, and fringe FITS files. If `None`,
        the corresponding process is not done. These can be provided in
        addition to `mbias`, `mdark`, `mflat`, and/or `mfrin`.

    mbias, mdark, mflat, mfrin : `~astropy.nddata.CCDData`, optional.
        The master bias, dark, and flat in `~astropy.nddata.CCDData`. If this
        is given, the files provided by `mbiaspath`, `mdarkpath`, `mflatpath`
        and/or `mfrin` are **not** loaded, but these paths will be used for
        header (``BIASFRM``, ``DARKFRM``, ``FLATFRM`` and/or ``FRINFRM``). If
        the paths are not given, ``xxxxFRM`` will be ``<User>``.

    fringe_scale : `int`, `float`, ndarray, function object, ``{"exp", "exposure", "exptime"}``, optional.
        The scale to be applied to the fringe frame. If numeric or `~numpy.ndarray`, it
        will directly be multiplied to the fringe before fringe subtraction. If
        function object, it will be applied to the fringe before fringe
        subtraction (using `fringe_scale_section`). If "exp", "exposure", or
        "exptime", the exposure time of the fringe frame will be used. (using
        either `exptime_frin` or `exptime_key`). If `None`, the fringe
        will be subtracted without modification.
        Default: `None`.

    fringe_scale_region : `~numpy.ndarray`(`bool`), `str`, optional.
        The mask or FITS-convention section of the fringe and object (science)
        frames to match the fringe pattern before the subtraction. If `~numpy.ndarray`,
        it will be forced to be changed into `bool` array. The scale will be
        ``fringe_scale(object_frame[fringe_scale_region]) /
        fringe_scale(fringe_frame[fringe_scale_region])``.
        Default: `None`.

    fringe_scale_kw : `dict`, optional.
        The kwargs that can be passed to `fringe_scale` if it is a function.

    fringe_flat_fielded : `bool`, optional.
        Whether the fringe frame is flat-fielded. If `True`, fringe is
        subtracted AFTER flat-fielding the input frame. Otherwise (default),
        fringe is subtracted BEFORE flat-fielding the input frame.

    calc_err : `bool`, optional.
        Whether to calculate the error map based on Poisson and readnoise error
        propagation.

        .. note::
            Currently it's encouraged to make error-map manually, as the API is
            not stable.

    unit : `~astropy.units.Unit` or `str`, optional.
        The units of the data.
        Default is `None`.

    gain, rdnoise : `None`, `float`, astropy.`~astropy.units.Quantity`, optional.
        The gain and readnoise value. These are not used if `do_crrej` is
        `False`. If `gain` or `readnoise` is specified, they are interpreted
        with `gain_unit` and `rdnoise_unit`, respectively. If they are not
        specified, this function will seek for the header with keywords of
        `gain_key` and `rdnoise_key`, and interprete the header value in the
        unit of `gain_unit` and `rdnoise_unit`, respectively.

    gain_key, rdnoise_key : `str`, optional.
        See `gain`, `rdnoise` explanation above.
        These are not used if ``do_crrej=False``.

    gain_unit, rdnoise_unit : `str`, astropy.`~astropy.units.Unit`, optional.
        See `gain`, `rdnoise` explanation above.
        These are not used if ``do_crrej=False``.

    dark_exposure, data_exposure : `None`, `float`, astropy `~astropy.units.Quantity`, optional.
        The exposure times of dark and data frame, respectively. They should
        both be specified or both `None`. These are not used if
        ``mdarkpath=None``. If both are not specified while `mdarkpath` is
        given, then the code automatically seeks for header's `exposure_key`.
        Then interprete the value as the quantity with unit `exposure_unit`. If
        `mdkarpath` is not `None`, then these are passed to
        `~ccdproc.subtract_dark`.

    exposure_key : `str`, optional.
        The header keyword for exposure time.

    exposure_unit : astropy `~astropy.units.Unit`, optional.
        The unit of the exposure time.
        Used in `~ccdproc.subtract_dark`.

    normalize_exposure : `bool`, optional.
        Whether to normalize the values by the exposure time of each frame.
        Maybe useful for long exposure darks to make 1-sec darks.
        Default is `False`.

    normalize_average, normalize_median : `bool`, optional.
        Whether to normalize the values by the average or median value of each
        frame before combining. Only up to one of these must be `True`. Maybe
        useful for flat.
        Default is `False`.

    flat_min_value : `float` or `None`, optional.
        min_value of `ccdproc.flat_correct`. Minimum value for flat field. The
        value can either be `None` and no minimum value is applied to the flat or
        specified by a `float` which will replace all values in the flat by the
        min_value.
        Default is `None`.

    flat_norm_value : `float` or `None`, optional.
        The norm_value of `ccdproc.flat_correct`. If `None`, the flat is
        internally normalized by its mean before the flat correction, i.e., the
        flat correction will be like ``image/flat*mean(flat)``.
        If not `None`, the flat correction will be like
        ``image/flat*flat_norm_value``. Default is 1 (**different** from
        `ccdproc` which uses `None` as default).

    crrej_kwargs : `dict` or `None`, optional.
        If `None` (default), uses some default values (see `crrej`). It is
        always discouraged to use default except for quick validity-checking,
        because even the official L.A. Cosmic codes in different versions
        (IRAF, IDL, Python, etc) have different default parameters, i.e., there
        is nothing which can be regarded as *the default*. To see all possible
        keywords, do ``print(astroscrappy.detect_cosmics.__doc__)`` Also refer
        to
        https://nbviewer.jupyter.org/github/ysbach/AO2019/blob/master/Notebooks/07-Cosmic_Ray_Rejection.ipynb

    propagate_crmask : `bool`, optional.
        Whether to save (propagate) the mask from CR rejection (`astroscrappy`)
        to the CCD's mask. Default is `False`.

    output_verify : `str`
        Output verification option.  Must be one of ``"fix"``, ``"silentfix"``,
        ``"ignore"``, ``"warn"``, or ``"exception"``. May also be any
        combination of ``"fix"`` or ``"silentfix"`` with ``"+ignore"``,
        ``+warn``, or ``+exception" (e.g. ``"fix+warn"``).  See the astropy
        documentation below:
        http://docs.astropy.org/en/stable/io/fits/api/verification.html#verify

    dtype : `str` or `numpy.dtype` or `None`, optional.
        Allows user to set dtype. See `numpy.array` `dtype` parameter
        description. If `None` it uses ``np.float64``.
        Default is `None`.
    """

    # This reduction process will ignore `uncertainty` attribute of all
    # input/master calibration frames. This is because (1) speed matters more
    # than such an error calculation for cases when this simple generalized
    # function is used (2) such uncertainties are anyway not accurate in most
    # cases.
    def _load_master(path, master):
        if path is None and master is None:
            return False, None, None

        if path is not None and master is None:
            master = load_ccd(
                path, ccddata=False
            )  # because it will be forced to CCDData

        do = True
        master, imname, _ = _parse_image(master, name=path, force_ccddata=True)
        return do, master, imname

    # ************************************************************************************ #
    # *                                  INITIAL SETTING                                 * #
    # ************************************************************************************ #
    ccd, _, _ = _parse_image(ccd, extension=extension, force_ccddata=True)
    proc = ccd.copy()

    # == Set for BIAS ==================================================================== #
    do_b, mbias, mbiaspath = _load_master(mbiaspath, mbias)  # mbias in ndarray
    do_d, mdark, mdarkpath = _load_master(mdarkpath, mdark)  # mdark in ndarray
    do_f, mflat, mflatpath = _load_master(mflatpath, mflat)  # mflat in ndarray
    do_r, mfrin, mfrinpath = _load_master(mfrinpath, mfrin)  # mfrin in ndarray

    # ************************************************************************************ #
    # *                                 RUN PREPROCESSING                                * #
    # ************************************************************************************ #
    # == Do TRIM ========================================================================= #
    if trimsec is not None:
        sect = dict(trimsec=trimsec, fill_value=None, update_header=False)
        proc = imslice(proc, trimsec=trimsec, fill_value=None)  # update header
        mbias = imslice(mbias, **sect) if do_b else None
        mdark = imslice(mdark, **sect) if do_d else None
        mflat = imslice(mflat, **sect) if do_f else None
        mfrin = imslice(mfrin, **sect) if do_r else None

    prockw = dict(copy=False, verbose=verbose_bdf)
    # == Do BIAS ========================================================================= #
    if do_b:
        proc = biascor(proc, mbias=mbias, mbiaspath=mbiaspath, **prockw)

    # == Do DARK ========================================================================= #
    if do_d:
        proc = darkcor(
            proc,
            mdark=mdark,
            mdarkpath=mdarkpath,
            exptime_key=exptime_key,
            exptime_data=exptime_data,
            exptime_dark=exptime_dark,
            dark_scale=dark_scale,
            **prockw,
        )

    # == Do FRINGE **before** flat if not `fringe_flat_fielded` ========================== #
    if do_r and not fringe_flat_fielded:
        proc = frincor(
            proc,
            mfrin,
            mfrinpath=mfrinpath,
            fringe_scale=fringe_scale,
            fringe_scale_region=fringe_scale_region,
            fringe_scale_kw=fringe_scale_kw,
            exptime_key=exptime_key,
            exptime_data=exptime_data,
            exptime_frin=exptime_frin,
            **prockw,
        )

    # == Do FLAT ========================================================================= #
    if do_f:
        proc = flatcor(
            proc,
            mflat=mflat,
            mflatpath=mflatpath,
            flat_mask=flat_mask,
            flat_fill=flat_fill,
            flat_norm_value=flat_norm_value,
            **prockw,
        )

    # == Do FRINGE **after** flat if `fringe_flat_fielded` =============================== #
    if do_r and fringe_flat_fielded:
        proc = frincor(
            proc,
            mfrin,
            mfrinpath=mfrinpath,
            fringe_scale=fringe_scale,
            fringe_scale_region=fringe_scale_region,
            fringe_scale_kw=fringe_scale_kw,
            exptime_key=exptime_key,
            exptime_data=exptime_data,
            exptime_frin=exptime_frin,
            **prockw,
        )

    # == Do CRREJ ======================================================================== #
    if do_crrej:
        if crrej_kw is None:
            crrej_kw = LACOSMIC_CRREJ.copy()
            logger.warning("Using default CR-rejection parameters.")

        _proc = proc.header["PROCESS"]
        if (("B" in _proc) + ("D" in _proc) + ("F" in _proc)) < 2:
            logger.warning(
                "L.A. Cosmic should be run AFTER bias, dark, flat process. "
                "You have only done %s. "
                "See http://www.astro.yale.edu/dokkum/lacosmic/notes.html",
                proc.header["PROCESS"],
            )

        proc, _ = crrej(
            proc,
            propagate_crmask=propagate_crmask,
            update_header=True,
            gain=hdrval(gain, proc.header, gain_key, 1, unit=gain_unit),
            rdnoise=hdrval(rdnoise, proc.header, rdnoise_key, 0, unit=rdnoise_unit),
            verbose=verbose_crrej,
            **crrej_kw,
        )

    # ************************************************************************************ #
    # *                                  PREPARE OUTPUT                                  * #
    # ************************************************************************************ #
    # To avoid ``pssl`` in cr rejection, subtract fringe AFTER the CRREJ.
    if pixel_min is not None:
        proc.data[proc.data < pixel_min] = pixel_min_fill
    if pixel_max is not None:
        proc.data[proc.data > pixel_max] = pixel_max_fill
    proc = CCDData_astype(proc, dtype=dtype)
    update_tlm(proc.header)

    if output is not None:
        if verbose_bdf:
            logger.info("Writing FITS to %s...", output)
        proc.write(output, output_verify=output_verify, overwrite=overwrite)
        if verbose_bdf:
            logger.info("Saved.")
    return proc


def run_reduc_plan(
    plan,
    output=None,
    extension=None,
    col_file="file",
    col_bias="BIASFRM",
    col_dark="DARKFRM",
    col_flat="FLATFRM",
    col_mask="MASKFILE",
    col_fringe="FRINFRM",
    fixpix_kw=None,
    do_crrej=False,
    preload_cals=False,
    return_ccd=False,
    verbose=False,
):
    """Run `ccdred` for each row in a reduction plan.

    Parameters
    ----------
    plan : `~pandas.DataFrame`
        Table containing input files and calibration-frame columns.
    output : path-like or sequence of path-like, optional
        Output path for each row. Required unless ``return_ccd=True``.
    extension : int, str, or tuple, optional
        FITS extension to load.
    col_file, col_bias, col_dark, col_flat, col_mask, col_fringe : str, optional
        The column names for bias, dark, flat, mask, and fringe frames in
        `plan`. Default values follow IRAF convention.
    fixpix_kw, do_crrej : optional
        Options for bad-pixel fixing and cosmic-ray rejection.
    preload_cals, return_ccd, verbose : bool, optional
        Whether to preload calibration frames, return reduced CCDs, and log
        progress.

    Returns
    -------
    list of `~astropy.nddata.CCDData` or None
        Reduced CCDs if ``return_ccd=True``; otherwise writes outputs and
        returns `None`.
    """

    def _get_frms(df, col):
        if col not in df:
            return {}
        ccds = {}
        for fpath in df[col].unique():
            if pd.isna(fpath):
                continue
            ccds[fpath] = load_ccd(fpath)
        return ccds

    def _get_path(row, col):
        value = row.get(col)
        return None if pd.isna(value) else value

    if not isinstance(plan, pd.DataFrame):
        raise TypeError("plan must be a ~pandas.DataFrame.")

    if output is None and not return_ccd:
        raise ValueError(
            "No output file and return_ccd is False. "
            + "Nothing will be saved and nothing will be returned."
        )

    output = [None] * len(plan) if output is None else listify(output)

    if len(output) != len(plan):
        raise ValueError("output must have the same length as the plan.")
    if fixpix_kw is None:
        fixpix_kw = dict(priority=None, verbose=False)

    mbiass = _get_frms(plan, col_bias) if preload_cals else {}
    mdarks = _get_frms(plan, col_dark) if preload_cals else {}
    mflats = _get_frms(plan, col_flat) if preload_cals else {}
    mfrins = _get_frms(plan, col_fringe) if preload_cals else {}
    mmasks = _get_frms(plan, col_mask) if preload_cals else {}

    if return_ccd:
        ccds = []

    for (_, row), outpath in zip(plan.iterrows(), output):
        mbiaspath = _get_path(row, col_bias)  # either path or None
        mdarkpath = _get_path(row, col_dark)  # either path or None
        mflatpath = _get_path(row, col_flat)  # either path or None
        mfrinpath = _get_path(row, col_fringe)  # either path or None
        maskpath = _get_path(row, col_mask)
        ccd = ccdred(
            load_ccd(row[col_file]),
            extension=extension,
            mbias=mbiass.get(mbiaspath),  # = None if not preload_cals or mbiaspath=None
            mdark=mdarks.get(mdarkpath),  # (same as above)
            mflat=mflats.get(mflatpath),  # (same as above)
            mfrin=mfrins.get(mfrinpath),  # (same as above)
            mbiaspath=mbiaspath,
            mdarkpath=mdarkpath,
            mflatpath=mflatpath,
            mfrinpath=mfrinpath,
            verbose_bdf=verbose,
        )
        # ^ Better to load as CCDData here rather than pass filepath to
        #   `ccdred`, to avoid parsing overhead in `_parse_image`.

        ccd = fixpix(
            ccd,
            mmasks.get(maskpath),  # = None if not preload_cals or mmaskpath=None
            maskpath=maskpath,
            **fixpix_kw,
        )

        if do_crrej:
            ccd, _ = crrej(
                ccd,
                mask=mmasks.get(maskpath),
                gain=row.get("gain", LACOSMIC_CRREJ.get("gain")),
                rdnoise=row.get("rdnoise", LACOSMIC_CRREJ.get("rdnoise")),
                sigclip=row.get("sigclip", LACOSMIC_CRREJ.get("sigclip")),
                sigfrac=row.get("sigfrac", LACOSMIC_CRREJ.get("sigfrac")),
                objlim=row.get("objlim", LACOSMIC_CRREJ.get("objlim")),
                satlevel=row.get("satlevel", LACOSMIC_CRREJ.get("satlevel")),
                niter=row.get("niter", LACOSMIC_CRREJ.get("niter")),
                sepmed=row.get("sepmed", LACOSMIC_CRREJ.get("sepmed")),
                cleantype=row.get("cleantype", LACOSMIC_CRREJ.get("cleantype")),
                fs=row.get("fs", LACOSMIC_CRREJ.get("fs")),
                psffwhm=row.get("psffwhm", LACOSMIC_CRREJ.get("psffwhm")),
                psfsize=row.get("psfsize", LACOSMIC_CRREJ.get("psfsize")),
                psfbeta=row.get("psfbeta", LACOSMIC_CRREJ.get("psfbeta")),
                verbose=verbose,
            )

        if outpath is not None:
            ccd.write(outpath, overwrite=True, output_verify="fix")
        if return_ccd:
            ccds.append(ccd)

    if return_ccd:
        return ccds
