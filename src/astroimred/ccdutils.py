"""CCDData manipulation helpers."""

from copy import deepcopy

import numpy as np
from astro_ndslice import calc_offset_physical, offseted_shape, slicefy
from astropy import units as u
from astropy.nddata import CCDData, Cutout2D
from astropy.time import Time
from astropy.wcs import WCS

from . import headers, io as _io
from .io import inputs2list
from .logging import logger
from .mathutils import binning

__all__ = [
    "CCDData_astype",
    "set_ccd_attribute",
    "set_ccd_gain_rdnoise",
    "propagate_ccdmask",
    "imslice",
    "trim_overlap",
    "cut_ccd",
    "bin_ccd",
    "convert_bit",
]


def _normalize_ccd_binning_factors(data_shape, factors):
    if factors is None:
        factors = (1,) * len(data_shape)
    raw_factors = tuple(np.asarray(factors, dtype=object).ravel())
    if len(raw_factors) != len(data_shape):
        raise ValueError(
            "bin_ccd factors must have the same length as ccd.data.ndim "
            f"({len(data_shape)}); got {len(raw_factors)}."
        )

    # binning() defaults to xyz-style factor order, so map None values against
    # the reversed NumPy shape for header reporting.
    xyz_shape = tuple(data_shape[::-1])
    effective = []
    for axis, (factor, axis_size) in enumerate(zip(raw_factors, xyz_shape)):
        if factor is None:
            effective.append(int(axis_size))
            continue
        if isinstance(factor, (bool, np.bool_)):
            raise ValueError(f"factor for axis {axis} must be a positive integer.")
        if not isinstance(factor, (int, np.integer)):
            raise ValueError(f"factor for axis {axis} must be a positive integer.")
        factor = int(factor)
        if factor < 1:
            raise ValueError(f"factor for axis {axis} must be a positive integer.")
        effective.append(factor)
    return raw_factors, tuple(effective)


def _update_binning_header(header, factors):
    if len(factors) in (2, 3):
        keys = ("XBINNING", "YBINNING", "ZBINNING")
        axis_names = ("X", "Y", "Z")
        for key, axis_name, factor in zip(keys, axis_names, factors, strict=False):
            header[key] = (
                factor,
                f"Binning done after the observation in {axis_name} direction",
            )
        return

    for idx, factor in enumerate(factors, start=1):
        header[f"BINNING{idx}"] = (
            factor,
            f"Binning factor {idx} applied after the observation",
        )


def CCDData_astype(ccd, dtype="float32", uncertainty_dtype=None, copy=True):
    """Assign dtype to the `~astropy.nddata.CCDData` object (numpy uses float64 default).

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The ccd to be astyped.

    dtype : dtype-like, optional.
        The dtype to be applied to the data
        Default: ``'float32'``.

    uncertainty_dtype : dtype-like, optional.
        The dtype to be applied to the uncertainty. Be default, use the same
        dtype as data (``uncertainty_dtype=dtype``).
        Default: `None`.

    Examples
    -------

    >>> from astropy.nddata import CCDData
    >>> import numpy as np
    >>> ccd = CCDData.read("image_unitygain001.fits", 0)
    >>> ccd.uncertainty = np.sqrt(ccd.data)
    >>> ccd = air.CCDData_astype(ccd, dtype='int16', uncertainty_dtype='float32')
    """
    if copy:
        nccd = ccd.copy()
    else:
        nccd = ccd
    nccd.data = nccd.data.astype(dtype)

    try:
        if uncertainty_dtype is None:
            uncertainty_dtype = dtype
        nccd.uncertainty.array = nccd.uncertainty.array.astype(uncertainty_dtype)
    except AttributeError:
        # If there is no uncertainty attribute in the input `ccd`
        pass

    headers.update_tlm(nccd.header)
    return nccd


def set_ccd_attribute(
    ccd,
    name,
    value=None,
    key=None,
    default=None,
    unit=None,
    header_comment=None,
    update_header=True,
    verbose=True,
    wrapper=None,
    wrapper_kw={},
):
    """Set CCDData attributes from explicit values or header keywords.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The ccd to add attribute.

    value : Any, optional.
        The value to be set as the attribute. If `None`, the
        ``ccd.header[key]`` will be searched.
        Default: `None`.

    name : `str`, optional.
        The name of the attribute.

    key : `str`, optional.
        The key in the ``ccd.header`` to be searched if ``value=None``.
        Default: `None`.

    unit : astropy.`~astropy.units.Unit`, optional.
        The unit that will be applied to the found value.
        Default: `None`.

    header_comment : `str`, optional.
        The comment string to the header if ``update_header=True``. If `None`
        (default), search for existing comment in the original header by
        ``ccd.comments[key]`` and only overwrite the value by
        ``ccd.header[key]=found_value``. If it's not `None`, the comments will
        also be overwritten if ``update_header=True``.
        Default: `None`.

    wrapper : function object, `None`, optional.
        The wrapper function that will be applied to the found value. Other
        keyword arguments should be given as a `dict` to `wrapper_kw`.
        Default: `None`.

    wrapper_kw : `dict`, optional.
        The keyword argument to `wrapper`.
        Default: ``{}``.

    Examples
    -------

    >>> set_ccd_attribute(ccd, 'gain', value=2, unit='electron/adu')
    >>> set_ccd_attribute(ccd, 'ra', key='RA', unit=u.deg, default=0)

    Notes
    -----
    """
    _t_start = Time.now()
    str_history = "From {}, {} = {} [unit = {}]"
    #                   value_from, name, value_Q.value, value_Q.unit

    if unit is None:
        try:
            unit = value.unit
        except AttributeError:
            unit = u.dimensionless_unscaled

    value_Q, value_from = headers.hdrval(
        value=value,
        header=ccd.header,
        key=key,
        unit=unit,
        verbose=verbose,
        default=default,
        return_source=True,
    )
    if wrapper is not None:
        value_Q = wrapper(value_Q, **wrapper_kw)

    if update_header:
        s = [str_history.format(value_from, name, value_Q.value, value_Q.unit)]
        if key is not None:
            if header_comment is None:
                try:
                    header_comment = ccd.header.comments[key]
                except (KeyError, ValueError):
                    header_comment = ""

            try:
                v = ccd.header[key]
                s.append(f"[air.set_ccd_attribute] (Original {key} = {v} is overwritten.)")

            except (KeyError, ValueError):
                pass

            ccd.header[key] = (value_Q.value, header_comment)
        # add as history
        headers.cmt2hdr(
            ccd.header,
            "h",
            s,
            t_ref=_t_start,
        )

    setattr(ccd, name, value_Q)
    headers.update_tlm(ccd.header)


def set_ccd_gain_rdnoise(
    ccd,
    verbose=True,
    update_header=True,
    gain=None,
    rdnoise=None,
    gain_key="GAIN",
    rdnoise_key="RDNOISE",
    gain_unit=u.electron / u.adu,
    rdnoise_unit=u.electron,
):
    """A convenience set_ccd_attribute for gain and readnoise.

    Parameters
    ----------
    gain, rdnoise : `None`, `float`, astropy.`~astropy.units.Quantity`, optional.
        The gain and readnoise value. If `gain` or `readnoise` is specified,
        they are interpreted with `gain_unit` and `rdnoise_unit`, respectively.
        If they are not specified, this function will seek for the header with
        keywords of `gain_key` and `rdnoise_key`, and interpret the header
        value in the unit of `gain_unit` and `rdnoise_unit`, respectively.

    gain_key, rdnoise_key : `str`, optional.
        See `gain`, `rdnoise` explanation above.

    gain_unit, rdnoise_unit : `str`, astropy.`~astropy.units.Unit`, optional.
        See `gain`, `rdnoise` explanation above.

    verbose : `bool`, optional.
        The verbose option.
        Default: `True`.

    update_header : `bool`, optional
        Whether to update the given header.
        Default: `True`.
    """
    gain_str = f"[{gain_unit:s}] Gain of the detector"
    rdn_str = f"[{rdnoise_unit:s}] Readout noise of the detector"
    set_ccd_attribute(
        ccd=ccd,
        name="gain",
        value=gain,
        key=gain_key,
        unit=gain_unit,
        default=1.0,
        header_comment=gain_str,
        update_header=update_header,
        verbose=verbose,
    )
    set_ccd_attribute(
        ccd=ccd,
        name="rdnoise",
        value=rdnoise,
        key=rdnoise_key,
        unit=rdnoise_unit,
        default=0.0,
        header_comment=rdn_str,
        update_header=update_header,
        verbose=verbose,
    )


def propagate_ccdmask(ccd, additional_mask=None):
    """Propagate the `~astropy.nddata.CCDData`'s mask and additional mask.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`, `~numpy.ndarray`
        The ccd to extract mask. If `~numpy.ndarray`, it will only return a copy of
        `additional_mask`.

    additional_mask : mask-like, `None`, optional.
        The mask to be propagated.
        Default: `None`.

    Notes
    -----
    The original ``ccd.mask`` is not modified. To do so,

    >>> ccd.mask = propagate_ccdmask(ccd, additional_mask=mask2)
    """
    if additional_mask is None:
        try:
            mask = ccd.mask.copy()
        except AttributeError:  # i.e., if ccd.mask is None
            mask = None
    else:
        try:
            mask = ccd.mask | additional_mask
        except (TypeError, AttributeError):  # i.e., if ccd.mask is None:
            mask = deepcopy(additional_mask)
    return mask


def imslice(
    ccd, trimsec, fill_value=None, order_xyz=True, update_header=True, verbose=False
):
    """Slice the `~astropy.nddata.CCDData` using one of trimsec, bezels, or slices.

    Parameters
    ---------
    ccd : `~astropy.nddata.CCDData`, `~numpy.ndarray`
        The ccd to be sliced. If `~numpy.ndarray`, it will be converted to `~astropy.nddata.CCDData` with
        dummy unit ("ADU").

    trimsec : `str`, `int`, `list` of `int`, `list` of slice, `None`, optional
        It can have several forms::

          * str: The FITS convention section to trim (e.g., IRAF TRIMSEC).
          * [list of] int: The number of pixels to trim from the edge of the
            image (bezel). If list, it must be [bezel_lower, bezel_upper].
          * [list of] slice: The slice of each axis (`slice(start, stop,
            step)`)

        If a single `int`/slice is given, it will be applied to all the axes.

    order_xyz : `bool`, optional
        Whether the order of trimsec is in xyz order. Works only if the
        `trimsec` is bezel-like (`int` or `list` of `int`). If it is slice-like,
        `trimsec` must be in the pythonic order (i.e., ``[slice_for_axis0,
        slice_for_axis1, ...]``).
        Default: `True`.

    fill_value : `None` or `float`-like, optional.
        If `None`, it removes the pixels outside of it. If given as `float`-like
        (including `np.nan`), the bezel pixels will be replaced with this
        value.
        Default: `None`.

    Notes
    -----
    Similar to ccdproc.trim_image or imcopy. Compared to ccdproc, it has
    flexibility, and can add LTV/LTM to header.

    """
    _t = Time.now()

    # Parse
    sl = slicefy(trimsec, ndim=ccd.ndim, order_xyz=order_xyz)

    if isinstance(ccd, np.ndarray):
        ccd = CCDData(ccd, unit=u.adu)

    if fill_value is None:
        nccd = ccd[sl].copy()  # CCDData supports this kind of slicing
    else:
        nccd = ccd.copy()
        nccd.data = np.ones(nccd.shape) * fill_value
        nccd.data[sl] = ccd.data[sl]

    if update_header:  # update LTV/LTM
        ltms = [1 if s.step is None else 1 / s.step for s in sl]
        ndim = ccd.ndim  # ndim == NAXIS keyword
        shape = ccd.shape
        if trimsec is not None:
            ltvs = []
            for axis_i_py, naxis_i in enumerate(shape):
                # example: "[10:110]", we must have LTV = -9, not -10.
                ltvs.append(-1 * sl[axis_i_py].indices(naxis_i)[0])
            ltvs = ltvs[::-1]  # zyx -> xyz order
        else:
            ltvs = [0.0] * ndim

        hdr = nccd.header
        for i, ltv in enumerate(ltvs):
            if (key := f"LTV{i+1}") in hdr:
                hdr[key] += ltv
            else:
                hdr[key] = ltv

        for i in range(ndim):
            for j in range(ndim):
                if i == j:
                    hdr[f"LTM{i+1}_{i+1}"] = hdr.get(f"LTM{i+1}", ltms[i])
                else:
                    hdr.setdefault(f"LTM{i+1}_{j+1}", 0.0)

        if trimsec is not None:
            infostr = [f"[air.imslice] Sliced using `{trimsec = }`: converted to {sl}. "]
            if fill_value is not None:
                infostr.append(f"Filled background with `{fill_value = }`.")
            headers.cmt2hdr(
                hdr,
                "h",
                infostr,
                t_ref=_t,
                verbose=verbose,
            )
            headers.update_process(hdr, "T")

    return nccd


def trim_overlap(inputs, extension=None, coordinate="image"):
    """Trim only the overlapping regions of the two CCDs

    Parameters
    ----------
    coordinate : `str`, optional.
        Ways to find the overlapping region. If ``'image'`` (default), output
        size will be ``np.min([ccd.shape for ccd in ccds], axis=0)``. If
        ``'physical'``, overlapping region will be found based on the physical
        coordinates.
        Default: ``'image'``.

    extension : `int`, `str`, (`str`, `int`), optional.
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.
        Default: `None`.

    Notes
    -----
    `~astropy.wcs.WCS` is not acceptable because no rotation/scaling is supported.
    """
    items = inputs2list(inputs, sort=False, accept_ccdlike=True, check_coherency=False)
    if len(items) < 2:
        raise ValueError("inputs must have at least 2 objects.")

    offsets = []
    shapes = []
    reference = _io._parse_image(
        items[0], extension=extension, name=None, force_ccddata=True
    )
    for item in items:
        ccd, _, _ = _io._parse_image(
            item, extension=extension, name=None, force_ccddata=True
        )
        shapes.append(ccd.data.shape)
        offsets.append(
            calc_offset_physical(ccd, reference, order_xyz=False, ignore_ltm=True)
        )

    offsets, new_shape = offseted_shape(
        shapes, offsets, method="overlap", intify_offsets=False
    )


def cut_ccd(ccd, position, size, wcs=None, mode="trim", fill_value=np.nan, warnings=True,
            update_header=True, verbose=0):
    """ Converts the Cutout2D object to proper CCDData.

    Parameters
    ----------
    ccd: CCDData
        The ccd to be trimmed.

    position : tuple or `~astropy.coordinates.SkyCoord`
        The position of the cutout array's center with respect to the ``data``
        array. The position can be specified either as a ``(x, y)`` tuple of
        pixel coordinates or a `~astropy.coordinates.SkyCoord`, in which case
        wcs is a required input.

    size : int, array-like, `~astropy.units.Quantity`
        The size of the cutout array along each axis. If `size` is a scalar
        number or a scalar `~astropy.units.Quantity`, then a square cutout of
        `size` will be created. If `size` has two elements, they should be in
        ``(ny, nx)`` order. Scalar numbers in `size` are assumed to be in units
        of pixels. `size` can also be a `~astropy.units.Quantity` object or
        contain `~astropy.units.Quantity` objects. Such
        `~astropy.units.Quantity` objects must be in pixel or angular units.
        For all cases, `size` will be converted to an integer number of pixels,
        rounding the the nearest integer. See the `mode` keyword for additional
        details on the final cutout size.

        .. note::
            If `size` is in angular units, the cutout size is converted to
            pixels using the pixel scales along each axis of the image at the
            ``CRPIX`` location.  Projection and other non-linear distortions
            are not taken into account.

    wcs : `~astropy.wcs.WCS`, optional
        A WCS object associated with the input `data` array.  If `wcs` is not
        `None`, then the returned cutout object will contain a copy of the
        updated WCS for the cutout data array.

    mode : {'trim', 'partial', 'strict'}, optional
        The mode used for creating the cutout data array.  For the
        ``'partial'`` and ``'trim'`` modes, a partial overlap of the cutout
        array and the input `data` array is sufficient. For the ``'strict'``
        mode, the cutout array has to be fully contained within the `data`
        array, otherwise an `~astropy.nddata.utils.PartialOverlapError` is
        raised.   In all modes, non-overlapping arrays will raise a
        `~astropy.nddata.utils.NoOverlapError`.  In ``'partial'`` mode,
        positions in the cutout array that do not overlap with the `data` array
        will be filled with `fill_value`.  In ``'trim'`` mode only the
        overlapping elements are returned, thus the resulting cutout array may
        be smaller than the requested `shape`.

    fill_value : number, optional
        If ``mode='partial'``, the value to fill pixels in the cutout array
        that do not overlap with the input `data`. `fill_value` must have the
        same `dtype` as the input `data` array.
    """
    cutout = Cutout2D(
        data=ccd.data,
        position=position,
        size=size,
        wcs=wcs or getattr(ccd, "wcs", WCS(ccd.header)),
        mode=mode,
        fill_value=fill_value,
        copy=True,
    )
    # Copy True just to avoid any contamination to the original ccd.

    # TODO: add mask/flags/uncertainty support
    nccd = CCDData(
        data=cutout.data,
        header=ccd.header.copy(),
        wcs=cutout.wcs,
        unit=ccd.unit
    )
    ny, nx = nccd.data.shape
    if update_header:
        nccd.header["NAXIS1"] = nx
        nccd.header["NAXIS2"] = ny
        nccd.header["LTV1"] = nccd.header.get("LTV1", 0) - cutout.origin_original[0]
        nccd.header["LTV2"] = nccd.header.get("LTV2", 0) - cutout.origin_original[1]

    if warnings:
        nonlin = False
        try:
            for ctype in ccd.wcs.get_axis_types():
                if ctype["scale"] != "linear":
                    nonlin = True
                    break
        except AttributeError:
            nonlin = False

        if nonlin:
            logger.warning(
                "Nonlinear WCS in `ccd.wcs.get_axis_types()`. "
                "This may result in slightly inaccurate WCS calculation."
            )

    return nccd, cutout


def bin_ccd(
    ccd,
    factors=None,
    binfunc=np.mean,
    trim_end=False,
    update_header=True,
    copy=True,
):
    """Bins the given ccd.

    Parameters
    ---------
    ccd : `~astropy.nddata.CCDData`
        The ccd to be binned

    factors : `list`-like of `int`, optional.
        The binning factors. The order matches
        ``mathutils.binning(..., order_xyz=True)``. For 2-D data this is
        ``(x, y)``; for 3-D data this is ``(x, y, z)``. If `None`, every
        factor is treated as ``1``.

    binfunc : callable, optional.
        The function to be applied for binning, such as ``np.sum``,
        ``np.mean``, and ``np.median``.
        Default: ``np.mean``.

    trim_end : `bool`, optional.
        Whether to trim the end of x, y axes such that binning is done without
        error.
        Default: `False`.

    update_header : `bool`, optional.
        Whether to update header. Defaults to `True`.
        Default: `True`.

    Notes
    -----
    This is ~ 20-30 to up to 10^5 times faster than astropy.nddata's
    block_reduce:

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
    _t_start = Time.now()

    if not isinstance(ccd, CCDData):
        raise TypeError("ccd must be CCDData object.")

    factors, header_factors = _normalize_ccd_binning_factors(ccd.data.shape, factors)

    if all(factor == 1 for factor in header_factors):
        return ccd

    if copy:
        _ccd = ccd.copy()
    else:
        _ccd = ccd

    _ccd.data = binning(
        _ccd.data,
        factors=factors,
        binfunc=binfunc,
        trim_end=trim_end,
    )
    if update_header:
        _ccd.header["BINFUNC"] = (binfunc.__name__, "The function used for binning.")
        _update_binning_header(_ccd.header, header_factors)
        # add as history
        headers.cmt2hdr(
            _ccd.header,
            "h",
            t_ref=_t_start,
            s=f"[air.bin_ccd] Binned by factors = {header_factors} ",
        )
    return _ccd


def convert_bit(
    ccd, original_bit=12, target_bit=16, dtype="int16", bunit=None, copy=True
):
    """Converts a FIT(S) file's bit.

    Parameters
    ----------
    ccd: `~astropy.nddata.CCDData`
        The `~astropy.nddata.CCDData` object to be converted.

    original_bit, target_bit: `int`
        The original and target bit of the `~astropy.nddata.CCDData` object. For example, if
        these are 12 and 16, respectively, the effect will be dividing the
        original data by 2^4 = 16.

    dtype : `str`, optional.
        The data type of the output `~astropy.nddata.CCDData` object.
        Default: ``'int16'``.

    bunit : `str`, optional.
        The unit of the output `~astropy.nddata.CCDData` object. Set it to `None` to keep the
        original ``"BUNIT"`` in the header.
        Default: `None`.

    Notes
    -----
    In ASI1600MM, for example, the output data is 12-bit but since FITS
    standard do not accept 12-bit (but the closest integer is 16-bit), so, for
    example, the pixel values can have 0 and 15, but not any integer between
    these two. So it is better to convert to 16-bit.
    """
    _t = Time.now()
    dscale = 2 ** (target_bit - original_bit)
    _ccd = ccd.copy() if copy else ccd
    _ccd.data = (_ccd.data / dscale).astype(dtype)
    _ccd.header["MAXDATA"] = (
        2**original_bit - 1,
        "maximum valid physical value in raw data",
    )
    if bunit is not None:
        _ccd.header["BUNIT"] = bunit
    headers.cmt2hdr(
        _ccd.header,
        "h",
        t_ref=_t,
        s="[air.convert_bit] Converted {}-bit to {}-bit".format(
            original_bit,
            target_bit,
        ),
    )
    return _ccd
