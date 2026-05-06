import glob
import re
from copy import deepcopy
from pathlib import Path, PosixPath, WindowsPath

import numpy as np
import pandas as pd
from astro_ndslice import (
    bezel2slice,
    calc_offset_physical,
    is_list_like,
    listify,
    offseted_shape,
    slicefy,
)
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData, Cutout2D
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS

# from scipy.interpolate import griddata
from . import io as _io
from .logging import logger
from .misc import binning, change_to_quantity, cmt2hdr, update_process, update_tlm

__all__ = [
    # ! file io related:
    "write2fits",
    # ! loaders:
    "load_ccd",
    "inputs2list",
    # ! setters:
    "CCDData_astype",
    "set_ccd_attribute",
    "set_ccd_gain_rdnoise",
    "propagate_ccdmask",
    # ! ccd processes
    "imslice",
    "trim_overlap",
    "cut_ccd",
    "bin_ccd",
    "fixpix",
    # "make_errormap",
    "find_extpix",
    "find_satpix",
    "convert_bit",
    # ! header update:
    "hedit",
    "key_remover",
    "key_mapper",
    "chk_keyval",
    # ! header accessor:
    "valinhdr",
    "get_from_header",
    "get_if_none",
    # ! WCS related:
    "midtime_obs",
]

def write2fits(data, header, output, return_ccd=False, **kwargs):
    """A convenience function to write proper FITS file.

    Parameters
    ----------
    data : `~numpy.ndarray`
        The data

    header : `~astropy.io.fits.Header`
        The header

    output : path-like
        The output file path

    return_ccd : `bool`, optional.
        Whether to return the generated `~astropy.nddata.CCDData`.

    **kwargs :
        The keyword arguements to write FITS file by
        `~astropy.nddata.fits_data_writer`, such as ``output_verify=True``,
        ``overwrite=True``.
        Default: `False`.
    """
    ccd = CCDData(data=data, header=header, unit=header.get("BUNIT", "adu"))

    try:
        ccd.write(output, **kwargs)
    except fits.VerifyError:
        logger.warning("Try using output_verify='fix' to avoid this error.")
    if return_ccd:
        return ccd


# **************************************************************************************** #
# *                                        FILE IO                                       * #
# **************************************************************************************** #
from .io import load_ccd


def inputs2list(
    inputs, sort=True, accept_ccdlike=True, path_to_text=False, check_coherency=False
):
    """Convert glob pattern or `list`-like of path-like to `list` of `~pathlib.Path`

    Parameters
    ----------
    inputs : `str`, path-like, `~astropy.nddata.CCDData`, `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~pandas.DataFrame`-convertable.
        If `~pandas.DataFrame`-convertable, e.g., `dict`, `~pandas.DataFrame` or
        `~astropy.table.Table`, it must have column named ``"file"``, such that
        ``outlist = `list`(inputs["file"])`` is possible. Otherwise, please use,
        e.g., ``inputs = `list`(that_table["filenamecolumn"])``. If a `str` starts
        with ``"@"`` (e.g., ``"@darks.`list`"``), it assumes the file contains a
        `list` of paths separated by ``"\\n"``, as in IRAF.

    sort : `bool`, optional.
        Whether to sort the output `list`.
        Default: `True`.

    accept_ccdlike: `bool`, optional.
        Whether to accept `~astropy.nddata.CCDData`-like objects and simpley
        return ``[inputs]``.
        Default: `True`.

    path_to_text: `bool`, optional.
        Whether to convert the `pathlib.Path` object to `str`.
        Default: `True`.

    check_coherence: `bool`, optional.
        Whether to check if all elements of the `inputs` have the identical
        type.
        Default: `False`.
    """
    contains_ccdlike = False
    if inputs is None:
        return None
    elif isinstance(inputs, str):
        if inputs.startswith("@"):
            with open(inputs[1:]) as ff:
                outlist = ff.read().splitlines()
        else:
            # If str, "dir/file.fits" --> [Path("dir/file.fits")]
            #         "dir/*.fits"    --> [Path("dir/file.fits"), ...]
            outlist = glob.glob(inputs)
    elif isinstance(inputs, (PosixPath, WindowsPath)):
        # If Path, ``TOP/"file*.fits"`` --> [Path("top/file1.fits"), ...]
        outlist = glob.glob(str(inputs))
    elif isinstance(inputs, _io.ASTROPY_CCD_TYPES):
        if accept_ccdlike:
            outlist = [inputs]
        else:
            raise TypeError(
                f"{type(inputs)} is given as `inputs`. "
                + "Turn off accept_ccdlike or use path-like."
            )
    elif isinstance(inputs, (Table, dict, pd.DataFrame)):
        # Do this before is_list_like because DataFrame returns True in
        # is_list_like as it is iterable.
        try:
            outlist = list(inputs["file"])
        except KeyError:
            raise KeyError(
                "If inputs is DataFrame convertible, it must have column named 'file'."
            )
    elif is_list_like(inputs):
        type_ref = type(inputs[0])
        outlist = []
        for i, item in enumerate(inputs):
            if check_coherency and not isinstance(item, type_ref):
                raise TypeError(
                    f"The 0-th item has {type_ref} while {i}-th has {type(item)}."
                )
            if isinstance(item, _io.ASTROPY_CCD_TYPES):
                contains_ccdlike = True
                if accept_ccdlike:
                    outlist.append(item)
                else:
                    raise TypeError(
                        f"{type(item)} is given in the {i}-th element. "
                        + "Turn off accept_ccdlike or use path-like."
                    )
            else:  # assume it is path-like
                if path_to_text:
                    outlist.append(str(item))
                else:
                    outlist.append(Path(item))
    else:
        raise TypeError(f"inputs type ({type(inputs)})not accepted.")

    if sort and not contains_ccdlike:
        outlist.sort()

    return outlist


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
    >>> ccd = fm.CCDData_astype(ccd, dtype='int16', uncertainty_dtype='float32')
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

    update_tlm(nccd.header)
    return nccd


# **************************************************************************************** #
# *                                         SETTER                                        * #
# **************************************************************************************** #
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
    """Set attributes from given paramters.

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

    value_Q, value_from = get_if_none(
        value=value,
        header=ccd.header,
        key=key,
        unit=unit,
        verbose=verbose,
        default=default,
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
                s.append(
                    f"[fitsmgmt.set_ccd_attribute] (Original {key} = {v} is overwritten.)"
                )

            except (KeyError, ValueError):
                pass

            ccd.header[key] = (value_Q.value, header_comment)
        # add as history
        cmt2hdr(ccd.header, "h", s, t_ref=_t_start)

    setattr(ccd, name, value_Q)
    update_tlm(ccd.header)


# TODO: This is quite much overlapping with get_gain_rdnoise...
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
        keywords of `gain_key` and `rdnoise_key`, and interprete the header
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


# **************************************************************************************** #
# *                                   CCD MANIPULATIONS                                  * #
# **************************************************************************************** #
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

    fill_value : `None`, `float`-like, optinoal, optional.
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
            infostr = [
                f"[fitsmgmt.imslice] Sliced using `{trimsec = }`: converted to {sl}. "
            ]
            if fill_value is not None:
                infostr.append(f"Filled background with `{fill_value = }`.")
            cmt2hdr(hdr, "h", infostr, t_ref=_t, verbose=verbose)
            update_process(hdr, "T")

    return nccd


# FIXME: not finished.
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


# FIXME: docstring looks strange about wcs..
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
    factor_x=1,
    factor_y=1,
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

    factor_x, factor_y : `int`, optional.
        The binning factors in x, y direction.

    binfunc : funciton object, optional.
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
    This is ~ 20-30 to upto 10^5 times faster than astropy.nddata's
    block_reduce:

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
    Tested on  MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16
    GB (2400MHz DDR4), Radeon Pro 560X (4GB)]
    """
    _t_start = Time.now()

    if not isinstance(ccd, CCDData):
        raise TypeError("ccd must be CCDData object.")

    if factor_x == 1 and factor_y == 1:
        return ccd

    if copy:
        _ccd = ccd.copy()
    else:
        _ccd = ccd

    _ccd.data = binning(
        _ccd.data,
        factor_x=factor_x,
        factor_y=factor_y,
        binfunc=binfunc,
        trim_end=trim_end,
    )
    if update_header:
        _ccd.header["BINFUNC"] = (binfunc.__name__, "The function used for binning.")
        _ccd.header["XBINNING"] = (
            factor_x,
            "Binning done after the observation in X direction",
        )
        _ccd.header["YBINNING"] = (
            factor_y,
            "Binning done after the observation in Y direction",
        )
        # add as history
        cmt2hdr(
            _ccd.header,
            "h",
            t_ref=_t_start,
            s=f"[bin_ccd] Binned by (xbin, ybin) = ({factor_x}, {factor_y}) ",
        )
    return _ccd


# TODO: Need something (e.g., cython with pythran) to boost the speed of this function.
def fixpix(
    ccd,
    mask=None,
    maskpath=None,
    extension=None,
    mask_extension=None,
    priority=None,
    update_header=True,
    verbose=True,
):
    """Interpolate the masked location (N-D generalization of IRAF PROTO.FIXPIX)

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, or number-like
        The CCD data to be "fixed".

    mask : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, optional.
        The mask to be used for fixing pixels (pixels to be fixed are where
        `mask` is `True`). If `None`, nothing will happen and `ccd` is
        returned.

    extension, mask_extension: `int`, `str`, (`str`, `int`), `None`
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.
        Default: `None`.

    priority: `tuple` of `int`, `None`, optional.
        The priority of axis as a `tuple` of non-repeating `int` from ``0`` to
        `ccd.ndim`. It will be used if the mask has the same size along two or
        more of the directions. To specify, use the integers for axis
        directions, descending priority. For example,  ``(2, 1, 0)`` will be
        identical to `priority=None` (default) for 3-D images.
        Default is `None` to follow IRAF's PROTO.FIXPIX: Priority is higher for
        larger axis number (e.g., in 2-D, x-axis (axis=1) has higher priority
        than y-axis (axis=0)).

    Examples
    --------
    Timing test: MBP 15" [2018, macOS 11.4, i7-8850H (2.6 GHz; 6-core), RAM 16
    GB (2400MHz DDR4), Radeon Pro 560X (4GB)], 2021-11-05 11:14:04 (KST:
    GMT+09:00)

    >>> np.random.RandomState(123)  # RandomState(MT19937) at 0x7FAECA768D40
    >>> data = np.random.normal(size=(1000, 1000))
    >>> mask = np.zeros_like(data).astype(bool)
    >>> mask[10, 10] = True
    >>> %timeit fm.fixpix(data, mask)
    19.7 ms +- 1.53 ms per loop (mean +- std. dev. of 7 runs, 100 loops each)

    >>> print(data[9:12, 9:12], fm.fixpix(data, mask)[9:12, 9:12])
    # [[ 1.64164502 -1.00385046 -1.24748504]
    #  [-1.31877621  1.37965928  0.66008966]
    #  [-0.7960262  -0.14613834 -1.34513327]]
    # [[ 1.64164502 -1.00385046 -1.24748504]
    #  [-1.31877621 -0.32934328  0.66008966]
    #  [-0.7960262  -0.14613834 -1.34513327]] adu
    """
    if mask is None:
        return ccd.copy()

    _t_start = Time.now()

    _ccd, _, _ = _io._parse_image(ccd, extension=extension, force_ccddata=True)
    mask, maskpath, _ = _io._parse_image(
        mask, extension=mask_extension, name=maskpath, force_ccddata=True
    )
    mask = mask.data.astype(bool)
    data = _ccd.data
    naxis = _ccd.shape

    if _ccd.shape != mask.shape:
        raise ValueError(
            f"ccd and mask must have the identical shape; now {_ccd.shape} VS {mask.shape}."
        )

    ndim = data.ndim

    if priority is None:
        priority = tuple([i for i in range(ndim)][::-1])
    elif len(priority) != ndim:
        raise ValueError(
            "len(priority) and ccd.ndim must be the same; "
            + f"now {len(priority)} VS {ccd.ndim}."
        )
    elif not isinstance(priority, tuple):
        priority = tuple(priority)
    elif (np.min(priority) != 0) or (np.max(priority) != ndim - 1):
        raise ValueError(
            f"`priority` must be a tuple of int (0 <= int <= {ccd.ndim-1=}). "
            + f"Now it's {priority=}"
        )

    structures = [np.zeros([3] * ndim) for _ in range(ndim)]
    for i in range(ndim):
        sls = [[slice(1, 2, None)] * ndim for _ in range(ndim)][0]
        sls[i] = slice(None, None, None)
        structures[i][tuple(sls)] = 1
    # structures[i] is the structure to obtain the num. of connected pix. along axis=i

    pixels = []
    n_axs = []
    labels = []

    for structure in structures:
        from scipy.ndimage import label as ndlabel

        _label, _nlabel = ndlabel(mask, structure=structure)
        _pixels = {}
        _n_axs = {}
        for k in range(1, _nlabel + 1):
            _label_k = _label == k
            _pixels[k] = np.where(_label_k)
            _n_axs[k] = np.count_nonzero(_label_k)
        labels.append(_label)
        pixels.append(_pixels)
        n_axs.append(_n_axs)

    idxs = np.where(mask)
    for pos in np.array(idxs).T:
        # The label of this position in each axis
        label_pos = [lab.item(*pos) for lab in labels]
        # number of pixels of the same label for each direction
        n_ax = [_n_ax[lab] for _n_ax, lab in zip(n_axs, label_pos)]

        # The shortest axis along which the interpolation will happen,
        # OR, if 1+ directions having same minimum length, select this axis
        #   according to `priority`
        interp_ax = np.where(n_ax == np.min(n_ax))[0]
        if len(interp_ax) > 1:
            for i_ax in priority:  # check in the identical order to `priority`
                if i_ax in interp_ax:
                    interp_ax = i_ax
                    break
        else:
            interp_ax = interp_ax[0]
        # The coordinates of the pixels having the identical label to this
        # pixel position, along the shortest axis
        coord_samelabel = pixels[interp_ax][label_pos[interp_ax]]
        coord_slice = []
        coord_init = []
        coord_last = []
        for i in range(ndim):
            invalid = False
            if i == interp_ax:
                init = np.min(coord_samelabel[i]) - 1
                last = np.max(coord_samelabel[i]) + 1
                # distance between the initial/last points to be used for the
                # interpolation, along the interpolation axis:
                delta = last - init
                # grid for interpolation:
                grid = np.arange(1, delta - 0.1, 1)
                # Slice to be used for interpolation:
                sl = slice(init + 1, last, None)
                # Should be done here, BEFORE the if clause below.

                # Check if lower/upper are all outside the image
                if init < 0 and last >= naxis[i]:
                    invalid = True
                    break
                elif init < 0:  # if only one of lower/upper is outside the image
                    init = last
                elif last >= naxis[i]:
                    last = init
            else:
                init = coord_samelabel[i][0]
                last = coord_samelabel[i][0]
                # coord_samelabel[i] is nothing but an array of same numbers
                sl = slice(init, last + 1, None)

            coord_init.append(init)
            coord_last.append(last)
            coord_slice.append(sl)

        if not invalid:
            val_init = data.item(tuple(coord_init))
            val_last = data.item(tuple(coord_last))
            data[tuple(coord_slice)].flat = (
                val_last - val_init
            ) / delta * grid + val_init

    if update_header:
        nfix = np.count_nonzero(mask)
        _ccd.header["MASKNPIX"] = (nfix, "No. of pixels masked (fixed) by fixpix.")
        _ccd.header["MASKFILE"] = (maskpath, "Applied mask for fixpix.")
        _ccd.header["MASKORD"] = (
            str(priority),
            "Axis priority for fixpix (python order)",
        )
        # MASKFILE: name identical to IRAF
        # add as history
        cmt2hdr(
            _ccd.header,
            "h",
            t_ref=_t_start,
            verbose=verbose,
            s="[fixpix] Pixel values interpolated.",
        )
        update_process(_ccd.header, "P")

    return _ccd


# # FIXME: Remove this after fixpix is completed
# def fixpix_griddata(ccd, mask, extension=None, method='nearest',
#     fill_value=0, update_header=True):
#     """ Interpolate the masked location (cf. IRAF's PROTO.FIXPIX)
#     Parameters
#     ----------
#     ccd : CCDData-like (e.g., PrimaryHDU, ImageHDU, HDUList), ndarray, path-like, or number-like
#         The CCD data to be "fixed".

#     mask : ndarray (bool)
#         The mask to be used for fixing pixels (pixels to be fixed are where
#         `mask` is `True`).

#     extension: int, str, (str, int)
#         The extension of FITS to be used. It can be given as integer
#         (0-indexing) of the extension, ``EXTNAME`` (single str), or a tuple
#         of str and int: ``(EXTNAME, EXTVER)``. If `None` (default), the
#         *first extension with data* will be used.

#     method: str
#         The interpolation method. Even the ``'linear'`` method takes too long
#         time in many cases, so the default is ``'nearest'``.
#     """
#     _t_start = Time.now()

#     _ccd, _, _ = _io._parse_image(ccd, extension=extension, force_ccddata=True)
#     data = _ccd.data

#     x_idx, y_idx = np.meshgrid(np.arange(0, data.shape[1] - 0.1),
#                                np.arange(0, data.shape[0] - 0.1))
#     mask = mask.astype(bool)
#     x_valid = x_idx[~mask]
#     y_valid = y_idx[~mask]
#     z_valid = data[~mask]
#     _ccd.data = griddata((x_valid, y_valid),
#                          z_valid, (x_idx, y_idx), method=method, fill_value=fill_value)

#     if update_header:
#         _ccd.header["MASKMETH"] = (method,
#                                    "The interpolation method for fixpix")
#         _ccd.header["MASKFILL"] = (fill_value,
#                                    "The fill value if interpol. fails in fixpix")
#         _ccd.header["MASKNPIX"] = (np.count_nonzero(mask),
#                                    "Total num of pixels fixed by fixpix.")
#         # add as history
#         cmt2hdr(_ccd.header, 'h', t_ref=_t_start, s="Pixel values fixed by fixpix")
#     update_tlm(_ccd.header)

#     return _ccd


def find_extpix(
    ccd,
    mask=None,
    npixs=(1, 1),
    bezels=None,
    order_xyz=True,
    sort=True,
    update_header=True,
    verbose=0,
):
    """Finds the N extrema pixel values excluding masked pixels.

    Parameters
    ---------
    ccd : `~astropy.nddata.CCDData`
        The ccd to find extreme values

    mask : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, or number-like, optional.
        The mask to be used. To reduce file I/O time, better to provide
        `~numpy.ndarray`.
        Default: `None`.

    npixs : length-2 `tuple` of `int`, optional
        The numbers of extrema to find, in the form of ``[small, large]``, so
        that ``small`` number of smallest and ``large`` number of largest pixel
        values will be found. If `None`, no extrema is found (`None` is
        returned for that extremum).
        Default: ``(1, 1)`` (find minimum and maximum)
        Default: ``(1, 1)``.

    bezels : `list` of `list` of `int`, optional.
        If given, must be a `list` of `list` of `int`. Each `list` of `int` is in the
        form of ``[lower, upper]``, i.e., the first ``lower`` and last
        ``upper`` rows/columns are ignored.
        Default: `None`.

    order_xyz : `bool`, optional.
        Whether `bezel` in xyz order or not (python order:
        ``xyz_order[::-1]``).
        Default: `True`.

    sort: `bool`, optional.
        Whether to sort the extrema in ascending order.
        Default: `True`.

    Returns
    -------
    min
        The `list` of extrema pixel values.
    """
    if not len(npixs) == 2:
        raise ValueError("npixs must be a length-2 tuple of int.")
    _t = Time.now()
    data = ccd.data.copy().astype("float32")  # Not float64 to reduce memory usage
    # slice first to reduce computation time
    if bezels is not None:
        sls = bezel2slice(bezels, order_xyz=order_xyz)
        data = data[sls]
        if mask is not None:
            mask = mask[sls]

    if mask is None:
        maskname = "No mask"
        mask = ~np.isfinite(data)
    else:
        if not isinstance(mask, np.ndarray):
            mask, maskname, _ = _io._parse_image(mask, force_ccddata=True)
            mask = mask.data | ~np.isfinite(data)
        else:
            maskname = "User-provided mask"

    exts = []
    for npix, sign, minmaxval in zip(npixs, [1, -1], [np.inf, -np.inf]):
        if npix is None:
            exts.append(None)
            continue
        data[mask] = minmaxval
        # ^ if getting maximum/minimum pix vals, replace with minimum/maximum
        extvals = np.partition(data.ravel(), sign * npix)
        #         ^^^^^^^^^^^^
        # bn.partitoin has virtually no speed gain.
        extvals = extvals[:npix] if sign > 0 else extvals[-npix:]
        if sort:
            extvals = np.sort(extvals)[::sign]
        exts.append(extvals)

    if update_header:
        for ext, mm in zip(exts, ["min", "max"]):
            if ext is not None:
                for i, extval in enumerate(ext):
                    ccd.header.set(
                        f"{mm.upper()}V{i+1:03d}", extval, f"{mm} pixel value"
                    )
        bezstr = ""
        if bezels is not None:
            order = "xyz order" if order_xyz else "pythonic order"
            bezstr = f" and bezel: {bezels} in {order}"
        cmt2hdr(
            ccd.header,
            "h",
            verbose=verbose,
            t_ref=_t,
            s=(
                "[fitsmgmt.find_extpix] Extrema pixel values found N(smallest, largest) = "
                + f"{npixs} excluding mask ({maskname}){bezstr}. "
                + "See MINViii and MAXViii."
            ),
        )
    return exts


def find_satpix(
    ccd,
    mask=None,
    satlevel=65535,
    bezels=None,
    order_xyz=True,
    update_header=True,
    verbose=0,
):
    """Finds saturated pixel values excluding masked pixels.

    Parameters
    ---------
    ccd : `~astropy.nddata.CCDData`, `~numpy.ndarray`
        The ccd to find extreme values. If `ndarray`, `update_header` will
        automatically be set to `False`.

    mask : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, or number-like, optional.
        The mask to be used. To reduce file I/O time, better to provide
        `~numpy.ndarray`.
        Default: `None`.

    satlevel: numeric, optional.
        The saturation level. Pixels >= `satlevel` will be retarded as
        saturated pixels, except for those masked by `mask`.
        Default: ``65535``.

    bezels : `list` of `list` of `int`, optional.
        If given, must be a `list` of `list` of `int`. Each `list` of `int` is in the
        form of ``[lower, upper]``, i.e., the first ``lower`` and last
        ``upper`` rows/columns are ignored.
        Default: `None`.

    order_xyz : `bool`, optional.
        Whether `bezel` in xyz order or not (python order:
        ``xyz_order[::-1]``).
        Default: `True`.

    Returns
    -------
    min
        The `list` of extrema pixel values.
    """
    _t = Time.now()
    if isinstance(ccd, CCDData):
        data = ccd.data.copy()
    else:
        data = ccd.copy()
        update_header = False
    satmask = np.zeros(data.shape, dtype=bool)
    # slice first to reduce computation time
    if bezels is not None:
        sls = bezel2slice(bezels, order_xyz=order_xyz)
        data = data[sls]
        if mask is not None:
            mask = mask[sls]
    else:
        sls = [slice(None, None, None) for _ in range(data.ndim)]

    if mask is None:
        maskname = "No mask"
        satmask[sls] = data >= satlevel
    else:
        if not isinstance(mask, np.ndarray):
            mask, maskname, _ = _io._parse_image(mask, force_ccddata=True)
            mask = mask.data
        else:
            maskname = "User-provided mask"
        satmask[sls] = (data >= satlevel) & (~mask)  # saturated && not masked

    if update_header:
        nsat = np.count_nonzero(satmask[sls])
        ccd.header["NSATPIX"] = (nsat, "No. of saturated pix")
        ccd.header["SATLEVEL"] = (satlevel, "Saturation: pixels >= this value")
        bezstr = ""
        if bezels is not None:
            order = "xyz order" if order_xyz else "pythonic order"
            bezstr = f" and bezel: {bezels} in {order}"
        cmt2hdr(
            ccd.header,
            "h",
            verbose=verbose,
            t_ref=_t,
            s=(
                "[fitsmgmt.find_satpix] Saturated pixels calculated based on satlevel = "
                + f"{satlevel}, excluding mask ({maskname}){bezstr}. "
                + "See NSATPIX and SATLEVEL."
            ),
        )
    return satmask


# **************************************************************************************** #
# *                                  HEADER MANIPULATION                                 * #
# **************************************************************************************** #
def hedit(
    item,
    keys,
    values,
    comments=None,
    befores=None,
    afters=None,
    add=False,
    output=None,
    overwrite=False,
    output_verify="fix",
    verbose=True,
):
    """Edit the header key (usu. to update value of a keyword).

    Parameters
    ----------
    item : `astropy` header, path-like, `~astropy.nddata.CCDData`-like
        The FITS file or header to edit. If `~astropy.io.fits.Header`, it is updated
        **inplace**.

    keys : `str`, `list`-like of `str`
        The key to edit.

    values : `str`, numeric, or `list`-like of such
        The new value. To pass one single iterable (e.g., `[1, 2, 3]`) for one
        single `key`, use a `list` of it (e.g., `[[1, 2, 3]]`) to circumvent
        problem.

    comment : `str`, `list`-like of `str` optional.
        The comment to add.

    add : `bool`, optional.
        Whether to add the key if it is not in the header.
        Default: `False`.

    befores : `str`, `int`, `list`-like of such, optional
        Name of the keyword, or index of the `Card` before which this card
        should be located in the header. The argument `before` takes
        precedence over `after` if both specified.
        Default: `None`.

    after : `str`, `int`, `list`-like of such, optional
        Name of the keyword, or index of the `Card` after which this card
        should be located in the header.

    output: path-like, optional
        The output file.
        Default: `None`.

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        The header-updated `~astropy.nddata.CCDData.` `None` if `item` was pure `~astropy.io.fits.Header`.
    """

    def _add_key(header, key, val, infostr, cmt=None, before=None, after=None):
        header.set(key, value=val, comment=cmt, before=before, after=after)
        # infostr += " (comment: {})".format(comment) if comment is not None else ""
        if before is not None:
            infostr += f" (moved: {before=})"
        elif after is not None:  # `after` is ignored if `before` is given
            infostr += f" (moved: {after=})"
        cmt2hdr(header, "h", infostr, verbose=verbose)
        update_tlm(header)

    if isinstance(item, fits.header.Header):
        header = item
        if verbose:
            logger.info("item is astropy Header. (any `output` is ignored).")
        output = None
        ccd = None
    elif isinstance(item, _io.ASTROPY_CCD_TYPES):
        ccd, imname, _ = _io._parse_image(item, force_ccddata=True, copy=False)
        #                                                   ^^^^^^^^^^
        # Use copy=False to update header of the input CCD inplace.z
        header = ccd.header
    else:
        ccd = load_ccd(item)
        imname = str(item)
        header = ccd.header
        if output is None and overwrite:
            output = item

    keys, values, comments, befores, afters = listify(
        keys, values, comments, befores, afters
    )

    for key, val, cmt, bef, aft in zip(keys, values, comments, befores, afters):
        if key in header:
            oldv = header[key]
            infostr = (
                f"[fitsmgmt.HEDIT] {key}={oldv} ({type(oldv).__name__}) "
                + f"--> {val} ({type(val).__name__})"
            )
            _add_key(header, key, val, infostr, cmt=cmt, before=bef, after=aft)
        else:
            if add:  # add key only if `add` is True.
                infostr = f"[fitsmgmt.HEDIT add] {key}= {val} ({type(val).__name__})"
                _add_key(header, key, val, infostr, cmt=cmt, before=bef, after=aft)
            elif verbose:
                logger.info(
                    "%s does not exist in the header. Skipped. (add=True to proceed)", key
                )

    if output is not None:
        ccd.write(output, overwrite=overwrite, output_verify=output_verify)
        if verbose:
            logger.info("%s --> %s", imname, output)

    return ccd


def key_remover(header, remove_keys, deepremove=True):
    """Removes keywords from the header.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        The header to be modified

    remove_keys : `list` of `str`
        The header keywords to be removed.

    deepremove : `True`, optional
        FITS standard does not have any specification of duplication of
        keywords as discussed in the following issue:
        https://github.com/astropy/ccdproc/issues/464
        If it is set to `True`, ALL the keywords having the name specified in
        `remove_keys` will be removed. If not, only the first occurence of each
        key in `remove_keys` will be removed. It is more sensical to set it
        `True` in most of the cases.
        Default: `True`.
    """
    nhdr = header.copy()
    if deepremove:
        for key in remove_keys:
            while True:
                try:
                    nhdr.remove(key)
                except KeyError:
                    break
    else:
        for key in remove_keys:
            try:
                nhdr.remove(key)
            except KeyError:
                continue

    return nhdr


def key_mapper(header, keymap=None, deprecation=False, remove=False):
    """Update the header to meed the standard (keymap).

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        The header to be modified

    keymap : `dict`, optional.
        The dictionary contains ``{<standard_key>:<original_key>}``
        information. If it is `None` (default), the copied version of the
        header is returned without any change.
        Default: `None`.

    deprecation : `bool`, optional
        Whether to change the original keywords' comments to contain
        deprecation warning. If `True`, the original keywords' comments will
        become ``DEPRECATED. See <standard_key>.``. It has no effect if
        ``remove=True``.
        Default is `False`.

    remove : `bool`, optional.
        Whether to remove the original keyword. `deprecation` is ignored if
        ``remove=True``.
        Default is `False`.

    Returns
    -------
    newhdr: `~astropy.io.fits.Heade`r
        The updated (key-mapped) header.

    Notes
    -----
    If the new keyword already exist in the given header, virtually nothing
    will happen. If ``deprecation=True``, the old one's comment will be
    changed, and if ``remove=True``, the old one will be removed; the new
    keyword will never be changed or overwritten.
    """

    def _rm_or_dep(hdr, old, new):
        if remove:
            hdr.remove(old)
        elif deprecation:  # do not remove but deprecate
            hdr.comments[old] = f"DEPRECATED. See {new}"

    newhdr = header.copy()
    if keymap is not None:
        for k_new, k_old in keymap.items():
            if k_new == k_old:
                continue

            if k_old is not None:
                if (
                    k_new in newhdr
                ):  # if k_new already in the header, JUST deprecate k_old.
                    _rm_or_dep(newhdr, k_old, k_new)
                else:  # if not, copy k_old to k_new and deprecate k_old.
                    try:
                        comment_ori = newhdr.comments[k_old]
                        newhdr[k_new] = (newhdr[k_old], comment_ori)
                        _rm_or_dep(newhdr, k_old, k_new)
                    except (KeyError, IndexError):
                        # don't even warn
                        pass

    return newhdr


def chk_keyval(type_key, type_val, group_key):
    """Checks the validity of key and values used heavily in combutil.

    Parameters
    ----------
    type_key : `None`, `str`, `list` of `str`, optional
        The header keyword for the ccd type you want to use for match.

    type_val : `None`, `int`, `str`, `float`, etc and `list` of such
        The header keyword values for the ccd type you want to match.


    group_key : `None`, `str`, `list` of `str`, optional
        The header keyword which will be used to make groups for the CCDs that
        have selected from `type_key` and `type_val`. If `None` (default), no
        grouping will occur, but it will return the `~pandas.DataFrameGroupBy`
        object will be returned for the sake of consistency.

    Returns
    -------
    type_key, type_val, group_key
    """
    # Make type_key to list
    if type_key is None:
        type_key = []
    elif is_list_like(type_key):
        try:
            type_key = list(type_key)
            if not all(isinstance(x, str) for x in type_key):
                raise TypeError("Some of type_key are not str.")
        except TypeError:
            raise TypeError("type_key should be str or convertible to list.")
    elif isinstance(type_key, str):
        type_key = [type_key]
    else:
        raise TypeError(
            f"`type_key` not understood (type = {type(type_key)}): {type_key}"
        )

    # Make type_val to list
    if type_val is None:
        type_val = []
    elif is_list_like(type_val):
        try:
            type_val = list(type_val)
        except TypeError:
            raise TypeError("type_val should be str or convertible to list.")
    elif isinstance(type_val, str):
        type_val = [type_val]
    else:
        raise TypeError(
            f"`type_val` not understood (type = {type(type_val)}): {type_val}"
        )

    # Make group_key to list
    if group_key is None:
        group_key = []
    elif is_list_like(group_key):
        try:
            group_key = list(group_key)
            if not all(isinstance(x, str) for x in group_key):
                raise TypeError("Some of group_key are not str.")
        except TypeError:
            raise TypeError("group_key should be str or convertible to list.")
    elif isinstance(group_key, str):
        group_key = [group_key]
    else:
        raise TypeError(
            f"`group_key` not understood (type = {type(group_key)}): {group_key}"
        )

    if len(type_key) != len(type_val):
        raise ValueError("`type_key` and `type_val` must have the same length!")

    # If there is overlap
    overlap = set(type_key).intersection(set(group_key))
    if len(overlap) > 0:
        logger.warning(
            f"{overlap} appear in both `type_key` and `group_key`."
            "It may not be harmful but better to avoid."
        )

    return type_key, type_val, group_key


def valinhdr(val=None, header=None, key=None, default=None, unit=None):
    """Get the value by priority: val > header[key] > default.

    Parameters
    ----------
    val : object, optional.
        If not `None`, `header`, `key`, and `default` will **not** be used.
        This is different from `header.get(key, default)`. It is therefore
        useful if the API wants to override the header value by the
        user-provided one.
        Default: `None`.

    header : `~astropy.io.fits.Header`, optional.
        The header to extract the value if `value` is `None`.
        Default: `None`.

    key : `str`, optional.
        The header keyword to extract if `value` is `None`.
        Default: `None`.

    default : object, optional.
        The default value. If `value` is `None`, then ``header.get(key,
        default)``.
        Default: `None`.

    unit : `str`, optional.
        `None` to ignore unit. ``''`` (empty string) means `Unit(dimensionless)`.
        Better to leave it as `None` unless astropy unit is truely needed.
        Default: `None`.

    Notes
    -----
    It takes << 10 us (when unit=`None`) or for any case for a reasonably lengthy
    header. See `Tests` below. Tested on MBP 15" [2018, macOS 11.6, i7-8850H
    (2.6 GHz; 6-core), RAM 16 GB (2400MHz DDR4), Radeon Pro 560X (4GB)].

    Tests
    -----

    .. code-block:: python

        real_q = 20*u.s
        real_v = 20
        default_q = 0*u.s
        default_v = 0
        test_q = 3*u.s
        test_v = 3

        # w/o unit  Times are the %timeit result of the LHS
        assert valinhdr(None,   hdr, "EXPTIME", default=0) == real_v  # ~ 6.5 us
        assert valinhdr(None,   hdr, "EXPTIxx", default=0) == default_v # ~ 3.5 us
        assert valinhdr(test_v, hdr, "EXPTIxx", default=0) == test_v  # ~ 0.3 us
        assert valinhdr(test_q, hdr, "EXPTIxx", default=0) == test_v  # ~ 0.6 us
        # w/ unit  Times are the %timeit result of the LHS
        assert valinhdr(None,   hdr, "EXPTIME", default=0, unit='s') == real_q  # ~ 23 us
        assert valinhdr(None,   hdr, "EXPTIxx", default=0, unit='s') == default_q # ~ 16 us
        assert valinhdr(test_v, hdr, "EXPTIxx", default=0, unit='s') == test_q  # ~ 11 us
        assert valinhdr(test_q, hdr, "EXPTIxx", default=0, unit='s') == test_q  # ~ 15 us

        For a test astropy.nddata.CCDData, the following timing gave ~ 0.5 ms on MBP 15" [2018,
        macOS 11.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB (2400MHz DDR4), Radeon
        Pro 560X (4GB)]
        %timeit ((fm.valinhdr(None, ccd.header, "EXPTIME", unit=u.s)
                 / fm.valinhdr(3*u.s, ccd.header, "EXPTIME", unit=u.s)).si.value)
    """
    uu = 1 if unit is None else u.Unit(unit)
    #    ^ NOT 1.0 to preserve the original dtype (e.g., int)
    val = header.get(key, default) if val is None else val

    if isinstance(val, u.Quantity):
        return val.value if unit is None else val.to(unit)
    else:
        try:
            return val * uu
        except TypeError:  # e.g., val is a str
            return val


def get_from_header(header, key, unit=None, verbose=True, default=0):
    """Get a variable from the header object.

    Parameters
    ----------
    header : astropy.Header
        The header to extract the value.

    key : `str`
        The header keyword to extract.

    unit : astropy unit, optional.
        The unit of the value.
        Default: `None`.

    default : `str`, `int`, `float`, ..., or `~astropy.units.Quantity`, optional.
        The default if not found from the header.
        Default: ``0``.

    Returns
    -------
    q: `~astropy.units.Quantity` or any object
        The extracted quantity from the header. It's a `~astropy.units.Quantity` if the unit is
        given. Otherwise, appropriate type will be assigned.
    """
    # If using q = header.get(key, default=default),
    # we cannot give any meaningful verboses infostr.
    # Anyway the `header.get` sourcecode contains only 4-line:
    # ``try: return header[key] // except (KeyError, IndexError): return default.
    key = key.upper()
    try:
        q = change_to_quantity(header[key], desired=unit)
        if verbose:
            logger.info("header: %-8s = %s", key, q)
    except (KeyError, IndexError):
        q = change_to_quantity(default, desired=unit)
        logger.warning("The key %s not found in header: setting to %s.", key, default)

    return q


def get_if_none(value, header, key, unit=None, verbose=True, default=0, to_value=False):
    """Similar to get_from_header, but a convenience wrapper."""
    if value is None:
        value_Q = get_from_header(
            header, key, unit=unit, verbose=verbose, default=default
        )
        value_from = f"{key} in header"
    else:
        value_Q = change_to_quantity(value, unit, to_value=False)
        value_from = "the user"

    if to_value:
        return value_Q.value, value_from
    else:
        return value_Q, value_from


def midtime_obs(
    header=None,
    dateobs="DATE-OBS",
    format=None,
    scale=None,
    precision=None,
    in_subfmt=None,
    out_subfmt=None,
    location=None,
    exptime="EXPTIME",
    exptime_unit=u.s,
):
    """Calculates the mid-obs time (exposure start + exposure/2)

    Parameters
    ----------
    header : astropy.Header, optional.
        The header to extract the value. `midtime_obs` can be used without
        header. But to do so, `dateobs` must be in `~astropy.time.Time` and
        `exptime` must be given as `float` or `~astropy.units.Quantity`.
        Default: `None`.

    dateobs : `str`, `~astropy.Time`, optional.
        The header keyword for DATE-OBS (start of exposure) or the
        `~astropy.Time` object.
        Default: ``'DATE-OBS'``.

    exptime : `str`, `float`, `~astropy.units.Quantity`, optional.
        The header keyword for exposure time or the exposure time as `float` (in
        seconds) or `~astropy.units.Quantity`.
        Default: ``'EXPTIME'``.

    """
    if isinstance(dateobs, str):
        try:
            time_0 = Time(
                header[dateobs],
                format=format,
                scale=scale,
                precision=precision,
                in_subfmt=in_subfmt,
                out_subfmt=out_subfmt,
                location=location,
            )
        except (KeyError, IndexError):
            raise KeyError(f"The key '{dateobs=}' not found in header.")
    else:
        time_0 = dateobs

    if isinstance(exptime, str):
        try:
            exptime = header.get(exptime, default=0) * exptime_unit
        except (KeyError, IndexError):
            raise KeyError(f"The key '{exptime=}' not found in header.")
    elif isinstance(exptime, (int, float)):
        exptime = exptime * exptime_unit
    elif not isinstance(exptime, u.Quantity):
        raise TypeError(f"exptime type ({type(exptime)}) not understood.")

    return time_0 + exptime / 2




# def center_coord(header, skycoord=False):
#     """ Gives the sky coordinate of the center of the image field of view.
#     Parameters
#     ----------
#     header: astropy.header.Header
#         The header to be used to extract WCS information (and image size)
#     skycoord: bool
#         Whether to return in the ~astropy.coordinates.SkyCoord object. If
#         `False`, a numpy array is returned.
#     """
#     wcs = WCS(header)
#     cx = float(header['naxis1']) / 2 - 0.5
#     cy = float(header['naxis2']) / 2 - 0.5
#     center_coo = wcs.wcs_pix2world(cx, cy, 0)

#     if skycoord:
#         return SkyCoord(*center_coo, unit='deg')

#     return np.array(center_coo)


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
    cmt2hdr(
        _ccd.header,
        "h",
        t_ref=_t,
        s="Converted {}-bit to {}-bit".format(original_bit, target_bit),
    )
    return _ccd
