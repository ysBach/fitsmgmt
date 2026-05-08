"""FITS image loading helpers."""

import glob
import sys
from pathlib import Path, PosixPath, WindowsPath

import numpy as np
import pandas as pd
from astro_ndslice import is_list_like, listify, slicefy
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.wcs import WCS

from .logging import logger

try:
    import fitsio

    HAS_FITSIO = True
except ImportError:
    HAS_FITSIO = False

__all__ = [
    "ASTROPY_CCD_TYPES",
    "_parse_data_header",
    "_parse_image",
    "_has_header",
    "_parse_extension",
    "get_size",
    "write2fits",
    "inputs2list",
    "load_ccd",
    "load_ccds",
]

ASTROPY_CCD_TYPES = (CCDData, fits.PrimaryHDU, fits.ImageHDU)  # fits.CompImageHDU ?


def get_size(obj, seen=None):
    """Recursively estimate an object's memory size in bytes.

    Based on the recursive recipe from
    https://goshippo.com/blog/measure-real-size-any-python-object/.
    """
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        objv = obj.values()
        objk = obj.keys()
        for kv in [objk, objv]:
            for v in kv:
                if not (isinstance(v, np.ndarray) and v.ndim == 0):
                    size += get_size(v, seen)
        # size += sum([get_size(v, seen) for v in obj.values()])
        # size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


# **************************************************************************************** #
# *                                         PARSERS                                       * #
# **************************************************************************************** #
def _parse_data_header(
    ccdlike, extension=None, parse_data=True, parse_header=True, copy=True
):
    """Parse data and header from a CCD-like object, array, header, or path.

    Parameters
    ---------
    ccdlike : `~astropy.nddata.CCDData`, `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`, `~astropy.io.fits.Header`, `~numpy.ndarray`, number-like, path-like, `None`
        The object to be parsed into data and header.

    extension : `int`, `str`, (`str`, `int`), optional.
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.

    parse_data, parse_header : `bool`, optional
        Because this function uses ``.copy()`` for safety, it may take a bit of
        time if this function is used iteratively. One then can turn off one of
        these to ignore either data or header part.
        Default: `True`.

    Returns
    -------
    data : `~numpy.ndarray`, `None`
        The data part of the input `ccdlike`. If `ccdlike` is ``''`` or `None`,
        `None` is returned.

    hdr : `~astropy.io.fits.Header`, `None`
        The header if header exists; otherwise, `None` is returned.

    Notes
    -----
    _parse_data_header and _parse_image have different purposes:
    _parse_data_header is to get a quick copy of the data and/or header,
    especially to CHECK if it has header, while _parse_image is to deal mainly
    with the data (and has options to return as `~astropy.nddata.CCDData`).
    """
    if ccdlike is None or (isinstance(ccdlike, str) and ccdlike == ""):
        data = None
        hdr = None
    elif isinstance(ccdlike, ASTROPY_CCD_TYPES):
        if parse_data:
            data = ccdlike.data.copy() if copy else ccdlike.data
        else:
            data = None
        if parse_header:
            hdr = ccdlike.header.copy() if copy else ccdlike.header
        else:
            hdr = None
    elif isinstance(ccdlike, fits.HDUList):
        extension = _parse_extension(extension) if (parse_data or parse_header) else 0
        # ^ don't even do _parse_extension if both are False
        if parse_data:
            data = ccdlike[extension].data.copy() if copy else ccdlike[extension].data
        else:
            data = None
        if parse_header:
            hdr = (
                ccdlike[extension].header.copy() if copy else ccdlike[extension].header
            )
        else:
            hdr = None
    elif isinstance(ccdlike, (np.ndarray, list, tuple)):
        if parse_data:
            data = np.array(ccdlike, copy=copy)
        else:
            data = None
        hdr = None  # regardless of parse_header
    elif isinstance(ccdlike, fits.Header):
        data = None  # regardless of parse_data
        if parse_header:
            hdr = ccdlike.copy() if copy else ccdlike
        else:
            hdr = None
    elif HAS_FITSIO and isinstance(ccdlike, fitsio.FITSHDR):
        import copy

        data = None  # regardless of parse_data
        if parse_header:
            hdr = copy.deepcopy(ccdlike) if copy else ccdlike
        else:
            hdr = None
    else:
        try:
            data = float(ccdlike) if (parse_data or parse_header) else None
            hdr = None
        except (ValueError, TypeError):  # Path-like
            # NOTE: This try-except cannot be swapped cuz ``Path("2321.3")``
            # can be PosixPath without error...
            extension = _parse_extension(extension) if parse_data or parse_header else 0
            # fits.getheader is ~ 10-20 times faster than load_ccd.
            # 2020-11-09 16:06:41 (KST: GMT+09:00) ysBach
            try:
                if parse_header:
                    hdu = fits.open(Path(ccdlike), memmap=False)[extension]
                    # No need to copy because they've been read (loaded) for
                    # the first time here.
                    data = hdu.data if parse_data else None
                    hdr = hdu.header if parse_header else None
                else:
                    if isinstance(extension, tuple):
                        if HAS_FITSIO:
                            data = fitsio.read(
                                Path(ccdlike), ext=extension[0], extver=extension[1]
                            )
                        else:
                            data = fits.getdata(Path(ccdlike), *extension)
                    else:
                        if HAS_FITSIO:
                            data = fitsio.read(Path(ccdlike), ext=extension)
                            # fitsio returns None for empty HDUs (e.g., PrimaryHDU
                            # with no data). Fall back to first extension with data.
                            if data is None:
                                with fits.open(Path(ccdlike), memmap=False) as hdul:
                                    for hdu in hdul:
                                        if hdu.data is not None:
                                            data = hdu.data
                                            break
                        else:
                            data = fits.getdata(Path(ccdlike), extension)
                    hdr = None
            except TypeError:
                raise TypeError(
                    f"ccdlike type ({type(ccdlike)}) is not acceptable "
                    + "to find header and data."
                )

    return data, hdr


# TODO: str(pathlibPath)
def _parse_image(
    ccdlike,
    extension=None,
    name=None,
    force_ccddata=False,
    prefer_ccddata=False,
    copy=True,
):
    """Parse an image-like input as an array or `~astropy.nddata.CCDData`.

    Parameters
    ----------
    ccdlike : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, or number-like
        The "image" that will be parsed. A string that can be converted to
        `float` (``float(im)``) will be interpreted as numbers; if not, it will
        be interpreted as a path to the FITS file.

    extension : `int`, `str`, (`str`, `int`), optional.
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.
        Default: `None`.

    force_ccddata, prefer_ccddata : `bool`, optional
        `force_ccddata` forces return as `~astropy.nddata.CCDData`. This is
        useful when error calculation is turned on.
        `prefer_ccddata` returns `~astropy.nddata.CCDData` only if `im` was
        `~astropy.nddata.CCDData`,
        HDU object, or `~pathlib.Path`-like to a FITS file, but **not** if it was `~numpy.ndarray`
        or numbers.
        Default: `False`.

    Returns
    -------
    new_im : `~numpy.ndarray` or `~astropy.nddata.CCDData`
        Depending on the options `force_ccddata` and `prefer_ccddata`.

    imname : `str`
        The name of the image.

    imtype : `str`
        The type of the image.

    Notes
    -----
    _parse_data_header and _parse_image have different purposes:
    _parse_data_header is to get a quick copy of the data and/or header,
    especially to CHECK if it has header, while _parse_image is to deal mainly
    with the data (and has options to return as `~astropy.nddata.CCDData`).

    Timing on MBP 14" [2021, macOS 12.2.1, M1Pro(6P+2E/G16c/N16c/32G)]:


    >>> np.random.RandomState(123)
    >>> data = np.random.normal(size=(100,100))
    >>> ccd = CCDData(data, unit='adu')
    >>> fpath = "img/0001.fits"  # doctest: +SKIP
    >>> %timeit air._parse_image(data, name="test", force_ccddata=True)
    >>> %timeit air._parse_image(ccd, name="test", force_ccddata=True)
    >>> %timeit air._parse_image(fpath, name="test", force_ccddata=True) # doctest: +SKIP
    >>> %timeit air._parse_image(fpath, name="test", force_ccddata=False)[0]*1.0 # doctest: +SKIP
    # 14.2 µs +- 208 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)
    # 16.6 µs +- 298 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)
    # 20.8 ms +- 133 µs per loop (mean +- std. dev. of 7 runs, 10000 loops each)
    # 156 µs +- 3.3 µs per loop (mean +- std. dev. of 7 runs, 10000 loops each)

    `fpath` contains a FITS file of 276KB. Note that path with `force_ccddata =
    True` consumes tremendous amount of time, because of astropy's header
    parsing scheme.
    """

    def __extract_extension(ext):
        extension = _parse_extension(ext)
        if extension is None:
            extstr = ""
        else:
            if isinstance(extension, (tuple, list)):
                extstr = f"[{extension[0]}, {extension[1]}]"
            else:
                extstr = f"[{extension}]"
        return extension, extstr

    def __extract_from_hdu(hdu, force_ccddata, prefer_ccddata):
        if force_ccddata or prefer_ccddata:
            unit = ccdlike.header.get("BUNIT", default=u.adu)
            if isinstance(unit, str):
                unit = unit.lower()
            if copy:
                return CCDData(
                    data=hdu.data.copy(), header=hdu.header.copy(), unit=unit
                )
            else:
                return CCDData(data=hdu.data, header=hdu.header, unit=unit)
            # The two lines above took ~ 5 us and 10-30 us for the simplest
            # header and 1x1 pixel data case (regardless of BUNIT exists), on
            # MBP 15" [2018, macOS 10.14.6, i7-8850H (2.6 GHz; 6-core), RAM 16
            # GB (2400MHz DDR4), Radeon Pro 560X (4GB)]
        else:
            return hdu.data.copy() if copy else hdu.data

    ccd_kw = dict(force_ccddata=force_ccddata, prefer_ccddata=prefer_ccddata)
    has_no_name = name is None
    extension, extstr = __extract_extension(extension)
    imname = (
        f"User-provided {ccdlike.__class__.__name__}{extstr}" if has_no_name else name
    )

    if isinstance(ccdlike, CCDData):
        # force_ccddata: CCDData // prefer_ccddata: CCDData // else: ndarray
        if force_ccddata or prefer_ccddata:
            new_im = ccdlike.copy() if copy else ccdlike
        else:
            new_im = ccdlike.data.copy() if copy else ccdlike.data
        imtype = "CCDData"
        imname = str(imname).replace("[0]", "")
    elif isinstance(ccdlike, (fits.PrimaryHDU, fits.ImageHDU)):
        # force_ccddata: CCDData // prefer_ccddata: CCDData // else: ndarray
        new_im = __extract_from_hdu(ccdlike, **ccd_kw)
        imtype = "hdu"
        imname = str(imname).replace("[0]", "")
    elif isinstance(ccdlike, fits.HDUList):
        # force_ccddata: CCDData // prefer_ccddata: CCDData // else: ndarray
        new_im = __extract_from_hdu(ccdlike[extension], **ccd_kw)
        imtype = "HDUList"
    elif isinstance(ccdlike, np.ndarray):
        # force_ccddata: CCDData // prefer_ccddata: ndarray // else: ndarray
        if copy:
            new_im = (
                CCDData(data=ccdlike.copy(), unit="adu")
                if force_ccddata
                else ccdlike.copy()
            )
        else:
            new_im = CCDData(data=ccdlike, unit="adu") if force_ccddata else ccdlike
        imtype = "ndarray"
    else:
        try:  # IF number (ex: im = 1.3)
            # force_ccddata: CCDData // prefer_ccddata: array // else: array
            imname = f"{imname} {ccdlike}" if has_no_name else name
            _im = float(ccdlike)
            new_im = CCDData(data=_im, unit="adu") if force_ccddata else np.asarray(_im)
            imtype = "num"
            # imname can be "int", "float", "str", etc, so imtype might be useful.
        except (ValueError, TypeError):
            try:
                fpath = Path(ccdlike)
            except TypeError:
                raise TypeError(
                    "input must be CCDData-like, ndarray, path-like (to FITS), or a number."
                )

            # If we are here, it is a path-like.
            imname = f"{str(fpath)}{extstr}" if has_no_name else name
            # set redundant extensions to None so that only the part
            # specified by `extension` be loaded:
            new_im = load_ccd(
                fpath,
                extension,
                ccddata=prefer_ccddata or force_ccddata,
                extension_uncertainty=None,
                extension_mask=None,
            )
            imtype = "path"

    return new_im, imname, imtype


def _has_header(ccdlike, extension=None, open_if_file=True):
    """Return whether an object has, or points to, a FITS-like header.

    Parameters
    ---------
    ccdlike : `~astropy.nddata.CCDData`, `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`, `~numpy.ndarray`, number-like, path-like
        The object to be parsed into data and header.

    extension : `int`, `str`, (`str`, `int`), optional.
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used. Used only if `ccdlike` is `~astropy.io.fits.HDUList` or
        path-like.
        Default: `None`.

    open_if_file : `bool`, optional.
        Whether to open the file to check if it has a header when `ccdlike` is
        path-like. Any FITS file has a header, so this means it will check the
        existence and validity of the file. If set to `False`, all path-like
        input will return `False` because the path itself has no header.
        Default: `True`.

    Notes
    -----
    It first checks if the input is one of ``(`~astropy.nddata.CCDData`, `~astropy.io.fits.PrimaryHDU`,
    `~astropy.io.fits.ImageHDU`)``, then if `~astropy.io.fits.HDUList`, then if `np.ndarray`, then if
    number-like, and then finally if path-like. Although this has a bit of
    disadvantage considering we may use file-path for most of the time, the
    overhead is only ~ 1 us, tested on MBP 15" [2018, macOS 10.14.6, i7-8850H
    (2.6 GHz; 6-core), RAM 16 GB (2400MHz DDR4), Radeon Pro 560X (4GB)].
    """
    hashdr = True
    if isinstance(ccdlike, ASTROPY_CCD_TYPES):  # extension not used
        try:
            hashdr = ccdlike.header is not None
        except AttributeError:
            hashdr = False
    elif isinstance(ccdlike, fits.HDUList):
        extension = _parse_extension(extension)
        try:
            hashdr = ccdlike[extension].header is not None
        except AttributeError:
            hashdr = False
    elif is_list_like(ccdlike):
        hashdr = False
    else:
        try:  # if number-like
            _ = float(ccdlike)
            hashdr = False
        except (ValueError, TypeError):  # if path-like
            # NOTE: This try-except cannot be swapped cuz ``Path("2321.3")``
            # can be PosixPath without error...
            if open_if_file:
                try:
                    # fits.getheader is ~ 10-20 times faster than load_ccd.
                    # 2020-11-09 16:06:41 (KST: GMT+09:00) ysBach
                    _ = fits.getheader(Path(ccdlike), extension)
                except (AttributeError, FileNotFoundError):
                    hashdr = False
            else:
                hashdr = False

    return hashdr


def _parse_extension(*args, ext=None, extname=None, extver=None):
    """
    Open the input file, return the `~astropy.io.fits.HDUList` and the extension.

    This supports several different styles of extension selection.  See the
    :func:`getdata()` documentation for the different possibilities.

    Direct copy from astropy, but removing "opening `~astropy.io.fits.HDUList`" part
    https://github.com/astropy/astropy/blob/master/astropy/io/fits/convenience.py#L988

    This is essential for fits_ccddata_reader, because it only has `hdu`, not
    all three of ext, extname, and extver.

    Notes
    -----
    extension parser itself is not a time-consuming process:

    %timeit air._parse_extension()
    # 1.52 µs +- 69.3 ns per loop (mean +- std. dev. of 7 runs, 1000000 loops each)
    """

    err_msg = "Redundant/conflicting extension arguments(s): {}".format(
        {"args": args, "ext": ext, "extname": extname, "extver": extver}
    )

    # This code would be much simpler if just one way of specifying an
    # extension were picked.  But now we need to support all possible ways for
    # the time being.
    if len(args) == 1:
        # Must be either an extension number, an extension name, or an
        # (extname, extver) tuple
        if isinstance(args[0], (int, np.integer)) or (
            isinstance(ext, tuple) and len(ext) == 2
        ):
            if ext is not None or extname is not None or extver is not None:
                raise TypeError(err_msg)
            ext = args[0]
        elif isinstance(args[0], str):
            # The first arg is an extension name; it could still be valid to
            # provide an extver kwarg
            if ext is not None or extname is not None:
                raise TypeError(err_msg)
            extname = args[0]
        else:
            # Take whatever we have as the ext argument; we'll validate it below
            ext = args[0]
    elif len(args) == 2:
        # Must be an extname and extver
        if ext is not None or extname is not None or extver is not None:
            raise TypeError(err_msg)
        extname = args[0]
        extver = args[1]
    elif len(args) > 2:
        raise TypeError("Too many positional arguments.")

    if ext is not None and not (
        isinstance(ext, (int, np.integer))
        or (
            isinstance(ext, tuple)
            and len(ext) == 2
            and isinstance(ext[0], str)
            and isinstance(ext[1], (int, np.integer))
        )
    ):
        raise ValueError(
            "The ext keyword must be either an extension number (zero-indexed) "
            + "or a (extname, extver) tuple."
        )
    if extname is not None and not isinstance(extname, str):
        raise ValueError("The extname argument must be a string.")
    if extver is not None and not isinstance(extver, (int, np.integer)):
        raise ValueError("The extver argument must be an integer.")

    if ext is None and extname is None and extver is None:
        ext = 0
    elif ext is not None and (extname is not None or extver is not None):
        raise TypeError(err_msg)
    elif extname:
        if extver:
            ext = (extname, extver)
        else:
            ext = (extname, 1)
    elif extver and extname is None:
        raise TypeError("extver alone cannot specify an extension.")

    return ext




def _parse_extension_or_none(ext):
    """Return `None` if ext is `None`, otherwise parse it."""
    if ext is None:
        return None
    return _parse_extension(ext)


def _build_ccd_reader_kwargs(
    path,
    extension,
    extension_uncertainty,
    extension_mask,
    extension_flags,
    key_uncertainty_type,
    memmap,
    use_wcs,
    **kwd,
):
    reader_kw = dict(
        hdu=extension,
        hdu_uncertainty=_parse_extension_or_none(extension_uncertainty),
        hdu_mask=_parse_extension_or_none(extension_mask),
        hdu_flags=_parse_extension_or_none(extension_flags),
        key_uncertainty_type=key_uncertainty_type,
        memmap=memmap,
        **kwd,
    )
    # ^ If hdu_flags is not None, CCDData raises this Error:
    #   NotImplementedError: loading flags is currently not supported.

    # FIXME: Remove this if block in the future if WCS issue is resolved.
    if use_wcs:  # Because of the TPV WCS issue
        hdr = fits.getheader(path)
        reader_kw["wcs"] = WCS(hdr)
        del hdr

    return reader_kw


def _load_ccd_astropy(
    path,
    extension,
    trimsec,
    unit,
    extension_uncertainty,
    extension_mask,
    extension_flags,
    key_uncertainty_type,
    memmap,
    use_wcs,
    **kwd,
):
    reader_kw = _build_ccd_reader_kwargs(
        path=path,
        extension=extension,
        extension_uncertainty=extension_uncertainty,
        extension_mask=extension_mask,
        extension_flags=extension_flags,
        key_uncertainty_type=key_uncertainty_type,
        memmap=memmap,
        use_wcs=use_wcs,
        **kwd,
    )

    try:  # Use BUNIT if unit is None
        ccd = CCDData.read(path, unit=unit, **reader_kw)
    except ValueError:  # e.g., user did not give unit and there's no BUNIT
        ccd = CCDData.read(path, unit=u.adu, **reader_kw)

    if trimsec is not None:
        # Do imslice AFTER loading the data to easily add LTV/LTM...
        from ..imops.ccdutils import imslice

        ccd = imslice(ccd, trimsec=trimsec)

    return ccd, reader_kw


def _read_fitsio_extension(hdul, ext, trimsec=None):
    if ext is None:
        return None
    ext = _parse_extension_or_none(ext)
    try:
        if trimsec is not None:
            sl = slicefy(trimsec)
            if is_list_like(ext):
                # length == 2 is already checked in _parse_extension.
                arr = hdul[ext[0], ext[1]].read()[sl]
            else:
                arr = hdul[ext].read()[sl]
        else:
            if is_list_like(ext):
                # length == 2 is already checked in _parse_extension.
                arr = hdul[ext[0], ext[1]].read()
            else:
                arr = hdul[ext].read()
        return arr
    except (OSError, ValueError) as e:
        logger.debug("Error reading extension: %s", e)
        # "Extension `{ext}` is not found (file: `{path}`)")
        return None


def _load_ccd_fitsio(
    path,
    extension,
    trimsec,
    full,
    extension_uncertainty,
    extension_mask,
    extension_flags,
):
    # Use fitsio and only load the data as soon as possible.
    # This is much quicker than astropy's getdata
    with fitsio.FITS(path) as hdul:
        if full:
            dat = _read_fitsio_extension(hdul, extension, trimsec)
            unc = _read_fitsio_extension(hdul, extension_uncertainty, trimsec)
            msk = _read_fitsio_extension(hdul, extension_mask, trimsec)
            flg = _read_fitsio_extension(hdul, extension_flags, trimsec)
            return dat, unc, msk, flg

        return _read_fitsio_extension(hdul, extension, trimsec)


def _astropy_raw_return(
    ccd, full, extension_uncertainty, extension_mask, extension_flags
):
    if full:
        try:
            unc = (
                None
                if extension_uncertainty is None
                else np.array(ccd.uncertainty.array)
            )
        except AttributeError:
            unc = None
        mask = None if extension_mask is None else np.array(ccd.mask.array)
        flag = None if extension_flags is None else np.array(ccd.flags)
        return ccd.data, unc, mask, flag
    return ccd.data


def load_ccd(
    path,
    extension=None,
    trimsec=None,
    ccddata=True,
    use_wcs=True,
    unit=None,
    extension_uncertainty="UNCERT",
    extension_mask="MASK",
    extension_flags=None,
    full=False,
    key_uncertainty_type="UTYPE",
    memmap=False,
    as_ccd=True,  # DEPRECATED
    **kwd,
):
    """Load CCD-like image data from a FITS file.

    Parameters
    ---------
    path : path-like
        The path to the FITS file to load.

    trimsec : `str`, optional.
        Region of `~astropy.nddata.CCDData` from which the data is extracted.
        Default: `None`.

    extension : `int`, `str`, (`str`, `int`), optional.
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.
        Default: `None`.

    ccddata : `bool`, optional.
        Whether to return `~astropy.nddata.CCDData`. Default is `True`. If it
        is `False`, **all the arguments below are ignored**, except for the
        keyword arguments that will be passed to `fitsio.read`, and an `~numpy.ndarray`
        will be returned without astropy unit.

    as_ccd : `bool`, optional.
        Deprecated. (identical to `ccddata`)
        Default: `True`.

    use_wcs : `bool`, optional.
        Whether to load `~astropy.wcs.WCS` by `fits.getheader`, **not** by
        `~astropy.nddata.fits_ccdddata_reader`. This is necessary as of now
        because TPV `~astropy.wcs.WCS` is not properly understood by the latter. It can
        degrade the performance, so if the user is sure the file is **not** in
        TPV, it is recommended to set it to `False`.
        Default : `True`.
        Used only if ``ccddata=True``.

    unit : `~astropy.units.Unit`, optional
        Units of the image data. If this argument is provided and there is a
        unit for the image in the FITS header (the keyword ``BUNIT`` is used as
        the unit, if present), this argument is used for the unit.
        Default: `None`.
        Used only if ``ccddata=True``.

    full : `bool`, optional.
        Whether to return full `(data, unc, mask, flag)` when using
        `fitsio` (i.e., when `ccddata=False`). If `False`(default), only `data`
        will be returned.
        Default: `False`.

    extension_uncertainty : `str` or `None`, optional
        FITS extension from which the uncertainty should be initialized. If the
        extension does not exist the uncertainty is `None`. Name is changed
        from `hdu_uncertainty` in ccdproc to `extension_uncertainty` here. See
        explanation of `extension`.
        Default: ``'UNCERT'``.

    extension_mask : `str` or `None`, optional
        FITS extension from which the mask should be initialized. If the
        extension does not exist the mask is `None`. Name is changed from
        `hdu_mask` in ccdproc to `extension_mask` here.  See explanation of
        `extension`.
        Default: ``'MASK'``.

    extension_flags : `str` or `None`, optional
        Currently not implemented. Name is changed from `hdu_flags` in ccdproc
        to `extension_flags` here.
        Default: `None`.

    key_uncertainty_type : `str`, optional
        The header key name where the class name of the uncertainty is stored
        in the hdu of the uncertainty (if any).
        Default: ``UTYPE``.
        Used only if ``ccddata=True``.

    memmap : `bool`, optional
        Is memory mapping to be used? This value is obtained from the
        configuration item `astropy.io.fits.Conf.use_memmap`.
        Default: `False` (**opposite of astropy**).
        Used only if ``ccddata=True``.

    kwd :
        Any additional keyword parameters that will be used in
        `~astropy.nddata.fits_ccddata_reader` (if ``ccddata=True``) or
        `fitsio.read()` (if ``ccddata=False``).
    """
    try:
        path = Path(path)
    except TypeError:
        raise TypeError(f"You must provide Path-like, not {type(path)}.")

    extension = _parse_extension(extension)

    if HAS_FITSIO and not (ccddata and as_ccd):
        return _load_ccd_fitsio(
            path=path,
            extension=extension,
            trimsec=trimsec,
            full=full,
            extension_uncertainty=extension_uncertainty,
            extension_mask=extension_mask,
            extension_flags=extension_flags,
        )

    e_u = _parse_extension_or_none(extension_uncertainty)
    e_m = _parse_extension_or_none(extension_mask)
    e_f = _parse_extension_or_none(extension_flags)

    ccd, _ = _load_ccd_astropy(
        path=path,
        extension=extension,
        trimsec=trimsec,
        unit=unit,
        extension_uncertainty=extension_uncertainty,
        extension_mask=extension_mask,
        extension_flags=extension_flags,
        key_uncertainty_type=key_uncertainty_type,
        memmap=memmap,
        use_wcs=use_wcs,
        **kwd,
    )

    # Force them to be None if extension is not specified
    # (astropy.NDData.CCDData forces them to be loaded, which is not desirable imho)
    ccd.uncertainty = None if e_u is None else ccd.uncertainty
    ccd.mask = None if e_m is None else ccd.mask

    if ccddata and as_ccd:
        if full:  # Just for API consistency
            return ccd, ccd.uncertainty, ccd.mask, ccd.flags
        return ccd

    return _astropy_raw_return(
        ccd,
        full=full,
        extension_uncertainty=e_u,
        extension_mask=e_m,
        extension_flags=e_f,
    )


def load_ccds(
    paths,
    extension=None,
    trimsec=None,
    ccddata=True,
    as_ccd=True,
    use_wcs=True,
    unit=None,
    extension_uncertainty="UNCERT",
    extension_mask="MASK",
    extension_flags=None,
    full=False,
    key_uncertainty_type="UTYPE",
    memmap=False,
    **kwd,
):
    """Load multiple FITS files with `load_ccd`.

    Parameters
    ---------
    paths : [`list` of] path-like
        The path, glob pattern, or `list` of such, e.g., ``"a.fits"``,
        ``"c*.fits"``, ``["a.fits", "c*.fits"]``

    Notes
    -----
    Timing on MBP 14" [2021, macOS 12.2, M1Pro(6P+2E/G16c/N16c/32G)] using 10
    FITS (each 4.3 MB) with ~ 100 header cards:

    .. code-block:: python

        %timeit ccds = air.load_ccds("h_20191021_000*")
        105 ms +- 2.11 ms per loop (mean +- std. dev. of 7 runs, 10 loops each)
    """
    paths2load = []
    for p in listify(paths):
        paths2load += inputs2list(p, sort=True, accept_ccdlike=False)
    return [
        load_ccd(
            p,
            extension=extension,
            trimsec=trimsec,
            ccddata=ccddata,
            as_ccd=as_ccd,
            use_wcs=use_wcs,
            unit=unit,
            extension_uncertainty=extension_uncertainty,
            extension_mask=extension_mask,
            extension_flags=extension_flags,
            full=full,
            key_uncertainty_type=key_uncertainty_type,
            memmap=memmap,
            **kwd,
        )
        for p in np.array(paths2load).ravel()
    ]


def write2fits(data, header, output, return_ccd=False, **kwargs):
    """Write data and a header to a FITS file via `~astropy.nddata.CCDData`.

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
        Keyword arguments passed to `~astropy.nddata.CCDData.write`, such as
        ``output_verify="fix"`` or ``overwrite=True``.
    """
    ccd = CCDData(data=data, header=header, unit=header.get("BUNIT", "adu"))

    try:
        ccd.write(output, **kwargs)
    except fits.VerifyError:
        logger.warning("Try using output_verify='fix' to avoid this error.")
    if return_ccd:
        return ccd


def inputs2list(
    inputs, sort=True, accept_ccdlike=True, path_to_text=False, check_coherency=False
):
    """Convert paths, globs, tables, or CCD-like objects to a list.

    Parameters
    ----------
    inputs : `str`, path-like, CCD-like, or table-like
        If `~pandas.DataFrame`-convertible, e.g., `dict`, `~pandas.DataFrame` or
        `~astropy.table.Table`, it must have column named ``"file"``, such that
        ``outlist = `list`(inputs["file"])`` is possible. Otherwise, please use,
        e.g., ``inputs = `list`(that_table["filenamecolumn"])``. If a `str` starts
        with ``"@"`` (e.g., ``"@darks.`list`"``), it assumes the file contains a
        `list` of paths separated by ``"\\n"``, as in IRAF.

    sort : `bool`, optional.
        Whether to sort the output `list`.
        Default: `True`.

    accept_ccdlike : `bool`, optional
        Whether to accept `~astropy.nddata.CCDData`-like objects and simply
        return ``[inputs]``.
        Default: `True`.

    path_to_text : `bool`, optional
        Whether to convert the `pathlib.Path` object to `str`.
        Default: `True`.

    check_coherency : `bool`, optional
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
    elif isinstance(inputs, ASTROPY_CCD_TYPES):
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
            if isinstance(item, ASTROPY_CCD_TYPES):
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
