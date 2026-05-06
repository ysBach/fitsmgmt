"""FITS image loading helpers."""

from pathlib import Path

import numpy as np
from astro_ndslice import is_list_like, listify, slicefy
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.wcs import WCS

from .logging import logger

try:
    import fitsio

    HAS_FITSIO = True
except ImportError:
    HAS_FITSIO = False

__all__ = ["load_ccd", "load_ccds"]


def _parse_extension_or_none(ext):
    """Return `None` if ext is `None`, otherwise parse it."""
    if ext is None:
        return None
    from .hduutil import _parse_extension

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
        from .hduutil import imslice

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
    """Loads FITS file of CCD image data (not table, etc).

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

    hdu_flags : `str` or `None`, optional
        Currently not implemented.N ame is changed from `hdu_flags` in ccdproc
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
    from .hduutil import _parse_extension

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
    """Simple recursion of `~fitsmgmt.io.load_ccd`

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

        %timeit ccds = fm.load_ccds("h_20191021_000*")
        105 ms +- 2.11 ms per loop (mean +- std. dev. of 7 runs, 10 loops each)
    """
    from .hduutil import inputs2list

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
