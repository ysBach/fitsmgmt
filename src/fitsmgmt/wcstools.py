"""WCS helper utilities from the raw hduutil port."""

import re

import erfa
import numpy as np
from astro_ndslice import listify
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.wcs import WCS, Wcsprm

from .logging import logger

__all__ = ["wcs_crota", "center_radec", "fov_radius", "wcsremove", "pixel_scale"]


def wcs_crota(wcs, degree=True):
    """

    Notes
    -----
    https://iraf.net/forum/viewtopic.php?showtopic=108893
    CROTA2 = arctan (-CD1_2 / CD2_2) = arctan ( CD2_1 / CD1_1)
    """
    if isinstance(wcs, WCS):
        wcsprm = wcs.wcs
    elif isinstance(wcs, Wcsprm):
        wcsprm = wcs
    else:
        raise TypeError(
            "wcs type not understood. "
            + "It must be either ~astropy.wcs.WCS or astropy.wcs.Wcsprm"
        )

    # numpy arctan2 gets y-coord (numerator) and then x-coord(denominator)
    crota = np.arctan2(wcsprm.cd[0, 0], wcsprm.cd[1, 0])
    if degree:
        crota = np.rad2deg(crota)

    return crota


def center_radec(
    ccd_or_header,
    center_of_image=True,
    ra_key="RA",
    dec_key="DEC",
    equinox=None,
    frame=None,
    equinox_key="EPOCH",
    frame_key="RADECSYS",
    ra_unit=u.hourangle,
    dec_unit=u.deg,
    mode="all",
    verbose=True,
    plain=False,
):
    """Returns the central ra/dec from header or `~astropy.wcs.WCS`.

    Notes
    -----
    Even though RA or DEC is in sexagesimal, e.g., "20 53 20", astropy
    correctly reads it in such a form, so no worries.

    Parameters
    ----------
    ccd_or_header : CCD-like, `~astropy.io.fits.Header`
        The ccd or header to extract the central RA/DEC from keywords or `~astropy.wcs.WCS`.

    center_of_image : `bool`, optional
        If `True`, `~astropy.wcs.WCS` information will be extracted from the ccd or header,
        rather than relying on the `ra_key` and `dec_key` keywords directly. If
        `False`, `ra_key` and `dec_key` from the header will be understood as
        the "center" and the RA, DEC of that location will be returned.

    equinox, frame : `str`, optional
        The `equinox` and `frame` for SkyCoord. Default (`None`) will use the
        default of SkyCoord. Important only if ``usewcs=False``.
        Default: `True`.

    XX_key : `str`, optional
        The header key to find XX if ``XX`` is `None`. Important only if
        ``usewcs=False``.

    XX_unit : `~astropy.units.Quantity`, optional
        The unit of ``XX``. Important only if ``usewcs=False``.

    mode : 'all' or 'wcs', optional
        Whether to do the transformation including distortions (``'all'``) or
        only including only the core `~astropy.wcs.WCS` transformation (``'wcs'``). Important
        only if ``usewcs=True``.
        Default: ``'all'``.

    plain : `bool`, optional.
        If `True`, only the values of RA/DEC in degrees will be returned.
        Default: `False`.
    """
    from .headers import get_from_header

    if isinstance(ccd_or_header, CCDData):
        header = ccd_or_header.header
        w = ccd_or_header.wcs
    elif isinstance(ccd_or_header, fits.Header):
        header = ccd_or_header
        w = WCS(header)

    if center_of_image:
        nx, ny = float(header["NAXIS1"]), float(header["NAXIS2"])
        centx = nx / 2 - 0.5
        centy = ny / 2 - 0.5
        coo = SkyCoord.from_pixel(centx, centy, wcs=w, origin=0, mode=mode)
    else:
        ra = get_from_header(header, ra_key, verbose=verbose)
        dec = get_from_header(header, dec_key, verbose=verbose)
        if equinox is None:
            equinox = get_from_header(
                header, equinox_key, verbose=verbose, default=None
            )
        if frame is None:
            frame = get_from_header(
                header, frame_key, verbose=verbose, default=None
            ).lower()
        coo = SkyCoord(
            ra=ra, dec=dec, unit=(ra_unit, dec_unit), frame=frame, equinox=equinox
        )

    if plain:
        return coo.ra.value, coo.dec.value
    return coo


def fov_radius(header=None, wcs=None, unit=u.deg):
    """Calculates the rough radius (cone) of the (square) FOV using `~astropy.wcs.WCS`.

    Parameters
    ----------
    header : `~astropy.io.fits.Heade`r, optional.
        The header to extract `~astropy.wcs.WCS` information.
        Default: `None`.

    wcs : `~astropy.wcs.WCS`, optional.
        The `~astropy.wcs.WCS` to extract the information. If `None`, it will be extracted
        from `header`.
        Default: `None`.

    Returns
    -------
    radius: `~astropy.Quantity`
        The radius in degrees
    """
    w = WCS(header) if wcs is None else wcs
    nx, ny = float(header["NAXIS1"]), float(header["NAXIS2"])
    # Rough calculation, so use mode='wcs'
    c1 = SkyCoord.from_pixel(0, 0, wcs=w, origin=0, mode="wcs")
    c2 = SkyCoord.from_pixel(nx, 0, wcs=w, origin=0, mode="wcs")
    c3 = SkyCoord.from_pixel(0, ny, wcs=w, origin=0, mode="wcs")
    c4 = SkyCoord.from_pixel(nx, ny, wcs=w, origin=0, mode="wcs")

    # TODO: Can't we just do ``return max(r1, r2).to(unit)``???
    #   Why did I do this? I can't remember...
    #   2020-11-09 14:29:29 (KST: GMT+09:00) ysBach
    r1 = c1.separation(c3).value / 2
    r2 = c2.separation(c4).value / 2
    r = max(r1, r2) * u.deg
    return r.to(unit)


def _parse_wcsremove_extension(extension):
    if extension is None:
        return 0
    if isinstance(extension, (int, np.integer)):
        return extension
    if (
        isinstance(extension, tuple)
        and len(extension) == 2
        and isinstance(extension[0], str)
        and isinstance(extension[1], (int, np.integer))
    ):
        return extension
    if isinstance(extension, str):
        return (extension, 1)
    raise ValueError(
        "The extension must be an integer, an EXTNAME string, or an (EXTNAME, EXTVER) tuple."
    )


# TODO: do not load data extension if not explicitly ordered
def wcsremove(
    path_or_header=None,
    additional_keys=None,
    ccddata=True,
    extension=None,
    output=None,
    output_verify="fix",
    overwrite=False,
    checksum=False,
    verbose=True,
):
    """Remove most `~astropy.wcs.WCS` related keywords from the header.

    Parameters
    ---------
    path_or_header : `str`, `~astropy.io.fits.Header`, optional.
        The path to the FITS file, or the header to be modified. If it is
        header, `ccddata`, `extension`, `output`, `output_verify`, `overwrite`,
        and `checksum` will be ignored.
        Default: `None`.

    additional_keys : `list` of regex `str`, optional
        Additional keys given by the user to be 'reset'. It must be in regex
        expression. Of course regex accepts just string, like 'NAXIS1'.
        Default: `None`.

    ccddata : `bool`, optional.
        Whether to return `~astropy.nddata.CCDData`. Default is `True`. If
        `False`, it will return `~astropy.io.fits.PrimaryHDU` of the
        `extension`.

        .. note::
            If there is no need to use the returned astropy.nddata.CCDData, it is better to
            set `ccddata=False` to improve the performance.

        .. warning::
            The returned astropy.nddata.CCDData will have ccd.wcs as None, while if the
            saved output is read by `~astropy.nddata.CCDData.read(filename)`, it will have
            the proper ccd.wcs.

    extension : `int`, `str`, (`str`, `int`), optional.
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.
        Default: `None`.

    output : `str` or `~pathlib.Path`, optional.
        The output file path.
        Default: `None`.

    output_verify : `str`, optional.
        Output verification option.  Must be one of ``"fix"``, ``"silentfix"``,
        ``"ignore"``, ``"warn"``, or ``"exception"``.  May also be any
        combination of ``"fix"`` or ``"silentfix"`` with ``"+ignore"``,
        ``+warn``, or ``+exception" (e.g. ``"fix+warn"``).  See
        :ref:`astropy:verify` for more info.
        Default: ``'fix'``.

    overwrite : `bool`, optional
        If `True`, overwrite the output file if it exists. Raises an `OSError`
        if `False` and the output file exists. Default is `False`.

    checksum : `bool`, optional
        If `True`, adds both ``DATASUM`` and ``CHECKSUM`` cards to the headers
        of all HDU's written to the file.
        Default: `False`.

    Notes
    -----
    For ``fm.wcsremove("test.fit")`` with a simple 33.6MB FITS file (71
    keywords, 20 `~astropy.wcs.WCS`-related keywords, 5 COMMENTs) on MBP 14" [2021, macOS
    13.1, M1Pro(6P+2E/G16c/N16c/32G)]:

    .. code-block:: text

        V A C
        O X O = 10.6 ± 0.2 ms (DEFAULT)
        X X O = 10.8 ± 0.3 ms (almost no benefit)  - effect of verbose

    Here::

        * A : additional_keys=["COMMEnT"]
        * V : verbose=True
        * C : ccddata=True

    With `additional_keys` (the payoff is not that big):

    .. code-block:: text

        V A C
        O O O = 11.2 ± 0.4 ms
        X O O = 10.6 ± 0.2 ms

    Return `~astropy.io.fits.PrimaryHDU` without converting to `~astropy.nddata.CCDData` (almost 5x faster):

    .. code-block:: text

        V A C
        X X X =  1.9 ± 0.0 ms
        X O X =  2.1 ± 0.0 ms

    The time it takes to parse the header and open the file is 0.4 ms, so the
    key removal part is changed from ~ 10 ms to ~ 1.5 ms (6-7 times faster)


    """
    # Define header keywords to be deleted in regex:
    re2remove = [
        # Coordinate system (widely used)
        "CD[0-9]_[0-9]",  # Coordinate Description matrix
        "CTYPE[0-9]",  # e.g., 'RA---TAN' and 'DEC--TAN'
        "C[0-9]YPE[0-9]",  # FOCAS
        "CUNIT[0-9]",  # e.g., 'deg'
        "C[0-9]NIT[0-9]",  # FOCAS
        "CRPIX[0-9]",  # The reference pixels in image coordinate
        "C[0-9]PIX[0-9]",  # FOCAS
        "CRVAL[0-9]",
        "C[0-9]VAL[0-9]",  # FOCAS
        "CDELT[0-9]",  # with CROTA, older version of CD matrix.
        "C[0-9]ELT[0-9]",  # FOCAS
        "CROTA[0-9]",
        "CRDELT[0-9]",
        "CFINT[0-9]",
        "WAT[0-9]_[0-9]",  # For TNX and ZPX, e.g., "WAT1_001"
        "C0[0-9]_[0-9]",  # polynomial CD by imwcs
        "P[C,V,S][0-9]_[0-9]",  # coordinate transformation
        "P[A-Z]?[0-9]?[0-9][0-9][0-9][0-9][0-9][0-9]",  # obsolete PC notation
        "RADE[C]?SYS*",
        "LONPOLE",
        "LONGPOLE",
        "LATPOLE",
        "EQUINOX",
        "EPOCH",  # not sure if this is safe to remove
        "WCS[A-Z]",  # see below
        "CRDER[0-9]",  # Coord. RanDom ERror (WCS paperI Sect 2.6)
        "CSYER[0-9]",  # Coord. RanDom ERror (WCS paperI Sect 2.6)
        # "MJD-OBS",  # I think we can just keep it there...?
        # Physical
        "LTM[0-9]_[0-9]",  # for PHYSICAL
        "LTV[0-9]*",  # for PHYSICAL
        # Others, usually added by WCS softwares
        "WCS-ORIG",  # RA/DEC system (frame)  # FOCAS
        "PIXXMIT",
        "PIXOFFST",
        "[A,B][P]?_[0-9]_[0-9]",  # astrometry.net
        "[A,B][P]?_ORDER",  # astrometry.net
        "[A,B][P]?_DMAX",  # astrometry.net
        "AST_[A-Z]",  # astrometry.net
        "ASTIRMS[0-9]",  # astrometry.net
        "ASTRRMS[0-9]",  # astrometry.net
        "PLTSOLVD",  # ASTAP
        "FGROUPNO",  # SCAMP field group label
        "ASTINST",  # SCAMP astrometric instrument label
        "FLXSCALE",  # SCAMP relative flux scale
        "MAGZEROP",  # SCAMP zero-point
        "PHOTIRMS",  # mag dispersion RMS (internal, high S/N)
        "PHOTINST",  # SCAMP photometric instrument label
        "PHOTLINK",  # True if linked to a photometric field
        "SECPIX[0-9]",
    ]
    # WCS[A-Z] captures, e.g., WCS[AXES, DIM, NAME, RFCAT, IMCAT, MATCH, NREF,
    # TOL, SEP], but not [IM]WCS, for example. These are likely to have been
    # inserted by WCS updating tools like astrometry.net or WCSlib/WCSTools. I
    # intentionally ignored IMWCS just for future reference.

    if additional_keys is not None:
        re2remove += [k.upper() for k in listify(additional_keys)]

    # If following str is in comment, suggest it if verbose
    candidate_re = ["wcs", "axis", "axes", "coord", "distortion", "reference"]
    candidate_key = []

    removed_keys = []  # Collect removed keys for logging

    if verbose:
        logger.info("Removed keywords: ")

    if isinstance(path_or_header, fits.Header):
        hdu = None
        hdr = path_or_header.copy()
    else:
        extension = _parse_wcsremove_extension(extension)
        with fits.open(path_or_header) as hdul:
            hdu = hdul[extension].copy()
        hdr = hdu.header

    for k in list(hdr.keys()):
        com = hdr.comments[k]
        deleted = False
        for re_i in re2remove:
            if re.match(re_i, k) is not None and not deleted:
                hdr.remove(k)
                deleted = True
                removed_keys.append(k)
                continue
        if not deleted and com:  # do only if com != ""
            for re_cand in candidate_re:
                if re.match(re_cand, com):
                    candidate_key.append(k)
                    break  # break here for minor performance boost
    if verbose:
        if removed_keys:
            logger.info("%s", " ".join(removed_keys))
        if len(candidate_key) != 0:
            logger.info("Following keys may be related to WCS too: %s", candidate_key)

    if hdu is None:
        return hdr  # Do not save. Do not try to return CCDData.

    if output is not None:
        hdu.writeto(
            output, output_verify=output_verify, overwrite=overwrite, checksum=checksum
        )

    return (
        hdu
        if not ccddata
        else CCDData(
            data=hdu.data,
            header=hdu.header,
            unit=hdu.header.get("BUNIT", default="adu"),
        )
    )


def pixel_scale(header=None, wcs=None, unit=u.arcsec, position=None):
    """Calculates the rough pixel scale using `~astropy.wcs.WCS`.

    Parameters
    ----------
    header : `~astropy.io.fits.Heade`r, optional.
        The header to extract `~astropy.wcs.WCS` information.
        It is used when `wcs` is `None` or `position` is `"physical"`.
        Default: `None`.

    wcs : `~astropy.wcs.WCS`, optional.
        The `~astropy.wcs.WCS` to extract the information. If `None`, it will be extracted
        from `header`.
        Default: `None`.

    unit : astropy unit, optional.
        The desired output unit. Default is arcsec. If `None`, the output will be
        in radians.

    position : `tuple` of `float`, optional
        The position (x, y) in pixel coordinates to calculate the pixel scale.
        If `None` (default), the center of the image will be used.
        If `"physical"`, the physical center, i.e., the center of the image
        minus LTV1 and LTV2 will be used.
        Default: `None`.

    Returns
    -------
    pscale: `~astropy.Quantity` or `float`
        The pixel scale in `unit`/pixel. If `unit` is `None`, it will be in radians.
    """
    if wcs is None:
        w = WCS(header)
        nx, ny = float(header["NAXIS1"]), float(header["NAXIS2"])
    else:
        w = wcs
        nx, ny = w.array_shape[::-1]  # w.array_shape is (ny, nx)

    if position is None:
        x, y = nx / 2 - 0.5, ny / 2 - 0.5
    elif position == "physical":
        x, y = nx / 2 - 0.5 - float(header.get("LTV1", 0)), ny / 2 - 0.5 - float(
            header.get("LTV2", 0)
        )
    else:
        x, y = position

    # pixel_to_world_values internally uses self._all_pix2world
    lons, lats = np.deg2rad(
        w.pixel_to_world_values([x - 0.5, x + 0.5], [y - 0.5, y + 0.5])
    )
    pscale = (erfa.seps(lons[0], lats[0], lons[1], lats[1])) / np.sqrt(2)

    if unit is None:
        return pscale

    return (pscale << u.rad).to(unit)
