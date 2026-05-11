import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import CCDData, Cutout2D
from astropy.wcs import WCS
from photutils.aperture import (
    Aperture,
    CircularAnnulus,
    CircularAperture,
    EllipticalAnnulus,
    EllipticalAperture,
)

from .pillbox import (
    PillBoxAnnulus,
    PillBoxAperture,
    PillBoxMaskMixin,
    SkyPillBoxAnnulus,
    SkyPillBoxAperture,
)

__all__ = [
    "cutout_from_ap",
    "ap_to_cutout_position",
    "circ_ap_an",
    "ellip_ap_an",
    "pill_ap_an",
    "pa2xytheta",
    "PillBoxMaskMixin",
    "PillBoxAperture",
    "PillBoxAnnulus",
    "SkyPillBoxAperture",
    "SkyPillBoxAnnulus",
]


def cutout_from_ap(
    ap: Aperture,
    ccd: CCDData | np.ndarray,
    method: str = "bbox",
    subpixels: int = 5,
    fill_value: float = np.nan,
) -> Cutout2D | list[Cutout2D]:
    """Returns a Cutout2D object from bounding boxes of aperture/annulus.

    Parameters
    ----------
    ap : `~photutils.aperture.Aperture`
        Aperture or annulus to cut the ccd.

    ccd : `astropy.nddata.CCDData` or ndarray
        The ccd to be cutout.

    method : {"bbox", "exact", "center", "subpixel"}
        The method to cutout.

          * ``"bbox"`` : use the bounding box of the aperture/annulus.
          * {``"exact"``, ``"center"``, ``"subpixel"``} : See `~photutils.aperture.ApertureMask`

        Default is ``"bbox"`` which uses the bounding box to cutout rectangular
        region. Otherwise, ``"center"`` is a reasonable option to cutout circular
        region around the aperture/annulus.

    subpixels : int, optional
        Resolution of the subpixel sampling. Default is ``5``.

    fill_value : float, optional
        Value to fill pixels outside of the aperture. Default is ``np.nan``.

    Note
    ----
    `~photutils.aperture.ApertureMask` has ``.cutout`` and ``.multiply``, but
    they are not "Cutout2D" object. But do I really need Cutout2D instead of
    ndarray?
    """
    # if not isinstance(ccd, CCDData):
    #     ccd = CCDData(ccd, unit="adu")  # dummy unit

    positions = np.atleast_2d(ap.positions)
    cuts = []
    # for ap in np.atleast_1d(ap):
    #     msk = ap.to_mask(method=method, subpixels=subpixels)
    #     if method == "bbox":
    #         cuts.append(msk.cutout(ccd, fill_value=fill_value))
    #     else:
    #         cuts.append(msk.multiply(ccd, fill_value=fill_value))

    bboxes = np.atleast_1d(ap.bbox)
    sizes = [bbox.shape for bbox in bboxes]
    for pos, size in zip(positions, sizes, strict=False):
        cut = Cutout2D(ccd.data, position=pos, size=size)
        if method != "bbox":
            cut.data = ap.to_mask(method, subpixels=subpixels).multiply(
                ccd, fill_value=fill_value
            )
        cuts.append(cut)

    if len(cuts) == 1:
        return cuts[0]
    else:
        return cuts


def ap_to_cutout_position(ap: Aperture, cutout2d: Cutout2D) -> Aperture:
    """Returns a new aperture/annulus only by updating ``positions``.

    Parameters
    ----------
    ap : `~photutils.aperture.Aperture`
        Aperture or annulus to update the ``.positions``.

    cutout2d : `astropy.nddata.Cutout2D`
        The cutout ccd to update ``ap.positions``.
    """
    import copy

    newap = copy.deepcopy(ap)
    pos_old = np.atleast_2d(newap.positions)  # Nx2 positions
    newpos = []
    for pos in pos_old:
        newpos.append(cutout2d.to_cutout_position(pos))
    newap.positions = newpos
    return newap


def _sanitize_apsize(size=None, fwhm=None, factor=None, name="size", repeat=False):
    def __repeat(item, repeat=False, rep=2):
        if repeat and np.isscalar(item):
            return np.repeat(item, rep)
        else:
            return np.atleast_1d(item) if repeat else np.atleast_1d(item)[0]

    if size is None:
        try:
            fwhm = __repeat(fwhm, repeat=repeat, rep=2)
            factor = __repeat(factor, repeat=repeat, rep=2)
            return factor * fwhm
        except TypeError as err:
            raise ValueError(f"{name} is None; fwhm must be given.") from err
    else:
        size = __repeat(size, repeat=repeat, rep=2)
        return size


def circ_ap_an(
    positions,
    r_ap: float | None = None,
    r_in: float | None = None,
    r_out: float | None = None,
    fwhm: float | None = None,
    f_ap: float = 1.5,
    f_in: float = 4.0,
    f_out: float = 6.0,
) -> tuple[CircularAperture, CircularAnnulus]:
    """A convenience function for pixel circular aperture/annulus.

    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

          * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
          * `tuple`, `list`, or `~numpy.ndarray` of ``(x, y)`` pairs
          * `~astropy.units.Quantity` instance of ``(x, y)`` pairs in pixel units

    r_ap, r_in, r_out : float, optional
        The aperture, annular inner, and annular outer radii.

    fwhm : float, optional
        The FWHM in pixel unit.

    f_ap, f_in, f_out: int or float, optional
        The factors multiplied to ``fwhm`` to set the aperture radius, inner
        sky radius, and outer sky radius, respectively. Defaults are ``1.5``,
        ``4.0``, and ``6.0``, respectively, which are de facto standard values
        used by classical IRAF users.

    Returns
    -------
    ap, an : `~photutils.aperture.CircularAperture` and `~photutils.aperture.CircularAnnulus`
        The object aperture and sky annulus.
    """
    r_ap = _sanitize_apsize(r_ap, fwhm=fwhm, factor=f_ap, name="r_ap")
    r_in = _sanitize_apsize(r_in, fwhm=fwhm, factor=f_in, name="r_in")
    r_out = _sanitize_apsize(r_out, fwhm=fwhm, factor=f_out, name="r_out")

    ap = CircularAperture(positions=positions, r=r_ap)
    an = CircularAnnulus(positions=positions, r_in=r_in, r_out=r_out)
    return ap, an


def ellip_ap_an(
    positions,
    r_ap: float | tuple[float, float] | None = None,
    r_in: float | tuple[float, float] | None = None,
    r_out: float | tuple[float, float] | None = None,
    fwhm: float | None = None,
    theta: float = 0.0,
    f_ap: float | tuple[float, float] = (1.5, 1.5),
    f_in: float | tuple[float, float] = (4.0, 4.0),
    f_out: float | tuple[float, float] = (6.0, 6.0),
) -> tuple[EllipticalAperture, EllipticalAnnulus]:
    """A convenience function for pixel elliptical aperture/annulus.

    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        The pixel coordinates of the aperture center(s) in one of the following
        formats::

          * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
          * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs
          * `~astropy.units.Quantity` instance of ``(x, y)`` pairs in pixel units

    r_ap, r_in, r_out: int or float, list or tuple of such, optional
        The aperture, annular inner, and annular outer radii. If list-like, the
        0-th element is regarded as the "semi-major" axis, even though it is
        smaller than the 1-th element. Thus, ``a, b = r_xx[0], r_xx[1]``

    fwhm : float
        The FWHM in pixel unit.

    theta : float, optional
        The rotation angle in radians of the ellipse semimajor axis (0-th
        element of radii or f parameters, not necessarily the longer axis) from
        the positive ``x`` axis.  The rotation angle increases
        counterclockwise.
        Default: ``0``.

    f_ap, f_in, f_out: int or float, list or tuple of such, optional
        The factors multiplied to ``fwhm`` to set the aperture ``a`` and ``b``,
        inner sky ``a`` and ``b``, and outer sky ``a`` and ``b``, respectively.
        If scalar, it is assumed to be identical for both ``a`` and ``b``
        parameters. Defaults are ``(1.5, 1.5)``, ``(4.0, 4.0)``, and ``(6.0,
        6.0)``, respectively, which are de facto standard values used by
        classical IRAF users. If list-like, the 0-th element is regarded as the
        "semi-major" axis, even though it is smaller than the 1-th element.

    Returns
    -------
    ap, an : `~photutils.aperture.EllipticalAperture` and `~photutils.aperture.EllipticalAnnulus`
        The object aperture and sky annulus.
    """
    a_ap, b_ap = _sanitize_apsize(r_ap, fwhm, factor=f_ap, name="r_ap", repeat=True)
    a_in, b_in = _sanitize_apsize(r_in, fwhm, factor=f_in, name="r_in", repeat=True)
    a_out, b_out = _sanitize_apsize(
        r_out, fwhm, factor=f_out, name="r_out", repeat=True
    )

    pt = {"positions": positions, "theta": theta}

    ap = EllipticalAperture(**pt, a=a_ap, b=b_ap)
    an = EllipticalAnnulus(**pt, a_in=a_in, a_out=a_out, b_in=b_in, b_out=b_out)

    return ap, an


def pill_ap_an(
    positions,
    fwhm,
    trail,
    theta=0.0,
    f_ap=(1.5, 1.5),
    f_in=(4.0, 4.0),
    f_out=(6.0, 6.0),
    f_w=1.0,
):
    """A convenience function for pixel pill box aperture/annulus.

    Parameters
    ----------
    positions : array_like or `~astropy.units.Quantity`
        The pixel coordinates of the aperture center(s) in one of the following
        formats:

        * single ``(x, y)`` pair as a tuple, list, or `~numpy.ndarray`
        * tuple, list, or `~numpy.ndarray` of ``(x, y)`` pairs
        * `~astropy.units.Quantity` instance of ``(x, y)`` pairs in pixel units

    fwhm : float
        The FWHM in pixel unit.

    trail : float
        The trail length in pixel unit. The trail is assumed to be extended
        along the ``x`` axis.

    theta : float, optional
        The rotation angle in radians of the ellipse semimajor axis from the
        positive ``x`` axis.  The rotation angle increases counterclockwise.
        Default: ``0``.

    f_ap, f_in, f_out: int or float, list or tuple of such, optional
        The factors multiplied to ``fwhm`` to set the aperture ``a`` and ``b``,
        inner sky ``a`` and ``b``, and outer sky ``a`` and ``b``, respectively,
        for the elliptical component of the pill box. If scalar, it is assumed
        to be identical for both ``a`` and ``b`` parameters. Defaults are
        ``(1.5, 1.5)``, ``(4.0, 4.0)``, and ``(6.0, 6.0)``, respectively, which
        are de facto standard values used by classical IRAF users.

    f_w : int or float
        The factor multiplied to ``trail`` to make a rectangular component of
        the pill box (both `~PillBoxAperture` and `~PillBoxAnnulus`). Note that
        this width is identical for both aperture and annulus.

    Returns
    -------
    ap, an : `~PillBoxAperture` and `~PillBoxAnnulus`
        The object aperture and sky annulus.
    """
    if np.isscalar(fwhm):
        fwhm = np.repeat(fwhm, 2)

    if np.isscalar(f_ap):
        f_ap = np.repeat(f_ap, 2)

    if np.isscalar(f_in):
        f_in = np.repeat(f_in, 2)

    if np.isscalar(f_out):
        f_out = np.repeat(f_out, 2)

    a_ap = f_ap[0] * fwhm[0]
    b_ap = f_ap[1] * fwhm[1]
    a_in = f_in[0] * fwhm[0]
    a_out = f_out[0] * fwhm[0]
    b_out = f_out[1] * fwhm[1]

    w = f_w * trail

    ap = PillBoxAperture(positions=positions, a=a_ap, b=b_ap, w=w, theta=theta)
    an = PillBoxAnnulus(
        positions=positions, a_in=a_in, a_out=a_out, b_out=b_out, w=w, theta=theta
    )
    return ap, an


def eofn_ccw(
    wcs: WCS,
    full: bool = False,
    tol: float = 5.0,
) -> bool | tuple[bool, float, float]:
    """Checks whether the East of North is counter-clockwise in the image.

    Parameters
    ----------
    wcs : `~astropy.wcs.WCS`
        The WCS object.
    full : bool, optional
        If `True`, return the PA of x- and y-axes.
    tol : float, optional
        The tolerance in degrees for the difference of the two PA.
    """
    center = np.array(wcs._naxis) / 2
    coo = SkyCoord(*wcs.wcs_pix2world(*center, 0), unit="deg")
    plusx = wcs.wcs_pix2world(
        *(center + np.array((1, 0))), 0
    )  # basically (CD1_1, CD1_2)
    plusy = wcs.wcs_pix2world(
        *(center + np.array((0, 1))), 0
    )  # basically (CD2_1, CD2_2)
    pa_x = coo.position_angle(SkyCoord(plusx[0], plusx[1], unit="deg")).to_value(u.deg)
    pa_y = coo.position_angle(SkyCoord(plusy[0], plusy[1], unit="deg")).to_value(u.deg)
    dpa = pa_y - pa_x
    if (-270 - tol <= dpa <= -270 + tol) or (90 - tol <= dpa <= 90 + tol):
        # PA (East of North) is CCW in XY coordinate
        if full:
            return True, pa_x, pa_y
        return True
    elif (270 - tol <= dpa <= 270 + tol) or (-90 - tol <= dpa <= -90 + tol):
        # PA (East of North) is CW in XY coordinate
        if full:
            return False, pa_x, pa_y
        return False
    else:
        raise ValueError("PA calculation is problematic.")


def pa2xytheta(
    pa: float,
    wcs: WCS,
    location: str | tuple[float, float] = "crpix",
    step_pix: float = 0.1,
) -> float:
    """
    pa : float
        The position angle in degrees, East of North.
    wcs : `~astropy.wcs.WCS`
        The WCS object.
    location : tuple or str, optional
        The location to convert the position angle. If ``"crpix"``, the
        location is the CRPIX of the WCS. If ``"center"``, the position angle
        is converted at the center of the image. Otherwise, it should be a
        tuple of ``(x, y)`` pixel coordinates.
    step_pix : float, optional
        The step in pixel unit to calculate the Jacobian of the WCS. It should
        be small enough to approximate the local linearity of the WCS, but not
        too small to cause numerical issues. Default is ``0.1`` pixel.

    Return
    ------
    theta: float
        The rotation angle in degrees from the positive ``x`` axis.  The
        angle increases counterclockwise.
    """
    if location == "crpix":
        try:
            location = np.array((wcs.wcs.crpix[0] - 1, wcs.wcs.crpix[1] - 1))
            # coo = SkyCoord(*wcs.wcs.crval, unit="deg")
        except AttributeError as err:
            raise AttributeError(
                "The WCS object does not have CRPIX and/or CRVAL. "
                + "Try with, e.g., `location`='center'."
            ) from err
    elif location == "center":
        location = np.array(wcs._naxis) / 2
        # coo = SkyCoord(*wcs.wcs_pix2world(*location, 0), unit="deg")
    else:
        location = np.array(location)
        # coo = SkyCoord(*wcs.wcs_pix2world(*location, 0), unit="deg")

    x, y = location

    # base world coord
    ra0, dec0 = wcs.all_pix2world(x, y, 0)

    # move slightly in pixel space
    ra_dx, dec_dx = wcs.all_pix2world(x + step_pix, y, 0)
    ra_dy, dec_dy = wcs.all_pix2world(x, y + step_pix, 0)

    # build Jacobian (world per pixel)
    dra_dx = (ra_dx - ra0) / step_pix
    ddec_dx = (dec_dx - dec0) / step_pix
    dra_dy = (ra_dy - ra0) / step_pix
    ddec_dy = (dec_dy - dec0) / step_pix

    # desired sky direction (unit vector in RA/Dec coords)
    pa_rad = np.deg2rad(pa)
    v_sky = np.array(
        [
            np.sin(pa_rad) / np.cos(np.deg2rad(dec0)),  # dRA corrected
            np.cos(pa_rad),  # dDec
        ]
    )

    # invert Jacobian: world -> pixel
    jacob = np.array([[dra_dx, dra_dy], [ddec_dx, ddec_dy]])
    v_pix = np.linalg.solve(jacob, v_sky)
    return np.degrees(np.arctan2(v_pix[1], v_pix[0]))
