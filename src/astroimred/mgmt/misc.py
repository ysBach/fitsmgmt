"""
Objects that are
(1) too fundamental, so used in various places,
(2) completely INDEPENDENT of all other modules of this package.
"""

import numpy as np
from astro_ndslice import listify
from astropy import units as u
from astropy.time import Time

from .logging import logger

try:
    import numba as nb
except ImportError:
    class _NoNumba:
        def njit(self, *args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]

            def _decorator(func):
                return func

            return _decorator

        @staticmethod
        def prange(*args):
            return range(*args)

    nb = _NoNumba()

__all__ = [
    "MEDCOMB_KEYS_INT",
    "SUMCOMB_KEYS_INT",
    "MEDCOMB_KEYS_FLT32",
    "sigclip_dataerr",
    "circular_mask",
    "circular_mask_2d",
    "enclosing_circle_radius",
    "str_now",
    "change_to_quantity",
]


MEDCOMB_KEYS_INT = dict(
    dtype="int16",
    combine_method="median",
    reject_method=None,
    unit=u.adu,
    combine_uncertainty_function=None,
)

SUMCOMB_KEYS_INT = dict(
    dtype="int16",
    combine_method="sum",
    reject_method=None,
    unit=u.adu,
    combine_uncertainty_function=None,
)

MEDCOMB_KEYS_FLT32 = dict(
    dtype="float32",
    combine_method="median",
    reject_method=None,
    unit=u.adu,
    combine_uncertainty_function=None,
)

# !FIXME: not finished
# TODO: add err_lower, err_upper, sigma_lower, sigma_upper
def sigclip_dataerr(val, err, cenfunc="wvg", sigma=3, maxiters=3):
    """Sigma-clip values using per-point error estimates.

    Parameters
    ----------
    val, err : array-like
        Values and corresponding 1-sigma errors.

    cenfunc : {"wvg", "avg", "average", "mean"}, optional
        Center estimator. ``"wvg"`` uses weighted average.

    sigma : float, optional
        Rejection threshold in units of `err`.

    maxiters : int, optional
        Maximum clipping iterations.
    """
    if cenfunc == "wvg":
        from ..imops.mathutils import weighted_avg

        cenfunc = lambda val, err: weighted_avg(val, err)[0]
    elif cenfunc in ["avg", "average", "mean"]:
        cenfunc = lambda val, err: np.mean(val)[0]  # err is dummy
    else:
        raise ValueError(f"cenfunc={cenfunc} is not implemented yet.")

    val = np.ma.array(val)
    val_clipped = val.compressed()
    err_clipped = err[val.mask]
    cen = cenfunc(val_clipped, err_clipped)

    for i in range(maxiters):
        # calculate deviation for all (even masked) elements:
        deviation = np.abs(val.data - cen)
        mask = deviation > sigma * err

    return val, mask


def circular_mask(shape, center=None, radius=None, center_xyz=True):
    """Creates an N-D circular (circular, spherical, ...) mask.

    Parameters
    ----------
    shape : `tuple`
        The pythonic shape, i.e., `arr.shape` (not xyz order).

    center : `tuple`, `None`, optional.
        The center of the circular mask. If `None` (default), the central
        position is used.
        Default: `None`.

    radius : `float`, `None`, optional.
        The radius of the mask. If `None`, the distance to the closest edge of
        the image is used.
        Default: `None`.

    center_xyz : `bool`, optional.
        Whether the center is in xyz order.
        Default: `True`.

    Notes
    -----
    Idea copied from
    https://stackoverflow.com/questions/44865023/how-can-i-create-a-circular-mask-for-a-numpy-array

    Note that this is slow due to the "general" N-D nature of the mask.
    If you need a 2-D mask, use `circular_mask_2d`
    """
    if center is None:  # use the middle of the image
        center = [npix / 2 for npix in shape[::-1]]

    if center_xyz:
        center = center[::-1]

    shape = np.array(shape)
    center = np.array(center)

    if radius is None:  # use the smallest distance between the center and image walls
        radius = np.min([center, shape - center])

    slices = tuple([slice(None, npix, None) for npix in shape])

    zyx = np.ogrid[slices]
    dist_sq = [((zyx[i] - center[i]) ** 2) for i in range(len(shape))]
    dist_from_center = np.sqrt(np.sum(np.array(dist_sq, dtype=object)))

    mask = dist_from_center <= radius
    return mask


def circular_mask_2d(
    shape,
    center=None,
    radius=0.5,
    method="center",
    subpixels=5,
    maskmin=0,
    return_apertures=False,
):
    """Creates a 2-D circular mask using photutils CircularAperture.

    Parameters
    ----------
    shape : `tuple`
        The shape of the 2-D image in *pythonic* order, i.e., `arr.shape`
        (height, width).

    center : array-like, `None`, optional.
        The pixel coordinates of the aperture center(s) in one of the
        following formats:

        * single ``(x, y)`` pair as a `tuple`, `list`, or `~numpy.ndarray`
        * `tuple`, `list`, or `~numpy.ndarray` of ``(x, y)`` pairs

        If `None`, the center is set to the middle of the image, i.e.,
        ``(shape[0] / 2, shape[1] / 2)``.
        Default is `None`.

    radius : `float`, array-like optional.
        The radius (radii) of the circular mask(s).
        Default: ``0.5``.

    method : ``{'exact', 'center', 'subpixel'}``, optional
        The method used to determine the overlap of the aperture on the pixel
        grid. Not all options are available for all aperture types. Note that
        the more precise methods are generally slower. The following methods
        are available:

        * ``'exact'`` (default):
        The exact fractional overlap of the aperture and each pixel is
        calculated. The aperture weights will contain values between 0 and 1.

        * ``'center'``:
        A pixel is considered to be entirely in or out of the aperture
        depending on whether its center is in or out of the aperture. The
        aperture weights will contain values only of 0 (out) and 1 (in).

        * ``'subpixel'``:
        A pixel is divided into subpixels (see the ``subpixels`` keyword), each
        of which are considered to be entirely in or out of the aperture
        depending on whether its center is in or out of the aperture. If
        ``subpixels=1``, this method is equivalent to ``'center'``. The
        aperture weights will contain values between 0 and 1.
        Default: ``'center'``.

    subpixels : `int`, optional
        For the ``'subpixel'`` method, resample pixels by this factor in each
        dimension. That is, each pixel is divided into ``subpixels**2``
        subpixels. This keyword is ignored unless ``method='subpixel'``.
        Default: ``5``.

    maskmin : `float`, optional
        The minimum value for the mask. If the aperture weights are greater
        than this value, the pixel is considered to be in the aperture. This
        keyword is ignored unless ``method='exact'`` or ``method='subpixel'``.
        Default: ``0``.

    return_apertures : `bool`, optional
        If `True`, return the `CircularAperture` objects and the masks
        instead of the 2D mask. This is useful if you want to use the
        Default: `False`.
    """
    try:
        from photutils.aperture import CircularAperture
    except ImportError:
        if method != "center" or return_apertures:
            raise
        if center is None:
            center = (shape[0] / 2, shape[1] / 2)
        yy, xx = np.indices(shape)
        apmask2d = np.zeros(shape, dtype=bool)
        centers = np.atleast_2d(center)
        radii = np.atleast_1d(radius)
        if radii.size == 1:
            radii = np.repeat(radii, len(centers))
        for c, r in zip(centers, radii):
            apmask2d |= (xx - c[0]) ** 2 + (yy - c[1]) ** 2 < r**2
        return apmask2d

    if center is None:  # use the middle of the image
        center = (shape[0] / 2, shape[1] / 2)

    try:
        apertures = CircularAperture(center, radius)
        apmasks = apertures.to_mask(method=method, subpixels=subpixels)
        if not isinstance(apmasks, list):
            apmasks = [apmasks]
    except ValueError:
        # multiple radii and "ValueError: 'r' must be a positive scalar" happens.
        if center.shape[0] != np.size(radius):
            raise ValueError(
                "If `radius` is an array-like, it must have the same length as `center`; "
                f"({center.shape[0] = }) != ({np.size(radius)} = )."
            )
        apertures = [CircularAperture(c, r=r) for c, r in zip(center, radius)]
        apmasks = [ap.to_mask(method=method, subpixels=subpixels) for ap in apertures]

    apmask2d = np.zeros(shape, dtype=bool)

    if method == "center":
        # Use the center of the pixel to determine if it is in the aperture
        for m in apmasks:
            apmask2d |= m.to_image(shape, dtype=bool)

    elif method == "exact" or method == "subpixel":
        # Use the exact overlap of the aperture and each pixel
        for m in apmasks:
            apmask2d |= m.to_image(shape, dtype=float) > maskmin
    else:
        raise ValueError(
            f"Method {method} not supported. Use 'exact', 'center', or 'subpixel'."
        )

    return apmask2d


@nb.njit(fastmath=False, parallel=True)
def _enclosing_circle_radius(segm, center, segm_id, output):

    for i in nb.prange(len(segm_id)):
        _segm_id = segm_id[i]
        mask = segm == _segm_id
        y, x = np.nonzero(mask)

        # if center is None:
        #     # Calculate the centroid of the masked region
        #     center = (np.mean(x), np.mean(y))

        # Calculate the distances from the center to all non-zero pixels
        rsq_max = np.max((x - center[i][0]) ** 2 + (y - center[i][1]) ** 2)
        output[i] = np.sqrt(rsq_max)


def enclosing_circle_radius(segm, center, segm_id=None):
    """
    Calculate the radius of the smallest enclosing circle for a given mask.

    Parameters
    ----------
    segm : 2D array-like
        The input segmentation map (binary image) where non-zero values are
        considered as the region of interest.

    center : 2-D array, optional
        The (x, y) coordinates of the center of the circles. If not provided,
        the center will be calculated as the centroid of the masked region.

    segm_id : `list` of `int`, optional
        The `list` of segmentation IDs to calculate the radius for. If not provided,
        it defaults to `[1]`, which is equivalent to `True` for binary masks.
        Default: `None`.

    Returns
    -------
    `~numpy.ndarray`
        The radius of the smallest enclosing circle.

    Notes
    -----
    Since it calculates distances from the center to the pixel center, one may
    want to add ~0.5 (or sqrt(2)*0.5) to enclose the full pixel area.

    By using numba, single segmentation radius finding is ~5 times faster than
    pure numpy, and it is boosted further if `parallel=True` is used.
    """

    if segm_id is None:
        segm_id = np.array([1], dtype=segm.dtype)  # same as `True`

    center = np.atleast_2d(center)
    if center.shape[1] != 2:
        raise ValueError("Center must be a 2D array with shape (N, 2)")

    radii = np.empty(len(segm_id), dtype=np.float64)

    _enclosing_circle_radius(segm, center, segm_id, radii)

    return radii


def str_now(
    precision=3, fmt="{:.>72s}", t_ref=None, dt_fmt="(dt = {:.3f} s)", return_time=False
):
    """Get stringified time now in UT ISOT format.

    Parameters
    ----------
    precision : `int`, optional.
        The precision of the isot format time.
        Default: ``3``.

    fmt : `str`, optional.
        The Python 3 format string to format the time. Examples::

          * ``"{:s}"``: plain time ``2020-01-01T01:01:01.23``
          * ``"({:s})"``: plain time in parentheses ``(2020-01-01T01:01:01.23)``
          * ``"{:_^72s}"``: center align, filling with _.
        Default: ``'{:.>72s}'``.

    t_ref : `~astropy.time.Time`, optional.
        The reference time. If not `None`, delta time is calculated.
        Default: `None`.

    dt_fmt : `str`, optional.
        The Python 3 format string to format the delta time.
        Default: ``'(dt = {:.3f} s)'``.

    return_time : `bool`, optional.
        Whether to return the time at the start of this function and the delta
        time (`dt`), as well as the time information string. If `t_ref` is
        `None`, `dt` is automatically set to `None`.
        Default: `False`.
    """
    now = Time(Time.now(), precision=precision)
    timestr = now.isot
    if t_ref is not None:
        dt = (now - Time(t_ref)).sec  # float in seconds unit
        timestr = dt_fmt.format(dt) + " " + timestr
    else:
        dt = None

    if return_time:
        return fmt.format(timestr), now, dt
    else:
        return fmt.format(timestr)


def change_to_quantity(x, desired="", to_value=False):
    """Convert an object to `~astropy.units.Quantity`, or to a scalar value.

    Parameters
    ----------
    x : object convertible to `~astropy.units.Quantity`
        Input value. If a `~astropy.units.Quantity` is given, `x` is converted
        to `desired`, i.e., ``x.to(desired)``.

    desired : `str` or astropy `~astropy.units.Unit`, optional.
        The desired unit for `x`. If `''` (default), it will be interpreted as
        `Unit(dimensionless)`.
        Default: ``''``.

    to_value : `bool`, optional.
        Whether to return as scalar value. If `True`, just the value(s) of the
        `desired` unit will be returned after conversion.
        Default: `False`.

    Returns
    -------
    ux: `~astropy.units.Quantity`

    Notes
    -----
    If `~astropy.units.Quantity`, transform to `desired`. If `desired` is `None`, return it as
    is. If not `Quantity`, multiply the `desired`. `desired` is `None`, return
    `x` with dimensionless unscaled unit.
    """

    def _copy(xx):
        try:
            xcopy = xx.copy()
        except AttributeError:
            import copy

            xcopy = copy.deepcopy(xx)
        return xcopy

    if x is None:
        return None

    try:
        ux = x.to(desired).value if to_value else x.to(desired)
    except AttributeError:  # if not Quantity
        if not to_value:
            if isinstance(desired, str):
                desired = u.Unit(desired)
            try:
                ux = x * desired
            except TypeError:
                ux = _copy(x)
        else:
            ux = _copy(x)
    except TypeError:
        ux = _copy(x)
    except u.UnitConversionError:
        raise ValueError(
            "If you use astropy.Quantity, you should use unit convertible to `desired`."
            + f'\nYou gave "{x.unit}", unconvertible with "{desired}".'
        )

    return ux
