import numpy as np
from astropy import units as u
from photutils.aperture import (
    ApertureMask,
    EllipticalAperture,
    PixelAperture,
    RectangularAperture,
    SkyAperture,
)
from photutils.aperture.attributes import (
    PixelPositions,
    PositiveScalar,
    PositiveScalarAngle,
    ScalarAngle,
    ScalarAngleOrValue,
    SkyCoordPositions,
)

__all__ = [
    "PillBoxMaskMixin",
    "PillBoxAperture",
    "PillBoxAnnulus",
    "SkyPillBoxAperture",
    "SkyPillBoxAnnulus",
]

# Pill-Box aperture related str (base descriptions):
_PBSTRS = dict(
    w="trailed distance of the pillbox",
    a="semimajor axis of ellipse part (parallel to the trail direction)",
    b="semiminor axis of ellipse part (perpendicular to the trail direction)",
    theta_pix=(
        "The counterclockwise rotation angle as an angular Quantity or value in "
        + "radians from the positive x axis."
    ),
    theta_sky="The position angle in angular units of the trail direction.",
)


class PillBoxMaskMixin:
    def _define_patch_params(self, origin=(0, 0), **kwargs):
        """Define patch positions and keyword arguments."""
        origin = np.array(origin)
        xy_positions = np.atleast_2d(self.positions) - origin
        return xy_positions, kwargs

    @staticmethod
    def _make_annulus_path(inner_patch, outer_patch):
        """Make a compound path for an annulus patch."""
        import matplotlib.path as mpath

        outer_path = outer_patch.get_path()
        inner_path = inner_patch.get_path()

        inner_vertices = inner_path.vertices[::-1]
        inner_codes = np.full(len(inner_vertices), mpath.Path.LINETO)
        inner_codes[0] = mpath.Path.MOVETO
        inner_codes[-1] = mpath.Path.CLOSEPOLY

        return mpath.Path(
            np.concatenate([outer_path.vertices, inner_vertices]),
            np.concatenate([outer_path.codes, inner_codes]),
        )

    @property
    def _set_aperture_elements(self):
        """Set internal aperture elements.
        ``self._ap_rect``, ``self.ap_el_1``, ``self.ap_el_2`` and their ``_in``
        counterparts are always made by ``np.atleast_2d(self.position)``, so
        their results are always in the ``N x 2`` shape.
        """
        if hasattr(self, "a"):
            w = self.w
            a = self.a
            b = self.b
            h = self.h
            theta = self.theta
        elif hasattr(self, "a_in"):  # annulus
            w = self.w
            a = self.a_out
            b = self.b_out
            h = self.h_out
            theta = self.theta
        else:
            raise ValueError("Cannot determine the aperture shape.")

        # positions only accepted in the shape of (N, 2), so shape[0]
        # gives the number of positions:
        pos = np.atleast_2d(self.positions)
        self.offset = np.array([w * np.cos(theta) / 2, w * np.sin(theta) / 2])
        offsets = np.repeat(
            np.array(
                [
                    self.offset,
                ]
            ),
            pos.shape[0],
            0,
        )

        # aperture elements for aperture,
        # OUTER aperture elements for annulus:
        self._ap_rect = RectangularAperture(positions=pos, w=w, h=h, theta=theta)
        self._ap_el_1 = EllipticalAperture(
            positions=pos - offsets, a=a, b=b, theta=theta
        )
        self._ap_el_2 = EllipticalAperture(
            positions=pos + offsets, a=a, b=b, theta=theta
        )

        if hasattr(self, "a_in"):  # inner components of annulus
            self._ap_rect_in = RectangularAperture(
                positions=pos, w=self.w, h=self.h_in, theta=self.theta
            )
            self._ap_el_1_in = EllipticalAperture(
                positions=pos - offsets, a=self.a_in, b=self.b_in, theta=self.theta
            )
            self._ap_el_2_in = EllipticalAperture(
                positions=pos + offsets, a=self.a_in, b=self.b_in, theta=self.theta
            )

    @staticmethod
    def _prepare_mask(bbox, ap_r, ap_1, ap_2, method, subpixels, min_mask=0):
        """Make the pill box mask array.
        Notes
        -----
        To make an ndarray to represent the overlapping mask, the three (a
        rectangular and two elliptical) apertures are generated, but in parallel
        shifted such that the bounding box has ``ixmin`` and ``iymin`` both
        zero. Then proper mask is generated as an ndarray. It is then used by
        ``PillBoxMaskMixin.to_mask`` to make an ``ApertureMask`` object by
        combining this mask with the original bounding box.

        Parameters
        ----------
        bbox : `~photutils.aperture.BoundingBox`
            The bounding box of the original aperture.

        ap_r : `~photutils.aperture.RectangularAperture`
            The rectangular aperture of a pill box.

        ap_1, ap_2 : `~photutils.aperture.EllipticalAperture`
            The elliptical apertures of a pill box. The order of left/right
            ellipses is not important for this method.

        method : See `~photutils.aperture.PillBoxMaskMixin.to_mask`

        subpixels : See `~photutils.aperture.PillBoxMaskMixin.to_mask`

        min_mask : float, optional
            The mask values smaller than this value is ignored (set to 0). This
            is required because the subtraction of elliptical and rectangular
            masks give some negative values. One can set it to be
            ``1/subpixels**2`` because ``RectangularAperture`` does not support
            ``method='exact'`` yet.

        Returns
        -------
        mask_pill : ndarray
            The mask of the pill box.
        """
        aps = []
        for ap in [ap_r, ap_1, ap_2]:
            pos_cent = ap.positions
            tmp_cent = pos_cent - np.array([bbox.ixmin, bbox.iymin])
            if hasattr(ap, "w"):
                tmp_ap = RectangularAperture(
                    positions=tmp_cent, w=ap.w, h=ap.h, theta=ap.theta
                )
            else:
                tmp_ap = EllipticalAperture(
                    positions=tmp_cent, a=ap.a, b=ap.b, theta=ap.theta
                )
            aps.append(tmp_ap)

        bbox_shape = bbox.shape

        mask_kw = dict(method=method, subpixels=subpixels)
        mask_r = aps[0].to_mask(**mask_kw).to_image(bbox_shape)
        mask_1 = aps[1].to_mask(**mask_kw).to_image(bbox_shape)
        mask_2 = aps[2].to_mask(**mask_kw).to_image(bbox_shape)

        # Remove both machine epsilon artifact & negative mask values:
        mask_pill_1 = mask_1 - mask_r
        mask_pill_1[mask_pill_1 < min_mask] = 0
        mask_pill_2 = mask_2 - mask_r
        mask_pill_2[mask_pill_2 < min_mask] = 0

        mask_pill = mask_r + mask_pill_1 + mask_pill_2

        # Overlap of elliptical parts may make value > 1:
        mask_pill[mask_pill > 1] = 1

        return mask_pill

    def to_mask(self, method="exact", subpixels=5):
        """Return a mask for the aperture.

        Parameters
        ----------
        method : {'exact', 'center', 'subpixel'}, optional
            The method used to determine the overlap of the aperture on the
            pixel grid.  Not all options are available for all aperture types.
            Note that the more precise methods are generally slower.  The
            following methods are available:

                * ``'exact'`` (default):
                  The the exact fractional overlap of the aperture and each
                  pixel is calculated.  The returned mask will contain values
                  between 0 and 1.

                * ``'center'``:
                  A pixel is considered to be entirely in or out of the
                  aperture depending on whether its center is in or out of the
                  aperture.  The returned mask will contain values only of 0
                  (out) and 1 (in).

                * ``'subpixel'``:
                  A pixel is divided into subpixels (see the `subpixels`
                  keyword), each of which are considered to be entirely in or
                  out of the aperture depending on whether its center is in or
                  out of the aperture.  If ``subpixels=1``, this method is
                  equivalent to ``'center'``.  The returned mask will contain
                  values between 0 and 1.

        subpixels : int, optional
            For the ``'subpixel'`` method, resample pixels by this factor
            in each dimension.  That is, each pixel is divided into
            ``subpixels ** 2`` subpixels.

        Returns
        -------
        mask : `~photutils.aperture.ApertureMask` or list of `~photutils.aperture.ApertureMask`
            The aperture mask. If the aperture is scalar, a single
            `~photutils.aperture.ApertureMask` is returned, otherwise a list of
            `~photutils.aperture.ApertureMask` is returned.
        """
        _, subpixels = self._translate_mask_mode(method, subpixels)
        min_mask = min(1.0e-6, 1 / (subpixels**2))
        masks = []
        bboxes = np.atleast_1d(self.bbox)
        is_annulus = True if hasattr(self, "a_in") else False

        for i, (bbox, ap_r, ap_1, ap_2) in enumerate(
            zip(bboxes, self._ap_rect, self._ap_el_1, self._ap_el_2)
        ):
            mask = self._prepare_mask(
                bbox,
                ap_r=ap_r,
                ap_1=ap_1,
                ap_2=ap_2,
                method=method,
                subpixels=subpixels,
                min_mask=min_mask,
            )

            if is_annulus:
                mask -= self._prepare_mask(
                    bbox,
                    ap_r=self._ap_rect_in[i],
                    ap_1=self._ap_el_1_in[i],
                    ap_2=self._ap_el_2_in[i],
                    method=method,
                    subpixels=subpixels,
                    min_mask=min_mask,
                )

            masks.append(ApertureMask(mask, bbox))

        if self.isscalar:
            return masks[0]
        else:
            return masks

    @staticmethod
    def _pill_patches(ellipse_1, ellipse_2, **patch_kwargs):
        """Make matplotlib.patches from ellipses."""
        import matplotlib.patches as mpatches
        import matplotlib.path as mpath

        path_1 = ellipse_1.get_path()
        tran_1 = ellipse_1.get_transform()
        trpath_1 = tran_1.transform_path(path_1)
        trpath_1_v = trpath_1.vertices
        trpath_1_c = trpath_1.codes
        pill_1_v = trpath_1_v[: len(trpath_1_v) // 2, :]
        pill_1_c = trpath_1_c[: len(trpath_1_c) // 2]

        path_2 = ellipse_2.get_path()
        tran_2 = ellipse_2.get_transform()
        trpath_2 = tran_2.transform_path(path_2)
        trpath_2_v = trpath_2.vertices
        trpath_2_c = trpath_2.codes
        pill_2_v = trpath_2_v[-(len(trpath_2_v) // 2 + 1) :, :]
        pill_2_c = trpath_2_c[-(len(trpath_2_c) // 2 + 1) :]

        pill_v = np.concatenate([pill_1_v, pill_2_v])
        pill_c = np.concatenate([pill_1_c, [mpath.Path.LINETO], pill_2_c[1:]])
        pill_path = mpath.Path(pill_v, pill_c)
        pill_patch = mpatches.PathPatch(pill_path, **patch_kwargs)

        return pill_patch


class PillBoxAperture(PillBoxMaskMixin, PixelAperture):
    """A pill box aperture defined in pixel coordinates.

    The aperture has a single fixed size/shape, but it can have multiple
    positions (see the ``positions`` input).

    """

    _params = ("positions", "w", "a", "b", "theta")
    positions = PixelPositions("The center pixel position(s).")
    w = PositiveScalar(f"The {_PBSTRS['w']} in pixels.")
    a = PositiveScalar(f"The {_PBSTRS['a']} in pixels.")
    b = PositiveScalar(f"The {_PBSTRS['b']} in pixels.")
    theta = ScalarAngleOrValue(_PBSTRS["theta_pix"])

    def __init__(self, positions, w, a, b, theta=0.0):
        self.positions = positions
        self.w = w
        self.a = a
        self.b = b
        self.h = 2 * b
        self.theta = theta
        self._set_aperture_elements

    @property
    def _xy_extents(self):
        return np.abs(self.offset) + self._ap_el_1._xy_extents

    @property
    def area(self):
        return self.w * self.h + np.pi * self.a * self.b

    def _to_patch(self, origin=(0, 0), indices=None, **kwargs):
        """ """
        import matplotlib.patches as mpatches

        # xy_positions is already atleast_2d'ed.
        xy_positions, patch_kwargs = self._define_patch_params(origin=origin, **kwargs)
        # There used to be `indices=indices` in this function, but it gives an
        # error (AttributeError: 'PathPatch' object has no property 'indices').
        # Without this, it works perfectly. I am not sure what happened...
        # -2022-04-25 23:26:12 (KST: GMT+09:00) ysBach

        patches = []
        theta_deg = self.theta.to_value(u.deg)

        for xy_position in xy_positions:
            # The ellipse on the "right" when theta = 0
            ellipse_1 = mpatches.Ellipse(
                xy_position + self.offset, 2.0 * self.a, 2.0 * self.b, angle=theta_deg
            )
            # The ellipse on the "left" when theta = 0
            ellipse_2 = mpatches.Ellipse(
                xy_position - self.offset, 2.0 * self.a, 2.0 * self.b, angle=theta_deg
            )
            p = self._pill_patches(ellipse_1, ellipse_2, **patch_kwargs)

            patches.append(p)

        if self.isscalar:
            return patches[0]
        else:
            return patches

    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyPillBoxAperture` object
        defined in celestial coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        Returns
        -------
        aperture : `SkyPillBoxAperture` object
            A `SkyPillBoxAperture` object.
        """
        sky_params = self._to_sky_params(wcs)
        return SkyPillBoxAperture(**sky_params)


class PillBoxAnnulus(PillBoxMaskMixin, PixelAperture):
    """ """

    _params = ("positions", "w", "a_in", "a_out", "b_out", "theta")
    positions = PixelPositions("The center pixel position(s).")
    w = PositiveScalar(f"The {_PBSTRS['w']} in pixels.")
    a_in = PositiveScalar(f"The inner {_PBSTRS['a']} in pixels.")
    a_out = PositiveScalar(f"The outer {_PBSTRS['a']} in pixels.")
    b_out = PositiveScalar(f"The outer {_PBSTRS['b']} in pixels.")
    theta = ScalarAngleOrValue(_PBSTRS["theta_pix"])

    def __init__(self, positions, w, a_in, a_out, b_out, theta=0.0):
        self.positions = positions
        self.w = w
        self.a_out = a_out
        self.a_in = a_in
        self.b_out = b_out
        self.b_in = self.b_out * self.a_in / self.a_out
        self.h_out = self.b_out * 2
        self.h_in = self.b_in * 2
        self.theta = theta
        self._set_aperture_elements

    @property
    def _xy_extents(self):
        return np.abs(self.offset) + self._ap_el_1._xy_extents

    @property
    def area(self):
        return self.w * (self.h_out - self.h_in) + np.pi * (
            self.a_out * self.b_out - self.a_in * self.b_in
        )

    def _to_patch(self, origin=(0, 0), indices=None, **kwargs):
        import matplotlib.patches as mpatches

        # xy_positions is already atleast_2d'ed.
        xy_positions, patch_kwargs = self._define_patch_params(origin=origin, **kwargs)
        # There used to be `indices=indices` in this function, but it gives an
        # error (AttributeError: 'PathPatch' object has no property 'indices').
        # Without this, it works perfectly. I am not sure what happened...
        # -2022-04-25 23:26:12 (KST: GMT+09:00) ysBach

        patches = []
        theta_deg = self.theta.to_value(u.deg)

        for xy_position in xy_positions:
            # The ellipse on the "right" when theta = 0
            ellipse_1_in = mpatches.Ellipse(
                xy_position + self.offset,
                2.0 * self.a_in,
                2.0 * self.b_in,
                angle=theta_deg,
            )
            # The ellipse on the "left" when theta = 0
            ellipse_2_in = mpatches.Ellipse(
                xy_position - self.offset,
                2.0 * self.a_in,
                2.0 * self.b_in,
                angle=theta_deg,
            )
            p_inner = self._pill_patches(ellipse_1_in, ellipse_2_in)

            # The ellipse on the "right" when theta = 0
            ellipse_1_out = mpatches.Ellipse(
                xy_position + self.offset,
                2.0 * self.a_out,
                2.0 * self.b_out,
                angle=theta_deg,
            )
            # The ellipse on the "left" when theta = 0
            ellipse_2_out = mpatches.Ellipse(
                xy_position - self.offset,
                2.0 * self.a_out,
                2.0 * self.b_out,
                angle=theta_deg,
            )
            p_outer = self._pill_patches(ellipse_1_out, ellipse_2_out)

            p = self._make_annulus_path(p_inner, p_outer)
            patches.append(mpatches.PathPatch(p, **patch_kwargs))

        if self.isscalar:
            return patches[0]
        else:
            return patches

    def to_sky(self, wcs):
        """
        Convert the aperture to a `SkyPillBoxAnnulus` object defined
        in celestial coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        Returns
        -------
        aperture : `SkyPillBoxAnnulus` object
            A `SkyPillBoxAnnulus` object.
        """
        sky_params = self._to_sky_params(wcs)
        return SkyPillBoxAnnulus(**sky_params)


class SkyPillBoxAperture(SkyAperture):
    """A pill box aperture defined in sky coordinates."""

    _params = ("positions", "w", "a", "b", "theta")
    positions = SkyCoordPositions("'The center position(s) in sky coordinates.'")
    w = PositiveScalarAngle(f"The {_PBSTRS['w']} in angular units.")
    a = PositiveScalarAngle(f"The {_PBSTRS['a']} in angular units.")
    b = PositiveScalarAngle(f"The {_PBSTRS['b']} in angular units.")
    theta = ScalarAngle(_PBSTRS["theta_sky"])

    def __init__(self, positions, w, a, b, theta=0.0 * u.deg):
        if not (w.unit.physical_type == a.unit.physical_type == b.unit.physical_type):
            raise ValueError("'w', 'a', and 'b' should all be angles or in pixels")

        self.positions = positions
        self.w = w
        self.a = a
        self.b = b
        self.theta = theta

    def to_pixel(self, wcs):
        """
        Convert the aperture to an `PillBoxAperture` object defined in pixel
        coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        Returns
        -------
        aperture : `PillBoxAperture` object
            An `PillBoxAperture` object.
        """
        pixel_params = self._to_pixel_params(wcs)
        return PillBoxAperture(**pixel_params)


class SkyPillBoxAnnulus(SkyAperture):
    _params = ("positions", "w", "a_in", "a_out", "b_out", "theta")
    positions = SkyCoordPositions("positions")
    w = PositiveScalarAngle(f"The {_PBSTRS['w']} in angular units.")
    a_in = PositiveScalarAngle(f"The inner {_PBSTRS['a']} in angular units.")
    a_out = PositiveScalarAngle(f"The outer {_PBSTRS['a']} in angular units.")
    b_out = PositiveScalarAngle(f"The outer {_PBSTRS['b']} in angular units.")
    theta = ScalarAngle(_PBSTRS["theta_sky"])

    def __init__(self, positions, w, a_in, a_out, b_out, theta=0.0 * u.deg):
        if not (
            w.unit.physical_type
            == a_in.unit.physical_type
            == a_out.unit.physical_type
            == b_out.unit.physical_type
        ):
            raise ValueError(
                "'w', 'a_in', 'a_out', and 'b_out' should all be " "angles or in pixels"
            )

        self.positions = positions
        self.w = w
        self.a_out = a_out
        self.a_in = a_in
        self.b_out = b_out
        self.b_in = self.b_out * self.a_in / self.a_out
        self.h_out = self.b_out * 2
        self.h_in = self.b_in * 2
        self.theta = theta

    def to_pixel(self, wcs):
        """
        Convert the aperture to an `PillBoxAnnulus` object defined in
        pixel coordinates.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system (WCS) transformation to use.

        Returns
        -------
        aperture : `PillBoxAnnulus` object
            An `PillBoxAnnulus` object.
        """
        pixel_params = self._to_pixel_params(wcs)
        return PillBoxAnnulus(**pixel_params)
