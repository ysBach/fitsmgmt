import warnings

import numpy as np
import pytest
from astropy.io import fits
from astropy.nddata import CCDData
from astropy import units as u

import astroimred.reduction as imred
from astroimred.reduction.imutil.util_comb import get_zsw


def _write_fits(path, data, header=None):
    hdr = fits.Header() if header is None else header.copy()
    hdr["BUNIT"] = "adu"
    fits.PrimaryHDU(data=np.asarray(data, dtype="float32"), header=hdr).writeto(path)


def _wcs_header_for_offset(raw_offset_yx, shape):
    dy, dx = raw_offset_yx
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = shape[1]
    hdr["NAXIS2"] = shape[0]
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = 20.0 - dx
    hdr["CRPIX2"] = 20.0 - dy
    hdr["CRVAL1"] = 10.0
    hdr["CRVAL2"] = 20.0
    hdr["CDELT1"] = 1.0 / 3600.0
    hdr["CDELT2"] = 1.0 / 3600.0
    return hdr


def _physical_header_for_offset(raw_offset_yx, shape):
    dy, dx = raw_offset_yx
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = shape[1]
    hdr["NAXIS2"] = shape[0]
    hdr["LTV1"] = float(dx)
    hdr["LTV2"] = float(dy)
    return hdr


def _regularized_offsets(raw_offsets):
    raw_offsets = np.asarray(raw_offsets, dtype=int)
    return raw_offsets - raw_offsets.min(axis=0)


def _offset_stack(images, raw_offsets):
    origins = _regularized_offsets(raw_offsets)
    shapes = np.array([im.shape for im in images], dtype=int)
    out_shape = tuple(np.max(origins + shapes, axis=0))
    stack = np.full((len(images), *out_shape), np.nan)
    for i, (image, origin) in enumerate(zip(images, origins)):
        y0, x0 = origin
        y1, x1 = origin + image.shape
        stack[i, y0:y1, x0:x1] = image
    return stack, origins, out_shape


def _normalized_zs(nimage, zero=None, scale=None):
    zeros = np.zeros(nimage) if zero is None else np.asarray(zero, dtype=float)
    scales = np.ones(nimage) if scale is None else np.asarray(scale, dtype=float)
    return zeros - zeros[0], scales / scales[0]


def _apply_zs(stack, zero=None, scale=None):
    zeros, scales = _normalized_zs(stack.shape[0], zero=zero, scale=scale)
    out = stack.copy()
    for i, (z, s) in enumerate(zip(zeros, scales)):
        out[i] = (out[i] - z) / s
    return out


def _nanlmedian_axis0(stack):
    out = np.full(stack.shape[1:], np.nan)
    for idx in np.ndindex(out.shape):
        vals = stack[(slice(None), *idx)]
        vals = np.sort(vals[np.isfinite(vals)])
        if vals.size:
            out[idx] = vals[(vals.size - 1) // 2]
    return out


def _combine_expected(stack, combine):
    with np.errstate(invalid="ignore"):
        if combine in ["average", "mean"]:
            return np.nanmean(stack, axis=0)
        if combine == "sum":
            return np.nansum(stack, axis=0)
        if combine in ["median", "med"]:
            return np.nanmedian(stack, axis=0)
        if combine in ["lmedian", "lmed"]:
            return _nanlmedian_axis0(stack)
    raise ValueError(f"Unsupported analytical combine: {combine}")


def _expected_no_reject(images, raw_offsets, combine="average", zero=None, scale=None):
    stack, _origins, out_shape = _offset_stack(images, raw_offsets)
    stack = _apply_zs(stack, zero=zero, scale=scale)
    mask = np.zeros(stack.shape, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return dict(
            comb=_combine_expected(stack, combine),
            err=np.nanstd(stack, axis=0, ddof=1),
            low=np.nanmin(stack, axis=0),
            upp=np.nanmax(stack, axis=0),
            mask_total=mask,
            mask_rej=mask.copy(),
            mask_thresh=mask.copy(),
            nit=None,
            rejcode=None,
            out_shape=out_shape,
        )


def _assert_imcombine_full(result, expected):
    np.testing.assert_allclose(
        result["comb"].data, expected["comb"], rtol=1e-6, atol=1e-6, equal_nan=True
    )
    for key in ["err", "low", "upp"]:
        np.testing.assert_allclose(
            result[key],
            expected[key],
            rtol=1e-6,
            atol=1e-6,
            equal_nan=True,
            err_msg=key,
        )
    for key in ["mask_total", "mask_rej", "mask_thresh"]:
        np.testing.assert_array_equal(result[key], expected[key])

    if expected["nit"] is None:
        assert result["nit"] is None
    else:
        nit = np.asarray(result["nit"]) + np.zeros(expected["out_shape"], dtype=np.uint8)
        np.testing.assert_array_equal(nit, expected["nit"])

    if expected["rejcode"] is None:
        assert result["rejcode"] is None
    else:
        np.testing.assert_array_equal(result["rejcode"], expected["rejcode"])


def test_stack_fits_removed():
    """`stack_FITS` has been replaced by `select_fits`."""
    assert not hasattr(imred.combutil, "stack_FITS")
    assert not hasattr(imred, "stack_FITS")


def test_get_zsw_does_not_mutate_kwargs():
    """Caller-owned sigma-clip kwargs must survive repeated calls."""
    arr = np.arange(27, dtype=float).reshape(3, 3, 3)
    zero_kw = {"sigma": 2.0, "maxiters": 1, "axis": 0}

    get_zsw(
        arr,
        zero="median_sc",
        scale=None,
        weight=None,
        zero_kw=zero_kw,
        scale_kw=None,
        zero_to_0th=True,
        scale_to_0th=True,
        zero_section=None,
        scale_section=None,
    )

    assert zero_kw == {"sigma": 2.0, "maxiters": 1, "axis": 0}


class TestNDCombine:
    """Tests for `~imred.imutil.ndcombine` function (core algorithmic logic)."""

    def test_basic_average(self):
        """Test simple average combination."""
        # 3 images, 10x10, values 1, 2, 3.
        # Average should be 2.
        arr = np.zeros((3, 10, 10))
        arr[0] += 1
        arr[1] += 2
        arr[2] += 3

        combined = imred.ndcombine(arr, combine="average")
        np.testing.assert_allclose(combined, 2.0, rtol=1e-6)

    def test_basic_median(self):
        """Test simple median combination."""
        arr = np.zeros((3, 10, 10))
        arr[0] += 1
        arr[1] += 10
        arr[2] += 100

        # Median is 10.
        combined = imred.ndcombine(arr, combine="median")
        np.testing.assert_allclose(combined, 10.0, rtol=1e-6)

    def test_basic_lmedian(self):
        """Test lower-median combination (even n: take lower of two middle)."""
        # 4 images: 1, 2, 3, 4 -> lower median = 2 (index 1 of sorted)
        arr = np.zeros((4, 10, 10))
        arr[0] += 1
        arr[1] += 2
        arr[2] += 3
        arr[3] += 4
        combined = imred.ndcombine(arr, combine="lmedian")
        np.testing.assert_allclose(combined, 2.0, rtol=1e-6)

    def test_sigma_clip(self):
        """Test sigma clipping."""
        # 5 images. 4 have value 10, 1 has value 100 (outlier).
        arr = np.ones((5, 10, 10)) * 10.0
        arr[4] = 100.0

        # Sigma clip with sigma=3.
        # Mean ~ 28. Std ~ 36.
        # 100 is (100-28)/36 = 2 sigma...
        # Wait, if we use sample std?
        # Let's make it more extreme.
        arr[4] = 1000.0

        # combine="average", reject="sigclip", sigma=[3, 3]
        combined = imred.ndcombine(
            arr,
            combine="average",
            reject="sigclip",
            sigma=[1.0, 1.0],
            verbose=False
        )

        # Should reject 1000.0 and average the rest (10.0).
        np.testing.assert_allclose(combined, 10.0, rtol=1e-6)

    def test_minmax_clip(self):
        """Test minmax rejection."""
        # 0, 10, 10, 10, 100
        arr = np.array([0, 10, 10, 10, 100])
        # Reshape to (N, 1, 1) needed for ndcombine?
        # ndcombine expects (N, y, x).
        arr = arr[:, None, None] * np.ones((5, 2, 2))

        # nlow=1, nhigh=1 -> reject lowest and highest.
        combined = imred.ndcombine(
            arr,
            combine="average",
            reject="minmax",
            n_minmax=[1, 1]
        )

        # Remaining: 10, 10, 10 -> Average 10.
        np.testing.assert_allclose(combined, 10.0, rtol=1e-6)

    def test_numba_sigclip_aux_analytical(self):
        """The Numba path must preserve NaN-aware rejection and combine math."""
        old = imred.imutil_config.IMUTIL_USE_NUMBA
        imred.set_imutil_use_numba(True)
        try:
            yy, xx = np.mgrid[:2, :3]
            base = yy * 10.0 + xx
            arr = np.stack([base + value for value in [10.0, 10.0, 10.0, 1000.0]])
            comb, err, mask_rej, mask_thresh, low, upp, nit, rejcode = imred.ndcombine(
                arr,
                combine="average",
                reject="sigclip",
                sigma=[1.0, 1.0],
                cenfunc="median",
                maxiters=5,
                ddof=1,
                nkeep=1,
                irafmode=False,
                full=True,
            )
        finally:
            imred.set_imutil_use_numba(old)

        expected_mask = np.zeros(arr.shape, dtype=bool)
        expected_mask[3] = True
        np.testing.assert_allclose(comb, base + 10.0, rtol=1e-6)
        np.testing.assert_allclose(err, 0.0, atol=1e-6)
        np.testing.assert_array_equal(mask_rej, expected_mask)
        np.testing.assert_array_equal(mask_thresh, np.zeros_like(expected_mask))
        np.testing.assert_allclose(low, base + 10.0, rtol=1e-6)
        np.testing.assert_allclose(upp, base + 10.0, rtol=1e-6)
        np.testing.assert_array_equal(nit, 2 * np.ones(base.shape, dtype=np.uint8))
        np.testing.assert_array_equal(
            rejcode, 2 * np.ones(base.shape, dtype=np.uint8)
        )


class TestImCombine:
    """Tests for `~imred.imutil.imcombine` wrapper with FITS files."""

    def test_imcombine_files(self, tmp_path):
        """Test combining FITS files."""
        # Create 3 files
        vals = [10.0, 20.0, 30.0]
        paths = []
        for i, v in enumerate(vals):
            d = np.ones((10, 10)) * v
            p = tmp_path / f"test_{i}.fits"
            CCDData(d, unit="adu").write(p)
            paths.append(p)

        # Combine
        outpath = tmp_path / "combined.fits"

        res = imred.imcombine(
            paths,
            output=outpath,
            combine="average",
            reject="none"
        )

        # Check result object
        np.testing.assert_allclose(res.data, 20.0, rtol=1e-6)

        # Check file
        loaded = CCDData.read(outpath)
        np.testing.assert_allclose(loaded.data, 20.0, rtol=1e-6)

    @pytest.mark.parametrize("memlimit", [1e9, 300])
    def test_default_extension_uses_first_image_hdu(self, tmp_path, memlimit):
        paths = []
        for i, value in enumerate([7.0, 9.0]):
            path = tmp_path / f"mef_{i}.fits"
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(
                        data=np.full((2, 3), value, dtype="float32"),
                        header=fits.Header({"BUNIT": "adu"}),
                        name="SCI",
                    ),
                ]
            ).writeto(path)
            paths.append(path)

        result = imred.imcombine(
            paths,
            combine="average",
            reject="none",
            memlimit=memlimit,
            verbose=False,
        )

        np.testing.assert_allclose(result.data, 8.0, rtol=1e-6)

    def test_imcombine_memlimit_chunks_match_full_with_offsets(self, tmp_path):
        """Chunked FITS loading should reproduce the full-stack result."""
        paths = []
        yy, xx = np.mgrid[:6, :8]
        for i in range(3):
            d = yy * 10 + xx + i * 100
            p = tmp_path / f"offset_{i}.fits"
            CCDData(d.astype("float32"), unit="adu").write(p)
            paths.append(p)

        offsets = np.array([[0, 2], [2, 0], [1, 1]])
        common = dict(
            offsets=offsets,
            combine="average",
            reject="none",
            full=True,
            return_dict=True,
            dtype="float32",
        )
        full = imred.imcombine(paths, memlimit=1e9, **common)
        chunked = imred.imcombine(paths, memlimit=500, **common)

        np.testing.assert_allclose(
            chunked["comb"].data, full["comb"].data, rtol=1e-6, equal_nan=True
        )
        for key in ["err", "low", "upp"]:
            np.testing.assert_allclose(
                chunked[key], full[key], rtol=1e-6, equal_nan=True
            )
        for key in ["mask_total", "mask_rej", "mask_thresh"]:
            np.testing.assert_array_equal(chunked[key], full[key])
        assert chunked["nit"] is None
        assert chunked["rejcode"] is None

    def test_user_offsets_zero_scale_average_aux_analytical(self, tmp_path):
        """User offsets plus zero/scale should match direct pixel math."""
        yy, xx = np.mgrid[:4, :5]
        sky = yy * 10.0 + xx
        raw_offsets = np.array([[0, 1], [2, 0], [1, 2]])
        zeros = np.array([0.0, 10.0, 20.0])
        scales = np.array([1.0, 2.0, 4.0])
        weights = np.array([1.0, 3.0, 5.0])
        images = [scales[i] * sky + zeros[i] for i in range(3)]
        paths = []
        for i, image in enumerate(images):
            p = tmp_path / f"user_offset_{i}.fits"
            _write_fits(p, image)
            paths.append(p)

        result = imred.imcombine(
            paths,
            offsets=raw_offsets,
            zero=zeros,
            scale=scales,
            weight=weights,
            combine="average",
            reject="none",
            full=True,
            return_dict=True,
            memlimit=400,
        )
        expected = _expected_no_reject(
            images, raw_offsets, combine="average", zero=zeros, scale=scales
        )
        _assert_imcombine_full(result, expected)

        # Weights are recorded in the header; weighted combine is still listed
        # as an unimplemented imcombine option.
        for i, weight in enumerate(weights, start=1):
            assert result["comb"].header[f"WEIGH{i:03d}"] == weight

    def test_wcs_offsets_median_aux_analytical(self, tmp_path):
        """WCS-derived offsets should place each image at the analytical origin."""
        yy, xx = np.mgrid[:4, :5]
        images = [yy * 10.0 + xx + 100.0 * i for i in range(3)]
        raw_offsets = np.array([[0, 0], [1, 2], [2, 1]])
        paths = []
        for i, (image, raw_offset) in enumerate(zip(images, raw_offsets)):
            p = tmp_path / f"wcs_offset_{i}.fits"
            _write_fits(p, image, _wcs_header_for_offset(raw_offset, image.shape))
            paths.append(p)

        result = imred.imcombine(
            paths,
            offsets="wcs",
            combine="median",
            reject="none",
            full=True,
            return_dict=True,
            memlimit=500,
        )
        expected = _expected_no_reject(images, raw_offsets, combine="median")
        _assert_imcombine_full(result, expected)
        assert result["comb"].header["OFFSTMOD"] == "WCS"

    def test_physical_offsets_lmedian_aux_analytical(self, tmp_path):
        """LTV/LTM physical offsets should match direct lower-median math."""
        yy, xx = np.mgrid[:4, :5]
        images = [yy * 10.0 + xx + 50.0 * i for i in range(3)]
        raw_offsets = np.array([[0, 0], [2, 1], [1, 3]])
        paths = []
        for i, (image, raw_offset) in enumerate(zip(images, raw_offsets)):
            p = tmp_path / f"physical_offset_{i}.fits"
            _write_fits(p, image, _physical_header_for_offset(raw_offset, image.shape))
            paths.append(p)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = imred.imcombine(
                paths,
                offsets="physical",
                combine="lmedian",
                reject="none",
                full=True,
                return_dict=True,
                memlimit=300,
            )
        expected = _expected_no_reject(images, raw_offsets, combine="lmedian")
        _assert_imcombine_full(result, expected)
        assert result["comb"].header["OFFSTMOD"] == "Physical"

    @pytest.mark.parametrize(
        ("combine", "comb_expected"),
        [
            ("average", lambda base: base + 15.0),
            ("sum", lambda base: 2 * base + 30.0),
            ("median", lambda base: base + 15.0),
            ("lmedian", lambda base: base + 10.0),
        ],
    )
    def test_minmax_rejection_combine_aux_analytical(
        self, tmp_path, combine, comb_expected
    ):
        """Minmax rejection should return exact bounds, masks, and codes."""
        yy, xx = np.mgrid[:4, :5]
        base = yy * 10.0 + xx
        images = [base + value for value in [0.0, 10.0, 20.0, 100.0]]
        paths = []
        for i, image in enumerate(images):
            p = tmp_path / f"minmax_{combine}_{i}.fits"
            _write_fits(p, image)
            paths.append(p)

        result = imred.imcombine(
            paths,
            combine=combine,
            reject="minmax",
            n_minmax=[1, 1],
            full=True,
            return_dict=True,
            memlimit=300,
        )
        expected_mask = np.zeros((4, *base.shape), dtype=bool)
        expected_mask[0] = True
        expected_mask[3] = True
        expected = dict(
            comb=comb_expected(base),
            err=np.sqrt(50.0) * np.ones_like(base),
            low=base,
            upp=base + 100.0,
            mask_total=expected_mask,
            mask_rej=expected_mask.copy(),
            mask_thresh=np.zeros_like(expected_mask),
            nit=np.ones(base.shape, dtype=np.uint8),
            rejcode=np.zeros(base.shape, dtype=np.uint8),
            out_shape=base.shape,
        )
        _assert_imcombine_full(result, expected)

    def test_sigclip_rejection_aux_analytical(self, tmp_path):
        """Sigma clipping should reject the high outlier and expose final bounds."""
        yy, xx = np.mgrid[:4, :5]
        base = yy * 10.0 + xx
        images = [base + value for value in [10.0, 10.0, 10.0, 1000.0]]
        paths = []
        for i, image in enumerate(images):
            p = tmp_path / f"sigclip_{i}.fits"
            _write_fits(p, image)
            paths.append(p)

        result = imred.imcombine(
            paths,
            combine="average",
            reject="sigclip",
            sigma=[1.0, 1.0],
            cenfunc="median",
            maxiters=5,
            ddof=1,
            nkeep=1,
            irafmode=False,
            full=True,
            return_dict=True,
            memlimit=300,
        )

        expected_mask = np.zeros((4, *base.shape), dtype=bool)
        expected_mask[3] = True
        expected = dict(
            comb=base + 10.0,
            err=np.zeros_like(base),
            low=base + 10.0,
            upp=base + 10.0,
            mask_total=expected_mask,
            mask_rej=expected_mask.copy(),
            mask_thresh=np.zeros_like(expected_mask),
            nit=2 * np.ones(base.shape, dtype=np.uint8),
            rejcode=2 * np.ones(base.shape, dtype=np.uint8),
            out_shape=base.shape,
        )
        _assert_imcombine_full(result, expected)
