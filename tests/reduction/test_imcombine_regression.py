"""
Regression tests for imcombine/ndcombine and components.

Compares current code output against ground-truth saved by
generate_imcombine_regression_data.py. Re-run that script to refresh
ground-truth after intentional logic changes.

Usage:
  pytest tests/test_imcombine_regression.py -v
  pytest tests/test_imcombine_regression.py -v -k ndcombine
"""

import pickle
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest

import astroimred.reduction as imred

imred_uc = import_module("astroimred.reduction.imutil.util_comb")
imred_ur = import_module("astroimred.reduction.imutil.util_reject")
ndcombine = imred.ndcombine
_get_dtype_limits = imred_uc._get_dtype_limits
_set_cenfunc = imred_uc._set_cenfunc
_set_combfunc = imred_uc._set_combfunc
_set_gain_rdns = imred_uc._set_gain_rdns
_set_int_dtype = imred_uc._set_int_dtype
_set_keeprej = imred_uc._set_keeprej
_set_mask = imred_uc._set_mask
_set_minmax = imred_uc._set_minmax
_set_reject_name = imred_uc._set_reject_name
_set_sigma = imred_uc._set_sigma
_set_thresh_mask = imred_uc._set_thresh_mask
do_zs = imred_uc.do_zs
get_zsw = imred_uc.get_zsw
ccdclip_mask = imred_ur.ccdclip_mask
minmax_mask = imred_ur.minmax_mask
sigclip_mask = imred_ur.sigclip_mask

# Reuse same stack generator as in data generator
from tests.generate_imcombine_regression_data import (
    DEFAULT_SEED,
    make_stack,
)

REGRESSION_DATA_DIR = Path(__file__).resolve().parent / "data"
RTOL = 1e-6
ATOL = 1e-8
# Slightly looser for combined array (Numba vs bottleneck float order can differ)
RTOL_COMB = 1e-5
ATOL_COMB = 1e-6


def _load_ndcombine_cases():
    p = REGRESSION_DATA_DIR / "ndcombine_regression.pkl"
    if not p.exists():
        pytest.skip("Regression data not found. Run: python -m tests.generate_imcombine_regression_data")
    with open(p, "rb") as f:
        return pickle.load(f)


def _load_component_cases():
    p = REGRESSION_DATA_DIR / "component_regression.pkl"
    if not p.exists():
        pytest.skip("Regression data not found. Run: python -m tests.generate_imcombine_regression_data")
    with open(p, "rb") as f:
        return pickle.load(f)


class TestNDCombineRegression:
    """Regression tests for ndcombine: compare current output to stored ground-truth."""

    @pytest.fixture(scope="class")
    def ndcombine_cases(self):
        return _load_ndcombine_cases()

    def test_ndcombine_regression_success_cases(self, ndcombine_cases):
        """All success cases must reproduce stored comb (and full outputs when full=True)."""
        success = [c for c in ndcombine_cases if c.get("error") is None]
        assert len(success) > 0, "No success cases in regression data"
        for case in success:
            shape = case["shape"]
            seed = case["seed"]
            params = case["params"]
            stored = case["output"]
            arr = make_stack(shape, seed=seed)
            out = ndcombine(arr, **params)
            if params.get("full"):
                comb, err, mask_rej, mask_thresh, low, upp, nit, rejcode = out
                np.testing.assert_allclose(comb, stored["comb"], rtol=RTOL_COMB, atol=ATOL_COMB, err_msg=case.get("case_id"))
                np.testing.assert_allclose(err, stored["err"], rtol=RTOL, atol=ATOL, err_msg=case.get("case_id"))
                np.testing.assert_array_equal(mask_rej, stored["mask_rej"], err_msg=case.get("case_id"))
                np.testing.assert_array_equal(mask_thresh, stored["mask_thresh"], err_msg=case.get("case_id"))
                np.testing.assert_allclose(low, stored["low"], rtol=RTOL, atol=ATOL, err_msg=case.get("case_id"))
                np.testing.assert_allclose(upp, stored["upp"], rtol=RTOL, atol=ATOL, err_msg=case.get("case_id"))
                np.testing.assert_array_equal(nit, stored["nit"], err_msg=case.get("case_id"))
                if stored["rejcode"] is not None:
                    np.testing.assert_array_equal(rejcode, stored["rejcode"], err_msg=case.get("case_id"))
            else:
                np.testing.assert_allclose(out, stored["comb"], rtol=RTOL_COMB, atol=ATOL_COMB, err_msg=case.get("case_id"))

    def test_ndcombine_regression_error_cases(self, ndcombine_cases):
        """Cases that raised during generation must still raise (same behavior)."""
        error_cases = [c for c in ndcombine_cases if c.get("error") is not None]
        for case in error_cases:
            shape = case["shape"]
            seed = case["seed"]
            params = case["params"]
            arr = make_stack(shape, seed=seed)
            try:
                ndcombine(arr, **params)
            except Exception:
                continue
            if "_median_nancheck" in case.get("error", ""):
                continue
            pytest.fail(f"Expected error did not raise: {case.get('case_id')}")


class TestGetZSWRegression:
    """Regression tests for get_zsw."""

    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["get_zsw"]

    def test_get_zsw_regression(self, cases):
        for i, rec in enumerate(cases):
            params = dict(rec["params"])
            arr = params.pop("arr")
            expected = rec["output"]
            z, s, w = get_zsw(
                arr,
                zero=params["zero"],
                scale=params["scale"],
                weight=params["weight"],
                zero_kw=params["zero_kw"],
                scale_kw=params["scale_kw"],
                zero_to_0th=params["zero_to_0th"],
                scale_to_0th=params["scale_to_0th"],
                zero_section=params["zero_section"],
                scale_section=params["scale_section"],
            )
            np.testing.assert_allclose(z, expected[0], rtol=RTOL, atol=ATOL, err_msg=f"get_zsw zeros case {i}")
            np.testing.assert_allclose(s, expected[1], rtol=RTOL, atol=ATOL, err_msg=f"get_zsw scales case {i}")
            np.testing.assert_allclose(w, expected[2], rtol=RTOL, atol=ATOL, err_msg=f"get_zsw weights case {i}")


class TestDoZSRegression:
    """Regression tests for do_zs."""

    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["do_zs"]

    def test_do_zs_regression(self, cases):
        for i, rec in enumerate(cases):
            params = dict(rec["params"])
            arr = params.pop("arr").copy()
            zeros = params["zeros"]
            scales = params["scales"]
            expected = rec["output"]
            result = do_zs(arr, zeros=zeros, scales=scales, copy=False)
            np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL, err_msg=f"do_zs case {i}")


class TestSigclipMaskRegression:
    """Regression tests for sigclip_mask."""

    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["sigclip_mask"]

    def test_sigclip_mask_regression(self, cases):
        for i, rec in enumerate(cases):
            params = dict(rec["params"])
            arr = params.pop("arr").copy()
            full = params.get("full", True)
            sigma = params.get("sigma")
            if isinstance(sigma, (list, tuple)):
                params.setdefault("sigma_lower", sigma[0])
                params.setdefault("sigma_upper", sigma[1])
            r = sigclip_mask(arr, **params)
            expected = rec["output"]
            if full:
                np.testing.assert_array_equal(r[0], expected[0], err_msg=f"sigclip_mask mask case {i}")
                np.testing.assert_allclose(r[1], expected[1], rtol=RTOL, atol=ATOL, err_msg=f"sigclip_mask low case {i}")
                np.testing.assert_allclose(r[2], expected[2], rtol=RTOL, atol=ATOL, err_msg=f"sigclip_mask upp case {i}")
                np.testing.assert_array_equal(r[3], expected[3], err_msg=f"sigclip_mask nit case {i}")
                np.testing.assert_array_equal(r[4], expected[4], err_msg=f"sigclip_mask code case {i}")
            else:
                np.testing.assert_array_equal(r, expected, err_msg=f"sigclip_mask case {i}")


class TestMinmaxMaskRegression:
    """Regression tests for minmax_mask."""

    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["minmax_mask"]

    def test_minmax_mask_regression(self, cases):
        for i, rec in enumerate(cases):
            params = dict(rec["params"])
            arr = params.pop("arr").copy()
            r = minmax_mask(arr, **params)
            expected = rec["output"]
            if params.get("full", True):
                np.testing.assert_array_equal(r[0], expected[0], err_msg=f"minmax_mask mask case {i}")
                np.testing.assert_allclose(r[1], expected[1], rtol=RTOL, atol=ATOL, err_msg=f"minmax_mask low case {i}")
                np.testing.assert_allclose(r[2], expected[2], rtol=RTOL, atol=ATOL, err_msg=f"minmax_mask upp case {i}")
                np.testing.assert_array_equal(r[3], expected[3], err_msg=f"minmax_mask nit case {i}")
                np.testing.assert_array_equal(r[4], expected[4], err_msg=f"minmax_mask code case {i}")
            else:
                np.testing.assert_array_equal(r, expected, err_msg=f"minmax_mask case {i}")


class TestCcdclipMaskRegression:
    """Regression tests for ccdclip_mask."""

    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["ccdclip_mask"]

    def test_ccdclip_mask_regression(self, cases):
        for i, rec in enumerate(cases):
            params = dict(rec["params"])
            arr = params.pop("arr").copy()
            r = ccdclip_mask(arr, **params)
            expected = rec["output"]
            if params.get("full", True):
                np.testing.assert_array_equal(r[0], expected[0], err_msg=f"ccdclip_mask mask case {i}")
                np.testing.assert_allclose(r[1], expected[1], rtol=RTOL, atol=ATOL, err_msg=f"ccdclip_mask low case {i}")
                np.testing.assert_allclose(r[2], expected[2], rtol=RTOL, atol=ATOL, err_msg=f"ccdclip_mask upp case {i}")
                np.testing.assert_array_equal(r[3], expected[3], err_msg=f"ccdclip_mask nit case {i}")
                np.testing.assert_array_equal(r[4], expected[4], err_msg=f"ccdclip_mask code case {i}")
            else:
                np.testing.assert_array_equal(r, expected, err_msg=f"ccdclip_mask case {i}")


class TestSetIntDtypeRegression:
    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["set_int_dtype"]

    def test_set_int_dtype_regression(self, cases):
        for i, rec in enumerate(cases):
            r = _set_int_dtype(**rec["params"])
            assert str(r) == rec["output"], f"set_int_dtype case {i}"


class TestSetSigmaRegression:
    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["set_sigma"]

    def test_set_sigma_regression(self, cases):
        for i, rec in enumerate(cases):
            if rec.get("error"):
                with pytest.raises(Exception):
                    _set_sigma(**rec["params"])
            else:
                r = _set_sigma(**rec["params"])
                assert (float(r[0]), float(r[1])) == rec["output"], f"set_sigma case {i}"


class TestSetKeeprejRegression:
    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["set_keeprej"]

    def test_set_keeprej_regression(self, cases):
        for i, rec in enumerate(cases):
            params = dict(rec["params"])
            arr = np.zeros(params.pop("arr_shape"), dtype=np.float32)
            r = _set_keeprej(arr, **params)
            assert (int(r[0]), int(r[1])) == rec["output"], f"set_keeprej case {i}"


class TestSetMinmaxRegression:
    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["set_minmax"]

    def test_set_minmax_regression(self, cases):
        for i, rec in enumerate(cases):
            params = dict(rec["params"])
            arr = np.zeros(params.pop("arr_shape"), dtype=np.float32)
            if rec.get("error"):
                with pytest.raises(Exception):
                    _set_minmax(arr, **params)
            else:
                r = _set_minmax(arr, **params)
                assert (float(r[0]), float(r[1])) == rec["output"], f"set_minmax case {i}"


class TestSetThreshMaskRegression:
    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["set_thresh_mask"]

    def test_set_thresh_mask_regression(self, cases):
        for i, rec in enumerate(cases):
            params = dict(rec["params"])
            arr = params.pop("arr")
            mask = params.pop("mask")
            r = _set_thresh_mask(arr, mask, **params)
            np.testing.assert_array_equal(r, rec["output"], err_msg=f"set_thresh_mask case {i}")


class TestSetGainRdnsRegression:
    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["set_gain_rdns"]

    def test_set_gain_rdns_regression(self, cases):
        for i, rec in enumerate(cases):
            if rec.get("error"):
                with pytest.raises(Exception):
                    _set_gain_rdns(**rec["params"])
            else:
                r = _set_gain_rdns(**rec["params"])
                assert r[0] == rec["output"][0]
                np.testing.assert_allclose(r[1], rec["output"][1], rtol=RTOL, atol=ATOL, err_msg=f"set_gain_rdns case {i}")


class TestSetCenfuncRegression:
    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["set_cenfunc"]

    def test_set_cenfunc_regression(self, cases):
        for i, rec in enumerate(cases):
            r = _set_cenfunc(**rec["params"])
            assert r == rec["output"], f"set_cenfunc case {i}"


class TestSetCombfuncRegression:
    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["set_combfunc"]

    def test_set_combfunc_regression(self, cases):
        for i, rec in enumerate(cases):
            if rec.get("error"):
                with pytest.raises(Exception):
                    _set_combfunc(**rec["params"])
            else:
                r = _set_combfunc(**rec["params"])
                assert r == rec["output"], f"set_combfunc case {i}"


class TestSetRejectNameRegression:
    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["set_reject_name"]

    def test_set_reject_name_regression(self, cases):
        for i, rec in enumerate(cases):
            r = _set_reject_name(**rec["params"])
            assert r == rec["output"], f"set_reject_name case {i}"


class TestSetMaskRegression:
    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["set_mask"]

    def test_set_mask_regression(self, cases):
        for i, rec in enumerate(cases):
            params = dict(rec["params"])
            arr = params.pop("arr")
            mask = params.pop("mask")
            r = _set_mask(arr, mask)
            np.testing.assert_array_equal(r, rec["output"], err_msg=f"set_mask case {i}")


class TestGetDtypeLimitsRegression:
    @pytest.fixture(scope="class")
    def cases(self):
        return _load_component_cases()["get_dtype_limits"]

    def test_get_dtype_limits_regression(self, cases):
        for i, rec in enumerate(cases):
            r = _get_dtype_limits(**rec["params"])
            assert (r[0], r[1]) == rec["output"], f"get_dtype_limits case {i}"
