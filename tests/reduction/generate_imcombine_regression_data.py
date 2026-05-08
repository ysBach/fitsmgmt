"""
Generate ground-truth regression data for imcombine/ndcombine and components.

Run from repo root:
  python -m tests.generate_imcombine_regression_data

Or from tests/:
  python generate_imcombine_regression_data.py

Output: tests/data/ndcombine_regression.pkl, tests/data/component_regression.pkl
"""

from __future__ import annotations

import itertools
import pickle
from importlib import import_module
from pathlib import Path

import numpy as np

# Import after path is set if needed
try:
    import astroimred.reduction as fir
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import astroimred.reduction as fir

fir_uc = import_module("astroimred.reduction.imutil.util_comb")
fir_ur = import_module("astroimred.reduction.imutil.util_reject")
ndcombine = fir.ndcombine
_get_dtype_limits = fir_uc._get_dtype_limits
_set_cenfunc = fir_uc._set_cenfunc
_set_combfunc = fir_uc._set_combfunc
_set_gain_rdns = fir_uc._set_gain_rdns
_set_int_dtype = fir_uc._set_int_dtype
_set_keeprej = fir_uc._set_keeprej
_set_mask = fir_uc._set_mask
_set_minmax = fir_uc._set_minmax
_set_reject_name = fir_uc._set_reject_name
_set_sigma = fir_uc._set_sigma
_set_thresh_mask = fir_uc._set_thresh_mask
do_zs = fir_uc.do_zs
get_zsw = fir_uc.get_zsw
ccdclip_mask = fir_ur.ccdclip_mask
minmax_mask = fir_ur.minmax_mask
sigclip_mask = fir_ur.sigclip_mask


REGRESSION_DATA_DIR = Path(__file__).resolve().parent / "data"
REGRESSION_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Fixed seed for reproducible arrays
DEFAULT_SEED = 12345


def make_stack(shape, seed=DEFAULT_SEED, dtype=np.float32):
    """Reproducible (N, H, W) stack."""
    rng = np.random.default_rng(seed)
    n, h, w = shape
    x = rng.normal(loc=100.0, scale=10.0, size=(n, h, w)).astype(dtype)
    return x


def _ndcombine_default_params():
    return dict(
        copy=True,
        blank=np.nan,
        offsets=None,
        zero_kw={"cenfunc": "median", "stdfunc": "std", "std_ddof": 1},
        scale_kw={"cenfunc": "median", "stdfunc": "std", "std_ddof": 1},
        zero_section=None,
        scale_section=None,
        dtype="float32",
        memlimit=2.5e9,
        verbose=False,
    )


def build_ndcombine_param_grid():
    """Build an exhaustive but finite parameter grid for ndcombine."""
    base = _ndcombine_default_params()
    # Parameter names and list of values (to be combined in meaningful subsets)
    combine_vals = ["average", "median", "sum", "min", "max", "lmedian"]
    reject_vals = [None, "sigclip", "minmax", "ccdclip"]
    zero_vals = [None, "mean", "median"]
    scale_vals = [None, "mean", "median"]
    weight_vals = [None, "mean", "median"]
    zero_to_0th_vals = [True, False]
    scale_to_0th_vals = [True, False]
    thresholds_vals = [[-np.inf, np.inf], [0.0, 100.0]]
    sigma_vals = [[3.0, 3.0], [2.0, 4.0]]
    maxiters_vals = [1, 3]
    ddof_vals = [0, 1]
    nkeep_vals = [1, 2]
    maxrej_vals = [None, 2]
    n_minmax_vals = [[1, 1], [2, 2]]
    cenfunc_vals = ["median", "mean", "lmedian"]
    irafmode_vals = [True, False]
    full_vals = [True, False]
    return_variance_vals = [True, False]

    cases = []

    # ---- 1) No rejection: vary combine, zero, scale, weight, to_0th ----
    for combine in combine_vals:
        for zero in zero_vals:
            for scale in scale_vals:
                for zero_to_0th in zero_to_0th_vals:
                    for scale_to_0th in scale_to_0th_vals:
                        for weight in weight_vals:
                            p = base.copy()
                            p.update(
                                combine=combine,
                                reject=None,
                                zero=zero,
                                scale=scale,
                                weight=weight,
                                zero_to_0th=zero_to_0th,
                                scale_to_0th=scale_to_0th,
                                thresholds=[-np.inf, np.inf],
                                full=False,
                                return_variance=False,
                            )
                            cases.append(p)

    # ---- 2) Sigclip: vary combine, sigma, maxiters, cenfunc, irafmode, full ----
    for combine in ["average", "median"]:
        for sigma in sigma_vals:
            for maxiters in maxiters_vals:
                for cenfunc in ["median", "mean"]:
                    for irafmode in irafmode_vals:
                        for full in full_vals:
                            for return_variance in [True, False] if full else [False]:
                                p = base.copy()
                                p.update(
                                    combine=combine,
                                    reject="sigclip",
                                    zero=None,
                                    scale=None,
                                    weight=None,
                                    sigma=sigma,
                                    maxiters=maxiters,
                                    ddof=1,
                                    nkeep=1,
                                    maxrej=None,
                                    cenfunc=cenfunc,
                                    irafmode=irafmode,
                                    full=full,
                                    return_variance=return_variance if full else False,
                                )
                                cases.append(p)

    # ---- 3) Minmax: vary combine, n_minmax, full ----
    for combine in ["average", "median", "sum"]:
        for n_minmax in n_minmax_vals:
            for full in full_vals:
                p = base.copy()
                p.update(
                    combine=combine,
                    reject="minmax",
                    zero=None,
                    scale=None,
                    weight=None,
                    n_minmax=n_minmax,
                    full=full,
                    return_variance=False,
                )
                cases.append(p)

    # ---- 4) Ccdclip: vary combine, sigma, full; fixed gain/rdnoise/snoise ----
    for combine in ["average", "median"]:
        for sigma in sigma_vals:
            for full in full_vals:
                p = base.copy()
                p.update(
                    combine=combine,
                    reject="ccdclip",
                    zero=None,
                    scale=None,
                    weight=None,
                    sigma=sigma,
                    maxiters=2,
                    gain=1.0,
                    rdnoise=5.0,
                    snoise=0.0,
                    full=full,
                    return_variance=False,
                )
                cases.append(p)

    # ---- 5) Thresholds with no reject and with sigclip ----
    for thresholds in thresholds_vals:
        for reject in [None, "sigclip"]:
            p = base.copy()
            p.update(
                combine="average",
                reject=reject,
                zero=None,
                scale=None,
                weight=None,
                thresholds=thresholds,
                full=False,
            )
            if reject == "sigclip":
                p["sigma"] = [3.0, 3.0]
                p["maxiters"] = 2
            cases.append(p)

    # ---- 6) nkeep / maxrej variants (sigclip) ----
    for nkeep in nkeep_vals:
        for maxrej in maxrej_vals:
            p = base.copy()
            p.update(
                combine="average",
                reject="sigclip",
                zero=None,
                scale=None,
                weight=None,
                nkeep=nkeep,
                maxrej=maxrej,
                sigma=[3.0, 3.0],
                maxiters=2,
                full=False,
            )
            cases.append(p)

    # ---- 7) ddof variants ----
    for ddof in ddof_vals:
        p = base.copy()
        p.update(
            combine="average",
            reject="sigclip",
            zero=None,
            scale=None,
            weight=None,
            ddof=ddof,
            maxiters=2,
            full=True,
            return_variance=True,
        )
        cases.append(p)

    return cases


def run_ndcombine_case(shape, seed, params):
    """Run ndcombine once; return (params, result dict with comb, err, ...)."""
    arr = make_stack(shape, seed=seed)
    p = dict(params)
    full = p.get("full", False)
    try:
        out = ndcombine(arr, **p)
    except Exception as e:
        return {"params": p, "shape": shape, "seed": seed, "error": str(e), "output": None}
    if full:
        comb, err, mask_rej, mask_thresh, low, upp, nit, rejcode = out
        output = {
            "comb": np.asarray(comb).copy(),
            "err": np.asarray(err).copy(),
            "mask_rej": np.asarray(mask_rej).copy(),
            "mask_thresh": np.asarray(mask_thresh).copy(),
            "low": np.asarray(low).copy(),
            "upp": np.asarray(upp).copy(),
            "nit": np.asarray(nit).copy(),
            "rejcode": np.asarray(rejcode).copy() if rejcode is not None else None,
        }
    else:
        output = {
            "comb": np.asarray(out).copy(),
            "err": None,
            "mask_rej": None,
            "mask_thresh": None,
            "low": None,
            "upp": None,
            "nit": None,
            "rejcode": None,
        }
    return {"params": p, "shape": shape, "seed": seed, "error": None, "output": output}


def generate_ndcombine_regression(shapes=None, param_grid=None):
    """Run full ndcombine grid and return list of case results."""
    if shapes is None:
        shapes = [(3, 4, 4), (5, 6, 6), (7, 8, 8)]
    if param_grid is None:
        param_grid = build_ndcombine_param_grid()
    cases = []
    case_idx = 0
    for shape in shapes:
        for p in param_grid:
            case = run_ndcombine_case(shape, DEFAULT_SEED, p)
            case["case_id"] = f"nd_{shape[0]}_{shape[1]}_{shape[2]}_{case_idx}"
            case_idx += 1
            cases.append(case)
    return cases


def build_component_param_grids():
    """Grids for get_zsw, do_zs, sigclip_mask, minmax_mask, ccdclip_mask."""
    rng = np.random.default_rng(DEFAULT_SEED)
    arr_small = rng.normal(10.0, 2.0, (5, 4, 4)).astype(np.float32)

    get_zsw_cases = []
    for zero in [None, "mean", "median"]:
        for scale in [None, "mean", "median"]:
            for weight in [None, "mean"]:
                for zero_to_0th in [True, False]:
                    for scale_to_0th in [True, False]:
                        get_zsw_cases.append({
                            "arr": arr_small.copy(),
                            "zero": zero,
                            "scale": scale,
                            "weight": weight,
                            "zero_kw": {"cenfunc": "median", "stdfunc": "std", "std_ddof": 1},
                            "scale_kw": {"cenfunc": "median", "stdfunc": "std", "std_ddof": 1},
                            "zero_to_0th": zero_to_0th,
                            "scale_to_0th": scale_to_0th,
                            "zero_section": None,
                            "scale_section": None,
                        })

    do_zs_cases = []
    for _ in range(10):
        n, h, w = 5, 4, 4
        a = rng.normal(10.0, 1.0, (n, h, w)).astype(np.float32)
        z = rng.uniform(-1, 1, n).astype(np.float32)
        s = rng.uniform(0.5, 2.0, n).astype(np.float32)
        do_zs_cases.append({"arr": a, "zeros": z, "scales": s})

    sigclip_cases = []
    arr5 = rng.normal(100.0, 10.0, (7, 6, 6)).astype(np.float32)
    for sigma in [[3.0, 3.0], [2.0, 4.0]]:
        for maxiters in [1, 3]:
            for cenfunc in ["median", "mean"]:
                for irafmode in [True, False]:
                    for full in [True, False]:
                        sigclip_cases.append({
                            "arr": arr5.copy(),
                            "mask": None,
                            "sigma": sigma[0] if sigma[0] == sigma[1] else sigma,
                            "sigma_lower": sigma[0],
                            "sigma_upper": sigma[1],
                            "maxiters": maxiters,
                            "ddof": 1,
                            "nkeep": 1,
                            "maxrej": None,
                            "cenfunc": cenfunc,
                            "irafmode": irafmode,
                            "axis": 0,
                            "full": full,
                        })

    minmax_cases = []
    for n_minmax in [[1, 1], [2, 2], [0, 1]]:
        for full in [True, False]:
            minmax_cases.append({
                "arr": arr5.copy(),
                "mask": None,
                "n_minmax": n_minmax,
                "full": full,
            })

    ccdclip_cases = []
    for sigma in [3.0, [2.0, 4.0]]:
        for full in [True, False]:
            ccdclip_cases.append({
                "arr": arr5.copy(),
                "mask": None,
                "sigma": sigma,
                "maxiters": 2,
                "gain": 1.0,
                "rdnoise": 5.0,
                "snoise": 0.0,
                "scale_ref": 1.0,
                "zero_ref": 0.0,
                "full": full,
            })

    # _set_* / _get_dtype_limits grids (exhaustive small param sets)
    arr_set = rng.normal(10.0, 2.0, (5, 4, 4)).astype(np.float32)
    set_int_dtype_cases = [{"ncombine": n} for n in [1, 10, 255, 256, 1000, 65535, 65536, 100000]]
    set_sigma_cases = []
    for sigma in [3.0, [3.0, 3.0], [2.0, 4.0], np.array([1.5])]:
        for sl in [None, 1.0]:
            for su in [None, 5.0]:
                set_sigma_cases.append({"sigma": sigma, "sigma_lower": sl, "sigma_upper": su})
    set_keeprej_cases = []
    for nkeep in [None, 1, 2, 0.5]:
        for maxrej in [None, 1, 2, 0.5]:
            set_keeprej_cases.append({"arr": arr_set.copy(), "nkeep": nkeep, "maxrej": maxrej, "axis": 0})
    set_minmax_cases = []
    for n_minmax in [[1, 1], [2, 2], [0, 1], [1], 0.1]:
        set_minmax_cases.append({"arr": arr_set.copy(), "n_minmax": n_minmax, "axis": 0})
    set_thresh_mask_cases = []
    for thresh in [[-np.inf, np.inf], [0, 100], [-1, 1], [50, 50]]:
        for update in [True, False]:
            mask = np.zeros_like(arr_set, dtype=bool)
            set_thresh_mask_cases.append({"arr": arr_set.copy(), "mask": mask.copy(), "thresholds": thresh, "update_mask": update})
    set_gain_rdns_cases = []
    for val in ["header", 1.0, 5.0, np.array([1.0, 2.0, 3.0, 4.0, 5.0])]:
        set_gain_rdns_cases.append({"gain_or_rdnoise": val, "ncombine": 5, "dtype": "float32"})
    set_cenfunc_cases = []
    for cf in [None, "median", "med", "mean", "avg", "lmedian", "lmd"]:
        for shorten in [True, False]:
            set_cenfunc_cases.append({"cenfunc": cf, "shorten": shorten, "nameonly": True, "nan": True})
    set_combfunc_cases = []
    for cf in [None, "median", "average", "sum", "min", "max", "lmedian", "and", "or"]:
        set_combfunc_cases.append({"combfunc": cf, "shorten": False, "nameonly": True, "nan": True})
    set_reject_name_cases = [{"reject": r} for r in [None, "sigclip", "sig", "minmax", "mm", "ccdclip", "ccd", "pclip"]]
    set_mask_cases = []
    set_mask_cases.append({"arr": arr_set.copy(), "mask": None})
    set_mask_cases.append({"arr": arr_set.copy(), "mask": np.zeros_like(arr_set, dtype=bool)})
    set_mask_cases.append({"arr": arr_set.copy(), "mask": (arr_set > 12).astype(bool)})
    get_dtype_limits_cases = [{"dtype": d} for d in [np.uint8, np.int32, np.float32, np.float64]]

    return {
        "get_zsw": get_zsw_cases,
        "do_zs": do_zs_cases,
        "sigclip_mask": sigclip_cases,
        "minmax_mask": minmax_cases,
        "ccdclip_mask": ccdclip_cases,
        "set_int_dtype": set_int_dtype_cases,
        "set_sigma": set_sigma_cases,
        "set_keeprej": set_keeprej_cases,
        "set_minmax": set_minmax_cases,
        "set_thresh_mask": set_thresh_mask_cases,
        "set_gain_rdns": set_gain_rdns_cases,
        "set_cenfunc": set_cenfunc_cases,
        "set_combfunc": set_combfunc_cases,
        "set_reject_name": set_reject_name_cases,
        "set_mask": set_mask_cases,
        "get_dtype_limits": get_dtype_limits_cases,
    }


def run_component_cases():
    """Run all component grids and return dict of name -> list of {params, output}."""
    grids = build_component_param_grids()
    out = {}

    # get_zsw
    out["get_zsw"] = []
    for i, g in enumerate(grids["get_zsw"]):
        g = dict(g)
        arr = g.pop("arr")
        z, s, w = get_zsw(
            arr,
            zero=g["zero"],
            scale=g["scale"],
            weight=g["weight"],
            zero_kw=g["zero_kw"],
            scale_kw=g["scale_kw"],
            zero_to_0th=g["zero_to_0th"],
            scale_to_0th=g["scale_to_0th"],
            zero_section=g["zero_section"],
            scale_section=g["scale_section"],
        )
        out["get_zsw"].append({
            "params": {"arr": arr.copy(), **g},
            "output": (np.asarray(z).copy(), np.asarray(s).copy(), np.asarray(w).copy()),
        })

    # do_zs (do_zs mutates arr in place; store original arr in params, result in output)
    out["do_zs"] = []
    for i, g in enumerate(grids["do_zs"]):
        arr_original = g["arr"].copy()
        arr_work = arr_original.copy()
        result = do_zs(arr_work, zeros=g["zeros"], scales=g["scales"], copy=False)
        out["do_zs"].append({
            "params": {"arr": arr_original, "zeros": g["zeros"].copy(), "scales": g["scales"].copy()},
            "output": np.asarray(result).copy(),
        })

    # sigclip_mask
    out["sigclip_mask"] = []
    for g in grids["sigclip_mask"]:
        g = dict(g)
        arr = g.pop("arr")
        sigma = g.get("sigma")
        if isinstance(sigma, (list, tuple)):
            sl, su = sigma[0], sigma[1]
        else:
            sl = su = sigma
        r = sigclip_mask(
            arr,
            mask=g.get("mask"),
            sigma=sigma,
            sigma_lower=g.get("sigma_lower", sl),
            sigma_upper=g.get("sigma_upper", su),
            maxiters=g["maxiters"],
            ddof=g["ddof"],
            nkeep=g["nkeep"],
            maxrej=g.get("maxrej"),
            cenfunc=g["cenfunc"],
            irafmode=g["irafmode"],
            axis=g.get("axis", 0),
            full=g["full"],
        )
        if g["full"]:
            out["sigclip_mask"].append({"params": {"arr": arr.copy(), **g}, "output": (np.asarray(r[0]).copy(), np.asarray(r[1]).copy(), np.asarray(r[2]).copy(), np.asarray(r[3]).copy(), np.asarray(r[4]).copy())})
        else:
            out["sigclip_mask"].append({"params": {"arr": arr.copy(), **g}, "output": np.asarray(r).copy()})

    # minmax_mask
    out["minmax_mask"] = []
    for g in grids["minmax_mask"]:
        g = dict(g)
        arr = g.pop("arr")
        r = minmax_mask(arr, mask=g.get("mask"), n_minmax=g["n_minmax"], full=g["full"])
        if g["full"]:
            out["minmax_mask"].append({"params": {"arr": arr.copy(), **g}, "output": tuple(np.asarray(x).copy() for x in r)})
        else:
            out["minmax_mask"].append({"params": {"arr": arr.copy(), **g}, "output": np.asarray(r).copy()})

    # ccdclip_mask
    out["ccdclip_mask"] = []
    for g in grids["ccdclip_mask"]:
        g = dict(g)
        arr = g.pop("arr")
        r = ccdclip_mask(
            arr,
            mask=g.get("mask"),
            sigma=g["sigma"],
            maxiters=g["maxiters"],
            gain=g["gain"],
            rdnoise=g["rdnoise"],
            snoise=g["snoise"],
            scale_ref=g["scale_ref"],
            zero_ref=g["zero_ref"],
            full=g["full"],
        )
        if g["full"]:
            out["ccdclip_mask"].append({"params": {"arr": arr.copy(), **g}, "output": tuple(np.asarray(x).copy() for x in r)})
        else:
            out["ccdclip_mask"].append({"params": {"arr": arr.copy(), **g}, "output": np.asarray(r).copy()})

    # _set_* and _get_dtype_limits
    def _run_setter(name, run_fn, cases_list):
        out[name] = []
        for g in cases_list:
            try:
                result = run_fn(**g)
                # Serialize: arrays -> copy(), dtype -> str, tuples stay, None stays
                if hasattr(result, "copy"):
                    ser = np.asarray(result).copy()
                elif isinstance(result, tuple):
                    ser = tuple(
                        np.asarray(x).copy() if hasattr(x, "copy") else x
                        for x in result
                    )
                elif hasattr(result, "dtype"):
                    ser = str(result)
                else:
                    ser = result
                out[name].append({"params": g, "output": ser})
            except Exception as e:
                out[name].append({"params": g, "output": None, "error": str(e)})

    for g in grids["set_int_dtype"]:
        g_ = dict(g)
        r = _set_int_dtype(**g_)
        out.setdefault("set_int_dtype", []).append({"params": g_, "output": str(r)})
    for g in grids["set_sigma"]:
        g_ = dict(g)
        try:
            r = _set_sigma(**g_)
            out.setdefault("set_sigma", []).append({"params": g_, "output": (float(r[0]), float(r[1]))})
        except Exception as e:
            out.setdefault("set_sigma", []).append({"params": g_, "output": None, "error": str(e)})
    for g in grids["set_keeprej"]:
        g_ = dict(g)
        arr = g_.pop("arr")
        r = _set_keeprej(arr, **g_)
        out.setdefault("set_keeprej", []).append({"params": {"arr_shape": arr.shape, **g_}, "output": (int(r[0]), int(r[1]))})
    for g in grids["set_minmax"]:
        g_ = dict(g)
        arr = g_.pop("arr")
        try:
            r = _set_minmax(arr, **g_)
            out.setdefault("set_minmax", []).append({"params": {"arr_shape": arr.shape, **g_}, "output": (float(r[0]), float(r[1]))})
        except Exception as e:
            out.setdefault("set_minmax", []).append({"params": {"arr_shape": arr.shape, **g_}, "output": None, "error": str(e)})
    for g in grids["set_thresh_mask"]:
        g_ = dict(g)
        arr = g_.pop("arr")
        mask = g_.pop("mask")
        mask_orig = mask.copy()
        r = _set_thresh_mask(arr, mask, **g_)
        out.setdefault("set_thresh_mask", []).append({"params": {"arr": arr.copy(), "mask": mask_orig, **g_}, "output": np.asarray(r).copy()})
    for g in grids["set_gain_rdns"]:
        g_ = dict(g)
        try:
            r = _set_gain_rdns(**g_)
            out.setdefault("set_gain_rdns", []).append({"params": g_, "output": (r[0], np.asarray(r[1]).copy())})
        except Exception as e:
            out.setdefault("set_gain_rdns", []).append({"params": g_, "output": None, "error": str(e)})
    for g in grids["set_cenfunc"]:
        g_ = dict(g)
        r = _set_cenfunc(**g_)
        out.setdefault("set_cenfunc", []).append({"params": g_, "output": r})
    for g in grids["set_combfunc"]:
        g_ = dict(g)
        try:
            r = _set_combfunc(**g_)
            out.setdefault("set_combfunc", []).append({"params": g_, "output": r})
        except Exception as e:
            out.setdefault("set_combfunc", []).append({"params": g_, "output": None, "error": str(e)})
    for g in grids["set_reject_name"]:
        g_ = dict(g)
        r = _set_reject_name(**g_)
        out.setdefault("set_reject_name", []).append({"params": g_, "output": r})
    for g in grids["set_mask"]:
        g_ = dict(g)
        arr = g_.pop("arr")
        mask = g_.pop("mask")
        r = _set_mask(arr, mask)
        out.setdefault("set_mask", []).append({
            "params": {"arr": arr.copy(), "mask": mask.copy() if mask is not None else None},
            "output": np.asarray(r).copy(),
        })
    for g in grids["get_dtype_limits"]:
        g_ = dict(g)
        r = _get_dtype_limits(**g_)
        out.setdefault("get_dtype_limits", []).append({"params": g_, "output": (r[0], r[1])})

    return out


def main():
    print("Generating ndcombine regression cases...")
    ndcombine_cases = generate_ndcombine_regression()
    n_nd = len(ndcombine_cases)
    n_err = sum(1 for c in ndcombine_cases if c.get("error"))
    print(f"  ndcombine: {n_nd} cases ({n_err} errors)")

    out_path_nd = REGRESSION_DATA_DIR / "ndcombine_regression.pkl"
    with open(out_path_nd, "wb") as f:
        pickle.dump(ndcombine_cases, f, protocol=4)
    print(f"  Wrote {out_path_nd}")

    print("Generating component regression cases...")
    component_cases = run_component_cases()
    for name, lst in component_cases.items():
        print(f"  {name}: {len(lst)} cases")
    out_path_comp = REGRESSION_DATA_DIR / "component_regression.pkl"
    with open(out_path_comp, "wb") as f:
        pickle.dump(component_cases, f, protocol=4)
    print(f"  Wrote {out_path_comp}")
    print("Done.")


if __name__ == "__main__":
    main()
