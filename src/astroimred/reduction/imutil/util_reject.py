import bottleneck as bn
import numpy as np

from astroimred.logging import logger

from . import config, docstrings
from .numba_reject import ccdclip_std_numba, reject_minmax_numba, reject_sigclip_numba
from .util_comb import (
    _get_dtype_limits,
    _set_cenfunc,
    _set_gain_rdns,
    _set_keeprej,
    _set_mask,
    _set_minmax,
    _set_sigma,
    _setup_reject,
    do_zs,
)

__all__ = ["sigclip_mask", "ccdclip_mask", "minmax_mask"]


def _iter_rej(
    arr,
    mask=None,
    sigma_lower=3.0,
    sigma_upper=3.0,
    maxiters=5,
    ddof=0,
    nkeep=3,
    maxrej=None,
    cenfunc="median",
    ccdclip=False,
    irafmode=True,
    rdnoise_ref=0.0,
    snoise_ref=0.0,
    scale_ref=1,
    zero_ref=0,
):
    """The common function for iterative rejection algorithms.

    Parameters
    ----------
    arr : `~numpy.ndarray`
        The array to find the mask. It must be gain-corrected if
        ``ccdclip=True``.

    rdnoise_ref, snoise_ref : `float`
        The representative readnoise and sensitivity noise to estimate the
        error-bar for ``ccdclip=True``.

    scale_ref, zero_ref : `float`
        The representative scaling and zeroing value to estimate the error-bar
        for ``ccdclip=True``.
    """

    def __calc_censtd(_arr):
        # most are defined in upper _iter_rej function
        cen = cenfunc(_arr, axis=0)
        if ccdclip:  # use abs(pix value) to avoid NaN from negative pixels.
            # Calculate: sqrt((1 + snoise_ref) * abs(cen + zero_ref) * scale_ref + rdnoise_ref**2)
            std = ccdclip_std_numba(
                cen=cen,
                snoise_ref=snoise_ref,
                zero_ref=zero_ref,
                scale_ref=scale_ref,
                rdnoise_ref=rdnoise_ref,
            )
            if std is None:
                std = np.sqrt(
                    (1 + snoise_ref) * np.abs(cen + zero_ref) * scale_ref
                    + rdnoise_ref**2
                )
        else:
            std = bn.nanstd(_arr, axis=0, ddof=ddof)

        return cen, std

    # General setup
    _arr, _masks, keeprej, cenfunc, _nvals, lowupp = _setup_reject(
        arr=arr, mask=mask, nkeep=nkeep, maxrej=maxrej, cenfunc=cenfunc
    )
    mask_nan, mask_nkeep, mask_maxrej, mask_pix = _masks
    nkeep, maxrej = keeprej
    nit, ncombine, n_finite_old = _nvals
    low, upp, low_new, upp_new = lowupp

    nrej = ncombine - n_finite_old
    k = 0
    # mask_pix is where **NO** rejection should occur.
    if (nkeep == 0) and (maxrej == ncombine):
        logger.info("nkeep, maxrej turned off.")
        # no need to check mask_pix iteratively
        while k < maxiters:
            cen, std = __calc_censtd(_arr=_arr)
            low_new[~mask_pix] = (cen - sigma_lower * std)[~mask_pix]
            upp_new[~mask_pix] = (cen + sigma_upper * std)[~mask_pix]

            # In numpy, > or < automatically applies along axis=0!!
            mask_bound = (_arr < low_new) | (_arr > upp_new) | ~np.isfinite(_arr)
            _arr[mask_bound] = np.nan

            n_finite_new = ncombine - np.count_nonzero(mask_bound, axis=0)
            n_change = n_finite_old - n_finite_new
            total_change = np.sum(n_change)

            mask_nochange = n_change == 0  # identical to say "max-iter reached"

            # no need to backup
            if total_change == 0:
                break

            # I put the test below because I thought it will be quicker to halt
            # clipping if all pixels are masked. But now I feel testing this in
            # every iteration is an unnecessary overhead for "nearly
            # impossible" situation.
            # - ysBach (2020-10-14 21:15:44 (KST: GMT+09:00))
            # if np.all(mask_pix):
            #     break

            # update only non-masked pixels
            nrej[~mask_pix] = n_change[~mask_pix]
            # update only changed pixels
            nit[~mask_nochange] += 1
            k += 1
            n_finite_old = n_finite_new

    else:
        while k < maxiters:
            cen, std = __calc_censtd(_arr=_arr)
            low_new[~mask_pix] = (cen - sigma_lower * std)[~mask_pix]
            upp_new[~mask_pix] = (cen + sigma_upper * std)[~mask_pix]

            # In numpy, > or < automatically applies along axis=0!!
            mask_bound = (_arr < low_new) | (_arr > upp_new) | ~np.isfinite(_arr)
            _arr[mask_bound] = np.nan

            n_finite_new = ncombine - np.count_nonzero(mask_bound, axis=0)
            n_change = n_finite_old - n_finite_new
            total_change = np.sum(n_change)

            mask_nochange = n_change == 0  # identical to say "max-iter reached"
            mask_nkeep = (ncombine - nrej) < nkeep
            mask_maxrej = nrej > maxrej

            # mask pixel position if any of these happened. Including
            # mask_nochange here will not change results but only spend more
            # time.
            mask_pix = mask_nkeep | mask_maxrej

            # revert to the previous ones if masked.
            # By doing this, pixels which was mask_nkeep now, e.g., will again
            # be True in mask_nkeep in the next iter but unchanged. This should
            # be done at every iteration (unfortunately) because, e.g., if
            # nkeep is very large, excessive rejection may happen for many
            # times, and the restoration CANNOT be done after all the
            # iterations.
            low_new[mask_pix] = low[mask_pix].copy()
            upp_new[mask_pix] = upp[mask_pix].copy()
            low = low_new
            upp = upp_new

            if total_change == 0:
                break

            # I put the test below because I thought it will be quicker to halt
            # clipping if all pixels are masked. But now I feel testing this in
            # every iteration is an unnecessary overhead for
            # "nearly impossible" situation.
            # - ysBach (2020-10-14 21:15:44 (KST: GMT+09:00))
            # if np.all(mask_pix):
            #     break

            # update only non-masked pixels
            nrej[~mask_pix] = n_change[~mask_pix]
            # update only changed pixels
            nit[~mask_nochange] += 1
            k += 1
            n_finite_old = n_finite_new

    mask = mask_nan | (arr < low_new) | (arr > upp_new)

    code = np.zeros(_arr.shape[1:], dtype=np.uint8)
    if maxiters == 0:
        code += 1
    else:
        code += (2 * mask_nochange + 4 * mask_nkeep + 8 * mask_maxrej).astype(np.uint8)

    if irafmode:
        n_minimum = max(nkeep, ncombine - maxrej)
        if n_minimum > 0:
            try:
                resid = np.abs(_arr - cen)
            except UnboundLocalError:  # cen undefined when maxiters=0
                resid = np.abs(_arr - cenfunc(_arr, axis=0))
            # need this cuz bn.argpartition cannot handle NaN:
            resid[np.isnan(resid)] = _get_dtype_limits(resid.dtype)[1]
            # ^ replace with max of dtype
            # after this, resid is guaranteed to have **NO** NaN values.

            resid_cut = np.max(
                bn.partition(resid, n_minimum, axis=0)[:n_minimum,], axis=0
            )
            mask[resid <= resid_cut] = False

    # Note the mask returned here is mask from rejection PROPAGATED with the
    # input mask. So to extract the pixels masked PURELY from rejection, you
    # need ``mask_output^mask_input`` because the input mask is a subset of the
    # output one.

    return (mask, low, upp, nit, code)


# TODO: let `cenfunc` be function object...?
# ************************************************************************************ #
# *                                    SIGMA-CLIPPING                                * #
# ************************************************************************************ #
def sigclip_mask(
    arr: np.ndarray,
    mask: np.ndarray | None = None,
    sigma: float = 3.0,
    sigma_lower: float | None = None,
    sigma_upper: float | None = None,
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 3,
    maxrej: int | None = None,
    cenfunc: str = "median",
    irafmode: bool = False,
    axis: int = 0,
    full: bool = True,
) -> np.ndarray | tuple:
    if axis != 0:
        raise ValueError("Currently only axis=0 is supported")

    mask = _set_mask(arr, mask)
    sigma_lower, sigma_upper = _set_sigma(sigma, sigma_lower, sigma_upper)
    nkeep, maxrej = _set_keeprej(arr, nkeep, maxrej, axis)
    cenfunc = _set_cenfunc(cenfunc)
    maxiters = int(maxiters)
    ddof = int(ddof)

    # Original path (fallback when IMUTIL_USE_NUMBA is False or arr not 3D):
    # o_mask, o_low, o_upp, o_nit, o_code = _iter_rej(
    #     arr=arr,
    #     mask=mask,
    #     sigma_lower=sigma_lower,
    #     sigma_upper=sigma_upper,
    #     maxiters=maxiters,
    #     ddof=ddof,
    #     nkeep=nkeep,
    #     maxrej=maxrej,
    #     cenfunc=cenfunc,
    #     ccdclip=False,
    #     irafmode=irafmode,
    # )
    if config.IMUTIL_USE_NUMBA and arr.ndim == 3:
        result = reject_sigclip_numba(
            arr,
            mask,
            sigma_lower,
            sigma_upper,
            maxiters,
            ddof,
            nkeep,
            maxrej,
            cenfunc,
            irafmode,
            ccdclip=False,
            rdnoise_ref=0.0,
            snoise_ref=0.0,
            scale_ref=1.0,
            zero_ref=0.0,
        )
        if result is not None:
            o_mask, o_low, o_upp, o_nit, o_code = result
        else:
            o_mask, o_low, o_upp, o_nit, o_code = _iter_rej(
                arr=arr,
                mask=mask,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                maxiters=maxiters,
                ddof=ddof,
                nkeep=nkeep,
                maxrej=maxrej,
                cenfunc=cenfunc,
                ccdclip=False,
                irafmode=irafmode,
            )
    else:
        o_mask, o_low, o_upp, o_nit, o_code = _iter_rej(
            arr=arr,
            mask=mask,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            maxiters=maxiters,
            ddof=ddof,
            nkeep=nkeep,
            maxrej=maxrej,
            cenfunc=cenfunc,
            ccdclip=False,
            irafmode=irafmode,
        )
    if full:
        return o_mask, o_low, o_upp, o_nit, o_code
    else:
        return o_mask


sigclip_mask.__doc__ = f""" Finds masks of `arr` by sigma-clipping.

    Parameters
    ----------
    arr : `~numpy.ndarray`
        The array to be subjected for masking. `arr` and `mask` must
        have the identical shape.

    {docstrings.REJECT_PARAMETERS_COMMON(indent=4)}
    {docstrings.REJECT_PARAMETERS_SIGMA(indent=4)}

    Returns
    -------
    {docstrings.REJECT_RETURNS_SIGMA(indent=4)}

    """


# ************************************************************************************ #
# *                                    MINMAX CLIPPING                               * #
# ************************************************************************************ #
def _minmax(arr, mask=None, q_low=0, q_upp=0, calc_low=True, calc_upp=True):
    # General setup (nkeep and maxrej as dummy)
    _arr, _masks, _, _, _nvals, _lowupp = _setup_reject(
        arr=arr, mask=mask, nkeep=1, maxrej=None, cenfunc=None
    )
    # mask == input_mask | ~isfinite(arr)
    mask, _, _, mask_skiprej = _masks  # nkeep and maxrej not used in MINMAX.
    _, ncombine, n_old = _nvals  # nit is not used in MINMAX.
    low, upp, _, _ = _lowupp  # low_new, upp_new are not used in MINMAX.

    # adding 0.001 following IRAF; per-pixel counts to avoid over-rejecting
    # pixels that have fewer finite values due to an inhomogeneous input mask.
    n_rej_low = (n_old * q_low + 0.001).astype(int)  # shape == _arr.shape[1:]
    n_rej_upp = (n_old * q_upp + 0.001).astype(int)

    if np.any(n_rej_low > 0):
        _arr[mask] = np.inf  # push already-masked to the high end when sorting
        # Double-argsort gives per-element rank (0 = smallest) along axis 0.
        rank_low = np.argsort(np.argsort(_arr, axis=0), axis=0)
        mask |= rank_low < n_rej_low[np.newaxis]
        # low stays as the pre-rejection nanmin from _setup_reject (same as
        # the original argpartition path and the numba path).

    if np.any(n_rej_upp > 0):
        _arr[mask] = -np.inf  # push rejected/masked to the low end
        # Negate so that the largest original values get the smallest ranks.
        rank_upp = np.argsort(np.argsort(-_arr, axis=0), axis=0)
        mask |= rank_upp < n_rej_upp[np.newaxis]
        # upp stays as the pre-rejection nanmax from _setup_reject.

    code = np.zeros(_arr.shape[1:], dtype=np.uint8)
    no_rej = (n_rej_low == 0) | (n_rej_upp == 0)
    code += (1 * no_rej).astype(np.uint8)

    return (mask, low, upp, 1, code)


def minmax_mask(
    arr: np.ndarray,
    mask: np.ndarray | None = None,
    n_minmax: list[int] | None = None,
    full: bool = True,
) -> np.ndarray | tuple:
    n_minmax = [1, 1] if n_minmax is None else n_minmax
    mask = _set_mask(arr, mask)
    q_low, q_upp = _set_minmax(arr, n_minmax, axis=0)
    # Original path (fallback when IMUTIL_USE_NUMBA is False or arr not 3D):
    # o_mask, o_low, o_upp, o_nit, o_code = _minmax(
    #     arr, mask=mask, q_low=q_low, q_upp=q_upp, calc_low=full, calc_upp=full
    # )
    if config.IMUTIL_USE_NUMBA and arr.ndim == 3:
        result = reject_minmax_numba(
            arr, mask, q_low, q_upp, calc_low=full, calc_upp=full
        )
        if result is not None:
            o_mask, o_low, o_upp, code = result
            o_nit = np.ones(arr.shape[1:], dtype=np.uint8)  # minmax always 1 iteration
            o_code = code
        else:
            o_mask, o_low, o_upp, o_nit, o_code = _minmax(
                arr, mask=mask, q_low=q_low, q_upp=q_upp, calc_low=full, calc_upp=full
            )
    else:
        o_mask, o_low, o_upp, o_nit, o_code = _minmax(
            arr, mask=mask, q_low=q_low, q_upp=q_upp, calc_low=full, calc_upp=full
        )
    if full:
        return o_mask, o_low, o_upp, o_nit, o_code
    else:
        return o_mask


minmax_mask.__doc__ = f""" Finds masks of `arr` after rejecting `n_minmax` pixels.

    Parameters
    ----------
    arr : `~numpy.ndarray`
        The array to be subjected for masking. `arr` and `mask` must have the
        identical shape. It must be in DN, i.e., **not** gain corrected.

    {docstrings.REJECT_PARAMETERS_COMMON(indent=4)}
    {docstrings.REJECT_PARAMETERS_SIGMA(indent=4)}

    Returns
    -------
    {docstrings.REJECT_RETURNS_SIGMA(indent=4)}

    """

# ************************************************************************************ #
# *                              PERCENTILE CLIPPING (PCLIP)                         * #
# ************************************************************************************ #


# ************************************************************************************ #
# *                          CCD NOISE MODEL CLIPPING (CCDCLIP)                      * #
# ************************************************************************************ #
def ccdclip_mask(
    arr: np.ndarray,
    mask: np.ndarray | None = None,
    sigma: float = 3.0,
    sigma_lower: float | None = None,
    sigma_upper: float | None = None,
    maxiters: int = 5,
    ddof: int = 0,
    nkeep: int = 3,
    maxrej: int | None = None,
    cenfunc: str = "median",
    irafmode: bool = False,
    axis: int = 0,
    gain: float = 1.0,
    rdnoise: float = 0.0,
    snoise: float = 0.0,
    scale_ref: float = 1,
    zero_ref: float = 0,
    dtype: str = "float32",
    full: bool = True,
) -> np.ndarray | tuple:
    if axis != 0:
        raise ValueError("Currently only axis=0 is supported")

    mask = _set_mask(arr, mask)
    ncombine = arr.shape[0]
    sigma_lower, sigma_upper = _set_sigma(sigma, sigma_lower, sigma_upper)
    nkeep, maxrej = _set_keeprej(arr, nkeep, maxrej, axis)
    cenfunc = _set_cenfunc(cenfunc)
    maxiters = int(maxiters)
    ddof = int(ddof)
    _, gns = _set_gain_rdns(gain, ncombine, dtype=dtype)
    _, rds = _set_gain_rdns(rdnoise, ncombine, dtype=dtype)
    _, sns = _set_gain_rdns(snoise, ncombine, dtype=dtype)

    # Convert to gain-corrected (Use copy!)
    arr = do_zs(arr, zeros=None, scales=1 / gns, copy=True)

    # Original path (fallback when IMUTIL_USE_NUMBA is False or arr not 3D):
    # o_mask, o_low, o_upp, o_nit, o_code = _iter_rej(
    #     arr=arr,
    #     mask=mask,
    #     sigma_lower=sigma_lower,
    #     sigma_upper=sigma_upper,
    #     maxiters=maxiters,
    #     scale_ref=scale_ref,
    #     zero_ref=zero_ref,
    #     ddof=ddof,
    #     nkeep=nkeep,
    #     maxrej=maxrej,
    #     cenfunc=cenfunc,
    #     ccdclip=True,
    #     rdnoise_ref=np.mean(rds),  # Use mean as the representative value
    #     snoise_ref=np.mean(sns),  # Use mean as the representative value
    #     irafmode=irafmode,
    # )
    rdnoise_mean = np.mean(rds)
    snoise_mean = np.mean(sns)
    if config.IMUTIL_USE_NUMBA and arr.ndim == 3:
        result = reject_sigclip_numba(
            arr,
            mask,
            sigma_lower,
            sigma_upper,
            maxiters,
            ddof,
            nkeep,
            maxrej,
            cenfunc,
            irafmode,
            ccdclip=True,
            rdnoise_ref=rdnoise_mean,
            snoise_ref=snoise_mean,
            scale_ref=scale_ref,
            zero_ref=zero_ref,
        )
        if result is not None:
            o_mask, o_low, o_upp, o_nit, o_code = result
        else:
            o_mask, o_low, o_upp, o_nit, o_code = _iter_rej(
                arr=arr,
                mask=mask,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                maxiters=maxiters,
                scale_ref=scale_ref,
                zero_ref=zero_ref,
                ddof=ddof,
                nkeep=nkeep,
                maxrej=maxrej,
                cenfunc=cenfunc,
                ccdclip=True,
                rdnoise_ref=rdnoise_mean,
                snoise_ref=snoise_mean,
                irafmode=irafmode,
            )
    else:
        o_mask, o_low, o_upp, o_nit, o_code = _iter_rej(
            arr=arr,
            mask=mask,
            sigma_lower=sigma_lower,
            sigma_upper=sigma_upper,
            maxiters=maxiters,
            scale_ref=scale_ref,
            zero_ref=zero_ref,
            ddof=ddof,
            nkeep=nkeep,
            maxrej=maxrej,
            cenfunc=cenfunc,
            ccdclip=True,
            rdnoise_ref=rdnoise_mean,
            snoise_ref=snoise_mean,
            irafmode=irafmode,
        )

    # Revert to ADU (DN)
    arr = do_zs(arr, zeros=None, scales=gns)

    if full:
        return o_mask, o_low, o_upp, o_nit, o_code
    else:
        return o_mask


ccdclip_mask.__doc__ = f""" Finds masks of `arr` by CCD noise model.

    Parameters
    ----------
    arr : `~numpy.ndarray`
        The array to be subjected for masking. `arr` and `mask` must have the
        identical shape. It must be in DN, i.e., **not** gain corrected.

    {docstrings.REJECT_PARAMETERS_COMMON(indent=4)}
    {docstrings.REJECT_PARAMETERS_SIGMA(indent=4)}

    Returns
    -------
    {docstrings.REJECT_RETURNS_SIGMA(indent=4)}

    """
