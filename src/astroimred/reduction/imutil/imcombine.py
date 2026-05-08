from pathlib import Path

import bottleneck as bn
import numpy as np
import pandas as pd
from astro_ndslice import (
    is_list_like,
    listify,
    offseted_shape,
)
from astropy.nddata import CCDData
from astropy.time import Time

from ..combutil import group_fits
from astroimred.mgmt.summary import fits_summary
from astroimred.mgmt.io import (
    inputs2list,
    load_ccd,
)
from astroimred.mgmt.headers import cmt2hdr
from astroimred.mgmt.io import _parse_extension
from astroimred.mgmt.misc import str_now
from astroimred.logging import logger
from . import docstrings
from ._imcombine_fits import (
    apply_output_offsets,
    calculate_zsw,
    check_stack_memory,
    extract_stack_metadata,
    init_log_table,
    load_full_stack,
    load_stack_chunk,
    log_zsw_table,
    update_hdr,
    write_imcombine_logfile,
    write_imcombine_outputs,
)
from .util_comb import (
    _set_cenfunc,
    _set_combfunc,
    _set_int_dtype,
    _set_keeprej,
    _set_mask,
    _set_reject_name,
    _set_sigma,
    _set_thresh_mask,
    _default_zsw_kw,
    do_zs,
    get_zsw,
)
from .numba_combine import combine_along_axis0_numba
from .util_reject import ccdclip_mask, minmax_mask, sigclip_mask
from . import config

__all__ = ["group_combine", "group_save", "imcombine", "ndcombine"]

"""
removed : headers, project, masktype, maskvalue, sigscale, grow
partial removal:
    * combine in ["quadrature", "nmodel"]
replaced
    * reject in ["crreject", "avsigclip"] --> ccdclip with certain params
    * offsets in ["grid", <filename>]  --> offsets in `~numpy.ndarray`

bpmasks                : ?
rejmask                : output_mask
nrejmasks              : output_nrej
expmasks               : Should I implement???
sigma                  : output_err
outtype                : dtype
outlimits              : trimsec
expname                : exposure_key

# ALGORITHM PARAMETERS ====================================================== #
lthreshold, hthreshold : thresholds (`tuple`)
nlow      , nhigh      : n_minmax (`tuple`)
nkeep                  : nkeep & maxrej
                        (IRAF nkeep > 0 && < 0 case, resp.)
mclip                  : cenfunc
lsigma    , hsigma     : sigma uple
"""


def group_combine(
    inputs,
    type_key=None,
    type_val=None,
    group_key=None,
    fmt=None,
    outdir=None,
    verbose=1,
    **kwargs,
):
    """Combine sub-groups of FITS files from the given input.

    Parameters
    ----------
    inputs : `~pandas.DataFrame`, glob pattern, `list`-like of path-like
        If `DataFrame`, it must be the summary table made by ``fm.fits_summary``.
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or `list` of
        files (each element must be path-like or `~astropy.nddata.CCDData`). Although it is not a
        good idea, a mixed `list` of `~astropy.nddata.CCDData` and paths to the files is also
        acceptable. For the purpose of ``imred.imcombine``, the best use is to
        use the `~glob` pattern or `list` of paths.

    type_key, type_val : `str`, `list` of `str`
        The header keyword for the ccd type, and the value you want to match.

    group_key : `None`, `str`, `list` of `str`, optional
        The header keyword which will be used to make groups for the CCDs that
        have selected from `type_key` and `type_val`. If `None` (default), no
        grouping will occur, but it will return the `~pandas.DataFrameGroupBy`
        object will be returned for the sake of consistency.
        Default: `None`.

    verbose : `int`, optional.
        Larger number means it becomes more verbose:

        * 0: print nothing
        * 1: only essential messages from this function
        * 2: also pass verbose mode to ``imred.imcombine``

        Default: ``1``.

    fmt : `str`, optional
        The f-string for the output file names.

        Example: if `group_key="EXPTIME"` and there are two groups where
          ``EXPTIME`` is 1.0 and 2.0,

        * ``"dark_{:.1f}s"`` gives ``dark_1.0s.fits`` and ``dark_2.0s.fits``.
        * For `float`, non-specification such as ``"d{}"`` is not recommended
          because filenames can contain long floating-point representations.

        If two `group_key` values are used, resulting in ``("B", 2.0)``,
        ``("V", 12.0)``, ...:

        * ``"flat_{2:04.1f}_{1:s}"`` gives ``"flat_02.0_B.fits"`` and
          ``"flat_12.0_V.fits"``.

        Default: `None`.

    outdir : path-like, optional
        The directory where the output fits files will be saved.

    **kwargs :
        The keyword arguments for `imcombine`.
        Default: `None`.

    Returns
    -------
    combined : `dict` of `~astropy.nddata.CCDData`
        The `dict` object where keys are the header value of the `group_key` and
        the values are the combined images in `~astropy.nddata.CCDData` object. If multiple keys
        for `group_key` is given, the key of this `dict` is a `tuple`.
    """

    def _group_save(ccd, groupname, fmt=None, verbose=1, outdir=None):
        """Saves the results."""
        outdir = Path(".") if outdir is None else Path(outdir)
        if verbose >= 1 and not outdir.exists():
            logger.info(
                "Output directory: '%s' <- does not exist! It will be newly made.", outdir
            )

        outdir.mkdir(exist_ok=True, parents=True)

        if fmt is None:
            nk = len(group_key) if is_list_like(group_key) else 1  # 1 if str
            fmt = "_".join(["{}"] * nk)
            if verbose >= 1:
                logger.warning(
                    "fmt is not specified! Output file names might be ugly."
                )

        if isinstance(groupname, tuple):
            fname = fmt.format(*groupname) + ".fits"
        else:
            fname = fmt.format(groupname) + ".fits"

        fname = fname.replace(".fits.fits", ".fits")

        fpath = outdir / fname
        if verbose >= 1:
            if fpath.exists():
                logger.info("%s will be overridden.", fpath)
            else:
                logger.info("%s", fpath)
        ccd.write(fpath, overwrite=True)

    _t = Time.now()

    if isinstance(inputs, pd.DataFrame):
        load_fits = True
        summary = inputs.copy()
    elif isinstance(inputs, str):  # glob pattern
        load_fits = True
        summary = fits_summary(inputs, verbose=verbose >= 2)
    else:
        inputs = listify(inputs)
        load_fits = False if isinstance(inputs[0], CCDData) else True
        # Assume all are CCDData if the first element is CCDData
        summary = fits_summary(inputs, verbose=verbose >= 2)

    gs, gt_key = group_fits(
        summary, type_key=type_key, type_val=type_val, group_key=group_key
    )
    if verbose >= 1:
        logger.info("Group and combine by %s (total %d groups)", group_key, len(gs))

    combined = {}
    for g_val, group in gs:
        if is_list_like(g_val):
            if len(g_val) == 1:
                g_val = g_val[0]
        files = group["file"].to_list()
        if verbose >= 1:
            logger.info("* %s... (%d files)", g_val, len(files))
        if len(files) == 0:
            if verbose >= 1:
                logger.info("No FITS to combine.")
            combined[g_val] = None
        elif len(files) == 1:
            if verbose >= 1:
                logger.info(
                    "Only 1 FITS to combine -- returning it without any modification."
                )
            combined[g_val] = load_ccd(files[0]) if load_fits else inputs[0]
            if outdir is not None or fmt is not None:
                _group_save(combined[g_val], g_val, fmt=fmt, outdir=outdir)
        else:
            combined[g_val] = imcombine(
                files if load_fits else inputs,
                verbose=verbose >= 2,
                full=False,
                **kwargs,
            )
            if outdir is not None or fmt is not None:
                _group_save(combined[g_val], g_val, fmt=fmt, outdir=outdir)

    if verbose >= 1:
        logger.info(str_now(t_ref=_t))

    return combined


def group_save(combined, fmt="", verbose=1, outdir=None):
    """Saves the group_combine results.
    Parameters
    ---------
    combined : `dict`
        The result from `group_combine` function.
    """
    outdir = Path(".") if outdir is None else Path(outdir)
    if verbose and not outdir.exists():
        logger.info(
            "Output directory: '%s' <- does not exist! It will be newly made.", outdir
        )

    outdir.mkdir(exist_ok=True, parents=True)

    if not fmt:
        fmt = "_".join(["{}"] * len(list(combined.keys())[0]))
        if verbose:
            logger.warning("fmt is not specified! Output file names might be ugly.")

    for k, ccd in combined.items():
        if isinstance(k, tuple):
            fname = fmt.format(*k) + ".fits"
        else:
            fname = fmt.format(k) + ".fits"
        fpath = outdir / fname
        if verbose >= 1 and fpath.exists():
            logger.info("The pre-existing file %s will be overridden.", fpath)
        ccd.write(fpath, overwrite=True)


def imcombine(
    inputs,
    mask=None,
    extension=None,
    extension_uncertainty=None,
    extension_mask=None,
    uncertainty_type="stddev",
    trimsec=None,
    blank=np.nan,
    offsets=None,
    thresholds=None,
    zero=None,
    zero_to_0th=True,
    zero_section=None,
    scale=None,
    scale_to_0th=True,
    scale_section=None,
    zero_kw=None,
    scale_kw=None,
    weight=None,
    reject=None,
    sigma=None,
    cenfunc="median",
    maxiters=50,
    ddof=1,
    nkeep=1,
    maxrej=None,
    n_minmax=None,
    rdnoise=0.0,
    gain=1.0,
    snoise=0.0,
    pclip=-0.5,
    logfile=None,
    combine="average",
    dtype="float32",
    dtype_err="float32",
    dtype_low=None,
    dtype_upp=None,
    irafmode=True,
    memlimit=2.5e9,
    verbose=False,
    full=False,
    return_variance=False,
    imcmb_key="$I",
    exposure_key="EXPTIME",
    output=None,
    output_mask=None,
    output_nrej=None,
    output_err=None,
    output_low=None,
    output_upp=None,
    output_rejcode=None,
    return_dict=False,
    output_verify="exception",
    overwrite=False,
    checksum=False,
):
    # === 1. Normalize defaults that must not use mutable signature values ===
    thresholds = [-np.inf, np.inf] if thresholds is None else list(thresholds)
    zero_kw = _default_zsw_kw() if zero_kw is None else dict(zero_kw)
    scale_kw = _default_zsw_kw() if scale_kw is None else dict(scale_kw)
    sigma = [3.0, 3.0] if sigma is None else sigma
    n_minmax = [1, 1] if n_minmax is None else n_minmax

    if verbose:
        _t1 = Time.now()
        logger.info(_t1.iso)
        logger.info("- Organizing...")

    # === 2. Organize inputs and output mode ===
    full = (
        full
        or output_mask is not None
        or output_nrej is not None
        or output_err is not None
        or output_low is not None
        or output_upp is not None
        or output_rejcode is not None
    )

    items = inputs2list(inputs, sort=True, accept_ccdlike=True, check_coherency=True)
    ncombine = len(items)
    reject_fullname = _set_reject_name(reject)
    int_dtype = _set_int_dtype(ncombine)
    extension = _parse_extension(extension)
    # If extensions are given as `None`, don't parse them and leave it as `None`.
    e_u = (
        None
        if extension_uncertainty is None
        else _parse_extension(extension_uncertainty)
    )
    e_m = None if extension_mask is None else _parse_extension(extension_mask)

    logfile, table_dict = init_log_table(items, logfile)

    # === 3. Read only the metadata needed to plan the full output stack ===
    metadata = extract_stack_metadata(
        items=items,
        ncombine=ncombine,
        extension=extension,
        trimsec=trimsec,
        imcmb_key=imcmb_key,
        scale=scale,
        exposure_key=exposure_key,
        reject_fullname=reject_fullname,
        gain=gain,
        rdnoise=rdnoise,
        snoise=snoise,
        dtype=dtype,
        offsets=offsets,
    )
    hdr0 = metadata["hdr0"]
    ndim = metadata["ndim"]
    shapes = metadata["shapes"]
    raw_shapes = metadata["raw_shapes"]
    offsets = metadata["offsets"]
    offset_mode = metadata["offset_mode"]
    use_wcs = metadata["use_wcs"]
    use_phy = metadata["use_phy"]
    imcmb_val = metadata["imcmb_val"]
    extract_exptime = metadata["extract_exptime"]
    scales = metadata["scales"]
    gns = metadata["gns"]
    rds = metadata["rds"]
    sns = metadata["sns"]

    # == Check the size of the temporary array for combination =========================== #
    offsets, sh_comb = offseted_shape(
        shapes, offsets, method="outer", offset_order_xyz=False, intify_offsets=True
    )

    mem_req, num_chunk, chunks = check_stack_memory(
        ncombine=ncombine,
        sh_comb=sh_comb,
        dtype=dtype,
        combine=combine,
        memlimit=memlimit,
    )
    if verbose:
        logger.info("Done.")
        if num_chunk > 1:
            logger.info("memlimit reached: Split combine by %d chunks.", num_chunk)

    if verbose:
        logger.info("- Loading, calculating offsets with zero/scale...")

    _t = Time.now()

    if num_chunk == 1:
        # == Setup offset-ed array ======================================================= #
        # NOTE: Using NaN does not set array with dtype of int... Any solution?
        arr_full, mask_full, var_full, zeros, scales, weights = load_full_stack(
            items=items,
            offsets=offsets,
            shapes=shapes,
            sh_comb=sh_comb,
            dtype=dtype,
            mask=mask,
            trimsec=trimsec,
            extension=extension,
            extension_mask=e_m,
            extension_uncertainty=e_u,
            extract_exptime=extract_exptime,
            scale=scale,
            zero=zero,
            weight=weight,
            zero_kw=zero_kw,
            scale_kw=scale_kw,
            zero_section=zero_section,
            scale_section=scale_section,
            scales=scales,
        )
    else:
        zeros, scales, weights = calculate_zsw(
            items=items,
            dtype=dtype,
            trimsec=trimsec,
            extension=extension,
            extension_mask=e_m,
            extension_uncertainty=e_u,
            extract_exptime=extract_exptime,
            scale=scale,
            zero=zero,
            weight=weight,
            zero_kw=zero_kw,
            scale_kw=scale_kw,
            zero_section=zero_section,
            scale_section=scale_section,
            scales=scales,
        )

    log_zsw_table(items, zeros, scales, weights, verbose)
    # ------------------------------------------------------------------------------------ #

    cmt2hdr(
        hdr0,
        "h",
        t_ref=_t,
        verbose=verbose,
        s=f"Loaded {ncombine} FITS, calculated zero, scale, weights",
    )

    ndcombine_kw = dict(
        combine=combine,
        reject=reject_fullname,
        scale=scales,  # it is scales , NOT scale , as it was updated above.
        zero=zeros,  # it is zeros  , NOT zero  , as it was updated above.
        weight=weights,  # it is weights, NOT weight, as it was updated above.
        zero_to_0th=zero_to_0th,
        scale_to_0th=scale_to_0th,
        scale_kw=scale_kw,
        zero_kw=zero_kw,
        thresholds=thresholds,
        n_minmax=n_minmax,
        nkeep=nkeep,
        maxrej=maxrej,
        cenfunc=cenfunc,
        sigma=sigma,
        maxiters=maxiters,
        ddof=ddof,
        rdnoise=rds,  # it is rds, not rdnoise, as it was updated above.
        gain=gns,  # it is gns, not gain   , as it was updated above.
        snoise=sns,  # it is sns, not snoise , as it was updated above.
        pclip=pclip,
        irafmode=irafmode,
        full=full,
        return_variance=return_variance,
        verbose=verbose,
    )

    # == Combine with rejection! ========================================================= #
    _t = Time.now()

    if num_chunk == 1:
        comb = ndcombine(
            arr=arr_full,
            mask=mask_full,
            copy=False,  # No need to retain arr_full.
            **ndcombine_kw,
        )

        if full:  # unpack the output
            comb, err, mask_rej, mask_thresh, low, upp, nit, rejcode = comb
            mask_total = mask_full | mask_thresh | mask_rej
        else:
            err = low = upp = mask_total = rejcode = None
    else:
        if verbose:
            logger.info("- Combining by %d chunks", num_chunk)

        comb = np.empty(sh_comb, dtype=dtype)
        err = mask_total = mask_rej = mask_thresh = low = upp = nit = rejcode = None

        for i_chunk, chunk_slices in enumerate(chunks, start=1):
            if verbose:
                logger.info("-- chunk %d/%d: %s", i_chunk, num_chunk, chunk_slices)

            arr_chunk, mask_chunk, var_chunk = load_stack_chunk(
                items=items,
                offsets=offsets,
                shapes=shapes,
                raw_shapes=raw_shapes,
                chunk_slices=chunk_slices,
                dtype=dtype,
                mask=mask,
                trimsec=trimsec,
                extension=extension,
                extension_mask=e_m,
                extension_uncertainty=e_u,
            )

            combined_chunk = ndcombine(
                arr=arr_chunk,
                mask=mask_chunk,
                copy=False,
                **ndcombine_kw,
            )

            if full:
                (
                    comb_chunk,
                    err_chunk,
                    mask_rej_chunk,
                    mask_thresh_chunk,
                    low_chunk,
                    upp_chunk,
                    nit_chunk,
                    rejcode_chunk,
                ) = combined_chunk
                mask_total_chunk = mask_chunk | mask_thresh_chunk | mask_rej_chunk

                if err is None:
                    err = np.empty(sh_comb, dtype=err_chunk.dtype)
                    mask_shape = (ncombine, *sh_comb)
                    mask_total = np.empty(mask_shape, dtype=bool)
                    mask_rej = np.empty(mask_shape, dtype=bool)
                    mask_thresh = np.empty(mask_shape, dtype=bool)
                    low = np.empty(sh_comb, dtype=low_chunk.dtype)
                    upp = np.empty(sh_comb, dtype=upp_chunk.dtype)
                    if nit_chunk is not None:
                        nit = np.empty(sh_comb, dtype=np.asarray(nit_chunk).dtype)
                    if rejcode_chunk is not None:
                        rejcode = np.empty(
                            sh_comb, dtype=np.asarray(rejcode_chunk).dtype
                        )

                comb[chunk_slices] = comb_chunk
                err[chunk_slices] = err_chunk
                mask_slices = (slice(None), *chunk_slices)
                mask_total[mask_slices] = mask_total_chunk
                mask_rej[mask_slices] = mask_rej_chunk
                mask_thresh[mask_slices] = mask_thresh_chunk
                low[chunk_slices] = low_chunk
                upp[chunk_slices] = upp_chunk
                if nit is not None:
                    nit[chunk_slices] = nit_chunk
                if rejcode is not None:
                    rejcode[chunk_slices] = rejcode_chunk
            else:
                comb[chunk_slices] = combined_chunk

            del arr_chunk, mask_chunk, var_chunk

        if not full:
            err = low = upp = mask_total = rejcode = None

    # == Update header properly ========================================================== #
    # Update WCS or PHYSICAL keywords so that "lock frame wcs", etc, on SAO
    # ds9, for example, to give proper visualization:
    apply_output_offsets(hdr0, ndim, offsets, use_wcs, use_phy)

    update_hdr(
        hdr0,
        ncombine,
        imcmb_key=imcmb_key,
        imcmb_val=imcmb_val,
        offset_mode=offset_mode,
        offsets=offsets,
        zeros=zeros,
        scales=scales,
        weights=weights,
    )

    try:
        unit = hdr0["BUNIT"].lower()
    except (KeyError, IndexError):
        unit = "adu"

    cmt2hdr(hdr0, "h", t_ref=_t, verbose=verbose, s="Rejection and combination done")
    comb = comb.astype(dtype)
    comb = CCDData(data=comb, header=hdr0, unit=unit)

    if verbose:
        logger.info("- Writing output FITS...")

    # == Save FITS files ================================================================= #
    write_imcombine_outputs(
        comb=comb,
        hdr0=hdr0,
        output=output,
        output_err=output_err,
        output_low=output_low,
        output_upp=output_upp,
        output_nrej=output_nrej,
        output_mask=output_mask,
        output_rejcode=output_rejcode,
        err=err,
        low=low,
        upp=upp,
        mask_total=mask_total,
        rejcode=rejcode,
        int_dtype=int_dtype,
        dtype=dtype,
        dtype_err=dtype_err,
        dtype_low=dtype_low,
        dtype_upp=dtype_upp,
        output_verify=output_verify,
        overwrite=overwrite,
        checksum=checksum,
    )

    if verbose:
        logger.info("Done.")

    # == Return memory... ================================================================ #
    if num_chunk == 1:
        del arr_full, mask_full
    del hdr0

    # == Write logfile =================================================================== #
    write_imcombine_logfile(
        logfile=logfile,
        table_dict=table_dict,
        ndim=ndim,
        offsets=offsets,
        zeros=zeros,
        scales=scales,
        weights=weights,
        gns=gns,
        rds=rds,
        sns=sns,
        verbose=verbose,
    )

    if verbose:
        _t2 = Time.now()
        logger.info("")
        logger.info("%s (TOTAL dt = %.3f sec)", _t2.iso, (_t2 - _t1).sec)

    # == Return ========================================================================== #
    if full:
        if return_dict:
            return dict(
                comb=comb,
                err=err,
                mask_total=mask_total,
                mask_rej=mask_rej,
                mask_thresh=mask_thresh,
                low=low,
                upp=upp,
                nit=nit,
                rejcode=rejcode,
            )
        else:
            return (
                comb,
                err,
                mask_total,
                mask_rej,
                mask_thresh,
                low,
                upp,
                nit,
                rejcode,
            )
    else:
        return comb


imcombine.__doc__ = """A helper function for ``imred.ndcombine`` to cope with FITS files.

    {}

    Parameters
    ----------

    inputs : glob pattern, `list`-like of path-like, `list`-like of `~astropy.nddata.CCDData`-like
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or `list` of
        files (each element must be path-like or `~astropy.nddata.CCDData`). Although it is not a
        good idea, a mixed `list` of `~astropy.nddata.CCDData` and paths to the files is also
        acceptable. For the purpose of ``imred.imcombine``, the best use is to
        use the `~glob` pattern or `list` of paths.

    mask : `~numpy.ndarray`, optional.
        The mask of bad pixels. If given, it must satisfy
        ``mask.shape[0]`` identical to the number of images.

        .. note::
            If the user ever want to use masking, it's more convenient to use
            ``'MASK'`` extension to the FITS files or replace bad pixel to very
            large or small numbers and use thresholds.

    extension, extension_uncertainty, extension_mask : `int`, `str`, (`str`, `int`)
        The extension of FITS, uncertainty, and mask to be used. It can be
        given as integer (0-indexing) of the extension, ``EXTNAME`` (single
        `str`), or a `tuple` of `str` and `int`: ``(EXTNAME, EXTVER)``. If `None`
        (default), the *first extension with data* will be used. If
        `extension_uncertainty` or `extension_mask` is `None` (default),
        uncertainty and mask are all ignored (turned off). Currently
        error-propagation or weighted combine is not supported, so only
        `extension_mask` can give difference to the output.

    {}

    {}

    imcmb_key : `str`
        The thing to add as ``IMCMBnnn`` in the output FITS file header. If
        ``"$I"``, following the default of IRAF, the file's name will be added.
        Otherwise, it should be a header keyword. If the key does not exist in
        ``nnn``-th file, a null string will be added. If a null string
        (``imcmb_key=""``), it does not set the ``IMCMBnnn`` keywords nor
        deletes any existing keyword.

        .. warning::
            If more than 999 files are combined, only the first 999 files will
            be recorded in the header.

    exposure_key : `str`, optional.
        The header keyword which contains the information about the exposure
        time of each FITS file. This is used only if scaling is done for
        exposure time (see `scale`).

    irafmode : `bool`, optional.
        Whether to use IRAF-like pixel restoration scheme.

    memlimit : float, optional
        Approximate memory limit in bytes for the temporary FITS stack. If the
        planned stack is larger, FITS inputs are read in row/column sections
        after offsets are applied, each section is combined, and the final
        output is stitched from the chunk results.

    output : path-like, optional
        The path to the final combined FITS file. It has dtype of `dtype` and
        dimension identical to each input image. Optional keyword arguments for
        ``fits.writeto()`` can be provided as ``**kwargs``.

    output_xxx : path-like, optional
        The output path to the mask, number of rejected pixels at each
        position, final ``nanstd(combined, ddof=ddof, axis=0)`` (if
        `return_variance` is `False`) or ``nanvar(combined, ddof=ddof,
        axis=0)`` (if `return_variance` is `True`) result, lower and upper
        bounds for rejection, and the integer codes for the rejection algorithm
        (see `mask_total`, `mask_rej`, `err`, `low`, `upp`, and `rejcode` in
        Returns.)

    return_dict : `bool`, optional.
        Whether to return the results as `dict` (works only if ``full=True``).

    Returns
    -------
    Returns the followings depending on `full` and `return_dict`.

    comb : `astropy.nddata.CCDData` (dtype `dtype`)
        The combined data.

    {}

    {}
    """.format(
    docstrings.NDCOMB_NOT_IMPLEMENTED(indent=4),
    docstrings.OFFSETS_LONG(indent=4),
    docstrings.NDCOMB_PARAMETERS_COMMON(indent=4),
    docstrings.NDCOMB_RETURNS_COMMON(indent=4),
    docstrings.IMCOMBINE_LINK(indent=4),
)


# ---------------------------------------------------------------------------------------- #
def ndcombine(
    arr,
    mask=None,
    copy=True,
    blank=np.nan,
    offsets=None,
    thresholds=None,
    zero=None,
    scale=None,
    weight=None,
    zero_kw=None,
    scale_kw=None,
    zero_to_0th=True,
    scale_to_0th=True,
    zero_section=None,
    scale_section=None,
    reject=None,
    cenfunc="median",
    sigma=None,
    maxiters=3,
    ddof=1,
    nkeep=1,
    maxrej=None,
    n_minmax=None,
    rdnoise=0.0,
    gain=1.0,
    snoise=0.0,
    pclip=-0.5,
    combine="average",
    dtype="float32",
    memlimit=2.5e9,
    irafmode=True,
    verbose=False,
    full=False,
    return_variance=False,
):
    thresholds = [-np.inf, np.inf] if thresholds is None else list(thresholds)
    zero_kw = _default_zsw_kw() if zero_kw is None else dict(zero_kw)
    scale_kw = _default_zsw_kw() if scale_kw is None else dict(scale_kw)
    sigma = [3.0, 3.0] if sigma is None else sigma
    n_minmax = [1, 1] if n_minmax is None else n_minmax

    if copy:
        arr = arr.copy()

    if np.array(arr).ndim == 1:
        raise ValueError("1-D array combination is not supported!")

    _mask = _set_mask(arr, mask)  # _mask = propagated through this function.
    sigma_lower, sigma_upper = _set_sigma(sigma)
    nkeep, maxrej = _set_keeprej(arr, nkeep, maxrej, axis=0)
    cenfunc = _set_cenfunc(cenfunc)
    reject_fullname = _set_reject_name(reject)
    maxiters = int(maxiters)
    ddof = int(ddof)

    combfunc = _set_combfunc(combine, nameonly=False, nan=True)

    if verbose and reject is not None:
        logger.info("- Rejection")
        if thresholds != [-np.inf, np.inf]:
            logger.info("-- thresholds (low, upp) = %s", thresholds)
        logger.info("-- reject=%s (irafmode=%s)", reject, irafmode)
        logger.info("--       params: nkeep=%s, maxrej=%s, maxiters=%s, cenfunc=%s", nkeep, maxrej, maxiters, cenfunc)
        if reject_fullname == "sigclip":
            logger.info("  (for sigclip): sigma=%s, ddof=%s", sigma, ddof)
        elif reject_fullname == "ccdclip":
            logger.info("  (for ccdclip): gain=%s, rdnoise=%s, snoise=%s", gain, rdnoise, snoise)
        # elif reject_fullnme == "pclip":
        #   print(f"    (for pclip)  : spclip={pclip}")
        # elif reject_fullname == "minmax":
        # print(f" (for minmaxclip): n_minmax={n_minmax}")

    # == 01 - Thresholding + Initial masking ============================================= #
    # Updating mask: _mask = _mask | mask_thresh
    mask_thresh = _set_thresh_mask(
        arr=arr, mask=_mask, thresholds=thresholds, update_mask=True
    )

    # if safemode:
    #     # Backup the pixels which are rejected by thresholding and # initial
    #     mask for future restoration (see below) for debugging # purpose.
    #     backup_thresh = arr[mask_thresh]
    #     backup_thresh_inmask = arr[_mask]

    # TODO: remove this np.nan and instead, let `get_zsw` to accept mask.
    arr[_mask] = np.nan
    # ------------------------------------------------------------------------------------ #

    # == 02 - Calculate zero, scale, weights ============================================= #
    # This should be done before rejection but after threshold masking..
    zeros, scales, weights = get_zsw(
        arr=arr,
        zero=zero,
        scale=scale,
        weight=weight,
        zero_kw=zero_kw,
        scale_kw=scale_kw,
        zero_to_0th=zero_to_0th,
        scale_to_0th=scale_to_0th,
        zero_section=zero_section,
        scale_section=scale_section,
    )
    arr = do_zs(arr, zeros=zeros, scales=scales)
    # ------------------------------------------------------------------------------------ #

    # == 02 - Rejection ================================================================== #
    if isinstance(reject_fullname, str):
        if reject_fullname == "sigclip":
            _mask_rej = sigclip_mask(
                arr,
                mask=_mask,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                maxiters=maxiters,
                ddof=ddof,
                nkeep=nkeep,
                maxrej=maxrej,
                cenfunc=cenfunc,
                axis=0,
                irafmode=irafmode,
                full=full,
            )
        elif reject_fullname == "minmax":
            _mask_rej = minmax_mask(arr, mask=_mask, n_minmax=n_minmax, full=full)
        elif reject_fullname == "ccdclip":
            _mask_rej = ccdclip_mask(
                arr,
                mask=_mask,
                sigma_lower=sigma_lower,
                sigma_upper=sigma_upper,
                scale_ref=np.mean(scales),
                zero_ref=np.mean(zeros),
                maxiters=maxiters,
                ddof=ddof,
                nkeep=nkeep,
                maxrej=maxrej,
                cenfunc=cenfunc,
                axis=0,
                gain=gain,
                rdnoise=rdnoise,
                snoise=snoise,
                irafmode=irafmode,
                full=full,
            )
        elif reject_fullname == "pclip":
            pass
        else:
            raise ValueError("reject not understood.")
        if full:
            _mask_rej, low, upp, nit, rejcode = _mask_rej
        # _mask is a subset of _mask_rej, so to extract pixels which are
        # masked PURELY due to the rejection is:
        mask_rej = _mask_rej ^ _mask
    elif reject_fullname is None:
        mask_rej = _set_mask(arr, None)
        if full:
            low = bn.nanmin(arr, axis=0)
            upp = bn.nanmax(arr, axis=0)
            nit = None
            rejcode = None
    else:
        raise ValueError("reject not understood.")

    if reject is not None and verbose:
        logger.info("Done.")

    _mask |= mask_rej

    # ------------------------------------------------------------------------------------ #

    # TODO: add "grow" rejection here?

    # == 03 - combine ==================================================================== #
    # Replace rejected / masked pixel to NaN and backup for debugging purpose.
    # This is done to reduce memory (instead of doing _arr = arr.copy())
    # backup_nan = arr[_mask]
    if verbose:
        logger.info("- Combining")
        logger.info("-- combine = %s", combine)
    arr[_mask] = np.nan

    # Combine and calc sigma
    # Original path (fallback when IMUTIL_USE_NUMBA is False or combine not supported):
    # comb = combfunc(arr, axis=0)
    if config.IMUTIL_USE_NUMBA:
        has_nan = np.any(_mask)
        comb_numba = combine_along_axis0_numba(arr, combine, has_nan=has_nan)
        if comb_numba is not None:
            comb = comb_numba
        else:
            comb = combfunc(arr, axis=0)
    else:
        comb = combfunc(arr, axis=0)
    if verbose:
        logger.info("Done.")

    # Restore NaN-replaced pixels of arr for debugging purpose.
    # arr[_mask] = backup_nan
    # arr[mask_thresh] = backup_thresh_inmask
    if full:
        if verbose:
            logger.info("- Error calculation")
            logger.info("-- to skip this, use `full=False`")
            logger.info("-- return_variance=%s, ddof=%s", return_variance, ddof)
        if return_variance:
            err = bn.nanvar(arr, ddof=ddof, axis=0)
        else:
            err = bn.nanstd(arr, ddof=ddof, axis=0)
        if verbose:
            logger.info("Done.")
        return comb, err, mask_rej, mask_thresh, low, upp, nit, rejcode
    else:
        return comb


ndcombine.__doc__ = """ Combines the given arr assuming no additional offsets.

    {}

    Offsets are not implemented in ``imred.ndcombine``; use ``imred.imcombine``
    for FITS inputs with offsets.

    Parameters
    ----------
    arr : `~numpy.ndarray`
        The array to be combined along axis 0.

    mask : `~numpy.ndarray`, optional.
        The mask of bad pixels. If given, it must satisfy ``mask.shape[0]``
        identical to the number of images.

    copy : `bool`, optional.
        Whether to copy the input array. Set to `True` if you want to keep the
        original array unchanged. Otherwise, the original array may be
        destroyed.

    {}

    {}

    Returns
    -------
    comb : `~numpy.ndarray`
        The combined array.

    {}

    {}
    """.format(
    docstrings.NDCOMB_NOT_IMPLEMENTED(indent=4),
    docstrings.OFFSETS_SHORT(indent=4),
    docstrings.NDCOMB_PARAMETERS_COMMON(indent=4),
    docstrings.NDCOMB_RETURNS_COMMON(indent=4),
    docstrings.IMCOMBINE_LINK(indent=4),
)
