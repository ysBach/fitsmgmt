from pathlib import Path

import numpy as np
from astro_ndslice import calc_offset_physical, calc_offset_wcs, slicefy
from astropy.io import fits
from astropy.io.fits.verify import VerifyError
from astropy.nddata import CCDData
from astropy.table import Table
from astropy.wcs import WCS

from astroimred.mgmt.headers import update_tlm
from astroimred.mgmt.io import _parse_data_header, get_size, load_ccd, write2fits
from astroimred.mgmt.logging import logger

from .util_comb import _set_combfunc, _set_gain_rdns, get_zsw


def _slice_shape(shape, slices):
    return tuple(
        len(range(*sl.indices(int(size)))) for size, sl in zip(shape, slices)
    )


def _trim_slices(trimsec, shape):
    if trimsec is None:
        return tuple(slice(None) for _ in shape)
    return tuple(slicefy(trimsec, ndim=len(shape)))


def _trimmed_shape(shape, trimsec):
    return _slice_shape(shape, _trim_slices(trimsec, shape))


def _compose_trim_data_slices(trimsec, data_slices, raw_shape):
    trim_slices = _trim_slices(trimsec, raw_shape)
    slices = []
    for raw_size, trim_slice, data_slice in zip(raw_shape, trim_slices, data_slices):
        t_start, _t_stop, t_step = trim_slice.indices(int(raw_size))
        if t_step <= 0:
            raise ValueError("Negative-step trimsec is not supported in chunked load.")

        trimmed_size = len(range(*trim_slice.indices(int(raw_size))))
        d_start, d_stop, d_step = data_slice.indices(trimmed_size)
        if d_step != 1:
            raise ValueError("Non-unit chunk slice steps are not supported.")
        slices.append(
            slice(t_start + d_start * t_step, t_start + d_stop * t_step, t_step)
        )
    return tuple(slices)


def _hdu_has_data(hdu):
    return hdu.header.get("NAXIS", 0) > 0 and all(
        hdu.header.get(f"NAXIS{i}", 0) > 0
        for i in range(1, hdu.header.get("NAXIS", 0) + 1)
    )


def _get_image_hdu(hdul, extension):
    try:
        hdu = hdul[extension]
    except (KeyError, IndexError, TypeError):
        return None

    if _hdu_has_data(hdu):
        return hdu

    if extension == 0:
        for hdu in hdul:
            if _hdu_has_data(hdu):
                return hdu
    return None


def _read_hdul_section(hdul, extension, section):
    if extension is None:
        return None

    hdu = _get_image_hdu(hdul, extension)
    if hdu is None:
        return None
    return np.asarray(hdu.section[section])


def update_hdr(
    header,
    ncombine,
    imcmb_key,
    imcmb_val,
    offset_mode=None,
    offsets=None,
    zeros=None,
    scales=None,
    weights=None,
):
    """**Inplace** update of the given header"""

    def __rm_and_add(hdr, keybase, values):
        for i in range(999):
            if f"{keybase}{i+1:03d}" in hdr:
                del hdr[f"{keybase}{i+1:03d}"]
            else:
                break

        for i in range(min(999, len(values))):
            hdr[f"{keybase}{i+1:03d}"] = values[i]

        return

    header["NCOMBINE"] = (ncombine, "Number of combined images")
    if imcmb_key != "":
        header["IMCMBKEY"] = (imcmb_key, "Key used in IMCMBiii ('$I': filepath)")
        __rm_and_add(header, "IMCMB", imcmb_val)
        # remove header keyword IMCMBiii if it exists:
        for i in range(999):
            if f"IMCMB{i+1:03d}" in header:
                del header[f"IMCMB{i+1:03d}"]
            else:
                break

        for i in range(min(999, len(imcmb_val))):
            header[f"IMCMB{i+1:03d}"] = imcmb_val[i]

    if offset_mode is not None:
        header["OFFSTMOD"] = (offset_mode, "Offset method used for combine.")
        for i in range(min(999, len(imcmb_val))):
            header[f"OFFST{i:03d}"] = str(offsets[i,][::-1].tolist())

    if not np.all(zeros == 0):
        __rm_and_add(header, "ZERO", zeros)

    if not np.all(scales == 1):
        __rm_and_add(header, "SCALE", scales)

    if not np.all(weights == 1):
        __rm_and_add(header, "WEIGH", weights)

    # Add "IRAF-TLM" like header key for continuity with IRAF.
    update_tlm(header)


def init_log_table(items, logfile):
    if logfile is None:
        return None, None

    logfile = Path(logfile)
    table_dict = dict(file=[], filesize=[])
    for item in items:
        try:
            fpath = Path(item)
            item_size = fpath.stat().st_size
        except (TypeError, ValueError, FileNotFoundError):
            fpath = f"User-provided {item.__class__.__name__}"
            item_size = get_size(item)
        table_dict["file"].append(fpath)
        table_dict["filesize"].append(item_size)

    return logfile, table_dict


def setup_offsets(offsets, ncombine, ndim, hdr0):
    use_wcs, use_phy = False, False
    w_ref = None

    if isinstance(offsets, str):
        if offsets.lower() in ["world", "wcs"]:
            w_ref = WCS(hdr0)
            use_wcs = True
            offset_mode = "WCS"
            offsets = np.zeros((ncombine, ndim))
        elif offsets.lower() in ["physical", "phys", "phy"]:
            use_phy = True
            offset_mode = "Physical"
            offsets = np.zeros((ncombine, ndim))
        else:
            raise ValueError("offsets not understood.")
    elif offsets is None:
        offset_mode = None
        offsets = np.zeros((ncombine, ndim))
    else:
        if offsets.shape[0] != ncombine:
            raise ValueError("offset.shape[0] must be num(images)")
        offset_mode = "User"
        offsets = np.array(offsets)

    return offsets, offset_mode, use_wcs, use_phy, w_ref


def extract_stack_metadata(
    items,
    ncombine,
    extension,
    trimsec,
    imcmb_key,
    scale,
    exposure_key,
    reject_fullname,
    gain,
    rdnoise,
    snoise,
    dtype,
    offsets,
):
    # == Extract header info ============================================================= #
    # TODO: if offsets is None and `fsize_tot` << memlimit, why not
    # just load all data here?
    _, hdr0 = _parse_data_header(items[0], extension=extension, parse_data=False)
    ndim = hdr0["NAXIS"]
    # N x ndim. sizes[i, :] = images[i].shape
    shapes = np.ones((ncombine, ndim), dtype=int)
    raw_shapes = np.ones((ncombine, ndim), dtype=int)
    extract_hdr = imcmb_key not in [None, "", "$I"]

    extract_exptime = False
    if isinstance(scale, str):
        if scale.lower() in ["exp", "expos", "exposure", "exptime"]:
            extract_exptime = True

    # === 1. Determine which calibration keywords are needed for rejection ===
    if reject_fullname == "ccdclip":
        extract_gain, gns = _set_gain_rdns(gain, ncombine, dtype=dtype)
        extract_rdnoise, rds = _set_gain_rdns(rdnoise, ncombine, dtype=dtype)
        extract_snoise, sns = _set_gain_rdns(snoise, ncombine, dtype=dtype)
    else:
        extract_gain, gns = False, 1
        extract_rdnoise, rds = False, 0
        extract_snoise, sns = False, 0

    # === 2. Interpret offset mode and initialize per-image offsets ===
    offsets, offset_mode, use_wcs, use_phy, w_ref = setup_offsets(
        offsets, ncombine, ndim, hdr0
    )

    scales = np.ones(shape=ncombine)
    imcmb_val = []
    extract_hdr = (
        extract_hdr
        or extract_exptime
        or extract_gain
        or extract_rdnoise
        or extract_snoise
        or use_wcs
        or use_phy
    )

    for i, item in enumerate(items):
        if extract_hdr:
            _, hdr = _parse_data_header(item, extension=extension, copy=False)
            if imcmb_key not in [None, ""]:
                if imcmb_key == "$I":
                    try:
                        imcmb_val.append(Path(item).name)
                    except TypeError:
                        imcmb_val.append(f"User-provided {type(item)}")
                else:
                    imcmb_val.append(hdr.get(imcmb_key, ""))

            if extract_exptime:
                scales[i] = float(hdr[exposure_key])
            if extract_gain:
                gns[i] = float(hdr[gain])
            if extract_rdnoise:
                rds[i] = float(hdr[rdnoise])
            if extract_snoise:
                sns[i] = float(hdr[snoise])

            if hdr["NAXIS"] != ndim:
                raise ValueError(
                    "All FITS files must have the identical ndim, "
                    + "though they can have different sizes."
                )

            # Update offsets if WCS or Physical should be used
            if use_wcs:
                # Code if using WCS, which may be much slower (but accurate?)
                # Find the center's pixel position in w_ref, in nearest integer value.
                offsets[i,] = calc_offset_wcs(
                    WCS(hdr),
                    w_ref,
                    intify_offset=True,
                    loc_target="center",
                    loc_reference="center",
                    order_xyz=False,
                )
                # For IRAF-like calculation, use
                #   offsets[i, ] = [hdr[f'CRPIX{i}'] for i in range(ndim, 0, -1)]
            elif use_phy:
                offsets[i,] = calc_offset_physical(
                    hdr, None, intify_offset=True, order_xyz=False, ignore_ltm=True
                )

            # NOTE: the indexing in python is [z, y, x] order!!
            raw_shape = tuple(int(hdr[f"NAXIS{i}"]) for i in range(ndim, 0, -1))
            raw_shapes[i,] = raw_shape
            shapes[i,] = _trimmed_shape(raw_shape, trimsec)
        else:
            if imcmb_key == "$I":
                try:
                    imcmb_val.append(Path(item).name)
                except TypeError:
                    imcmb_val.append(f"User-provided {type(item)}")
            data = _parse_data_header(item, extension=extension, parse_header=False)[0]
            raw_shapes[i,] = data.shape
            if trimsec is not None:
                shapes[i,] = _trimmed_shape(data.shape, trimsec)
            else:
                shapes[i,] = data.shape

    return dict(
        hdr0=hdr0,
        ndim=ndim,
        shapes=shapes,
        raw_shapes=raw_shapes,
        offsets=offsets,
        offset_mode=offset_mode,
        use_wcs=use_wcs,
        use_phy=use_phy,
        imcmb_val=imcmb_val,
        extract_exptime=extract_exptime,
        scales=scales,
        gns=gns,
        rds=rds,
        sns=sns,
    )


def check_stack_memory(ncombine, sh_comb, dtype, combine, memlimit):
    # Size of (N+1)-D array before combining along axis=0
    stacksize = np.prod((ncombine, *sh_comb)) * (np.dtype(dtype).itemsize)
    # size estimated by full-stacked array (1st term) plus combined image
    # (1/ncombine), low and upp bounds (each 1/ncombine), mask (bool8),
    # niteration (int8), and code(int8). temp_arr_size = stacksize*(1 +
    # 1/ncombine*4)

    # Copied from ccdproc v 2.0.1
    # https://github.com/astropy/ccdproc/blob/b9ec64dfb59aac1d9ca500ad172c4eb31ec305f8/ccdproc/combiner.py#L710
    # Set a memory use factor based on profiling
    combmeth = _set_combfunc(combine)
    memory_factor = 3 if combmeth == "median" else 2
    memory_factor *= 1.5
    mem_req = memory_factor * stacksize
    if memlimit is None or memlimit <= 0 or mem_req <= memlimit:
        return mem_req, 1, [tuple(slice(0, size) for size in sh_comb)]

    # FITS stores the last Python axis contiguously.  Prefer chunking the first
    # image axis so each read keeps the full fast axis and stays row-slab-like
    # for normal 2-D images.  If one row slab is still too large, move toward
    # the fast axis until at least one section fits.
    chunk_axis = None
    chunk_size = None
    min_required = np.inf
    for axis in range(len(sh_comb)):
        fast_shape = sh_comb[:axis] + sh_comb[axis + 1 :]
        bytes_per_axis_pixel = (
            memory_factor * ncombine * np.prod(fast_shape) * np.dtype(dtype).itemsize
        )
        min_required = min(min_required, bytes_per_axis_pixel)
        size = int(memlimit // bytes_per_axis_pixel)
        if size >= 1:
            chunk_axis = axis
            chunk_size = size
            break

    if chunk_axis is None:
        raise ValueError(
            "memlimit is too small to hold even one FITS chunk. "
            + f"Try memlimit > {min_required:.1e}."
        )

    chunks = []
    for start in range(0, sh_comb[chunk_axis], chunk_size):
        stop = min(start + chunk_size, sh_comb[chunk_axis])
        slices = [slice(0, size) for size in sh_comb]
        slices[chunk_axis] = slice(start, stop)
        chunks.append(tuple(slices))

    return mem_req, len(chunks), chunks


def calculate_zsw(
    items,
    dtype,
    trimsec,
    extension,
    extension_mask,
    extension_uncertainty,
    extract_exptime,
    scale,
    zero,
    weight,
    zero_kw,
    scale_kw,
    zero_section,
    scale_section,
    scales,
):
    ncombine = len(items)
    zeros = np.zeros(shape=ncombine)
    weights = np.ones(shape=ncombine)

    calc_zero = isinstance(zero, str)
    calc_scale = isinstance(scale, str) and not extract_exptime
    calc_weight = isinstance(weight, str)

    if zero is not None and not calc_zero:
        zeros = np.asarray(zero, dtype=float).ravel()
        if zeros.size != ncombine:
            raise ValueError("zero must have size equal to the number of images.")

    if scale is not None and not isinstance(scale, str):
        scales = np.asarray(scale, dtype=float).ravel()
        if scales.size != ncombine:
            raise ValueError("scale must have size equal to the number of images.")

    if weight is not None and not calc_weight:
        weights = np.asarray(weight, dtype=float).ravel()
        if weights.size != ncombine:
            raise ValueError("weight must have size equal to the number of images.")

    for i, item in enumerate(items):
        needs_data = calc_zero or calc_scale or calc_weight
        if needs_data:
            # Preserve the legacy global zero/scale/weight semantics.  These
            # statistics must not be recalculated per chunk.
            data, _var, _mask = load_imcombine_item(
                item,
                trimsec=trimsec,
                extension=extension,
                extension_mask=extension_mask,
                extension_uncertainty=extension_uncertainty,
            )
        else:
            continue

        z_i, s_i, w_i = get_zsw(
            arr=np.array(data[None, :]),  # make a fake (N+1)-D array
            zero=zero if calc_zero else None,
            scale=scale if calc_scale else None,
            weight=weight if calc_weight else None,
            zero_kw=zero_kw,
            scale_kw=scale_kw,
            zero_to_0th=False,  # to retain original zero
            scale_to_0th=False,  # to retain original scale
            zero_section=zero_section,
            scale_section=scale_section,
        )
        if calc_zero:
            zeros[i] = z_i[0]
        if calc_scale:
            scales[i] = s_i[0]
        if calc_weight:
            weights[i] = w_i[0]

    return zeros, scales, weights


def load_imcombine_item(
    item,
    trimsec,
    extension,
    extension_mask,
    extension_uncertainty,
):
    try:
        data, var, mask, _ = load_ccd(
            item,
            trimsec=trimsec,
            ccddata=False,
            extension=extension,
            extension_mask=extension_mask,
            extension_uncertainty=extension_uncertainty,
            full=True,
        )
    except TypeError:
        if isinstance(item, CCDData):
            slices = _trim_slices(trimsec, item.data.shape)
            data = item.data[slices].copy()
            if item.mask is None:
                mask = np.zeros(data.shape, dtype=bool)
            else:
                mask = item.mask[slices].copy()
            var = (
                None
                if item.uncertainty is None
                else np.asarray(item.uncertainty.array)[slices].copy()
            )
        else:
            raise ValueError("Each item is not path-like or CCDData.")

    return data, var, mask


def load_imcombine_item_region(
    item,
    data_slices,
    raw_shape,
    trimsec,
    extension,
    extension_mask,
    extension_uncertainty,
):
    section = _compose_trim_data_slices(trimsec, data_slices, raw_shape)
    try:
        path = Path(item)
    except TypeError:
        if not isinstance(item, CCDData):
            raise ValueError("Each item is not path-like or CCDData.")
        data = item.data[section].copy()
        if item.mask is None:
            mask = np.zeros(data.shape, dtype=bool)
        else:
            mask = item.mask[section].copy()
        var = (
            None
            if item.uncertainty is None
            else np.asarray(item.uncertainty.array)[section].copy()
        )
        return data, var, mask

    with fits.open(path, memmap=True) as hdul:
        data = _read_hdul_section(hdul, extension, section)
        if data is None:
            raise ValueError(f"No image data found in {path}.")
        var = _read_hdul_section(hdul, extension_uncertainty, section)
        mask = _read_hdul_section(hdul, extension_mask, section)
    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)
    else:
        mask = mask.astype(bool, copy=False)
    return data, var, mask


def load_full_stack(
    items,
    offsets,
    shapes,
    sh_comb,
    dtype,
    mask,
    trimsec,
    extension,
    extension_mask,
    extension_uncertainty,
    extract_exptime,
    scale,
    zero,
    weight,
    zero_kw,
    scale_kw,
    zero_section,
    scale_section,
    scales,
):
    ncombine = len(items)
    zeros = np.zeros(shape=ncombine)
    weights = np.ones(shape=ncombine)
    var_full = None
    if extension_uncertainty is not None:
        var_full = np.nan * np.zeros(shape=(ncombine, *sh_comb), dtype=dtype)

    arr_full = np.nan * np.zeros(shape=(ncombine, *sh_comb), dtype=dtype)
    mask_full = np.zeros(shape=(ncombine, *sh_comb), dtype=bool)

    for i, (item, offset, shape) in enumerate(zip(items, offsets, shapes)):
        # -- Set slice ------------------------------------------------------------------- #
        # offsets2slice is introduced much later than the code below was written,
        # so not used here..
        slices = [i]
        # offset & size at each j-th dimension axis
        for offset_j, shape_j in zip(offset, shape):
            slices.append(slice(offset_j, offset_j + shape_j, None))
        slices = tuple(slices)

        # -- Load data ------------------------------------------------------------------- #
        data, var, item_mask = load_imcombine_item(
            item,
            trimsec=trimsec,
            extension=extension,
            extension_mask=extension_mask,
            extension_uncertainty=extension_uncertainty,
        )

        if mask is not None:
            item_mask |= mask[i,]

        # -- zero and scale -------------------------------------------------------------- #
        # better to calculate here than from full array, as the
        # latter may contain too many NaNs due to offest shifting.
        # TODO: let get_zsw to get functionals for zsw, so _set_calc_zsw
        # will not be repeated for every iteration.
        scale_i = scales[i] if extract_exptime else scale
        z_i, s_i, w_i = get_zsw(
            arr=np.array(data[None, :]),  # make a fake (N+1)-D array
            zero=zero,
            scale=scale_i,
            weight=weight,
            zero_kw=zero_kw,
            scale_kw=scale_kw,
            zero_to_0th=False,  # to retain original zero
            scale_to_0th=False,  # to retain original scale
            zero_section=zero_section,
            scale_section=scale_section,
        )
        zeros[i] = z_i[0]
        scales[i] = s_i[0]
        weights[i] = w_i[0]

        # -- Insertion ------------------------------------------------------------------- #
        arr_full[slices] = data
        mask_full[slices] = item_mask
        if var is not None and var_full is not None:
            var_full[slices] = var

    return arr_full, mask_full, var_full, zeros, scales, weights


def load_stack_chunk(
    items,
    offsets,
    shapes,
    raw_shapes,
    chunk_slices,
    dtype,
    mask,
    trimsec,
    extension,
    extension_mask,
    extension_uncertainty,
):
    ncombine = len(items)
    chunk_shape = tuple(sl.stop - sl.start for sl in chunk_slices)
    var_chunk = None
    if extension_uncertainty is not None:
        var_chunk = np.nan * np.zeros(shape=(ncombine, *chunk_shape), dtype=dtype)

    arr_chunk = np.nan * np.zeros(shape=(ncombine, *chunk_shape), dtype=dtype)
    mask_chunk = np.zeros(shape=(ncombine, *chunk_shape), dtype=bool)

    chunk_starts = np.array([sl.start for sl in chunk_slices])
    chunk_stops = np.array([sl.stop for sl in chunk_slices])

    for i, (item, offset, shape, raw_shape) in enumerate(
        zip(items, offsets, shapes, raw_shapes)
    ):
        image_starts = offset
        image_stops = offset + shape
        starts = np.maximum(chunk_starts, image_starts)
        stops = np.minimum(chunk_stops, image_stops)
        if np.any(stops <= starts):
            continue

        data_slices = tuple(
            slice(int(start - image_start), int(stop - image_start))
            for start, stop, image_start in zip(starts, stops, image_starts)
        )
        insert_slices = tuple(
            slice(int(start - chunk_start), int(stop - chunk_start))
            for start, stop, chunk_start in zip(starts, stops, chunk_starts)
        )

        data, var, item_mask = load_imcombine_item_region(
            item=item,
            data_slices=data_slices,
            raw_shape=raw_shape,
            trimsec=trimsec,
            extension=extension,
            extension_mask=extension_mask,
            extension_uncertainty=extension_uncertainty,
        )

        if mask is not None:
            item_mask |= mask[i,][data_slices]

        full_insert_slices = (i, *insert_slices)
        arr_chunk[full_insert_slices] = data
        mask_chunk[full_insert_slices] = item_mask
        if var is not None and var_chunk is not None:
            var_chunk[full_insert_slices] = var

    return arr_chunk, mask_chunk, var_chunk


def log_zsw_table(items, zeros, scales, weights, verbose):
    if not verbose:
        return
    logger.info("Done.")
    if isinstance(items[0], str):
        logger.info("")
        logger.info("-" * 80)
        logger.info(
            "{:^45s}|{:^9s}|{:^9s}|{:^9s}".format("input", "zero", "scale", "weight")
        )
        logger.info("-" * 80)
        for item, z, s, w in zip(items, zeros, scales, weights):
            logger.info("{:>45s}|{:3e}|{:3e}|{:3e}".format(item[-45:], z, s, w))
        logger.info("-" * 80)
        logger.info("")


def apply_output_offsets(header, ndim, offsets, use_wcs, use_phy):
    if use_wcs:  # NOTE: the indexing in python is [z, y, x] order!!
        for i in range(ndim, 0, -1):
            header[f"CRPIX{i}"] += offsets[0][ndim - i]

    if use_phy:  # NOTE: the indexing in python is [z, y, x] order!!
        for i in range(ndim, 0, -1):
            header[f"LTV{i}"] += offsets[0][ndim - i]


def write_imcombine_outputs(
    comb,
    hdr0,
    output,
    output_err,
    output_low,
    output_upp,
    output_nrej,
    output_mask,
    output_rejcode,
    err,
    low,
    upp,
    mask_total,
    rejcode,
    int_dtype,
    dtype,
    dtype_err,
    dtype_low,
    dtype_upp,
    output_verify,
    overwrite,
    checksum,
):
    write_kw = dict(output_verify=output_verify, overwrite=overwrite, checksum=checksum)
    if output is not None:
        try:
            comb.write(output, **write_kw)
        except VerifyError:
            raise VerifyError("Use output_verify='fix'")

    if output_err is not None:
        err = err.astype(dtype_err)
        write2fits(err, hdr0, output_err, return_ccd=False, **write_kw)

    if output_low is not None:
        low = low.astype(dtype) if dtype_low is None else low.astype(dtype_low)
        write2fits(low, hdr0, output_low, return_ccd=False, **write_kw)

    if output_upp is not None:
        upp = upp.astype(dtype) if dtype_upp is None else upp.astype(dtype_upp)
        write2fits(upp, hdr0, output_upp, return_ccd=False, **write_kw)

    if output_nrej is not None:  # Do this BEFORE output_mask!!
        nrej = np.count_nonzero(mask_total, axis=0).astype(int_dtype)
        write2fits(nrej, hdr0, output_nrej, return_ccd=False, **write_kw)

    if output_mask is not None:  # Do this AFTER output_nrej!!
        # FITS does not accept boolean. We need uint8.
        write2fits(
            mask_total.astype(np.uint8), hdr0, output_mask, return_ccd=False, **write_kw
        )

    if output_rejcode is not None:
        write2fits(rejcode, hdr0, output_rejcode, return_ccd=False, **write_kw)


def write_imcombine_logfile(
    logfile,
    table_dict,
    ndim,
    offsets,
    zeros,
    scales,
    weights,
    gns,
    rds,
    sns,
    verbose,
):
    if logfile is None:
        return
    if verbose:
        logger.info("- Writing summary table...")

    table_dict["scales"] = list(scales)
    table_dict["zeros"] = list(zeros)
    table_dict["weights"] = list(weights)
    table = Table(table_dict)
    table["gains"] = gns
    table["readnoises"] = rds
    table["snoises"] = sns
    # NOTE: the indexing in python is [z, y, x] order!!
    for i in range(ndim, 0, -1):
        table[f"offset{i}"] = offsets[:, ndim - i]
    table.write(logfile, format="csv")
    if verbose:
        logger.info("Done.")
