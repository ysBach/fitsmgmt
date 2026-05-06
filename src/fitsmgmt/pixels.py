"""Pixel mask, interpolation, saturation, and extrema helpers."""

import numpy as np
from astro_ndslice import bezel2slice
from astropy.nddata import CCDData
from astropy.time import Time

from . import headers, io as _io

__all__ = [
    "fixpix",
    "find_extpix",
    "find_satpix",
]


def fixpix(
    ccd,
    mask=None,
    maskpath=None,
    extension=None,
    mask_extension=None,
    priority=None,
    update_header=True,
    verbose=True,
):
    """Interpolate the masked location (N-D generalization of IRAF PROTO.FIXPIX)

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, or number-like
        The CCD data to be "fixed".

    mask : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, optional.
        The mask to be used for fixing pixels (pixels to be fixed are where
        `mask` is `True`). If `None`, nothing will happen and `ccd` is
        returned.

    extension, mask_extension: `int`, `str`, (`str`, `int`), `None`
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.
        Default: `None`.

    priority: `tuple` of `int`, `None`, optional.
        The priority of axis as a `tuple` of non-repeating `int` from ``0`` to
        `ccd.ndim`. It will be used if the mask has the same size along two or
        more of the directions. To specify, use the integers for axis
        directions, descending priority. For example,  ``(2, 1, 0)`` will be
        identical to `priority=None` (default) for 3-D images.
        Default is `None` to follow IRAF's PROTO.FIXPIX: Priority is higher for
        larger axis number (e.g., in 2-D, x-axis (axis=1) has higher priority
        than y-axis (axis=0)).

    Examples
    --------
    Timing test: MBP 15" [2018, macOS 11.4, i7-8850H (2.6 GHz; 6-core), RAM 16
    GB (2400MHz DDR4), Radeon Pro 560X (4GB)], 2021-11-05 11:14:04 (KST:
    GMT+09:00)

    >>> np.random.RandomState(123)  # RandomState(MT19937) at 0x7FAECA768D40
    >>> data = np.random.normal(size=(1000, 1000))
    >>> mask = np.zeros_like(data).astype(bool)
    >>> mask[10, 10] = True
    >>> %timeit fm.fixpix(data, mask)
    19.7 ms +- 1.53 ms per loop (mean +- std. dev. of 7 runs, 100 loops each)

    >>> print(data[9:12, 9:12], fm.fixpix(data, mask)[9:12, 9:12])
    # [[ 1.64164502 -1.00385046 -1.24748504]
    #  [-1.31877621  1.37965928  0.66008966]
    #  [-0.7960262  -0.14613834 -1.34513327]]
    # [[ 1.64164502 -1.00385046 -1.24748504]
    #  [-1.31877621 -0.32934328  0.66008966]
    #  [-0.7960262  -0.14613834 -1.34513327]] adu
    """
    if mask is None:
        return ccd.copy()

    _t_start = Time.now()

    _ccd, _, _ = _io._parse_image(ccd, extension=extension, force_ccddata=True)
    mask, maskpath, _ = _io._parse_image(
        mask, extension=mask_extension, name=maskpath, force_ccddata=True
    )
    mask = mask.data.astype(bool)
    data = _ccd.data
    naxis = _ccd.shape

    if _ccd.shape != mask.shape:
        raise ValueError(
            f"ccd and mask must have the identical shape; now {_ccd.shape} VS {mask.shape}."
        )

    ndim = data.ndim

    if priority is None:
        priority = tuple([i for i in range(ndim)][::-1])
    elif len(priority) != ndim:
        raise ValueError(
            "len(priority) and ccd.ndim must be the same; "
            + f"now {len(priority)} VS {ccd.ndim}."
        )
    elif not isinstance(priority, tuple):
        priority = tuple(priority)
    elif (np.min(priority) != 0) or (np.max(priority) != ndim - 1):
        raise ValueError(
            f"`priority` must be a tuple of int (0 <= int <= {ccd.ndim-1=}). "
            + f"Now it's {priority=}"
        )

    structures = [np.zeros([3] * ndim) for _ in range(ndim)]
    for i in range(ndim):
        sls = [[slice(1, 2, None)] * ndim for _ in range(ndim)][0]
        sls[i] = slice(None, None, None)
        structures[i][tuple(sls)] = 1
    # structures[i] is the structure to obtain the num. of connected pix. along axis=i

    pixels = []
    n_axs = []
    labels = []

    for structure in structures:
        from scipy.ndimage import label as ndlabel

        _label, _nlabel = ndlabel(mask, structure=structure)
        _pixels = {}
        _n_axs = {}
        for k in range(1, _nlabel + 1):
            _label_k = _label == k
            _pixels[k] = np.where(_label_k)
            _n_axs[k] = np.count_nonzero(_label_k)
        labels.append(_label)
        pixels.append(_pixels)
        n_axs.append(_n_axs)

    idxs = np.where(mask)
    for pos in np.array(idxs).T:
        # The label of this position in each axis
        label_pos = [lab.item(*pos) for lab in labels]
        # number of pixels of the same label for each direction
        n_ax = [_n_ax[lab] for _n_ax, lab in zip(n_axs, label_pos)]

        # The shortest axis along which the interpolation will happen,
        # OR, if 1+ directions having same minimum length, select this axis
        #   according to `priority`
        interp_ax = np.where(n_ax == np.min(n_ax))[0]
        if len(interp_ax) > 1:
            for i_ax in priority:  # check in the identical order to `priority`
                if i_ax in interp_ax:
                    interp_ax = i_ax
                    break
        else:
            interp_ax = interp_ax[0]
        # The coordinates of the pixels having the identical label to this
        # pixel position, along the shortest axis
        coord_samelabel = pixels[interp_ax][label_pos[interp_ax]]
        coord_slice = []
        coord_init = []
        coord_last = []
        for i in range(ndim):
            invalid = False
            if i == interp_ax:
                init = np.min(coord_samelabel[i]) - 1
                last = np.max(coord_samelabel[i]) + 1
                # distance between the initial/last points to be used for the
                # interpolation, along the interpolation axis:
                delta = last - init
                # grid for interpolation:
                grid = np.arange(1, delta - 0.1, 1)
                # Slice to be used for interpolation:
                sl = slice(init + 1, last, None)
                # Should be done here, BEFORE the if clause below.

                # Check if lower/upper are all outside the image
                if init < 0 and last >= naxis[i]:
                    invalid = True
                    break
                elif init < 0:  # if only one of lower/upper is outside the image
                    init = last
                elif last >= naxis[i]:
                    last = init
            else:
                init = coord_samelabel[i][0]
                last = coord_samelabel[i][0]
                # coord_samelabel[i] is nothing but an array of same numbers
                sl = slice(init, last + 1, None)

            coord_init.append(init)
            coord_last.append(last)
            coord_slice.append(sl)

        if not invalid:
            val_init = data.item(tuple(coord_init))
            val_last = data.item(tuple(coord_last))
            data[tuple(coord_slice)].flat = (
                val_last - val_init
            ) / delta * grid + val_init

    if update_header:
        nfix = np.count_nonzero(mask)
        _ccd.header["MASKNPIX"] = (nfix, "No. of pixels masked (fixed) by fixpix.")
        _ccd.header["MASKFILE"] = (maskpath, "Applied mask for fixpix.")
        _ccd.header["MASKORD"] = (
            str(priority),
            "Axis priority for fixpix (python order)",
        )
        # MASKFILE: name identical to IRAF
        # add as history
        headers.cmt2hdr(
            _ccd.header,
            "h",
            t_ref=_t_start,
            verbose=verbose,
            s="[fixpix] Pixel values interpolated.",
        )
        headers.update_process(_ccd.header, "P")

    return _ccd


def find_extpix(
    ccd,
    mask=None,
    npixs=(1, 1),
    bezels=None,
    order_xyz=True,
    sort=True,
    update_header=True,
    verbose=0,
):
    """Finds the N extrema pixel values excluding masked pixels.

    Parameters
    ---------
    ccd : `~astropy.nddata.CCDData`
        The ccd to find extreme values

    mask : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, or number-like, optional.
        The mask to be used. To reduce file I/O time, better to provide
        `~numpy.ndarray`.
        Default: `None`.

    npixs : length-2 `tuple` of `int`, optional
        The numbers of extrema to find, in the form of ``[small, large]``, so
        that ``small`` number of smallest and ``large`` number of largest pixel
        values will be found. If `None`, no extrema is found (`None` is
        returned for that extremum).
        Default: ``(1, 1)`` (find minimum and maximum)
        Default: ``(1, 1)``.

    bezels : `list` of `list` of `int`, optional.
        If given, must be a `list` of `list` of `int`. Each `list` of `int` is in the
        form of ``[lower, upper]``, i.e., the first ``lower`` and last
        ``upper`` rows/columns are ignored.
        Default: `None`.

    order_xyz : `bool`, optional.
        Whether `bezel` in xyz order or not (python order:
        ``xyz_order[::-1]``).
        Default: `True`.

    sort: `bool`, optional.
        Whether to sort the extrema in ascending order.
        Default: `True`.

    Returns
    -------
    min
        The `list` of extrema pixel values.
    """
    if not len(npixs) == 2:
        raise ValueError("npixs must be a length-2 tuple of int.")
    _t = Time.now()
    data = ccd.data.copy().astype("float32")  # Not float64 to reduce memory usage
    # slice first to reduce computation time
    if bezels is not None:
        sls = bezel2slice(bezels, order_xyz=order_xyz)
        data = data[sls]
        if mask is not None:
            mask = mask[sls]

    if mask is None:
        maskname = "No mask"
        mask = ~np.isfinite(data)
    else:
        if not isinstance(mask, np.ndarray):
            mask, maskname, _ = _io._parse_image(mask, force_ccddata=True)
            mask = mask.data | ~np.isfinite(data)
        else:
            maskname = "User-provided mask"

    exts = []
    for npix, sign, minmaxval in zip(npixs, [1, -1], [np.inf, -np.inf]):
        if npix is None:
            exts.append(None)
            continue
        data[mask] = minmaxval
        # ^ if getting maximum/minimum pix vals, replace with minimum/maximum
        extvals = np.partition(data.ravel(), sign * npix)
        #         ^^^^^^^^^^^^
        # bn.partitoin has virtually no speed gain.
        extvals = extvals[:npix] if sign > 0 else extvals[-npix:]
        if sort:
            extvals = np.sort(extvals)[::sign]
        exts.append(extvals)

    if update_header:
        for ext, mm in zip(exts, ["min", "max"]):
            if ext is not None:
                for i, extval in enumerate(ext):
                    ccd.header.set(
                        f"{mm.upper()}V{i+1:03d}", extval, f"{mm} pixel value"
                    )
        bezstr = ""
        if bezels is not None:
            order = "xyz order" if order_xyz else "pythonic order"
            bezstr = f" and bezel: {bezels} in {order}"
        headers.cmt2hdr(
            ccd.header,
            "h",
            verbose=verbose,
            t_ref=_t,
            s=(
                "[fitsmgmt.find_extpix] Extrema pixel values found N(smallest, largest) = "
                + f"{npixs} excluding mask ({maskname}){bezstr}. "
                + "See MINViii and MAXViii."
            ),
        )
    return exts


def find_satpix(
    ccd,
    mask=None,
    satlevel=65535,
    bezels=None,
    order_xyz=True,
    update_header=True,
    verbose=0,
):
    """Finds saturated pixel values excluding masked pixels.

    Parameters
    ---------
    ccd : `~astropy.nddata.CCDData`, `~numpy.ndarray`
        The ccd to find extreme values. If `ndarray`, `update_header` will
        automatically be set to `False`.

    mask : `~astropy.nddata.CCDData`-like (e.g., `~astropy.io.fits.PrimaryHDU`, `~astropy.io.fits.ImageHDU`, `~astropy.io.fits.HDUList`), `~numpy.ndarray`, path-like, or number-like, optional.
        The mask to be used. To reduce file I/O time, better to provide
        `~numpy.ndarray`.
        Default: `None`.

    satlevel: numeric, optional.
        The saturation level. Pixels >= `satlevel` will be retarded as
        saturated pixels, except for those masked by `mask`.
        Default: ``65535``.

    bezels : `list` of `list` of `int`, optional.
        If given, must be a `list` of `list` of `int`. Each `list` of `int` is in the
        form of ``[lower, upper]``, i.e., the first ``lower`` and last
        ``upper`` rows/columns are ignored.
        Default: `None`.

    order_xyz : `bool`, optional.
        Whether `bezel` in xyz order or not (python order:
        ``xyz_order[::-1]``).
        Default: `True`.

    Returns
    -------
    min
        The `list` of extrema pixel values.
    """
    _t = Time.now()
    if isinstance(ccd, CCDData):
        data = ccd.data.copy()
    else:
        data = ccd.copy()
        update_header = False
    satmask = np.zeros(data.shape, dtype=bool)
    # slice first to reduce computation time
    if bezels is not None:
        sls = bezel2slice(bezels, order_xyz=order_xyz)
        data = data[sls]
        if mask is not None:
            mask = mask[sls]
    else:
        sls = [slice(None, None, None) for _ in range(data.ndim)]

    if mask is None:
        maskname = "No mask"
        satmask[sls] = data >= satlevel
    else:
        if not isinstance(mask, np.ndarray):
            mask, maskname, _ = _io._parse_image(mask, force_ccddata=True)
            mask = mask.data
        else:
            maskname = "User-provided mask"
        satmask[sls] = (data >= satlevel) & (~mask)  # saturated && not masked

    if update_header:
        nsat = np.count_nonzero(satmask[sls])
        ccd.header["NSATPIX"] = (nsat, "No. of saturated pix")
        ccd.header["SATLEVEL"] = (satlevel, "Saturation: pixels >= this value")
        bezstr = ""
        if bezels is not None:
            order = "xyz order" if order_xyz else "pythonic order"
            bezstr = f" and bezel: {bezels} in {order}"
        headers.cmt2hdr(
            ccd.header,
            "h",
            verbose=verbose,
            t_ref=_t,
            s=(
                "[fitsmgmt.find_satpix] Saturated pixels calculated based on satlevel = "
                + f"{satlevel}, excluding mask ({maskname}){bezstr}. "
                + "See NSATPIX and SATLEVEL."
            ),
        )
    return satmask
