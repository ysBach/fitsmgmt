import numpy as np
from astropy.nddata import CCDData

from astroimred.mgmt.logging import logger
from astroimred.imops.ccdutils import CCDData_astype, imslice
from astroimred.mgmt.headers import update_tlm
from astroimred.mgmt.io import _parse_image, inputs2list

__all__ = ["imcopy"]


# TODO: use fitsio if (outputs is None) and not return_ccd
def imcopy(
    inputs,
    trimsecs=None,
    outputs=None,
    extension=None,
    return_ccd=True,
    dtype=None,
    update_header=True,
    **kwargs,
):
    """Copy FITS images or sections, similar to IRAF IMCOPY.

    Parameters
    ----------
    inputs : glob pattern, path-like, CCD-like, or list-like
        Input files or CCD-like objects.

    trimsecs : str or list-like of str, optional
        FITS sections to extract. Bracket embraced, comma separated, XY order,
        1-indexing, and including the end index. If given as array-like format
        of length ``N``, all such sections in all FITS files will be extracted.

    outputs : path-like or list-like, optional
        Output paths. If list-like, shape must match ``(n_inputs, n_sections)``.

    extension : `int`, `str`, (`str`, `int`), optional.
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.
        Default: `None`.

    return_ccd, update_header : bool, optional
        Whether to return CCDData objects and update ``NAXIS*`` metadata.

    dtype : dtype-like, optional
        Output/return data dtype. If `None`, preserve input dtype.

    **kwargs
        Keyword arguments passed to ``CCDData.write``.

    Returns
    -------
    `~astropy.nddata.CCDData` or list of `~astropy.nddata.CCDData`
        Returned only when ``return_ccd=True``.

    Notes
    -----
    To make imcopy faster, use  update_header=`False` (2.8 ms -> 2.3 ms) and
    dtype=`None`.

    All the sections will be flattened if they are higher than 1-d. I think it
    will only increase the complexity of the code if I accept that...?

    Examples
    -------

    >>> import astroimred.reduction as imred
    >>> from pathlib import Path
    >>>
    >>> datapath = Path("./data")
    >>> files = datapath.glob("*.pcr.fits")
    >>> sections = ["[50:100, 50:100]", "[50:100, 50:150]"]
    >>> outputs = [datapath/"test1.fits", datapath/"test2.fits"]
    >>>
    >>> # single file, single section
    >>> trim = imred.imcopy(files[0], sections[0])
    >>>
    >>> # single file, multi sections
    >>> trims = imred.imcopy(files[0], sections)
    >>>
    >>> # Save with overwrite option
    >>> imred.imcopy(files[0], sections, outputs=outputs, overwrite=True)
    >>>
    >>> # multi file multi section
    >>> trims2d = imred.imcopy(files[:2], trimsecs=sections, outputs=None)
    """
    to_trim = False
    to_save = False

    inputs = inputs2list(inputs, sort=True, accept_ccdlike=True, check_coherency=False)

    m = len(inputs)

    if trimsecs is not None:
        sects = np.atleast_1d(trimsecs)
        to_trim = True
        if sects.ndim > 1:
            logger.info("`trimsecs` with > 1D are flattened. Now %d-D.", sects.ndim)
            sects = sects.ravel()
        n = sects.shape[0]
    else:
        sects = None
        n = 1

    if outputs is not None:
        outputs = np.atleast_2d(outputs)
        to_save = True
        if outputs.ndim > 2:
            raise ValueError("outputs should be lower than 3-d.")
        if outputs.shape != (m, n):
            raise ValueError(
                "If outputs is array-like, it's shape must have the shape of (fpaths.size, "
                + "trimsecs.size)= ({}, {}). Now it's ({}).".format(
                    m, n, *outputs.shape
                )
            )

    if return_ccd:
        results = []

    # TODO: Use fits.open rather than CCDData for speed issue.
    for i, item in enumerate(inputs):
        if isinstance(item, CCDData):
            ccd = item
        else:
            ccd = _parse_image(item, extension=extension, force_ccddata=True)[0]
        result = []
        if to_trim:  # n CCDData will be in `result`
            for sect in sects:
                nccd = imslice(ccd, trimsec=sect)
                if dtype is not None:
                    nccd = CCDData_astype(nccd, dtype=dtype)
                if update_header:
                    update_tlm(nccd.header)
                result.append(nccd)
        else:  # only one single CCDData will be in `result`
            nccd = ccd if dtype is None else CCDData_astype(ccd, dtype=dtype)
            if update_header:
                update_tlm(nccd.header)
            result.append(nccd)

        if to_save:
            for j, res in enumerate(result):
                res.write(outputs[i, j], **kwargs)

        if return_ccd:
            if len(result) == 1:
                results.append(result[0])
            else:
                results.append(result)

    if return_ccd:
        if len(results) == 1:
            return results[0]
        else:
            return results
