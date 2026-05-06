"""
Contains convenience functions which are
(1) more related to the file name or paths rather than the contents or
(2) related to the non-FITS files.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from astro_ndslice import listify, slice_from_string
from astropy.io import fits
from astropy.io.fits.verify import VerifyError
from astropy.nddata import CCDData

from .ccdutils import cut_ccd
from .headers import key_mapper, key_remover
from .io import _parse_extension, inputs2list
from .logging import logger

__all__ = [
    "mkdir",
    "make_summary",
    "df_selector",
    "fits_newpath",
    "fitsrenamer",
]


def mkdir(fpath, mode=0o777, exist_ok=True):
    """Convenience function for `~pathlib.Path`.mkdir()"""
    fpath = Path(fpath)
    Path.mkdir(fpath, mode=mode, exist_ok=exist_ok)


def make_summary(
    inputs=None,
    extension=None,
    verify_fix=False,
    fname_option="relative",
    output=None,
    keywords=None,
    example_header=None,
    sort_by="file",
    sort_map=None,
    fullmatch=None,
    flags=0,
    querystr=None,
    negate_fullmatch=False,
    nonunique_keys=False,
    verbose=True,
    **kwargs,
):
    """Extracts summary from the headers of FITS files.

    Parameters
    ----------
    inputs : glob pattern, `list`-like of path-like, `list`-like of `~astropy.nddata.CCDData`, `~pandas.DataFrame` convertible, optional.
        The `~glob` pattern for files (e.g., ``"2020*[012].fits"``) or `list` of
        files (each element must be path-like or `~astropy.nddata.CCDData`). Although it is not a
        good idea, a mixed `list` of `~astropy.nddata.CCDData` and paths to the files is also
        acceptable. If a `~pandas.DataFrame` or convertible (especially
        `~astropy.table.Table`) is given, it finds the ``"file"`` column and
        use it as the input files, make a summary table from the headers of
        those files.
        If `inputs` is `None`, any `output` is ignored and `None` is returned.
        Default: `None`.

    extension : `int`, `str`, (`str`, `int`), optional.
        The extension of FITS to be used. It can be given as integer
        (0-indexing) of the extension, ``EXTNAME`` (single `str`), or a `tuple` of
        `str` and `int`: ``(EXTNAME, EXTVER)``. If `None` (default), the *first
        extension with data* will be used.
        Default: `None`.

    verify_fix : `bool`, optional.
        Whether to do ``.verify('fix')`` to all FITS files to avoid
        VerifyError. It may take some time if turned on. Default is `False`.

    fname_option : `str` ``{'absolute', 'relative', 'name'}``, optional
        Whether to save full absolute/relative path or only the filename.
        Default: ``'relative'``.

    output : `str` or path-like, optional
        The directory and file name of the output summary file.
        Default: `None`.

    keywords : `list` or `str`(``"*"``), optional
        The `list` of the keywords to extract (keywords should be in `str`).
        Default: `None`.

    example_header : `None` or path-like, optional
        The path including the filename of the output summary text file. If
        specified, the header of the 0-th element of `inputs` will be extracted
        (if glob-pattern is given, the 0-th element is random, so be careful)
        and saved to `example_header`. Use `None` (default) to skip this.
        Default: `None`.

    sort_by : `str`, optional
        The column name to sort the results. It can be any element of
        `keywords` or `'file'`, which sorts the table by the file name.
        Default: ``'file'``.

    sort_map: `dict`, optional
        A subset of `key` parameter in `pandas.DataFrame.sort_values()`. If a
        `dict` is given, then ``key = lambda x: x.map(sort_map)`` is passed into
        `.sort_values()`.
        Default: `None`.

    fullmatch : `dict`, optional
        The ``{column: regex}`` style `dict` to be used for selecting rows by
        ``summarytab[column].`str`.fullmatch(regex, case=`True`)``.
        Default: `None`

    negate_fullmatch: `bool`, optional.
        Whether to negate the mask by `fullmatch`, in case the user does not
        want to think much about regex to negate it.
        Default: `False`.

    flags: `int`, optional.
        Regex module flags, e.g. re.IGNORECASE. Default: 0

    querystr : `str`, optional
        The query string used for ``summarytab.query(querystr)``. See
        `~pandas.DataFrame.query`.
        Default: `None`.

    nonunique_keys : `bool`, optional
        Whether to remove the keys that have only one unique value throughout
        *ALL* input objects. Even if it is unique, keys specified in `keywords`
        will not be removed.
        Default is `False`.

    verbose : `bool`, optional
        Whether to print the progress. Default is `True`.

    **kwargs :
        The keyword arguments to be passed to `~astropy.io.fits.open`.

    Returns
    -------
    summarytab: astropy.Table

    Notes
    -----
    I want to use ccdproc.ImageFileCollection instead of this, but it is about
    4 times slower than my `~fitsmgmt.filemgmt.make_summary`, so I cannot use it yet.

    Examples
    -------

    >>> from pathlib import Path
    >>> import fitsmgmt as fm
    >>> keys = ["OBS-TIME", "FILTER", "OBJECT"]
    >>> # actually it is case-insensitive
    >>> # The keywords you want to extract
    >>> # (from the headers of FITS files)
    >>> TOPPATH = Path(".", "observation_2018-01-01")
    >>> # The toppath
    >>> savepath = TOPPATH / "summary_20180101.csv"
    >>> # list of all the fits files in TOPPATH/rawdata:
    >>> summary = fm.make_summary(
    >>>     TOPPATH/"rawdata/*.fits",
    >>>     keywords=keys,
    >>>     fname_option='name',
    >>>     pandas=True,
    >>>     sort_by="DATE-OBS",
    >>>     output=savepath
    >>> )

    Select all rows with ``OBJECT`` starts with "DA":

    >>> # fullmatch = {"OBJECT": "DA.*"}
    Select all rows with ``OBJECT`` starts with "Ves", ``FILTER`` is "J", and
    ``EXPTIME`` is 2 or 3:

    >>> # fullmatch = {"OBJECT": "Ves.*", "FILTER": "J"},
    >>> # querystr="EXPTIME in [2, 3]
    """
    if inputs is None:
        return None

    if nonunique_keys:
        summ = make_summary(
            inputs=inputs,
            extension=extension,
            verify_fix=verify_fix,
            fname_option=fname_option,
            output=None,
            keywords=keywords,
            example_header=example_header,
            sort_by=sort_by,
            sort_map=sort_map,
            fullmatch=fullmatch,
            flags=flags,
            querystr=querystr,
            negate_fullmatch=negate_fullmatch,
            nonunique_keys=False,
            verbose=verbose,
            **kwargs,
        )
        if verbose:
            logger.info("Unique keys that will be removed:")
        for key in summ.keys():
            if keywords is not None and key in keywords:
                continue
            if len(_uniq := summ[key].unique()) == 1:
                if verbose:
                    logger.info(" * %-8s: %s", key, _uniq[0])
                summ.pop(key)
        if output is not None:
            output = Path(output)
            if verbose:
                logger.info('Saving the summary to "%s"', output)
            summ.to_csv(output, index=False)
        return summ

    # Although there's no need to sort here because the real "sort" will be
    # done later based on ``sort_by`` column, I did it here because the full
    # header keys will be inferred from the 0-th element (if `keywords` is not
    # given)
    fitslist = inputs2list(
        inputs, sort=True, accept_ccdlike=True, check_coherency=False
    )

    if len(fitslist) == 0:
        if verbose:
            logger.info("No FITS file found.")
        return None

    def _get_fname_fsize_hdr(item, idx, extension):
        if isinstance(item, CCDData):
            # NOTE: CCDData does not support extension (only available when it
            #   is being read)!
            fname = f"CCDData in fitslist[{idx:d}]"
            fsize = None
            hdr = item.header
        else:
            if fname_option == "relative":
                fname = str(item)
            elif fname_option == "absolute":
                fname = str(item.absolute())
            elif fname_option == "name":
                fname = item.name
            else:
                raise ValueError(f"fname_option `{fname_option}`not understood.")
            fsize = Path(item).stat().st_size
            # Don't change to MB/GB, which will make it float...
            with fits.open(item, **kwargs) as hdul:
                if verify_fix:
                    hdul.verify("fix")
                hdr = hdul[extension].header.copy()

        return fname, fsize, hdr

    skip_keys = ["COMMENT", "HISTORY"]

    if verbose and keywords is not None:
        if keywords == "*":
            logger.info("Extracting all keywords...")
        else:
            logger.info("Extracting keys: %s", keywords)

    extension = _parse_extension(extension)

    first_info = None
    if example_header is not None or keywords is None or keywords == "*":
        first_info = _get_fname_fsize_hdr(fitslist[0], 0, extension=extension)

    # Save example header
    if example_header is not None:
        fname0, _, hdr0 = first_info
        if verbose:
            logger.info("Header of 0-th: %s -> %s", fname0, example_header)
        hdr0.totextfile(example_header, overwrite=True)

    # load ALL keywords for special cases
    if (keywords is None) or (keywords is not None and keywords == "*"):
        fname0, _, hdr0 = first_info
        num_hkeys = len(hdr0.cards)
        keywords = []

        for i in range(num_hkeys):
            try:
                key_i = hdr0.cards[i][0]
            except VerifyError:
                raise VerifyError("Use verify_fix=True.")
            if key_i in skip_keys:
                continue
            elif key_i in keywords:
                logger.warning(
                    "Key %s is duplicated! Only the first one will be saved.",
                    key_i,
                )
                continue
            keywords.append(key_i)

        if verbose:
            logger.info(
                "All %d keywords (guessed from %s) will be loaded.", len(keywords), fname0
            )

    # Initialize
    summarytab = dict(file=[], filesize=[])
    missing_keys = set()
    for k in keywords:
        summarytab[k] = []

    # Run through all the fits files
    for i, item in enumerate(fitslist):
        if i == 0 and first_info is not None:
            fname, fsize, hdr = first_info
        else:
            fname, fsize, hdr = _get_fname_fsize_hdr(item, i, extension=extension)
        summarytab["file"].append(fname)
        summarytab["filesize"].append(fsize)
        for k in keywords:
            try:
                summarytab[k].append(hdr[k])
            except KeyError:
                if verbose:
                    str_keyerror_fill = (
                        "Key {:s} not found for {:s}, filling with None."
                    )
                    if isinstance(item, CCDData):
                        logger.warning(str_keyerror_fill.format(k, f"fitslist[{i}]"))
                    else:
                        logger.warning(str_keyerror_fill.format(k, str(item)))
                summarytab[k].append(None)
                missing_keys.add(k)

    summarytab = pd.DataFrame.from_dict(summarytab)
    summarytab = df_selector(
        summarytab,
        fullmatch=fullmatch,
        flags=flags,
        querystr=querystr,
        negate_fullmatch=negate_fullmatch,
    )
    if sort_by is not None:
        key = None if sort_map is None else lambda x: x.map(sort_map)
        summarytab.sort_values(sort_by, inplace=True, key=key)
    summarytab.reset_index(drop=True, inplace=True)
    for k in missing_keys:
        summarytab[k] = summarytab[k].astype(object).where(pd.notna(summarytab[k]), None)

    if output is not None:
        output = Path(output)
        if verbose:
            logger.info('Saving the summary to "%s"', output)
        summarytab.to_csv(output, index=False)

    return summarytab


def df_selector(
    summarytab,
    fullmatch=None,
    flags=0,
    negate_fullmatch=False,
    querystr=None,
    columns=None,
    columns_drop=None,
    reset_index=True,
):
    """Select rows from a summary table.

    Parameters
    ----------
    summarytab : `~pandas.DataFrame`
        The summary table to select from. Normally the table made from header
        information.
    fullmatch : `dict`, optional
        The ``{column: regex}`` style `dict` to be used for selecting rows by
        ``summarytab[column].`str`.fullmatch(regex, case=`True`)``. An example:
        ``{"OBJECT": "Ves.*"}``. All corresponding columns must have dtype of
        `str` to apply regex.
        Default: `None`
    negate_fullmatch: `bool`, optional.
        Whether to negate the mask by `fullmatch`, in case the user does not
        want to think much about regex to negate it.
        Default: `False`.
    flags: `int`, optional.
        Regex module flags, e.g. re.IGNORECASE. Default: 0
    querystr : `str`, optional
        The query string used for ``summarytab.query(querystr)``. See
        `~pandas.DataFrame.query`.
    columns, columns_drop: `str`, `list`, optional.
        The `list` of columns to be returned/dropped after selection. No need to
        setup both, but no Error will be raised even the user does so.
        Default: `None`.

    Returns
    -------
    summarytab
        The final summary table after selection. If everything is `None` (the
        default), the original summary table is returned.

    Raises
    ------
    AttributeError
        The column dtype is not `str`
    TypeError
        fullmatch must be in `dict`.

    Examples
    --------
    Select all rows with ``OBJECT`` starts with "DA":

    >>> # fullmatch = {"OBJECT": "DA.*"}
    Select all rows with ``OBJECT`` starts with "Ves", ``FILTER`` is "J", and
    ``EXPTIME`` is 2 or 3:

    >>> # fullmatch = {"OBJECT": "Ves.*", "FILTER": "J"},
    >>> # querystr="EXPTIME in [2, 3]"

    """
    df = summarytab.copy()

    if fullmatch is not None:
        if not isinstance(fullmatch, dict):
            raise TypeError("fullmatch must be a dict.")

        select_mask = np.ones(len(df), dtype=bool)
        for k, v in fullmatch.items():
            try:
                select_mask &= df[k].str.fullmatch(v, flags=flags, case=True)
            except AttributeError:
                try:
                    select_mask &= df[k] == v
                except (ValueError, TypeError, AttributeError):
                    raise TypeError(
                        "Both ``summarytab[k].str.fullmatch(v)`` and "
                        + f"``summarytab[{k}] == {v}`` failed.\n"
                        + "Maybe use `querystr` instead?"
                    )
        df = df[~select_mask] if negate_fullmatch else df[select_mask]

    if querystr is not None:
        df = df.query(querystr)

    if columns is not None:
        df = df[listify(columns)]

    if columns_drop is not None:
        df.drop(listify(columns_drop), axis=1, inplace=True)

    if reset_index:
        df = df.reset_index(drop=True)

    return df.copy()


# def df_matcher(
#     df1,
#     df2,
#     match_by=None
# ):
#     """

#     Parameters
#     ----------
#     df1 : `~pandas.DataFrame`
#         The first table for the rows to be picked out.
#     df2 : `~pandas.DataFrame`
#         The table for the rows to be matched based on `match_by`.
#     match_by : [type], optional
#         [description], by default None
#     """
#     if match_by is None:
#         if not all(df2.columns.isin(df1.columns)):
#             raise IndexError(
#                 f"Some column of `df2` not found in `df1`. "
#                 + "You may specify `match_by` to specify the columns to match."
#             )
#         match_by = list(df2.columns)
#     match_by = np.atleast_1d(match_by)

#     for idx, row in df1.iterrows():
#         try:
#             df2.loc[df2[match_by].eq(row[match_by]).all(axis=1), :]
#         for col, cal, mat, calcol in zip(cols, cals, mats, calcols):
#             mat = np.atleast_1d(mat)
#             # If just a str, cal[mat]==row[mat] gives `Series`, not `DataFrame`,
#             # hence `axis=1` raises error.
#             try:
#                 sel = cal[(cal[mat] == row[mat]).all(axis=1)]
#                 if len(sel) == 1:
#                     df.loc[idx, col] = sel[calcol].values[0]
#                 elif len(sel) > 1:
#                     raise ValueError(
#                         f"More than one calibration frame found for {mat} = {row[mat].values}"
#                     )
#                 else:
#                     continue
#             except (IndexError):  # no match
#                 continue


def fits_newpath(
    fpath,
    rename_by,
    mkdir_by=None,
    header=None,
    delimiter="_",
    fillnan="",
    fileext=".fits",
):
    """Gives the new path of the FITS file from header.

    Parameters
    ----------
    fpath : path-like
        The path to the original FITS file.

    rename_by : `list` of `str`, optional
        The keywords of the FITS header to rename by.

    mkdir_by : `list` of `str`, optional
        The keys which will be used to make subdirectories to classify files.
        If given, subdirectories will be made with the header value of the
        keys.
        Default: `None`.

    header : `~astropy.io.fits.Header` object, optional
        The header to extract `rename_by` and `mkdir_by`. If `None`, the
        function will do ``header = fits.getheader(fpath)``.
        Default: `None`.

    delimiter : `str`, optional
        The delimiter for the renaming.
        Default: ``'_'``.

    fillnan : `str`, optional
        The string that will be inserted if the keyword is not found from the
        header.
        Default: ``''``.

    fileext : `str`, optional
        The extension of the file name to be returned. Normally it should be
        ``'.fits'`` since this function is `fits_newname`, but you may prefer,
        e.g., ``'.fit'`` for some reason. If `fileext` does not start with a
        period (``"."``), it is automatically added to the final file name in
        front of the ``fileext``.
        Default: ``'.fits'``.

    Returns
    -------
    newpath : path
        The new path.
    """

    if header is None:
        hdr = fits.getheader(fpath)
    else:
        hdr = header.copy()

    if not fileext.startswith("."):
        fileext = f".{fileext}"

    newname = delimiter.join([str(hdr.get(k, fillnan)) for k in rename_by]) + fileext
    newpath = Path(fpath.parent)

    if mkdir_by is not None:
        for k in mkdir_by:
            newpath = newpath / hdr[k]

    newpath = newpath / newname

    return newpath


def fitsrenamer(
    fpath=None,
    header=None,
    newtop=None,
    rename_by=["OBJECT"],
    mkdir_by=None,
    delimiter="_",
    archive_dir=None,
    keymap=None,
    key_deprecation=True,
    remove_keys=None,
    overwrite=False,
    fillnan="",
    trimsec=None,
    verbose=True,
    add_header=None,
):
    """Renames a FITS file by ``rename_by`` with delimiter.

    Parameters
    ----------
    fpath : path-like, optional.
        The path to the target FITS file.
        Default: `None`.

    header : `~astropy.io.fits.Header`, optional
        The header of the fits file, especially if you want to just overwrite
        the header with this.
        Default: `None`.

    newtop : path-like, optional.
        The top path for the new FITS file. If `None`, the new path will share
        the parent path with `fpath`.
        Default: `None`.

    rename_by : `list` of `str`, optional
        The keywords of the FITS header to rename by.
        Default: ``['OBJECT']``.

    mkdir_by : `list` of `str`, optional
        The keys which will be used to make subdirectories to classify files.
        If given, subdirectories will be made with the header value of the
        keys.
        Default: `None`.

    delimiter : `str`, optional
        The delimiter for the renaming.
        Default: ``'_'``.

    archive_dir : path-like or `None`, optional
        Where to move the original FITS file. If `None`, the original file will
        remain there. Deleting original FITS is dangerous so it is only
        supported to move the files. You may delete files manually if needed.
        Default: `None`.

    keymap : `dict` or `None`, optional
        If not `None`, the keymapping is done by using the `dict` of `keymap` in
        the format of ``{<standard_key>:<original_key>}``.
        Default: `None`.

    key_deprecation : `bool`, optional
        Whether to change the original keywords' comments to contain
        deprecation warning. If `True`, the original keywords' comments will
        become ``Deprecated. See <standard_key>.``.
        Default: `True`.

    trimsec : `str` or `None`, optional
        Region of ``~astropy.nddata.CCDData`` from which the overscan is extracted; see
        `~ccdproc.subtract_overscan` for details. Default is `None`.

    fillnan : `str`, optional
        The string that will be inserted if the keyword is not found from the
        header.
        Default: ``''``.

    remove_keys : `list` of `str`, optional.
        The header keywords to be removed.
        Default: `None`.

    add_header : header or Card object, optional.
        The header keyword, value (and comment) to add after the renaming.
        Default: `None`.

    Notes
    -----
    MEF(Multi-Extension FITS) currently is not supported.

    """

    # Load fits file
    hdul = fits.open(fpath)
    data = hdul[0].data
    if header is None:
        hdr = hdul[0].header
    else:
        hdr = header.copy()
    hdul.close()

    # add keyword
    if add_header is not None:
        if not isinstance(add_header, fits.Header) and not isinstance(
            add_header, fits.header.Card
        ):
            logger.warning(
                "add_header is not either Header or Card. "
                "Be careful about possible error."
            )
        hdr += add_header

    # Copy keys based on KEYMAP
    if keymap is not None:
        hdr = key_mapper(hdr, keymap, deprecation=key_deprecation)

    if remove_keys is not None:
        hdr = key_remover(hdr, remove_keys, deepremove=True)

    # TODO: It is necessary to do this bothersome calculations to
    #   preserve the WCS information that may reside in the FITS (if use
    #   ``trim_image`` of ccdproc, it will not be preserved).
    # TODO: Maybe I can put some LTV-like keys to the header, rather
    #   than this crazy code...? (ysBach 2019-05-09)
    if trimsec is not None:
        slices = slice_from_string(trimsec, fits_convention=True)
        # initially guess start and stop indices as 0's and from shape in (ny, nx) order
        ny, nx = data[slices].shape
        starts = np.array([0, 0])  # yx order
        stops = np.array([ny, nx])  # yx order

        for i in range(2):
            if slices[i].start is not None:
                starts[i] = slices[i].start
            if slices[i].stop is not None:
                stops[i] = slices[i].stop

        cent = np.flip((stops - starts) / 2)  # xy order
        size = (ny, nx)  # yx order
        # Make CCDData instance as dummy object
        _ccd = CCDData(data, header=hdr, unit="adu")
        _ccd = cut_ccd(_ccd, cent, size)
        data = _ccd.data
        hdr = _ccd.header

    newhdul = fits.PrimaryHDU(data=data, header=hdr)

    # Set the new path
    if verbose:
        form = ""
        for rn in rename_by:
            form = form + f"<{rn:s}>{delimiter:s}"
        ndelimiter = len(delimiter)
        logger.info("Renaming file by %s", form[:-ndelimiter])
        if mkdir_by is not None:
            form = ""
            for md in mkdir_by:
                form = form + f"<{md:s}>/"
            logger.info("Make by %s", form[:-1])

    newpath = fits_newpath(
        fpath,
        rename_by,
        mkdir_by=mkdir_by,
        header=hdr,
        delimiter=delimiter,
        fillnan=fillnan,
        fileext="fits",
    )
    if newtop is not None:
        newpath = Path(newtop) / newpath.name

    mkdir(newpath.parent)

    if verbose:
        logger.info("Rename %s to %s", fpath.name, newpath)

    newhdul.writeto(newpath, output_verify="fix", overwrite=overwrite)

    if archive_dir is not None:
        archive_dir = Path(archive_dir)
        archive_path = archive_dir / fpath.name
        mkdir(archive_path.parent)
        if verbose:
            logger.info("Moving %s to %s", fpath.name, archive_path)
        fpath.rename(archive_path)

    return newpath
