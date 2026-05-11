"""FITS file summary and table-selection helpers."""

from pathlib import Path

import numpy as np
import pandas as pd
from astro_ndslice import listify
from astropy.io import fits
from astropy.io.fits.verify import VerifyError
from astropy.nddata import CCDData

from ..logging import logger
from ._types import HDUExt, StrPathLike
from .io import _parse_extension, inputs2list

__all__ = [
    "fits_summary",
    "df_selector",
]


def _write_summary(
    output: StrPathLike, summarytab: pd.DataFrame, verbose: bool = True
) -> None:
    """Write a summary table, choosing format from the file suffix."""
    output = Path(output)
    if verbose:
        logger.info('Saving the summary to "%s"', output)

    suffix = output.suffix.lower()
    if suffix in {".parq", ".parquet"}:
        summarytab.to_parquet(output, index=False)
    else:
        summarytab.to_csv(output, index=False)


def fits_summary(
    inputs=None,
    extension: HDUExt = None,
    verify_fix: bool = False,
    fname_option: str = "relative",
    output: StrPathLike | None = None,
    keywords: list[str] | str | None = None,
    example_header: StrPathLike | None = None,
    sort_by: str = "file",
    sort_map: dict | None = None,
    fullmatch: dict | None = None,
    flags: int = 0,
    querystr: str | None = None,
    negate_fullmatch: bool = False,
    nonunique_keys: bool = False,
    verbose: bool = True,
    **kwargs,
) -> pd.DataFrame | None:
    """Extract summary rows from FITS headers.

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
        Output summary file. ``.parq`` and ``.parquet`` use parquet; other
        suffixes use CSV.
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
        ``summarytab[column].str.fullmatch(regex, case=True)``.
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
        *ALL* input objects. Even if they are unique, keys specified in `keywords`
        will not be removed.
        Default is `False`.

    verbose : `bool`, optional
        Whether to print the progress. Default is `True`.

    **kwargs :
        The keyword arguments to be passed to `~astropy.io.fits.open`.

    Returns
    -------
    summarytab : `~pandas.DataFrame`
        Summary table with one row per input FITS file.

    Notes
    -----
    I want to use ccdproc.ImageFileCollection instead of this, but it is about
    4 times slower than my `~astroimred.summary.fits_summary`, so I cannot use it yet.

    Examples
    -------

    >>> from pathlib import Path
    >>> import astroimred as air
    >>> keys = ["OBS-TIME", "FILTER", "OBJECT"]
    >>> # actually it is case-insensitive
    >>> # The keywords you want to extract
    >>> # (from the headers of FITS files)
    >>> TOPPATH = Path(".", "observation_2018-01-01")
    >>> # The toppath
    >>> savepath = TOPPATH / "summary_20180101.csv"
    >>> # list of all the fits files in TOPPATH/rawdata:
    >>> summary = air.fits_summary(
    >>>     TOPPATH/"rawdata/*.fits",
    >>>     keywords=keys,
    >>>     fname_option='name',
    >>>     sort_by="DATE-OBS",
    >>>     output=savepath
    >>> )

    Select all rows with ``OBJECT`` starts with "DA":

    >>> # fullmatch = {"OBJECT": "DA.*"}
    Select all rows with ``OBJECT`` starts with "Ves", ``FILTER`` is "J", and
    ``EXPTIME`` is 2 or 3:

    >>> # fullmatch = {"OBJECT": "Ves.*", "FILTER": "J"},
    >>> # querystr="EXPTIME in [2, 3]"
    """
    if inputs is None:
        return None

    if nonunique_keys:
        summ = fits_summary(
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
        for key in list(summ.columns):
            if keywords is not None and key in keywords:
                continue
            if len(_uniq := summ[key].unique()) == 1:
                if verbose:
                    logger.info(" * %-8s: %s", key, _uniq[0])
                summ.pop(key)
        if output is not None:
            _write_summary(output, summ, verbose=verbose)
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
            except VerifyError as err:
                raise VerifyError("Use verify_fix=True.") from err
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
                "All %d keywords (guessed from %s) will be loaded.",
                len(keywords),
                fname0,
            )

    # Initialize
    summarytab = {"file": [], "filesize": []}
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
        summarytab[k] = (
            summarytab[k].astype(object).where(pd.notna(summarytab[k]), None)
        )

    if output is not None:
        _write_summary(output, summarytab, verbose=verbose)

    return summarytab


def df_selector(
    summarytab: pd.DataFrame,
    fullmatch: dict | None = None,
    flags: int = 0,
    negate_fullmatch: bool = False,
    querystr: str | None = None,
    columns: str | list[str] | None = None,
    columns_drop: str | list[str] | None = None,
    reset_index: bool = True,
) -> pd.DataFrame:
    """Select rows from a summary table.

    Parameters
    ----------
    summarytab : `~pandas.DataFrame`
        The summary table to select from. Normally the table made from header
        information.
    fullmatch : `dict`, optional
        The ``{column: regex}`` style `dict` to be used for selecting rows by
        ``summarytab[column].str.fullmatch(regex, case=True)``. An example:
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

    reset_index : `bool`, optional.
        Whether to reset the DataFrame index after selection.
        Default: `True`.

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
                except (ValueError, TypeError, AttributeError) as err:
                    raise TypeError(
                        "Both ``summarytab[k].str.fullmatch(v)`` and "
                        + f"``summarytab[{k}] == {v}`` failed.\n"
                        + "Maybe use `querystr` instead?"
                    ) from err
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
