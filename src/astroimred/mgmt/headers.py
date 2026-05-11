"""FITS header editing and accessor helpers."""

import contextlib

from astro_ndslice import is_list_like, listify
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.time import Time

from .._types import CCDLike, StrPathLike
from ..logging import logger
from . import io as _io
from .misc import change_to_quantity, str_now

__all__ = [
    "cmt2hdr",
    "update_tlm",
    "update_process",
    "hedit",
    "key_remover",
    "key_mapper",
    "chk_keyval",
    "hdrval",
    "midtime_obs",
]


def cmt2hdr(
    header: fits.Header,
    histcomm: str,
    s: str | list[str],
    precision: int = 3,
    time_fmt: str | None = "{:.>72s}",
    t_ref: Time | None = None,
    dt_fmt: str = "(dt = {:.3f} s)",
    set_kw: dict | None = None,
    verbose: bool = False,
) -> None:
    """Add HISTORY/COMMENT entries, optionally with a timestamp.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        The header.

    histcomm : `str` in ['h', 'hist', 'history', 'c', 'comm', 'comment']
        Whether to add history or comment.

    s : `str` or `list` of `str`
        The string to add as history or comment.

    precision : `int`, optional.
        The precision of the isot format time.
        Default: ``3``.

    time_fmt : `str`, `None`, optional.
        The Python 3 format string to format the time in the header. If `None`,
        the timestamp string will not be added.

        Examples::
          * ``"{:s}"``: plain time ``2020-01-01T01:01:01.23``
          * ``"({:s})"``: plain time in ``()``. ``(2020-01-01T01:01:01.23)``
          * ``"{:_^72s}"``: center align, filling with _.
        Default: ``'{:.>72s}'``.

    t_ref : `~astropy.time.Time`, optional.
        The reference time. If not `None`, delta time is calculated.
        Default: `None`.

    dt_fmt : `str`, optional.
        The Python 3 format string to format the delta time in the header.
        Default: ``'(dt = {:.3f} s)'``.

    verbose : `bool`, optional.
        Whether to log the same information.
        Default: `False`.

    set_kw : `dict`, optional.
        The keyword arguments added to `~astropy.io.fits.Header.set()`. Default is
        ``{'after': -1}``, i.e., the history or comment will be appended to the
        very last part of the header.

    Notes
    -----
    The timing benchmark for a reasonably long header (len(ccd.header.cards) =
    197) shows dt ~ 0.2-0.3 ms on MBP 15" [2018, macOS 11.6, i7-8850H (2.6 GHz;
    6-core), RAM 16 GB (2400MHz DDR4), Radeon Pro 560X (4GB)]:

    %timeit ccd.header.copy()
    1.67 ms +/- 33.3 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)
    %timeit air.cmt2hdr(ccd.header.copy(), 'h', 'test')
    1.89 ms +/- 141 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)
    %timeit air.cmt2hdr(ccd.header.copy(), 'hist', 'test')
    1.89 ms +/- 144 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)
    %timeit air.cmt2hdr(ccd.header.copy(), 'histORy', 'test')
    1.95 ms +/- 146 µs per loop (mean +/- std. dev. of 7 runs, 100 loops each)
    """
    if set_kw is None:
        set_kw = {"after": -1}

    # Normalize histcomm to canonical form
    _hc = histcomm.lower()
    if _hc in ["h", "hist", "history"]:
        histcomm = "HISTORY"
    elif _hc in ["c", "com", "comm", "comment"]:
        histcomm = "COMMENT"
    else:
        raise ValueError(
            f"histcomm must be one of 'h', 'hist', 'history', 'c', 'com', 'comm', 'comment'; "
            f"got {histcomm!r}"
        )

    def _add_content(header, content):
        try:
            header.set(histcomm, content, **set_kw)
        except AttributeError:
            # For a CCDData that has just initialized, header is in OrderedDict, not Header
            header[histcomm] = content

    for _s in listify(s):
        _add_content(header, _s)
        if verbose:
            logger.info("%-8s %s", histcomm, _s)

    if time_fmt is not None:
        timestr = str_now(precision=precision, fmt=time_fmt, t_ref=t_ref, dt_fmt=dt_fmt)
        _add_content(header, timestr)
        if verbose:
            logger.info("%-8s %s", histcomm, timestr)
    update_tlm(header)


def update_tlm(header: fits.Header) -> None:
    """Adds the IRAF-like ``FITS-TLM`` right after ``NAXISi``.

     Timing on MBP 15" [2018, macOS 11.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB
    (2400MHz DDR4), Radeon Pro 560X (4GB)]:
    %timeit air.update_tlm(ccd.header)
    # 443 µs +/- 19.5 µs per loop (mean +/- std. dev. of 7 runs, 1000 loops each)
    """
    now = Time(Time.now(), precision=0).isot
    with contextlib.suppress(KeyError):
        del header["FITS-TLM"]
    try:
        header.set(
            "FITS-TLM",
            value=now,
            comment="UT of last modification of this FITS file",
            after=1,
        )
    except AttributeError:  # If header is OrderedDict
        header["FITS-TLM"] = (now, "UT of last modification of this FITS file")


def update_process(
    header: fits.Header,
    process: str | list[str] | None = None,
    key: str = "PROCESS",
    delimiter: str = "",
    add_comment: bool = True,
    additional_comment: dict | None = None,
) -> None:
    """Update the process history keyword in the header.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        The header to update the ``PROCESS`` (tunable by `key` parameter)
        keyword.

    process : `str` or `list`-like of `str`, optional.
        The additional process keys to add to the header.
        Default: `None`.

    key : `str`, optional.
        The key for the process-related header keyword.
        Default: ``'PROCESS'``.

    delimiter : `str`, optional.
        The delimiter for each process. It can be null string (``''``). The
        best is to match it with the pre-existing delimiter of the
        ``header[key]``.
        Default: ``''``.

    add_comment : `bool`, optional.
        Whether to add a comment to the header if there was no `key`
        (``"PROCESS"`` by default) in the header.
        Default: `True`.

    additional_comment : `dict`, optional.
        The additional comment to add. For instance, ``dict(v="vertical
        pattern", f="fourier pattern")`` will add a new line of comment which
        reads "User added items for `key`: v=vertical pattern, f=fourier
        pattern."
        Default: `None`.
    """
    if additional_comment is None:
        additional_comment = {}
    process = listify(process)

    if key in header:
        if delimiter:
            process = header[key].split(delimiter) + process
        else:
            process = list(header[key]) + process
        # do not additionally add comment.
    elif add_comment:
        # add comment.
        cmt2hdr(
            header,
            "c",
            time_fmt=None,
            s=f"[air.update_process] Standard items for {key} includes B=bias, D=dark, "
            + "F=flat, T=trim, W=WCS, O=Overscan, I=Illumination, C=CRrej, R=fringe, "
            + "P=fixpix, X=crosstalk.",
        )

    header[key] = (delimiter.join(process), "Process (order: 1-2-3-...): see comment.")

    if additional_comment:
        addstr = ", ".join([f"{k}={v}" for k, v in additional_comment.items()])
        cmt2hdr(header, "c", f"User added items to {key}: {addstr}.", time_fmt=None)
    update_tlm(header)


def hedit(
    item: fits.Header | CCDLike,
    keys: str | list[str],
    values: object | list[object],
    comments: str | list[str] | None = None,
    befores: str | int | list[str | int] | None = None,
    afters: str | int | list[str | int] | None = None,
    add: bool = False,
    output: StrPathLike | None = None,
    overwrite: bool = False,
    output_verify: str = "fix",
    verbose: bool = True,
) -> CCDData | None:
    """Edit FITS header keyword values.

    Parameters
    ----------
    item : `astropy` header, path-like, `~astropy.nddata.CCDData`-like
        The FITS file or header to edit. If `~astropy.io.fits.Header`, it is updated
        **inplace**.

    keys : `str`, `list`-like of `str`
        The key to edit.

    values : `str`, numeric, or `list`-like of such
        The new value. To pass one single iterable (e.g., ``[1, 2, 3]``) for one
        single `key`, use a `list` of it (e.g., ``[[1, 2, 3]]``) to circumvent
        the problem.

    comments : `str`, `list`-like of `str`, optional
        Comments to add.

    add : `bool`, optional.
        Whether to add the key if it is not in the header.
        Default: `False`.

    befores, afters : `str`, `int`, `list`-like of such, optional
        Name of the keyword, or index of the `Card` before which this card
        should be located in the header, or after which this card should be
        located. The argument `before` takes precedence over `after` if both
        specified.
        Default: `None`.

    output: path-like, optional
        The output file.
        Default: `None`.

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        The header-updated `~astropy.nddata.CCDData`. `None` if `item` was
        pure `~astropy.io.fits.Header`.
    """

    def _add_key(header, key, val, infostr, cmt=None, before=None, after=None):
        header.set(key, value=val, comment=cmt, before=before, after=after)
        # infostr += " (comment: {})".format(comment) if comment is not None else ""
        if before is not None:
            infostr += f" (moved: {before=})"
        elif after is not None:  # `after` is ignored if `before` is given
            infostr += f" (moved: {after=})"
        cmt2hdr(header, "h", infostr, verbose=verbose)
        update_tlm(header)

    if isinstance(item, fits.header.Header):
        header = item
        if verbose:
            logger.info("item is astropy Header. (any `output` is ignored).")
        output = None
        ccd = None
    elif isinstance(item, _io.ASTROPY_CCD_TYPES):
        ccd, imname, _ = _io._parse_image(item, force_ccddata=True, copy=False)
        #                                                   ^^^^^^^^^^
        # Use copy=False to update header of the input CCD inplace.z
        header = ccd.header
    else:
        ccd = _io.load_ccd(item)
        imname = str(item)
        header = ccd.header
        if output is None and overwrite:
            output = item

    keys, values, comments, befores, afters = listify(
        keys, values, comments, befores, afters
    )

    for key, val, cmt, bef, aft in zip(
        keys, values, comments, befores, afters, strict=False
    ):
        if key in header:
            oldv = header[key]
            infostr = (
                f"[air.HEDIT] {key}={oldv} ({type(oldv).__name__}) "
                f"--> {val} ({type(val).__name__})"
            )
            _add_key(header, key, val, infostr, cmt=cmt, before=bef, after=aft)
        else:
            if add:  # add key only if `add` is True.
                infostr = f"[air.HEDIT] Add {key}= {val} ({type(val).__name__})"
                _add_key(header, key, val, infostr, cmt=cmt, before=bef, after=aft)
            elif verbose:
                logger.info(
                    "%s does not exist in the header. Skipped. (add=True to proceed)",
                    key,
                )

    if output is not None:
        ccd.write(output, overwrite=overwrite, output_verify=output_verify)
        if verbose:
            logger.info("%s --> %s", imname, output)

    return ccd


def key_remover(
    header: fits.Header, remove_keys: list[str], deepremove: bool = True
) -> fits.Header:
    """Remove keywords from a header.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        The header to be modified

    remove_keys : `list` of `str`
        The header keywords to be removed.

    deepremove : `bool`, optional
        FITS standard does not have any specification of duplication of
        keywords as discussed in the following issue:
        https://github.com/astropy/ccdproc/issues/464
        If it is set to `True`, ALL the keywords having the name specified in
        `remove_keys` will be removed. If not, only the first occurrence of each
        key in `remove_keys` will be removed. It is more sensible to set it
        `True` in most of the cases.
        Default: `True`.
    """
    nhdr = header.copy()
    if deepremove:
        for key in remove_keys:
            while True:
                try:
                    nhdr.remove(key)
                except KeyError:
                    break
    else:
        for key in remove_keys:
            try:
                nhdr.remove(key)
            except KeyError:
                continue

    return nhdr


def key_mapper(
    header: fits.Header,
    keymap: dict | None = None,
    deprecation: bool = False,
    remove: bool = False,
) -> fits.Header:
    """Update a header to match a keyword map.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        The header to be modified

    keymap : `dict`, optional.
        The dictionary contains ``{<standard_key>:<original_key>}``
        information. If it is `None` (default), the copied version of the
        header is returned without any change.
        Default: `None`.

    deprecation : `bool`, optional
        Whether to change the original keywords' comments to contain
        deprecation warning. If `True`, the original keywords' comments will
        become ``DEPRECATED. See <standard_key>.``. It has no effect if
        ``remove=True``.
        Default is `False`.

    remove : `bool`, optional.
        Whether to remove the original keyword. `deprecation` is ignored if
        ``remove=True``.
        Default is `False`.

    Returns
    -------
    newhdr : `~astropy.io.fits.Header`
        The updated (key-mapped) header.

    Notes
    -----
    If the new keyword already exist in the given header, virtually nothing
    will happen. If ``deprecation=True``, the old one's comment will be
    changed, and if ``remove=True``, the old one will be removed; the new
    keyword will never be changed or overwritten.
    """

    def _rm_or_dep(hdr, old, new):
        if remove:
            hdr.remove(old)
        elif deprecation:  # do not remove but deprecate
            hdr.comments[old] = f"DEPRECATED. See {new}"

    newhdr = header.copy()
    if keymap is not None:
        for k_new, k_old in keymap.items():
            if k_new == k_old:
                continue

            if k_old is not None:
                if (
                    k_new in newhdr
                ):  # if k_new already in the header, JUST deprecate k_old.
                    _rm_or_dep(newhdr, k_old, k_new)
                else:  # if not, copy k_old to k_new and deprecate k_old.
                    try:
                        comment_ori = newhdr.comments[k_old]
                        newhdr[k_new] = (newhdr[k_old], comment_ori)
                        _rm_or_dep(newhdr, k_old, k_new)
                    except (KeyError, IndexError):
                        # don't even warn
                        pass

    return newhdr


def chk_keyval(type_key, type_val, group_key) -> tuple[list[str], list, list[str]]:
    """Validate type and group keyword/value arguments.

    Parameters
    ----------
    type_key : `None`, `str`, `list` of `str`, optional
        The header keyword for the ccd type you want to use for match.

    type_val : `None`, `int`, `str`, `float`, etc and `list` of such
        The header keyword values for the ccd type you want to match.


    group_key : `None`, `str`, `list` of `str`, optional
        The header keyword which will be used to make groups for the CCDs that
        have selected from `type_key` and `type_val`. If `None` (default), no
        grouping will occur, but it will return the `~pandas.DataFrameGroupBy`
        object will be returned for the sake of consistency.

    Returns
    -------
    type_key, type_val, group_key
    """
    # Make type_key to list
    if type_key is None:
        type_key = []
    elif is_list_like(type_key):
        try:
            type_key = list(type_key)
            if not all(isinstance(x, str) for x in type_key):
                raise TypeError("Some of type_key are not str.")
        except TypeError as err:
            raise TypeError("type_key should be str or convertible to list.") from err
    elif isinstance(type_key, str):
        type_key = [type_key]
    else:
        raise TypeError(
            f"`type_key` not understood (type = {type(type_key)}): {type_key}"
        )

    # Make type_val to list
    if type_val is None:
        type_val = []
    elif is_list_like(type_val):
        try:
            type_val = list(type_val)
        except TypeError as err:
            raise TypeError("type_val should be str or convertible to list.") from err
    elif isinstance(type_val, str):
        type_val = [type_val]
    else:
        raise TypeError(
            f"`type_val` not understood (type = {type(type_val)}): {type_val}"
        )

    # Make group_key to list
    if group_key is None:
        group_key = []
    elif is_list_like(group_key):
        try:
            group_key = list(group_key)
            if not all(isinstance(x, str) for x in group_key):
                raise TypeError("Some of group_key are not str.")
        except TypeError as err:
            raise TypeError("group_key should be str or convertible to list.") from err
    elif isinstance(group_key, str):
        group_key = [group_key]
    else:
        raise TypeError(
            f"`group_key` not understood (type = {type(group_key)}): {group_key}"
        )

    if len(type_key) != len(type_val):
        raise ValueError("`type_key` and `type_val` must have the same length!")

    # If there is overlap
    overlap = set(type_key).intersection(set(group_key))
    if len(overlap) > 0:
        logger.warning(
            f"{overlap} appear in both `type_key` and `group_key`."
            "It may not be harmful but better to avoid."
        )

    return type_key, type_val, group_key


def hdrval(
    value=None,
    header: fits.Header | None = None,
    key: str | None = None,
    default=None,
    unit: str | u.Unit | None = None,
    verbose: bool = False,
    to_value: bool = False,
    return_source: bool = False,
):
    """Get a value by priority: ``value`` > ``header[key]`` > ``default``.

    Parameters
    ----------
    value : object, optional.
        If not `None`, `header`, `key`, and `default` will **not** be used.
        This is different from `header.get(key, default)`. It is therefore
        useful if the API wants to override the header value by the
        user-provided one.
        Default: `None`.

    header : `~astropy.io.fits.Header`, optional.
        The header to extract the value if `value` is `None`.
        Default: `None`.

    key : `str`, optional.
        The header keyword to extract if `value` is `None`.
        Default: `None`.

    default : object, optional.
        The default value. If `value` is `None` and ``header[key]`` is not
        available, this value is used.
        Default: `None`.

    unit : `str` or `~astropy.units.Unit`, optional.
        `None` to ignore unit. ``''`` (empty string) is interpreted as
        `~astropy.units.dimensionless_unscaled`.
        Better to leave it as `None` unless an astropy unit is truly needed.
        Default: `None`.

    verbose : `bool`, optional.
        Whether to log the source and missing-key fallback.
        Default: `False`.

    to_value : `bool`, optional.
        Whether to return only the scalar value when the result is an
        `~astropy.units.Quantity`.
        Default: `False`.

    return_source : `bool`, optional.
        Whether to return ``(value, source)``. ``source`` is one of
        ``"the user"``, ``"{KEY} in header"``, or ``"default"``.
        Default: `False`.

    Notes
    -----
    It takes << 10 us (when unit=`None`) or for any case for a reasonably lengthy
    header. See `Tests` below. Tested on MBP 15" [2018, macOS 11.6, i7-8850H
    (2.6 GHz; 6-core), RAM 16 GB (2400MHz DDR4), Radeon Pro 560X (4GB)].

    Tests
    -----

    .. code-block:: python

        real_q = 20*u.s
        real_v = 20
        default_q = 0*u.s
        default_v = 0
        test_q = 3*u.s
        test_v = 3

        # w/o unit  Times are the %timeit result of the LHS
        assert hdrval(None,   hdr, "EXPTIME", default=0) == real_v  # ~ 6.5 us
        assert hdrval(None,   hdr, "EXPTIxx", default=0) == default_v # ~ 3.5 us
        assert hdrval(test_v, hdr, "EXPTIxx", default=0) == test_v  # ~ 0.3 us
        assert hdrval(test_q, hdr, "EXPTIxx", default=0) == test_v  # ~ 0.6 us
        # w/ unit  Times are the %timeit result of the LHS
        assert hdrval(None,   hdr, "EXPTIME", default=0, unit='s') == real_q  # ~ 23 us
        assert hdrval(None,   hdr, "EXPTIxx", default=0, unit='s') == default_q # ~ 16 us
        assert hdrval(test_v, hdr, "EXPTIxx", default=0, unit='s') == test_q  # ~ 11 us
        assert hdrval(test_q, hdr, "EXPTIxx", default=0, unit='s') == test_q  # ~ 15 us

        For a test astropy.nddata.CCDData, the following timing gave ~ 0.5 ms on MBP 15" [2018,
        macOS 11.6, i7-8850H (2.6 GHz; 6-core), RAM 16 GB (2400MHz DDR4), Radeon
        Pro 560X (4GB)]
        %timeit ((air.hdrval(None, ccd.header, "EXPTIME", unit=u.s)
                 / air.hdrval(3*u.s, ccd.header, "EXPTIME", unit=u.s)).si.value)
    """
    if value is None:
        if key is None:
            raise ValueError("key must be given when value is None.")
        key = key.upper()
        try:
            value = header[key]
            source = f"{key} in header"
            if verbose:
                logger.info("header: %-8s = %s", key, value)
        except (KeyError, IndexError):
            value = default
            source = "default"
            if verbose:
                logger.warning(
                    "The key %s not found in header: setting to %s.", key, default
                )
    else:
        source = "the user"

    if isinstance(value, u.Quantity):
        value = value.value if unit is None else value.to(unit)
    elif unit is not None:
        value = change_to_quantity(value, unit, to_value=False)

    if to_value:
        with contextlib.suppress(AttributeError):
            value = value.value
    if return_source:
        return value, source
    return value


def midtime_obs(
    header: fits.Header | None = None,
    dateobs: str | Time = "DATE-OBS",
    format: str | None = None,
    scale: str | None = None,
    precision: int | None = None,
    in_subfmt: str | None = None,
    out_subfmt: str | None = None,
    location=None,
    exptime: str | float | u.Quantity = "EXPTIME",
    exptime_unit: u.Unit = u.s,
) -> Time:
    """Calculate the mid-observation time.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`, optional.
        The header to extract the value. `midtime_obs` can be used without
        header. But to do so, `dateobs` must be in `~astropy.time.Time` and
        `exptime` must be given as `float` or `~astropy.units.Quantity`.
        Default: `None`.

    dateobs : `str` or `~astropy.time.Time`, optional.
        The header keyword for DATE-OBS (start of exposure) or the
        `~astropy.Time` object.
        Default: ``'DATE-OBS'``.

    exptime : `str`, `float`, `~astropy.units.Quantity`, optional.
        The header keyword for exposure time or the exposure time as `float` (in
        seconds) or `~astropy.units.Quantity`.
        Default: ``'EXPTIME'``.

    format, scale, precision, in_subfmt, out_subfmt, location : optional
        Passed to `~astropy.time.Time` when `dateobs` is read from the header.

    exptime_unit : `~astropy.units.Unit`, optional
        Unit applied to numeric exposure times. Default is seconds.

    """
    if isinstance(dateobs, str):
        try:
            time_0 = Time(
                header[dateobs],
                format=format,
                scale=scale,
                precision=precision,
                in_subfmt=in_subfmt,
                out_subfmt=out_subfmt,
                location=location,
            )
        except (KeyError, IndexError) as err:
            raise KeyError(f"The key '{dateobs=}' not found in header.") from err
    else:
        time_0 = dateobs

    if isinstance(exptime, str):
        try:
            exptime = header.get(exptime, default=0) * exptime_unit
        except (KeyError, IndexError) as err:
            raise KeyError(f"The key '{exptime=}' not found in header.") from err
    elif isinstance(exptime, (int, float)):
        exptime = exptime * exptime_unit
    elif not isinstance(exptime, u.Quantity):
        raise TypeError(f"exptime type ({type(exptime)}) not understood.")

    return time_0 + exptime / 2
