"""FITS path, directory, and renaming helpers."""

from pathlib import Path

import numpy as np
from astro_ndslice import slice_from_string
from astropy.io import fits
from astropy.nddata import CCDData

from ..logging import logger
from .._types import StrPathLike
from .headers import key_mapper, key_remover

__all__ = [
    "mkdir",
    "fits_newpath",
    "fitsrenamer",
]


def mkdir(fpath: StrPathLike, mode: int = 0o777, exist_ok: bool = True) -> None:
    """Convenience function for `~pathlib.Path`.mkdir()"""
    fpath = Path(fpath)
    Path.mkdir(fpath, mode=mode, exist_ok=exist_ok)


def fits_newpath(
    fpath: StrPathLike,
    rename_by: list[str],
    mkdir_by: list[str] | None = None,
    header: fits.Header | None = None,
    delimiter: str = "_",
    fillnan: str = "",
    fileext: str = ".fits",
) -> Path:
    """Give the new path of a FITS file from header values.

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
        ``'.fits'`` since this function is ``fits_newpath``, but you may prefer,
        e.g., ``'.fit'`` for some reason. If `fileext` does not start with a
        period (``"."``), one is prepended automatically.
        Default: ``'.fits'``.

    Returns
    -------
    newpath : path
        The new path.
    """

    hdr = fits.getheader(fpath) if header is None else header.copy()

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
    fpath: StrPathLike | None = None,
    header: fits.Header | None = None,
    newtop: StrPathLike | None = None,
    rename_by: list[str] | None = None,
    mkdir_by: list[str] | None = None,
    delimiter: str = "_",
    archive_dir: StrPathLike | None = None,
    keymap: dict | None = None,
    key_deprecation: bool = True,
    remove_keys: list[str] | None = None,
    overwrite: bool = False,
    fillnan: str = "",
    trimsec: str | None = None,
    verbose: bool = True,
    add_header=None,
) -> Path:
    """Rename a FITS file using header values.

    Parameters
    ----------
    fpath : path-like, optional.
        The path to the target FITS file.
        Default: `None`.

    header : `~astropy.io.fits.Header`, optional
        The header of the fits file, especially if you want to just overwrite
        the header with this.
        Default: `None`.

    newtop : path-like, optional
        The top path for the new FITS file. If `None`, the new path will share
        the parent path with `fpath`.
        Default: `None`.

    rename_by : `list` of `str`, optional
        The keywords of the FITS header to rename by.
        Default: `None` which uses ``['OBJECT']``.

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
        become ``DEPRECATED. See <standard_key>.``.
        Default: `True`.

    trimsec : `str` or `None`, optional
        FITS-style section string used to trim the image before writing the
        renamed file. Default is `None`.

    fillnan : `str`, optional
        The string that will be inserted if the keyword is not found from the
        header.
        Default: ``''``.

    remove_keys : `list` of `str`, optional.
        The header keywords to be removed.
        Default: `None`.

    add_header : `~astropy.io.fits.Header` or `~astropy.io.fits.Card`, optional.
        The header keyword, value (and comment) to add after the renaming.
        Default: `None`.

    Notes
    -----
    MEF(Multi-Extension FITS) currently is not supported.

    """

    # Load fits file
    if rename_by is None:
        rename_by = ["OBJECT"]
    hdul = fits.open(fpath)
    data = hdul[0].data
    hdr = hdul[0].header if header is None else header.copy()
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
        from ..imops.ccdutils import cut_ccd

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
