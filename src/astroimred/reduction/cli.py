"""Command line tools for ``astroimred.reduction``."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import click
import numpy as np

from astroimred.mgmt.logging import enable_console_logging

from .imutil.imarith import imarith
from .imutil.imcombine import imcombine
from .imutil.imcopy import imcopy

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


def _none_if_blank(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"none", "null", "indef"}:
        return None
    return text


def _parse_input(value):
    """Parse IRAF-like input lists while leaving globs and @files intact."""
    value = _none_if_blank(value)
    if value is None:
        raise click.BadParameter("input must not be blank")
    if value.startswith("@"):
        return value
    if "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return value


def _parse_extension(value):
    value = _none_if_blank(value)
    if value is None:
        return None

    if "," in value:
        name, extver = [part.strip() for part in value.split(",", 1)]
        try:
            return (name, int(extver))
        except ValueError as exc:
            raise click.BadParameter(
                "extension tuples must look like 'EXTNAME,EXTVER'"
            ) from exc

    try:
        return int(value)
    except ValueError:
        return value


def _parse_float_vector(value):
    value = _none_if_blank(value)
    if value is None:
        return None
    if value.startswith("@"):
        try:
            data = np.loadtxt(value[1:], dtype=float)
        except OSError as exc:
            raise click.BadParameter(f"could not read {value}") from exc
        return np.atleast_1d(data).astype(float)

    parts = [part for part in re.split(r"[\s,]+", value) if part]
    if len(parts) > 1:
        try:
            return np.asarray([float(part) for part in parts], dtype=float)
        except ValueError as exc:
            raise click.BadParameter("expected numeric values") from exc

    try:
        return np.asarray([float(value)], dtype=float)
    except ValueError:
        return value


def _parse_offsets(value):
    value = _none_if_blank(value)
    if value is None:
        return None

    mode = value.lower()
    if mode in {"wcs", "world", "physical", "phys", "phy"}:
        return mode
    if mode == "none":
        return None

    try:
        if value.startswith("@"):
            offsets = np.loadtxt(value[1:], dtype=float)
        else:
            rows = [
                [float(part) for part in re.split(r"[\s,]+", row.strip()) if part]
                for row in value.split(";")
                if row.strip()
            ]
            offsets = np.asarray(rows, dtype=float)
    except (OSError, ValueError) as exc:
        raise click.BadParameter(
            "offsets must be none, wcs, world, physical, @file, "
            "or rows like '0,0; 2,-1; 3,4'"
        ) from exc

    if offsets.ndim == 1:
        offsets = offsets[:, None]
    return offsets


def _parse_memlimit(value):
    value = _none_if_blank(value)
    if value is None:
        return None

    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z]*)",
        value,
    )
    if match is None:
        raise click.BadParameter("expected bytes or a value like 512MB, 1GiB")

    number = float(match.group(1))
    unit = match.group(2).lower()
    factors = {
        "": 1,
        "b": 1,
        "k": 1e3,
        "kb": 1e3,
        "m": 1e6,
        "mb": 1e6,
        "g": 1e9,
        "gb": 1e9,
        "t": 1e12,
        "tb": 1e12,
        "ki": 1024,
        "kib": 1024,
        "mi": 1024**2,
        "mib": 1024**2,
        "gi": 1024**3,
        "gib": 1024**3,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    try:
        return number * factors[unit]
    except KeyError as exc:
        raise click.BadParameter(f"unknown memory unit {unit!r}") from exc


def _parse_dtype(value):
    value = _none_if_blank(value)
    if value is None:
        return "float32"
    aliases = {
        "real": "float32",
        "r": "float32",
        "double": "float64",
        "d": "float64",
        "short": "int16",
        "s": "int16",
        "ushort": "uint16",
        "u": "uint16",
        "integer": "int32",
        "i": "int32",
        "long": "int64",
        "l": "int64",
    }
    return aliases.get(value.lower(), value)


def _parse_float_or_key(value):
    value = _none_if_blank(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return value


def _parse_replace(value):
    value = _none_if_blank(value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise click.BadParameter("replace must be a number or none") from exc


def _thresholds(lthreshold, hthreshold):
    low = -np.inf if lthreshold is None else float(lthreshold)
    high = np.inf if hthreshold is None else float(hthreshold)
    return [low, high]


def _reject_name(value):
    value = _none_if_blank(value)
    if value is None:
        return None
    value = value.lower()
    if value == "none":
        return None
    return value


def _configure_logging(verbose):
    if verbose <= 0:
        return
    level = logging.DEBUG if verbose > 1 else logging.INFO
    enable_console_logging(level=level)


@click.group(context_settings=CONTEXT_SETTINGS)
def main():
    """astroimred.reduction FITS image-in/image-out reduction tools."""


@click.command(
    name="comb",
    context_settings=CONTEXT_SETTINGS,
    help=(
        "Combine FITS images pixel-by-pixel with astroimred.reduction. IRAF-like parameters: "
        "imred comb INPUT OUTPUT."
    ),
)
@click.argument("input_", metavar="input")
@click.argument(
    "output",
    metavar="output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.option(
    "--combine",
    type=click.Choice(
        ["average", "mean", "median", "lmedian", "sum", "minimum", "maximum"],
        case_sensitive=False,
    ),
    default="average",
    show_default=True,
    help="Final combine operation.",
)
@click.option(
    "--reject",
    type=click.Choice(["none", "minmax", "sigclip", "ccdclip"], case_sensitive=False),
    default="none",
    show_default=True,
    help="Pixel rejection algorithm.",
)
@click.option(
    "--offsets",
    callback=lambda _c, _p, v: _parse_offsets(v),
    help="none, wcs, world, physical, @file, or semicolon-separated rows.",
)
@click.option(
    "--extension",
    callback=lambda _c, _p, v: _parse_extension(v),
    help="Image extension: integer, EXTNAME, or EXTNAME,EXTVER.",
)
@click.option(
    "--extension-uncertainty",
    callback=lambda _c, _p, v: _parse_extension(v),
    help="Uncertainty extension.",
)
@click.option(
    "--extension-mask",
    callback=lambda _c, _p, v: _parse_extension(v),
    help="Mask extension.",
)
@click.option(
    "--trimsec",
    "--outlimits",
    "trimsec",
    help="Input section passed to imcombine trimsec.",
)
@click.option("--lthreshold", type=float, help="Low input-pixel threshold.")
@click.option("--hthreshold", type=float, help="High input-pixel threshold.")
@click.option(
    "--zero",
    callback=lambda _c, _p, v: _parse_float_vector(v),
    help="Zero correction: none, statistic name, @file, or numeric vector.",
)
@click.option(
    "--scale",
    callback=lambda _c, _p, v: _parse_float_vector(v),
    help="Scale correction: none, statistic/exposure name, @file, or numeric vector.",
)
@click.option(
    "--weight",
    callback=lambda _c, _p, v: _parse_float_vector(v),
    help="Weight: none, statistic name, @file, or numeric vector.",
)
@click.option(
    "--statsec", help="Section used for zero and scale statistics unless overridden."
)
@click.option("--zero-section", help="Section used for zero statistics.")
@click.option("--scale-section", help="Section used for scale statistics.")
@click.option(
    "--zero-to-0th/--no-zero-to-0th",
    default=True,
    show_default=True,
    help="Normalize zero values to the first image.",
)
@click.option(
    "--scale-to-0th/--no-scale-to-0th",
    default=True,
    show_default=True,
    help="Normalize scale values to the first image.",
)
@click.option(
    "--nlow",
    type=int,
    default=1,
    show_default=True,
    help="Number of low pixels rejected by minmax.",
)
@click.option(
    "--nhigh",
    type=int,
    default=1,
    show_default=True,
    help="Number of high pixels rejected by minmax.",
)
@click.option(
    "--nkeep",
    type=int,
    default=1,
    show_default=True,
    help="Minimum pixels to keep for clipping algorithms.",
)
@click.option(
    "--maxrej", type=int, help="Maximum pixels to reject for clipping algorithms."
)
@click.option(
    "--mclip/--no-mclip",
    default=True,
    show_default=True,
    help="Use median center for sigma clipping.",
)
@click.option(
    "--lsigma",
    type=float,
    default=3.0,
    show_default=True,
    help="Lower sigma clipping factor.",
)
@click.option(
    "--hsigma",
    type=float,
    default=3.0,
    show_default=True,
    help="Upper sigma clipping factor.",
)
@click.option(
    "--maxiters",
    type=int,
    default=50,
    show_default=True,
    help="Maximum sigma-clipping iterations.",
)
@click.option(
    "--ddof",
    type=int,
    default=1,
    show_default=True,
    help="Delta degrees of freedom for standard deviation.",
)
@click.option(
    "--rdnoise",
    default="0.0",
    show_default=True,
    callback=lambda _c, _p, v: _parse_float_or_key(v),
    help="CCD read noise value or header key for ccdclip.",
)
@click.option(
    "--gain",
    default="1.0",
    show_default=True,
    callback=lambda _c, _p, v: _parse_float_or_key(v),
    help="CCD gain value or header key for ccdclip.",
)
@click.option(
    "--snoise",
    default="0.0",
    show_default=True,
    callback=lambda _c, _p, v: _parse_float_or_key(v),
    help="Sensitivity noise for ccdclip.",
)
@click.option(
    "--outtype",
    "--dtype",
    "dtype",
    default="real",
    show_default=True,
    callback=lambda _c, _p, v: _parse_dtype(v),
    help="Output dtype or IRAF outtype alias.",
)
@click.option(
    "--memlimit",
    default="2.5GB",
    show_default=True,
    callback=lambda _c, _p, v: _parse_memlimit(v),
    help="Approximate stack memory limit.",
)
@click.option(
    "--imcmb",
    "imcmb_key",
    default="$I",
    show_default=True,
    help="Header keyword copied into IMCMBnnn; '$I' records filenames.",
)
@click.option(
    "--expname",
    "exposure_key",
    default="EXPTIME",
    show_default=True,
    help="Exposure-time header key.",
)
@click.option(
    "--logfile",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Write imcombine log table.",
)
@click.option(
    "--output-mask",
    "--rejmask",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output total rejection/exclusion mask.",
)
@click.option(
    "--output-nrej",
    "--nrejmasks",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output number of rejected/excluded pixels.",
)
@click.option(
    "--output-err",
    "--sigma-output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output standard deviation or variance image.",
)
@click.option(
    "--output-low",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output lower rejection bound image.",
)
@click.option(
    "--output-upp",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output upper rejection bound image.",
)
@click.option(
    "--output-rejcode",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Output integer rejection-code image.",
)
@click.option(
    "--return-variance/--return-stddev",
    default=False,
    show_default=True,
    help="Store variance instead of standard deviation in output-err.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite existing output files.",
)
@click.option(
    "--checksum/--no-checksum",
    default=False,
    show_default=True,
    help="Write FITS checksums.",
)
@click.option(
    "-v", "--verbose", count=True, help="Show progress logging; repeat for debug level."
)
def comb_command(
    input_,
    output,
    combine,
    reject,
    offsets,
    extension,
    extension_uncertainty,
    extension_mask,
    trimsec,
    lthreshold,
    hthreshold,
    zero,
    scale,
    weight,
    statsec,
    zero_section,
    scale_section,
    zero_to_0th,
    scale_to_0th,
    nlow,
    nhigh,
    nkeep,
    maxrej,
    mclip,
    lsigma,
    hsigma,
    maxiters,
    ddof,
    rdnoise,
    gain,
    snoise,
    dtype,
    memlimit,
    imcmb_key,
    exposure_key,
    logfile,
    output_mask,
    output_nrej,
    output_err,
    output_low,
    output_upp,
    output_rejcode,
    return_variance,
    overwrite,
    checksum,
    verbose,
):
    """Run ``astroimred.reduction.imcombine`` from the command line."""
    _configure_logging(verbose)
    inputs = _parse_input(input_)
    reject = _reject_name(reject)
    cenfunc = "median" if mclip else "average"
    zero_section = zero_section or statsec
    scale_section = scale_section or statsec

    try:
        imcombine(
            inputs=inputs,
            extension=extension,
            extension_uncertainty=extension_uncertainty,
            extension_mask=extension_mask,
            trimsec=trimsec,
            offsets=offsets,
            thresholds=_thresholds(lthreshold, hthreshold),
            zero=zero,
            zero_to_0th=zero_to_0th,
            zero_section=zero_section,
            scale=scale,
            scale_to_0th=scale_to_0th,
            scale_section=scale_section,
            weight=weight,
            reject=reject,
            sigma=[lsigma, hsigma],
            cenfunc=cenfunc,
            maxiters=maxiters,
            ddof=ddof,
            nkeep=nkeep,
            maxrej=maxrej,
            n_minmax=[nlow, nhigh],
            rdnoise=rdnoise,
            gain=gain,
            snoise=snoise,
            logfile=logfile,
            combine=combine,
            dtype=dtype,
            memlimit=memlimit,
            verbose=verbose,
            return_variance=return_variance,
            imcmb_key=imcmb_key,
            exposure_key=exposure_key,
            output=output,
            output_mask=output_mask,
            output_nrej=output_nrej,
            output_err=output_err,
            output_low=output_low,
            output_upp=output_upp,
            output_rejcode=output_rejcode,
            output_verify="exception",
            overwrite=overwrite,
            checksum=checksum,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@click.command(
    name="copy",
    context_settings=CONTEXT_SETTINGS,
    help="Copy FITS images or sections with astroimred.reduction: imred copy INPUT OUTPUT.",
)
@click.argument("input_", metavar="input")
@click.argument(
    "output",
    metavar="output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.option(
    "--trimsec",
    help="FITS section to copy, in the same syntax accepted by imred.imcopy.",
)
@click.option(
    "--extension",
    callback=lambda _c, _p, v: _parse_extension(v),
    help="Image extension: integer, EXTNAME, or EXTNAME,EXTVER.",
)
@click.option(
    "--outtype",
    "--dtype",
    "dtype",
    callback=lambda _c, _p, v: _parse_dtype(v),
    help="Output dtype or IRAF outtype alias.",
)
@click.option(
    "--update-header/--no-update-header",
    default=True,
    show_default=True,
    help="Update output FITS header timing/shape metadata.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite existing output files.",
)
def copy_command(input_, output, trimsec, extension, dtype, update_header, overwrite):
    """Run ``astroimred.reduction.imcopy`` from the command line."""
    try:
        imcopy(
            inputs=_parse_input(input_),
            trimsecs=trimsec,
            outputs=output,
            extension=extension,
            return_ccd=False,
            dtype=dtype,
            update_header=update_header,
            overwrite=overwrite,
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


@click.command(
    name="arith",
    context_settings=CONTEXT_SETTINGS,
    help="Apply image arithmetic with astroimred.reduction: imred arith IM1 OP IM2 OUTPUT.",
)
@click.argument("im1")
@click.argument("op")
@click.argument("im2")
@click.argument(
    "output",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
)
@click.option(
    "--extension1",
    callback=lambda _c, _p, v: _parse_extension(v),
    help="Extension for IM1.",
)
@click.option(
    "--extension2",
    callback=lambda _c, _p, v: _parse_extension(v),
    help="Extension for IM2.",
)
@click.option("--name1", help="Name recorded for IM1 in the output header history.")
@click.option("--name2", help="Name recorded for IM2 in the output header history.")
@click.option(
    "--offsets",
    callback=lambda _c, _p, v: _parse_offsets(v),
    help="none, wcs, world, physical, @file, or semicolon-separated rows.",
)
@click.option(
    "--replace",
    default="0",
    show_default=True,
    callback=lambda _c, _p, v: _parse_replace(v),
    help="Replacement for non-finite output pixels; use none to preserve them.",
)
@click.option(
    "--outtype",
    "--dtype",
    "dtype",
    default="real",
    show_default=True,
    callback=lambda _c, _p, v: _parse_dtype(v),
    help="Output dtype or IRAF outtype alias.",
)
@click.option(
    "--error-calc/--no-error-calc",
    default=False,
    show_default=True,
    help="Use CCDData arithmetic error propagation.",
)
@click.option(
    "--ignore-header/--no-ignore-header",
    default=False,
    show_default=True,
    help="Ignore header alignment checks.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    show_default=True,
    help="Overwrite existing output files.",
)
@click.option("-v", "--verbose", count=True, help="Show progress logging.")
def arith_command(
    im1,
    op,
    im2,
    output,
    extension1,
    extension2,
    name1,
    name2,
    offsets,
    replace,
    dtype,
    error_calc,
    ignore_header,
    overwrite,
    verbose,
):
    """Run ``astroimred.reduction.imarith`` from the command line."""
    _configure_logging(verbose)
    try:
        imarith(
            im1=im1,
            op=op,
            im2=im2,
            output=output,
            extension1=extension1,
            extension2=extension2,
            name1=name1,
            name2=name2,
            offsets=offsets,
            replace=replace,
            dtype=dtype,
            error_calc=error_calc,
            ignore_header=ignore_header,
            overwrite=overwrite,
            output_verify="silentfix",
            verbose=bool(verbose),
        )
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc


main.add_command(comb_command)
main.add_command(copy_command)
main.add_command(arith_command)


if __name__ == "__main__":
    main()
