# Package scope notes

This memo summarizes a local scan of:

- `/Users/ysbach/Dropbox/github/ysfitsutilpy`
- `/Users/ysbach/Dropbox/github/ysphotutilpy`
- `/Users/ysbach/Dropbox/github/ysvisutilpy`
- `/Users/ysbach/Dropbox/github/fitsmgmt`

## Current code shape

`ysfitsutilpy` is not one coherent layer. It mixes FITS I/O/header/file
management (`hduutil.py`, `filemgmt.py`), image combination (`imutil/`,
`combutil.py`), calibration/reduction (`preproc.py`), observation geometry
(`airmass.py`, WCS helpers), remote astrometry (`astrometry.py`), and generic
math helpers (`misc.py`, `fitting.py`). The current `fitsmgmt` repo has already
ported much of the FITS I/O/header/file-management layer and some WCS/stat
helpers into focused modules such as `io.py`, `headers.py`, `summary.py`,
`paths.py`, `ccdutils.py`, `pixels.py`, `imstat.py`, `misc.py`, `airmass.py`, and
`wcstools.py`.

`ysphotutilpy` is a photometry/image-analysis package. Its core is aperture
geometry, aperture photometry, sky/background estimation, centroiding, SEP
extraction, radial profiles, PSF-ish helpers, and polarimetry. It also contains
catalog/Horizons/Pan-STARRS query code that is useful but has different runtime
and testing constraints from local image analysis.

`ysvisutilpy` is compact and mostly independent: astronomy-aware image display
normalization, Matplotlib tick/colorbar helpers, histogram helpers, and optional
statistical diagnostic plots.

## Recommended split

Use a small package family rather than one large package:

1. `fitsmgmt` as the package family/root distribution.
   - Owns FITS file discovery, summaries, header editing, CCDData/HDU parsing,
     extension selection, basic WCS metadata extraction, and small visualization
     helpers that inspect FITS images.
   - Exposes modules such as `fitsmgmt.io`, `fitsmgmt.headers`,
     `fitsmgmt.ccdutils`, `fitsmgmt.wcstools`, and `fitsmgmt.viz`.
   - Should not own calibration policy, object photometry, polarimetry, catalog
     queries, or spectroscopy.
2. `fitsimred` for image-in/image-out reduction.
   - Bias/dark/flat/fringe/illumination correction, cosmic ray rejection,
     image arithmetic, stacking/combination, rejection masks, and reduction
     planning.
   - Depends on `fitsmgmt`; optional dependencies include `ccdproc`,
     `astroscrappy`, `numba`, and `bottleneck`.
3. `fitsimphot` for science measurement on imaging data.
   - Apertures, source detection, background estimation, centroiding, PSF/PRF
     helpers, radial profiles, growth curves, photometry tables, and
     polarimetry.
   - Depends on `fitsmgmt`; may optionally use `fitsimred` only in examples or
     pipeline glue.
4. `fitsimspec` later.
   - Keep spectroscopy out until there is real code pressure. Spectroscopy has
     a different data model, calibration vocabulary, and visualization surface.
5. Optional catalog/query package or module.
   - `ysphotutilpy/queryutil.py` is useful, but it mixes network services,
     catalog-specific schemas, and FOV filtering. Consider keeping it outside
     the core photometry package, or under an optional `fitsimphot.catalogs`
     extra.

## Boundary rules

Keep code in `fitsmgmt` when it can answer questions about files, HDUs, headers,
CCDData containers, image sections, pixel arrays as arrays, or WCS metadata
without making science decisions about sources.

Move code to `fitsimred` when it transforms one or more images into calibrated
or combined images, creates/rejects masks as part of calibration, or records
reduction history.

Move code to `fitsimphot` when it measures sources, estimates sky around
sources, fits centroids/PSFs, builds apertures, computes flux/magnitude/error,
or computes polarization.

Keep visualization small and optional. `ysvisutilpy/astro.py` fits naturally
inside `fitsmgmt.viz` if `fitsmgmt` is the user-facing umbrella. Heavier
Matplotlib/statistical diagnostic helpers should be optional extras or a
separate `astroviz` package if they grow.

WCS belongs in `fitsmgmt` only up to metadata: parse WCS, pixel scale, image
center, footprint/radius, rotation, and removing WCS keywords. Coordinate
science such as catalog matching, ephemerides, FOV membership for moving
objects, or source association belongs outside `fitsmgmt`.

## Naming

`fitsmgmt` is the low-level FITS management package for I/O, headers, WCS
metadata, summaries, and light visualization. It should not absorb full
reduction or photometry workflows.

Use `fitsimred` for the image-in/image-out reduction layer, where users will
expect calibration pipelines and combination tools.

`fitsimphot` owns local imaging photometry and source-measurement utilities.
It may still support `CCDData`, arrays, tables, and non-FITS intermediates,
but this wave keeps the FITS-imaging package family naming.

`astroviz` is preferable to `fitsviz` if it includes generic Matplotlib,
statistics, or catalog visualizations. Use `fitsmgmt.viz` for compact FITS image
inspection helpers before creating a separate package.

## Practical migration order

1. Keep this repo as `fitsmgmt` and harden the low-level FITS/CCDData layer.
2. Move `ysfitsutilpy/imutil` and reduction-specific `preproc.py` code into a
   separate `fitsimred` package, not into the low-level base.
3. Move `ysphotutilpy` local image-analysis code into `fitsimphot`; keep
   `queryutil.py` optional or separate.
4. Port only `ysvisutilpy/astro.py` into `fitsmgmt.viz` initially. Defer the rest
   unless plotting becomes a real supported product surface.
