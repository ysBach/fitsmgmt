# FITS Package Reorganization Plan

## Summary

This document coordinates the first-wave port into:

- `fitsmgmt` in `/Users/ysbach/Dropbox/github/fitsmgmt`
- `fitsimred` in `/Users/ysbach/Dropbox/github/fitsimred`
- `fitsimphot` in `/Users/ysbach/Dropbox/github/fitsimphot`

All packages target Python `>=3.9`. `fitsimspec` is deferred.

Preferred import aliases:

- `import fitsmgmt as fm`
- `import fitsimred as fir`
- `import fitsimphot as fip`

The first commit should preserve the raw, archaic legacy code style as much as
practical: keep original comments, docstrings, TODOs, notes, historical
explanations, and naming unless a minimal change is required for package
boundaries, imports, Python 3.9 compatibility, or tests.

## Package Boundaries

### `fitsmgmt`

Owns FITS I/O, HDU/CCDData parsing, headers, file summaries, WCS metadata,
airmass/observation metadata helpers, and lightweight FITS image visualization.

Ported from:

- `ysfitsutilpy/hduutil.py` -> `fitsmgmt.hduutil`
- `ysfitsutilpy/filemgmt.py` -> `fitsmgmt.filemgmt`
- `ysfitsutilpy/misc.py` -> `fitsmgmt.misc`
- `ysfitsutilpy/airmass.py` -> `fitsmgmt.airmass`
- `ysvisutilpy/astro.py` -> `fitsmgmt.viz`

Compatibility aliases retained for the disposable old skeleton:

- `fitsmgmt.images` -> `fitsmgmt.hduutil`
- `fitsmgmt.files` -> `fitsmgmt.filemgmt`
- `fitsmgmt.utils` -> `fitsmgmt.misc`
- `fitsmgmt.wcstools` -> WCS helpers from `fitsmgmt.hduutil`

Deferred from `fitsmgmt`:

- `ysfitsutilpy/imutil/`, `combutil.py`, `preproc.py`: belongs to `fitsimred`.
- `ysphotutilpy/*`: belongs to `fitsimphot`.
- `ysfitsutilpy/astrometry.py`: defer or optional extra.
- `ysvisutilpy/stats.py`: defer.

### `fitsimred`

Owns image-in/image-out reduction. Port near-verbatim from `ysfitsutilpy`:

- `preproc.py`
- `combutil.py`
- full `imutil/`: `imcombine.py`, `imarith.py`, `imcopy.py`, `imsmooth.py`,
  `util_comb.py`, `util_reject.py`, `util_lmedian.py`, `docstrings.py`,
  `config.py`, `numba_combine.py`, `numba_reject.py`

`fitsimred` depends on `fitsmgmt` and must not import `fitsimphot`.
Use `import fitsimred as fir` in new examples, docstrings, and user-facing
comments.

### `fitsimphot`

Owns imaging photometry/source measurement. Port near-verbatim from
`ysphotutilpy`:

- `aperture.py`, `aputil.py`, `apphot.py`, `background.py`, `center.py`,
  `seputil.py`, `radprof.py`, `growth.py`, `daopsf.py`, `polarimetry.py`,
  `util.py`

Defer `queryutil.py`, `query_cols.py`, and `smallbody.py` unless implemented
later as an optional catalog/network extra.
Use `import fitsimphot as fip` in new examples, docstrings, and user-facing
comments.

## Implementation Rules

- Prefer direct copy-and-paste over cleanup.
- Preserve legacy comments and docstrings as much as possible.
- Avoid stylistic modernization, type-hinting, renaming, typo fixes, and bug
  fixes unless necessary for importability, Python 3.9 syntax, package
  boundaries, or test execution.
- Do not port legacy broad package-level star imports directly if they pull in
  unrelated heavy dependencies.
- Fix only mechanical issues caused by the split: imports, package names,
  optional dependencies, broken `__all__`, Python 3.9 syntax, and missing files.
- Current `fitsmgmt` files are not authoritative and may be replaced.

## Test Plan

`fitsmgmt`:

- Port/expand tests for `hduutil`, `filemgmt`, `misc`, and `airmass`.
- Cover `make_summary`, `inputs2list`, `load_ccd`, `write2fits`, extension
  parsing, header edits, WCS helpers, FITS slicing, path/glob behavior, and CLI
  summary output.
- Confirm base import does not require photometry, reduction, network, or heavy
  plotting dependencies.

`fitsimred`:

- Port `ysfitsutilpy` tests for `imcombine`, `imcombine_regression`, `preproc`,
  and relevant combination helpers.
- Preserve regression fixtures under `ysfitsutilpy/tests/data`.

`fitsimphot`:

- Port `ysphotutilpy` tests for aperture geometry, photometry, sky/background,
  centroiding, SEP, radial profiles, polarimetry, and util helpers.
- Mock or skip network/catalog behavior.

All packages:

- Run Python 3.9 import smoke tests.
- Run editable-install checks and `pip check` where feasible.

## Prompts For Other Threads

Prompt for `fitsimred`:

```text
Load skill ysfitsutilpy. Work in /Users/ysbach/Dropbox/github/fitsimred. Read /Users/ysbach/Dropbox/github/fitsmgmt/docs/reorganization-plan.md first. Do not edit fitsmgmt.

Target Python >=3.9. This is the image-in/image-out reduction package. Spawn at least three subagents before editing: senior architect, code-reviewer, and tester/user advocate.

Port near-verbatim from ysfitsutilpy: preproc.py, combutil.py, and the full imutil package. Preserve original comments, docstrings, TODOs, and archaic style as much as possible. Only change imports/package names, Python 3.9 syntax issues, dependency boundaries, and tests needed for the package split.

Replace ysfitsutilpy base helper imports with fitsmgmt equivalents. Do not import fitsimphot. Port focused tests and regression fixtures, especially imcombine regression and preproc tests.
```

Prompt for `fitsimphot`:

```text
Load skill ysphotutilpy. Work in /Users/ysbach/Dropbox/github/fitsimphot. Read /Users/ysbach/Dropbox/github/fitsmgmt/docs/reorganization-plan.md first. Do not edit fitsmgmt or fitsimred.

Target Python >=3.9. This is the local imaging photometry/source-measurement package. Spawn at least three subagents before editing: senior architect, code-reviewer, and tester/user advocate.

Port near-verbatim from ysphotutilpy: aperture.py, aputil.py, apphot.py, background.py, center.py, seputil.py, radprof.py, growth.py, daopsf.py, polarimetry.py, util.py, and logging.py if needed. Preserve original comments, docstrings, TODOs, and archaic style as much as possible. Only change imports/package names, Python 3.9 syntax issues, dependency boundaries, and tests needed for the package split.

Keep photutils >=2,<3 explicit. Defer queryutil.py, query_cols.py, and smallbody.py. Port focused tests from ysphotutilpy/tests and avoid live network tests.
```

## Assumptions

- `fitsmgmt`, `fitsimred`, and `fitsimphot` are the final package names for
  this wave.
- Existing `fitsmgmt` structure is not authoritative and can be replaced.
- Initial commits should represent raw reorganization, not cleanup.
- Legacy repo dirty changes are user-owned source state and must not be
  reverted.
- `fitsimspec` receives no implementation in this wave.
