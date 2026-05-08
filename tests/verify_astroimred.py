import os
import shutil
import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits

# Import astroimred
try:
    from astroimred import ccdutils, headers, io, mathutils, misc, paths, summary
    from astroimred import logging as airlogging
except ImportError:
    # Use direct path if package not installed in env yet
    import sys

    sys.path.insert(0, os.path.abspath("src"))

    from astroimred import ccdutils, headers, io, mathutils, misc, paths, summary
    from astroimred import logging as airlogging

logger = airlogging.logger


def run_tests():
    airlogging.enable_console_logging()
    logger.info("Starting Verification of astroimred...")

    # Setup temp dir
    tmpdir = Path(tempfile.mkdtemp())
    logger.info("Using temp dir: %s", tmpdir)

    try:
        # ==========================================
        # 1. Test Logging
        # ==========================================
        logger.info("--- Testing Logging ---")
        airlogging.set_log_level("DEBUG")
        airlogging.enable_console_logging(level=10)

        # ==========================================
        # 2. Test Utils
        # ==========================================
        logger.info("--- Testing Utils ---")

        # listify
        assert misc.listify(1) == [1]
        assert misc.listify([1, 2]) == [1, 2]
        assert misc.listify("abc") == ["abc"]

        # str_now
        assert len(misc.str_now()) > 0

        # change_to_quantity
        q1 = misc.change_to_quantity(10, "km")
        assert q1.value == 10.0 and q1.unit == u.km
        q2 = misc.change_to_quantity(10 * u.m, "km")
        assert q2.value == 0.01 and q2.unit == u.km

        # binning
        arr = np.arange(16).reshape(4, 4)
        binned = mathutils.binning(arr, factors=(2, 2))
        expected_bin = np.array([[2.5, 4.5], [10.5, 12.5]])
        assert np.allclose(binned, expected_bin)

        # Header utils
        hdr = fits.Header()
        hdr["NAXIS"] = 2

        # cmt2hdr
        headers.cmt2hdr(hdr, "h", "Test history")
        assert "Test history" in str(hdr.get("HISTORY"))

        # update_process
        headers.update_process(hdr, "BiasSub")
        assert "BiasSub" in str(hdr.get("PROCESS"))

        # update_tlm
        headers.update_tlm(hdr)
        assert "FITS-TLM" in hdr

        # ==========================================
        # 3. Test Images
        # ==========================================
        logger.info("--- Testing Images ---")

        # Setup dummy FITS file
        data = np.zeros((10, 10))
        data[2:5, 2:5] = 100
        hdu = fits.PrimaryHDU(data=data, header=hdr)
        fpath = tmpdir / "test.fits"
        hdu.writeto(fpath)

        # load_ccd
        ccd = io.load_ccd(fpath)
        assert ccd.shape == (10, 10)

        # inputs2list
        inputs = io.inputs2list(str(tmpdir / "*.fits"))
        assert [Path(p).name for p in inputs] == ["test.fits"]

        # imslice
        sl_ccd = ccdutils.imslice(ccd, "[2:5, 2:5]")
        assert sl_ccd.shape == (4, 4)

        # cut_ccd
        # cut_ccd returns (nccd, cutout)
        cut, _ = ccdutils.cut_ccd(ccd, (5, 5), (4, 4))
        assert cut.shape == (4, 4)

        # bin_ccd
        binccd = ccdutils.bin_ccd(ccd, factors=(2, 2))
        assert binccd.shape == (5, 5)
        assert "XBINNING" in binccd.header
        assert "YBINNING" in binccd.header

        # hedit
        headers.hedit(fpath, "OBJECT", "TestObj", overwrite=True, add=True)
        assert fits.getval(fpath, "OBJECT") == "TestObj"

        # key_remover
        h2 = hdr.copy()
        h2["TEMP"] = 123
        h2 = headers.key_remover(h2, ["TEMP"])
        assert "TEMP" not in h2

        # set_ccd_attribute
        ccdutils.set_ccd_attribute(ccd, "gain", 2.0, unit="electron/adu")
        assert ccd.gain.value == 2.0
        assert ccd.gain.unit == u.electron / u.adu

        # write2fits
        outpath = tmpdir / "out.fits"
        io.write2fits(data, hdr, outpath)
        assert outpath.exists()

        # ==========================================
        # 4. Test Files
        # ==========================================
        logger.info("--- Testing Files ---")

        # mkdir
        dpath = tmpdir / "subdir"
        paths.mkdir(dpath)
        assert dpath.exists()

        # fits_summary
        df = summary.fits_summary([fpath, outpath], keywords=["OBJECT", "NAXIS"])
        df = df.sort_values("file").reset_index(drop=True)
        # out.fits is first (alphabetical o before t? No, fpath=test.fits,
        # outpath=out.fits)
        # test.fits (fpath) and out.fits (outpath)
        # 'out.fits' < 'test.fits'.
        # out.fits has OBJECT=TestObj? No, outpath was written from `hdr` which
        # was created BEFORE hedit on fpath.
        # But `hdr` object is updated inplace by `cmt2hdr` etc. But `hedit`
        # updated `fpath` (file on disk).
        # `hedit` was on `fpath`. `hdr` variable in memory might not reflect
        # `fpath` updates unless reloaded.
        # `write2fits` used `hdr` and `data`.
        # `hdr` has `HISTORY` and `PROCESS` and `FITS-TLM`.
        # `fpath` has `OBJECT=TestObj` because of `hedit`.
        # `outpath` was written using `data` and `hdr` (memory). It doesn't
        # have `OBJECT=TestObj` unless `hdr` had it.
        # `hdr` didn't have `OBJECT` set. `hedit` modified `fpath` on disk.
        # So `out.fits` (from `hdr`) has no `OBJECT`.
        # `test.fits` (on disk) has `OBJECT=TestObj`.
        # df sorted: out.fits (idx 0), test.fits (idx 1).

        # Actually verify content carefully.
        # summary table logic loads from disk.
        # So df[0] (out.fits) -> OBJECT=None (or whatever default)
        # df[1] (test.fits) -> OBJECT=TestObj

        logger.info("Summary DF:\n%s", df)
        assert str(df.iloc[1]["file"]).endswith("test.fits")
        assert df.iloc[1]["OBJECT"] == "TestObj"

    except Exception:
        shutil.rmtree(tmpdir)
        raise
    else:
        shutil.rmtree(tmpdir)
        logger.info("Verification Finished Successfully.")


if __name__ == "__main__":
    run_tests()
