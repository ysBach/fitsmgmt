import logging
import shutil
import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits

from astroimred import (
    ccdutils,
    headers,
    io,
    logging as airlogging,
    mathutils,
    misc,
    summary,
)


def test_canonical_subpackages_match_compatibility_wrappers():
    """Canonical subpackages and top-level wrappers expose the same objects."""
    import astroimred as air
    import astroimred.ccdutils as ccdutils_module
    import astroimred.io as io_module

    assert air.mgmt.io.load_ccd is io.load_ccd
    assert io_module is air.mgmt.io
    assert air.mgmt.headers.cmt2hdr is headers.cmt2hdr
    assert air.mgmt.summary.fits_summary is summary.fits_summary
    assert air.imops.ccdutils.bin_ccd is ccdutils.bin_ccd
    assert ccdutils_module is air.imops.ccdutils
    assert air.imops.mathutils.binning is mathutils.binning
    assert air.imops.imstat.give_stats is air.imstat.give_stats
    assert air.imops.pixels.fixpix is air.pixels.fixpix
    assert air.logger is air.mgmt.logger


@pytest.fixture
def temp_env():
    """Fixture to provide a clean temp directory."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir)

@pytest.fixture
def dummy_fits(temp_env):
    """Fixture to create a dummy FITS file."""
    hdr = fits.Header()
    hdr['NAXIS'] = 2
    hdr['EXPTIME'] = 10.0
    data = np.zeros((10, 10))
    data[2:5, 2:5] = 100
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    fpath = temp_env / "test.fits"
    hdu.writeto(fpath)
    return fpath

def test_logging():
    """Test logging configuration."""
    airlogging.set_log_level("DEBUG")
    airlogging.enable_console_logging(level=10)
    # Since we can't easily capture logger output configured to stdout in pytest
    # without caplog, we just verify the function runs and level is set.
    assert airlogging.logger.level == logging.DEBUG
    for handler in airlogging.logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            airlogging.logger.removeHandler(handler)

def test_listify():
    """Test listify utility."""
    assert misc.listify(1) == [1]
    assert misc.listify([1, 2]) == [1, 2]
    assert misc.listify("abc") == ["abc"]

def test_str_now():
    """Test str_now."""
    assert len(misc.str_now()) > 0

def test_change_to_quantity():
    """Test quantity conversion."""
    q1 = misc.change_to_quantity(10, "km")
    assert q1.value == 10.0 and q1.unit == u.km
    q2 = misc.change_to_quantity(10*u.m, "km")
    assert q2.value == 0.01 and q2.unit == u.km

def test_binning():
    """Test array binning."""
    arr = np.arange(16).reshape(4, 4)
    binned = mathutils.binning(arr, factors=(2, 2))
    expected_bin = np.array([[2.5, 4.5], [10.5, 12.5]])
    assert np.allclose(binned, expected_bin)

def test_header_utils(dummy_fits):
    """Test header utilities."""
    hdr = fits.getheader(dummy_fits)

    # cmt2hdr
    headers.cmt2hdr(hdr, 'h', "Test history")
    assert "Test history" in str(hdr.get("HISTORY"))

    # update_process
    headers.update_process(hdr, "BiasSub")
    assert "BiasSub" in str(hdr.get("PROCESS"))

    # update_tlm
    headers.update_tlm(hdr)
    assert "FITS-TLM" in hdr

def test_images_io(dummy_fits):
    """Test image loading and saving."""
    ccd = io.load_ccd(dummy_fits)
    assert ccd.shape == (10, 10)

    # Test inputs2list
    inputs = io.inputs2list(str(dummy_fits.parent / "*.fits"))
    assert [Path(p).name for p in inputs] == ['test.fits']

    # Test write2fits
    outpath = dummy_fits.parent / "out.fits"
    io.write2fits(ccd.data, ccd.header, outpath)
    assert outpath.exists()

def test_image_process(dummy_fits):
    """Test image processing."""
    ccd = io.load_ccd(dummy_fits)

    # imslice
    sl_ccd = ccdutils.imslice(ccd, "[2:5, 2:5]")
    assert sl_ccd.shape == (4, 4)

    # cut_ccd
    cut, _ = ccdutils.cut_ccd(ccd, (5, 5), (4, 4))
    assert cut.shape == (4, 4)

    # bin_ccd
    binccd = ccdutils.bin_ccd(ccd, factors=(2, 2))
    assert binccd.shape == (5, 5)
    assert "XBINNING" in binccd.header
    assert "YBINNING" in binccd.header

def test_header_edits(dummy_fits):
    """Test header edits via headers module."""
    # hedit
    headers.hedit(
        dummy_fits, "OBJECT", "TestObj", overwrite=True, add=True, output=dummy_fits
    )
    assert fits.getval(dummy_fits, "OBJECT") == "TestObj"

    # key_remover
    hdr = fits.getheader(dummy_fits)
    hdr['TEMP'] = 123
    hdr = headers.key_remover(hdr, ['TEMP'])
    assert "TEMP" not in hdr

def test_ccd_attributes(dummy_fits):
    """Test CCDData attribute setting."""
    ccd = io.load_ccd(dummy_fits)
    ccdutils.set_ccd_attribute(ccd, 'gain', 2.0, unit='electron/adu')
    assert ccd.gain.value == 2.0
    assert ccd.gain.unit == u.electron / u.adu

def test_files_summary(dummy_fits):
    """Test summary generation."""
    # Create another file for variety
    outpath = dummy_fits.parent / "out.fits"
    headers.hedit(
        dummy_fits, "OBJECT", "TestObj", overwrite=True, add=True, output=dummy_fits
    )
    io.write2fits(np.zeros((10,10)), fits.Header(), outpath)

    df = summary.fits_summary([dummy_fits, outpath], keywords=['OBJECT', 'NAXIS'])
    df = df.sort_values('file').reset_index(drop=True)

    # out.fits (no object)
    assert df.iloc[0]['OBJECT'] is None
    # test.fits (object=TestObj)
    assert df.iloc[1]['OBJECT'] == "TestObj"
