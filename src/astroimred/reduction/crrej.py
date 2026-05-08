"""Cosmic-ray rejection helpers."""

import astroscrappy
import numpy as np
from astro_ndslice import is_list_like, listify, slicefy
from astropy import units as u
from astropy.nddata import CCDData
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astroscrappy import detect_cosmics

from astroimred.mgmt.headers import cmt2hdr, update_process, update_tlm
from astroimred.imops.ccdutils import propagate_ccdmask
from astroimred.mgmt.io import _parse_image
from astroimred.mgmt.logging import logger
from astroimred.mgmt.misc import change_to_quantity

__all__ = [
    "ASTROSCRAPPY_DIVFACTOR",
    "LACOSMIC_KEYS",
    "LACOSMIC_CRREJ",
    "parse_crrej_psf",
    "crrej",
    "medfilt_bpm",
]


# I skipped two params in IRAF LACOSMIC: gain=2.0, readnoise=6.
LACOSMIC_KEYS = {
    "sigclip": 4.5,
    "sigfrac": 0.5,
    "objlim": 1.0,
    "satlevel": np.inf,
    "invar": None,
    "inbkg": None,
    "niter": 4,
    "sepmed": False,
    "cleantype": "medmask",
    "fsmode": "median",
    "psfmodel": "gauss",
    "psffwhm": 2.5,
    "psfsize": 7,
    "psfk": None,
    "psfbeta": 4.765,
}

# same as above, but simplify `fsmode`, `psfmodel`, and `psfk` into `fs`
LACOSMIC_CRREJ = {
    "sigclip": 4.5,
    "sigfrac": 0.5,
    "objlim": 1.0,
    "satlevel": np.inf,
    "invar": None,
    "inbkg": None,
    "niter": 4,
    "sepmed": False,
    "cleantype": "medmask",
    "fs": "median",
    "psffwhm": 2.5,
    "psfsize": 7,
    "psfbeta": 4.765,
}


ASTROSCRAPPY_DIVFACTOR = detect_cosmics(np.ones((3, 3)), gain=1.0, niter=0)[1][0, 0]
# astroscrappy used to return the data in e- unit, but suddenly changed around
# version 1.1.0.... Jeez...
# https://github.com/astropy/astroscrappy/issues/73
# %timeit detect_cosmics(np.ones((3, 3)), gain=1., niter=0)[1][0, 0] == 1.
# ^ takes ~ 40 us VS ~ 1800 us on MBP 15" [2018, macOS 11.6 i7-8850H (2.6 GHz;
#   6-core), RAM 16 GB (2400MHz DDR4), Radeon Pro 560X (4GB)] VS MBP 14" [2021,
#   macOS 12.2.1, M1 6c+2c, 32G, GPU 14c]. I dunno why they differ so much...
#   Does not change much w.r.t. gain, shape, etc. Maybe cuz I'm using Rosetta2
#   (Anaconda) on the latter?
#   2021-12-13 13:27:35 (KST: GMT+09:00) ysBach
#   2022-04-02 14:58:44 (KST: GMT+09:00) ysBach


def parse_crrej_psf(
    fs="median", psffwhm=2.5, psfsize=7, psfbeta=4.765, fill_with_none=True
):
    """Translate a compact fine-structure spec into ``detect_cosmics`` kwargs.

    Parameters
    ----------
    fs : str, ndarray, or list-like, optional
        Fine-structure model specification.

        - ``"median"`` maps to ``fsmode="median"``.
        - ``"gauss"``, ``"gaussx"``, ``"gaussy"``, and ``"moffat"`` map to
          ``fsmode="convolve"`` with the corresponding ``psfmodel``.
        - A `~numpy.ndarray` is treated as a user-provided convolution kernel
          and maps to ``fsmode="convolve", psfk=fs``.
        - A list-like value may mix strings and kernels. It must not be a
          single 2-D kernel array.
    psffwhm, psfsize, psfbeta : float, int, or list-like, optional
        PSF parameters passed through for convolved Gaussian or Moffat models.
        If any of these or ``fs`` is list-like, all list-like inputs must have
        length 1 or the same maximum length.
    fill_with_none : bool, optional
        When expanding list-like inputs, fill unused ``detect_cosmics`` PSF
        keywords with `None` if `True`; otherwise fill them with
        ``LACOSMIC_KEYS`` defaults. Scalar inputs always return only the
        minimal keyword set.

    Returns
    -------
    dict
        Keyword arguments for `astroscrappy.detect_cosmics`.

    Examples
    --------
    ``parse_crrej_psf()`` returns ``{"fsmode": "median"}``.

    ``parse_crrej_psf("gauss", psffwhm=2, psfsize=3)`` returns
    ``{"fsmode": "convolve", "psfmodel": "gauss", "psffwhm": 2,
    "psfsize": 3}``.
    """
    if is_list_like(psffwhm, psfsize, psfbeta, func=any) or (
        is_list_like(fs) and not isinstance(fs, np.ndarray)
    ):
        fs = listify(fs)
        psffwhm = listify(psffwhm)
        psfsize = listify(psfsize)
        psfbeta = listify(psfbeta)
        lengths = (len(fs), len(psffwhm), len(psfsize), len(psfbeta))
        length = max(lengths)
        if not all(_len in [1, length] for _len in lengths):
            raise ValueError(
                "`fs`, `psffwhm`, `psfsize`, and `psfbeta` must all be "
                f"length 1 or the same length (current maxlength = {length})."
            )

        fs = fs * length if len(fs) == 1 else fs
        psffwhm = psffwhm * length if len(psffwhm) == 1 else psffwhm
        psfsize = psfsize * length if len(psfsize) == 1 else psfsize
        psfbeta = psfbeta * length if len(psfbeta) == 1 else psfbeta

        def _allocate(_fs, _psffwhm, _psfsize, _psfbeta):
            if isinstance(_fs, str):
                if _fs == "median":
                    fsmode = "median"
                    psfmodel = None if fill_with_none else LACOSMIC_KEYS["psfmodel"]
                    psfk = None  # anyway, default in LACOSMIC_KEYS is `None`
                    psffwhm = None if fill_with_none else LACOSMIC_KEYS["psffwhm"]
                    psfsize = None if fill_with_none else LACOSMIC_KEYS["psfsize"]
                    psfbeta = None if fill_with_none else LACOSMIC_KEYS["psfbeta"]
                elif _fs == "moffat":
                    fsmode = "convolve"
                    psfmodel = "moffat"
                    psfk = None
                    psffwhm = _psffwhm
                    psfsize = _psfsize
                    psfbeta = _psfbeta
                elif _fs in ("gauss", "gaussx", "gaussy"):
                    fsmode = "convolve"
                    psfmodel = _fs
                    psfk = None
                    psffwhm = _psffwhm
                    psfsize = _psfsize
                    psfbeta = None if fill_with_none else LACOSMIC_KEYS["psfbeta"]
            elif isinstance(_fs, np.ndarray):
                fsmode = "convolve"
                psfmodel = None if fill_with_none else LACOSMIC_KEYS["psfmodel"]
                psfk = _fs
                psffwhm = None if fill_with_none else LACOSMIC_KEYS["psffwhm"]
                psfsize = None if fill_with_none else LACOSMIC_KEYS["psfsize"]
                psfbeta = None if fill_with_none else LACOSMIC_KEYS["psfbeta"]
            else:
                raise ValueError(f"fs ({fs}) not understood")
            return fsmode, psfmodel, psfk, psffwhm, psfsize, psfbeta

        res = dict(fsmode=[], psfmodel=[], psfk=[], psffwhm=[], psfsize=[], psfbeta=[])
        for _fs, _psffwhm, _psfsize, _psfbeta in zip(fs, psffwhm, psfsize, psfbeta):
            fsmode, psfmodel, psfk, psffwhm, psfsize, psfbeta = _allocate(
                _fs, _psffwhm, _psfsize, _psfbeta
            )
            res["fsmode"].append(fsmode)
            res["psfmodel"].append(psfmodel)
            res["psfk"].append(psfk)
            res["psffwhm"].append(psffwhm)
            res["psfsize"].append(psfsize)
            res["psfbeta"].append(psfbeta)
        return res

    elif isinstance(fs, np.ndarray):
        return dict(fsmode="convolve", psfk=fs)
    elif isinstance(fs, str):
        if fs == "median":
            return dict(fsmode=fs)
        elif fs == "moffat":
            return dict(
                fsmode="convolve",
                psfmodel="moffat",
                psffwhm=psffwhm,
                psfsize=psfsize,
                psfbeta=psfbeta,
            )
        elif fs in ["gauss", "gaussx", "gaussy"]:
            return dict(
                fsmode="convolve", psfmodel=fs, psffwhm=psffwhm, psfsize=psfsize
            )
    else:
        raise ValueError(f"fs ({fs}) not understood")


def crrej(
    ccd,
    mask=None,
    inbkg=None,
    invar=None,
    propagate_crmask=False,
    update_header=True,
    add_process=True,
    gain=None,
    rdnoise=None,
    sigclip=4.5,
    sigfrac=0.5,
    objlim=1.0,
    satlevel=np.inf,
    niter=4,
    sepmed=False,
    cleantype="medmask",
    fs="median",
    psffwhm=2.5,
    psfsize=7,
    psfbeta=4.765,
    verbose=True,
):
    """Do cosmic-ray rejection using L.A.Cosmic default parameters.

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The ccd to be processed. The data must be in ADU, not electrons.

    propagate_crmask : `bool`, optional.
        Whether to save (propagate) the mask from CR rejection
        (`~astroscrappy`) to the CCD's mask. Default is `False`.

    inbkg : `float`, `~numpy.ndarray`, path-like to FITS, optional
        A pre-determined background image, to be subtracted from `ccd` before
        running the main detection algorithm. This is used primarily with
        spectroscopic data, to remove sky lines and the cross-section of an
        object continuum during iteration, "protecting" them from spurious
        rejection (see the above paper). This background is not removed from
        the final, cleaned output (`cleanarr`). This should be in units of
        "counts", the same units of `ccd`. `inbkg` should be free from cosmic
        rays. When estimating the cosmic-ray free noise of the image, we will
        treat `inbkg` as a constant Poisson contribution to the variance.
        Default: `None`.

        .. note::
            Originally pssl, which stood for "previously subtracted sky
            level" in ADU (in astroscrappy < 1.1.0 or original L.A.Cosmic).
            Since astroscrappy ver > 1.1.0, a 2-D sky level is supported by
            inbkg (it was bkg in == 1.1.0, which is a hasty bug in argument
            naming).

    invar : `float` numpy array, path-like to FITS, optional
        A pre-determined estimate of the data variance (ie. noise squared) in
        each pixel, generated by previous processing of `ccd`. If provided,
        this is used in place of an internal noise model based on `ccd`, `gain`
        and `readnoise`. This still gets median filtered and cleaned
        internally, to estimate what the noise in each pixel *would* be in the
        absence of cosmic rays. This should be in units of "counts" squared.
        (it was `var` in == 1.1.0, which is a hasty bug in argument naming)

    gain, rdnoise : `None`, `float`, astropy.`~astropy.units.Quantity`, optional.
        The gain and readnoise value. If not ``Quantity``, they must be in
        electrons per adu and electron unit, respectively.

    sigclip : `float`, optional
        Laplacian-to-noise limit for cosmic ray detection. Lower values will
        flag more pixels as cosmic rays.
        Default: 4.5.

    sigfrac : `float`, optional
        Fractional detection limit for neighboring pixels. For cosmic ray
        neighbor pixels, a lapacian-to-noise detection limit of sigfrac *
        sigclip will be used.
        Default: 0.5.

    objlim : `float`, optional
        Minimum contrast between Laplacian image and the fine structure image.
        Increase this value if cores of bright stars are flagged as cosmic
        rays.
        Default: 1.0.

    satlevel : `float`, optional
        Saturation of level of the image (electrons). This value is used to
        detect saturated stars and pixels at or above this level are added to
        the mask.
        Default: ``np.inf``.

    niter : `int`, optional
        Number of iterations of the LA Cosmic algorithm to perform.
        Default: 4.

    sepmed : boolean, optional
        Use the separable median filter instead of the full median filter. The
        separable median is not identical to the full median filter, but they
        are approximately the same and the separable median filter is
        significantly faster and still detects cosmic rays well.
        Default: `True`

    cleantype : ``{'median', 'medmask', 'meanmask', 'idw'}``, optional
        Set which clean algorithm is used:

        * ``'median'``: An umasked 5x5 median filter
        * ``'medmask'``: A masked 5x5 median filter
        * ``'meanmask'``: A masked 5x5 mean filter
        * ``'idw'``: A masked 5x5 inverse distance weighted interpolation

        Default: ``"medmask"``.

    fs : ``{'median', 'gauss', 'gaussx', 'gaussy', 'moffat'}``, `~numpy.ndarray`, `None`, optional.
        Method to generate the fine structure. Combination of `fsmode`,
        `psfmodel`, `psfk` of `astroscrappy`.

        * ``'median'``: Use the median filter in the standard LA Cosmic
          algorithm. `None` of `psffwhm`, `psfsize`, and `psfbeta` are used.
        * other `str`: Use a Gaussian/Moffat model to generate the psf kernel.
          ``'gauss'|'moffat'`` produce circular PSF kernels.
          ``'gaussx'|'gaussy'`` produce Gaussian kernels in the x and y
          directions respectively. `psffwhm`, `psfsize` (plus `psfbeta` if
          "moffat") are used.
        * `~numpy.ndarray`: PSF kernel array to use for the fine structure image. `None`
          of `psffwhm`, `psfsize`, and `psfbeta` are used.

        Mapping between `astroscrappy` and ``imred`` options:

        * ``fsmode="median"`` == ``fs="median"``
        * ``fsmode="convolve", psfmodel=*`` == ``fs=*``, where ``*`` can be
          any of ``{'gauss', 'gaussx', 'gaussy', 'moffat'}``.
        * ``fsmode="convolve", psfk=<`~numpy.ndarray`>`` == ``fs=<`~numpy.ndarray`>``

        If `None`, CR rejection will not happen and copy of input `ccd` will be
        returned.

        Default: ``'median'``.

    psffwhm : `float`, optional
        Full Width Half Maximum of the PSF to use to generate the kernel.
        Default: 2.5.

    psfsize : `int`, optional
        Size of the kernel to calculate. Returned kernel will have size `psfsize`
        x `psfsize`. It should be an odd integer.
        Default: 7.

    psfbeta : `float`, optional
        Moffat beta parameter. Only used if ``fs='moffat'``.
        Default: 4.765.

    verbose : boolean, optional
        Print to the screen or not. Default: `False`.

    Returns
    -------
    _ccd : `~astropy.nddata.CCDData`
        The cosmic-ray cleaned `~astropy.nddata.CCDData` in ADU. `~astroscrappy` automatically
        does a gain correction, so I divided the `~astroscrappy` result by
        gain to restore to ADU (not to surprise the users).

    crmask : `~numpy.ndarray` (mask)
        The cosmic-ray mask from `~astroscrappy`, propagated by the original
        mask of the ccd (if ``ccd.mask`` is not `None`) and `mask` given by
        the user.

    update_header : `bool`, optional.
        Whether to update the header if there is any.

    add_process : `bool`, optional.
        Whether to add ``PROCESS`` key to the header.

    Notes
    -----
    Important detection parameters are `sigclip`, `sigfrac`, and `objlim`.
    Fine-structure parameters are mapped through `fs`, `psffwhm`, `psfsize`,
    `psfk`, and `psfbeta`. Detector-specific parameters are `gain` and
    `rdnoise`.

    (Note from `astroscrappy`)
    For best results on spectra, we recommend that you include an estimate of
    the background. One can generally obtain this by fitting columns with a
    smooth function. To efficiently identify cosmic rays, LA Cosmic and
    therefore astroscrappy estimates the cosmic ray free noise by smoothing the
    variance using a median filter. To minimize false positives on bright sky
    lines, if `inbkg` is provided, we do not smooth the variance contribution
    from the provided background. We only smooth the variance that is in
    addition to the Poisson contribution from the background so that we do not
    underestimate the noise (and therefore run the risk of flagging false
    positives) near narrow, bright sky lines.

    All defaults are based on the IRAF version of L.A. Cosmic. The default
    parameters differ between L.A. Cosmic versions, so these follow the IRAF
    version written by van Dokkum.
    """
    if fs is None:
        return ccd.copy(), None

    str_cr = (
        "Cosmic-Ray rejection (CRNFIX={:d} pixels fixed) by astroscrappy (v {}). "
        + "Parameters: {}"
    )

    _t = Time.now()

    if isinstance(gain, str):
        gain = ccd.header.get(gain, None)

    if gain is None:  # If it is still None...
        gain = getattr(ccd, "gain", 1)

    if isinstance(rdnoise, str):
        rdnoise = ccd.header.get(rdnoise, None)

    if rdnoise is None:  # If it is still None...
        rdnoise = getattr(ccd, "rdnoise", 0)

    _ccd = ccd.copy()
    if mask is None:
        inmask = None
    else:
        inmask = _parse_image(mask)[0]
        inmask = propagate_ccdmask(_ccd, additional_mask=inmask)

    # The L.A. Cosmic accepts only the gain in e/adu and rdnoise in e.
    gain = change_to_quantity(gain, u.electron / u.adu, to_value=True)
    rdnoise = change_to_quantity(rdnoise, u.electron, to_value=True)

    inbkg = None if inbkg is None else _parse_image(inbkg)[0]
    invar = None if invar is None else _parse_image(invar)[0]

    # remove the cosmic cosmic rays
    crrej_kwargs = dict(
        gain=gain,
        readnoise=rdnoise,
        sigclip=sigclip,
        sigfrac=sigfrac,
        objlim=objlim,
        satlevel=satlevel,
        niter=niter,
        sepmed=sepmed,
        cleantype=cleantype,
        **parse_crrej_psf(fs=fs, psffwhm=psffwhm, psfsize=psfsize, psfbeta=psfbeta),
    )
    try:
        crmask, cleanarr = detect_cosmics(
            _ccd.data,
            inmask=inmask,
            inbkg=inbkg,
            invar=invar,
            verbose=verbose,
            **crrej_kwargs,
        )
    except TypeError:  # astroscrappy < 1.1.1 (Commit on 2021-11-20) Jeez...
        try:
            crmask, cleanarr = detect_cosmics(
                _ccd.data,
                inmask=inmask,
                bkg=inbkg,
                var=invar,
                verbose=verbose,
                **crrej_kwargs,
            )
        except TypeError:  # astroscrappy < 1.1.0 (Commit on 2020-11-21) Jeez...
            # Error if inbkg is ndarray
            crmask, cleanarr = detect_cosmics(
                _ccd.data,
                inmask=inmask,
                pssl=0 if inbkg is None else inbkg,
                verbose=verbose,
                **crrej_kwargs,
            )

    # create the new ccd data object
    _ccd.data = cleanarr / ASTROSCRAPPY_DIVFACTOR

    if propagate_crmask:
        _ccd.mask = propagate_ccdmask(_ccd, additional_mask=crmask)

    if add_process and _ccd.header is not None:
        update_process(_ccd.header, process="C")

    if update_header and _ccd.header is not None:
        nrej_cr = np.sum(crmask)
        _ccd.header["CRNFIX"] = (nrej_cr, "Number of cosmic-ray pixels fixed.")
        cmt2hdr(
            _ccd.header,
            "h",
            verbose=verbose,
            t_ref=_t,
            s=str_cr.format(nrej_cr, astroscrappy.__version__, crrej_kwargs),
        )
    else:
        if verbose:
            nrej_cr = np.sum(crmask)
            logger.info(str_cr.format(nrej_cr, astroscrappy.__version__, crrej_kwargs))

    update_tlm(_ccd.header)

    return _ccd, crmask


# TODO: put niter
# TODO: put medfilt_min
#   to get std at each pixel by medfilt[<medfilt_min] = 0, and std =
#   sqrt((1+snoise)*medfilt/gain + rdn**2)
def medfilt_bpm(
    ccd,
    cadd=1.0e-10,
    std_model="std",
    gain=1.0,
    rdnoise=0.0,
    snoise=0.0,
    size=5,
    sigclip_kw=None,
    std_section=None,
    footprint=None,
    mode="reflect",
    cval=0.0,
    origin=0,
    med_sub_clip=None,
    med_rat_clip=[0.5, 2],
    std_rat_clip=[-5, 5],
    dtype="float32",
    update_header=True,
    verbose=False,
    logical="and",
    full=False,
):
    """Find bad pixels from median filtering technique (non standard..?)

    Parameters
    ----------
    ccd : `~astropy.nddata.CCDData`
        The CCD to find the bad pixels.

    cadd : `float`, optional.
        A very small const to be added to the input array to avoid resulting
        value of 0.0 in the median filtered image which raises zero-division in
        median ratio ``(image/|median_filtered|)``.

    std_model : ``{"std", "ccd"}``, `~numpy.ndarray`, numeric, optional.
        The model used to calculate the std (standard deviation) map.

        - ``"std"``: Simple standard deviation is calculated.
        - ``"ccd"``: Using CCD noise model (``sqrt{(1 + snoise)*med_filt/gain
          + (rdnoise/gain)**2}``)
        - `ndarray`: A pre-calculated std map to be used as the std map.

        For ``'std'``, the arguments `std_section` and `sigclip_kw` are used,
        while if ``'ccd'``, arguments `gain`, `rdnoise`, `snoise` will be used.

    size, footprint, mode, cval, origin : optional.
        The parameters to obtain the median-filtered map. See
        `~scipy.ndimage.median_filter`.

    sigclip_kw : `dict`, optional.
        The parameters used for `~astropy.stats.sigma_clipped_stats` when
        estimating the sky standard deviation at `std_section`. This is
        **ignored** if ``std_model='ccd'``.
        Default is ``dict(sigma=3.0, maxiters=5, std_ddof=1)``.

    std_section : `str`, optional.
        The region in FITS standard (1-indexing, end-inclusive, xyz order) to
        estimate the sky standard deviation to obtain the `std_ratio`. If
        `None` (default), the full region of the given array is used, which is
        many times not desirable due to the celestial objects in the FOV and
        computational cost. This is **ignored** if ``std_model='ccd'``.

    gain, rdnoise, snoise : `float`, optional.
        The gain (electrons/ADU), readout noise (electrons), and sensitivity
        noise (fractional error from flat fielding) of the frame. These are
        **ignored** if ``std_model="std"``.

    med_sub_clip : `list` of two `float` or `None`, optional.
        The thresholds to find bad pixel by ``med_sub = ccd.data -
        median_filter(ccd.data)``. The clipping will be turned off if it is
        `None` (default). If a `list`, must be in the order of ``[lower, upper]``
        and at most two of these can be `None`.

    med_rat_clip : `list` of two `float` or `None`, optional.
        The thresholds to find bad pixel by ``med_ratio =
        ccd.data/np.abs(median_filter(ccd.data))``. The clipping will be turned
        off if it is `None` (default). If a `list`, must be in the order of
        ``[lower, upper]`` and at most two of these can be `None`.

    std_rat_clip : `list` of two `float` or `None`, optional.
        The thresholds to find bad pixel by ``std_ratio = (ccd -
        median_filter(ccd))/std``. The clipping will be turned off if it is
        `None` (default). If a `list`, must be in the order of ``[lower, upper]``
        and at most two of these can be `None`.

    logical : {'and', '&', 'or', '|'} or `list` of these, optional.
        The logic to propagate masks determined by the ``_clip``'s. The mask is
        propagated such as ``posmask = med_sub > med_sub_clip[1] &/| med_ratio
        > med_rat_clip[1] &/| std_ratio > std_rat_clip[1]``. If a `list`, it must
        contain two `str` of these, in the order of ``[logical_negmask,
        logical_posmask]``.

    Returns
    -------
    ccd : `~astropy.nddata.CCDData`
        The badpixel removed result.

    The followings are returned as `dict` only if ``full=True``.

    posmask, negmask : ndarray of `bool`
        The masked pixels by positive/negative criteria.

    sky_std : `float`
        The (sigma-clipped) sky standard deviation. Returned only if
        ``full=True``.

    Notes
    -----
    ``med_sub_clip`` is usually unnecessary, but can be useful for detecting
    hot pixels in dark frames. The method generates median-subtracted,
    median-ratio, and stddev-ratio maps, combines the positive and negative
    masks with the requested logic, and replaces masked pixels with the median
    filtered frame.

    """
    from scipy.ndimage import median_filter

    if sigclip_kw is None:
        sigclip_kw = dict(sigma=3.0, maxiters=5, std_ddof=1)

    def _sanitize_clips(clips):
        clips = np.atleast_1d(clips)
        if clips.size == 1:
            clips = np.repeat(clips, 2)
        return clips

    if (med_sub_clip is None) and (med_rat_clip is None) and (std_rat_clip is None):
        logger.warning("No BPM is found because all clips are None.")
        if full:
            return ccd, dict(
                posmask=None,
                negmask=None,
                med_filt=None,
                med_sub=None,
                med_rat=None,
                std_rat=None,
                std=None,
            )
        else:
            return ccd

    logical = np.array(logical)
    if logical.size == 1:
        logical = np.repeat(logical, 2)
    elif logical.size > 2:
        raise ValueError("logical must be at most size 2.")

    _LOGICAL_AND = []
    _LOGICAL_STR = []
    for i, _logical in enumerate(logical):
        _logical_and = _logical in ["and", "&"]
        if not _logical_and and _logical not in ["or", "|"]:
            raise ValueError("logical not understood.")

        _LOGICAL_AND.append(_logical_and)
        _LOGICAL_STR.append("and" if _logical_and else "or")

    def _set_masks(arr2test, clips):
        if clips[0] is None:  # let lower clip does not affect final mask
            _negmask = _LOGICAL_AND[0]  # isinstance bool
        else:
            _negmask = arr2test < clips[0]  # isinstance ndarray

        if clips[1] is None:  # let upper clip does not affect final mask
            _posmask = _LOGICAL_AND[1]  # isinstance bool
        else:
            _posmask = arr2test > clips[1]  # isinstance ndarray

        return _negmask, _posmask

    if not isinstance(ccd, CCDData):
        raise TypeError("ccd should be CCDData")

    # Work with a copy of the data, converted to the target dtype
    # Use copy=False in astype to avoid redundant copy when dtype already matches
    arr = ccd.data.astype(dtype, copy=False).copy()  # single copy
    hdr = ccd.header.copy()

    # add very small const to avoid resulting value of 0.0 in med_filt
    # which results in zero-division in med_ratio below.
    arr += cadd

    if std_section is not None:
        slices = slicefy(std_section)
    else:
        slices = [slice(None, None, None)] * arr.ndim

    medfilt_kw = dict(
        size=size, footprint=footprint, mode=mode, cval=cval, origin=origin
    )

    _t = Time.now()
    med_filt = median_filter(arr, **medfilt_kw)

    if update_header:
        cmt2hdr(
            hdr,
            "h",
            verbose=verbose,
            t_ref=_t,
            s=f"Median filtered (convolved) frame calculated with {medfilt_kw}",
        )

    if std_model == "ccd":
        _t = Time.now()
        gain = change_to_quantity(gain, u.electron / u.adu, to_value=True)
        rdnoise = change_to_quantity(rdnoise, u.electron, to_value=True)

        std = np.sqrt((1 + snoise) * med_filt / gain + (rdnoise / gain) ** 2)
        if update_header:
            cmt2hdr(
                hdr,
                "h",
                verbose=verbose,
                t_ref=_t,
                s=(
                    "Stddev map is generated from median filtered frame by "
                    + "sqrt{(1 + snoise)*med_filt/gain + (rdnoise/gain)**2}"
                ),
            )
            hdr["MB_MODEL"] = (std_model, "Method used for getting stdev map")
            hdr["MB_GAIN"] = (gain, "gain used for stdev map in MBPM")
            hdr["MB_RDN"] = (rdnoise, "rdnoise used for stdev map in MBPM")
            hdr["MB_SSN"] = (snoise, "snoise used for stdev map in MBPM")

    elif std_model == "std":
        _t = Time.now()
        _, _, std = sigma_clipped_stats(arr[tuple(slices)], **sigclip_kw)

        if update_header:
            if std_section is None:
                std_section = "[" + ",".join([":"] * arr.ndim) + "]"
            hdr["MB_MODEL"] = (std_model, "Method used for getting stdev map")
            hdr["MB_SSKY"] = (std, "Sky stdev for median filter BPM (MBPM)")
            hdr["MB_SSECT"] = (
                f"{std_section}",
                "Sky stdev calculation section in MBPM",
            )
            cmt2hdr(
                hdr,
                "h",
                verbose=verbose,
                t_ref=_t,
                s=(
                    "Sky standard deviation (MB_SSKY) calculated by sigma clipping at "
                    + f"MB_SSECT with {sigclip_kw}; used for std_ratio map calculation."
                ),
            )

    elif isinstance(std_model, np.ndarray):
        if std_model.shape != ccd.data.shape:
            raise ValueError(
                f"std_model.shape (= {std_model.shape}) differs from "
                + f"ccd.shape ({ccd.data.shape}"
            )
        std = std_model
        if update_header:
            hdr["MB_MODEL"] = ("User input array", "Method used for getting stdev map")

    elif isinstance(std_model, (int, float)):
        std = std_model
        if update_header:
            hdr["MB_MODEL"] = ("Constant", "Method used for getting stdev map")

    elif std_model is None:
        hdr["MB_MODEL"] = ("None", "Method used for getting stdev map")
        std = 1  # so that med_ratio is nothing but med_sub itself below.
        std_rat_clip = None  # turn off clipping using std_ratio

    else:
        raise ValueError("std_model not understood.")

    med_sub_clip = _sanitize_clips(med_sub_clip)
    med_rat_clip = _sanitize_clips(med_rat_clip)
    std_rat_clip = _sanitize_clips(std_rat_clip)

    _t = Time.now()
    npmask = []
    for msc, mrc, src in zip(med_sub_clip, med_rat_clip, std_rat_clip):
        if isinstance(msc, bool) and isinstance(mrc, bool) and isinstance(src, bool):
            npmask.append(np.zeros_like(arr, dtype=bool))

    med_ratio = arr / np.abs(med_filt)
    # Above is identical to sign(arr)*abs(arr/med_filt)
    med_sub = arr - med_filt
    std_ratio = med_sub / std

    # mask in the order of negative and positive cases
    mask_ms = _set_masks(med_sub, med_sub_clip)
    mask_mr = _set_masks(med_ratio, med_rat_clip)
    mask_sr = _set_masks(std_ratio, std_rat_clip)

    masks = []
    for i, (ms, mr, sr) in enumerate(zip(mask_ms, mask_mr, mask_sr)):
        if isinstance(ms, bool) and isinstance(mr, bool) and isinstance(sr, bool):
            # i.e., if all of neg or pos were None
            masks.append(np.zeros_like(arr, dtype=bool))  # all False

        else:  # if at least one was not None:
            if _LOGICAL_AND[i]:
                masks.append(ms & mr & sr)
            else:
                masks.append(ms | mr | sr)

    replace_mask = masks[0] | masks[1]
    arr[replace_mask] = med_filt[replace_mask]

    if update_header:
        hdr["MB_NLOGI"] = (
            _LOGICAL_STR[0],
            "The logic used for negative MBPM masks (and/or)",
        )
        hdr["MB_PLOGI"] = (
            _LOGICAL_STR[1],
            "The logic used for positive MBPM masks (and/or)",
        )
        hdr["MB_RAT_U"] = (med_rat_clip[1], "Upper clip of (data/|medfilt|) map (MBPM)")
        hdr["MB_RAT_L"] = (med_rat_clip[0], "Lower clip of (data/|medfilt|) map (MBPM)")
        hdr["MB_SUB_U"] = (med_sub_clip[1], "Upper clip of (data-medfilt) map (MBPM)")
        hdr["MB_SUB_L"] = (med_sub_clip[0], "Lower clip of (data-medfilt) map (MBPM)")
        hdr["MB_STD_U"] = (
            std_rat_clip[1],
            "Upper clip of (data-medfilt)/std map (MBPM)",
        )
        hdr["MB_STD_L"] = (
            std_rat_clip[0],
            "Lower clip of (data-medfilt)/std map (MBPM)",
        )

        cmt2hdr(
            hdr,
            "h",
            verbose=verbose,
            t_ref=_t,
            s="[medfilt_bpm] Median-filter based Bad-Pixel Masking (MBPM) applied.",
        )
        # cmt2hdr(
        #     hdr, 'h', verbose=verbose, t_ref=_t,
        #     s=("(1) Median additive difference (data-medfilt) generated, "
        #        + "(2) Median ratio (data/|medfilt|) generated, "
        #        + "(3) Stddev ratio ((data-medfilt)/std) generated, "
        #        + "(4) posmask and negmask calculated by clips "
        #        + "MB_[ADD/RAT/STD]_[U/L] and logic MB_[N/P]LOG (see keywords),"
        #        + "(5) Pixels of (posmask | negmask) are replaced with median "
        #        + "filtered frame."
        #        ))

    nccd = CCDData(data=arr - cadd, header=hdr, unit=ccd.unit)

    if full:
        return nccd, dict(
            negmask=masks[0],
            posmask=masks[1],
            med_filt=med_filt,
            med_sub=med_sub,
            med_rat=med_ratio,
            std_rat=std_ratio,
            std=std,
        )
    else:
        return nccd
