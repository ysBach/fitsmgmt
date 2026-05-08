"""Visualization utilities for matplotlib and astronomical imaging.

This module provides convenience functions for displaying astronomical
images with appropriate normalization (ZScale, etc.) using astropy and
matplotlib.
"""

from collections.abc import Sequence
from typing import Any, TypeAlias
from warnings import warn

from astropy.visualization import (
    AsinhStretch,
    AsymmetricPercentileInterval,
    BaseInterval,
    BaseStretch,
    ImageNormalize,
    LinearStretch,
    LogStretch,
    PercentileInterval,
    PowerStretch,
    SinhStretch,
    SqrtStretch,
    SquaredStretch,
    ZScaleInterval,
)

__all__ = ["znorm", "zimshow", "norm_imshow", "astropy_stretch", "imshow_norm"]

ImageLike: TypeAlias = Any
TickOffsets: TypeAlias = Sequence[int] | None
ImshowResult: TypeAlias = Any

_STRETCH_MAP: dict[str, BaseStretch] = {
    "linear": LinearStretch(),
    "sqrt": SqrtStretch(),
    "asinh": AsinhStretch(),
    "log": LogStretch(),
    "power": PowerStretch(1.0),
    "sinh": SinhStretch(),
    "square": SquaredStretch(),
}


def astropy_stretch(name: str) -> BaseStretch:
    """Resolve a stretch name string to an astropy BaseStretch instance.

    Parameters
    ----------
    name : str
        Case-insensitive stretch name. Accepts both bare names (e.g.,
        ``"asinh"``) and full class names (e.g., ``"AsinhStretch"``).

    Returns
    -------
    stretch : `~astropy.visualization.BaseStretch`
        The corresponding pre-instantiated stretch object.

    Raises
    ------
    ValueError
        If *name* does not match any supported stretch.
    """
    key = name.strip().lower()
    if key.endswith("stretch"):
        key = key[: -len("stretch")]
    if key not in _STRETCH_MAP:
        supported = ", ".join(sorted(_STRETCH_MAP))
        raise ValueError(f"Unknown stretch {name!r}. Supported names: {supported}")
    return _STRETCH_MAP[key]


def znorm(
    image: ImageLike,
    stretch: BaseStretch = LinearStretch(),
    **kwargs: Any,
) -> ImageNormalize:
    """Create an ImageNormalize object using ZScale interval.

    Parameters
    ----------
    image : array-like
        The image data to normalize.
    stretch : `~astropy.visualization.BaseStretch`, optional
        The stretch function to apply. Default is `LinearStretch`.
    **kwargs
        Additional keyword arguments passed to `ZScaleInterval`.

    Returns
    -------
    norm : `~astropy.visualization.ImageNormalize`
        The normalization object suitable for use with `imshow`.
    """
    return ImageNormalize(image, interval=ZScaleInterval(**kwargs), stretch=stretch)


def zimshow(
    ax,
    image: ImageLike,
    stretch: BaseStretch = LinearStretch(),
    cmap: Any = None,
    origin: str = "lower",
    zscale_kw: dict[str, Any] | None = None,
    **kwargs: Any,
) -> ImshowResult:
    """Display an image with ZScale normalization.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The axes on which to display the image.
    image : array-like
        The 2D image data to display.
    stretch : `~astropy.visualization.BaseStretch`, optional
        The stretch function to apply. Default is `LinearStretch`.
    cmap : str or `~matplotlib.colors.Colormap`, optional
        The colormap to use.
    origin : {'upper', 'lower'}, optional
        The origin of the image. Default is 'lower'.
    zscale_kw : dict, optional
        Additional keyword arguments passed to `ZScaleInterval`.
    **kwargs
        Additional keyword arguments passed to `ax.imshow`.

    Returns
    -------
    im : `~matplotlib.image.AxesImage`
        The image object returned by `imshow`.
    """
    if zscale_kw is None:
        zscale_kw = {}
    im = ax.imshow(
        image,
        norm=znorm(image, stretch=stretch, **zscale_kw),
        origin=origin,
        cmap=cmap,
        **kwargs,
    )
    return im


def _symmetric_ticks(half: int, n_ticks: int) -> list[int]:
    """Return a symmetric list of integer offsets centred on 0.

    Always includes 0. Uses ``(n_ticks - 1) // 2`` steps per side so the
    result has the same approximate count as the original matplotlib ticks.

    Parameters
    ----------
    half : int
        Half-size of the axis in pixels (``dimension // 2``).
    n_ticks : int
        Number of original matplotlib ticks (used to derive step count).

    Returns
    -------
    list[int]
        Symmetric offsets, e.g. ``[-10, -5, 0, 5, 10]``.
    """
    import math

    if half == 0:
        return [0]
    n_side = max(1, (n_ticks - 1) // 2)
    raw_step = half / n_side
    magnitude = 10 ** math.floor(math.log10(max(raw_step, 1)))
    step = magnitude
    for factor in (1, 2, 5, 10):
        step = int(magnitude * factor)
        if step >= raw_step:
            break
    step = max(1, step)
    offsets = list(range(0, half + 1, step))
    return sorted({-o for o in offsets} | set(offsets))


def _apply_center_origin_ticks(
    ax,
    shape: tuple[int, int],
    xticks: TickOffsets = None,
    yticks: TickOffsets = None,
) -> None:
    """Relabel axes ticks so that coordinate 0 sits at the image center.

    Parameters
    ----------
    ax : `~matplotlib.axes.Axes`
        The axes whose tick labels will be updated.
    shape : tuple[int, int]
        Image shape as ``(n_rows, n_cols)``.
    xticks : array-like of int or None, optional
        Tick positions expressed as **offsets from the image center** along
        the x-axis.  If ``None``, symmetric ticks are generated automatically.
    yticks : array-like of int or None, optional
        Same as *xticks* but for the y-axis.
    """
    center_col = shape[1] // 2
    center_row = shape[0] // 2

    if xticks is None:
        n_x = len(ax.get_xticks())
        x_offsets = _symmetric_ticks(center_col, n_x)
    else:
        x_offsets = list(xticks)
    x_pixel = [o + center_col for o in x_offsets]
    from matplotlib.ticker import FixedLocator

    ax.xaxis.set_major_locator(FixedLocator(x_pixel))
    ax.set_xticklabels([str(o) for o in x_offsets])

    if yticks is None:
        n_y = len(ax.get_yticks())
        y_offsets = _symmetric_ticks(center_row, n_y)
    else:
        y_offsets = list(yticks)
    y_pixel = [o + center_row for o in y_offsets]
    ax.yaxis.set_major_locator(FixedLocator(y_pixel))
    ax.set_yticklabels([str(o) for o in y_offsets])


def imshow_norm(
    data: ImageLike,
    ax: Any = None,
    stretch: str | BaseStretch = "linear",
    interval: str | BaseInterval | None = None,
    origin: str = "lower",
    tickorigin2center: bool = False,
    xticks: TickOffsets = None,
    yticks: TickOffsets = None,
    return_norm: bool = False,
    # stretch tuning
    asinh_a: float = 0.1,
    log_a: float = 1000.0,
    power: float = 1.0,
    sinh_a: float = 0.3,
    # range / clipping
    vmin: float | None = None,
    vmax: float | None = None,
    min_percent: float | None = None,
    max_percent: float | None = None,
    percent: float | None = None,
    clip: bool = False,
    invalid: float | None = -1.0,
    **kwargs: Any,
) -> ImshowResult | tuple[ImshowResult, ImageNormalize]:
    """Display an image with astropy normalization.

    A unified wrapper around `astropy.visualization.imshow_norm` that
    resolves stretch names from strings, supports common interval shortcuts,
    defaults ``origin`` to ``"lower"``, and optionally relabels axes so that
    coordinate 0 sits at the image center.

    Parameters
    ----------
    data : array-like
        The 2-D image data to display.
    ax : `~matplotlib.axes.Axes` or None, optional
        Target axes. If ``None``, uses the current pyplot axes.
    stretch : str or `~astropy.visualization.BaseStretch`, optional
        Stretch specification. A string is resolved case-insensitively and
        accepts both bare names (``"asinh"``) and full class names
        (``"AsinhStretch"``). A ``BaseStretch`` instance is passed through
        unchanged. Default is ``"linear"``.
    interval : str, `~astropy.visualization.BaseInterval`, or None, optional
        Interval controlling the data range mapping:

        - ``None`` — uses ``vmin``/``vmax`` if given, otherwise the data
          min/max (equivalent to `~astropy.visualization.ManualInterval`).
        - ``"zscale"`` — `~astropy.visualization.ZScaleInterval`.
        - ``percent=<v>`` — `~astropy.visualization.PercentileInterval(v)`.
        - ``min_percent``/``max_percent`` —
          `~astropy.visualization.AsymmetricPercentileInterval`.
        - Any `~astropy.visualization.BaseInterval` instance — passed through
          unchanged.

        Default is ``None``.
    origin : str, optional
        Image origin convention. Default is ``"lower"``.
    tickorigin2center : bool, optional
        If ``True``, relabel axes ticks so that coordinate 0 sits at the
        image center. Default is ``False``.
    xticks : array-like of int or None, optional
        Only used when ``tickorigin2center=True``. Tick positions as offsets
        from the image center along the x-axis. ``None`` generates symmetric
        ticks automatically.
    yticks : array-like of int or None, optional
        Same as *xticks* but for the y-axis.
    return_norm : bool, optional
        If ``True``, return ``(AxesImage, ImageNormalize)``. If ``False``
        (default), return only the ``AxesImage`` for backward compatibility.
    asinh_a : float, optional
        The ``a`` parameter for ``AsinhStretch``. Default is ``0.1``.
    log_a : float, optional
        The ``a`` parameter for ``LogStretch``. Default is ``1000.0``.
    power : float, optional
        The ``a`` parameter for ``PowerStretch``. Default is ``1.0``.
    sinh_a : float, optional
        The ``a`` parameter for ``SinhStretch``. Default is ``0.3``.
    vmin, vmax : float or None, optional
        Explicit minimum/maximum data values for the normalization.
        Ignored when ``interval`` is set to a `~astropy.visualization.BaseInterval`
        instance or ``"zscale"``.
    min_percent, max_percent : float or None, optional
        Percentile-based minimum/maximum. Constructs
        `~astropy.visualization.AsymmetricPercentileInterval` when either is
        given and ``interval`` is ``None``.
    percent : float or None, optional
        Symmetric percentile for both min and max. Constructs
        `~astropy.visualization.PercentileInterval` when given and ``interval``
        is ``None``. Takes precedence over ``min_percent``/``max_percent``.
    clip : bool, optional
        Whether to clip values outside the normalized range. Default is ``False``.
    invalid : float or None, optional
        Value assigned to invalid (NaN/inf) pixels before normalization.
        Default is ``-1.0``.
    **kwargs
        Additional keyword arguments forwarded to ``ax.imshow``.

    Returns
    -------
    im : `~matplotlib.image.AxesImage`
        The image object. If ``return_norm=True``, returns
        ``(im, norm)`` where *norm* is the `~astropy.visualization.ImageNormalize`
        instance.
    """
    import numpy as np
    from matplotlib.axes import Axes

    # Detect old norm_imshow(ax, data, ...) positional convention and swap.
    if isinstance(data, Axes) and not isinstance(ax, Axes):
        warn(
            "Passing ax as the first positional argument is deprecated. "
            "Use imshow_norm(data, ax=ax, ...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        data, ax = ax, data

    # --- resolve stretch ---
    _stretch_tuning = {"asinh": asinh_a, "log": log_a, "power": power, "sinh": sinh_a}
    _stretch_class = {
        "asinh": AsinhStretch,
        "log": LogStretch,
        "power": PowerStretch,
        "sinh": SinhStretch,
    }

    if isinstance(stretch, str):
        key = stretch.strip().lower()
        if key.endswith("stretch"):
            key = key[: -len("stretch")]
        if key not in _STRETCH_MAP:
            supported = ", ".join(sorted(_STRETCH_MAP))
            raise ValueError(f"Unknown stretch {stretch!r}. Supported: {supported}")
        if key in _stretch_class:
            stretch_obj = _stretch_class[key](_stretch_tuning[key])
        else:
            stretch_obj = _STRETCH_MAP[key]
    else:
        stretch_obj = stretch  # BaseStretch instance passed through

    # --- resolve interval ---
    if isinstance(interval, str):
        if interval.lower() == "zscale":
            interval_obj = ZScaleInterval()
        else:
            raise ValueError(
                f"Unknown interval string {interval!r}. "
                "Use 'zscale' or a BaseInterval instance."
            )
    else:
        interval_obj = interval  # BaseInterval instance or None

    # --- build vmin/vmax from percent args if needed ---
    # Pass everything through ImageNormalize directly for full control.

    if percent is not None and interval_obj is None:
        interval_obj = PercentileInterval(percent)
    elif min_percent is not None or max_percent is not None:
        lo = min_percent if min_percent is not None else 0.0
        hi = max_percent if max_percent is not None else 100.0
        if interval_obj is None:
            interval_obj = AsymmetricPercentileInterval(lo, hi)

    norm = ImageNormalize(
        np.asarray(data, dtype=float),
        interval=interval_obj,
        vmin=vmin,
        vmax=vmax,
        stretch=stretch_obj,
        clip=clip,
        invalid=invalid,
    )

    if ax is None:
        import matplotlib.pyplot as plt

        ax = plt.gca()

    im = ax.imshow(np.asarray(data, dtype=float), origin=origin, norm=norm, **kwargs)

    if tickorigin2center:
        _apply_center_origin_ticks(ax, data.shape, xticks=xticks, yticks=yticks)

    if return_norm:
        return im, norm
    return im


def norm_imshow(
    *args: Any,
    **kwargs: Any,
) -> ImshowResult | tuple[ImshowResult, ImageNormalize]:
    """Deprecated alias for `imshow_norm`. Use `imshow_norm` instead."""
    warn(
        "norm_imshow is deprecated and will be removed in a future version. "
        "Use imshow_norm instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # norm_imshow had ax as first positional arg, data as second
    if args:
        ax, data = args[0], args[1]
        return imshow_norm(data, ax=ax, **kwargs)
    return imshow_norm(**kwargs)
