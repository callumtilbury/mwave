"""Visualization and utility functions ported from j-wave."""
import warnings
from typing import Set, Tuple, Union

import numpy as np
import mlx.core as mx
from mlxdf import Field

try:
    from matplotlib import pyplot as plt
    from matplotlib.figure import Figure
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def get_smallest_prime_factors(n: int) -> Set[int]:
    """Get the smallest prime factors of a number."""
    factors = []
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            factors.append(i)
            n //= i
    if n > 1:
        factors.append(n)
    return set(factors)


def load_image_to_numpy(filepath, padding=0, image_size=None):
    """Load an image as a numpy array."""
    if not HAS_PIL:
        raise ImportError("Pillow is required for load_image_to_numpy")
    img = Image.open(filepath).convert("L")
    if image_size is not None:
        img = img.resize(image_size)
    img = np.array(img).astype(np.float32)
    if padding is not None and padding > 0:
        img = np.pad(img, padding, mode="constant")
    return img


def is_numeric(x):
    """Check if x is a numeric value."""
    return isinstance(x, (int, float, complex))


def show_field(x, title="", figsize=(8, 6), vmax=None, aspect="auto"):
    """Plot a real-valued field."""
    if not HAS_MPL:
        raise ImportError("matplotlib is required for show_field")
    if isinstance(x, Field):
        x = np.array(x.on_grid)
    elif isinstance(x, mx.array):
        x = np.array(x)

    plt.figure(figsize=figsize)
    maxval = vmax or np.amax(np.abs(x))
    plt.imshow(x, cmap="RdBu_r", vmin=-maxval, vmax=maxval,
               interpolation="nearest", aspect=aspect)
    plt.colorbar()
    plt.title(title)
    plt.axis("off")


def display_complex_field(field, figsize=(15, 8), max_intensity=None):
    """Display real part and magnitude of a complex field."""
    if not HAS_MPL:
        raise ImportError("matplotlib is required for display_complex_field")
    if isinstance(field, Field):
        field = np.array(field.on_grid)
    elif isinstance(field, mx.array):
        field = np.array(field)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    if max_intensity is None:
        max_intensity = np.amax(np.abs(field))

    im1 = axes[0].imshow(field.real, vmin=-max_intensity,
                          vmax=max_intensity, cmap="seismic")
    axes[0].set_title("Real wavefield")
    im2 = axes[1].imshow(np.abs(field), vmin=0,
                          vmax=max_intensity, cmap="magma")
    axes[1].set_title("Wavefield magnitude")

    for ax, im in zip(axes, [im1, im2]):
        div = make_axes_locatable(ax)
        cax = div.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

    return fig, axes


def plot_complex_field(*args, **kwargs):
    warnings.warn("plot_complex_field is deprecated, use display_complex_field instead",
                  DeprecationWarning)
    return display_complex_field(*args, **kwargs)
