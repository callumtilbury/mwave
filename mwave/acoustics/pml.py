"""PML (Perfectly Matched Layer) construction for m-wave.
Uses numpy for array construction (init-time only), converts to mx.array at the end.
"""
from typing import Callable

import numpy as np
import mlx.core as mx

from mlxdf import Continuous, Field
from mlxdf.core import operator
from mlxdf.operators import compose
from mlxdf.geometry import Domain
from mwave.geometry import Medium


def _base_pml(transform_fun, medium, exponent=2.0, alpha_max=2.0, shift=0.0):
    """Build base PML grid using numpy."""
    def pml_edge(x):
        return x / 2 - medium.pml_size

    delta_pml = np.array(list(map(pml_edge, medium.domain.N)))
    coord_grid = np.array(Domain(
        N=medium.domain.N,
        dx=tuple([1.0] * len(medium.domain.N))
    ).grid)
    coord_grid = coord_grid + shift

    def _pml_fun(x, delta_pml):
        diff = (np.abs(x) - 1.0 * delta_pml) / medium.pml_size
        on_pml = np.where(diff > 0, diff, 0)
        alpha = alpha_max * (on_pml**exponent)
        return transform_fun(alpha)

    return _pml_fun(coord_grid, delta_pml)


def complex_pml_on_grid(medium, omega, exponent=4.0, alpha_max=2.0, shift=0.0):
    """Complex PML on grid for frequency-domain solvers."""
    transform_fun = lambda alpha: 1.0 / (1 + 1j * alpha)
    result = _base_pml(transform_fun, medium, exponent, alpha_max, shift=shift)
    return mx.array(result.astype(np.complex64))


def td_pml_on_grid(
    medium,
    dt,
    exponent=4.0,
    alpha_max=2.0,
    c0=1.0,
    dx=1.0,
    coord_shift=0.0,
):
    """Time-domain PML built with numpy, returned as mx.array."""
    if medium.domain.ndim not in [1, 2, 3]:
        raise NotImplementedError(
            f"Can't make a PML for a domain of dimensions {medium.domain.ndim}"
        )

    if medium.pml_size == 0:
        size = tuple(list(medium.domain.N) + [1])
        return mx.ones(size)

    pml_size = int(medium.pml_size)

    x_right = (np.arange(1, pml_size + 1, 1) + coord_shift) / pml_size
    x_left = (np.arange(pml_size, 0, -1) - coord_shift) / pml_size
    x_right = x_right**exponent
    x_left = x_left**exponent

    alpha_left = np.exp(alpha_max * (-1) * x_left * dt * c0 / 2 / dx)
    alpha_right = np.exp(alpha_max * (-1) * x_right * dt * c0 / 2 / dx)

    pml_shape = tuple(list(medium.domain.N) + [len(medium.domain.N)])
    pml = np.ones(pml_shape, dtype=np.float32)

    # Last axis (fastest varying spatial dimension)
    if medium.domain.ndim >= 1:
        pml[..., :pml_size, -1] = alpha_left
        pml[..., -pml_size:, -1] = alpha_right

    if medium.domain.ndim >= 2:
        al2 = np.expand_dims(alpha_left, -1)
        ar2 = np.expand_dims(alpha_right, -1)
        pml[..., :pml_size, :, -2] = al2
        pml[..., -pml_size:, :, -2] = ar2

    if medium.domain.ndim == 3:
        al3 = np.expand_dims(np.expand_dims(alpha_left, -1), -1)
        ar3 = np.expand_dims(np.expand_dims(alpha_right, -1), -1)
        pml[:pml_size, :, :, -3] = al3
        pml[-pml_size:, :, :, -3] = ar3

    return mx.array(pml)


@operator
def complex_pml(x: Continuous,
                medium: Medium,
                *,
                omega=1.0,
                sigma_star=10.0,
                alpha=2.0,
                params=None):
    """Complex PML for continuous fields."""
    dx = x.domain.dx
    N = x.domain.N

    def sigma(x):
        delta_pml = dx[0] * (N[0] / 2 - medium.pml_size)
        L_half = dx[0] * N[0] / 2
        abs_x = mx.abs(x)
        in_pml = (mx.abs(abs_x - delta_pml) / (L_half - delta_pml))**alpha
        return mx.where(abs_x > delta_pml, sigma_star * in_pml, mx.zeros_like(x))

    y = compose(x)(sigma)
    return 1.0 / (1.0 + 1j * y / omega)
