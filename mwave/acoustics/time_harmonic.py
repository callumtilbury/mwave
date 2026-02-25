"""Time-harmonic acoustic solvers ported from j-wave.

Note: helmholtz_solver (gmres/bicgstab) is stubbed — MLX has no sparse iterative solver.
Born series, angular spectrum, and Rayleigh integral are fully ported.
"""
from math import factorial
from typing import Union

import numpy as np
import mlx.core as mx

from mlxdf import Field, FourierSeries
from mlxdf.core import operator
from mlxdf.discretization import OnGrid
from mlxdf.geometry import Domain
from mlxdf.operators import functional
from mlxdf.operators.differential import laplacian

from mwave.geometry import Medium
from .operators import helmholtz, scale_source_helmholtz


@operator
def angular_spectrum(
    pressure: FourierSeries,
    *,
    z_pos,
    f0,
    medium,
    padding=0,
    angular_restriction=True,
    unpad_output=True,
    params=None,
) -> FourierSeries:
    """Angular spectrum method for projecting pressure fields."""
    c0 = medium.sound_speed
    k = 2 * np.pi * f0 / c0
    k_t_sq = k**2

    p = pressure.on_grid[..., 0]
    if padding > 0:
        p = mx.pad(p, [(padding, padding)] * len(pressure.domain.N))

    domain = Domain(
        tuple(int(s) for s in p.shape[:len(pressure.domain.N)]),
        pressure.domain.dx,
    )
    pressure_padded = FourierSeries(p, domain)

    freq_grid = pressure_padded._freq_grid
    k_x_sq = mx.sum(freq_grid**2, axis=-1)
    kz = mx.sqrt(k_t_sq - k_x_sq + 0j)

    H = mx.conj(mx.exp(1j * z_pos * kz))

    D = min(pressure_padded.domain.size)
    kc = k * mx.sqrt(mx.array(0.5 * D**2 / (0.5 * D**2 + z_pos**2)))
    H_restrict = mx.where(k_x_sq <= kc**2, H, mx.zeros_like(H))
    H = H_restrict if angular_restriction else H

    p_hat = mx.fft.fftn(pressure_padded.on_grid[..., 0])
    p_plane = mx.fft.ifftn(p_hat * H)

    if padding > 0 and unpad_output:
        slices = tuple(slice(padding, -padding) for _ in range(len(pressure.domain.N)))
        p_plane = p_plane[slices]
        return FourierSeries(p_plane, pressure.domain)
    else:
        return FourierSeries(p_plane, domain)


@operator
def scattering_potential(field: Field,
                         k_sq: Field,
                         *,
                         k0=1.0,
                         epsilon=0.1,
                         params=None) -> Field:
    """Scattering potential for CBS method."""
    k = k_sq - k0**2 - 1j * epsilon
    return field * k


@operator
def homogeneous_helmholtz_green(field: FourierSeries,
                                *,
                                k0=1.0,
                                epsilon=0.1,
                                params=None):
    """Green's operator for homogeneous Helmholtz equation."""
    freq_grid = field._freq_grid
    p_sq = mx.sum(freq_grid**2, axis=-1)

    g_fourier = 1.0 / (p_sq - (k0**2) - 1j * epsilon)
    u = field.on_grid[..., 0]
    u_fft = mx.fft.fftn(u)
    Gu_fft = g_fourier * u_fft
    Gu = mx.fft.ifftn(Gu_fft)
    return field.replace_params(Gu)


@operator
def born_iteration(field: Field,
                   k_sq: Field,
                   src: Field,
                   *,
                   k0,
                   epsilon,
                   params=None) -> FourierSeries:
    """One step of the Convergent Born Series method."""
    V1 = scattering_potential(field, k_sq, k0=k0, epsilon=epsilon)
    G = homogeneous_helmholtz_green(V1 + src, k0=k0, epsilon=epsilon)
    V2 = scattering_potential(field - G, k_sq, k0=k0, epsilon=epsilon)
    return field - (1j / epsilon) * V2


def _cbs_pml(field, k0=1.0, pml_size=32, alpha=1.0):
    """Construct PML for CBS solver using numpy."""
    medium = Medium(domain=field.domain, pml_size=pml_size)
    N_order = 4

    def pml_edge(x):
        return x / 2 - pml_size

    def num(x):
        return (alpha**2) * (N_order - alpha * x + 2j * k0 * x) * (
            (alpha * x)**(N_order - 1))

    def den(x):
        return sum([((alpha * x)**i) / float(factorial(i))
                    for i in range(N_order + 1)]) * factorial(N_order)

    def transform_fun(x):
        return num(x) / den(x)

    delta_pml = np.array(list(map(pml_edge, medium.domain.N)))
    coord_grid = np.array(Domain(
        N=medium.domain.N,
        dx=tuple([1.0] * len(medium.domain.N))).grid)

    diff = np.abs(coord_grid) - delta_pml
    diff = np.where(diff > 0, diff, 0) / 4

    dist = np.sqrt(np.sum(diff**2, -1))
    k_k0 = transform_fun(dist)
    k_k0 = np.expand_dims(k_k0, -1)
    return mx.array((k_k0 + k0**2).astype(np.complex64))


@operator
def born_series(
    medium: Medium,
    src: FourierSeries,
    *,
    omega=1.0,
    k0=None,
    max_iter=1000,
    tol=1e-8,
    alpha=1.0,
    remove_pml=True,
    params=None,
) -> FourierSeries:
    """Helmholtz solver using Convergent Born Series (CBS)."""

    def enlarge_domain(domain, ps):
        new_N = tuple([x + 2 * ps for x in domain.N])
        return Domain(new_N, domain.dx)

    pml_size = int(medium.pml_size)

    def pad_fun(u):
        pad_size = [(pml_size, pml_size)] * len(u.domain.N) + [(0, 0)]
        return FourierSeries(mx.pad(u.on_grid, pad_size),
                             enlarge_domain(u.domain, pml_size))

    def cbs_helmholtz(field, k_sq):
        return laplacian(field) + k_sq * field

    if k0 is None:
        k_max = omega / functional(medium.sound_speed)(mx.max)
        k_min = omega / functional(medium.sound_speed)(mx.min)
        k0 = mx.sqrt(mx.array(0.5) * (k_max**2 + k_min**2))

    src = scale_source_helmholtz(src, medium)
    src = pad_fun(src)
    norm_initial = float(mx.sqrt(mx.sum(mx.abs(src.on_grid)**2)))

    _sos = medium.sound_speed.on_grid if isinstance(
        medium.sound_speed, FourierSeries) else medium.sound_speed

    k_biggest = mx.max((omega / _sos))
    k_sq = _cbs_pml(src, float(k_biggest), pml_size, alpha)

    # Set interior k_sq
    interior = ((omega / _sos)**2 + 0j).astype(mx.complex64)
    # Build via numpy for slicing
    k_sq_np = np.array(k_sq)
    interior_np = np.array(interior)
    slices = tuple(slice(pml_size, -pml_size) for _ in range(len(medium.domain.N)))
    k_sq_np[slices] = interior_np
    k_sq = FourierSeries(mx.array(k_sq_np), src.domain)

    epsilon = float(mx.max(mx.abs(k_sq.on_grid - k0**2)))

    guess = FourierSeries.empty(src.domain)
    # Make complex
    guess = guess + 0j

    field = guess
    for i in range(max_iter):
        field = born_iteration(field, k_sq, src, k0=k0, epsilon=epsilon)
        resid = cbs_helmholtz(field, k_sq) + src
        resid_norm = float(mx.sqrt(mx.sum(mx.abs(resid.on_grid)**2)))
        if resid_norm / norm_initial < tol:
            break

    # Remove padding
    if remove_pml:
        slices_out = tuple(slice(pml_size, -pml_size) for _ in range(len(medium.domain.N)))
        _out = field.on_grid[slices_out]
    else:
        _out = field.on_grid

    out_field = -1j * omega * FourierSeries(_out, medium.domain)
    return out_field


@operator
def rayleigh_integral(
    pressure: FourierSeries,
    *,
    r,
    f0,
    sound_speed=1500.0,
    params=None,
):
    """Rayleigh integral for a FourierSeries field."""
    if pressure.domain.ndim != 2:
        raise ValueError("Only 2D domains are supported.")

    k = 2 * np.pi * f0 / sound_speed
    area = pressure.domain.cell_volume
    plane_grid = np.array(pressure.domain.grid)

    z_dim = np.zeros(plane_grid.shape[:-1] + (1,))
    plane_grid = np.concatenate((plane_grid, z_dim), axis=-1)

    r_np = np.array(r)
    R = np.abs(r_np - plane_grid)

    def exp_term(x, y, z):
        r_dist = np.sqrt(x**2 + y**2 + z**2)
        return np.exp(1j * k * r_dist) / r_dist

    def direc_exp_term(x, y, z):
        # Numerical derivative along z
        eps = 1e-6
        return (exp_term(x, y, z + eps) - exp_term(x, y, z - eps)) / (2 * eps)

    weights = direc_exp_term(R[..., 0], R[..., 1], R[..., 2])
    p_grid = np.array(pressure.on_grid)
    return mx.array(np.sum(weights * p_grid) * area)


def helmholtz_solver(*args, **kwargs):
    """Helmholtz solver using iterative methods.

    Note: Not implemented for MLX — requires gmres/bicgstab which
    MLX does not provide. Use born_series instead.
    """
    raise NotImplementedError(
        "helmholtz_solver requires jax.scipy.sparse.linalg.gmres/bicgstab, "
        "which have no MLX equivalent. Use born_series() instead."
    )


def helmholtz_solver_verbose(*args, **kwargs):
    """Verbose Helmholtz solver — not available in MLX."""
    raise NotImplementedError(
        "helmholtz_solver_verbose is not available in m-wave. Use born_series() instead."
    )
