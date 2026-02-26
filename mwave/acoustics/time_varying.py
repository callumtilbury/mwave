"""Time-domain wave propagation simulator ported from j-wave.
Uses explicit Python for-loop instead of jax.lax.scan.
"""
from typing import Callable, Dict, Tuple, Union

import numpy as np
import mlx.core as mx

from mlxdf import Field
from mlxdf.core import operator
from mlxdf.discretization import FourierSeries, Linear, OnGrid
from mlxdf.mods import Module
from mlxdf.operators import diag_jacobian, shift_operator, sum_over_dims

from mwave.acoustics.spectral import kspace_op
from mwave.geometry import Medium, Sources, TimeAxis
from mwave.logger import logger
from mwave.signal_processing import smooth

from .pml import td_pml_on_grid


class TimeWavePropagationSettings:
    """Settings for time-domain wave solvers."""

    def __init__(
        self,
        c_ref: Callable = lambda m: m.max_sound_speed,
        checkpoint: bool = True,
        smooth_initial: bool = True,
    ):
        self.c_ref = c_ref
        self.checkpoint = checkpoint
        self.smooth_initial = smooth_initial


def _shift_rho(rho0, direction, dx):
    if isinstance(rho0, OnGrid):
        rho0_params = rho0.params[..., 0]

        def linear_interp(u, axis):
            return 0.5 * (mx.roll(u, -direction, axis) + u)

        rho0 = mx.stack(
            [linear_interp(rho0_params, n) for n in range(rho0.domain.ndim)],
            axis=-1)
    elif isinstance(rho0, Field):
        rho0 = shift_operator(rho0, direction * dx)
    return rho0


@operator
def momentum_conservation_rhs(p: OnGrid,
                              u: OnGrid,
                              medium: Medium,
                              *,
                              c_ref=1.0,
                              dt=1.0,
                              params=None) -> OnGrid:
    """Momentum conservation RHS (staggered, OnGrid)."""
    dx = np.asarray(u.domain.dx)
    rho0 = _shift_rho(medium.density, 1, dx)
    dp = diag_jacobian(p, stagger=[0.5])
    return -dp / rho0


@operator
def momentum_conservation_rhs(
    p: FourierSeries,
    u: FourierSeries,
    medium: Medium,
    *,
    c_ref=1.0,
    dt=1.0,
    params=None,
) -> FourierSeries:
    """Momentum conservation RHS (Fourier, with k-space operator)."""
    if params is None:
        params = kspace_op(p.domain, c_ref, dt)

    dx = np.asarray(u.domain.dx)
    direction = 1

    rho0 = _shift_rho(medium.density, direction, dx)

    k_vec = params["k_vec"]
    k_space_op = params["k_space_op"]

    shift_and_k_op = [
        1j * k.astype(mx.complex64) * mx.exp(1j * k.astype(mx.complex64) * direction * delta / 2)
        for k, delta in zip(k_vec, dx)
    ]

    p_params = p.params[..., 0]
    Fu = mx.fft.fftn(p_params)

    def single_grad(axis):
        Fx = mx.moveaxis(Fu, axis, -1)
        k_op = mx.moveaxis(k_space_op, axis, -1)
        iku = mx.moveaxis(Fx * shift_and_k_op[axis] * k_op, -1, axis)
        return mx.fft.ifftn(iku).real

    dp = mx.stack([single_grad(i) for i in range(p.domain.ndim)], axis=-1)
    update = -p.replace_params(dp) / rho0
    return update


@operator
def mass_conservation_rhs(p: OnGrid,
                          u: OnGrid,
                          mass_source: object,
                          medium: Medium,
                          *,
                          c_ref,
                          dt,
                          params=None) -> OnGrid:
    """Mass conservation RHS (staggered, OnGrid)."""
    rho0 = medium.density
    c0 = medium.sound_speed
    dx = np.asarray(p.domain.dx)

    du = diag_jacobian(u, stagger=[-0.5])
    divergence = -du * rho0

    if isinstance(mass_source, (int, float)) and mass_source == 0.0:
        update = divergence
    else:
        scale = 2.0 / (p.domain.ndim * dx)
        if isinstance(c0, Field):
            inv_c0 = 1.0 / c0
            src_term = inv_c0 * (scale * mass_source)
        else:
            src_term = scale * mass_source / c0
        update = divergence + src_term
    return update


@operator
def mass_conservation_rhs(
    p: FourierSeries,
    u: FourierSeries,
    mass_source: object,
    medium: Medium,
    *,
    c_ref,
    dt,
    params=None,
) -> FourierSeries:
    """Mass conservation RHS (Fourier, with k-space operator)."""
    if params is None:
        params = kspace_op(p.domain, c_ref, dt)

    dx = np.asarray(p.domain.dx)
    direction = -1

    k_vec = params["k_vec"]
    k_space_op = params["k_space_op"]
    rho0 = medium.density
    c0 = medium.sound_speed

    shift_and_k_op = [
        1j * k.astype(mx.complex64) * mx.exp(1j * k.astype(mx.complex64) * direction * delta / 2)
        for k, delta in zip(k_vec, dx)
    ]

    def single_grad(axis, u):
        Fu = mx.fft.fftn(u)
        Fx = mx.moveaxis(Fu, axis, -1)
        k_op = mx.moveaxis(k_space_op, axis, -1)
        iku = mx.moveaxis(Fx * shift_and_k_op[axis] * k_op, -1, axis)
        return mx.fft.ifftn(iku).real

    du = mx.stack(
        [single_grad(i, u.params[..., i]) for i in range(p.domain.ndim)],
        axis=-1)
    divergence = -p.replace_params(du) * rho0

    # Source term: handle the case where c0 is a FourierSeries
    if isinstance(mass_source, (int, float)) and mass_source == 0.0:
        update = divergence
    else:
        # Compute 2 * mass_source / (c0 * ndim * dx)
        scale = 2.0 / (p.domain.ndim * dx)
        if isinstance(c0, Field):
            # mass_source / c0 -> (1/c0) * mass_source
            inv_c0 = 1.0 / c0
            src_term = inv_c0 * (scale * mass_source)
        else:
            src_term = scale * mass_source / c0
        update = divergence + src_term
    return update


@operator
def pressure_from_density(rho: Field, medium: Medium, *, params=None) -> Field:
    """Calculate pressure from acoustic density."""
    rho_sum = sum_over_dims(rho)
    c0 = medium.sound_speed
    return (c0**2) * rho_sum


# ── OnGrid wave propagation ──────────────────────────────────────────────────

def ongrid_wave_prop_params(medium, time_axis, *, settings, **kwargs):
    x = [
        x for x in [medium.sound_speed, medium.density, medium.attenuation]
        if isinstance(x, Field)
    ][0]

    dt = time_axis.dt
    c_ref = settings.c_ref(medium)

    def make_pml(staggering=0.0):
        pml_grid = td_pml_on_grid(medium, dt, c0=c_ref,
                                   dx=medium.domain.dx[0],
                                   coord_shift=staggering)
        return x.replace_params(pml_grid)

    pml_rho = make_pml()
    pml_u = make_pml(staggering=0.5)

    return {"pml_rho": pml_rho, "pml_u": pml_u, "c_ref": c_ref}


@operator(init_params=ongrid_wave_prop_params)
def simulate_wave_propagation(
    medium: Medium,
    time_axis: TimeAxis,
    *,
    settings: TimeWavePropagationSettings = TimeWavePropagationSettings(),
    sources=None,
    sensors=None,
    u0=None,
    p0=None,
    params=None,
):
    """Simulate wave propagation (OnGrid). Uses Python for-loop instead of scan."""
    if sensors is None:
        sensors = lambda p, u, rho: p

    dt = time_axis.dt
    Nt = int(time_axis.Nt)

    c_ref = params["c_ref"]
    pml_rho = params["pml_rho"]
    pml_u = params["pml_u"]

    shape = tuple(list(medium.domain.N) + [len(medium.domain.N)])
    shape_one = tuple(list(medium.domain.N) + [1])

    if u0 is None:
        u0 = pml_u.replace_params(mx.zeros(shape))
    if p0 is None:
        p0 = pml_rho.replace_params(mx.zeros(shape_one))
    else:
        if settings.smooth_initial:
            p0_params = p0.params[..., 0]
            p0_params = mx.expand_dims(smooth(p0_params), -1)
            p0 = p0.replace_params(p0_params)

        u0 = -dt * momentum_conservation_rhs(
            p0, u0, medium, c_ref=c_ref, dt=dt) / 2

    # Initialize acoustic density: repeat p0's single component ndim times
    p0_single = p0.params[..., 0]
    rho = (p0.replace_params(
        mx.stack([p0_single] * p0.domain.ndim, axis=-1)) / p0.domain.ndim)
    rho = rho / (medium.sound_speed**2)

    p, u = p0, u0
    recordings = []

    logger.debug("Starting simulation using generic OnGrid code")
    for n in range(Nt):
        mass_src = 0.0 if sources is None else sources.on_grid(n)

        du = momentum_conservation_rhs(p, u, medium, c_ref=c_ref, dt=dt)
        u = pml_u * (pml_u * u + dt * du)

        drho = mass_conservation_rhs(p, u, mass_src, medium,
                                      c_ref=c_ref, dt=dt)
        rho = pml_rho * (pml_rho * rho + dt * drho)

        p = pressure_from_density(rho, medium)
        recordings.append(sensors(p, u, rho))

    # Stack recordings
    if len(recordings) > 0 and isinstance(recordings[0], Field):
        # Stack the params of each field
        stacked = mx.stack([r.params for r in recordings])
        return recordings[0].__class__(stacked, recordings[0].domain)
    else:
        return mx.stack(recordings) if len(recordings) > 0 else None


# ── Fourier wave propagation ─────────────────────────────────────────────────

def fourier_wave_prop_params(medium, time_axis, *, settings, **kwargs):
    dt = time_axis.dt
    c_ref = settings.c_ref(medium)

    def make_pml(staggering=0.0):
        pml_grid = td_pml_on_grid(medium, dt, c0=c_ref,
                                   dx=medium.domain.dx[0],
                                   coord_shift=staggering)
        return FourierSeries(pml_grid, medium.domain)

    pml_rho = make_pml()
    pml_u = make_pml(staggering=0.5)

    fourier = kspace_op(medium.domain, c_ref, dt)

    return {
        "pml_rho": pml_rho,
        "pml_u": pml_u,
        "fourier": fourier,
        "c_ref": c_ref,
    }


def _make_compiled_step(domain, medium, dt, pml_u_params, pml_rho_params,
                        k_vec, k_space_op):
    """Build a compiled single-timestep function operating on raw mx.arrays.
    This bypasses plum dispatch and Field construction for maximum speed.
    """
    dx = np.asarray(domain.dx)
    ndim = domain.ndim
    direction_fwd = 1
    direction_bwd = -1

    # Pre-compute shifted k-operators for momentum (fwd) and mass (bwd)
    shift_k_fwd = [
        1j * k.astype(mx.complex64) * mx.exp(
            1j * k.astype(mx.complex64) * direction_fwd * delta / 2)
        for k, delta in zip(k_vec, dx)
    ]
    shift_k_bwd = [
        1j * k.astype(mx.complex64) * mx.exp(
            1j * k.astype(mx.complex64) * direction_bwd * delta / 2)
        for k, delta in zip(k_vec, dx)
    ]

    # Get sound speed as raw array (handle scalar or field)
    if isinstance(medium.sound_speed, Field):
        c0_arr = medium.sound_speed.params
    elif isinstance(medium.sound_speed, (int, float)):
        c0_arr = mx.array(medium.sound_speed, dtype=mx.float32)
    else:
        c0_arr = mx.array(medium.sound_speed)

    c0_sq = c0_arr ** 2
    # For source scaling in step_with_source we need c0 (not c0^2)
    c0_sq_for_src = c0_arr

    # Same for density
    if isinstance(medium.density, Field):
        rho0_arr = medium.density.params
    elif isinstance(medium.density, (int, float)):
        rho0_arr = mx.array(medium.density, dtype=mx.float32)
    else:
        rho0_arr = mx.array(medium.density)

    def _spectral_grad(p_scalar, shift_k_ops):
        """Spectral gradient: FFT → multiply by ik·kspace_op → IFFT."""
        Fp = mx.fft.fftn(p_scalar)
        grads = []
        for axis in range(ndim):
            Fx = mx.moveaxis(Fp, axis, -1)
            k_op = mx.moveaxis(k_space_op, axis, -1)
            iku = mx.moveaxis(Fx * shift_k_ops[axis] * k_op, -1, axis)
            grads.append(mx.fft.ifftn(iku).real)
        return mx.stack(grads, axis=-1)

    def _spectral_div(u_params, shift_k_ops):
        """Spectral divergence of a vector field."""
        divs = []
        for axis in range(ndim):
            Fu = mx.fft.fftn(u_params[..., axis])
            Fx = mx.moveaxis(Fu, axis, -1)
            k_op = mx.moveaxis(k_space_op, axis, -1)
            iku = mx.moveaxis(Fx * shift_k_ops[axis] * k_op, -1, axis)
            divs.append(mx.fft.ifftn(iku).real)
        return mx.stack(divs, axis=-1)

    def step(u_params, rho_params, p_params):
        """Single timestep: (u, rho, p) → (u', rho', p')"""
        # Momentum conservation: du = -grad(p) / rho0
        dp = _spectral_grad(p_params[..., 0], shift_k_fwd)
        du = -dp / rho0_arr
        u_params = pml_u_params * (pml_u_params * u_params + dt * du)

        # Mass conservation: drho = -div(u) * rho0
        div_u = _spectral_div(u_params, shift_k_bwd)
        drho = -div_u * rho0_arr
        rho_params = pml_rho_params * (pml_rho_params * rho_params + dt * drho)

        # Pressure from density: p = c0^2 * sum(rho)
        rho_sum = mx.sum(rho_params, axis=-1, keepdims=True)
        p_params = c0_sq * rho_sum

        return u_params, rho_params, p_params

    # Source-aware variant: mass_src is (Nx, Ny, 1) at a single time step.
    # Scale factor mirrors mass_conservation_rhs: 2 / (ndim * dx)
    _scale = mx.array(
        (2.0 / (domain.ndim * np.asarray(domain.dx))).astype(np.float32)
    )  # shape (ndim,)

    def step_with_source(u_params, rho_params, p_params, mass_src):
        dp = _spectral_grad(p_params[..., 0], shift_k_fwd)
        du = -dp / rho0_arr
        u_params = pml_u_params * (pml_u_params * u_params + dt * du)

        div_u = _spectral_div(u_params, shift_k_bwd)
        # mass_src: (Nx, Ny, 1); _scale: (ndim,) → broadcasts to (Nx, Ny, ndim)
        src_contribution = _scale * mass_src / c0_sq_for_src
        drho = -div_u * rho0_arr + src_contribution
        rho_params = pml_rho_params * (pml_rho_params * rho_params + dt * drho)

        rho_sum = mx.sum(rho_params, axis=-1, keepdims=True)
        p_params = c0_sq * rho_sum

        return u_params, rho_params, p_params

    return mx.compile(step), mx.compile(step_with_source)


@operator(init_params=fourier_wave_prop_params)
def simulate_wave_propagation(
    medium: Medium,
    time_axis: TimeAxis,
    *,
    settings: TimeWavePropagationSettings = TimeWavePropagationSettings(),
    sources=None,
    sensors=None,
    u0=None,
    p0=None,
    params=None,
):
    """Simulate wave propagation (FourierSeries PSTD).
    Uses mx.compile on the inner step for minimal dispatch overhead.
    """
    if sensors is None:
        sensors = lambda p, u, rho: p

    dt = time_axis.dt
    Nt = int(time_axis.Nt)

    c_ref = params["c_ref"]
    pml_rho = params["pml_rho"]
    pml_u = params["pml_u"]

    shape = tuple(list(medium.domain.N) + [len(medium.domain.N)])
    shape_one = tuple(list(medium.domain.N) + [1])

    if u0 is None:
        u0 = pml_u.replace_params(mx.zeros(shape))
    if p0 is None:
        p0 = pml_rho.replace_params(mx.zeros(shape_one))
    else:
        if settings.smooth_initial:
            p0_params = p0.params[..., 0]
            p0_params = mx.expand_dims(smooth(p0_params), -1)
            p0 = p0.replace_params(p0_params)

        u0_init = pml_u.replace_params(mx.zeros(shape))
        u0 = (-dt * momentum_conservation_rhs(
            p0, u0_init, medium, c_ref=c_ref, dt=dt,
            params=params["fourier"]) / 2)

    # Initialize acoustic density: repeat p0's single component ndim times
    p0_single = p0.params[..., 0]
    rho = (p0.replace_params(
        mx.stack([p0_single] * p0.domain.ndim, axis=-1)) / p0.domain.ndim)
    rho = rho / (medium.sound_speed**2)

    p, u = p0, u0
    recordings = []

    compiled_step, compiled_step_with_source = _make_compiled_step(
        p.domain, medium, dt,
        pml_u.params, pml_rho.params,
        params["fourier"]["k_vec"], params["fourier"]["k_space_op"],
    )

    u_p, rho_p, p_p = u.params, rho.params, p.params

    if sources is None:
        logger.debug("Starting simulation using COMPILED FourierSeries code")
        for n in range(Nt):
            u_p, rho_p, p_p = compiled_step(u_p, rho_p, p_p)
            p_field = p.replace_params(p_p)
            u_field = u.replace_params(u_p)
            rho_field = rho.replace_params(rho_p)
            recordings.append(sensors(p_field, u_field, rho_field))
    else:
        logger.debug("Starting simulation using COMPILED FourierSeries code (with sources)")
        # Pre-bake all source grids upfront so on_grid() is never called
        # inside the loop — keeps the hot path inside mx.compile.
        source_grid = sources.to_grid_array(Nt)  # (Nt, Nx, Ny, 1)
        mx.eval(source_grid)
        for n in range(Nt):
            u_p, rho_p, p_p = compiled_step_with_source(u_p, rho_p, p_p, source_grid[n])
            p_field = p.replace_params(p_p)
            u_field = u.replace_params(u_p)
            rho_field = rho.replace_params(rho_p)
            recordings.append(sensors(p_field, u_field, rho_field))

    if len(recordings) > 0 and isinstance(recordings[0], Field):
        stacked = mx.stack([r.params for r in recordings])
        return recordings[0].__class__(stacked, recordings[0].domain)
    else:
        return mx.stack(recordings) if len(recordings) > 0 else None

