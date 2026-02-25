"""Acoustic operators ported from j-wave: Helmholtz, laplacian_with_pml, wavevector."""
import mlx.core as mx

from mlxdf import Field, Continuous, FourierSeries
from mlxdf.core import operator
from mlxdf.discretization import FiniteDifferences, OnGrid
from mlxdf.operators import (compose, diag_jacobian, functional, gradient,
                             shift_operator, sum_over_dims)

from mwave.geometry import Medium
from .conversion import db2neper
from .pml import complex_pml, complex_pml_on_grid


@operator
def laplacian_with_pml(u: Continuous,
                       medium: Medium,
                       *,
                       omega=1.0,
                       params=None) -> Continuous:
    """Laplacian with PML for Continuous fields."""
    x = Continuous(None, u.domain, lambda p, x: x)
    pml = complex_pml(x, medium, omega)
    grad_u = gradient(u)
    mod_grad_u = grad_u * pml
    mod_diag_jac = diag_jacobian(mod_grad_u) * pml
    return sum_over_dims(mod_diag_jac)


def on_grid_pml_init(u, medium, omega, *args, **kwargs):
    return [
        u.replace_params(complex_pml_on_grid(medium, omega, shift=u.domain.dx[0] / 2)),
        u.replace_params(complex_pml_on_grid(medium, omega, shift=-u.domain.dx[0] / 2)),
    ]


@operator
def laplacian_with_pml(u: OnGrid,
                       medium: Medium,
                       *,
                       omega=1.0,
                       params=None) -> OnGrid:
    """Laplacian with PML for OnGrid fields."""
    pml_grid = complex_pml_on_grid(medium, omega)
    pml = u.replace_params(pml_grid)

    grad_u = gradient(u)
    mod_grad_u = grad_u * pml
    mod_diag_jac = diag_jacobian(mod_grad_u) * pml
    nabla_u = sum_over_dims(mod_diag_jac)

    rho0 = medium.density
    if not issubclass(type(rho0), Field):
        rho_u = 0.0
    else:
        grad_rho0 = gradient(rho0)
        rho_u = sum_over_dims(mod_grad_u * grad_rho0) / rho0

    return nabla_u - rho_u


def fd_laplacian_with_pml_init(u: FiniteDifferences, medium, omega, *args, **kwargs):
    return {
        "pml_on_grid": on_grid_pml_init(u, medium, omega),
        "stencils": {
            "gradient": gradient.default_params(u, stagger=[0.5]),
            "gradient_unstaggered": gradient.default_params(u),
            "diag_jacobian": diag_jacobian.default_params(u, stagger=[-0.5]),
        },
    }


@operator(init_params=fd_laplacian_with_pml_init)
def laplacian_with_pml(u: FiniteDifferences,
                       medium: Medium,
                       *,
                       omega=1.0,
                       params=None) -> FiniteDifferences:
    """Laplacian with PML for FiniteDifferences fields."""
    rho0 = medium.density
    pml = params["pml_on_grid"]
    stencils = params["stencils"]

    grad_u = gradient(u, stagger=[0.5], params=stencils["gradient"])
    mod_grad_u = grad_u * pml[0]
    mod_diag_jac = diag_jacobian(mod_grad_u, stagger=[-0.5],
                                  params=stencils["diag_jacobian"])
    nabla_u = sum_over_dims(mod_diag_jac * pml[1])

    if not issubclass(type(rho0), Field):
        rho_u = 0.0
    else:
        grad_u2 = gradient(u, params=stencils["gradient_unstaggered"])
        grad_rho0 = gradient(rho0, stagger=[0],
                             params=stencils["gradient_unstaggered"])
        rho_u = sum_over_dims(mod_grad_u * grad_rho0) / rho0

    return nabla_u - rho_u


def fourier_laplacian_with_pml_init(u: FourierSeries, medium, omega, *args, **kwargs):
    return {
        "pml_on_grid": on_grid_pml_init(u, medium, omega),
        "fft_u": gradient.default_params(u),
    }


@operator(init_params=fourier_laplacian_with_pml_init)
def laplacian_with_pml(u: FourierSeries,
                       medium: Medium,
                       *,
                       omega=1.0,
                       params=None) -> FourierSeries:
    """Laplacian with PML for FourierSeries fields."""
    rho0 = medium.density
    pml = params["pml_on_grid"]

    grad_u = gradient(u, stagger=[0.5], correct_nyquist=False,
                      params=params["fft_u"])
    mod_grad_u = grad_u * pml[0]
    mod_diag_jac = (diag_jacobian(mod_grad_u, stagger=[-0.5],
                                   correct_nyquist=False,
                                   params=params["fft_u"]) * pml[1])
    nabla_u = sum_over_dims(mod_diag_jac)

    if not issubclass(type(rho0), Field):
        rho_u = 0.0
    else:
        assert isinstance(rho0, FourierSeries)
        if "fft_rho0" not in params:
            params["fft_rho0"] = gradient.default_params(rho0)
        grad_rho0 = gradient(rho0, stagger=[0.5], params=params["fft_rho0"])
        dx = list(map(lambda x: -x / 2, u.domain.dx))
        _ru = shift_operator(mod_grad_u * grad_rho0, dx=dx)
        rho_u = sum_over_dims(_ru) / rho0

    return nabla_u - rho_u


@operator
def wavevector(u: Field, medium: Medium, *, omega=1.0, params=None) -> Field:
    """Wavevector operator."""
    c = medium.sound_speed
    alpha = medium.attenuation
    trans_fun = lambda x: db2neper(x, 2.0)
    alpha = compose(alpha)(trans_fun)
    k_mod = (omega / c)**2 + 2j * (omega**3) * alpha / c
    return u * k_mod


def helmholtz_init_params(u, medium, omega, *args, **kwargs):
    return {
        "laplacian": laplacian_with_pml.default_params(u, medium, omega=omega),
        "wavevector": wavevector.default_params(u, medium, omega=omega),
    }


@operator(init_params=helmholtz_init_params)
def helmholtz(u: Field, medium: Medium, *, omega=1.0, params=None) -> Field:
    """Helmholtz operator with PML."""
    lapl_params, wv_params = params["laplacian"], params["wavevector"]
    L = laplacian_with_pml(u, medium, omega, params=lapl_params)
    k = wavevector(u, medium, omega, params=wv_params)
    return L + k


def ongrid_helmholtz_init_params(u: OnGrid, medium, omega, *args, **kwargs):
    return laplacian_with_pml.default_params(u, medium, omega=omega)


@operator(init_params=ongrid_helmholtz_init_params)
def helmholtz(u: OnGrid, medium: Medium, *, omega=1.0, params=None) -> OnGrid:
    """Helmholtz operator with PML for OnGrid fields."""
    L = laplacian_with_pml(u, medium, omega=omega, params=params)
    k = wavevector(u, medium, omega=omega)
    return L + k


def scale_source_helmholtz(source, medium):
    if isinstance(medium.sound_speed, Field):
        min_sos = functional(medium.sound_speed)(mx.min)
    else:
        min_sos = mx.min(mx.array(medium.sound_speed))
    source = source * 2 / (source.domain.dx[0] * min_sos)
    return source
