import math
from typing import List, Tuple, Union

import numpy as np
import mlx.core as mx

from mlxdf import Field, FourierSeries
from mlxdf.geometry import Domain
from mlxdf.mods import Module
from mlxdf.operators import functional

from mwave.logger import logger

Number = Union[float, int]


class Medium(Module):
    """Acoustic medium containing sound speed, density,
    attenuation, and PML configuration.

    Args:
        domain (Domain): The simulation domain.
        sound_speed: Sound speed (scalar, mx.array, or Field).
        density: Density (scalar, mx.array, or Field).
        attenuation: Attenuation (scalar, mx.array, or Field).
        pml_size (float): Size of PML layer in grid points.
    """
    def __init__(
        self,
        domain: Domain,
        sound_speed=1.0,
        density=1.0,
        attenuation=1.0,
        pml_size: float = 20.0,
    ):
        super().__init__()
        self.domain = domain
        self.pml_size = pml_size

        # Convert mx.array inputs to FourierSeries
        for name, val in [("sound_speed", sound_speed),
                          ("density", density),
                          ("attenuation", attenuation)]:
            if isinstance(val, mx.array) and val.ndim > 0:
                val = FourierSeries(val, domain)
            setattr(self, name, val)

    def _field_extremum(self, field_val, func):
        """Return extremum of a field or scalar."""
        if isinstance(field_val, Field):
            return functional(field_val)(func)
        return field_val  # already a scalar

    @property
    def max_sound_speed(self):
        return self._field_extremum(self.sound_speed, mx.max)

    @property
    def min_sound_speed(self):
        return self._field_extremum(self.sound_speed, mx.min)

    @property
    def max_density(self):
        return self._field_extremum(self.density, mx.max)

    @property
    def min_density(self):
        return self._field_extremum(self.density, mx.min)

    @property
    def max_attenuation(self):
        return self._field_extremum(self.attenuation, mx.max)

    @property
    def min_attenuation(self):
        return self._field_extremum(self.attenuation, mx.min)

    @property
    def int_pml_size(self) -> int:
        return int(self.pml_size)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def points_on_circle(
        n: int,
        radius: float,
        centre: Tuple[float, float],
        cast_int: bool = True,
        angle: float = 0.0,
        max_angle: float = 2 * np.pi) -> Tuple[List[float], List[float]]:
    """Generate points on a circle."""
    angles = np.linspace(0, max_angle, n, endpoint=False)
    x = (radius * np.cos(angles + angle) + centre[0]).tolist()
    y = (radius * np.sin(angles + angle) + centre[1]).tolist()
    if cast_int:
        x = list(map(int, x))
        y = list(map(int, y))
    return x, y


def unit_fibonacci_sphere(samples: int = 128):
    """Generate evenly distributed points on a unit sphere."""
    points = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2
        radius = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return points


def fibonacci_sphere(n, radius, centre, cast_int=True):
    """Generate points on a sphere."""
    points = np.array(unit_fibonacci_sphere(n))
    points = points * radius + np.array(centre)
    if cast_int:
        points = points.astype(int)
    return points[:, 0], points[:, 1], points[:, 2]


def circ_mask(N, radius, centre):
    """Generate a 2D circular binary mask."""
    x, y = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((x - centre[0])**2 + (y - centre[1])**2)
    return (dist < radius).astype(int)


def sphere_mask(N, radius, centre):
    """Generate a 3D spherical binary mask."""
    x, y, z = np.mgrid[0:N[0], 0:N[1], 0:N[2]]
    dist = np.sqrt((x - centre[0])**2 + (y - centre[1])**2 +
                   (z - centre[2])**2)
    return (dist < radius).astype(int)


# ── Sources ───────────────────────────────────────────────────────────────────

class Sources:
    """Source structure for time-varying simulations."""

    def __init__(self, positions, signals, dt, domain):
        self.positions = positions
        self.signals = signals
        self.dt = dt
        self.domain = domain

    def on_grid(self, n):
        src = np.zeros(self.domain.N)
        if len(self.signals) == 0:
            return mx.array(np.expand_dims(src, -1).astype(np.float32))

        idx = int(n) if not isinstance(n, int) else n
        signals = self.signals[:, idx]

        # Build source grid using numpy (because of index-add)
        for i in range(len(signals)):
            pos = tuple(p[i] for p in self.positions)
            src[pos] += float(signals[i])

        return mx.array(np.expand_dims(src, -1).astype(np.float32))

    @staticmethod
    def no_sources(domain):
        return Sources(positions=([], []), signals=[], dt=1.0, domain=domain)


# ── Distributed Transducer ────────────────────────────────────────────────────

class DistributedTransducer:
    """A transducer represented by a mask field and a time signal."""

    def __init__(self, mask, signal, dt, domain):
        self.mask = mask
        self.signal = signal
        self.dt = dt
        self.domain = domain

    def __call__(self, u: Field):
        from mlxdf.operators import dot_product
        return dot_product(self.mask, u)

    def set_signal(self, s):
        return DistributedTransducer(self.mask, s, self.dt, self.domain)

    def set_mask(self, m):
        return DistributedTransducer(m, self.signal, self.dt, m.domain)

    def on_grid(self, n):
        if len(self.signal) == 0:
            return 0.0
        idx = int(n) if not isinstance(n, int) else n
        signal = self.signal[idx]
        return signal * self.mask


def get_line_transducer(domain, position, width, angle=0):
    """Construct a 2D line transducer."""
    if angle != 0:
        raise NotImplementedError("Angle not implemented yet")

    mask = np.zeros(domain.N, dtype=np.float32)
    start_col = (domain.N[1] - width) // 2
    end_col = (domain.N[1] + width) // 2
    mask[position, start_col:end_col] = 1.0
    mask = mx.array(np.expand_dims(mask, -1))
    mask = FourierSeries(mask, domain)
    return DistributedTransducer(mask, [], 0.0, domain)


# ── Time-Harmonic Source ──────────────────────────────────────────────────────

class TimeHarmonicSource:
    """Time-harmonic source with complex amplitude and angular frequency."""

    def __init__(self, amplitude, omega, domain):
        self.amplitude = amplitude
        self.omega = omega
        self.domain = domain

    def on_grid(self, t=0.0):
        return self.amplitude * mx.exp(1j * self.omega * t)

    @staticmethod
    def from_point_sources(domain, x, y, value, omega):
        src_field = np.zeros(domain.N, dtype=np.complex64)
        src_field[x, y] = value
        return TimeHarmonicSource(mx.array(src_field), omega, domain)


# ── Sensors ───────────────────────────────────────────────────────────────────

class Sensors:
    """Sensor structure that records field values at given positions."""

    def __init__(self, positions):
        self.positions = positions

    def __call__(self, p: Field, u: Field, rho: Field):
        if len(self.positions) == 1:
            return p.on_grid[self.positions[0]]
        elif len(self.positions) == 2:
            return p.on_grid[self.positions[0], self.positions[1]]
        elif len(self.positions) == 3:
            return p.on_grid[self.positions[0], self.positions[1],
                             self.positions[2]]
        else:
            raise ValueError(
                f"Sensors positions must be 1, 2 or 3 dimensional. Not {len(self.positions)}"
            )


# ── BLI Sensors ───────────────────────────────────────────────────────────────

def bli_function(x0, x, n, include_imag=False):
    """Band limited interpolation function."""
    dx = mx.where(
        (x - x0[:, None]) == 0, mx.array(1.0),
        x - x0[:, None])
    dx_nonzero = (x - x0[:, None]) != 0

    if n % 2 == 0:
        y = mx.sin(mx.array(math.pi) * dx) / \
            mx.tan(mx.array(math.pi) * dx / n) / n
        y = y - mx.sin(mx.array(math.pi) * x0[:, None]) * mx.sin(mx.array(math.pi) * x) / n
        if include_imag:
            y = y + 1j * mx.cos(mx.array(math.pi) * x0[:, None]) * mx.sin(mx.array(math.pi) * x) / n
    else:
        y = mx.sin(mx.array(math.pi) * dx) / \
            mx.sin(mx.array(math.pi) * dx / n) / n

    all_nonzero = mx.all(dx_nonzero, axis=1)
    y = y * all_nonzero[:, None] + (1 - dx_nonzero.astype(mx.float32)) * (
        ~all_nonzero)[:, None]
    return y


class BLISensors:
    """Band-limited interpolant (off-grid) sensors."""

    def __init__(self, positions, n):
        self.positions = positions
        self.n = n

        x = mx.arange(n[0])[None]
        self.bx = mx.expand_dims(bli_function(positions[0], x, n[0]),
                                  axis=tuple(range(2, 2 + len(n))))

        if len(n) > 1:
            y = mx.arange(n[1])[None]
            self.by = mx.expand_dims(bli_function(positions[1], y, n[1]),
                                      axis=tuple(range(2, 2 + len(n) - 1)))
        else:
            self.by = None

        if len(n) > 2:
            z = mx.arange(n[2])[None]
            self.bz = mx.expand_dims(bli_function(positions[2], z, n[2]),
                                      axis=tuple(range(2, 2 + len(n) - 2)))
        else:
            self.bz = None

    def __call__(self, p: Field, u, v):
        if len(self.positions) == 1:
            return mx.sum(p.on_grid[None] * self.bx, axis=1)
        elif len(self.positions) == 2:
            pw = mx.sum(p.on_grid[None] * self.bx, axis=1)
            return mx.sum(pw * self.by, axis=1)
        elif len(self.positions) == 3:
            pw = mx.sum(p.on_grid[None] * self.bx, axis=1)
            pw = mx.sum(pw * self.by, axis=1)
            return mx.sum(pw * self.bz, axis=1)
        else:
            raise ValueError(
                f"Sensors positions must be 1, 2 or 3 dimensional. Not {len(self.positions)}"
            )


# ── TimeAxis ──────────────────────────────────────────────────────────────────

class TimeAxis:
    """Temporal vector for acoustic simulations."""

    def __init__(self, dt, t_end):
        self.dt = dt
        self.t_end = t_end

    @property
    def Nt(self):
        return np.ceil(self.t_end / self.dt)

    def to_array(self):
        out_steps = mx.arange(0, int(self.Nt), 1)
        return out_steps * self.dt

    @staticmethod
    def from_medium(medium: Medium, cfl: float = 0.3, t_end=None):
        dt = cfl * min(medium.domain.dx) / functional(medium.sound_speed)(
            np.max)
        if t_end is None:
            t_end = np.sqrt(
                sum((float(x[-1]) - float(x[0]))**2
                    for x in medium.domain.spatial_axis)) / functional(
                        medium.sound_speed)(np.min)
        return TimeAxis(dt=float(dt), t_end=float(t_end))
