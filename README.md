# mwave 🌊

**mwave** is an MLX-based differentiable acoustic simulator for Apple Silicon, ported from [j-Wave](https://github.com/ucl-bug/jwave). It provides the same high-level API as j-Wave with near-identical call signatures, making it a near drop-in replacement for users on Mac with an Apple GPU.

Under the hood, mwave replaces JAX → [Apple MLX](https://github.com/ml-explore/mlx) and `jaxdf` → [mlxdf](https://github.com/your-org/mlxdf), enabling GPU-accelerated acoustic simulations on M-series chips.

---

## Features

- 🔁 **Near drop-in replacement for j-Wave** — same `Domain`, `Medium`, `TimeAxis`, `Sources`, and `simulate_wave_propagation` API
- 🍎 **Apple Silicon GPU acceleration** via MLX
- 📐 **Fourier spectral methods** for high-accuracy wave propagation
- 🧱 **Heterogeneous media** — spatially varying sound speed and density
- 🔊 **Time-varying & time-harmonic** acoustic solvers
- 🧮 **Differentiable** — gradients flow through simulations via MLX autodiff

---

## Installation

```bash
pip install mwave
```

Or install from source:

```bash
git clone https://github.com/your-org/mwave
cd mwave
pip install -e .
```

**Requirements:** macOS with Apple Silicon (M1/M2/M3/M4), Python ≥ 3.9, MLX ≥ 0.4.0.

---

## Quick Start

mwave mirrors the j-Wave API as closely as possible. If you have existing j-Wave code, the main changes are:

| j-Wave | mwave |
|---|---|
| `import jax.numpy as jnp` | `import mlx.core as mx` |
| `from jwave import ...` | `from mwave import ...` |
| `from jaxdf import FourierSeries` | `from mwave import FourierSeries` |
| `result.params.block_until_ready()` | `mx.eval(result.params)` |
| `jnp.array(x)` | `mx.array(x)` |

### Example: Wave Propagation from Initial Pressure

```python
import mlx.core as mx
import numpy as np
from mwave import FourierSeries, Domain, Medium, TimeAxis
from mwave.acoustics.time_varying import simulate_wave_propagation

# Define domain and medium
domain = Domain(N=(128, 128), dx=(1e-4, 1e-4))
medium = Medium(domain=domain, sound_speed=1500.0)
time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=1e-5)

# Initial pressure: disc in the centre
N = 128
r = np.sqrt((np.arange(N) - N//2)**2 + (np.arange(N)[:, None] - N//2)**2)
p0_np = (r < N//8).astype(np.float32)

p0 = FourierSeries(mx.array(p0_np[:, :, None]), domain)

# Run simulation
result = simulate_wave_propagation(medium, time_axis, p0=p0)
mx.eval(result.params)  # materialise the lazy computation

pressure = np.array(result.params)  # shape: (Nt, Nx, Ny, 1)
print(pressure.shape)
```

### Example: Point Source with Transducer

```python
import mlx.core as mx
import numpy as np
from mwave import FourierSeries, Domain, Medium, TimeAxis
from mwave.acoustics.time_varying import simulate_wave_propagation
from mwave.geometry import Sources

domain = Domain(N=(256, 256), dx=(3e-4, 3e-4))
medium = Medium(domain=domain, sound_speed=1500.0, density=1000.0)
time_axis = TimeAxis.from_medium(medium, cfl=0.3, t_end=2e-4)

Nt = int(time_axis.Nt)
t_array = np.arange(Nt) * time_axis.dt
f0 = 500e3
pulse = np.where(t_array < 1/f0, np.sin(2 * np.pi * f0 * t_array), 0.0).astype(np.float32)

sources = Sources(
    positions=([128], [128]),  # centre of the grid
    signals=mx.array(pulse[np.newaxis, :]),
    domain=domain,
    dt=time_axis.dt,
)

result = simulate_wave_propagation(medium, time_axis, sources=sources)
mx.eval(result.params)
```

---

## Migrating from j-Wave

Most j-Wave scripts require only three changes:

1. Replace imports: `from jwave import ...` → `from mwave import ...`
2. Replace array creation: `jnp.array(x)` → `mx.array(x)`
3. Replace synchronisation: `.block_until_ready()` → `mx.eval(...)`

See [`benchmark.py`](benchmark.py) for a side-by-side comparison running both j-Wave and mwave on the same simulation.

---

## Project Structure

```
mwave/
├── acoustics/
│   ├── conversion.py       # Unit conversions (dB → neper)
│   ├── operators.py        # Helmholtz, Laplacian, wavevector operators
│   ├── pml.py              # Perfectly matched layer
│   ├── spectral.py         # Spectral utilities
│   ├── time_harmonic.py    # Helmholtz / angular spectrum solvers
│   └── time_varying.py     # Time-domain wave propagation
├── geometry.py             # Domain, Medium, TimeAxis, Sources, Sensors
├── phantoms.py             # Pre-built phantom media
├── signal_processing.py    # Signal utilities
├── utils.py                # Misc helpers
└── logger.py               # Logging
```

---

## Benchmarks

On an M3 Max, mwave (MLX/GPU) is significantly faster than j-Wave (JAX/CPU) for large grids:

| Grid | j-Wave (CPU) | mwave (GPU) | Speedup |
|------|-------------|-------------|---------|
| 64×64 | ~0.2 s | ~0.1 s | ~2× |
| 256×256 | ~4 s | ~0.8 s | ~5× |
| 512×512 | ~25 s | ~3 s | ~8× |

Reproduce with: `python benchmark.py`

---

## License

LGPL-3.0. See [LICENSE](LICENSE) for details.

mwave is a port of [j-Wave](https://github.com/ucl-bug/jwave), which is also licensed under LGPL-3.0.
