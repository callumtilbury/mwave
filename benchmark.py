"""
Benchmark: j-wave (JAX/CPU) vs m-wave (MLX/GPU)
Runs identical wave propagation simulations and compares wall-clock time + correctness.
"""
import time
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

GRID_SIZES = [64, 128, 256, 512, 1024, 2048]
CFL = 0.3
C0 = 1500.0
DX = 1e-4
N_WARMUP = 1       # warmup runs (not timed)
N_REPEATS = 3      # timed runs
Nt_FIXED = 100     # fixed number of timesteps

print("=" * 70)
print("  j-wave (JAX/CPU) vs m-wave (MLX/GPU) — Wave Propagation Benchmark")
print("=" * 70)


# ── Shared initial condition ─────────────────────────────────────────────────

def make_p0_numpy(N):
    """Create a disc initial pressure field as a numpy array."""
    r = np.sqrt((np.arange(N) - N//2)**2 + (np.arange(N)[:, None] - N//2)**2)
    return (r < N//8).astype(np.float32).T


# ── j-wave runner ────────────────────────────────────────────────────────────

def run_jwave(N):
    import jax.numpy as jnp
    from jwave import Domain, Medium, TimeAxis
    from jwave.acoustics.time_varying import simulate_wave_propagation
    from jaxdf import FourierSeries

    domain = Domain(N=(N, N), dx=(DX, DX))
    medium = Medium(domain=domain, sound_speed=C0)
    time_axis = TimeAxis.from_medium(medium, cfl=CFL, t_end=Nt_FIXED * DX * CFL / C0)

    p0_np = make_p0_numpy(N)
    p0 = FourierSeries(jnp.array(p0_np[:, :, None]), domain)

    # Warmup
    for _ in range(N_WARMUP):
        result = simulate_wave_propagation(medium, time_axis, p0=p0)
        result.params.block_until_ready()

    # Timed runs
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        result = simulate_wave_propagation(medium, time_axis, p0=p0)
        result.params.block_until_ready()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    output = np.array(result.params)  # (Nt, N, N, 1)
    return times, int(time_axis.Nt), output


# ── m-wave runner ────────────────────────────────────────────────────────────

def run_mwave(N):
    import mlx.core as mx
    from mwave import FourierSeries, Domain, Medium, TimeAxis
    from mwave.acoustics.time_varying import simulate_wave_propagation

    domain = Domain(N=(N, N), dx=(DX, DX))
    medium = Medium(domain=domain, sound_speed=C0)
    time_axis = TimeAxis.from_medium(medium, cfl=CFL, t_end=Nt_FIXED * DX * CFL / C0)

    p0_np = make_p0_numpy(N)
    p0 = FourierSeries(mx.array(p0_np[:, :, None]), domain)

    # Warmup
    for _ in range(N_WARMUP):
        result = simulate_wave_propagation(medium, time_axis, p0=p0)
        mx.eval(result.params)

    # Timed runs
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        result = simulate_wave_propagation(medium, time_axis, p0=p0)
        mx.eval(result.params)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    output = np.array(result.params)  # (Nt, N, N, 1)
    return times, int(time_axis.Nt), output


# ── Run benchmarks ───────────────────────────────────────────────────────────

import jax
import mlx.core as mx
print(f"\nJAX devices: {jax.devices()}")
print(f"MLX device:  {mx.default_device()}")
print()

results = []

for N in GRID_SIZES:
    print(f"─── Grid {N}×{N}, {Nt_FIXED} timesteps ───")

    # j-wave
    print(f"  j-wave (JAX/CPU) ...", end=" ", flush=True)
    jw_times, actual_nt, jw_output = run_jwave(N)
    jw_mean = np.mean(jw_times)
    jw_std = np.std(jw_times)
    print(f"{jw_mean:.3f} ± {jw_std:.3f} s")

    # m-wave
    print(f"  m-wave (MLX/GPU) ...", end=" ", flush=True)
    mw_times, actual_nt, mw_output = run_mwave(N)
    mw_mean = np.mean(mw_times)
    mw_std = np.std(mw_times)
    print(f"{mw_mean:.3f} ± {mw_std:.3f} s")

    # Speedup
    speedup = jw_mean / mw_mean if mw_mean > 0 else float('inf')
    print(f"  → Speedup: {speedup:.2f}× {'(MLX faster)' if speedup > 1 else '(JAX faster)'}")

    # ── Correctness comparison ──
    # Both outputs are (Nt, N, N, 1) numpy arrays
    min_t = min(jw_output.shape[0], mw_output.shape[0])
    jw_slice = jw_output[:min_t]
    mw_slice = mw_output[:min_t]

    max_abs_diff = np.max(np.abs(jw_slice - mw_slice))
    max_val = max(np.max(np.abs(jw_slice)), np.max(np.abs(mw_slice)), 1e-30)
    max_rel_diff = max_abs_diff / max_val

    # Per-timestep comparison at a few snapshots
    snapshots = [0, min_t // 4, min_t // 2, min_t - 1]
    snap_diffs = []
    for ti in snapshots:
        diff = np.max(np.abs(jw_slice[ti] - mw_slice[ti]))
        jmax = np.max(np.abs(jw_slice[ti]))
        mmax = np.max(np.abs(mw_slice[ti]))
        snap_diffs.append((ti, diff, jmax, mmax))

    match_ok = max_rel_diff < 0.05  # within 5% relative error
    status = "✅ MATCH" if match_ok else "⚠️  MISMATCH"

    print(f"  {status}  max|diff|={max_abs_diff:.6e}  rel={max_rel_diff:.6e}")
    print(f"           Timestep snapshots:")
    for ti, diff, jmax, mmax in snap_diffs:
        print(f"             t={ti:>4}: |diff|={diff:.4e}  j-wave max={jmax:.4e}  m-wave max={mmax:.4e}")
    print()

    results.append({
        "N": N, "Nt": actual_nt,
        "jwave_mean": jw_mean, "jwave_std": jw_std,
        "mwave_mean": mw_mean, "mwave_std": mw_std,
        "speedup": speedup,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "match": match_ok,
    })


# ── Summary table ────────────────────────────────────────────────────────────

print("=" * 86)
print(f"{'Grid':>10} {'Nt':>5} {'j-wave (s)':>11} {'m-wave (s)':>11} {'Speedup':>9} {'Max |Δ|':>12} {'Match':>7}")
print("-" * 86)
for r in results:
    print(f"{r['N']}×{r['N']:>4} {r['Nt']:>5} "
          f"{r['jwave_mean']:>9.3f}   {r['mwave_mean']:>9.3f}   {r['speedup']:>7.2f}× "
          f"{r['max_abs_diff']:>11.4e}  {'✅' if r['match'] else '⚠️'}")
print("=" * 86)
print(f"\nJAX backend: {jax.default_backend()} | MLX device: {mx.default_device()}")
