"""
Focused Ultrasound Tutorial — m-wave (MLX on Apple Silicon)
Adapted from the j-wave tutorial by Callum Rhys Tilbury.

Run with: python tutorial.py
"""

import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from mwave import FourierSeries, Domain, Medium, TimeAxis
from mwave.acoustics.time_varying import simulate_wave_propagation
from mwave.geometry import Sources

# ────────────────────────────────────────────────────────────────────────────
# Simulation Config
# ────────────────────────────────────────────────────────────────────────────

cfl = 0.3
ppw = 5          # Points per wavelength

c0 = 1500        # [m/s] Speed of sound (water)
rho0 = 1000      # [kg/m³] Density

Dx = 0.1         # [m] Domain width
Dy = 0.1         # [m] Domain height

f0 = 500e3       # [Hz] Ultrasound frequency
wavelen = c0 / f0

dx = wavelen / ppw
dy = dx

Nx = int(Dx / dx)
Ny = int(Dy / dy)

print(f"Grid: {Nx}×{Ny}, dx={dx*1e3:.2f} mm, λ={wavelen*1e3:.2f} mm")

# ────────────────────────────────────────────────────────────────────────────
# Create domain, medium, time axis
# ────────────────────────────────────────────────────────────────────────────

domain = Domain(N=(Nx, Ny), dx=(dx, dy))
medium = Medium(domain=domain, sound_speed=c0, density=rho0)

t_end = Dx / c0
t_axis = TimeAxis.from_medium(medium=medium, cfl=cfl, t_end=t_end)
Nt = int(t_axis.Nt)
t_array = np.arange(Nt) * t_axis.dt

print(f"Time: dt={t_axis.dt:.2e} s, Nt={Nt}, t_end={t_end:.2e} s")

# ────────────────────────────────────────────────────────────────────────────
# Define a single-cycle pulse
# ────────────────────────────────────────────────────────────────────────────

pulse = np.where(
    t_array < (1 / f0),
    np.sin(2 * np.pi * f0 * t_array),
    0.0,
).astype(np.float32)

plt.figure(figsize=(8, 3))
plt.plot(t_array * 1e6, pulse)
plt.xlabel("Time [µs]")
plt.ylabel("Amplitude")
plt.title("Transmitted Pulse")
plt.tight_layout()
plt.savefig("01_pulse.png", dpi=150)
plt.close()
print("✓ Saved 01_pulse.png")

# ────────────────────────────────────────────────────────────────────────────
# Helper: percentage-based coordinates
# ────────────────────────────────────────────────────────────────────────────

def loc_perc_to_i(loc_perc):
    return (loc_perc * np.array([[Nx, Ny]]) // 100).astype(int)


def make_sources(loc_i, signals):
    """Create Sources from integer grid positions and signal array."""
    return Sources(
        positions=(list(loc_i[:, 0]), list(loc_i[:, 1])),
        signals=mx.array(signals),
        domain=domain,
        dt=t_axis.dt,
    )

# ────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ────────────────────────────────────────────────────────────────────────────

def plot_frames(result, t_indices, filename, title_prefix=""):
    data = np.array(result.params)
    vmax = np.max(np.abs(data))
    extent = [0, Nx * dx * 1e3, 0, Ny * dy * 1e3]

    _, axs = plt.subplots(1, len(t_indices), figsize=(5 * len(t_indices), 4))
    if len(t_indices) == 1:
        axs = [axs]
    for i, ti in enumerate(t_indices):
        axs[i].imshow(
            data[ti, :, :, 0].T,
            vmax=vmax, vmin=-vmax,
            extent=extent, origin="lower", cmap="seismic",
        )
        axs[i].set_title(f"{title_prefix}t={ti * t_axis.dt * 1e6:.1f} µs")
        axs[i].set_xlabel("x [mm]")
    axs[0].set_ylabel("y [mm]")
    plt.colorbar(axs[-1].images[0], ax=axs, shrink=0.6, label="Pressure")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✓ Saved {filename}")


def max_pressure_plot(result, filename, target_i=None):
    data = np.array(result.params)
    extent = [0, Nx * dx * 1e3, 0, Ny * dy * 1e3]
    max_p = np.max(np.abs(data[:, :, :, 0]), axis=0)

    plt.figure(figsize=(6, 5))
    plt.imshow(max_p.T, extent=extent, origin="lower")
    if target_i is not None:
        plt.scatter(
            target_i[:, 0] * dx * 1e3,
            target_i[:, 1] * dy * 1e3,
            marker="o", color="r", s=100, label="Target",
        )
        plt.legend()
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.colorbar(label="Max |p|")
    plt.title("Maximum Pressure Over Time")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✓ Saved {filename}")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Single Point Source
# ═══════════════════════════════════════════════════════════════════════════

print("\n── 1. Single Point Source ──")

tx1_loc_i = loc_perc_to_i(np.array([[50, 50]]))
tx1 = make_sources(tx1_loc_i, pulse[np.newaxis, :])

result1 = simulate_wave_propagation(medium, t_axis, sources=tx1)
print(f"  Result shape: {result1.params.shape}")

plot_frames(result1, [10, Nt // 4, Nt // 2], "02_point_source.png")
max_pressure_plot(result1, "03_point_source_max.png")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Line Array (Multiple Sources)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── 2. Line Array ──")

txA_loc_perc = np.array([[20, y] for y in range(30, 71, 10)])
txA_loc_i = loc_perc_to_i(txA_loc_perc)
txA = make_sources(txA_loc_i, np.tile(pulse, (len(txA_loc_i), 1)))

resultA = simulate_wave_propagation(medium, t_axis, sources=txA)
plot_frames(resultA, [10, 50, Nt // 2, Nt - 1], "04_line_array.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Phased Array Focusing (Ray Tracing)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── 3. Phased Array (Ray Tracing) ──")

Ntx = 16
tx_loc_perc = np.array([[20, int(y)] for y in np.linspace(30, 70, Ntx)])
target_loc_perc = np.array([[70, 50]])

def ray_tracing(tx_loc_perc, target_loc_perc):
    """Apply time-delays to focus on a target."""
    target_i = loc_perc_to_i(target_loc_perc)
    tx_i = loc_perc_to_i(tx_loc_perc)

    diffs_m = (target_i - tx_i) * np.array([dx, dy])
    dist_m = np.linalg.norm(diffs_m, axis=-1)

    delays_s = dist_m / c0
    delays_i = (delays_s / t_axis.dt).astype(int)
    delays_i = np.max(delays_i) - delays_i

    shifted = np.array([np.roll(pulse, d) for d in delays_i])
    tx = make_sources(tx_i, shifted)

    return simulate_wave_propagation(medium, t_axis, sources=tx)


result_focus = ray_tracing(tx_loc_perc, target_loc_perc)
target_i = loc_perc_to_i(target_loc_perc)

plot_frames(result_focus, [10, 50, Nt // 3, Nt // 2], "05_phased_focus.png")
max_pressure_plot(result_focus, "06_phased_focus_max.png", target_i=target_i)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Heterogeneous Medium
# ═══════════════════════════════════════════════════════════════════════════

print("\n── 4. Heterogeneous Medium ──")

c1 = 2800.0     # Speed of sound in aberrator [m/s]
rho1 = 1850.0   # Density in aberrator [kg/m³]

c_map = np.ones((Nx, Ny), dtype=np.float32) * c0
rho_map = np.ones((Nx, Ny), dtype=np.float32) * rho0

# Insert aberrator layer
x_start = int(Nx * 36 / 100)
x_end = int(Nx * 44 / 100)
c_map[x_start:x_end, :] = c1
rho_map[x_start:x_end, :] = rho1

medium_het = Medium(
    domain=domain,
    sound_speed=mx.array(c_map[:, :, np.newaxis]),
    density=mx.array(rho_map[:, :, np.newaxis]),
)

# Plot speed of sound map
extent = [0, Nx * dx * 1e3, 0, Ny * dy * 1e3]
plt.figure(figsize=(6, 5))
plt.imshow(c_map.T, extent=extent, origin="lower")
plt.colorbar(label="Sound Speed [m/s]")
plt.xlabel("x [mm]")
plt.ylabel("y [mm]")
plt.title("Heterogeneous Speed of Sound Map")
plt.tight_layout()
plt.savefig("07_het_medium.png", dpi=150)
plt.close()
print("✓ Saved 07_het_medium.png")

# Simulate focusing through aberrator
target_loc_perc_het = np.array([[70, 70]])


def ray_tracing_het(tx_loc_perc, target_loc_perc, med):
    """Ray tracing with custom medium."""
    target_i = loc_perc_to_i(target_loc_perc)
    tx_i = loc_perc_to_i(tx_loc_perc)

    diffs_m = (target_i - tx_i) * np.array([dx, dy])
    dist_m = np.linalg.norm(diffs_m, axis=-1)

    delays_s = dist_m / c0  # Note: still using c0 for delay computation
    delays_i = (delays_s / t_axis.dt).astype(int)
    delays_i = np.max(delays_i) - delays_i

    shifted = np.array([np.roll(pulse, d) for d in delays_i])
    tx = make_sources(tx_i, shifted)

    return simulate_wave_propagation(med, t_axis, sources=tx)


result_het = ray_tracing_het(tx_loc_perc, target_loc_perc_het, medium_het)
target_het_i = loc_perc_to_i(target_loc_perc_het)

plot_frames(result_het, [10, 50, Nt // 3, Nt // 2], "08_het_focus.png")
max_pressure_plot(result_het, "09_het_focus_max.png", target_i=target_het_i)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Time Reversal
# ═══════════════════════════════════════════════════════════════════════════

print("\n── 5. Time Reversal ──")

def time_reversal(tx_loc_perc, target_loc_perc, med):
    """Time-reversal focusing through heterogeneous medium."""
    target_i = loc_perc_to_i(target_loc_perc)
    tx_i = loc_perc_to_i(tx_loc_perc)

    # Step 1: Beacon — emit from target, record at sources
    beacon = make_sources(target_i, pulse[np.newaxis, :])
    beacon_result = simulate_wave_propagation(med, t_axis, sources=beacon)

    # Record signals at transmitter locations
    beacon_data = np.array(beacon_result.params)  # (Nt, Nx, Ny, 1)
    beacon_signals = np.array([
        beacon_data[:, tx_i[j, 0], tx_i[j, 1], 0]
        for j in range(len(tx_i))
    ])  # (Ntx, Nt)

    # Step 2: Time-reverse the signals
    reversed_signals = beacon_signals[:, ::-1].copy()

    # Step 3: Transmit the reversed signals
    tx = make_sources(tx_i, reversed_signals)
    tr_result = simulate_wave_propagation(med, t_axis, sources=tx)

    return beacon_result, tr_result


beacon_result, tr_result = time_reversal(
    tx_loc_perc, target_loc_perc_het, medium_het
)

plot_frames(beacon_result, [10, Nt // 4, Nt // 2, Nt - 1],
            "10_beacon.png", title_prefix="Beacon: ")

plot_frames(tr_result, [10, Nt // 4, Nt // 2, Nt - 1],
            "11_time_reversal.png", title_prefix="TR: ")

max_pressure_plot(tr_result, "12_time_reversal_max.png",
                  target_i=target_het_i)


# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Tutorial complete! Generated 12 figures.")
print("=" * 60)
