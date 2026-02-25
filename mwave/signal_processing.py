"""Signal processing utilities ported from j-wave to MLX."""
import numpy as np
import mlx.core as mx


def analytic_signal(x, axis=-1):
    """Compute analytic signal from real signal using FFT."""
    x_np = np.array(x)
    spectrum = np.fft.fft(x_np, axis=axis)

    # Zero negative frequencies
    positive_indices = slice(0, spectrum.shape[axis] // 2)
    slices = [slice(None)] * spectrum.ndim
    slices[axis] = positive_indices
    spectrum[tuple(slices)] = 0.0
    spectrum = spectrum * 2.0

    result = np.fft.ifft(spectrum, axis=axis)
    return mx.array(result)


def fourier_downsample(x, subsample=2, discard_last=True):
    """Downsample signal via Fourier truncation."""
    if subsample == 1:
        return x

    x_np = np.array(x)

    def _single_downsample(arr):
        Fx = np.fft.fftshift(np.fft.fftn(arr))
        cuts = [int((subsample - 1) * s / 2 / subsample) for s in Fx.shape]
        slices = tuple([slice(cut, -cut) for cut in cuts])
        return np.fft.ifftn(np.fft.ifftshift(Fx[slices])) / (subsample**arr.ndim)

    if discard_last:
        results = []
        for i in range(x_np.shape[-1]):
            results.append(_single_downsample(x_np[..., i]))
        result = np.stack(results, axis=-1)
    else:
        result = _single_downsample(x_np)

    return mx.array(result)


def fourier_upsample(x, upsample=2, discard_last=True):
    """Upsample signal via Fourier zero-padding."""
    if upsample == 1:
        return x

    x_np = np.array(x)

    def _single_upsample(arr):
        new_size = [s * upsample for s in arr.shape]
        Fx = np.fft.fftshift(np.fft.fftn(arr))
        new_Fx = np.zeros(new_size, dtype=Fx.dtype)
        cuts = [(ns - os) // 2 for os, ns in zip(Fx.shape, new_size)]
        slices = tuple([slice(c, c + os) for c, os in zip(cuts, Fx.shape)])
        new_Fx[slices] = Fx
        return np.fft.ifftn(np.fft.ifftshift(new_Fx)) * (upsample**arr.ndim)

    if discard_last:
        results = []
        for i in range(x_np.shape[-1]):
            results.append(_single_upsample(x_np[..., i]))
        result = np.stack(results, axis=-1)
    else:
        result = _single_upsample(x_np)

    return mx.array(result)


def apply_ramp(signal, dt, center_freq, warmup_cycles=3):
    """Apply ramp to signal."""
    if center_freq <= 0:
        raise ValueError(f"Center frequency must be positive, got {center_freq}")

    t = mx.arange(signal.shape[0]) * dt
    period = 1 / center_freq
    ramp_length = warmup_cycles * period
    return signal * mx.where(t < ramp_length, t / ramp_length, mx.ones_like(t))


def blackman(N):
    """Blackman window of length N."""
    i = np.arange(N, dtype=np.float32)
    return mx.array(
        0.42 - 0.5 * np.cos(2 * np.pi * i / N) +
        0.08 * np.cos(4 * np.pi * i / N)
    )


def gaussian_window(signal, time, mu, sigma):
    """Gaussian window."""
    return signal * mx.exp(-((time - mu)**2) / sigma**2)


def smooth(x, exponent=1.0):
    """Smooth an n-dimensional signal by multiplying its spectrum
    by a blackman window."""
    x_np = np.array(x)
    dimensions = x_np.shape
    axis = [np.array(blackman(d)) for d in dimensions]

    if len(dimensions) == 1:
        fk = np.fft.fftshift(axis[0])
    elif len(axis) == 2:
        fk = np.fft.fftshift(np.outer(*axis))
    elif len(axis) == 3:
        fk_2d = np.outer(*axis[1:])
        third = np.expand_dims(np.expand_dims(axis[0], 1), 2)
        fk = np.fft.fftshift(third * fk_2d)
    else:
        fk = np.fft.fftshift(axis[0])

    fk = fk**exponent
    result = np.fft.ifftn(fk * np.fft.fftn(x_np)).real
    return mx.array(result.astype(np.float32))


def tone_burst(sample_freq, signal_freq, num_cycles):
    """Generate a tone burst signal."""
    def gaussian(x, magnitude, mean, variance):
        return magnitude * np.exp(-((x - mean)**2) / (2 * variance))

    dt = 1 / sample_freq
    tone_length = num_cycles / signal_freq
    tone_t = np.arange(0, tone_length + dt, dt)
    burst = np.sin(2 * np.pi * signal_freq * tone_t)

    x_lim = 3
    window_x = np.linspace(-x_lim, x_lim, burst.shape[0])
    window = gaussian(window_x, 1, 0, 1)
    return mx.array((burst * window).astype(np.float32))
