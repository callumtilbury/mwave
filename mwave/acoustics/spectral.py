import numpy as np
import mlx.core as mx

from mlxdf.geometry import Domain


def kspace_op(domain: Domain, c_ref: float, dt: float):
    r"""Returns the k-space operator for the given domain and reference
    speed of sound.

    $$\kappa = \text{sinc}(c_{ref} k \Delta t / 2)$$

    Args:
        domain: The simulation domain.
        c_ref: Reference speed of sound.
        dt: Time step.

    Returns:
        dict with 'k_vec' and 'k_space_op'.
    """
    def f(N, dx):
        return mx.array(np.fft.fftfreq(N, dx).astype(np.float32)) * 2 * np.pi

    k_vec = [f(n, delta) for n, delta in zip(domain.N, domain.dx)]

    # Building k-space operator
    K = mx.stack(mx.meshgrid(*k_vec, indexing="ij"))
    k_magnitude = mx.sqrt(mx.sum(K**2, axis=0))

    # sinc(x) = sin(pi*x) / (pi*x) — numpy convention
    arg = c_ref * k_magnitude * dt / (2 * np.pi)
    k_space_op = mx.array(np.sinc(np.array(arg)))

    return {"k_vec": k_vec, "k_space_op": k_space_op}
