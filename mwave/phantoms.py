"""Phantom generators ported from j-wave."""
import numpy as np
import mlx.core as mx

from mwave.geometry import circ_mask


def three_circles(N):
    """Generate a 3-circle phantom."""
    assert len(N) == 2, "N must be of length 2"

    radius = sum(N) / float(len(N))
    mask1 = circ_mask(N, radius * 0.05,
                      (int(N[0] / 2 + N[0] / 8), int(N[1] / 2)))
    mask2 = circ_mask(N, radius * 0.1,
                      (int(N[0] / 2 - N[0] / 8), int(N[1] / 2 + N[1] / 6)))
    mask3 = circ_mask(N, radius * 0.15, (int(N[0] / 2), int(N[1] / 2)))
    p0 = 5.0 * mask1 + 3.0 * mask2 + 4.0 * mask3
    return mx.array(np.expand_dims(p0, -1).astype(np.float32))
