import numpy as np


def db2neper(alpha, y):
    r"""Transforms absorption units from decibels to nepers.
    See http://www.k-wave.org/documentation/db2neper.php

    Args:
        alpha: Absorption coefficient in decibels.
        y: Exponent of the absorption coefficient.

    Returns:
        Absorption coefficient in nepers.
    """
    return 100 * alpha * ((1e-6 / (2 * np.pi))**y) / (20 * np.log10(np.exp(1)))
