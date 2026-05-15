"""
REVO core functions for WESTPA.

Provides the distance matrix computation, novelty function, and variation
calculation used by REVO_driver.py. The REVO greedy optimization loop
lives in the driver itself (matching the wepy implementation pattern).

Reference: Donyapour et al., J. Chem. Phys. 150, 244112 (2019)
"""

import numpy as np


def compute_distance_matrix(features, importance=None, sigmas=None):
    """Compute pairwise Euclidean distance matrix with variance normalization.

    Parameters
    ----------
    features : ndarray, shape (n_walkers, n_features)
    importance : ndarray, shape (n_features,), optional
        Per-feature importance weights applied after normalization.
        Scales each dimension's contribution to the distance.
        If None, all features weighted equally (1.0).
    sigmas : ndarray, shape (n_features,), optional
        Per-feature standard deviations for normalization.
        If None, computed from the features.

    Returns
    -------
    dist_matrix : ndarray, shape (n_walkers, n_walkers)
    sigmas : ndarray, shape (n_features,)
    """
    if sigmas is None:
        sigmas = np.std(features, axis=0)
        sigmas[sigmas < 1e-12] = 1.0
    
    if importance is None:
        importance = np.ones(features.shape[1])
    normed = features / sigmas
    diff = normed[:, np.newaxis, :] - normed[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum((diff ** 2) * importance[np.newaxis, np.newaxis, :], axis=2))

    return dist_matrix, sigmas


def novelty(weight, n_copies, pmin=1e-12, use_weights=True):
    """Compute the novelty function for a walker.

    phi(w, n) = max(0, log(w/n) - log(pmin/100))

    Parameters
    ----------
    weight : float
    n_copies : float or int
        Number of copies (can be fractional for probability-weighted test).
    pmin : float
    use_weights : bool
        If False, return 1.0 (unweighted REVO).
    """
    if weight <= 0 or n_copies <= 0:
        return 0.0
    if not use_weights:
        return 1.0
    val = np.log(weight / n_copies) - np.log(pmin / 100)
    return max(0.0, val)


def calc_variation(weights, n_copies, dist_matrix, char_dist, dist_exponent=4,
                   pmin=1e-12, use_weights=True):
    """Compute total ensemble variation and per-walker contributions.

    V = sum_{i<j} (d_ij / d_0)^alpha * phi_i * phi_j * n_i * n_j

    Parameters
    ----------
    weights : ndarray, shape (n_walkers,)
    n_copies : ndarray, shape (n_walkers,)
        Can be int (actual) or float (probability-weighted for testing).
    dist_matrix : ndarray, shape (n_walkers, n_walkers)
    char_dist : float
    dist_exponent : int
    pmin : float
    use_weights : bool

    Returns
    -------
    variation : float
    walker_variations : ndarray, shape (n_walkers,)
    """
    n = len(weights)
    walker_vars = np.zeros(n)
    total_var = 0.0

    phi = np.array([novelty(weights[i], n_copies[i], pmin, use_weights) for i in range(n)])

    for i in range(n):
        if n_copies[i] <= 0:
            continue
        for j in range(i + 1, n):
            if n_copies[j] <= 0:
                continue
            d_scaled = (dist_matrix[i, j] / char_dist) ** dist_exponent
            contrib = d_scaled * phi[i] * phi[j]
            pair_var = contrib * n_copies[i] * n_copies[j]
            total_var += pair_var
            walker_vars[i] += contrib * n_copies[j]
            walker_vars[j] += contrib * n_copies[i]

    return total_var, walker_vars
