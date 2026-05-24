"""
Pluggable distance metrics for the REVO weighted-ensemble driver.

Subclass DistanceMetric to implement a custom distance function and pass
the class name as 'distance_metric' in revo.cfg, or use one of the
built-in implementations:

    adaptive_sigma  (default)
        Sigma is computed from the ensemble and updated at the end of each
        iteration using exponential smoothing, so it slowly tracks the
        evolving ensemble without changing within an iteration. On the first
        iteration (or after a restart with no saved state) sigma is
        initialised from the current ensemble.

    zscore
        Sigma is recomputed from the current ensemble at the start of each
        iteration and held fixed within the iteration. No state is
        persisted between restarts.

    fixed
        A fixed normalization vector is supplied by the user via
        'sigma_fixed' in revo.cfg. Sigma never changes.

    euclidean
        Raw Euclidean distance with no normalization. Use when features
        are already on a common scale or normalization is handled upstream.
"""

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ensemble_sigma(features, n_copies):
    """Per-feature weighted std using n_copies as population weights.

    Walkers with n_copies <= 0 are excluded. Returns sigma >= 1e-8 per
    feature to guard against collapsed dimensions.
    """
    n_copies = np.asarray(n_copies, dtype=float)
    alive = n_copies > 0
    wt = n_copies[alive] / n_copies[alive].sum()
    mean = np.average(features[alive], weights=wt, axis=0)
    var = np.average((features[alive] - mean) ** 2, weights=wt, axis=0)
    return np.maximum(np.sqrt(var), 1e-8)


def _pairwise_distance(normed_features, importance):
    """Pairwise distance matrix from pre-normalized features.

    Parameters
    ----------
    normed_features : ndarray, shape (n, d)
    importance : ndarray, shape (d,) or None
        Per-feature importance weights applied inside the sqrt.
        If None, all features are weighted equally.

    Returns
    -------
    dist_matrix : ndarray, shape (n, n)
    """
    diff = normed_features[:, np.newaxis, :] - normed_features[np.newaxis, :, :]
    sq = diff ** 2
    if importance is not None:
        sq = sq * importance[np.newaxis, np.newaxis, :]
    return np.sqrt(sq.sum(axis=2))


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DistanceMetric:
    """Abstract base class for REVO distance metrics.

    The REVO planning loop calls compute() once at the start of each
    iteration to obtain a fixed distance matrix, which it uses throughout
    the planning phase.  update() is called at the end of each iteration
    with the post-resampling n_copies, allowing stateful metrics to track
    the evolving ensemble across iterations.

    To implement a custom metric subclass this class, override compute()
    (and optionally update / save / load), then register it by passing its
    name to _make_distance_metric.
    """

    @property
    def sigma(self):
        """Per-feature normalization scale, or None if not applicable.

        Exposed so the driver can log sigma and r/s values. Metrics that
        do not use sigma-based normalization should leave this as None.
        """
        return None

    def compute(self, features, n_copies=None):
        """Return the pairwise distance matrix for the current ensemble.

        Parameters
        ----------
        features : ndarray, shape (n_walkers, n_features)
        n_copies : ndarray, shape (n_walkers,), optional
            Current copy counts. Provided as a hint for weight-aware
            sigma computation; may be ignored.

        Returns
        -------
        dist_matrix : ndarray, shape (n_walkers, n_walkers)
        """
        raise NotImplementedError

    def update(self, features, n_copies):
        """Update internal state at the end of an iteration.

        Called after the planning loop with the final post-resampling
        n_copies (integer). Override if the metric maintains state that
        should track the ensemble across iterations.

        Parameters
        ----------
        features : ndarray, shape (n_walkers, n_features)
        n_copies : ndarray, shape (n_walkers,)
            Integer copy counts after resampling.
        """

    def save(self, path):
        """Persist state to disk. Override if the metric has state."""

    def load(self, path):
        """Restore state from disk. Override if the metric has state."""


# ---------------------------------------------------------------------------
# Built-in implementations
# ---------------------------------------------------------------------------

class AdaptiveSigmaDistance(DistanceMetric):
    """Sigma updated between iterations, fixed within each iteration.

    At the end of each iteration sigma is blended toward the current
    ensemble's sigma:

        sigma_new = update_rate * sigma_current + (1 - update_rate) * sigma

    The distance matrix returned by compute() uses the sigma from the
    *previous* update, so the metric is constant throughout the planning
    loop. This avoids scale changes between the test step and accepted-move
    bookkeeping.

    Parameters
    ----------
    update_rate : float
        Controls how quickly sigma tracks the ensemble. 0 freezes sigma
        after the first iteration; 1 is equivalent to ZScoreDistance.
        Default 0.1.
    importance : ndarray, shape (n_features,), optional
        Per-feature importance weights applied inside the sqrt.
    """

    def __init__(self, update_rate=0.1, importance=None):
        self.update_rate = float(update_rate)
        self.importance = np.asarray(importance, dtype=float) if importance is not None else None
        self._sigma = None

    @property
    def sigma(self):
        return self._sigma

    def compute(self, features, n_copies=None):
        if self._sigma is None:
            if n_copies is None:
                n_copies = np.ones(len(features))
            self._sigma = _ensemble_sigma(features, n_copies)
        return _pairwise_distance(features / self._sigma, self.importance)

    def update(self, features, n_copies):
        sigma_current = _ensemble_sigma(features, n_copies)
        if self._sigma is None:
            self._sigma = sigma_current
        else:
            self._sigma = (
                self.update_rate * sigma_current
                + (1.0 - self.update_rate) * self._sigma
            )

    def save(self, path):
        if self._sigma is not None:
            np.save(path, self._sigma)

    def load(self, path):
        path = Path(path)
        if path.exists():
            self._sigma = np.load(str(path))


class ZScoreDistance(DistanceMetric):
    """Fresh sigma computed from the current ensemble each iteration.

    Sigma is the ensemble-weighted standard deviation across walkers,
    computed at the start of each iteration and held fixed within it.
    No state is persisted between restarts.

    Parameters
    ----------
    importance : ndarray, shape (n_features,), optional
        Per-feature importance weights applied inside the sqrt.
    """

    def __init__(self, importance=None):
        self.importance = np.asarray(importance, dtype=float) if importance is not None else None
        self._sigma = None

    @property
    def sigma(self):
        return self._sigma

    def compute(self, features, n_copies=None):
        if n_copies is None:
            n_copies = np.ones(len(features))
        self._sigma = _ensemble_sigma(features, n_copies)
        return _pairwise_distance(features / self._sigma, self.importance)


class FixedNormDistance(DistanceMetric):
    """Sigma supplied by the user; constant for the entire simulation.

    Useful when the expected feature scales are known in advance and
    should not depend on the ensemble state.

    Parameters
    ----------
    sigma : array-like, shape (n_features,)
        Per-feature normalization values.
    importance : ndarray, shape (n_features,), optional
        Per-feature importance weights applied inside the sqrt.
    """

    def __init__(self, sigma, importance=None):
        self._sigma = np.asarray(sigma, dtype=float)
        self.importance = np.asarray(importance, dtype=float) if importance is not None else None

    @property
    def sigma(self):
        return self._sigma

    def compute(self, features, n_copies=None):
        return _pairwise_distance(features / self._sigma, self.importance)


class EuclideanDistance(DistanceMetric):
    """Raw Euclidean distance with no feature normalization.

    Use this when features are already on a common scale, or when
    normalization is handled upstream (e.g., in the progress coordinate
    pipeline). Importance weights can still be applied.

    Parameters
    ----------
    importance : ndarray, shape (n_features,), optional
        Per-feature importance weights applied inside the sqrt.
    """

    def __init__(self, importance=None):
        self.importance = np.asarray(importance, dtype=float) if importance is not None else None

    def compute(self, features, n_copies=None):
        return _pairwise_distance(features, self.importance)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _make_distance_metric(name, config):
    """Instantiate a DistanceMetric from a name and config dict.

    Parameters
    ----------
    name : str
        One of 'adaptive_sigma', 'zscore', 'fixed', 'euclidean'.
    config : dict
        REVO configuration dict (as returned by _load_revo_config).

    Returns
    -------
    DistanceMetric
    """
    importance = config.get("importance")
    if importance is not None:
        importance = np.asarray(importance, dtype=float)

    if name == "adaptive_sigma":
        return AdaptiveSigmaDistance(
            update_rate=config.get("sigma_update_rate", 0.1),
            importance=importance,
        )
    elif name == "zscore":
        return ZScoreDistance(importance=importance)
    elif name == "fixed":
        sigma_fixed = config.get("sigma_fixed")
        if sigma_fixed is None:
            raise ValueError(
                "distance_metric 'fixed' requires a 'sigma_fixed' list in revo.cfg."
            )
        return FixedNormDistance(sigma=sigma_fixed, importance=importance)
    elif name == "euclidean":
        return EuclideanDistance(importance=importance)
    else:
        raise ValueError(
            f"Unknown distance_metric '{name}'. "
            "Choose from: adaptive_sigma, zscore, fixed, euclidean."
        )
