# WESTPA REVO Driver

[WESTPA](https://github.com/westpa/westpa) implementation of the Resampling of Ensembles by Variation Optimisation (REVO) method, first described in:

> Donyapour, N. et al. *The Journal of Chemical Physics*, 150, 244112 (2019). https://doi.org/10.1063/1.5100521

This implementation is based on the reference implementation in [WEPY](https://github.com/ADicksonLab/wepy) (Copyright © 2017, 2020 ADicksonLab).

## Overview

REVO replaces WESTPA's default bin-based resampler with a diversity-optimising algorithm. Each iteration it greedily identifies clone/merge pairs that maximise an ensemble *variation* score — a weighted sum of pairwise distances scaled by walker novelty. The goal is to prevent the walker population from collapsing onto a single configuration, which is particularly important near kinetic barriers.

The driver consists of three files:

| File | Role |
|------|------|
| `REVO_driver.py` | WESTPA `WEDriver` subclass; planning and execution phases |
| `revo_resampler.py` | `calc_variation` and `novelty` — pure functions, no WESTPA dependency |
| `revo_distance.py` | Pluggable distance metric interface and built-in implementations |

## Installation

Copy the three Python files into your WESTPA simulation root (the directory containing `west.cfg`) and register the driver:

```yaml
# west.cfg
west:
  drivers:
    module_path: $WEST_SIM_ROOT
    we_driver: REVO_driver.REVODriver
```

Requires: `numpy`, `pyyaml`, and a working WESTPA installation.

## Configuration

Driver parameters are read from a YAML file. By default the driver looks for `revo.cfg` next to `REVO_driver.py`. Point to a different file with the `REVO_CONFIG` environment variable.

Only `feature_names` is required — everything else falls back to the defaults shown below.

```yaml
# Names of the progress coordinate dimensions used as REVO features.
# Must match the order and length of the pcoord returned by your propagator.
feature_names:
  - feature_0
  - feature_1

# --- Algorithm parameters ---
pmin: 1.0e-12          # minimum weight floor for the novelty function
pmax: 0.1              # maximum combined weight allowed for a merge pair
dist_exponent: 4       # exponent on (d / char_dist) in the variation sum
merge_dist_fraction: 0.5  # merge pairs must be within this fraction of mean_dist
use_weights: true      # weight novelty by walker probability (false = unweighted REVO)
merge_alg: pairs       # 'pairs': minimise variation loss; 'greedy': lowest-Vi first
char_dist: 1.0         # characteristic distance in the variation formula

# --- Distance metric ---
distance_metric: adaptive_sigma  # see Distance Metrics section below
sigma_update_rate: 0.1           # smoothing rate for adaptive_sigma (0–1)
sigma_state_file: revo_sigma_state.npy  # where to persist sigma between restarts

# --- Feature importance ---
# Per-feature weights applied inside the distance sqrt.
# A scalar list of length n_features, or null for equal weighting.
importance: null
```

## Distance Metrics

The distance metric determines how pairwise distances between walkers are computed. Select one via `distance_metric` in `revo.cfg`.

### `adaptive_sigma` (default)

Sigma (per-feature standard deviation) is computed from the ensemble at the end of each iteration and updated via exponential smoothing:

```
sigma_new = update_rate * sigma_current + (1 - update_rate) * sigma
```

Sigma is **fixed within each iteration**, so the distance metric does not change during the planning loop. This keeps the acceptance criterion on a consistent scale and avoids systematic drift in the variation score. State is persisted to `revo_sigma_state.npy` so it survives WESTPA restarts.

`sigma_update_rate` controls how quickly sigma tracks the current ensemble: 0 freezes sigma after the first iteration; 1 is equivalent to `zscore`.

### `zscore`

Sigma is recomputed fresh from the current ensemble at the start of each iteration and held fixed within it. No state is persisted between restarts. Use this if you want sigma to respond immediately to changes in the ensemble rather than smoothly.

### `fixed`

A fixed normalization vector supplied by the user. Sigma never changes. Requires a `sigma_fixed` list in `revo.cfg` with one value per feature:

```yaml
distance_metric: fixed
sigma_fixed:
  - 1.0
  - 0.5
```

### `euclidean`

Raw Euclidean distance with no normalization. Use when features are already on a common scale or normalization is handled upstream. `importance` weights still apply.

### Custom metrics

Subclass `DistanceMetric` from `revo_distance.py` and override `compute()`. Optionally override `update()`, `save()`, and `load()` if your metric maintains state across iterations.

```python
from revo_distance import DistanceMetric

class MyMetric(DistanceMetric):
    def compute(self, features, n_copies=None):
        # return ndarray of shape (n_walkers, n_walkers)
        ...
```

Then instantiate it directly in a subclass of `REVODriver`:

```python
from REVO_driver import REVODriver
from my_metric import MyMetric

class MyDriver(REVODriver):
    def _load_config(self):
        super()._load_config()
        self.distance_metric = MyMetric()
```

## Algorithm

### Variation formula

```
V = Σ_{i<j} (d_ij / char_dist)^α × φ_i × φ_j × n_i × n_j

φ_i = max(0, log(w_i / n_i) − log(p_min / 100))   # novelty
d_ij = sqrt( Σ_k importance_k × ((x_ik − x_jk) / σ_k)² )
```

- `α` = `dist_exponent` (default 4)
- `n_i` = number of copies of walker *i* in the planned ensemble
- `w_i` = weight of walker *i*
- `σ_k` = per-feature normalization (from the distance metric)

### Planning phase

1. Compute the distance matrix once via the configured metric.
2. Find the clone candidate (walker with highest per-walker variation).
3. Find the merge pair (two walkers within `merge_dist` that minimise variation loss, or the lowest-variation walker and its nearest neighbour for `merge_alg: greedy`).
4. Accept the move if it increases variation on the fixed distance matrix.
5. Repeat until no improving move exists.

### Execution phase

Clone/merge decisions from the planning phase are executed using WESTPA's `_merge_walkers` and `_split_walker` API. The pre-selected keeper is guaranteed by setting all `cumul_weight` entries to the total group weight.

## Bin structure

REVO ignores WESTPA bins for resampling. Bins are still needed to trigger recycling (absorbing states). A minimal configuration with a single recycling boundary on the first pcoord dimension looks like:

```yaml
# west.cfg (binning section)
bins:
  type: RectilinearBinMapper
  boundaries:
    - [-inf, 1.0, inf]   # dim 0: recycle below 1.0
    - [-inf, inf]         # all other dims: single bin
```

## Acknowledgements

- REVO algorithm: Donyapour et al., J. Chem. Phys. 150, 244112 (2019)
- Reference implementation: [WEPY](https://github.com/ADicksonLab/wepy) (ADicksonLab)
- Weighted ensemble framework: [WESTPA](https://github.com/westpa/westpa)
