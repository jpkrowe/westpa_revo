"""
REVO (Resampling of Ensembles by Variation Optimization) driver for WESTPA.

The REVO optimization runs as a planning phase on a fixed distance matrix
(Donyapour et al., J. Chem. Phys. 2019), then clone/merge decisions are
executed via WESTPA's _merge_walkers / _split_walker API.

Register in west.cfg:
    west:
      drivers:
        module_path: $WEST_SIM_ROOT
        we_driver: REVO_driver.REVODriver

Driver parameters are read from a YAML file. The default location is
revo.cfg next to this module, or set REVO_CONFIG to point to a
different file.

Requires revo_resampler.py in the same directory.
"""

import logging
import operator
import os
from pathlib import Path

import numpy as np

import westpa
from westpa.core.we_driver import WEDriver

from revo_resampler import compute_distance_matrix, calc_variation

log = logging.getLogger(__name__)

DEFAULT_REVO_CONFIG = {
    "pmin": 1e-12,
    "pmax": 0.1,
    "dist_exponent": 4,
    "merge_dist_fraction": 0.5,
    "use_weights": True,
    "merge_alg": "pairs",
    "importance": None,
    "pcoord_ranges": None,
}


def _load_revo_config(config_path=None):
    """Load REVO configuration from YAML and apply defaults."""
    if config_path is None:
        config_path = os.environ.get(
            "REVO_CONFIG", Path(__file__).with_name("revo.cfg")
        )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"REVO configuration file not found: {config_path}. "
            "Set REVO_CONFIG or create revo.cfg next to REVO_driver.py."
        )

    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to load REVO configuration files. "
            "Install the 'pyyaml' package to use revo.cfg."
        ) from exc

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"REVO configuration in {config_path} must be a YAML mapping.")

    config = DEFAULT_REVO_CONFIG.copy()
    config.update({key: value for key, value in loaded.items() if key in config})

    feature_names = loaded.get("feature_names")
    if not feature_names:
        raise ValueError(
            f"REVO configuration in {config_path} must define a non-empty 'feature_names' list."
        )
    if not isinstance(feature_names, (list, tuple)):
        raise ValueError("'feature_names' must be a YAML list.")

    config["feature_names"] = [str(name) for name in feature_names]

    if config["importance"] is not None:
        config["importance"] = np.asarray(config["importance"], dtype=float)
    if config["pcoord_ranges"] is not None:
        config["pcoord_ranges"] = np.asarray(config["pcoord_ranges"], dtype=float)

    return config


class REVODriver(WEDriver):
    """WESTPA WE driver implementing REVO diversity-optimizing resampling.

    Planning phase: greedy REVO on a fixed distance matrix. Modifies only
    weights, n_copies, and merge_groups (bookkeeping). Variation is
    recomputed each step with updated bookkeeping.

    Execution phase: one _merge_walkers call per keeper group with
    forced cumul_weight to guarantee the planned keeper is selected.
    Splits are executed after all merges.

    WESTPA API (confirmed on WESTPA 2022.15):
        _merge_walkers(segments, cumul_weight, bin)
            -> (glom, gparent_seg). Does NOT modify bin.
        _split_walker(segment, m, bin)
            -> list of new segments. Removes segment from bin, adds new ones.
    """

    PMIN = 1e-12
    PMAX = 0.1
    DIST_EXPONENT = 4
    MERGE_DIST_FRACTION = 0.5
    USE_WEIGHTS = True
    MERGE_ALG = "pairs"  # 'pairs' (wepy default): find pair minimizing variation loss
    # 'greedy' (paper): lowest Vi first, then nearest neighbor
    IMPORTANCE = (
        None  # Per-feature importance weights for the distance sum. None = equal.
    )
    PCOORD_RANGES = (
        None  # Per-feature expected ranges for normalization, shape (n_features,).
    )
    # If set, sigmas = PCOORD_RANGES instead of computing from ensemble std.
    # This prevents sigma oscillation between iterations.
    # Example: np.array([0.1, 2.0, 3.0, ...]) for 19 features.

    def _load_config(self):
        if hasattr(self, "_revo_config"):
            return self._revo_config

        self._revo_config = _load_revo_config()
        self.FEATURE_NAMES = self._revo_config["feature_names"]
        self.PMIN = self._revo_config["pmin"]
        self.PMAX = self._revo_config["pmax"]
        self.DIST_EXPONENT = self._revo_config["dist_exponent"]
        self.MERGE_DIST_FRACTION = self._revo_config["merge_dist_fraction"]
        self.USE_WEIGHTS = self._revo_config["use_weights"]
        self.MERGE_ALG = self._revo_config["merge_alg"]
        self.IMPORTANCE = self._revo_config["importance"]
        self.PCOORD_RANGES = self._revo_config["pcoord_ranges"]
        return self._revo_config

    def _run_we(self):
        self._load_config()
        self._recycle_walkers()
        self._check_pre()

        for bin in self.next_iter_binning:
            if len(bin) == 0:
                continue

            segments = np.array(
                sorted(bin, key=operator.attrgetter("weight")), dtype=np.object_
            )
            n_walkers = len(segments)
            if n_walkers < 3:
                continue

            weights = np.array([s.weight for s in segments])
            features = np.array([s.pcoord[-1, :] for s in segments])

            if np.allclose(features, features[0]):
                westpa.rc.pstatus("REVO: All walkers identical, skipping")
                continue

            # Distance matrix: computed once, fixed for the entire optimization.
            # If PCOORD_RANGES is set, use it as fixed sigmas to prevent oscillation.
            dist_matrix, sigmas = compute_distance_matrix(
                features, self.IMPORTANCE, sigmas=self.PCOORD_RANGES
            )
            char_dist = dist_matrix[np.triu_indices(n_walkers, k=1)].mean()
            merge_dist = self.MERGE_DIST_FRACTION * char_dist

            n_copies = np.ones(n_walkers, dtype=int)
            w = weights.copy()
            variation, walker_vars = calc_variation(
                w,
                n_copies,
                dist_matrix,
                char_dist,
                self.DIST_EXPONENT,
                self.PMIN,
                self.USE_WEIGHTS,
            )

            # Log iteration stats
            westpa.rc.pstatus("\n========== REVO ITERATION STATS ==========")
            westpa.rc.pstatus(f"Walkers: {n_walkers}")
            westpa.rc.pstatus(f"Char distance: {char_dist:.4f}")
            westpa.rc.pstatus(f"Merge distance: {merge_dist:.4f}")
            westpa.rc.pstatus(f"Initial variation: {variation:.4e}")
            westpa.rc.pstatus("--- Feature ranges (min / max) ---")
            for dim in range(features.shape[1]):
                vals = features[:, dim]
                name = (
                    self.FEATURE_NAMES[dim]
                    if dim < len(self.FEATURE_NAMES)
                    else f"dim{dim}"
                )
                westpa.rc.pstatus(f"  {name}: {vals.min():.4f} / {vals.max():.4f}")

            # === PLANNING PHASE ===
            merge_groups = [[] for _ in range(n_walkers)]
            n_ops = 0
            productive = True

            while productive:
                productive = False

                # Clone candidate: highest variation, alive, splittable
                clone_idx = None
                for idx in np.argsort(-walker_vars):
                    if n_copies[idx] < 1:
                        continue
                    if w[idx] / (n_copies[idx] + 1) <= self.PMIN:
                        continue
                    clone_idx = idx
                    break

                if clone_idx is None:
                    break

                # Find merge pair
                m1_idx = None
                m2_idx = None

                if self.MERGE_ALG == "pairs":
                    best_loss = np.inf
                    for i in range(n_walkers):
                        if i == clone_idx or n_copies[i] != 1:
                            continue
                        for j in range(i + 1, n_walkers):
                            if j == clone_idx or n_copies[j] != 1:
                                continue
                            if w[i] + w[j] >= self.PMAX:
                                continue
                            if dist_matrix[i, j] >= merge_dist:
                                continue
                            v_loss = (w[j] * walker_vars[i] + w[i] * walker_vars[j]) / (
                                w[i] + w[j]
                            )
                            if v_loss < best_loss:
                                best_loss = v_loss
                                m1_idx, m2_idx = i, j

                else:  # 'greedy'
                    for idx in np.argsort(walker_vars):
                        if idx == clone_idx or n_copies[idx] != 1:
                            continue
                        if w[idx] >= self.PMAX:
                            continue
                        m1_idx = idx
                        break

                    if m1_idx is not None:
                        best_dist = np.inf
                        for j in range(n_walkers):
                            if j == clone_idx or j == m1_idx or n_copies[j] != 1:
                                continue
                            if w[m1_idx] + w[j] >= self.PMAX:
                                continue
                            if dist_matrix[m1_idx, j] >= merge_dist:
                                continue
                            if dist_matrix[m1_idx, j] < best_dist:
                                best_dist = dist_matrix[m1_idx, j]
                                m2_idx = j

                if m1_idx is None or m2_idx is None:
                    break

                # Test with probability-weighted n_copies (expected variation)
                old_copies = n_copies.copy()
                old_w = w.copy()

                n_copies = n_copies.astype(float)
                n_copies[clone_idx] += 1
                tempsum = w[m1_idx] + w[m2_idx]
                n_copies[m1_idx] = w[m1_idx] / tempsum
                n_copies[m2_idx] = w[m2_idx] / tempsum

                test_var, _ = calc_variation(
                    w,
                    n_copies,
                    dist_matrix,
                    char_dist,
                    self.DIST_EXPONENT,
                    self.PMIN,
                    self.USE_WEIGHTS,
                )

                if test_var <= variation:
                    n_copies = old_copies
                    w = old_w
                    break

                # Accepted: random keeper selection
                n_copies = old_copies.copy()
                n_copies[clone_idx] += 1

                if np.random.random() < w[m1_idx] / tempsum:
                    keep_idx, squash_idx = m1_idx, m2_idx
                else:
                    keep_idx, squash_idx = m2_idx, m1_idx

                w[keep_idx] += w[squash_idx]
                w[squash_idx] = 0.0
                n_copies[squash_idx] = 0

                merge_groups[keep_idx].append(squash_idx)
                merge_groups[keep_idx].extend(merge_groups[squash_idx])
                merge_groups[squash_idx] = []

                variation, walker_vars = calc_variation(
                    w,
                    n_copies,
                    dist_matrix,
                    char_dist,
                    self.DIST_EXPONENT,
                    self.PMIN,
                    self.USE_WEIGHTS,
                )

                n_ops += 1
                productive = True

            # Log results
            westpa.rc.pstatus("\n--- REVO optimization result ---")
            westpa.rc.pstatus(f"  Clone/merge ops: {n_ops}")
            westpa.rc.pstatus(f"  Final variation: {variation:.4e}")
            westpa.rc.pstatus(f"  Cloned: {int((n_copies > 1).sum())}")
            westpa.rc.pstatus(f"  Removed: {int((n_copies == 0).sum())}")
            westpa.rc.pstatus(f"  Unchanged: {int((n_copies == 1).sum())}")
            westpa.rc.pstatus(f"  Weight conservation: {w.sum():.6f}")
            westpa.rc.pflush()
            westpa.rc.pstatus("\n--- Post Optimisation Weight stats ---")
            westpa.rc.pstatus(f"  min: {w.min():.4e}")
            westpa.rc.pstatus(f"  max: {w.max():.4e}")
            westpa.rc.pstatus(f"  sum: {w.sum():.6f}")
            westpa.rc.pflush()

            # === EXECUTION PHASE ===
            # One _merge_walkers call per keeper group. forced_cumul ensures
            # the planned keeper (index 0 in the segment list) is selected.
            glom_refs = {}
            for keep_i in range(n_walkers):
                squash_list = merge_groups[keep_i]
                if len(squash_list) == 0:
                    continue
                # to_merge is a list per segment that describes the segments that will be merged into that segment
                to_merge = [segments[keep_i]] + [segments[si] for si in squash_list]
                total_w = sum(s.weight for s in to_merge)
                forced_cumul = np.full(len(to_merge), total_w)

                bin.difference_update(to_merge)
                # Since we have already selected the 'keeper' walker according to the weights of the two walkers
                # we want the merge to always select the keeper walker as the parent segment.
                # To do this, we set the cumul_weight for all segments in the merge group to the total weight of the group,
                # which ensures that the first segment (the keeper) is always selected as the parent.
                glom, gparent = self._merge_walkers(to_merge, forced_cumul, bin)
                bin.add(glom)
                glom_refs[keep_i] = glom

            for i in range(n_walkers):
                if n_copies[i] > 1:
                    seg = glom_refs.get(i, segments[i])
                    if seg in bin:
                        bin.remove(seg)
                        new_segs = self._split_walker(seg, n_copies[i], bin)
                        bin.update(new_segs)

            westpa.rc.pstatus(f"  Final walker count: {len(bin)}")
            westpa.rc.pstatus("==========================================")
            westpa.rc.pflush()

        self._check_post()
        self.new_weights = self.new_weights or []

        log.debug("used initial states: {!r}".format(self.used_initial_states))
        log.debug("available initial states: {!r}".format(self.avail_initial_states))
