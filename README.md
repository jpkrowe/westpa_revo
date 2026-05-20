[WESTPA](https://github.com/westpa/westpa) (Copyright (c) 2013 WESTPA Developers) implementation of the Resampling of Ensembles by Variation Optimisation method (REVO), first described in: Donyapour, Nazanin, et al. The Journal of Chemical Physics, vol. 150, no. 24, June 2019, p. 244112.  [https://doi.org/10.1063/1.5100521](https://doi.org/10.1063/1.5100521).

This implementation is heavily based off the implementation found in [WEPY](https://github.com/ADicksonLab/wepy) (Copyright (c) 2017, 2020 ADicksonLab).

## Configuration

REVO driver parameters are read from a YAML file. By default the driver looks
for `revo.cfg` next to `REVO_driver.py`, or you can point to another
file with the `REVO_CONFIG` environment variable.

Only `feature_names` is required. The rest of the parameters fall back to the
current driver defaults when omitted.

Example:

```yaml
feature_names:
  - pcoord_x
  - pcoord_y

pmin: 1.0e-12
pmax: 0.1
dist_exponent: 4
merge_dist_fraction: 0.5
use_weights: true
merge_alg: pairs
importance: null
pcoord_ranges: null
```
