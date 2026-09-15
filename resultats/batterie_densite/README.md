# Visual control battery across densities

30 s videos, 100 × 10 mm bridge, seed 0, default configuration (`standoff` wall
law with `c_wall = 2`, torus with transverse remixing, **without** positional
non-penetration or wall-escape).

Reproduction: `python scripts/run_density_battery.py` from `simulateur/`.
Each `.config.json` replays its run identically
(`python scripts/replay.py <file> --check`).

| N | k (cm⁻²) | u | \|vx\| | %stopped | %crushed | segreg. | mean nearest-neighbour distance |
|---|---|---|---|---|---|---|---|
| 1 | 0.1 | 17.93 | 17.89 | 0.00 | 0.00 | — | — |
| 2 | 0.2 | 16.29 | 16.22 | 0.00 | 0.03 | — | 24.28 |
| 60 | 6.0 | 11.51 | 11.24 | 0.00 | 0.03 | 0.31 | 2.09 |
| 100 | 10.0 | 9.58 | 9.29 | 0.00 | 0.04 | 0.25 | 1.79 |
| 150 | 15.0 | 7.71 | 7.42 | 0.01 | 0.06 | 0.19 | 1.55 |
| 200 | 20.0 | 6.33 | 6.06 | 0.03 | 0.10 | 0.13 | 1.43 |
| 400 | 40.0 | 3.27 | 3.09 | 0.14 | 0.18 | 0.07 | 1.03 |

Measurements over the second half of each run. Contact at 2R = 2 mm;
experimental range of Poissonnier et al.: k ≤ 18; segregation 0 = mixed,
1 = clear lanes (meaningless at N = 1 and 2, too few agents).

*These figures were produced with the earlier parameter set (ξ = 15); they are
kept as a visual reference. The quantitative results in
`resultats/diagramme_fondamental/` use the current defaults (ξ = 14).*

> **Note.** Ces runs datent d'avant le retrait du terme d'amortissement de
> paroi (`c_wall_damping`). Leurs `config.json` ne se rechargent donc plus
> (`load_config` refuse les champs inconnus, par construction). Les vidéos
> restent valables comme illustration ; pour des runs rejouables, relancer
> le script de la section « Reproduction ».
