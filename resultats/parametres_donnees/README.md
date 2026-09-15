# Parameters extracted directly from the real trajectories

```bash
python scripts/extract_params_from_data.py
```

Datasets: `New_data_12800` (100×20 bridge, k = 2.55) and the two `Old_data`
recordings (10 and 20 mm bridges, colonies 800 and 3200). Positions in mm,
25 frames/s. Each measurement is independent of the others and **does not go
through a fit of the fundamental diagram**.

| Quantity | Measured on the data | Model | Verdict |
|---|---|---|---|
| body size | 3.2 × 1.8 mm → equivalent radius **1.2 mm** | `radius` = 1.0 | ✔ consistent |
| free-flow speed | **12.5–13 mm/s** (plateau of the braking profile) | `xi_cruise` = 14 | ✔ close |
| speed relaxation | τ = 0.227 s | `nu_cruise` = 20 | ✔ see below |
| braking range | **~5 mm** (speed drops below 5 mm) | `l_brake` = 3 | length frozen |
| transverse profile | edge preferred, dip at contact | `standoff` law | ✔ qualitatively |

## Free-flow speed — via the braking profile

The best measurement is not "the speed of isolated ants" (rare at this density,
only 804 measurements, and the criterion also selects stopped ants) but the
**plateau of speed as a function of the distance to the neighbour ahead**:

| distance to frontal neighbour (mm) | 1.25 | 2.25 | 3.25 | 4.25 | 5.25 | 6.25 | > 8 |
|---|---|---|---|---|---|---|---|
| mean speed (mm/s) | 5.35 | 7.83 | 11.30 | 12.85 | 13.20 | 12.71 | ~12.2 |

Two direct readings:

- **the plateau gives `ξ ≈ 12.5–13 mm/s`** — the speed when nothing is in the
  way;
- **the drop-off begins around 5 mm**, not 3: the measured braking range is
  longer than the model's. (Length frozen by decision; recorded for later.)

This profile also gives a direct estimate of the braking strength. At 1.25 mm,
`U/ξ = 0.41`, hence `ν_brake = ν_c(ξ/U − 1) ≈ 29 /s` for ONE frontal neighbour.
The model's kernel would predict `c_brake·(ℓ−ρ)/ρ = 300 × 1.4 = 420 /s`, i.e.
**14 times too strong**. The aggregate braking that yields the right fundamental
diagram is therefore much weaker, at short range, than the kernel says: the same
conclusion as the hand-off report ("the first-principles β̄ over-estimates the
braking"), measured this time on the trajectories.

## Speed relaxation — a trap

The correlation time of speed along the tracks is **0.227 s**, which *seems* to
impose `ν ≈ 4.4`. **This is an over-interpretation**: that time is not
`1/ν_cruise`, it is dominated by braking switching on and off as neighbours
pass. Measured with the same estimator on the simulation:

| | data | `ν_cruise` = 20 | `ν_cruise` = 10 | `ν_cruise` = 4.4 |
|---|---|---|---|---|
| correlation time (s) | 0.227 | **0.268** | 0.324 | 0.422 |

`ν_cruise = 20` is therefore the value **closest to the data**; lowering it moves
away. And on the fundamental diagram, at constant `ν_c/c_brake` ratio (so with
`U(k)` unchanged and only the relaxation time varying):

| | slope | `k_c` | `q_j` |
|---|---|---|---|
| `ν`=20, `c_b`=300 | 15.3 | 17.6 | 10.5 |
| `ν`=10, `c_b`=150 | 15.1 | 18.0 | 10.4 |
| `ν`=4.4, `c_b`=66 | 14.7 | 17.9 | 10.1 |

**No effect**: the diagram only sees the ratio, not the relaxation time. A
useful result in itself — constraining `ν_cruise` will require a dynamical
observable, not the diagram.

Once recomputed with `ν_c = 20`, the measured `β̄` is **196 mm²/s** against 175
in the model: `c_brake = 300` is confirmed.

## Transverse profile

Density relative to a uniform distribution, as a function of distance to the
nearest wall:

| bridge | 0.25 | 1.25 | 2.75 | 3.75 | centre |
|---|---|---|---|---|---|
| 10 mm (colony 800) | 0.49 | 0.99 | **1.32** | 1.26 | 0.79 |
| 20 mm (colony 3200) | **1.36** | 1.35 | 1.27 | 1.13 | 0.70 |
| 20 mm (colony 12800) | 0.00 | 0.17 | 1.14 | 1.83 | 0.73 |

What all three share is clear: **the edge is preferred over the centre**. But
they disagree very close to the wall. On the 10 mm bridge and in the 12800
dataset there is a DIP at contact and a peak a few millimetres out — the
signature of a standoff distance, i.e. of the `standoff` law. In the 3200
dataset the maximum is right against the wall.

Caution: in the 12800 dataset the ordinates never go below 1.78 mm, which
betrays a **detection limit** near the edge (the centre of the bounding box
cannot get closer, and occlusions are frequent there). Its dip is therefore at
least partly an artefact. The 10 mm dataset, however, resolves down to 0.25 mm
and still shows the dip.

Measured conclusion: the edge preference is real and the `standoff` law points
the right way, but the model's target distance (1 mm) is probably too short —
the data suggest 2.5 to 4 mm.
