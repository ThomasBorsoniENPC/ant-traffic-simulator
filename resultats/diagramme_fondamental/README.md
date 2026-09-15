# Quantitative analysis — fundamental diagram and contacts

Protocol and reference values from Poissonnier, Motsch, Gautrais, Buhl &
Dussutour (2019, *eLife*). Default model configuration: `standoff` wall law
(`c_wall = 2`), torus with transverse remixing, **without** positional
non-penetration or wall-escape.

`xi_cruise = 14` mm/s — an acknowledged compromise between the real
trajectories, which give 12.5–13 but only cover `k ≤ 4.4`, and the published
plateau of the fundamental diagram, which would call for 15 (see
`resultats/parametres_donnees/` and `resultats/comparaison_donnees/`).
`nu_cruise = 20`, `c_brake = 300` and all LENGTHS are unchanged — the data
confirm them (measured β̄ = 196 mm²/s against 175 in the model).

Sweep: **29 densities (N = 10 to 280), 6 seeds**, 40 s simulated, 40 % warm-up
— 174 runs. Densities are refined between `k` = 11 and 19 (where the transition
happens) and above all between `k` = 20 and 28, to populate the plateau.

```bash
python scripts/run_fd.py --seeds 6 --duration 40
python scripts/analyze_fd.py
```

## Fundamental diagram — `diagramme_fondamental.png`

Three flow estimators measured on the same runs (central line alone = eLife
protocol, average over 7 cross-sections, whole domain); they agree, and
averaging over cross-sections cuts the uncertainty without introducing bias.

| Quantity | Simulation | eLife 2019 |
|---|---|---|
| Shape | linear then **strict plateau, no decline** | two-phase (idem) |
| Phase-1 slope | `ξ_eff` = **14.4 mm/s** | `v_f` ≈ 12–21 |
| Effective braking | `β̄` = 174 mm²/s | — |
| Plateau | `q_j` = **10.21** ants·cm⁻¹·s⁻¹ | `q_j` ≈ **10** |
| Transition | `k_c` = **18.6** ants·cm⁻² | `k_j` ≈ 8 |
| Two-phase fit | R² = 1.000 | — |

Across the 11 densities between `k` = 19 and 28 the flow stays between **10.11
and 10.29**: the plateau is **strictly flat, with no decline whatsoever**, over
a range extending to 1.5× the maximum experimental density.

The phase-1 slope equals the model's own cruising speed: it is **predicted, not
fitted**. The plateau is not the asymptote of the mono-kinetic law (16.4): it is
a **capacity**, reached before it.

### Why the knee does not move

The diagnosis lies in a single quantity: **the speed at the transition**,
`U(k_c) = 10 q_j / k_c`. The eLife description requires `10 × 10 / 8 =
12.5 mm/s`, i.e. **the free-flow speed**: in their two-phase fit, ants do not
slow down at all before the knee. Our model gives 4 to 6 mm/s. (The article's
raw point cloud, however, starts bending at `k ≈ 3` and saturates towards
`k ≈ 10–12`: the "`k_j` = 8" is a parameter of the two-phase fit, not what the
data show. Underwood describes the cloud just as well, and it is that shape the
model approaches.)

Magnitude sweep (lengths frozen, 10 densities, 3 seeds):

| Configuration | slope | `k_c` | `q_j` | `U(k_c)` |
|---|---|---|---|---|
| reference (ξ=13) | 12.7 | 16.1 | 8.7 | 5.4 |
| `k_obj` 5 / 20 | 13.0 / 13.1 | 15.9 / 17.4 | 8.8 / 8.4 | 5.5 / 4.8 |
| `c_avoid_steer` 15 / 150 | 12.0 / 13.1 | 18.2 / 13.4 | 7.7 / 8.6 | 4.3 / 6.4 |
| `c_repulse` 10 | 12.8 | 16.7 | 8.8 | 5.3 |
| `nu_cruise` 8, `c_brake` 120 | 12.6 | 16.5 | 8.6 | 5.2 |
| `c_brake` 450 / 700 | 12.7 / 12.6 | 14.4 / 10.8 | 6.8 / 4.9 | 4.7 / 4.5 |
| `nu_cruise` 10 / 4.4 (constant `ν/c_b` ratio) | 15.1 / 14.7 | 18.0 / 17.9 | 10.4 / 10.1 | 5.8 / 5.6 |
| **eLife target** | **12.5** | **8.0** | **10.0** | **12.5** |

No magnitude brings `U(k_c)` anywhere near 12.5. And this is structural: in the
Monod law, `q(8) = 10` with `ξ = 12.5` requires `β̄ → 0`, i.e. no braking at all
below `k = 8`. Raising `c_brake` does move `k_c` towards 8, but lowers the
plateau in the same proportion (`q_j` = 4.9 at `c_brake` = 700): `k_c` and `q_j`
are tied by the same constant `β̄` and cannot be set independently.

### But the experiment may contradict its own description

The real trajectories at `k = 2.55` give a mean speed of **10.4 mm/s** for a
free-flow speed of 18–23 (percentiles 90–99): at that density, far below
`k_j = 8`, ants are **already at ~55 % of their free-flow speed**. This
contradicts the "no slowdown before the knee" reading imposed by the two-phase
fit, and instead supports a gradual saturation — the model's. Before trying to
reproduce a sharp knee, one should check on the article's raw point cloud
whether a smooth saturation is genuinely excluded.

## Contacts — `contacts.png`

**The counting method is described in detail in `CONTACT_METHOD.md`**: it is
the most consequential convention in the whole analysis.

| Law | Simulation | eLife 2019 |
|---|---|---|
| Contacts per transit (`k` ≤ 20) | **C = 0.67 k + 0.54** | **C = 0.61 k** |

`C` grows **linearly** with density, and with the default settings (antennal
range 2.5 mm, mutual ±60° cone) the slope is 0.65, against 0.61 in the article.
Two caveats, detailed in `CONTACT_METHOD.md`:

- the **linearity** is robust up to `k` ≈ 20: it holds for every threshold
  convention tested and for every velocity parameter set (R² ≥ 0.98). That is
  what is a property of the model. Beyond `k` = 20 it degrades (R² = 0.93) and
  the fit stops there: once the flow has split into clusters, complete transits
  of the section become long and rare, and the per-transit count becomes noisy.
  This is also the bound of the experimental range (`k` ≤ 18).
- the **value** of the slope varies from 0.58 to 0.95 depending on the antennal
  range adopted (2.2 to 3.5 mm). Agreement at 0.61 therefore constrains the
  convention as much as it validates the model. A proper calibration would
  measure `d_contact` on the real trajectories before comparing slopes.

Contact duration and time cost `ΔT`: see `bonus/`.

## Space–time structure — `bonus/spatio_temporel.png`

Local density along the bridge over time.

- **k = 6 (phase 1)**: diagonal streaks in both directions, no persistent
  structure — free flow.
- **k = 14 (near the transition)**: first concentrations.
- **k = 25 (phase 2)**: **bright, persistent vertical bands** — near-stationary
  clusters lasting tens of seconds, coexisting with free-flowing regions. This
  is the phase separation invoked to explain the plateau: beyond `k_c` the flow
  splits, throughput is set by flux continuity between regions, and extra
  density only grows the clusters.

## Mixing

The **excess** segregation (raw index minus the statistical baseline obtained by
reshuffling group labels at fixed positions) stays between −0.03 and +0.09 over
the whole range: **the flow is statistically indistinguishable from perfectly
mixed**. Beware the RAW index, which falls from 0.61 to 0.11: that decrease is
almost entirely the statistical baseline shrinking (`√(2/πn)` per strip), not
mixing improving. See `resultats/melange/`.

The stopped fraction stays low across the range: 0.2 % at `k` = 6, 3 % at
`k` = 20, **12 % at `k` = 28** — the flow slows a great deal but does not clog,
even at 1.5× the maximum experimental density.

## Error bars

The bars plotted are the **standard deviation across the 6 seeds**, not the
standard error (`fd_agrege.csv` contains both, columns `_std` and `_sem`). They
are small — at k = 10, standard deviation 0.10 and standard error 0.04 on a flow
of 9.1 — and legitimately so: **each point is already a time average over ~24 s
of stationary regime**, i.e. about 480 frames. The instantaneous fluctuation is
averaged out within a run; all that remains between seeds is the variability of
the initial condition, which is quickly forgotten on a torus with transverse
remixing. The high-density points do show visibly larger scatter — that is the
phase-2 instability showing up point by point.

## Files

| | |
|---|---|
| `diagramme_fondamental.png` | q(k), the three estimators, the fit, published eLife points |
| `vitesse_densite.png` | U(k), each branch over its own domain of validity |
| `contacts.png` | C(k) alone — the fundamental contact observable |
| `CONTACT_METHOD.md` | how a contact is defined and counted |
| `bonus/` | contact duration, time cost, space–time diagram |
| `fd_runs.csv`, `fd_agrege.csv` | one run per line / per-density averages |
| `run_N*/` | three archived runs, replayable |
