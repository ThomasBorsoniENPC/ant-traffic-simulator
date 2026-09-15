# Calibration against the real speed distribution

Data: `Data/New_data_12800/newdata_cleaned_with_or.csv` — **100 × 20 mm**
bridge, 104,138 frames, 50.9 agents per frame, i.e.
**k = 2.55 ants·cm⁻²**.

## Units

Positions are in **mm** (`x ∈ [0, 100]`, `y ∈ [0, 20]`) and `vmag` is in **mm
per frame**. The frame rate is **25 frames/s** (`fps = 25` in
`Programmes/Data Management/…`), so `vmag × 25` gives mm/s.

Independent check: in `velocity_ID_4792` the ant crosses the full 100 mm of the
bridge in **200 frames**, i.e. 8 s, i.e. **12.1 mm/s** — consistent with the
mode of the histogram (0.47 mm/frame × 25 = 11.8 mm/s).

## Real distribution (mm/s)

| percentile | 5 | 10 | 25 | 50 | 75 | 90 | 95 | 99 |
|---|---|---|---|---|---|---|---|---|
| speed | 1.07 | 2.35 | 5.83 | **10.47** | 14.80 | 18.17 | 19.97 | 23.02 |

Mean 10.44, standard deviation 5.81, mode 11.7. Fractions below 1 / 2 / 3 mm/s:
**4.7 %** / 8.6 % / 12.6 %.

## What the calibration changed

Grid over `xi_cruise`, `xi_spread` and `c_brake` (lengths unchanged), objective
= Wasserstein distance to the real distribution:

| | median | mean | p90 | < 1 mm/s | Wasserstein |
|---|---|---|---|---|---|
| **real data** | 10.5 | 10.4 | 18.2 | 0.047 | — |
| old defaults (ξ=17, σ=0.2) | 15.5 | 14.2 | 19.1 | 0.000 | 3.96 |
| ξ=15, σ=0.5 | 12.5 | 12.4 | 19.8 | 0.001 | 2.02 |

Two conclusions:

1. **ξ = 17 was too fast.** Lowering it improves the match markedly. A larger
   individual spread (σ = 0.5) improves it further still, but `xi_spread` was
   **capped at 0.2 by decision**: beyond that the individual heterogeneity
   becomes implausible, even though it would bring the distribution closer. The
   adopted value is `ξ = 14`, `σ = 0.2` (see
   `resultats/comparaison_donnees/` for the arbitration).
2. **`c_brake` is not constrained by these data**: at `k = 2.55` encounters are
   too rare for braking to leave a signature. This dataset constrains the
   free-flow speed and its heterogeneity, not the braking.

## What still does not match

The data contain **4.7 % of speeds below 1 mm/s** — the spike at zero, clearly
visible in the histogram — against **0.1 %** in the model. The model has no
genuinely **stopped** ants at low density. This is precisely the mechanism on
which Dobramysl et al. (2024) base their explanation of the plateau. It is the
sharpest remaining gap between the model and the trajectories, and the first
candidate for an extension.

## Reproduction

```bash
python scripts/fit_speed_distribution.py          # full grid
python scripts/fit_speed_distribution.py --quick  # reduced grid
```
