# Fundamental diagram: simulation against the real trajectories

```bash
python scripts/analyze_real_fd.py
```

The diagram is rebuilt **from the raw data**, with the same estimator as for the
simulations — no detour through the article's fitted curves. Each recording is
split into 10 s windows; in each, density and flow are measured.

Two flow estimators are computed and they agree, which validates the
measurement: crossings of the central line, and the integrated flow
`q = ⟨Σ|v_x|⟩/(L·W)` (which is insensitive to track fragmentation).

> **A trap, now fixed**: in the 12800 dataset the identifiers are RECYCLED, so a
> label jumps from one end of the bridge to the other between two frames. These
> teleportations were being counted as crossings and produced absurd flows (`q`
> up to 66). We therefore require a displacement below 4 mm per frame, i.e.
> 100 mm/s — far above the real maximum speed (p99 = 31 mm/s).

## Range covered

| dataset | bridge | measured k |
|---|---|---|
| colony 800 | 10 mm | 0.10 → 1.70 |
| colony 3200 | 20 mm | 0.05 → 2.27 |
| colony 12800 | 20 mm | 0.11 → 4.44 |

**The available data stop at k ≈ 4.4**, well below the transition. They
constrain the RISE of the diagram, not the knee or the plateau; those can only
be tested against the published figure, which aggregates 170 experiments.

## Comparison over the common range

| k | q data (speeds) | q data (line) | q simulation | difference |
|---|---|---|---|---|
| 1 | 1.16 | 1.12 | 1.27 | +10 % |
| 2 | 1.88 | 1.85 | 2.41 | +29 % |
| 3 | 2.98 | 3.03 | 3.42 | +15 % |
| 4 | 3.76 | 3.82 | 4.22 | +12 % |

Fits on the data: Underwood gives `v_f` = 10.9 mm/s, Monod gives `ξ` = 11.0 mm/s
— the simulation has a slope of 14.4.

## The tension, quantified

| what one wants to match | required `ξ` |
|---|---|
| the slope of the diagram measured on the data | ~11 mm/s |
| the free-flow speed measured on the trajectories | 12.5–13 mm/s |
| the article's plateau `q_j` = 10 | ~15 mm/s |

The three cannot be satisfied together as long as `ξ` and `β̄` jointly set both
the slope and the plateau — and **the data do not arbitrate**, since they stop
well before the plateau.

`ξ` = **14** is the compromise adopted: the plateau stays matched (10.20 against
10.0) and the discrepancy on the rise falls from +20/+35 % (at `ξ` = 15) to
+9/+30 %. Both extremes were measured:

| | slope | `q_j` | discrepancy on the rise |
|---|---|---|---|
| `ξ` = 13 | 12.7 | 8.7 | +15 % |
| **`ξ` = 14** | **14.4** | **10.20** | **+9 to +30 %** |
| `ξ` = 15 | 15.3 | 10.9 | +17 to +37 % |

## Caveat — the two data sources disagree

The published speed curve (figure 4A of the article) extrapolates to
**~17.5 mm/s** as `k → 0`, whereas the three recordings available here give a
mean scalar speed of 10–12 mm/s. **The gap is a factor ~1.4 and has not been
resolved.** Before recalibrating `ξ` against either source, its origin must be
understood: trajectory smoothing protocol, inclusion or exclusion of stopped
ants, or simply different sets of experiments.
