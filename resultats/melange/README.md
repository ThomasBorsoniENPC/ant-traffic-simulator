# Continuous ablation of antennation

The hand-off report shows only two points (`c_comm = 0` → lanes, `c_comm = 12`
→ mixed). Here the whole interval is swept, at two densities, 6 seeds, 40 s.

```bash
python scripts/run_mixing.py --seeds 6 --duration 40
python scripts/analyze_mixing.py
```

## Result — `melange.png`

The three panels move together, which is the expected signature if antennation
really is the mixing mechanism: as `c_comm` rises, the contact rate rises and
segregation falls.

The index plotted is the **segregation excess**, corrected for the statistical
baseline (see below): 0 = indistinguishable from perfect mixing, > 0 = real
lanes.

| | N = 60 (k = 6) | N = 150 (k = 15) |
|---|---|---|
| excess without antennation | **+0.227** | **+0.213** |
| excess at `c_comm = 12` | +0.042 | **+0.005** |
| halfway reached by | `c_comm` = 2 | `c_comm` = 3 |
| flow, `c=0` → `c=12` | 6.39 → 5.60 | 9.97 → 9.38 |

At the working density (k = 15) antennation brings the excess to **+0.005**:
the flow becomes statistically **indistinguishable from perfectly mixed**.

**This is not a fine tuning**: segregation has finished falling by `c_comm ≈ 8`
and does not move up to 24. The nominal value 12 sits on the flat part of the
curve, not on its slope. Mixing costs a little throughput (−11 % at k = 6, −6 %
at k = 15): antennation slows the flow but does not jam it.

## The statistical baseline — why the raw index misleads

The raw index is the occupancy-weighted mean of `|n₊ − n₋| / (n₊ + n₋)` over 8
horizontal strips. It is **not zero for perfect mixing**: with `n` agents per
strip and randomly assigned groups it averages `√(2/πn)` — 0.29 for 7 agents per
strip, 0.14 for 30. It therefore decreases mechanically as density rises,
without any additional mixing taking place.

`segregation_index` therefore **measures** the baseline instead of assuming it:
at each frame the group labels are reshuffled at random among the agents present
(**positions are unchanged**) and the index is recomputed. This is the exact null
"observed positions, groups independent of position". The excess is the
difference. The columns `segregation_raw`, `segregation_null` and
`segregation_excess` are all three in the CSV.

## Caution — transverse remixing damps this measurement

The periodic boundary condition redraws the ordinate at random on re-entry
(`x_mode = "torus"`). That is desirable for the fundamental diagram — on a torus
of finite length a transverse structure would freeze in — but **it also destroys
lanes by construction**, and therefore underestimates the very effect we are
trying to measure here. Control at N = 150, 4 seeds, segregation excess:

| `x_mode` | `c_comm = 0` | `c_comm = 12` |
|---|---|---|
| `torus` (remixing) | +0.25 | +0.01 |
| `torus_keep_y` (ordinate preserved) | **+0.63** | +0.03 |

Without remixing, switching antennation off does produce clear lanes (+0.63) and
the effect is three times stronger. **For this observable specifically,
measurements should be made with `torus_keep_y`**; remixing remains the right
choice for the fundamental diagram. The numbers in the table above are therefore
a LOWER bound on the effect of antennation.
