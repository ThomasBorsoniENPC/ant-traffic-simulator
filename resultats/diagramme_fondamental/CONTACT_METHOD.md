# How contacts are counted

Code: `antsim/observables.py`, functions `_contact_pairs`, `contact_stats`,
`_section_transits`.

## 1. What counts as a contact

At each recorded frame, a pair (i, j) is **in contact** if all three of the
following hold simultaneously:

1. **opposite directions**: `g_i · g_j < 0`;
2. **antennal proximity**: centre-to-centre distance `ρ_ij ≤ d_contact`, with
   `d_contact = 2.5 mm` by default. The body is `2R = 2 mm`, so the threshold
   places contact a quarter of a diameter beyond bodily contact, which
   represents antennae touching;
3. **facing each other**: each sees the other within its frontal cone,
   `cos(φ_ij − θ_i) ≥ cos_min` **and** `cos(φ_ji − θ_j) ≥ cos_min`, with
   `cos_min = 0.5`, i.e. a 60° half-cone. The condition is **mutual**: an ant
   being overtaken from behind is not an antennal contact.

## 2. Episodes

An **episode** starts at the first frame where the condition becomes true for
the pair, and ends at the first frame where it becomes false again. From this
we derive:

- the **number** of contacts (one per episode, counted for both agents);
- the **duration** of each episode.

Time resolution: frames are recorded at 20 Hz, so **0.05 s**. A contact shorter
than that is missed, and durations are quantised in steps of 0.05 s. There is
**no hysteresis**: a pair oscillating around the threshold is counted several
times. This is the main limitation of the method.

## 3. `C`: contacts per transit

`C` is defined as in the article: the number of contacts an ant experiences
**during a transit of a central section**, here **20 mm** long (parameter
`section_mm`).

A **transit** is a continuous stay of the agent within the central band:

- both the **entry** and the **exit** must be observed (transits truncated by
  the start or end of the measurement window are not counted);
- no periodic wrap in the middle (an agent that leaves by one end of the bridge
  and comes back by the other has not "transited");
- the contacts counted are those that **begin** during the stay.

`T` is the duration of that stay, and `T = T₀ + C·ΔT` is fitted by least squares
on the individual transits of a run.

## 4. What depends on conventions, and what does not

Sensitivity of the slope of `C(k)` to the two thresholds (3 densities, seed 0):

| `d_contact` (mm) | half-cone | C(k=6) | C(k=14) | C(k=25) | **slope** |
|---|---|---|---|---|---|
| 2.2 | 60° | 3.58 | 8.86 | 14.67 | 0.58 |
| **2.5** | **60°** | **4.09** | **9.93** | **16.34** | **0.64** |
| 3.0 | 60° | 4.73 | 11.44 | 19.60 | 0.78 |
| 3.5 | 60° | 5.46 | 13.33 | 23.60 | 0.95 |
| 2.5 | 90° | 5.08 | 12.19 | 20.90 | 0.83 |
| 2.5 | 45° | 3.17 | 8.24 | 14.16 | 0.58 |

**What to take away, and not to oversell**: the **linearity** of `C` in `k` is
robust — it holds for every convention tested, and it is that which is a
property of the model. The **value** of the slope, however, depends on the
threshold: from 0.58 to 0.95 over the reasonable range. The fact that the
default setting (2.5 mm, ±60°) lands on the experimental slope of 0.61
**constrains the convention as much as it validates the model**. Put differently:
the model predicts that the number of contacts grows linearly with density; it
does not on its own predict the coefficient 0.61 without fixing the antennal
range.

A proper calibration would measure `d_contact` on the real trajectories (the
typical distance at which two ants of opposite direction slow down), then check
that the slope follows — rather than choosing `d_contact` so that the slope
comes out right.
