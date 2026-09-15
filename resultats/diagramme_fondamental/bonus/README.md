# Secondary material

Figures that help understand the mechanism, but which do not serve as
validation in the same way as `q(k)` and `C(k)`.

## `contacts_duree_et_cout.png`

Mean contact duration, and the slope `ΔT` of `T = T₀ + C·ΔT`.

| | Simulation | eLife 2019 |
|---|---|---|
| Mean contact duration | 0.19 s | — |
| Time lost per contact `ΔT` | **0.15 s** | ≈ **0.24 s** |
| `T₀` (20 mm section) | 1.57 s | ≈ 0.95 s |

The per-contact time cost is **emergent and non-zero**: the slowdown produced by
braking and avoidance during an encounter costs about 0.15 s, against 0.24 s
measured — the right order of magnitude. An explicit antennation timer (item
B.1 of the project backlog) would close the remaining gap.

Both quantities depend on conventions (distance threshold, definition of the end
of a contact, length of the measurement section — and `T₀` depends directly on
that length): read them as orders of magnitude.

## `spatio_temporel.png`

Local density along the bridge over time, one panel per density. Hard to read as
it stands, but the content is clear:

- **k = 6**: diagonal streaks in both directions, no persistent structure —
  free flow;
- **k = 14**: first concentrations;
- **k = 25**: bright, persistent vertical bands — near-stationary clusters
  lasting tens of seconds alongside free-flowing regions.

This is the phase separation invoked to explain the plateau: beyond `k_c` the
flow splits, throughput is set by flux continuity between regions, and extra
density only grows the clusters.
