"""
fit_speed_distribution.py
=========================
Cale les paramètres de vitesse sur la DISTRIBUTION DES VITESSES mesurée sur
les trajectoires réelles, plutôt que sur le diagramme fondamental.

Données : `Data/New_data_12800/newdata_cleaned_with_or.csv`, pont 100 x 20 mm,
25 images/s. Les positions y sont en mm et `vmag` en mm/image ; on multiplie
donc par 25 pour obtenir des mm/s. (Vérification indépendante : sur
`velocity_ID_4792`, la fourmi traverse les 100 mm en 200 images, soit 8 s,
soit 12.1 mm/s — cohérent avec le mode de l'histogramme à 11.8 mm/s.)

La vitesse simulée est mesurée EXACTEMENT comme dans les données : différence
finie des positions enregistrées, divisée par le pas d'enregistrement — pas la
variable `u` du modèle, qui ignore le fait que l'agent tourne pendant le pas.

    python scripts/fit_speed_distribution.py
    python scripts/fit_speed_distribution.py --quick
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import _bootstrap  # noqa: F401
import _data_root

from antsim import ModelParams, SimParams, run

DATA_REL = Path("New_data_12800") / "newdata_cleaned_with_or.csv"
FPS = 25.0
BRIDGE = (100.0, 20.0)
V_CUT = 1.1 * FPS          # même coupure que la figure d'origine (27.5 mm/s)


def real_speeds(path):
    """Vitesses réelles en mm/s, et densité moyenne de l'enregistrement."""
    d = pd.read_csv(path, usecols=["frame", "vmag"])
    v = (d["vmag"].dropna() * FPS).to_numpy()
    n_per_frame = d.groupby("frame").size().mean()
    area_cm2 = BRIDGE[0] / 10.0 * BRIDGE[1] / 10.0
    return v[v < V_CUT], n_per_frame, n_per_frame / area_cm2


def simulated_speeds(model, n_agents, duration, seed):
    """Vitesses simulées, mesurées par différence finie comme dans les données."""
    result = run(model, SimParams(num_agents=n_agents, duration=duration,
                                  seed=seed), progress=False)
    start = result.n_frames // 3
    dt = float(result.times[1] - result.times[0])
    dx = np.diff(result.x[start:], axis=0)
    dy = np.diff(result.y[start:], axis=0)
    # exclure les pas repliés par la périodicité (x) et le remixage (y)
    ok = (np.abs(dx) < 0.5 * model.length) & (np.abs(dy) < 0.5 * model.height)
    v = np.hypot(dx, dy)[ok] / dt
    return v[v < V_CUT]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="grille réduite")
    ap.add_argument("--out", default="resultats/calibration_vitesses")
    _data_root.add_argument(ap)
    args = ap.parse_args()

    v_real, n_frame, k_real = real_speeds(
        _data_root.data_root(args.data_root) / DATA_REL)
    print(f"Données : {len(v_real):,} vitesses, {n_frame:.1f} agents/image, "
          f"k = {k_real:.2f} fourmis/cm²")
    print(f"  médiane {np.median(v_real):.2f}   moyenne {v_real.mean():.2f}   "
          f"p90 {np.percentile(v_real, 90):.2f}   p99 {np.percentile(v_real, 99):.2f}")
    print(f"  fraction sous 1 mm/s : {np.mean(v_real < 1):.3f}")

    n_agents = int(round(n_frame))
    if args.quick:
        grid_xi, grid_spread, grid_brake = (15.0, 19.0), (0.2, 0.5), (300.0, 900.0)
    else:
        grid_xi = (13.0, 15.0, 17.0, 19.0, 21.0)
        grid_spread = (0.2, 0.35, 0.5, 0.65)
        grid_brake = (150.0, 300.0, 600.0, 1200.0)

    rows = []
    for xi in grid_xi:
        for spread in grid_spread:
            for brake in grid_brake:
                model = ModelParams(length=BRIDGE[0], height=BRIDGE[1],
                                    xi_cruise=xi, xi_spread=spread, c_brake=brake)
                v = np.concatenate([simulated_speeds(model, n_agents,
                                                     args.duration, s)
                                    for s in range(args.seeds)])
                rows.append(dict(xi_cruise=xi, xi_spread=spread, c_brake=brake,
                                 wasserstein=wasserstein_distance(v_real, v),
                                 median=np.median(v), mean=v.mean(),
                                 p90=np.percentile(v, 90),
                                 frac_slow=float(np.mean(v < 1))))

    df = pd.DataFrame(rows).sort_values("wasserstein").reset_index(drop=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "grille.csv", index=False)

    print(f"\nMeilleurs jeux (distance de Wasserstein aux données, en mm/s) :")
    print(df.head(8).round(3).to_string(index=False))
    best = df.iloc[0]

    # --- figure de comparaison ------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_best = ModelParams(length=BRIDGE[0], height=BRIDGE[1],
                             xi_cruise=best.xi_cruise, xi_spread=best.xi_spread,
                             c_brake=best.c_brake)
    v_best = np.concatenate([simulated_speeds(model_best, n_agents,
                                              args.duration, s)
                             for s in range(args.seeds)])
    model_def = ModelParams(length=BRIDGE[0], height=BRIDGE[1])
    v_def = np.concatenate([simulated_speeds(model_def, n_agents, args.duration, s)
                            for s in range(args.seeds)])

    bins = np.linspace(0, V_CUT, 100)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.hist(v_real, bins=bins, density=True, color="#8fb8de",
            label=f"données réelles (k = {k_real:.1f})")
    ax.hist(v_def, bins=bins, density=True, histtype="step", lw=1.8,
            color="#c0392b", label="modèle, défauts (ξ=17, σ=0.2, c_b=300)")
    ax.hist(v_best, bins=bins, density=True, histtype="step", lw=2.0,
            color="#1b3b6f",
            label=f"modèle ajusté (ξ={best.xi_cruise:g}, σ={best.xi_spread:g}, "
                  f"c_b={best.c_brake:g})")
    ax.set_xlabel("vitesse (mm/s)")
    ax.set_ylabel("densité de probabilité")
    ax.set_title(f"Distribution des vitesses — pont {BRIDGE[0]:.0f}×{BRIDGE[1]:.0f} mm, "
                 f"{n_agents} agents")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "distribution_vitesses.png", dpi=150)
    print(f"\n{out / 'distribution_vitesses.png'}")


if __name__ == "__main__":
    main()
