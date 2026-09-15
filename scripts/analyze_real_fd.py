"""
analyze_real_fd.py
==================
Construit le diagramme fondamental À PARTIR DES TRAJECTOIRES RÉELLES, avec
exactement le même estimateur que pour les simulations, et superpose les deux.

C'est la comparaison directe simulation <-> données, sans passer par les
courbes ajustées de l'article.

Jeux de données (tous à 25 images/s, positions en mm) :
  Old_data/bridge_width=10_colony_size=800     pont 100 x 10
  Old_data/bridge_width=20_colony_size=3200    pont 100 x 20
  New_data_12800/newdata_cleaned_with_or       pont 100 x 20

Chaque enregistrement est découpé en fenêtres de `--window` secondes ; dans
chacune on mesure la densité (fenêtre centrale de 1 cm de long sur toute la
largeur) et le flux (franchissements de la ligne centrale, les deux sens
confondus, par cm de largeur et par seconde). Un enregistrement fournit donc un
NUAGE de points couvrant sa propre gamme de densité.

    python scripts/analyze_real_fd.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
import _data_root

FPS = 25.0

#: (libellé, largeur du pont, chemin RELATIF sous la racine des données, colonnes)
DATASETS = [
    ("colony 800, 10 mm bridge", 10.0,
     Path("Old_data") / "bridge_width=10_colony_size=800_day=11_month=02_nr_of_tracks=4029_run_nr=1.csv",
     dict(step="step", ident="id", x="position_x", y="position_y")),
    ("colony 3200, 20 mm bridge", 20.0,
     Path("Old_data") / "bridge_width=20_colony_size=3200_day=16_month=01_nr_of_tracks=23884_run_nr=1.csv",
     dict(step="step", ident="id", x="position_x", y="position_y")),
    ("colony 12800, 20 mm bridge", 20.0,
     Path("New_data_12800") / "newdata_cleaned_with_or.csv",
     dict(step="frame", ident="id", x="x", y="y")),
]


def real_points(path, cols, width, window_s, length=100.0, window_mm=10.0,
                max_step_mm=4.0):
    """Nuage (k, q) mesuré sur un enregistrement, une fenêtre temporelle par point."""
    d = pd.read_csv(path, usecols=list(cols.values()))
    d = d.rename(columns={v: k for k, v in cols.items()})
    d = d.sort_values(["ident", "step"])

    x_line = 0.5 * length
    half = 0.5 * window_mm
    width_cm = width / 10.0
    area_cm2 = (window_mm / 10.0) * width_cm

    # Franchissements : changement de signe de (x - x_line) entre deux pas
    # consécutifs D'UN MÊME individu.
    # Filtre indispensable : dans le jeu 12800 les identifiants sont RECYCLÉS,
    # si bien qu'une même étiquette saute d'un bout à l'autre du pont entre deux
    # images. Ces téléportations étaient comptées comme des franchissements et
    # produisaient des flux aberrants (q jusqu'à 66). On exige donc un
    # déplacement physiquement possible : moins de `max_step_mm` en une image,
    # soit 100 mm/s à 25 images/s — bien au-delà de la vitesse maximale observée
    # (p99 = 31 mm/s).
    prev_x = d.groupby("ident")["x"].shift()
    prev_step = d.groupby("ident")["step"].shift()
    plausible = (d["x"] - prev_x).abs() <= max_step_mm
    crossed = ((np.sign(d["x"] - x_line) != np.sign(prev_x - x_line))
               & (prev_step == d["step"] - 1) & plausible)
    d = d.assign(crossed=crossed.fillna(False),
                 in_window=(d["x"] >= x_line - half) & (d["x"] <= x_line + half))

    n_frames = int(round(window_s * FPS))
    d["block"] = (d["step"] - d["step"].min()) // n_frames

    # Second estimateur, insensible à la fragmentation des pistes : le flux
    # intégré sur toutes les sections, q = <somme des |v_x|> / (L . W). Une
    # piste interrompue fait perdre un franchissement au premier estimateur,
    # jamais sa contribution en vitesse au second — et le suivi réel PERD des
    # pistes (identifiants recyclés, occlusions), ce qui biaise le comptage de
    # ligne vers le bas.
    vx = (d["x"] - prev_x) * FPS
    d = d.assign(abs_vx=np.where(plausible & (prev_step == d["step"] - 1),
                                 vx.abs(), np.nan))

    rows = []
    for block, sub in d.groupby("block"):
        frames = sub["step"].nunique()
        if frames < 0.8 * n_frames:
            continue
        duration = frames / FPS
        n_mean = len(sub) / frames
        rows.append({
            "k": sub["in_window"].sum() / frames / area_cm2,
            "k_domain": n_mean / (length / 10.0 * width_cm),
            "q_line": sub["crossed"].sum() / (duration * width_cm),
            "q_speed": np.nanmean(sub["abs_vx"]) * n_mean / (length * width_cm),
        })
    return pd.DataFrame(rows)


def underwood(k, v_f, k0):
    """q = k.v_f.exp(-k/k0) — vitesse décroissant exponentiellement avec k."""
    return k * v_f * np.exp(-k / k0)


def monod(k, xi, beta, nu=20.0):
    return xi * nu * k / (nu + beta * k)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=float, default=10.0,
                    help="durée d'une fenêtre de mesure (s)")
    ap.add_argument("--sim", default="resultats/diagramme_fondamental/fd_agrege.csv")
    _data_root.add_argument(ap)
    ap.add_argument("--out", default="resultats/comparaison_donnees")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    root = _data_root.data_root(args.data_root)
    clouds = []
    for label, width, rel, cols in DATASETS:
        path = root / rel
        if not path.exists():
            print(f"  (absent) {path}")
            continue
        pts = real_points(path, cols, width, args.window)
        pts["dataset"] = label
        clouds.append(pts)
        print(f"{label:<28} {len(pts):>4} fenêtres   "
              f"k de {pts['k_domain'].min():.2f} à {pts['k_domain'].max():.2f}   "
              f"q(vitesses) de {pts['q_speed'].min():.2f} à {pts['q_speed'].max():.2f}")
    real = pd.concat(clouds, ignore_index=True)
    real.to_csv(out / "diagramme_reel.csv", index=False)

    sim = pd.read_csv(args.sim)

    # ajustements sur les DONNÉES, dans la gamme couverte
    from scipy.optimize import curve_fit
    real = real.dropna(subset=["k_domain", "q_speed"])
    (v_f, k0), _ = curve_fit(underwood, real["k_domain"], real["q_speed"],
                             p0=(15.0, 12.0), maxfev=20000)
    (xi_r, beta_r), _ = curve_fit(lambda k, xi, b: monod(k, xi, b),
                                  real["k_domain"], real["q_speed"],
                                  p0=(1.5, 2.0), maxfev=20000)
    print(f"\nAJUSTEMENTS SUR LES DONNÉES RÉELLES (k <= {real['k_domain'].max():.1f})")
    print(f"  Underwood : v_f = {10 * v_f:.1f} mm/s, k0 = {k0:.1f} fourmis/cm²")
    print(f"  Monod     : xi  = {10 * xi_r:.1f} mm/s, beta_bar = {100 * beta_r:.0f} mm²/s")

    mask = sim["k_domain"] <= real["k"].max()
    print(f"\nCOMPARAISON DIRECTE dans la gamme commune (k <= {real['k_domain'].max():.1f})")
    print(f"{'k':>6} {'q vitesses':>12} {'q lignes':>10} {'q simu':>9} {'écart':>8}")
    for kk in (1, 2, 3, 4, 5):
        near = real[(real["k_domain"] - kk).abs() < 0.4]
        if len(near) < 5:
            continue
        q_real = near["q_speed"].mean()
        q_sim = np.interp(kk, sim["k_domain"], sim["q_domain"])
        print(f"{kk:>6} {q_real:>12.2f} {near['q_line'].mean():>10.2f} "
              f"{q_sim:>9.2f} {100 * (q_sim - q_real) / q_real:>7.1f}%")

    # ------------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    colors = {"colony 800, 10 mm bridge": "#7ea8c4",
              "colony 3200, 20 mm bridge": "#4b8fb8",
              "colony 12800, 20 mm bridge": "#1f5f85"}
    for label, sub in real.groupby("dataset"):
        ax.scatter(sub["k_domain"], sub["q_speed"], s=9, alpha=0.35, lw=0,
                   color=colors.get(label, "gray"), label=f"data — {label}")

    k_max_real = real["k_domain"].max()
    kk = np.linspace(0, max(k_max_real, sim["k_domain"].max()) * 1.05, 400)
    ax.plot(kk, underwood(kk, v_f, k0), "-", color="purple", lw=1.8,
            label=f"Underwood fitted on the data ($v_f$={10*v_f:.1f} mm/s)")
    ax.errorbar(sim["k_domain"], sim["q_domain"], yerr=sim["q_domain_std"],
                fmt="o", ms=5, lw=0, elinewidth=1, capsize=2, color="#c0392b",
                label="simulation", zorder=4)
    ax.axvspan(0, k_max_real, color="gray", alpha=0.07)
    ax.text(0.5 * k_max_real, 0.5, "density range covered by the data",
            ha="center", fontsize=8, color="gray")

    ax.set_xlabel(r"density $k$ (ants$\cdot$cm$^{-2}$)")
    ax.set_ylabel(r"flow $q$ (ants$\cdot$cm$^{-1}\cdot$s$^{-1}$)")
    ax.set_title("Simulation against the real trajectories")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "diagramme_simu_vs_donnees.png", dpi=150)
    print(f"\n{out / 'diagramme_simu_vs_donnees.png'}")


if __name__ == "__main__":
    main()
