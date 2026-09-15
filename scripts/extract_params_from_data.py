"""
extract_params_from_data.py
===========================
Extrait directement des trajectoires réelles les paramètres du modèle qui ont
une observable évidente, sans passer par un ajustement du diagramme
fondamental. Chaque mesure est indépendante des autres.

Ce qu'on peut lire sans effort :

  radius            taille corporelle (colonnes `width` / `height` du suivi)
  xi_cruise         vitesse des fourmis ISOLÉES (aucun voisin à portée)
  nu_cruise         taux de relaxation de la vitesse, = 1 / temps de corrélation
                    de la vitesse d'une fourmi isolée
  lambda_theta      taux de virage, = 1 / temps de corrélation du cap
  l_brake           distance au voisin frontal à partir de laquelle la vitesse
                    chute
  thigmotactisme    profil transverse : les fourmis longent-elles les bords ?

    python scripts/extract_params_from_data.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import _bootstrap  # noqa: F401
import _data_root

FPS = 25.0
DT = 1.0 / FPS
NEW_REL = Path("New_data_12800") / "newdata_cleaned_with_or.csv"

ISOLATION_MM = 10.0        # « isolée » = aucun voisin à moins de 10 mm


def load_new(path, max_rows=None):
    d = pd.read_csv(path, usecols=["frame", "id", "x", "y", "vmag", "orientation",
                                  "width", "height"], nrows=max_rows)
    d["speed"] = d["vmag"] * FPS
    return d


def body_size(d):
    """Taille corporelle : la boîte englobante du suivi."""
    return d["width"].median(), d["height"].median()


def isolated_speed(d, stride=10):
    """Vitesse des fourmis sans voisin à moins de ISOLATION_MM."""
    speeds = []
    frames = np.sort(d["frame"].unique())[::stride]
    sub = d[d["frame"].isin(frames)]
    for _, g in sub.groupby("frame"):
        if len(g) < 2:
            speeds.append(g["speed"].to_numpy())
            continue
        pts = g[["x", "y"]].to_numpy()
        tree = cKDTree(pts)
        nn = tree.query(pts, k=2)[0][:, 1]
        speeds.append(g["speed"].to_numpy()[nn > ISOLATION_MM])
    return np.concatenate(speeds)


def correlation_time(d, column, min_len=40, n_tracks=1500, max_lag=25):
    """Temps de corrélation d'une grandeur le long des pistes longues.

    On ajuste une exponentielle sur l'autocorrélation moyenne des écarts à la
    moyenne de la piste ; l'inverse est le taux de relaxation.
    """
    counts = d.groupby("id").size()
    ids = counts[counts >= min_len].index[:n_tracks]
    acc = np.zeros(max_lag)
    n = 0
    for i in ids:
        s = d.loc[d["id"] == i, column].to_numpy()
        s = s - s.mean()
        var = np.dot(s, s) / len(s)
        if var <= 0:
            continue
        for lag in range(max_lag):
            acc[lag] += np.dot(s[:len(s) - lag], s[lag:]) / (len(s) - lag) / var
        n += 1
    if n == 0:
        return np.nan, None
    acc /= n
    lags = np.arange(max_lag) * DT
    positive = acc > 0.05
    if positive.sum() < 3:
        return np.nan, acc
    slope = np.polyfit(lags[positive], np.log(acc[positive]), 1)[0]
    return -1.0 / slope, acc


def braking_profile(d, stride=20, bins=np.arange(1.0, 12.0, 0.5)):
    """Vitesse moyenne en fonction de la distance au voisin situé DEVANT."""
    dists, speeds = [], []
    frames = np.sort(d["frame"].unique())[::stride]
    sub = d[d["frame"].isin(frames)]
    for _, g in sub.groupby("frame"):
        if len(g) < 2:
            continue
        pts = g[["x", "y"]].to_numpy()
        th = g["orientation"].to_numpy()
        v = g["speed"].to_numpy()
        tree = cKDTree(pts)
        for a, (nbrs) in enumerate(tree.query_ball_point(pts, 12.0)):
            best = np.inf
            for b in nbrs:
                if b == a:
                    continue
                dx, dy = pts[b] - pts[a]
                rho = np.hypot(dx, dy)
                if rho < 1e-9:
                    continue
                if (np.cos(th[a]) * dx + np.sin(th[a]) * dy) / rho > 0.7:  # cône avant
                    best = min(best, rho)
            if np.isfinite(best):
                dists.append(best)
                speeds.append(v[a])
    dists = np.array(dists)
    speeds = np.array(speeds)
    idx = np.digitize(dists, bins) - 1
    out = []
    for b in range(len(bins) - 1):
        m = idx == b
        if m.sum() > 50:
            out.append((0.5 * (bins[b] + bins[b + 1]), speeds[m].mean(), m.sum()))
    return out


def wall_profile(d, width=20.0, n_bins=20):
    """Distribution de la distance à la paroi la plus proche."""
    dist = np.minimum(d["y"], width - d["y"])
    h, edges = np.histogram(dist, bins=np.linspace(0, width / 2, n_bins + 1),
                            density=True)
    return h, 0.5 * (edges[:-1] + edges[1:])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-rows", type=int, default=1_500_000)
    ap.add_argument("--out", default="resultats/parametres_donnees")
    _data_root.add_argument(ap)
    args = ap.parse_args()

    d = load_new(_data_root.data_root(args.data_root) / NEW_REL, args.max_rows)
    print(f"{len(d):,} lignes, {d['id'].nunique():,} pistes, "
          f"{d['frame'].nunique():,} images\n")

    w, h = body_size(d)
    print("TAILLE CORPORELLE")
    print(f"  longueur {w:.2f} mm, largeur {h:.2f} mm  ->  rayon équivalent "
          f"{0.5 * np.sqrt(w * h):.2f} mm    (modèle : radius = 1.0)")

    v_iso = isolated_speed(d)
    print(f"\nVITESSE LIBRE (fourmis sans voisin à moins de {ISOLATION_MM:.0f} mm, "
          f"{len(v_iso):,} mesures)")
    print(f"  moyenne {v_iso.mean():.2f}   médiane {np.median(v_iso):.2f}   "
          f"p90 {np.percentile(v_iso, 90):.2f} mm/s    (modèle : xi_cruise = 15)")
    print(f"  écart-type / moyenne = {v_iso.std() / v_iso.mean():.2f}"
          f"                        (modèle : xi_spread = 0.2)")

    tau_v, _ = correlation_time(d, "speed")
    print(f"\nRELAXATION DE LA VITESSE")
    print(f"  temps de corrélation {tau_v:.3f} s  ->  nu ≈ {1 / tau_v:.1f} /s"
          f"          (modèle : nu_cruise = 20)")

    d["unwrapped"] = d["orientation"]
    tau_t, _ = correlation_time(d, "unwrapped")
    print(f"\nRELAXATION DU CAP")
    print(f"  temps de corrélation {tau_t:.3f} s  ->  lambda ≈ {1 / tau_t:.1f} /s"
          f"    (modèle : lambda_theta = 10, mode field_norm par défaut)")

    print(f"\nFREINAGE — vitesse selon la distance au voisin de devant")
    print(f"  {'distance (mm)':>14} {'vitesse (mm/s)':>16} {'effectif':>10}")
    for rho, v, n in braking_profile(d):
        print(f"  {rho:>14.2f} {v:>16.2f} {n:>10}")

    print(f"\nPROFIL TRANSVERSE (distance à la paroi, pont 20 mm)")
    h_prof, centers = wall_profile(d)
    uniform = 1.0 / 10.0
    for c, v in zip(centers, h_prof):
        bar = "#" * int(60 * v / max(h_prof))
        print(f"  {c:>5.2f} mm | {v / uniform:>5.2f}x uniforme | {bar}")

    Path(args.out).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
