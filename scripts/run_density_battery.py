"""
run_density_battery.py
======================
Batterie de contrôle visuel en densité : une vidéo de 30 s par effectif, plus
un tableau de diagnostics. Sert à voir CE QUI CASSE quand on pousse la densité
bien au-delà de la gamme expérimentale.

    python scripts/run_density_battery.py
    python scripts/run_density_battery.py --agents 1 2 60 --duration 10

Repères de densité (pont 100 x 10 mm, soit 10 cm²) : N agents -> k = N/10
fourmis/cm². La gamme expérimentale de Poissonnier et al. va jusqu'à
k ~ 18 (N ~ 180) ; au-delà on est en régime extrême.
"""

import argparse
import time

import numpy as np

import _bootstrap  # noqa: F401

from antsim import ModelParams, SimParams, run, save_run
from viz import save_snapshot, save_video

DEFAULT_AGENTS = (1, 2, 60, 100, 150, 200, 400)


def diagnostics(result):
    """Mesures agrégées sur la seconde moitié du run."""
    half = result.n_frames // 2
    h = result.model.height
    y = result.y[half:]
    d_wall = np.minimum(y, h - y)

    # ségrégation transverse : 0 = mélangé, 1 = voies nettes
    seg = []
    for f in range(0, y.shape[0], 10):
        idx = np.clip((y[f] / h * 8).astype(int), 0, 7)
        num = den = 0.0
        for s in range(8):
            m = idx == s
            npos = np.sum(result.group[m] > 0)
            nneg = np.sum(result.group[m] < 0)
            if npos + nneg:
                num += abs(npos - nneg)
                den += npos + nneg
        if den:
            seg.append(num / den)

    # recouvrement : distance moyenne au plus proche voisin (contact à 2R)
    nn = []
    for f in range(0, y.shape[0], 40):
        pts = np.stack([result.x[half + f], y[f]], 1)
        if len(pts) < 2:
            continue
        m = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
        np.fill_diagonal(m, 1e9)
        nn.append(m.min(1).mean())

    return {
        "k": result.n_agents / (result.model.length * result.model.height / 100.0),
        "u": float(result.u[half:].mean()),
        "vx": float(np.abs(result.u[half:] * np.cos(result.theta[half:])).mean()),
        "arret": float(np.mean(result.u[half:] < 1.0)),
        "ecrase": float(np.mean(d_wall < 0.5)),
        "seg": float(np.mean(seg)) if seg else np.nan,
        "nn": float(np.mean(nn)) if nn else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, nargs="+", default=list(DEFAULT_AGENTS))
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/batterie")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    rows = []
    for n in args.agents:
        t0 = time.perf_counter()
        result = run(ModelParams(),
                     SimParams(num_agents=n, duration=args.duration, seed=args.seed),
                     progress=False)
        t_sim = time.perf_counter() - t0
        folder = f"{args.out}/N{n:04d}"
        save_run(result, folder)
        save_snapshot(result, f"{folder}/snapshot.png")
        t_vid = 0.0
        if not args.no_video:
            t0 = time.perf_counter()
            save_video(result, f"{folder}/dynamics", progress=False)
            t_vid = time.perf_counter() - t0
        d = diagnostics(result)
        rows.append((n, d))
        print(f"N={n:>4}  k={d['k']:>5.1f}/cm²  simu {t_sim:>5.1f}s  "
              f"vidéo {t_vid:>5.1f}s  -> {folder}/")

    print(f"\n{'N':>5} {'k/cm²':>7} {'u':>7} {'|vx|':>7} {'%arrêt':>8} "
          f"{'%écrasé':>9} {'ségrég.':>8} {'d plus proche':>14}")
    print("-" * 70)
    for n, d in rows:
        print(f"{n:>5} {d['k']:>7.1f} {d['u']:>7.2f} {d['vx']:>7.2f} "
              f"{d['arret']:>8.2f} {d['ecrase']:>9.2f} {d['seg']:>8.2f} {d['nn']:>14.2f}")
    print("\nRepères : contact à 2R = 2 mm ; gamme expérimentale k <= 18 ; "
          "ségrégation 0 = mélangé, 1 = voies.")


if __name__ == "__main__":
    main()
