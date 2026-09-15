"""
quickstart.py
=============
Un run, une image, une vidéo — le point d'entrée pour vérifier en une commande
que tout fonctionne.

    python scripts/quickstart.py
    python scripts/quickstart.py --agents 120 --duration 45 --width 20 --seed 3
    python scripts/quickstart.py --preset minimal --no-video

Sorties dans `--out` (par défaut `out/quickstart/`) :
    config.json        paramètres complets + graine (suffit à rejouer le run)
    trajectories.npz   trajectoires
    snapshot.png       dernière image
    dynamics.mp4       vidéo (ou .gif si ffmpeg est absent)
"""

import argparse
import time

import _bootstrap  # noqa: F401  (ajoute la racine du projet au sys.path)

from antsim import SimParams, make_model, run, save_run
from viz import save_snapshot, save_video, sprites


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="full", choices=("full", "minimal", "report"),
                    help="jeu de termes actifs : full | minimal | report (défaut : full)")
    ap.add_argument("--agents", type=int, default=60, help="nombre d'agents")
    ap.add_argument("--duration", type=float, default=30.0, help="durée simulée (s)")
    ap.add_argument("--width", type=float, default=10.0,
                    help="largeur du pont en mm (expériences : 5, 10, 20)")
    ap.add_argument("--length", type=float, default=100.0, help="longueur du pont (mm)")
    ap.add_argument("--seed", type=int, default=0, help="graine du run")
    ap.add_argument("--out", default="out/quickstart", help="dossier de sortie")
    ap.add_argument("--fps", type=int, default=20, help="images par seconde de la vidéo")
    ap.add_argument("--sprite", default=sprites.DEFAULT, choices=sprites.available(),
                    help="rendu des agents : ellipse (défaut, figures d'analyse), "
                         "ant, rabbit, rally_car, grandpa")
    ap.add_argument("--sprite-scale", type=float, default=1.0,
                    help="multiplicateur de taille des silhouettes (1 = taille "
                         "réelle mesurée, 3.2 mm ; 0.63 = taille du disque "
                         "d'interaction du modèle)")
    ap.add_argument("--natural", action="store_true",
                    help="toutes les fourmis de la même teinte brune, sans le "
                         "code bleu/rouge des groupes (illustration seulement)")
    ap.add_argument("--size-jitter", type=float, default=0.0,
                    help="dispersion relative du gabarit d'un individu à l'autre "
                         "(0.15 = plus ou moins 15 %%)")
    ap.add_argument("--no-video", action="store_true", help="ne pas produire la vidéo")
    return ap.parse_args()


def main():
    args = parse_args()

    model = make_model(args.preset, length=args.length, height=args.width)
    sim = SimParams(num_agents=args.agents, duration=args.duration, seed=args.seed)

    print(f"[run] {args.agents} agents, pont {args.length:.0f} x {args.width:.0f} mm, "
          f"{args.duration:.0f} s simulées, graine {args.seed}, preset {args.preset}")
    t0 = time.perf_counter()
    result = run(model, sim)
    elapsed = time.perf_counter() - t0
    print(f"[run] terminé en {elapsed:.1f} s "
          f"({result.n_frames} images, {sim.substeps_per_frame} sous-pas/image)")

    out = save_run(result, args.out)
    print(f"[io] run sauvegardé -> {out}/")

    look = dict(sprite=args.sprite, sprite_scale=args.sprite_scale,
                color_mode="natural" if args.natural else "group",
                size_jitter=args.size_jitter)

    snap = save_snapshot(result, f"{args.out}/snapshot.png", **look)
    print(f"[viz] image -> {snap}")

    if not args.no_video:
        video = save_video(result, f"{args.out}/dynamics", fps=args.fps, **look)
        print(f"[viz] vidéo -> {video}")

    speed = result.u[result.n_frames // 2:].mean()
    print(f"[diag] vitesse scalaire moyenne (seconde moitié du run) : {speed:.2f} mm/s")


if __name__ == "__main__":
    main()
