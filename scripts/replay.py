"""
replay.py
=========
Rejoue un run à partir de son SEUL fichier de configuration, et vérifie que les
trajectoires obtenues sont identiques à celles enregistrées.

    python scripts/replay.py out/quickstart/config.json
    python scripts/replay.py out/quickstart/config.json --check

C'est la garantie pratique de reproductibilité : un run archivé se reconstitue
sans autre information que son JSON.
"""

import argparse
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from antsim import load_run, replay, save_run


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="chemin vers config.json")
    ap.add_argument("--check", action="store_true",
                    help="comparer aux trajectoires enregistrées à côté du JSON")
    ap.add_argument("--out", default=None, help="dossier où sauvegarder le run rejoué")
    return ap.parse_args()


def main():
    args = parse_args()
    config = Path(args.config)
    result = replay(config)
    print(f"[replay] {result.n_agents} agents, {result.n_frames} images")

    if args.check:
        reference = load_run(config.parent)
        same = (np.array_equal(reference.x, result.x)
                and np.array_equal(reference.y, result.y)
                and np.array_equal(reference.u, result.u)
                and np.array_equal(reference.theta, result.theta))
        if same:
            print("[check] trajectoires IDENTIQUES au run enregistré.")
        else:
            worst = max(np.abs(reference.x - result.x).max(),
                        np.abs(reference.y - result.y).max())
            raise SystemExit(f"[check] ÉCHEC : écart maximal de position {worst:.3e} mm")

    if args.out:
        print(f"[io] run rejoué -> {save_run(result, args.out)}/")


if __name__ == "__main__":
    main()
