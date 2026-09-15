"""
run_fd.py
=========
Balayage de densité : produit le CSV brut dont sortent le diagramme
fondamental, l'analyse des contacts et les diagrammes spatio-temporels.

    python scripts/run_fd.py
    python scripts/run_fd.py --seeds 10 --duration 60 --out resultats/fd

REPRENABLE : le CSV est écrit ligne par ligne et les couples (N, graine) déjà
faits sont sautés au redémarrage. On peut donc interrompre (Ctrl-C) et relancer.
"""

import argparse
import csv
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import antsim.observables as obs
from antsim import ModelParams, SimParams, run, save_run

DENSITIES = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
             120, 140, 160, 180, 200, 225, 250)

FIELDS = ("n_agents", "seed", "height", "k_line", "q_line", "k_lines", "q_lines",
          "k_domain", "q_domain", "u_mean", "u_from_flux", "stopped_frac",
          "segregation_raw", "segregation_null", "segregation_excess",
          "contacts_per_ant_per_s", "contacts_per_transit",
          "n_transits", "mean_duration", "median_duration", "T0", "delta_T",
          "d_contact")


def existing(path):
    if not path.exists():
        return set()
    with path.open() as f:
        return {(int(r["n_agents"]), int(r["seed"])) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--densities", type=int, nargs="+", default=list(DENSITIES))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--warmup", type=float, default=0.4)
    ap.add_argument("--d-contact", type=float, default=2.5,
                    help="portée antennaire pour la détection des contacts (mm)")
    ap.add_argument("--height", type=float, default=10.0)
    ap.add_argument("--xi", type=float, default=None, help="xi_cruise")
    ap.add_argument("--xi-spread", type=float, default=None)
    ap.add_argument("--c-brake", type=float, default=None)
    ap.add_argument("--l-brake", type=float, default=None,
                    help="portée du freinage frontal (mm)")
    ap.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR",
                    help="surcharge d'un paramètre de ModelParams, répétable")
    ap.add_argument("--no-contacts", action="store_true",
                    help="saute l'analyse des contacts (balayage de repérage)")
    ap.add_argument("--out", default="resultats/diagramme_fondamental")
    ap.add_argument("--keep-runs", type=int, nargs="+", default=[60, 150, 250],
                    help="effectifs dont la graine 0 est archivée (spatio-temporel)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "fd_runs.csv"
    done = existing(csv_path)
    if not csv_path.exists():
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    overrides = {k: v for k, v in
                 (("xi_cruise", args.xi), ("xi_spread", args.xi_spread),
                  ("c_brake", args.c_brake), ("l_brake", args.l_brake))
                 if v is not None}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key] = float(value)
    if overrides:
        print(f"surcharges du modèle : {overrides}")

    todo = [(n, s) for n in args.densities for s in range(args.seeds)
            if (n, s) not in done]
    print(f"{len(todo)} runs à faire ({len(done)} déjà présents)")

    t_start = time.perf_counter()
    for i, (n, seed) in enumerate(todo, 1):
        result = run(ModelParams(height=args.height, **overrides),
                     SimParams(num_agents=n, duration=args.duration, seed=seed),
                     progress=False)
        row = obs.fundamental_point(result, warmup_frac=args.warmup)
        if not args.no_contacts:
            row.update(obs.contact_stats(result, d_contact=args.d_contact,
                                         warmup_frac=args.warmup))
            seg = obs.segregation_index(result, warmup_frac=args.warmup)
            row.update({f"segregation_{k}": v for k, v in seg.items()})
        with csv_path.open("a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow({k: row.get(k) for k in FIELDS})
        if seed == 0 and n in args.keep_runs:
            save_run(result, out / f"run_N{n:04d}")
        elapsed = time.perf_counter() - t_start
        print(f"  [{i}/{len(todo)}] N={n:>4} graine {seed}  "
              f"k={row['k_domain']:.1f}  q={row['q_domain']:.2f}  "
              f"({elapsed / i:.1f} s/run)")

    print(f"\nCSV -> {csv_path}")


if __name__ == "__main__":
    main()
