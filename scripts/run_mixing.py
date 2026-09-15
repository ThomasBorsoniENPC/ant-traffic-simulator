"""
run_mixing.py
=============
Ablation CONTINUE de l'antennation : quelle intensité de communication faut-il
pour tenir le mélange, et le résultat est-il robuste ou tient-il à un réglage
fin ?

Le rapport de passation ne montre que deux points (`c_comm = 0` -> voies,
`c_comm = 12` -> mélangé). On balaie ici l'intervalle, à plusieurs densités,
en mesurant simultanément la ségrégation transverse, le flux et le taux de
contacts — ces trois grandeurs devant bouger ensemble si l'antennation est bien
le mécanisme du mélange.

    python scripts/run_mixing.py
    python scripts/run_mixing.py --agents 60 150 --seeds 8
"""

import argparse
import csv
from pathlib import Path

import _bootstrap  # noqa: F401

import antsim.observables as obs
from antsim import ModelParams, SimParams, run, save_run

C_COMM_VALUES = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0)
FIELDS = ("n_agents", "c_comm", "seed", "segregation_raw", "segregation_null",
          "segregation_excess", "k_domain", "q_domain", "u_mean",
          "contacts_per_ant_per_s", "stopped_frac")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, nargs="+", default=[60, 150])
    ap.add_argument("--c-comm", type=float, nargs="+", default=list(C_COMM_VALUES))
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--warmup", type=float, default=0.4)
    ap.add_argument("--out", default="resultats/melange")
    ap.add_argument("--keep", type=float, nargs="+", default=[0.0, 4.0, 12.0],
                    help="valeurs de c_comm dont la graine 0 est archivée (vidéos)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "melange_runs.csv"
    done = set()
    if csv_path.exists():
        with csv_path.open() as f:
            done = {(int(r["n_agents"]), float(r["c_comm"]), int(r["seed"]))
                    for r in csv.DictReader(f)}
    else:
        with csv_path.open("w", newline="") as f:
            csv.DictWriter(f, FIELDS).writeheader()

    todo = [(n, c, s) for n in args.agents for c in args.c_comm
            for s in range(args.seeds) if (n, c, s) not in done]
    print(f"{len(todo)} runs à faire")

    for i, (n, c_comm, seed) in enumerate(todo, 1):
        result = run(ModelParams(c_comm=c_comm),
                     SimParams(num_agents=n, duration=args.duration, seed=seed),
                     progress=False)
        point = obs.fundamental_point(result, warmup_frac=args.warmup)
        seg = obs.segregation_index(result, warmup_frac=args.warmup)
        row = {
            "n_agents": n, "c_comm": c_comm, "seed": seed,
            "segregation_raw": seg["raw"], "segregation_null": seg["null"],
            "segregation_excess": seg["excess"],
            "k_domain": point["k_domain"], "q_domain": point["q_domain"],
            "u_mean": point["u_mean"], "stopped_frac": point["stopped_frac"],
            "contacts_per_ant_per_s": obs.contact_stats(
                result, warmup_frac=args.warmup)["contacts_per_ant_per_s"],
        }
        with csv_path.open("a", newline="") as f:
            csv.DictWriter(f, FIELDS).writerow(row)
        if seed == 0 and c_comm in args.keep and n == args.agents[-1]:
            save_run(result, out / f"run_N{n}_ccomm{c_comm:g}")
        print(f"  [{i}/{len(todo)}] N={n:>4} c_comm={c_comm:>5.1f} graine {seed}  "
              f"ségrég. {row['segregation_excess']:+.2f}  q {row['q_domain']:.2f}")

    print(f"\nCSV -> {csv_path}")


if __name__ == "__main__":
    main()
