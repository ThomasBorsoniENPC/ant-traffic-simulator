"""
analyze_mixing.py
=================
Analyse l'ablation continue de l'antennation produite par `run_mixing.py`.

    python scripts/analyze_mixing.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from viz.plots import plot_mixing


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="resultats/melange")
    args = ap.parse_args()
    d = Path(args.dir)

    runs = pd.read_csv(d / "melange_runs.csv")
    g = runs.groupby(["n_agents", "c_comm"])
    agg = g.mean(numeric_only=True)
    err = g.sem(numeric_only=True)
    for col in ("segregation_excess", "segregation_raw", "q_domain",
                "contacts_per_ant_per_s", "u_mean"):
        agg[f"{col}_err"] = err[col]
    agg = agg.reset_index()

    print("ABLATION CONTINUE DE L'ANTENNATION")
    for n, sub in agg.groupby("n_agents"):
        sub = sub.sort_values("c_comm")
        seg0 = float(sub.iloc[0]["segregation_excess"])
        seg_nom = float(sub[sub["c_comm"] == 12.0]["segregation_excess"].iloc[0])
        # plus petite intensité ramenant la ségrégation à mi-chemin
        target = 0.5 * (seg0 + seg_nom)
        below = sub[sub["segregation_excess"] <= target]
        c_half = float(below.iloc[0]["c_comm"]) if len(below) else np.nan
        print(f"\n  N = {n} (k = {n / 10:.0f} cm^-2)")
        print(f"    sans antennation   : excès {seg0:+.3f}")
        print(f"    valeur nominale 12 : excès {seg_nom:+.3f}")
        print(f"    mi-chemin atteint dès c_comm = {c_half:g}")
        print(f"    flux : {float(sub.iloc[0]['q_domain']):.2f} (c=0) -> "
              f"{float(sub[sub['c_comm'] == 12.0]['q_domain'].iloc[0]):.2f} (c=12)")

    agg.to_csv(d / "melange_agrege.csv", index=False)
    print(f"\n{plot_mixing(agg, d / 'melange.png')}")


if __name__ == "__main__":
    main()
