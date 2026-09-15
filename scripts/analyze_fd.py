"""
analyze_fd.py
=============
Analyse le CSV produit par `run_fd.py` : ajustement en deux phases du
diagramme fondamental, lois de contact, diagrammes spatio-temporels.

    python scripts/analyze_fd.py
    python scripts/analyze_fd.py --dir resultats/diagramme_fondamental

Ajustement en deux phases (méthode du rapport de passation) :
  phase 1  q = xi.nu_c.k / (nu_c + beta_bar.k)   — fermeture mono-cinétique,
           ajustée sur les densités basses et moyennes ;
  phase 2  q = q_j constant                      — capacité, ajustée sur les
           densités hautes SEULES ;
  la coupure k_split est choisie pour maximiser le R² global, et k_c est
  l'abscisse où les deux branches se rejoignent.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

import antsim.observables as obs
from antsim import load_run, ModelParams as _MP  # noqa: F401
from viz.plots import (plot_contact_number, plot_contact_timing,
                       plot_fundamental, plot_spacetime, plot_speed_density)

from antsim import ModelParams

DEFAULTS = ModelParams()
NU_CRUISE = DEFAULTS.nu_cruise


def aggregate(runs):
    """Moyenne sur les graines, avec DEUX mesures de dispersion.

    `_std` : écart-type d'un run à l'autre (dispersion réelle entre graines) —
             c'est celle qu'on trace, la plus conservatrice ;
    `_sem` : erreur standard de la moyenne, = _std / sqrt(nombre de graines).

    Les barres sont petites parce que chaque point est DÉJÀ une moyenne
    temporelle sur ~24 s de régime stationnaire (environ 480 images) : la
    fluctuation instantanée est moyennée à l'intérieur d'un run, et il ne reste
    entre graines que la variabilité de la condition initiale.
    """
    g = runs.groupby("n_agents")
    agg = g.mean(numeric_only=True)
    std = g.std(numeric_only=True)
    sem = g.sem(numeric_only=True)
    n_seeds = g.size()
    for col in ("q_domain", "q_lines", "q_line", "contacts_per_transit",
                "mean_duration", "delta_T"):
        agg[f"{col}_std"] = std[col]
        agg[f"{col}_sem"] = sem[col]
    agg["u_std"] = std["u_from_flux"]
    agg["u_sem"] = sem["u_from_flux"]
    agg["n_seeds"] = n_seeds
    return agg.reset_index()


def fit_two_phase(agg, nu_cruise=NU_CRUISE):
    """Ajuste les deux phases et renvoie xi_eff, beta_bar, q_j, k_c, R²."""
    from scipy.optimize import curve_fit

    k = agg["k_domain"].to_numpy()
    q = agg["q_domain"].to_numpy()

    def model(kk, xi, beta):
        return obs.monod_flux(kk, xi, beta, nu_cruise)

    best = None
    for split in np.unique(k):
        low, high = k <= split, k > split
        if low.sum() < 4 or high.sum() < 3:
            continue
        try:
            (xi, beta), _ = curve_fit(model, k[low], q[low], p0=(17.0, 40.0),
                                      maxfev=20000)
        except RuntimeError:
            continue
        q_j = float(q[high].mean())
        pred = np.where(low, model(k, xi, beta), q_j)
        ss_res = float(np.sum((q - pred) ** 2))
        ss_tot = float(np.sum((q - q.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        if best is None or r2 > best["r2"]:
            best = dict(xi_eff=float(xi), beta_bar=float(beta), q_j=q_j,
                        r2=r2, k_split=float(split))

    # k_c : abscisse où la branche mono-cinétique atteint le plateau
    xi, beta, q_j = best["xi_eff"], best["beta_bar"], best["q_j"]
    denom = xi * nu_cruise - q_j * beta
    best["k_c"] = float(q_j * nu_cruise / denom) if denom > 0 else np.nan
    best["monod"] = lambda kk: obs.monod_flux(kk, xi, beta, nu_cruise)
    best["q_asympt"] = xi * nu_cruise / beta
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="resultats/diagramme_fondamental")
    args = ap.parse_args()
    d = Path(args.dir)

    runs = pd.read_csv(d / "fd_runs.csv")
    agg = aggregate(runs)
    fit = fit_two_phase(agg)

    print("DIAGRAMME FONDAMENTAL — ajustement en deux phases")
    # k est en cm^-2 et q en cm^-1.s^-1 : la pente xi de la loi de Monod sort
    # donc en cm/s, et beta_bar en cm^2/s. On les affiche dans les unités du
    # modèle (mm/s et mm^2/s).
    print(f"  phase 1 : xi_eff = {10 * fit['xi_eff']:.2f} mm/s   "
          f"(vitesse de croisière du modèle : {DEFAULTS.xi_cruise:g} — "
          f"PRÉDITE, pas ajustée)")
    print(f"            beta_bar = {100 * fit['beta_bar']:.0f} mm²/s   "
          f"(rapport de passation : ~228)")
    print(f"  phase 2 : plateau q_j = {fit['q_j']:.2f} fourmis/cm/s   "
          f"(eLife : ~10)")
    print(f"  transition k_c = {fit['k_c']:.1f} fourmis/cm²          (eLife : ~8)")
    print(f"  asymptote de la loi mono-cinétique = {fit['q_asympt']:.1f} "
          f"(le plateau n'est PAS cette asymptote)")
    print(f"  R² (deux phases) = {fit['r2']:.3f}")

    from viz.plots import K_FIT_MAX
    k = agg["k_domain"].to_numpy()
    c = agg["contacts_per_transit"].to_numpy()
    ok = np.isfinite(c) & (k <= K_FIT_MAX)
    slope, intercept = np.polyfit(k[ok], c[ok], 1)
    dt_mean = float(np.nanmean(agg["delta_T"]))
    print("\nCONTACTS")
    print(f"  C(k) = {slope:.2f} k + {intercept:.2f}  (k <= {K_FIT_MAX:.0f})"
          f"      (eLife : C = 0.61 k)")
    print(f"  durée moyenne d'un contact = {np.nanmean(agg['mean_duration']):.3f} s")
    print(f"  temps perdu par contact dT = {dt_mean:.3f} s            (eLife : 0.24 s)")
    print(f"  T0 moyen = {np.nanmean(agg['T0']):.2f} s (section de 20 mm)")

    print("\nMÉLANGE (indice corrigé du plancher statistique)")
    print(f"  excès de ségrégation : {agg['segregation_excess'].min():+.3f} à "
          f"{agg['segregation_excess'].max():+.3f}")
    print("  (0 = indiscernable d'un mélange parfait ; > 0 = voies réelles)")

    print(f"\nDISPERSION (exemple à k = 10) : écart-type entre graines = "
          f"{float(agg.loc[agg['n_agents'] == 100, 'q_domain_std'].iloc[0]):.3f}, "
          f"erreur standard = "
          f"{float(agg.loc[agg['n_agents'] == 100, 'q_domain_sem'].iloc[0]):.3f} "
          f"sur {int(agg.loc[agg['n_agents'] == 100, 'n_seeds'].iloc[0])} graines")

    agg.to_csv(d / "fd_agrege.csv", index=False)
    bonus = d / "bonus"
    bonus.mkdir(exist_ok=True)

    print(f"\nFONDAMENTAL\n  {plot_fundamental(agg, fit, d / 'diagramme_fondamental.png')}")
    print(f"  {plot_speed_density(agg, fit, d / 'vitesse_densite.png')}")
    print(f"  {plot_contact_number(agg, d / 'contacts.png')}")
    print(f"BONUS\n  {plot_contact_timing(agg, bonus / 'contacts_duree_et_cout.png')}")

    panels = []
    for folder in sorted(d.glob("run_N*")):
        result = load_run(folder)
        dens, x, t = obs.density_spacetime(result, n_bins=50, warmup_frac=0.2)
        panels.append((f"N = {result.n_agents}  "
                       f"(k = {result.n_agents / 10:.0f} cm$^{{-2}}$)", dens, x, t))
    if panels:
        print(f"  {plot_spacetime(panels, bonus / 'spatio_temporel.png')}")


if __name__ == "__main__":
    main()
