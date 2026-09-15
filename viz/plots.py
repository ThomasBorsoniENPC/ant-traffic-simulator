"""
plots.py
========
Figures d'analyse : diagramme fondamental, vitesse-densité, contacts,
diagramme spatio-temporel. Toutes prennent les données déjà agrégées et se
contentent de tracer — l'ajustement est dans `scripts/analyze_fd.py`.

Repères expérimentaux tracés en pointillés, d'après Poissonnier et al. 2019 :
densité de transition k_j ~ 8 fourmis/cm², plateau q_j ~ 10 fourmis/cm/s,
contacts C = 0.61 k, temps perdu par contact dT ~ 0.24 s.
"""

from pathlib import Path

import numpy as np

#: Points relevés à la main sur la figure 4 de l'article (panneaux A et B).
#: Lecture graphique : à ne pas utiliser comme donnée de précision, mais c'est
#: la seule façon de comparer aux MESURES publiées plutôt qu'à leur ajustement.
ELIFE_POINTS = Path(__file__).resolve().parents[1] / "donnees_publiees" / "elife2019_figure4.csv"


def _elife_points():
    """(k, v en mm/s, q) relevés sur la figure 4. None si le fichier manque."""
    if not ELIFE_POINTS.exists():
        return None
    a = np.loadtxt(ELIFE_POINTS, delimiter=",", comments="#")
    return {"k": a[:, 0], "v_mm_s": a[:, 1], "q": a[:, 2]}


ELIFE_K_J = 8.0
ELIFE_Q_J = 10.0
ELIFE_CONTACT_SLOPE = 0.61
ELIFE_DELTA_T = 0.24


def _plt(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return plt


def _errbar(ax, k, q, qerr, label, color, marker="o"):
    ax.errorbar(k, q, yerr=qerr, fmt=marker, ms=4, lw=0, elinewidth=1,
                capsize=2, label=label, color=color)


def elife_two_phase(k, k_j=ELIFE_K_J, q_j=ELIFE_Q_J):
    """Courbe two-phase de l'article : linéaire jusqu'à k_j, constante ensuite."""
    return np.minimum(q_j * np.asarray(k) / k_j, q_j)


def plot_fundamental(agg, fit, path, estimators=("domain", "lines", "line"),
                     errorbar="std"):
    """q(k) : les trois estimateurs, l'ajustement en deux phases, et l'eLife."""
    plt = _plt(path)
    fig, ax = plt.subplots(figsize=(7.4, 5.2))

    colors = {"domain": "#1b3b6f", "lines": "#e07b39", "line": "#8fb8de"}
    names = {"domain": "whole domain", "lines": "average over 7 cross-sections",
             "line": "central line (eLife protocol)"}
    for est in estimators:
        _errbar(ax, agg[f"k_{est}"], agg[f"q_{est}"],
                agg[f"q_{est}_{errorbar}"], names[est], colors[est])

    k = np.linspace(0, agg["k_domain"].max() * 1.05, 400)
    ax.plot(k, fit["monod"](k), "-", color="#2e7d32", lw=1.8,
            label=r"phase 1: $q=\xi\nu_c k/(\nu_c+\bar\beta k)$")
    ax.axhline(fit["q_j"], color="#c0392b", lw=1.8,
               label=f"phase 2: plateau $q_j$ = {fit['q_j']:.1f}")
    ax.axvline(fit["k_c"], color="k", ls="--", lw=1,
               label=f"transition $k_c$ = {fit['k_c']:.1f}")

    pts = _elife_points()
    if pts is not None:
        ax.plot(pts["k"], pts["q"], "s", ms=4.5, mfc="none", mec="purple",
                mew=1.2, label="eLife 2019 — published points (fig. 4B)")
    ax.plot(k, elife_two_phase(k), "--", color="purple", lw=1.4, alpha=0.6,
            label="eLife 2019 — two-phase fit")

    ax.set_xlabel(r"density $k$ (ants$\cdot$cm$^{-2}$)")
    ax.set_ylabel(r"flow $q$ (ants$\cdot$cm$^{-1}\cdot$s$^{-1}$)")
    ax.set_title("Fundamental diagram")
    # L'axe est cadré sur la densité GLOBALE. L'estimateur par fenêtre centrale
    # mesure une densité LOCALE, qui peut dépasser largement la globale quand un
    # amas occupe la fenêtre — c'est une mesure correcte, pas une aberration,
    # mais la laisser fixer l'échelle écraserait le reste du nuage.
    ax.set_xlim(0, agg["k_domain"].max() * 1.12)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_speed_density(agg, fit, path, errorbar="std"):
    """U(k) = 10 q / k.

    Chaque branche n'est tracée que sur SON domaine de validité : la loi
    mono-cinétique jusqu'à k_c, le plateau au-delà. Tracer U = 10 q_j / k
    jusqu'à k -> 0 la ferait diverger et écraserait toute l'échelle.
    """
    plt = _plt(path)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    k_max = agg["k_domain"].max() * 1.05

    pts = _elife_points()
    if pts is not None:
        ax.plot(pts["k"], pts["v_mm_s"], "s", ms=4.5, mfc="none", mec="purple",
                mew=1.2, label="eLife 2019 — published points (fig. 4A)", zorder=2)
    ax.errorbar(agg["k_domain"], agg["u_from_flux"], yerr=agg["u_" + errorbar],
                fmt="o", ms=4, lw=0, elinewidth=1, capsize=2,
                color="#1b3b6f", label="simulation", zorder=3)

    k_low = np.linspace(0.3, fit["k_c"], 200)
    ax.plot(k_low, 10.0 * fit["monod"](k_low) / k_low, "-", color="#2e7d32",
            lw=1.8, label=r"phase 1 (mono-kinetic), $k<k_c$")
    k_high = np.linspace(fit["k_c"], k_max, 200)
    ax.plot(k_high, 10.0 * fit["q_j"] / k_high, "-", color="#c0392b", lw=1.8,
            label=r"phase 2 (plateau $q_j/k$), $k>k_c$")
    ax.axvline(fit["k_c"], color="k", ls="--", lw=1,
               label=f"transition $k_c$ = {fit['k_c']:.1f}")
    ax.axhline(10 * fit["xi_eff"], color="gray", ls=":", lw=1.2,
               label=fr"free-flow speed $\xi_{{eff}}$ = {10 * fit['xi_eff']:.1f} mm/s")

    ax.set_xlabel(r"density $k$ (ants$\cdot$cm$^{-2}$)")
    ax.set_ylabel(r"speed $U$ (mm$\cdot$s$^{-1}$)")
    ax.set_title("Speed-density relation")
    ax.set_xlim(0, k_max)
    ax.set_ylim(0, 1.15 * 10 * fit["xi_eff"])
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


#: Au-delà de cette densité, les traversées de la section deviennent longues et
#: rares (le flot s'est scindé en amas) et le comptage par traversée devient
#: trop bruité pour porter un ajustement : R² passe de 0.98 à 0.93. C'est aussi
#: la borne de la gamme expérimentale (k <= 18).
K_FIT_MAX = 20.0


def plot_contact_number(agg, path, errorbar="std", k_fit_max=K_FIT_MAX):
    """C(k) seul : le nombre de contacts par traversée, l'observable de fond.

    C'est la grandeur que l'article mesure directement (C = 0.61 k) et la seule
    du bloc « contacts » qui n'exige aucune convention sur la durée.

    L'ajustement ne porte que sur `k <= k_fit_max` (voir K_FIT_MAX) ; les points
    au-delà sont tracés mais grisés.
    """
    plt = _plt(path)
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    k = agg["k_domain"].to_numpy()
    c = agg["contacts_per_transit"].to_numpy()
    err = agg["contacts_per_transit_" + errorbar].to_numpy()
    ok = np.isfinite(c) & (k <= k_fit_max)
    beyond = np.isfinite(c) & (k > k_fit_max)

    ax.errorbar(k[ok], c[ok], yerr=err[ok], fmt="o", ms=5, lw=0, elinewidth=1,
                capsize=2, color="#1b3b6f", label="simulation", zorder=3)
    if beyond.any():
        ax.errorbar(k[beyond], c[beyond], yerr=err[beyond], fmt="o", ms=4, lw=0,
                    elinewidth=1, capsize=2, color="#9bb4c9", zorder=2,
                    label=f"$k$ > {k_fit_max:.0f}: transits rare, excluded from fit")
        ax.axvspan(k_fit_max, k.max() * 1.05, color="gray", alpha=0.07)
    if ok.sum() > 2:
        slope, intercept = np.polyfit(k[ok], c[ok], 1)
        kk = np.linspace(0, k.max() * 1.05, 100)
        ax.plot(kk, slope * kk + intercept, "-", color="#2e7d32", lw=1.8,
                label=f"fit ($k \\leq$ {k_fit_max:.0f}): "
                      f"$C$ = {slope:.2f} $k$ + {intercept:.2f}")
    ax.plot(k, ELIFE_CONTACT_SLOPE * k, "--", color="purple", lw=2.0,
            label=f"eLife 2019: $C$ = {ELIFE_CONTACT_SLOPE} $k$")

    ax.set_xlabel(r"density $k$ (ants$\cdot$cm$^{-2}$)")
    ax.set_ylabel("contacts per transit of the section")
    ax.set_title("Number of antennal contacts")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_contact_timing(agg, path, errorbar="std"):
    """Durée des contacts et temps perdu par contact — matériel secondaire.

    Ces deux grandeurs dépendent de conventions (seuil de distance, définition
    de la fin d'un contact, longueur de la section de mesure) : elles éclairent
    le mécanisme mais ne servent pas de validation au même titre que C(k).
    """
    plt = _plt(path)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))
    k = agg["k_domain"].to_numpy()

    axes[0].errorbar(k, agg["mean_duration"], yerr=agg["mean_duration_" + errorbar],
                     fmt="s", ms=4, lw=0, elinewidth=1, capsize=2, color="#e07b39")
    axes[0].set_ylabel("mean contact duration (s)")
    axes[0].set_title("Contact duration")

    axes[1].errorbar(k, agg["delta_T"], yerr=agg["delta_T_" + errorbar], fmt="^",
                     ms=4, lw=0, elinewidth=1, capsize=2, color="#1b3b6f",
                     label="simulation")
    axes[1].axhline(ELIFE_DELTA_T, ls="--", color="purple", lw=2.0,
                    label=fr"eLife: $\Delta T$ = {ELIFE_DELTA_T} s")
    axes[1].set_ylabel(r"$\Delta T$: time lost per contact (s)")
    axes[1].set_title(r"Slope of $T = T_0 + C\,\Delta T$")
    axes[1].legend(fontsize=8)
    for a in axes:
        a.set_xlabel(r"density $k$ (ants$\cdot$cm$^{-2}$)")
        a.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_spacetime(panels, path, vmax=None):
    """Densité le long du pont au cours du temps, un panneau par effectif.

    `panels` : liste de (titre, densités (n_images, n_bins), x, t).
    Les bandes sombres inclinées, si elles apparaissent, sont des ondes
    d'arrêt-redémarrage remontant le flot.
    """
    plt = _plt(path)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.4), squeeze=False)
    if vmax is None:
        vmax = max(np.percentile(d, 99) for _, d, _, _ in panels)
    for ax, (title, dens, x, t) in zip(axes[0], panels):
        im = ax.imshow(dens, aspect="auto", origin="lower", cmap="magma",
                       vmin=0, vmax=vmax,
                       extent=[x[0], x[-1], t[0], t[-1]])
        ax.set_xlabel("x (mm)")
        ax.set_title(title)
    axes[0][0].set_ylabel("time (s)")
    fig.colorbar(im, ax=axes[0], label=r"local density (ants$\cdot$cm$^{-2}$)")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mixing(agg, path):
    """Ablation continue de l'antennation : ségrégation, flux et contacts
    en fonction de `c_comm`, pour chaque densité.

    Les trois panneaux doivent bouger ensemble si l'antennation est bien le
    mécanisme du mélange : la ségrégation chute quand le taux de contacts monte.
    La ligne verticale marque la valeur nominale du modèle.
    """
    plt = _plt(path)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    colors = ("#1b3b6f", "#e07b39", "#2e7d32", "#c0392b")

    panels = (("segregation_excess", "segregation excess (baseline-corrected)",
               "Transverse mixing"),
              ("q_domain", r"flow $q$ (ants$\cdot$cm$^{-1}\cdot$s$^{-1}$)", "Flow"),
              ("contacts_per_ant_per_s", "contacts per ant per second",
               "Contact rate"))

    for ax, (col, ylabel, title) in zip(axes, panels):
        for c, (n, sub) in zip(colors, agg.groupby("n_agents")):
            ax.errorbar(sub["c_comm"], sub[col], yerr=sub[f"{col}_err"],
                        fmt="o-", ms=4, lw=1.2, capsize=2, color=c,
                        label=f"N = {n}  (k = {n / 10:.0f} cm$^{{-2}}$)")
        ax.axvline(12.0, color="gray", ls=":", lw=1.2)
        ax.set_xlabel(r"$c_{\mathrm{comm}}$ (antennation strength)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    axes[0].axhline(0.0, color="k", lw=0.5)
    axes[0].legend(fontsize=8)
    axes[0].text(12.3, axes[0].get_ylim()[1] * 0.95, "nominal value",
                 fontsize=8, color="gray", va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
