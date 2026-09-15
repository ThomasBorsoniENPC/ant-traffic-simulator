"""
diagnose_wall.py
================
Départage les lois de thigmotactisme (`ModelParams.wall_law`) sur trois
critères, qui sont le cahier des charges du comportement voulu :

  A. un agent SEUL longe la paroi — il ne s'en détache pas, mais ne s'y écrase
     pas non plus ;
  B. deux agents de sens opposés qui se rencontrent CONTRE la paroi se
     désengagent — ils se croisent sans se superposer ;
  C. en foule, pas de condensation contre une paroi.

A et B pilotent directement le noyau `step` sur une configuration construite à
la main : ce sont des scénarios contrôlés, pas des runs statistiques.

    python scripts/diagnose_wall.py
"""

import numpy as np

import _bootstrap  # noqa: F401

from antsim import ModelParams, SimParams, build_kernel_params, run
from antsim.step import step

LAWS = ("tangential", "outward_damping", "standoff")
DT = 0.01


def drive(model, state0, group, duration, seed=0, tie_sign=None):
    """Pilote le noyau sur un état initial imposé. Renvoie (n_pas+1, 4, n)."""
    p = build_kernel_params(model)
    n = state0.shape[1]
    rng = np.random.default_rng(seed)
    state = state0.copy()
    new = np.empty_like(state)
    buf = np.empty(n, dtype=np.int64)
    flag = np.empty(n, dtype=np.bool_)
    xi_cruise = np.full(n, model.xi_cruise)
    if tie_sign is None:
        tie_sign = np.ones(n)

    n_steps = int(duration / DT)
    rec = np.empty((n_steps + 1, 4, n))
    rec[0] = state
    for k in range(n_steps):
        step(state, new, xi_cruise, group, tie_sign,
             rng.standard_normal(n), rng.random(n), p, DT, buf, flag)
        state, new = new, state
        rec[k + 1] = state
    return rec


def criterion_a(law):
    """Agent isolé : suivi de bord depuis la paroi, et capture depuis le large."""
    model = ModelParams(wall_law=law, sigma_theta=0.0)
    out = {}
    for name, y0 in (("part du bord (y=1)", 1.0), ("part du large (y=3.5)", 3.5)):
        state0 = np.array([[20.0], [y0], [model.xi_cruise], [0.0]])
        rec = drive(model, state0, np.array([1.0]), duration=30.0)
        d = np.minimum(rec[:, 1, 0], model.height - rec[:, 1, 0])
        out[name] = (d[len(d) // 2:].mean(), d.min(), np.mean(d < 0.2))
    return out


def criterion_b(law, symmetric):
    """Paire frontale contre la paroi : se croisent-ils sans se superposer ?

    `symmetric` : les deux agents exactement à la même ordonnée ET avec le même
    signe de départage — c'est le cas dégénéré où les braquages d'évitement
    s'annulent. Sinon un décalage réaliste de 0.2 mm.

    La durée est courte (1.5 s) pour qu'aucun agent ne fasse le tour du tore :
    la comparaison des abscisses finales garde alors un sens.
    """
    model = ModelParams(wall_law=law, sigma_theta=0.0)
    y0 = [1.0, 1.0] if symmetric else [1.0, 1.2]
    tie = np.array([1.0, 1.0]) if symmetric else np.array([1.0, -1.0])
    state0 = np.array([[47.0, 53.0], y0,
                       [model.xi_cruise, model.xi_cruise], [0.0, np.pi]])
    rec = drive(model, state0, np.array([1.0, -1.0]), duration=1.5, tie_sign=tie)
    sep = np.hypot(rec[:, 0, 0] - rec[:, 0, 1], rec[:, 1, 0] - rec[:, 1, 1])
    return sep.min(), rec[-1, 0, 0] > rec[-1, 0, 1]


def criterion_c(law, seeds=(0, 1, 2)):
    """Foule : profil transverse, écrasement contre la paroi, vitesse, recouvrement.

    Deux mesures distinctes, là où une seule confondait deux choses :
      - `%écrasé`  : agents à moins de 0.5 mm de la paroi, c'est-à-dire enfoncés
        dedans (le rayon vaut 1) — c'est ça, la pathologie ;
      - `%moitié centrale` : agents dans la moitié centrale du canal — un suivi
        de bord légitime en laisse peu, une condensation n'en laisse aucun.
    """
    model = ModelParams(wall_law=law, sigma_theta=0.0)
    crushed, central, speed, dmin = [], [], [], []
    for s in seeds:
        r = run(model, SimParams(num_agents=80, duration=60.0, seed=s), progress=False)
        half = r.n_frames // 2
        y = r.y[half:]
        h = model.height
        d_wall = np.minimum(y, h - y)
        crushed.append(np.mean(d_wall < 0.5))
        central.append(np.mean(np.abs(y - 0.5 * h) < 0.25 * h))
        speed.append(r.u[half:].mean())
        d = []
        for f in range(0, r.x.shape[0] - half, 40):
            pts = np.stack([r.x[half + f], y[f]], 1)
            m = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
            np.fill_diagonal(m, 1e9)
            d.append(m.min())
        dmin.append(np.mean(d))
    return np.mean(crushed), np.mean(central), np.mean(speed), np.mean(dmin)


def main():
    print("A. AGENT ISOLÉ — distance à la paroi (mm). Idéal : se stabilise autour")
    print("   d'une distance de l'ordre du rayon (1 mm), sans contact (0 % à <0.2).")
    print(f"\n{'loi':>17} {'scénario':>24} {'d moyen':>9} {'d min':>7} {'%contact':>9}")
    for law in LAWS:
        for name, (dm, dmin, frac) in criterion_a(law).items():
            print(f"{law:>17} {name:>24} {dm:>9.2f} {dmin:>7.2f} {frac:>9.2f}")

    print("\nB. PAIRE FRONTALE CONTRE LA PAROI — séparation minimale (mm).")
    print("   Idéal : se croisent sans descendre sous ~1 mm (2R = 2 : contact).")
    print(f"\n{'loi':>17} {'sépar. min':>12} {'croisent':>10}   cas")
    for law in LAWS:
        for symmetric in (False, True):
            sep, crossed = criterion_b(law, symmetric)
            case = "décalage 0.2 mm" if not symmetric else "exactement symétrique"
            print(f"{law:>17} {sep:>12.2f} {'oui' if crossed else 'NON':>10}   {case}")

    print("\nC. FOULE (N=80, 60 s, 3 graines).")
    print("   %écrasé = à moins de 0.5 mm de la paroi (rayon = 1) : c'est la pathologie.")
    print("   %centre = dans la moitié centrale du canal. Uniforme donnerait 0.50.")
    print(f"\n{'loi':>17} {'%écrasé':>9} {'%centre':>9} {'vitesse':>9} {'d_min paire':>13}")
    for law in LAWS:
        cr, ce, u, d = criterion_c(law)
        print(f"{law:>17} {cr:>9.2f} {ce:>9.2f} {u:>9.2f} {d:>13.2f}")


if __name__ == "__main__":
    main()
