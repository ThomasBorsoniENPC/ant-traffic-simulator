"""
braking.py
==========
La dynamique RADIALE : le taux de freinage `nu_brake` et l'intégration exacte
de la vitesse scalaire.

C'est ici que se trouve le mécanisme central du modèle. Le freinage n'est pas
une force soustractive (qui donnerait une capacité maximale, à la Greenshields)
mais un TAUX ajouté à la relaxation :

    du/dt = nu_c (xi - u) - nu_brake u
          = (nu_c + nu_brake) (u* - u),      u* = nu_c xi / (nu_c + nu_brake).

Sous une fermeture mono-cinétique où nu_brake croît linéairement avec la densité
locale (nu_brake ~ beta_bar k), la vitesse d'équilibre est

    U(k) = xi nu_c / (nu_c + beta_bar k)       (loi de type Monod)

et le flux q = kU SATURE au lieu de s'effondrer : c'est l'origine mécanistique
du plateau « two-phase » de Poissonnier et al. 2019. Voir `rapport_LLM.tex`,
eq. micro-speed et beta-kernel.

Attention : cette saturation suppose que `nu_brake` voie la densité PHYSIQUE.
Si les agents se recouvrent sans borne à haute densité, la proportionnalité
`nu_brake ~ k` se rompt. C'est la raison d'être de `contacts.py` (désactivé par
défaut).
"""

import numpy as np
from numba import njit

from .geometry import wall_y
from .kernels import cone_factor_pow, profile_singular


@njit(cache=True, inline="always")
def brake_from_neighbor(xi, yi, ti, xj, yj, p):
    """Contribution du voisin j au taux de freinage de i.

    Freine si j est proche (rho <= l_brake) et DEVANT (cône avant d'ouverture
    eps_brake). S'applique quel que soit le groupe : c'est un arrêt de contact,
    pas une interaction sociale. L'exposant `alpha_brake` concentre l'effet
    droit devant.
    """
    if p.use_brake == 0:
        return 0.0
    dx = xj - xi
    dy = yj - yi
    rho2 = dx * dx + dy * dy
    if rho2 <= 0.0 or rho2 > p.l_brake * p.l_brake:
        return 0.0
    rho = np.sqrt(rho2)
    cos_i = (np.cos(ti) * dx + np.sin(ti) * dy) / rho
    if cos_i < 1.0 - p.eps_brake:
        return 0.0
    return (p.c_brake
            * profile_singular(rho, p.l_brake, p.rho_reg)
            * cone_factor_pow(cos_i, p.eps_brake, p.alpha_brake))


@njit(cache=True, inline="always")
def brake_from_wall(x, y, p):
    """Moitié RADIALE de la condition de non-flux : un agent qui a atteint une
    paroi cesse d'avancer.

    La moitié angulaire (retrait ou renversement de la composante normale du
    cap) est dans `boundaries.py`. `nu_wall_stop` est volontairement énorme
    (arrêt quasi instantané) ; le remplacer par un profil lisse en distance est
    une piste identifiée du backlog — c'est le seul endroit à modifier.
    """
    y_bot, y_top = wall_y(x, p)
    if y <= y_bot or y >= y_top:
        return p.nu_wall_stop
    return 0.0


@njit(cache=True, inline="always")
def relax_speed(u, nu_brake, xi_cruise_i, p, dt):
    """Intégration EXACTE de la vitesse sur un pas dt (freinage gelé).

        u <- u e^{-r dt} + u* (1 - e^{-r dt}),   r = nu_c + nu_brake.

    Inconditionnellement stable, sans dépassement, et u >= 0 est préservé
    exactement (combinaison convexe de deux quantités positives) — ce qui
    importe puisque le modèle interdit la marche arrière.
    """
    rate = p.nu_cruise + nu_brake
    coef = np.exp(-rate * dt)
    u_target = p.nu_cruise * xi_cruise_i / rate
    return u * coef + u_target * (1.0 - coef)
