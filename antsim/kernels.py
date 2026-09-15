"""
kernels.py
==========
Profils scalaires élémentaires dont sont faites TOUTES les magnitudes de force
du modèle. Chaque magnitude se factorise en

    magnitude = c x (profil radial) x (facteur angulaire)

conformément au § « Noyaux d'interaction sans priorité » de `rapport_LLM.tex`.

Ces fonctions sont volontairement sans état, sans paramètre global et sans
branchement sur la géométrie : elles ne prennent que des scalaires. Les seuils
de support (rho <= l, cos >= 1 - eps) sont testés par l'APPELANT, dans
`forces.py` — ici on suppose l'argument dans le support.
"""

import numpy as np
from numba import njit


# ──────────────────────────────────────────────────────────────────────────
#  Profils radiaux
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True, inline="always")
def profile_singular(rho, ell, rho_reg):
    """(ell - rho) / rho : décroissance singulière, nulle en rho = ell.

    C'est le profil `(l - rho)_+ / rho` du rapport. `rho_reg` est un plancher
    qui évite la division par zéro en cas de superposition exacte ; il est
    choisi assez petit pour être inerte (cf. `ModelParams.rho_reg`). La
    régularisation lisse `1/sqrt(rho^2 + delta^2)` suggérée par le backlog
    (piste 5) se substituerait ici, et ici seulement.
    """
    d = rho if rho > rho_reg else rho_reg
    return (ell - rho) / d


@njit(cache=True, inline="always")
def profile_ramp(rho, a, b):
    """(rho - a) / (b - a) : rampe linéaire croissante de 0 en a à 1 en b."""
    return (rho - a) / (b - a)


# ──────────────────────────────────────────────────────────────────────────
#  Facteur angulaire (cône de perception)
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True, inline="always")
def cone_factor(cos_alpha, eps):
    """[(eps - (1 - cos_alpha)) / eps]_+ : 1 droit devant, 0 au bord du cône.

    `cos_alpha` est le cosinus de l'angle entre le cap de l'agent et la
    direction du voisin ; `eps` l'ouverture effective (le support est
    cos_alpha >= 1 - eps). Le résultat est tronqué à 0 par le bas mais pas
    par le haut : avec eps > 1 le facteur dépasse 1 et le terme devient
    quasi omnidirectionnel (c'est le cas voulu de `eps_repulse = 2`).
    """
    z = (cos_alpha - (1.0 - eps)) / eps
    return z if z > 0.0 else 0.0


@njit(cache=True, inline="always")
def cone_factor_pow(cos_alpha, eps, alpha_exp):
    """`cone_factor` élevé à la puissance `alpha_exp` (sévérité des contacts
    latéraux ; `alpha_brake = 4` concentre le freinage droit devant)."""
    z = cone_factor(cos_alpha, eps)
    if z <= 0.0:
        return 0.0
    return z ** alpha_exp


# ──────────────────────────────────────────────────────────────────────────
#  Utilitaires vectoriels / angulaires
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True, inline="always")
def wrap_angle(a):
    """Ramène un angle dans (-pi, pi]."""
    return (a + np.pi) % (2.0 * np.pi) - np.pi


@njit(cache=True, inline="always")
def sign_nonzero(v):
    """Signe de v, avec sign(0) = +1 (les cas v = 0 sont traités en amont)."""
    return 1.0 if v >= 0.0 else -1.0
