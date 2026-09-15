"""
boundaries.py
=============
Conditions aux bords du pont.

Parois latérales — les deux modes réalisent le non-flux (q . n = 0) et ne
diffèrent que par la distribution des orientations au voisinage du bord :
  - `specular` : la composante normale du cap est RENVERSÉE (réflexion miroir).
    C'est l'idéalisation cinétique du rapport (eq. specular) et la variante
    retenue par les tests antérieurs du projet, qui dénoue mieux les files
    plaquées contre la paroi.
  - `slide`    : la composante normale sortante est RETIRÉE, l'agent longe la
    paroi. Plus proche du thigmotactisme, mais concentre la masse sortante sur
    les directions tangentes (ce n'est pas un opérateur de réflexion cinétique).

Extrémités du pont :
  - `torus`  : x périodique, avec l'ordonnée RETIRÉE AU HASARD à la ré-entrée.
    Sur un tore de longueur finie, un agent repasse indéfiniment au même
    endroit ; conserver son ordonnée laisse alors une structure transverse se
    figer. Le retirage remixe le flot — l'agent qui revient est, du point de
    vue transverse, une nouvelle fourmi entrant sur le pont.
  - `torus_keep_y` : x périodique, ordonnée conservée (trajectoire continue).
  - `inflow` : l'agent qui sort est réinjecté à l'entrée de SON groupe (g = +1
    à gauche, g = -1 à droite), y tiré uniformément, vitesse et cap remis à
    l'état d'une fourmi entrante. Le nombre d'agents reste constant, ce qui est
    ce qu'on veut pour un diagramme fondamental à densité contrôlée.
"""

import numpy as np
from numba import njit

from .geometry import inward_normal, wall_y

_SPECULAR = 0
_SLIDE = 1
_TORUS = 0
_INFLOW = 1
_TORUS_KEEP_Y = 2

#: Marge de sécurité au repositionnement sur une paroi (unités internes = mm).
_WALL_INSET = 1e-6


@njit(cache=True, inline="always")
def apply_wall(x, y, theta, p):
    """Replace l'agent dans le canal et corrige son cap. Renvoie (y, theta)."""
    y_bot, y_top = wall_y(x, p)
    if y_bot <= y <= y_top:
        return y, theta

    if y < y_bot:
        y = y_bot + _WALL_INSET
    else:
        y = y_top - _WALL_INSET

    nx, ny, _ = inward_normal(x, y, p)
    vx = np.cos(theta)
    vy = np.sin(theta)
    v_dot_n = vx * nx + vy * ny
    if v_dot_n < 0.0:                      # le cap pointe encore vers l'extérieur
        if p.wall_mode == _SLIDE:
            vx -= v_dot_n * nx
            vy -= v_dot_n * ny
        else:                              # _SPECULAR
            vx -= 2.0 * v_dot_n * nx
            vy -= 2.0 * v_dot_n * ny
        norm_v = np.sqrt(vx * vx + vy * vy)
        if norm_v > 1e-12:
            theta = np.arctan2(vy / norm_v, vx / norm_v)
    return y, theta


@njit(cache=True, inline="always")
def apply_ends(x, y, u, theta, group, xi_cruise_i, p, uniform_draw):
    """Traite la sortie par une extrémité. Renvoie (x, y, u, theta).

    `uniform_draw` est un tirage uniforme sur [0, 1) pré-calculé côté NumPy
    (aucun générateur aléatoire n'est appelé dans le code compilé) ; il n'est
    consommé qu'en mode `inflow`.
    """
    if 0.0 <= x <= p.length:
        return x, y, u, theta

    if p.x_mode != _INFLOW:
        x_new = x % p.length
        if p.x_mode == _TORUS_KEEP_Y:
            return x_new, y, u, theta
        # _TORUS : remixage transverse
        y_bot, y_top = wall_y(x_new, p)
        lo = y_bot + p.radius
        hi = y_top - p.radius
        if hi <= lo:
            lo, hi = y_bot, y_top
        return x_new, lo + uniform_draw * (hi - lo), u, theta

    # _INFLOW : réinjection à l'entrée du groupe
    if group > 0.0:
        x_new = _WALL_INSET
        theta_new = 0.0
    else:
        x_new = p.length - _WALL_INSET
        theta_new = np.pi
    y_bot, y_top = wall_y(x_new, p)
    lo = y_bot + p.radius
    hi = y_top - p.radius
    if hi <= lo:                            # canal plus étroit qu'une fourmi
        lo = y_bot
        hi = y_top
    return x_new, lo + uniform_draw * (hi - lo), xi_cruise_i, theta_new
