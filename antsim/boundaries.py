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
from .kernels import wrap_angle

_SPECULAR = 0
_SLIDE = 1
_TORUS = 0
_INFLOW = 1
_TORUS_KEEP_Y = 2

#: Marge de sécurité au repositionnement sur une paroi (unités internes = mm).
_WALL_INSET = 1e-6


@njit(cache=True, inline="always")
def apply_wall(x, y, u, theta, p, dt):
    """Replace l'agent dans le canal, corrige son cap et sa vitesse.
    Renvoie (y, u, theta).

    La correction de cap est PLAFONNÉE au même taux de virage maximal que la
    dynamique angulaire (`lambda_max`), sous `cap_wall_turn`. Sans ce plafond,
    la réflexion spéculaire est un saut instantané — jusqu'à pi en un pas,
    mesuré à 0.195 rad en médiane et 2.65 rad au maximum, alors que le pas de
    braquage libre vaut 0.034 rad. Elle échappait ainsi entièrement au plafond
    angulaire (72 % des réflexions le dépassaient) et entretenait un
    CHATTERING : 44 % des contacts successifs d'un même agent se produisaient
    en moins de 0.05 s, avec un dixième à deux pas d'intervalle.

    Le non-flux reste garanti sans condition : il est assuré par le
    repositionnement de `y` dans le canal, à chaque pas, indépendamment du cap.
    Un agent dont le cap pointe encore vers l'extérieur longe donc la paroi en
    tournant à vitesse bornée, au lieu d'être renversé d'un coup.

    Le plafond seul ne suffit pas à stabiliser : pendant les quelques pas que
    dure le demi-tour, le cap pointe toujours dehors et l'agent continue de
    pousser dans la paroi, pour être repositionné à chaque pas. Sous
    `wall_cut_normal_speed`, la COMPOSANTE NORMALE SORTANTE de la vitesse est
    donc retirée : il ne reste que la part tangentielle,

        u <- u * sqrt(1 - (e_theta . n)^2),

    ce qui annule la vitesse d'un agent face à la paroi (il tourne sur place) et
    la laisse intacte pour un agent quasi tangent. La relaxation de `u` la
    ramène ensuite vers xi dès que le cap s'est dégagé : la coupure est un
    amortissement transitoire, pas une pénalité durable.
    """
    y_bot, y_top = wall_y(x, p)
    if y_bot <= y <= y_top:
        return y, u, theta

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
            theta_target = np.arctan2(vy / norm_v, vx / norm_v)
            if p.cap_wall_turn == 1:
                d = wrap_angle(theta_target - theta)
                cap = p.lambda_max * dt
                if d > cap:
                    d = cap
                elif d < -cap:
                    d = -cap
                theta = wrap_angle(theta + d)
            else:
                theta = theta_target
        if p.wall_cut_normal_speed == 1:
            # v_dot_n < 0 : part sortante. Il ne reste que le tangentiel.
            tang2 = 1.0 - v_dot_n * v_dot_n
            if tang2 < 0.0:
                tang2 = 0.0
            u = u * np.sqrt(tang2)
    return y, u, theta


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
