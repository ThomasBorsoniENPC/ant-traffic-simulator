"""
forces.py
=========
Les contributions au CHAMP D'ORIENTATION `F` (dynamique angulaire). Le freinage,
qui appartient à la dynamique radiale, est dans `braking.py`.

Chaque terme se factorise en (magnitude) x (direction), conformément au
§ « Noyaux d'interaction sans priorité » de `rapport_LLM.tex` :

    groupes OPPOSÉS (g_i g_j = -1)
      AVOID  évitement latéral, direction sigma_perp choisie pour ne pas
             aggraver la collision                        (eq. avoid-opp)
      COMM   antennation : attraction vers l'autre, à moyenne portée
             [l_comm_min, l_comm]                          (eq. comm-kernel)
    MÊME groupe (g_i g_j = +1)
      REPULSE   répulsion latérale (non-pénétration « soft »)  (eq. avoid-same)
      OVERTAKE  dépassement d'un co-directionnel plus lent     (eq. overtake)
      FOLLOW    suivi d'un co-directionnel                     (eq. follow)
    toujours
      OBJECTIVE force constante k_obj * (g, 0)
      WALL      thigmotactisme, pondéré par la composante TANGENTIELLE du cap

Toutes les fonctions renvoient un couple (fx, fy) : aucun tableau n'est alloué
dans la boucle chaude.

Départage déterministe
----------------------
Deux directions d'évitement sont dégénérées lorsque le voisin est exactement
dans l'axe (produit scalaire nul). Le code de référence tirait à pile ou face
dans la boucle compilée, ce qui imposait de gérer un second générateur
aléatoire. Ici le départage utilise `tie_sign`, un signe +/-1 tiré UNE FOIS par
agent à l'initialisation depuis l'unique générateur du run : même effet,
reproductibilité sans second RNG (cf. README, § reproductibilité).
"""

import numpy as np
from numba import njit

from .geometry import closest_wall, inward_normal
from .kernels import cone_factor, profile_ramp, profile_singular, sign_nonzero


# ──────────────────────────────────────────────────────────────────────────
#  Directions de braquage
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True, inline="always")
def steer_direction_avoid(sx, sy, ti, tj, tie_sign):
    """Direction latérale d'évitement d'un agent de sens opposé.

    On braque du côté `-sign(e_ti . sigma_perp)` lorsque les deux agents
    tournent déjà du même côté et que i « mène » la rencontre, et du côté
    opposé sinon : le but est de ne pas choisir une correction qui accentue la
    collision. (Comme le code de référence, la comparaison des composantes
    radiales se fait sans valeur absolue ; l'évitement n'étant actif que dans
    le cône avant, l'écart avec l'écriture du rapport ne porte que sur le signe
    de e_tj . sigma.)
    """
    perp_x = -sy
    perp_y = sx
    si = np.cos(ti) * perp_x + np.sin(ti) * perp_y
    if si == 0.0:
        return tie_sign * perp_x, tie_sign * perp_y
    sj = np.cos(tj) * perp_x + np.sin(tj) * perp_y
    ci = np.cos(ti) * sx + np.sin(ti) * sy
    cj = np.cos(tj) * sx + np.sin(tj) * sy
    if si * sj >= 0.0 and ci >= cj:
        s = -sign_nonzero(si)
    else:
        s = sign_nonzero(si)
    return s * perp_x, s * perp_y


@njit(cache=True, inline="always")
def steer_direction_overtake(sx, sy, group, tie_sign):
    """Direction de dépassement : le côté de `sigma_perp` qui va vers l'objectif."""
    perp_x = -sy
    perp_y = sx
    proj = perp_x * group          # sigma_perp . (g, 0)
    if proj == 0.0:
        return tie_sign * perp_x, tie_sign * perp_y
    s = sign_nonzero(proj)
    return s * perp_x, s * perp_y


# ──────────────────────────────────────────────────────────────────────────
#  Termes indépendants des voisins
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True, inline="always")
def objective_force(group, p):
    """Cap global : vers la source (+x) si g = +1, vers le nid (-x) si g = -1."""
    return p.k_obj * group, 0.0


_WALL_TANGENTIAL = 0
_WALL_OUTWARD_DAMPING = 1
_WALL_STANDOFF = 2


@njit(cache=True, inline="always")
def wall_force(x, y, theta, p):
    """Thigmotactisme. Trois lois, sélectionnées par `p.wall_law`.

    Toutes renvoient une force colinéaire à `sigma_wall` (vecteur unitaire de
    l'agent vers la paroi la plus proche) ; elles ne diffèrent que par le signe
    et la pondération angulaire. `cos_hw = e_theta . sigma_wall` est positif
    quand l'agent se dirige vers la paroi, nul quand il la longe.

    `tangential` — la loi du rapport de passation. Attraction pondérée par la
    composante tangentielle du cap, donc MAXIMALE quand l'agent longe déjà la
    paroi, et nulle quand il la vise de face. Elle produit bien un suivi de
    bord, mais c'est un état ABSORBANT : la force ne change jamais de signe, si
    bien que rien ne ramène jamais vers le centre du canal, et l'agent qu'un
    évitement écarte de la paroi y est aussitôt ramené — le désengagement n'est
    jamais mené à terme. Voir README §8 pour les mesures.

    `outward_damping` — ne retient que l'agent qui S'ÉLOIGNE de la paroi ; nulle
    s'il la longe ou s'en approche. Corrige le défaut de pondération de la
    précédente, mais reste unidirectionnelle : elle ne peut que dissiper
    l'éloignement, jamais le provoquer.

    `standoff` — rappel vers une distance de consigne `l_wall_standoff` :
    RÉPULSIVE en deçà, attractive au-delà, nulle à la consigne. L'agent qui
    longe la paroi à la bonne distance ne subit AUCUNE force : il est libre de
    s'en écarter si un voisin l'y pousse, et il n'y est plus écrasé.

    Ce qu'elle produit, MESURÉ aux paramètres par défaut : une PRÉFÉRENCE de
    bord, pas un équilibre. Un agent isolé oscille sur toute la largeur du
    canal (distance à la paroi de 0.5 à 5.0 mm, moyenne 1.9 sur une demi-
    largeur de 5), mais ne s'écrase jamais contre le bord (3 % d'agents à moins
    de 0.5 mm, contre 30 % pour `tangential`). C'est le gain réel de cette loi ;
    la distance de consigne est un attracteur mou, pas un point fixe.

    """
    if p.use_wall_attraction == 0:
        return 0.0, 0.0
    dist, sx, sy = closest_wall(x, y, p)
    if dist > p.l_wall_max:
        return 0.0, 0.0
    cos_hw = np.cos(theta) * sx + np.sin(theta) * sy

    if p.wall_law == _WALL_TANGENTIAL:
        if cos_hw < 0.0 or dist <= p.l_wall_min:
            return 0.0, 0.0
        tangential = 1.0 - cos_hw * cos_hw
        if tangential < 0.0:
            tangential = 0.0
        mag = (p.c_wall * profile_ramp(dist, p.l_wall_min, p.l_wall_max)
               * np.sqrt(tangential))
        return mag * sx, mag * sy

    if p.wall_law == _WALL_OUTWARD_DAMPING:
        if cos_hw >= 0.0 or dist <= p.l_wall_min:
            return 0.0, 0.0
        mag = (p.c_wall * profile_ramp(dist, p.l_wall_min, p.l_wall_max)
               * (-cos_hw))
        return mag * sx, mag * sy

    # _WALL_STANDOFF
    d_star = p.l_wall_standoff
    if dist >= d_star:
        offset = (dist - d_star) / (p.l_wall_max - d_star)      # attraction
    else:
        offset = (dist - d_star) / d_star                       # répulsion (< 0)
    mag = p.c_wall * offset
    return mag * sx, mag * sy


@njit(cache=True, inline="always")
def wall_escape_force(x, y, u, xi_cruise_i, group, blocked, p):
    """Dégagement d'un agent bloqué contre une paroi (désactivé par défaut).

    Une configuration SYMÉTRIQUE — un agent plaqué au bord, la foule de l'autre
    côté — annule la force latérale de dégagement : la paire s'immobilise et une
    file s'accumule le long de la paroi. Le remède du rapport de passation
    (annexe « Numerical scheme ») pousse vers l'intérieur du canal les seuls
    agents réellement bloqués — collés au bord, voisin à moins de 2R, et lents —
    avec une intensité LÉGÈREMENT différente selon le groupe : c'est cette
    asymétrie qui brise la symétrie du face-à-face. Ne se déclenchant que sur
    des agents bloqués, le terme laisse l'écoulement libre intact.

    Il remplace un biais latéral appliqué à un seul groupe, écarté faute de
    justification biologique.
    """
    if p.use_wall_escape == 0 or blocked == 0:
        return 0.0, 0.0
    if u >= 0.5 * xi_cruise_i:                 # pas réellement bloqué
        return 0.0, 0.0
    nx, ny, dist = inward_normal(x, y, p)
    if dist > p.l_wall_stuck:                  # pas collé à une paroi
        return 0.0, 0.0
    if group > 0.0:
        gain = p.c_wall_escape * (1.0 + p.wall_escape_asym)
    else:
        gain = p.c_wall_escape * (1.0 - p.wall_escape_asym)
    return gain * nx, gain * ny


# ──────────────────────────────────────────────────────────────────────────
#  Interaction de paire
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def pair_force(xi, yi, ti, gi, xi_cruise_i,
               xj, yj, tj, gj, uj, tie_sign, p):
    """Contribution du voisin j au champ d'orientation de i.

    `xi_cruise_i` est la vitesse de croisière propre de i (porte du dépassement),
    `uj` la vitesse instantanée de j.
    """
    dx = xj - xi
    dy = yj - yi
    rho2 = dx * dx + dy * dy
    if rho2 <= 0.0:
        return 0.0, 0.0
    rho = np.sqrt(rho2)
    sx = dx / rho
    sy = dy / rho

    cos_i = np.cos(ti) * sx + np.sin(ti) * sy          # e_theta_i . sigma
    fx = 0.0
    fy = 0.0

    if gi * gj < 0.0:
        # ---------------- groupes opposés : évitement + antennation -------
        if (p.use_avoid_steer == 1 and rho <= p.l_avoid_steer
                and cos_i >= 1.0 - p.eps_avoid_steer):
            mag = (p.c_avoid_steer
                   * profile_singular(rho, p.l_avoid_steer, p.rho_reg)
                   * cone_factor(cos_i, p.eps_avoid_steer))
            dxs, dys = steer_direction_avoid(sx, sy, ti, tj, tie_sign)
            fx += mag * dxs
            fy += mag * dys

        if (p.use_comm == 1 and p.l_comm_min < rho <= p.l_comm
                and cos_i >= 1.0 - p.eps_comm):
            mag = (p.c_comm
                   * profile_ramp(rho, p.l_comm_min, p.l_comm)
                   * cone_factor(cos_i, p.eps_comm))
            fx += mag * sx
            fy += mag * sy

    else:
        # ---------------- même groupe : répulsion + dépassement + suivi ----
        sin_i = -np.sin(ti) * sx + np.cos(ti) * sy     # e_theta_i_perp . sigma
        abs_sin_i = abs(sin_i)

        if (p.use_repulse == 1 and rho <= p.l_repulse
                and abs_sin_i >= 1.0 - p.eps_repulse):
            mag = (p.c_repulse
                   * profile_singular(rho, p.l_repulse, p.rho_reg)
                   * cone_factor(abs_sin_i, p.eps_repulse))
            fx -= mag * sx
            fy -= mag * sy

        if (p.use_overtake == 1 and rho <= p.l_overtake
                and cos_i >= 1.0 - p.eps_overtake):
            u_gate = p.alpha_overtake * xi_cruise_i
            if uj <= u_gate:
                mag = (p.c_overtake
                       * profile_singular(rho, p.l_overtake, p.rho_reg)
                       * cone_factor(cos_i, p.eps_overtake))
                if p.overtake_smooth_gate == 1 and u_gate > 0.0:
                    # Porte lisse du rapport (eq. overtake-mag) : préférable
                    # pour la régularité de la limite cinétique.
                    mag *= (u_gate - uj) / u_gate
                dxs, dys = steer_direction_overtake(sx, sy, gi, tie_sign)
                fx += mag * dxs
                fy += mag * dys

        if (p.use_follow == 1 and p.l_follow1 < rho <= p.l_follow2
                and cos_i >= 1.0 - p.eps_follow):
            mag = (p.c_follow
                   * profile_ramp(rho, p.l_follow1, p.l_follow2)
                   * cone_factor(cos_i, p.eps_follow))
            fx += mag * sx
            fy += mag * sy

    return fx, fy
