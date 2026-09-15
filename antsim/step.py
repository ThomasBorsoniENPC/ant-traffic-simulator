"""
step.py
=======
LE noyau compilé : un pas d'intégration dt pour tous les agents.

Structure d'un pas (mise à jour SYNCHRONE : tout est lu dans `state`, écrit
dans `new_state`) :

  1. voisins visibles (cône, portée, occlusions)                 neighbors.py
  2. dynamique RADIALE   : nu_brake puis relaxation exacte de u  braking.py
  3. dynamique ANGULAIRE : cap cible omega = arg F, puis
     relaxation exacte de theta vers omega, puis bruit           ci-dessous
  4. déplacement, conditions aux bords                           boundaries.py

Dynamique angulaire — pourquoi trois modes
------------------------------------------
L'équation du rapport est `theta_dot = e_theta_perp . F`, soit
`theta_dot = |F| sin(omega - theta)` avec `omega = arg F`. Écrite ainsi, la
norme |F| joue DEUX rôles à la fois : elle fixe la direction visée et la
vitesse à laquelle on y va. Un terme de répulsion intense rend donc le virage
quasi instantané, ce qui n'a pas de sens biologique et oblige à plafonner
|dtheta| — plafond qui mélange alors des échelles hétérogènes (c'est le défaut
identifié dans le code de référence, où la constante de non-pénétration
saturait systématiquement le plafond).

On sépare donc les deux rôles :

    omega   = arg(somme des forces)        <- les c_* ne fixent plus qu'une
                                              PRIORITÉ RELATIVE (la direction)
    theta_dot = lambda sin(omega - theta)  <- lambda = taux de virage

    `field_norm` lambda = |F|, avec plafond |dtheta| <= lambda_max dt
                 C'est l'équation (heading) du rapport de passation, donc le
                 DÉFAUT : sans quoi le simulateur ne reproduirait pas le modèle
                 de référence. Contrepartie assumée : les c_* portent à la fois
                 la direction et l'urgence, et le plafond mélange des échelles.
    `fixed`      lambda = lambda_theta, taux de virage constant
                 Les c_* ne fixent plus qu'une priorité relative (la direction),
                 et lambda_theta — directement calibrable sur la distribution
                 des vitesses angulaires mesurées — EST le plafond. C'est
                 l'extension propre du Persistent Turning Walker (Gautrais
                 et al. 2009) et la variante recommandée pour la calibration.
    `saturating` lambda = lambda_max |F| / (|F| + force_ref)
                 réponse graduée à l'urgence, bornée par construction :
                 compromis entre les deux précédentes.

L'intégration de `theta_dot = lambda sin(omega - theta)` est EXACTE sur le pas
(F gelé) : avec psi = theta - omega,

    tan(psi_new / 2) = tan(psi / 2) e^{-lambda dt},

ce qui converge vers omega par le chemin court, sans dépassement et sans
condition de stabilité — contrairement à un Euler explicite.
"""

import numpy as np
from numba import njit

from .boundaries import apply_ends, apply_wall
from .braking import brake_from_neighbor, brake_from_wall, relax_speed
from .forces import (objective_force, pair_force, wall_escape_force,
                     wall_force)
from .kernels import wrap_angle
from .neighbors import build_cell_list, grid_shape, visible_neighbors

_MODE_FIELD_NORM = 0
_MODE_FIXED = 1
_MODE_SATURATING = 2
_INFLOW = 1


@njit(cache=True, inline="always")
def turn_rate(force_norm, p):
    """Taux de virage lambda selon `angular_mode` (cf. docstring du module)."""
    if p.angular_mode == _MODE_FIXED:
        return p.lambda_theta
    if p.angular_mode == _MODE_SATURATING:
        return p.lambda_max * force_norm / (force_norm + p.force_ref)
    return force_norm                                  # _MODE_FIELD_NORM


@njit(cache=True, inline="always")
def relax_heading(theta, omega, lam, p, dt):
    """Relaxation exacte du cap vers omega sur un pas dt. Renvoie le nouveau cap."""
    psi = wrap_angle(theta - omega)
    psi_new = 2.0 * np.arctan(np.tan(0.5 * psi) * np.exp(-lam * dt))
    dtheta = psi_new - psi
    if p.angular_mode == _MODE_FIELD_NORM:
        cap = p.lambda_max * dt
        if dtheta > cap:
            dtheta = cap
        elif dtheta < -cap:
            dtheta = -cap
    return theta + dtheta


@njit(cache=True)
def step(state, new_state, xi_cruise, group, tie_sign,
         gauss, uniform, p, dt, buf, flag):
    """Un pas dt pour les `n` agents.

    state, new_state : (4, n) = [x ; y ; u ; theta]  (lecture / écriture)
    gauss, uniform   : (n,) tirages pré-calculés côté NumPy — la boucle
                       compilée n'appelle AUCUN générateur aléatoire
    buf, flag        : tampons de voisinage, alloués une fois par pas
    """
    n = state.shape[1]
    px = state[0]
    py = state[1]
    pu = state[2]
    pt = state[3]

    ncx, ncy, cell = grid_shape(p)
    cell_start, order = build_cell_list(px, py, n, ncx, ncy, cell)

    for i in range(n):
        xi = px[i]
        yi = py[i]
        ui = pu[i]
        ti = pt[i]
        gi = group[i]
        xc = xi_cruise[i]
        ts = tie_sign[i]

        # --- termes indépendants des voisins ---------------------------
        nu_brake = brake_from_wall(xi, yi, p)
        fx, fy = objective_force(gi, p)
        wx, wy = wall_force(xi, yi, ti, p)
        fx += wx
        fy += wy

        # --- interactions ----------------------------------------------
        m = visible_neighbors(i, px, py, pt, p, n, cell_start, order,
                              ncx, ncy, cell, buf, flag)
        blocked = 0
        contact_range = 2.0 * p.radius
        for t in range(m):
            j = buf[t]
            nu_brake += brake_from_neighbor(xi, yi, ti, px[j], py[j], p)
            ax, ay = pair_force(xi, yi, ti, gi, xc,
                                px[j], py[j], pt[j], group[j], pu[j], ts, p)
            fx += ax
            fy += ay
            if p.use_wall_escape == 1 and blocked == 0:
                dxj = px[j] - xi
                dyj = py[j] - yi
                if dxj * dxj + dyj * dyj < contact_range * contact_range:
                    blocked = 1

        ex, ey = wall_escape_force(xi, yi, ui, xc, gi, blocked, p)
        fx += ex
        fy += ey

        # --- dynamique radiale ------------------------------------------
        u_new = relax_speed(ui, nu_brake, xc, p, dt)

        # --- dynamique angulaire ----------------------------------------
        force_norm = np.sqrt(fx * fx + fy * fy)
        if force_norm > 0.0:
            omega = np.arctan2(fy, fx)
            t_new = relax_heading(ti, omega, turn_rate(force_norm, p), p, dt)
        else:
            t_new = ti                                 # aucune direction visée
        if p.sigma_theta > 0.0:
            t_new += p.sigma_theta * np.sqrt(dt) * gauss[i]
        t_new = wrap_angle(t_new)

        # --- déplacement et bords ---------------------------------------
        x_new = xi + dt * u_new * np.cos(t_new)
        y_new = yi + dt * u_new * np.sin(t_new)
        x_new, y_new, u_new, t_new = apply_ends(
            x_new, y_new, u_new, t_new, gi, xc, p, uniform[i])
        y_new, t_new = apply_wall(x_new, y_new, t_new, p)

        new_state[0, i] = x_new
        new_state[1, i] = y_new
        new_state[2, i] = u_new
        new_state[3, i] = t_new

