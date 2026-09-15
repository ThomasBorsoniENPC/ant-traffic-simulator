"""
geometry.py
===========
Le canal : parois, normales, distance à la paroi, occlusion visuelle.

Phase 1 : géométrie `rectangle` seule (le pont expérimental), traitée
ANALYTIQUEMENT. Le code de référence balayait 250 points de paroi par agent et
par pas pour trouver le point de paroi le plus proche ; sur un rectangle c'est
exactement `min(y, H - y)`, d'où un gain de l'ordre de deux décades sur cette
partie de la boucle.

Ajouter une géométrie (goulot, double goulot, bosse — Phase 3)
--------------------------------------------------------------
1. déclarer son code entier dans `params.GEOMETRY_CODES` ;
2. ajouter la branche correspondante dans `wall_y`, `closest_wall` et
   `segment_leaves_channel` ci-dessous (branchement sur `p.geom_type`, JAMAIS
   un pointeur de fonction global : la boucle chaude doit rester compilée une
   seule fois) ;
3. pour une géométrie NON CONVEXE, `segment_leaves_channel` doit repasser à un
   échantillonnage du segment — l'implémentation exacte ci-dessous s'appuie sur
   la convexité du rectangle.
"""

import numpy as np
from numba import njit

_RECTANGLE = 0


# ──────────────────────────────────────────────────────────────────────────
#  Parois
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True, inline="always")
def wall_y(x, p):
    """Ordonnées (y_bas, y_haut) des parois à l'abscisse x."""
    if p.geom_type == _RECTANGLE:
        return 0.0, p.height
    # Géométrie non reconnue : on retombe sur le rectangle englobant plutôt que
    # de lever (impossible en nopython). `params._validate` a déjà filtré.
    return 0.0, p.height


@njit(cache=True, inline="always")
def is_outside(x, y, p):
    """True si (x, y) est hors du canal (les extrémités en x ne sont pas des parois)."""
    y_bot, y_top = wall_y(x, p)
    return y < y_bot or y > y_top


@njit(cache=True, inline="always")
def closest_wall(x, y, p):
    """(dist, sx, sy) : distance à la paroi la plus proche et vecteur unitaire
    de l'agent VERS cette paroi (le `sigma_wall` du rapport).

    La normale INTERNE est simplement (-sx, -sy).
    """
    if p.geom_type == _RECTANGLE:
        y_bot, y_top = 0.0, p.height
        d_bot = y - y_bot
        d_top = y_top - y
        if d_bot <= d_top:
            return d_bot, 0.0, -1.0
        return d_top, 0.0, 1.0
    return 0.0, 0.0, 1.0


@njit(cache=True, inline="always")
def inward_normal(x, y, p):
    """(nx, ny, dist) : normale unitaire dirigée vers l'intérieur du canal."""
    dist, sx, sy = closest_wall(x, y, p)
    return -sx, -sy, dist


# ──────────────────────────────────────────────────────────────────────────
#  Occlusion visuelle par les parois
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True, inline="always")
def segment_leaves_channel(x0, y0, x1, y1, p):
    """True si le segment [(x0,y0), (x1,y1)] sort du canal.

    Le rectangle est CONVEXE : le segment reste dans le canal si et seulement
    si ses deux extrémités y sont. Le test est donc exact, et non échantillonné.

    En pratique, sur un pont rectangulaire, la seule occlusion possible par une
    paroi concerne un point de visée situé hors du canal (la tête ou l'arrière
    d'une fourmi qui longe le bord) ; l'occlusion mutuelle entre agents, elle,
    est traitée dans `neighbors.py`.
    """
    return is_outside(x0, y0, p) or is_outside(x1, y1, p)
