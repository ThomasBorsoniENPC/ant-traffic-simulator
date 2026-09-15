"""
neighbors.py
============
Champ de vision d'un agent : voisins à portée, dans le cône, non masqués par
une paroi ni par un autre agent.

Deux niveaux :
  - une `cell list` (hachage spatial régulier de pas `l_vision`) qui ramène la
    recherche de voisins de O(N^2) à O(N) amorti — tout voisin à distance
    <= l_vision se trouve dans la cellule de l'agent ou l'une des 8 adjacentes ;
  - `visible_neighbors`, qui applique portée, cône, occlusion par la paroi puis
    occlusion mutuelle entre agents.

La grille est NON périodique en x, ce qui est cohérent avec un pont dont les
extrémités bornent la vision : un agent près de x = 0 ne voit pas à travers le
bord, même lorsque le mouvement, lui, est périodique (« semi-périodicité »).

Les indices renvoyés sont TRIÉS par ordre croissant : l'ordre de sommation des
forces est alors indépendant de l'implémentation de la recherche de voisins,
ce qui rend `visible_neighbors` et `visible_neighbors_bruteforce` bit-à-bit
équivalents (vérifié par `tests/test_neighbors.py`).

Aucune allocation dans la boucle par agent : les tampons `out` et `flag` sont
alloués une fois par pas et réutilisés.
"""

import numpy as np
from numba import njit

from .geometry import segment_leaves_channel
from .kernels import wrap_angle


# ──────────────────────────────────────────────────────────────────────────
#  Grille
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def grid_shape(p):
    """(nx, ny, cell_size) : la maille vaut la portée de vision."""
    cell = p.l_vision
    nx = int(p.length / cell)
    if nx < 1:
        nx = 1
    ny = int(p.height / cell)
    if ny < 1:
        ny = 1
    return nx, ny, cell


@njit(cache=True, inline="always")
def cell_index(x, y, nx, ny, cell):
    cx = int(x / cell)
    if cx < 0:
        cx = 0
    elif cx >= nx:
        cx = nx - 1
    cy = int(y / cell)
    if cy < 0:
        cy = 0
    elif cy >= ny:
        cy = ny - 1
    return cx, cy


@njit(cache=True)
def build_cell_list(px, py, n, nx, ny, cell):
    """Range les agents par cellule (tri par comptage).

    Renvoie (cell_start, order) : les agents de la cellule c sont
    `order[cell_start[c]:cell_start[c + 1]]`, avec c = cy * nx + cx.
    """
    n_cells = nx * ny
    counts = np.zeros(n_cells, dtype=np.int64)
    cell_of = np.empty(n, dtype=np.int64)

    for a in range(n):
        cx, cy = cell_index(px[a], py[a], nx, ny, cell)
        c = cy * nx + cx
        cell_of[a] = c
        counts[c] += 1

    cell_start = np.zeros(n_cells + 1, dtype=np.int64)
    for c in range(n_cells):
        cell_start[c + 1] = cell_start[c] + counts[c]

    order = np.empty(n, dtype=np.int64)
    cursor = cell_start[:n_cells].copy()
    for a in range(n):
        c = cell_of[a]
        order[cursor[c]] = a
        cursor[c] += 1

    return cell_start, order


@njit(cache=True, inline="always")
def _insertion_sort(buf, m):
    """Tri croissant en place de buf[:m] (m est petit : voisinage local)."""
    for a in range(1, m):
        key = buf[a]
        b = a - 1
        while b >= 0 and buf[b] > key:
            buf[b + 1] = buf[b]
            b -= 1
        buf[b + 1] = key


# ──────────────────────────────────────────────────────────────────────────
#  Occlusion mutuelle
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _filter_occluded(i, px, py, buf, m, radius, flag):
    """Retire de buf[:m] les agents masqués par un agent plus proche.

    Un agent k est masqué si un agent intercalé, strictement plus proche de i,
    couvre un angle `atan(radius / d)` autour de sa direction qui contient
    celle de k. Renvoie le nouveau nombre d'éléments (compactés en tête).
    """
    xi = px[i]
    yi = py[i]
    for a in range(m):
        flag[a] = True

    for a in range(m):
        ka = buf[a]
        dxa = px[ka] - xi
        dya = py[ka] - yi
        da = np.sqrt(dxa * dxa + dya * dya)
        ang_a = np.arctan2(dya, dxa)
        for b in range(m):
            kb = buf[b]
            if kb == ka:
                continue
            dxb = px[kb] - xi
            dyb = py[kb] - yi
            db = np.sqrt(dxb * dxb + dyb * dyb)
            if da < db:            # l'intercalé doit être plus proche
                continue
            ang_b = np.arctan2(dyb, dxb)
            if abs(wrap_angle(ang_a - ang_b)) < abs(np.arctan(radius / db)):
                flag[a] = False
                break

    k = 0
    for a in range(m):
        if flag[a]:
            buf[k] = buf[a]
            k += 1
    return k


# ──────────────────────────────────────────────────────────────────────────
#  Champ de vision
# ──────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def visible_neighbors(i, px, py, theta, p, n, cell_start, order,
                      nx, ny, cell, buf, flag):
    """Remplit `buf[:count]` des indices visibles par i (triés) et renvoie count."""
    xi = px[i]
    yi = py[i]
    ti = theta[i]
    half_cone = 0.5 * p.angle_vision
    reach = p.l_vision
    radius = p.radius

    cx, cy = cell_index(xi, yi, nx, ny, cell)

    m = 0
    for gx in range(cx - 1, cx + 2):
        if gx < 0 or gx >= nx:
            continue
        for gy in range(cy - 1, cy + 2):
            if gy < 0 or gy >= ny:
                continue
            c = gy * nx + gx
            for t in range(cell_start[c], cell_start[c + 1]):
                j = order[t]
                if j == i:
                    continue
                dx = px[j] - xi
                dy = py[j] - yi
                if dx * dx + dy * dy > reach * reach:
                    continue
                if abs(wrap_angle(np.arctan2(dy, dx) - ti)) >= half_cone:
                    continue
                # Occlusion par une paroi : le voisin n'est masqué que si sa
                # tête ET son arrière sont hors de vue.
                tj = theta[j]
                hx = px[j] + radius * np.cos(tj)
                hy = py[j] + radius * np.sin(tj)
                bx = px[j] - radius * np.cos(tj)
                by = py[j] - radius * np.sin(tj)
                if (segment_leaves_channel(xi, yi, hx, hy, p)
                        and segment_leaves_channel(xi, yi, bx, by, p)):
                    continue
                buf[m] = j
                m += 1

    _insertion_sort(buf, m)
    return _filter_occluded(i, px, py, buf, m, radius, flag)


@njit(cache=True)
def visible_neighbors_bruteforce(i, px, py, theta, p, n, buf, flag):
    """Référence O(N^2), sans grille. Sert d'oracle à `tests/test_neighbors.py`
    pour garantir que l'accélération spatiale ne change pas la physique."""
    xi = px[i]
    yi = py[i]
    ti = theta[i]
    half_cone = 0.5 * p.angle_vision
    reach = p.l_vision
    radius = p.radius

    m = 0
    for j in range(n):
        if j == i:
            continue
        dx = px[j] - xi
        dy = py[j] - yi
        if dx * dx + dy * dy > reach * reach:
            continue
        if abs(wrap_angle(np.arctan2(dy, dx) - ti)) >= half_cone:
            continue
        tj = theta[j]
        hx = px[j] + radius * np.cos(tj)
        hy = py[j] + radius * np.sin(tj)
        bx = px[j] - radius * np.cos(tj)
        by = py[j] - radius * np.sin(tj)
        if (segment_leaves_channel(xi, yi, hx, hy, p)
                and segment_leaves_channel(xi, yi, bx, by, p)):
            continue
        buf[m] = j
        m += 1

    return _filter_occluded(i, px, py, buf, m, radius, flag)
