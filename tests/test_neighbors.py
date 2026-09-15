"""La cell list doit renvoyer EXACTEMENT le même voisinage que la référence O(N^2).

C'est la garantie que l'accélération spatiale ne modifie pas la physique : même
ensemble de voisins, même ordre, donc même ordre de sommation des forces et
mêmes trajectoires au bit près.
"""

import numpy as np

import _bootstrap  # noqa: F401

from antsim import ModelParams, build_kernel_params
from antsim.neighbors import (build_cell_list, grid_shape, visible_neighbors,
                              visible_neighbors_bruteforce)


def _random_configuration(n, model, seed):
    rng = np.random.default_rng(seed)
    px = rng.random(n) * model.length
    py = rng.random(n) * model.height
    theta = rng.random(n) * 2 * np.pi - np.pi
    return px, py, theta


def _compare(model, n, seed):
    p = build_kernel_params(model)
    px, py, theta = _random_configuration(n, model, seed)
    ncx, ncy, cell = grid_shape(p)
    cell_start, order = build_cell_list(px, py, n, ncx, ncy, cell)

    buf_a = np.empty(n, dtype=np.int64)
    buf_b = np.empty(n, dtype=np.int64)
    flag = np.empty(n, dtype=np.bool_)

    for i in range(n):
        m_grid = visible_neighbors(i, px, py, theta, p, n, cell_start, order,
                                   ncx, ncy, cell, buf_a, flag)
        m_ref = visible_neighbors_bruteforce(i, px, py, theta, p, n, buf_b, flag)
        assert m_grid == m_ref, f"agent {i} : {m_grid} voisins vs {m_ref}"
        assert np.array_equal(buf_a[:m_grid], buf_b[:m_ref]), f"agent {i} : voisins différents"


def test_cell_list_matches_bruteforce_default_bridge():
    _compare(ModelParams(), n=120, seed=0)


def test_cell_list_matches_bruteforce_narrow_bridge():
    # pont de 5 mm : moins d'une cellule de haut, le cas limite de la grille
    _compare(ModelParams(height=5.0), n=80, seed=1)


def test_cell_list_matches_bruteforce_wide_bridge():
    _compare(ModelParams(height=20.0), n=150, seed=2)


def test_cell_list_matches_bruteforce_short_bridge():
    # pont plus court que quelques cellules : bornes de grille sollicitées
    _compare(ModelParams(length=20.0, height=10.0), n=60, seed=3)


def test_neighbors_are_sorted_and_exclude_self():
    model = ModelParams()
    p = build_kernel_params(model)
    n = 100
    px, py, theta = _random_configuration(n, model, seed=7)
    ncx, ncy, cell = grid_shape(p)
    cell_start, order = build_cell_list(px, py, n, ncx, ncy, cell)
    buf = np.empty(n, dtype=np.int64)
    flag = np.empty(n, dtype=np.bool_)
    for i in range(n):
        m = visible_neighbors(i, px, py, theta, p, n, cell_start, order,
                              ncx, ncy, cell, buf, flag)
        found = buf[:m]
        assert i not in found
        assert np.all(np.diff(found) > 0), "voisins non triés strictement croissants"
