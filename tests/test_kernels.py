"""Propriétés élémentaires des profils de force : supports, bornes, monotonie.

Ces tests protègent la forme des noyaux, qui est le point de contact entre le
code et les équations de `rapport_LLM.tex` : un profil dont le support se
décale, ou qui cesse de s'annuler à la portée, change le modèle sans que rien
ne casse visiblement.
"""

import numpy as np

import _bootstrap  # noqa: F401

from antsim.kernels import (cone_factor, cone_factor_pow, profile_ramp,
                            profile_singular, sign_nonzero, wrap_angle)


def test_cone_factor_bounds():
    eps = 0.7
    assert cone_factor(1.0, eps) == 1.0                 # droit devant
    assert cone_factor(1.0 - eps, eps) == 0.0           # bord du cône
    assert cone_factor(1.0 - 2 * eps, eps) == 0.0       # au-delà : tronqué à 0
    # croissance avec le cosinus à l'intérieur du cône
    values = [cone_factor(c, eps) for c in np.linspace(1 - eps, 1.0, 20)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_cone_factor_wide_eps_is_omnidirectional():
    # eps_repulse = 2 s'applique à |sin| dans [0, 1] : le facteur vaut alors
    # (|sin| + 1) / 2 >= 0.5, donc la répulsion latérale est omnidirectionnelle
    # — ce n'est pas un réglage anodin, c'est un choix de modèle.
    for abs_sin in np.linspace(0.0, 1.0, 11):
        assert cone_factor(abs_sin, 2.0) >= 0.5
    # le support s'étend jusqu'à cos = 1 - eps = -1, où le facteur s'annule
    assert cone_factor(-1.0, 2.0) == 0.0
    assert cone_factor(-0.99, 2.0) > 0.0


def test_cone_factor_pow_matches_power():
    eps, alpha = 0.7, 4.0
    for c in (0.4, 0.7, 1.0):
        assert np.isclose(cone_factor_pow(c, eps, alpha), cone_factor(c, eps) ** alpha)


def test_profile_singular_vanishes_at_range():
    ell, reg = 3.0, 1e-6
    assert profile_singular(ell, ell, reg) == 0.0
    # décroissance stricte sur (0, ell)
    values = [profile_singular(r, ell, reg) for r in np.linspace(0.2, ell, 30)]
    assert all(b < a for a, b in zip(values, values[1:]))
    # la régularisation évite la division par zéro
    assert np.isfinite(profile_singular(0.0, ell, reg))


def test_profile_ramp_endpoints():
    a, b = 2.3, 5.0
    assert np.isclose(profile_ramp(a, a, b), 0.0)
    assert np.isclose(profile_ramp(b, a, b), 1.0)
    assert np.isclose(profile_ramp(0.5 * (a + b), a, b), 0.5)


def test_wrap_angle_range():
    for a in np.linspace(-10.0, 10.0, 101):
        w = wrap_angle(a)
        assert -np.pi < w <= np.pi + 1e-12
        assert np.isclose(np.cos(w), np.cos(a)) and np.isclose(np.sin(w), np.sin(a))


def test_sign_nonzero():
    assert sign_nonzero(3.0) == 1.0
    assert sign_nonzero(-3.0) == -1.0
    assert sign_nonzero(0.0) == 1.0
