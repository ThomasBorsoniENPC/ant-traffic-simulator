"""Invariants de la dynamique : positivité, confinement, conservation.

Ce sont les propriétés qu'aucune modification du modèle ne doit casser
silencieusement. Elles sont peu coûteuses à vérifier et attrapent l'essentiel
des régressions d'intégration (dépassement de la relaxation, fuite au travers
d'une paroi, vitesse négative, NaN).
"""

import numpy as np

import _bootstrap  # noqa: F401

from antsim import ModelParams, SimParams, build_kernel_params, run
from antsim.braking import brake_from_neighbor, relax_speed
from antsim.kernels import wrap_angle
from antsim.step import relax_heading

TOL_WALL = 1e-6            # marge de repositionnement utilisée par boundaries


def _short_run(**model_kwargs):
    model = ModelParams(**model_kwargs)
    sim = SimParams(num_agents=60, duration=2.0, seed=0)
    return run(model, sim, progress=False), model


def test_speed_stays_non_negative():
    result, _ = _short_run()
    assert result.u.min() >= 0.0, f"vitesse négative : {result.u.min()}"


def test_no_nan():
    result, _ = _short_run()
    for name, arr in (("x", result.x), ("y", result.y),
                      ("u", result.u), ("theta", result.theta)):
        assert np.isfinite(arr).all(), f"valeurs non finies dans {name}"


def test_agents_stay_inside_the_bridge():
    result, model = _short_run()
    assert result.x.min() >= 0.0 and result.x.max() <= model.length
    assert result.y.min() >= -TOL_WALL
    assert result.y.max() <= model.height + TOL_WALL


def test_agent_count_and_groups_are_conserved():
    result, _ = _short_run()
    assert result.x.shape[1] == result.sim.num_agents
    assert set(np.unique(result.group)) <= {-1.0, 1.0}
    assert np.sum(result.group > 0) == result.sim.num_agents // 2


def test_inflow_mode_also_confines_agents():
    model = ModelParams(x_mode="inflow")
    result = run(model, SimParams(num_agents=40, duration=3.0, seed=1), progress=False)
    assert result.x.min() >= 0.0 and result.x.max() <= model.length
    assert result.y.min() >= -TOL_WALL and result.y.max() <= model.height + TOL_WALL
    assert result.u.min() >= 0.0


def test_isolated_agent_keeps_its_cruise_speed():
    """Sans voisin, sans bruit et sans paroi attractive, u reste à xi."""
    model = ModelParams(sigma_theta=0.0, use_wall_attraction=False)
    result = run(model, SimParams(num_agents=1, duration=2.0, seed=0), progress=False)
    assert np.allclose(result.u, result.xi_cruise[0], rtol=1e-6)


def test_crowding_slows_the_flow_down():
    """Contrôle de bon sens : à forte densité la vitesse moyenne chute."""
    sparse = run(ModelParams(), SimParams(num_agents=10, duration=3.0, seed=0),
                 progress=False)
    dense = run(ModelParams(), SimParams(num_agents=150, duration=3.0, seed=0),
                progress=False)
    half = sparse.n_frames // 2
    assert dense.u[half:].mean() < 0.9 * sparse.u[half:].mean()


def test_braking_rate_is_non_negative():
    p = build_kernel_params(ModelParams())
    rng = np.random.default_rng(0)
    for _ in range(500):
        xi, yi = rng.random(2) * 10.0
        xj, yj = rng.random(2) * 10.0
        ti = rng.random() * 2 * np.pi - np.pi
        assert brake_from_neighbor(xi, yi, ti, xj, yj, p) >= 0.0


def test_speed_relaxation_converges_to_the_monod_equilibrium():
    p = build_kernel_params(ModelParams())
    xi = 17.0
    for nu_brake in (0.0, 5.0, 50.0, 1e4):
        u = 0.0
        for _ in range(2000):
            u = relax_speed(u, nu_brake, xi, p, 0.01)
        expected = p.nu_cruise * xi / (p.nu_cruise + nu_brake)
        assert np.isclose(u, expected, rtol=1e-8), (nu_brake, u, expected)
        assert u <= xi + 1e-12               # jamais au-dessus de la vitesse libre


def test_heading_relaxation_never_overshoots():
    p = build_kernel_params(ModelParams(angular_mode="fixed", lambda_theta=10.0))
    rng = np.random.default_rng(1)
    for _ in range(500):
        theta = rng.random() * 2 * np.pi - np.pi
        omega = rng.random() * 2 * np.pi - np.pi
        before = abs(wrap_angle(theta - omega))
        after = abs(wrap_angle(relax_heading(theta, omega, p.lambda_theta, p, 0.05) - omega))
        assert after <= before + 1e-12, (theta, omega, before, after)


def test_angular_modes_all_run():
    for mode in ("field_norm", "fixed", "saturating"):
        result = run(ModelParams(angular_mode=mode),
                     SimParams(num_agents=30, duration=1.0, seed=0), progress=False)
        assert np.isfinite(result.theta).all(), mode


def test_wall_modes_run():
    for wall_mode in ("specular", "slide"):
        result = run(ModelParams(wall_mode=wall_mode),
                     SimParams(num_agents=30, duration=1.0, seed=0), progress=False)
        assert result.y.min() >= -TOL_WALL
