"""Reproductibilité : une graine détermine entièrement un run.

Trois garanties vérifiées ici :
  1. même graine  -> trajectoires identiques au bit près ;
  2. graine différente -> trajectoires différentes (le run dépend vraiment de l'aléa) ;
  3. un run se rejoue depuis son SEUL fichier de configuration.

Le quatrième test est structurel : il vérifie qu'aucun tirage aléatoire n'a été
réintroduit dans le code compilé. Numba possède en nopython un état aléatoire
distinct de celui de NumPy ; un `np.random` dans le noyau rendrait la graine du
run insuffisante. `engine.seed_numba` est la seule exception admise (c'est
précisément la fonction qui seeder ce générateur, par sécurité).
"""

import re
import tempfile
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

import antsim
from antsim import ModelParams, SimParams, load_run, replay, run, save_run

SHORT = dict(num_agents=40, duration=1.0)


def test_same_seed_gives_identical_trajectories():
    model = ModelParams()
    sim = SimParams(seed=12, **SHORT)
    a = run(model, sim, progress=False)
    b = run(model, sim, progress=False)
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.y, b.y)
    assert np.array_equal(a.u, b.u)
    assert np.array_equal(a.theta, b.theta)
    assert np.array_equal(a.xi_cruise, b.xi_cruise)


def test_different_seed_gives_different_trajectories():
    model = ModelParams()
    a = run(model, SimParams(seed=1, **SHORT), progress=False)
    b = run(model, SimParams(seed=2, **SHORT), progress=False)
    assert not np.array_equal(a.x, b.x)


def test_run_is_replayable_from_its_config():
    model = ModelParams()
    sim = SimParams(seed=5, **SHORT)
    original = run(model, sim, progress=False)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "run"
        save_run(original, out)

        reloaded = load_run(out)
        assert np.array_equal(original.x, reloaded.x)

        replayed = replay(out / "config.json", progress=False)
        assert np.array_equal(original.x, replayed.x)
        assert np.array_equal(original.theta, replayed.theta)


def test_compiled_kernel_draws_no_random_number():
    package = Path(antsim.__file__).parent
    offenders = []
    for path in sorted(package.glob("*.py")):
        if path.name == "engine.py":
            continue                      # contient seed_numba, la seule exception
        text = path.read_text()
        if not re.search(r"@njit", text):
            continue
        if re.search(r"np\.random", text):
            offenders.append(path.name)
    assert not offenders, (
        "tirage aléatoire dans du code compilé : " + ", ".join(offenders)
        + " — le générateur interne de Numba est distinct de celui de NumPy, "
          "la graine du run ne le contrôlerait pas.")


def test_seed_only_source_of_randomness_in_engine():
    # Dans engine.py, le seul usage de np.random hors du Generator est seed_numba.
    text = (Path(antsim.__file__).parent / "engine.py").read_text()
    uses = re.findall(r"np\.random\.\w+", text)
    assert set(uses) <= {"np.random.seed", "np.random.default_rng", "np.random.Generator"}, uses
