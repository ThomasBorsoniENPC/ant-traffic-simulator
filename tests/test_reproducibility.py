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


def test_numba_cache_is_isolated_per_field_layout():
    """Le cache disque de Numba doit être séparé PAR DISPOSITION DE CHAMPS.

    Numba identifie un type NamedTuple par le nom de sa classe et les types de
    ses membres, jamais par les noms de champs. Deux versions successives de
    `KERNEL_FIELDS` de même longueur et de mêmes types (ici : des int64 puis
    des float64) sont donc indistinguables pour lui, et le code machine
    compilé pour l'une lui est resservi pour l'autre — chaque paramètre étant
    alors lu à la MAUVAISE POSITION. La régression observée (freinage et
    braquage simultanément faux, sans qu'aucun paramètre ni aucune ligne de
    code ne soit en cause) est invisible aux autres tests, qui recompilent
    tous dans le même processus.

    La parade est un répertoire de cache par empreinte. On ne renomme PAS la
    classe : Numba pickle le type dans son index, et un nom qui change casse la
    relecture des index déjà écrits (AttributeError au lieu d'une simple
    recompilation).
    """
    import hashlib
    import os

    from antsim import params as P

    expected = hashlib.blake2s(
        "|".join(P.KERNEL_FIELDS).encode(), digest_size=4).hexdigest()
    assert P.KERNEL_FINGERPRINT == expected
    assert P.KernelParams.__name__ == "KernelParams", (
        "le nom de la classe doit rester stable pour que Numba puisse "
        "dépickler ses index de cache déjà écrits.")
    assert expected in os.environ.get("NUMBA_CACHE_DIR", ""), (
        "le répertoire de cache Numba ne porte pas l'empreinte des champs : "
        "une disposition périmée pourrait être resservie.")

    # Une disposition DIFFÉRENTE (mêmes champs, deux permutés : même longueur,
    # mêmes types) doit donner une empreinte différente, donc un autre cache.
    permuted = (P.KERNEL_FIELDS[1], P.KERNEL_FIELDS[0]) + P.KERNEL_FIELDS[2:]
    other = hashlib.blake2s("|".join(permuted).encode(), digest_size=4).hexdigest()
    assert other != expected


def test_no_kernel_field_is_declared_but_unused():
    """Tout champ de `ModelParams` transmis au noyau doit y être lu, et tout
    champ non transmis ne doit pas se faire passer pour un paramètre actif.

    `n_brake` avait survécu ainsi : documenté comme exposant du profil radial,
    absent de `KERNEL_FIELDS`, donc sans le moindre effet.
    """
    from antsim import ModelParams
    from antsim import params as P

    package = Path(antsim.__file__).parent
    sources = "\n".join(path.read_text() for path in sorted(package.glob("*.py"))
                        if path.name != "params.py")

    orphans = []
    for name in ModelParams().__dict__:
        if name in P.KERNEL_FIELDS:
            continue
        if re.search(rf"\bp\.{name}\b", sources):
            continue                      # lu autrement (hors noyau compilé)
        orphans.append(name)

    known = {"geom_type", "angular_mode", "wall_mode", "x_mode", "wall_law",
             "brake_law", "length", "height", "amplitude", "wavelength"}
    unexpected = [n for n in orphans if n not in known
                  and not re.search(rf"\b{n}\b", sources)]
    assert not unexpected, (
        "champs déclarés mais jamais lus (paramètres fantômes) : "
        + ", ".join(unexpected))
