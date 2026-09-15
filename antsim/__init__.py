"""
antsim — modèle d'agents pour le trafic bidirectionnel de fourmis d'Argentine
============================================================================

Simule l'expérience de Poissonnier, Motsch, Gautrais, Buhl & Dussutour (2019,
eLife) : deux flux opposés de *Linepithema humile* sur un pont étroit.

Unités internes : 1 mm, 1 s. Voir `params.py` pour les conversions vers les
unités de l'article, et le README pour la correspondance entre paramètres et
équations de `rapport_LLM.tex`.

Usage minimal
-------------
    from antsim import ModelParams, SimParams, run, save_run

    result = run(ModelParams(), SimParams(num_agents=60, duration=30.0, seed=0))
    save_run(result, "out/demo")
"""

from .params import (
    CODE_VERSION,
    ANGULAR_MODES,
    GEOMETRY_CODES,
    WALL_MODES,
    X_MODES,
    KernelParams,
    ModelParams,
    SimParams,
    build_kernel_params,
    config_dict,
    load_config,
    make_model,
    save_config,
)
from .engine import RunResult, make_groups, run, seed_numba
from .io import load_run, replay, save_run, to_dataframe

__all__ = [
    "CODE_VERSION",
    "ANGULAR_MODES", "GEOMETRY_CODES", "WALL_MODES", "X_MODES",
    "ModelParams", "SimParams", "KernelParams",
    "build_kernel_params", "config_dict", "load_config", "save_config",
    "make_model",
    "RunResult", "run", "make_groups", "seed_numba",
    "save_run", "load_run", "replay", "to_dataframe",
]
