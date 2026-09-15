"""
io.py
=====
Sauvegarde et rechargement d'un run.

Un run sauvegardé tient dans un dossier :

    <dossier>/config.json        TOUS les paramètres + la graine + la version
    <dossier>/trajectories.npz   les trajectoires (float32, compressé)

`config.json` suffit à REJOUER le run à l'identique (`scripts/replay.py`) :
c'est la contrepartie de l'unicité du générateur aléatoire. Les trajectoires
sont stockées en NPZ plutôt qu'en CSV : le CSV du code de référence pesait une
vingtaine de mégaoctets pour un seul run, le NPZ float32 est ~15 fois plus
compact et se relit sans analyse de texte. `to_dataframe` reste disponible pour
qui veut un format tabulaire.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .engine import RunResult
from .params import ModelParams, SimParams, load_config, save_config

CONFIG_NAME = "config.json"
TRAJECTORIES_NAME = "trajectories.npz"


def save_run(result: RunResult, out_dir: str | Path) -> Path:
    """Écrit config.json et trajectories.npz dans `out_dir`. Renvoie le dossier."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(result.model, result.sim, out_dir / CONFIG_NAME)
    np.savez_compressed(
        out_dir / TRAJECTORIES_NAME,
        x=result.x, y=result.y, u=result.u, theta=result.theta,
        group=result.group, xi_cruise=result.xi_cruise, times=result.times,
    )
    return out_dir


def load_run(out_dir: str | Path) -> RunResult:
    """Recharge un run sauvegardé par `save_run`."""
    out_dir = Path(out_dir)
    model, sim = load_config(out_dir / CONFIG_NAME)
    data = np.load(out_dir / TRAJECTORIES_NAME)
    return RunResult(
        x=data["x"], y=data["y"], u=data["u"], theta=data["theta"],
        group=data["group"], xi_cruise=data["xi_cruise"], times=data["times"],
        model=model, sim=sim,
    )


def replay(config_path: str | Path, progress: bool = True) -> RunResult:
    """Rejoue un run depuis son seul fichier de configuration."""
    from .engine import run
    model, sim = load_config(config_path)
    return run(model, sim, progress=progress)


def to_dataframe(result: RunResult):
    """Vue tabulaire longue (une ligne par agent et par image).

    Nécessite pandas. Colonnes : frame, time, agent_id, group, x, y, u, theta.
    """
    import pandas as pd

    n_frames, n = result.x.shape
    frame = np.repeat(np.arange(n_frames), n)
    time = np.repeat(result.times, n)
    agent = np.tile(np.arange(n), n_frames)
    return pd.DataFrame({
        "frame": frame,
        "time": time,
        "agent_id": agent,
        "group": np.tile(result.group, n_frames),
        "x": result.x.ravel(),
        "y": result.y.ravel(),
        "u": result.u.ravel(),
        "theta": result.theta.ravel(),
    })
