"""
engine.py
=========
Orchestration : initialisation, boucle de simulation, résultat.

Reproductibilité — UN SEUL générateur
-------------------------------------
Tout l'aléa du run vient de `np.random.default_rng(sim.seed)` :
  - vitesses de croisière individuelles, positions et caps initiaux ;
  - `tie_sign`, le signe qui départage les directions d'évitement dégénérées ;
  - le bruit angulaire et les tirages de réinjection, PRÉ-TIRÉS par bloc et
    passés en argument au noyau compilé.

Le code de référence tirait des nombres aléatoires à l'intérieur des fonctions
`@njit`. Or Numba possède en mode nopython un état aléatoire PROPRE, distinct
de celui de NumPy : il fallait alors seeder les deux générateurs, sous peine de
runs non reproductibles. On a préféré supprimer le problème plutôt que le
gérer : aucun appel à `np.random` ne subsiste dans le code compilé
(`tests/test_reproducibility.py` le vérifie automatiquement). `seed_numba` est
conservé et appelé comme filet de sécurité, au cas où un tirage serait
réintroduit dans le noyau.

Les tirages sont effectués à chaque sous-pas, que le bruit soit actif ou non :
la séquence aléatoire ne dépend donc pas des options du modèle, ce qui rend les
ablations comparables à graine égale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .params import ModelParams, SimParams, build_kernel_params
from .geometry import wall_y
from .step import step


@njit(cache=True)
def seed_numba(seed):
    """Graine du générateur INTERNE de Numba.

    Filet de sécurité : le noyau compilé n'utilise aujourd'hui aucun tirage
    aléatoire, mais si l'un venait à être réintroduit, cet appel garantit que
    la graine du run le contrôle aussi.
    """
    np.random.seed(seed)


# ──────────────────────────────────────────────────────────────────────────
#  Résultat
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """Trajectoires enregistrées.

    Les tableaux ont la forme (n_frames + 1, n_agents) : la ligne 0 est l'état
    INITIAL, la ligne k l'état à t = k / record_fps.
    """
    x: np.ndarray            # mm
    y: np.ndarray            # mm
    u: np.ndarray            # mm/s (vitesse scalaire)
    theta: np.ndarray        # rad (cap)
    group: np.ndarray        # +1 (vers +x, la source) ou -1 (vers -x, le nid)
    xi_cruise: np.ndarray    # mm/s, vitesse de croisière propre de chaque agent
    times: np.ndarray        # s
    model: ModelParams
    sim: SimParams

    @property
    def n_agents(self) -> int:
        return self.x.shape[1]

    @property
    def n_frames(self) -> int:
        return self.x.shape[0]

    def velocity(self):
        """(vx, vy) : composantes cartésiennes de la vitesse, en mm/s."""
        return self.u * np.cos(self.theta), self.u * np.sin(self.theta)


# ──────────────────────────────────────────────────────────────────────────
#  Initialisation
# ──────────────────────────────────────────────────────────────────────────

def make_groups(num_agents: int, fraction_plus: float) -> np.ndarray:
    """Vecteur de groupes : `round(fraction_plus * N)` agents en +1, le reste en -1."""
    n_plus = int(round(fraction_plus * num_agents))
    group = -np.ones(num_agents, dtype=np.float64)
    group[:n_plus] = 1.0
    return group


def _slot_spacing(model: ModelParams, p, num_agents: int = 0) -> float:
    """Pas de la grille initiale : un diamètre, resserré si nécessaire.

    Au-delà d'une certaine densité il n'existe plus de placement initial sans
    recouvrement (un pont de 100 x 10 n'offre que 188 sites à un pas de 2.1).
    Plutôt que d'échouer, on resserre la grille : les agents démarrent alors en
    recouvrement, ce qui est la situation physique à ces densités.
    """
    spacing = 2.0 * model.radius * 1.05
    if num_agents <= 0:
        return spacing
    while spacing > 0.2:
        n_cols = max(int(model.length / spacing), 1)
        y_bot, y_top = wall_y(0.5 * model.length, p)
        span = (y_top - model.radius) - (y_bot + model.radius)
        n_rows = int(span / spacing) + 1 if span >= 0 else 1
        if n_cols * n_rows >= num_agents:
            break
        spacing *= 0.9
    return spacing


def _candidate_slots(model: ModelParams, p, num_agents: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Grille de positions initiales sans recouvrement (x croissant).

    La grille est CENTRÉE dans les deux directions. Un simple
    `arange(y_bot + R, y_top - R, pas)` laisserait une marge inégale en haut et
    en bas — sur un pont de 10 mm avec R = 1 et un pas de 2.1, les sites
    tombent en y = 1, 3.1, 5.2, 7.3, de moyenne 4.15 au lieu de 5. Ce biais
    d'un demi-pas n'est pas anodin : il oriente la brisure de symétrie
    haut/bas de la condensation contre une paroi, qui se fait alors
    systématiquement vers le bas. En x, les sites sont répartis exactement sur
    le tore, sans couture entre le dernier et le premier.
    """
    step_len = _slot_spacing(model, p, num_agents)

    n_cols = max(int(model.length / step_len), 1)
    col_x = (np.arange(n_cols) + 0.5) * model.length / n_cols

    xs, ys = [], []
    for x in col_x:
        y_bot, y_top = wall_y(x, p)
        span = (y_top - model.radius) - (y_bot + model.radius)
        if span < 0:
            continue
        n_rows = int(span / step_len) + 1
        offset = y_bot + model.radius + 0.5 * (span - (n_rows - 1) * step_len)
        for k in range(n_rows):
            xs.append(x)
            ys.append(offset + k * step_len)
    return np.asarray(xs), np.asarray(ys)


def initial_state(model: ModelParams, sim: SimParams, group: np.ndarray,
                  xi_cruise: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """État initial (4, N) = [x ; y ; u ; theta].

    `init_mode = 'uniform'`   : positions tirées sur tout le pont ;
    `init_mode = 'separated'` : groupe +1 à gauche, groupe -1 à droite (chacun
    part de son côté, comme au début d'une expérience).
    """
    p = build_kernel_params(model)
    slots_x, slots_y = _candidate_slots(model, p, sim.num_agents)
    n = sim.num_agents

    if sim.init_mode == "uniform":
        if len(slots_x) < n:
            raise ValueError(
                f"canal trop petit : {len(slots_x)} positions sans recouvrement "
                f"pour {n} agents (pont {model.length} x {model.height}, R = {model.radius}).")
        chosen = rng.choice(len(slots_x), size=n, replace=False)
        pos_x = slots_x[chosen]
        pos_y = slots_y[chosen]
    else:                                   # separated
        left = np.flatnonzero(slots_x < 0.5 * model.length)
        right = np.flatnonzero(slots_x >= 0.5 * model.length)
        n_plus = int(np.sum(group > 0))
        n_minus = n - n_plus
        if len(left) < n_plus or len(right) < n_minus:
            raise ValueError("pas assez de positions pour le mode 'separated'.")
        take_left = left[rng.choice(len(left), size=n_plus, replace=False)]
        take_right = right[rng.choice(len(right), size=n_minus, replace=False)]
        pos_x = np.empty(n)
        pos_y = np.empty(n)
        it_l = it_r = 0
        for i in range(n):
            if group[i] > 0:
                k = take_left[it_l]; it_l += 1
            else:
                k = take_right[it_r]; it_r += 1
            pos_x[i] = slots_x[k]
            pos_y[i] = slots_y[k]

    # Gigue bornée à 0.4 R : les agents restent dans le canal sans rognage.
    pos_x = pos_x + (rng.random(n) - 0.5) * 0.8 * model.radius
    pos_y = pos_y + (rng.random(n) - 0.5) * 0.8 * model.radius
    pos_x = np.mod(pos_x, model.length)

    # Cap initial : dans un demi-secteur autour de la direction objectif.
    theta0 = (rng.random(n) - 0.5) * np.pi
    theta0[group < 0] += np.pi

    state = np.empty((4, n), dtype=np.float64)
    state[0] = pos_x
    state[1] = pos_y
    state[2] = xi_cruise
    state[3] = theta0
    return state


# ──────────────────────────────────────────────────────────────────────────
#  Boucle
# ──────────────────────────────────────────────────────────────────────────

def run(model: ModelParams, sim: SimParams, progress: bool = True) -> RunResult:
    """Lance une simulation et renvoie les trajectoires enregistrées."""
    rng = np.random.default_rng(sim.seed)
    seed_numba(sim.seed)

    p = build_kernel_params(model)
    n = sim.num_agents
    n_frames = sim.n_frames
    n_sub = sim.substeps_per_frame

    xi_cruise = model.xi_cruise * (1.0 + model.xi_spread * (2.0 * rng.random(n) - 1.0))
    group = make_groups(n, sim.fraction_plus)
    tie_sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    state = initial_state(model, sim, group, xi_cruise, rng)
    new_state = np.empty_like(state)
    buf = np.empty(n, dtype=np.int64)
    flag = np.empty(n, dtype=np.bool_)

    rec_x = np.empty((n_frames + 1, n), dtype=np.float32)
    rec_y = np.empty((n_frames + 1, n), dtype=np.float32)
    rec_u = np.empty((n_frames + 1, n), dtype=np.float32)
    rec_t = np.empty((n_frames + 1, n), dtype=np.float32)

    def record(k, s):
        rec_x[k] = s[0]
        rec_y[k] = s[1]
        rec_u[k] = s[2]
        rec_t[k] = s[3]

    record(0, state)

    frames = range(n_frames)
    if progress:
        try:
            from tqdm import tqdm
            frames = tqdm(frames, desc="frames", unit="f")
        except ImportError:
            pass

    for k in frames:
        # Tirages du bloc : une seule sollicitation du générateur par image.
        gauss_block = rng.standard_normal((n_sub, n))
        uniform_block = rng.random((n_sub, n))
        for s in range(n_sub):
            step(state, new_state, xi_cruise, group, tie_sign,
                 gauss_block[s], uniform_block[s], p, sim.dt, buf, flag)
            state, new_state = new_state, state
        record(k + 1, state)

    times = np.arange(n_frames + 1, dtype=np.float64) / sim.record_fps
    return RunResult(x=rec_x, y=rec_y, u=rec_u, theta=rec_t,
                     group=group, xi_cruise=xi_cruise, times=times,
                     model=model, sim=sim)
