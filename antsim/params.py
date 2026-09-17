"""
params.py
=========
Point unique de paramétrage : `ModelParams` (physique des agents) et
`SimParams` (boucle, enregistrement, graine), tous deux sérialisables en JSON,
plus la conversion vers le `NamedTuple` passé à la couche compilée.

Conventions d'unités (internes)
-------------------------------
    1 unité de longueur = 1 mm      1 unité de temps = 1 s
    fourmi de rayon R = 1 (soit 2 mm de diamètre comportemental)

Conversions vers les unités de Poissonnier et al. 2019 (eLife) :
    k [fourmis/cm²] = rho [fourmis/mm²] x 100
    q [fourmis/cm/s] = 10 x rho x U

Pourquoi un NamedTuple pour la couche compilée
----------------------------------------------
Les fonctions `@njit` ne doivent JAMAIS lire un paramètre comme variable
globale : Numba le capturerait à la compilation, ce qui interdit tout balayage
de paramètre sans recompilation. Tout circule donc via `KernelParams`, construit
par `build_kernel_params()` et passé en argument. Les choix discrets
(géométrie, mode angulaire, condition au bord) deviennent des ENTIERS branchés
à l'intérieur des fonctions — surtout pas des pointeurs de fonction globaux.

Équations et noyaux : voir
`Wrap up bidirectional ant trafic flow/Rapport existant LLM/rapport_LLM.tex`
(§ « Modèle microscopique sans règles de priorité » et § « Noyaux
d'interaction sans priorité »). Le README dresse la correspondance
paramètre <-> équation.
"""

from __future__ import annotations

import hashlib
import json
from collections import namedtuple
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

#: Version du code sérialisée avec chaque run (traçabilité des rejouages).
CODE_VERSION = "1.0.0-phase1"

# ──────────────────────────────────────────────────────────────────────────
#  Choix discrets -> codes entiers (branchés dans les fonctions compilées)
# ──────────────────────────────────────────────────────────────────────────

#: Géométries du canal. Phase 1 : rectangle seul. Pour en ajouter une
#: (goulot, bosse, ...), voir la marche à suivre en tête de `geometry.py`.
GEOMETRY_CODES = {"rectangle": 0}

#: Gestion du taux de rotation, cf. `step.py` et le README (§ force angulaire).
#:   field_norm : lambda = |F|, avec plafond |dtheta| <= lambda_max * dt.
#:                C'est l'équation (heading) du rapport de passation, donc le
#:                DÉFAUT : le simulateur reproduit le modèle de référence.
#:   fixed      : lambda = lambda_theta (taux de virage constant)
#:   saturating : lambda = lambda_max * |F| / (|F| + force_ref)
ANGULAR_MODES = {"field_norm": 0, "fixed": 1, "saturating": 2}

#: Traitement du contact avec une paroi latérale (les deux annulent le flux
#: normal ; la réflexion spéculaire est la configuration retenue par les tests
#: antérieurs du projet, le glissement est plus proche du thigmotactisme).
WALL_MODES = {"specular": 0, "slide": 1}

#: Loi de thigmotactisme (terme de paroi), cf. `forces.wall_force`.
#:   tangential       : loi du rapport de passation — attraction pondérée par la
#:                      composante TANGENTIELLE du cap, donc maximale quand
#:                      l'agent longe déjà la paroi.
#:   outward_damping  : ne retient que l'agent qui S'ÉLOIGNE ; nulle s'il longe
#:                      la paroi ou s'en approche.
#:   standoff         : rappel vers une distance de consigne l_wall_standoff,
#:                      RÉPULSIF en deçà et attractif au-delà, amorti sur la
#:                      vitesse normale. Seule des trois à admettre un suivi de
#:                      bord STABLE plutôt qu'un état absorbant.
WALL_LAWS = {"tangential": 0, "outward_damping": 1, "standoff": 2}

#: Extrémités du pont :
#:   torus        : x périodique, et l'ordonnée est RETIRÉE au hasard à la
#:                  ré-entrée — l'agent qui refait un tour revient à une
#:                  position transverse quelconque, ce qui remixe le flot et
#:                  empêche une structure transverse de se figer sur un tore
#:                  de longueur finie. C'est le comportement par défaut.
#:   torus_keep_y : x périodique, ordonnée CONSERVÉE. Trajectoire continue,
#:                  mais les corrélations transverses persistent indéfiniment.
#:   inflow       : l'agent sortant est réinjecté à l'entrée de son groupe,
#:                  vitesse et cap remis à l'état d'une fourmi entrante.
X_MODES = {"torus": 0, "inflow": 1, "torus_keep_y": 2}

#: Placement initial des agents.
INIT_MODES = ("uniform", "separated")


# ──────────────────────────────────────────────────────────────────────────
#  Paramètres du modèle
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class ModelParams:
    """Physique des agents. Les valeurs par défaut sont les valeurs CALIBRÉES
    du projet (celles qui donnent une pente de phase 1 ~17 mm/s, une transition
    k ~ 8.6 fourmis/cm² et un plateau q ~ 10 fourmis/cm/s) ; le README rappelle
    les valeurs historiques du `main.py` d'origine.
    """

    # --- Géométrie du domaine -------------------------------------------
    geometry: str = "rectangle"
    length: float = 100.0          # longueur du pont (mm)
    height: float = 10.0           # largeur du pont (mm) ; expériences : 5 / 10 / 20
    radius: float = 1.0            # rayon comportemental d'une fourmi (mm)

    # --- Perception ------------------------------------------------------
    angle_vision: float = np.pi    # ouverture TOTALE du cône de vision (rad)
    l_vision: float = 8.0          # portée de vision (mm)

    # --- Dynamique radiale (vitesse scalaire u) --------------------------
    nu_cruise: float = 20.0        # taux de relaxation vers la vitesse de croisière (1/s)
    xi_cruise: float = 14.0        # vitesse de croisière moyenne (mm/s).
                                   # Compromis assumé : les trajectoires réelles
                                   # donnent 12.5-13 (plateau du profil de
                                   # freinage) mais ne couvrent que k <= 4.4,
                                   # tandis que le plateau publié du diagramme
                                   # fondamental demanderait 15. Historique : 17.
    xi_spread: float = 0.2         # dispersion relative de xi entre individus.
                                   # Plafonné à 0.2 (choix utilisateur) : au-delà
                                   # l'hétérogénéité individuelle devient
                                   # invraisemblable, même si elle rapprocherait
                                   # la distribution des vitesses mesurée.
    nu_wall_stop: float = 1e10     # freinage radial imposé au contact d'une paroi
                                   # (moitié radiale de la condition de non-flux)

    # --- Dynamique angulaire (cap theta) ---------------------------------
    angular_mode: str = "field_norm"
    k_obj: float = 10.0            # intensité de la force d'objectif
    lambda_theta: float = 10.0     # taux de virage (1/s) — mode "fixed"
    lambda_max: float = 50.0       # taux de virage maximal (1/s) — plafond |theta_dot|
    force_ref: float = 10.0        # échelle de force de référence — mode "saturating"
    sigma_theta: float = 0.3       # bruit angulaire (rad/s^{1/2}) : brise-symétrie.
                                   # ABSENT du modèle de référence (mettre 0 pour le
                                   # reproduire à l'identique) ; retenu ici comme
                                   # brise-symétrie de principe, cf. README.

    # --- Freinage de contact (module nu_brake, tous groupes) -------------
    use_brake: bool = True
    l_brake: float = 3.0
    eps_brake: float = 0.7
    c_brake: float = 300.0         # historique main.py : 1e3 (surestimation d'un facteur ~2.5)
    alpha_brake: float = 4.0

    # --- Évitement latéral, groupes OPPOSÉS ------------------------------
    use_avoid_steer: bool = True
    l_avoid_steer: float = 2.3
    eps_avoid_steer: float = 0.7
    c_avoid_steer: float = 50.0

    # --- Répulsion latérale, MÊME groupe (non-pénétration « soft ») ------
    use_repulse: bool = True
    l_repulse: float = 1.5
    eps_repulse: float = 2.0       # > 1 : le facteur angulaire ne coupe jamais (omnidirectionnel)
    c_repulse: float = 40.0        # historique main.py : 1e5 — à ne PAS restaurer, cf. README

    # --- Communication / antennation, groupes OPPOSÉS --------------------
    use_comm: bool = True
    l_comm_min: float | None = None  # défaut : l_avoid_steer (borne interne de la rampe)
    l_comm: float = 5.0
    eps_comm: float = 1.0
    c_comm: float = 12.0           # historique : 20 (colmatage en run long)

    # --- Dépassement, MÊME groupe ----------------------------------------
    use_overtake: bool = True
    l_overtake: float = 4.0
    eps_overtake: float = 0.7
    c_overtake: float = 15.0
    alpha_overtake: float = 10.0   # porte de vitesse : actif si u_j <= alpha * xi_i.
                                   # 10 neutralise la porte (écart A du backlog) ;
                                   # une valeur dans (0, 1) la restaure.
    overtake_smooth_gate: bool = False  # True : porte lisse [(a.xi - u_j)/(a.xi)]_+ (écart B)

    # --- Suivi, MÊME groupe ----------------------------------------------
    use_follow: bool = True
    l_follow1: float = 3.0
    l_follow2: float = 8.0
    eps_follow: float = 0.7
    c_follow: float = 30.0

    # --- Thigmotactisme (attraction de paroi) ----------------------------
    use_wall_attraction: bool = True
    wall_law: str = "standoff"     # cf. README §8 ; "tangential" = loi du rapport
    l_wall_min: float = 0.5        # zone morte (lois tangential / outward_damping)
    l_wall_max: float = 4.0        # portée du terme de paroi
    c_wall: float = 2.0            # 10 avec la loi "tangential" (cf. PRESET_REPORT)
    l_wall_standoff: float = 1.0   # distance de consigne (loi standoff) ; 1 = R,
                                   # le flanc de la fourmi touche la paroi

    # --- Dégagement de paroi (« wall-escape ») ---------------------------
    # Débloque les impasses tête-à-tête contre une paroi : un agent collé au
    # bord, lent, et ayant un voisin à moins de 2R est poussé vers l'intérieur
    # du canal, avec une légère asymétrie par groupe qui brise la symétrie du
    # blocage. Fait partie de la configuration finale du rapport de passation
    # (annexe « Numerical scheme »), désactivé ici par défaut.
    use_wall_escape: bool = False
    c_wall_escape: float = 20.0
    l_wall_stuck: float = 1.5
    wall_escape_asym: float = 0.2

    # --- Conditions aux bords --------------------------------------------
    wall_mode: str = "specular"
    x_mode: str = "torus"

    # --- Régularisation numérique ----------------------------------------
    rho_reg: float = 1e-6          # plancher sur rho dans les noyaux en 1/rho

    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.l_comm_min is None:
            # Borne interne de la rampe de communication : la portée d'évitement
            # (cf. rapport_LLM.tex, eq. comm-mag). Champ explicite pour que
            # désactiver l'évitement ne déplace pas le support de la communication.
            self.l_comm_min = self.l_avoid_steer
        self._validate()

    def _validate(self):
        if self.geometry not in GEOMETRY_CODES:
            raise ValueError(f"géométrie inconnue : {self.geometry!r} "
                             f"(disponibles : {sorted(GEOMETRY_CODES)})")
        if self.angular_mode not in ANGULAR_MODES:
            raise ValueError(f"angular_mode inconnu : {self.angular_mode!r} "
                             f"(disponibles : {sorted(ANGULAR_MODES)})")
        if self.wall_law not in WALL_LAWS:
            raise ValueError(f"wall_law inconnue : {self.wall_law!r} "
                             f"(disponibles : {sorted(WALL_LAWS)})")
        if self.wall_mode not in WALL_MODES:
            raise ValueError(f"wall_mode inconnu : {self.wall_mode!r} "
                             f"(disponibles : {sorted(WALL_MODES)})")
        if self.x_mode not in X_MODES:
            raise ValueError(f"x_mode inconnu : {self.x_mode!r} "
                             f"(disponibles : {sorted(X_MODES)})")
        if not (self.length > 0 and self.height > 0 and self.radius > 0):
            raise ValueError("length, height et radius doivent être > 0.")
        if self.height <= 2 * self.radius:
            raise ValueError("canal plus étroit qu'une fourmi (height <= 2*radius).")
        if self.l_comm <= self.l_comm_min:
            raise ValueError("l_comm doit être > l_comm_min (rampe de communication vide).")
        if self.l_follow2 <= self.l_follow1:
            raise ValueError("l_follow2 doit être > l_follow1 (rampe de suivi vide).")
        if self.l_wall_max <= self.l_wall_min:
            raise ValueError("l_wall_max doit être > l_wall_min (bande de paroi vide).")
        if self.nu_cruise <= 0:
            raise ValueError("nu_cruise doit être > 0 (sinon la relaxation radiale diverge).")
        if self.sigma_theta < 0:
            raise ValueError("sigma_theta doit être >= 0.")

    # ------------------------------------------------------------------
    @property
    def l_interaction_max(self) -> float:
        """Portée d'interaction maximale : pas de la grille de voisinage."""
        return max(self.l_vision, self.l_brake, self.l_avoid_steer, self.l_repulse,
                   self.l_comm, self.l_overtake, self.l_follow2)


# ──────────────────────────────────────────────────────────────────────────
#  Paramètres de simulation
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SimParams:
    """Boucle, enregistrement et reproductibilité."""

    num_agents: int = 50
    fraction_plus: float = 0.5     # proportion d'agents du groupe +1 (vers +x)
    duration: float = 30.0         # durée simulée (s)
    dt: float = 0.01               # pas d'intégration (s)
    record_fps: float = 20.0       # images enregistrées par seconde simulée
    init_mode: str = "uniform"     # uniform | separated
    seed: int = 0                  # graine UNIQUE de tout l'aléa du run

    def __post_init__(self):
        if self.num_agents < 1:
            raise ValueError("num_agents doit être >= 1.")
        if not (0.0 <= self.fraction_plus <= 1.0):
            raise ValueError("fraction_plus doit être dans [0, 1].")
        if self.dt <= 0 or self.record_fps <= 0 or self.duration <= 0:
            raise ValueError("dt, record_fps et duration doivent être > 0.")
        if self.init_mode not in INIT_MODES:
            raise ValueError(f"init_mode inconnu : {self.init_mode!r} "
                             f"(disponibles : {list(INIT_MODES)})")
        ratio = 1.0 / (self.record_fps * self.dt)
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError(
                f"1/(record_fps*dt) = {ratio:.6f} n'est pas entier : une image "
                "enregistrée ne tomberait pas sur un pas d'intégration.")

    @property
    def substeps_per_frame(self) -> int:
        """Nombre de pas dt entre deux images enregistrées."""
        return int(round(1.0 / (self.record_fps * self.dt)))

    @property
    def n_frames(self) -> int:
        """Nombre d'images enregistrées (l'état initial n'en est pas une)."""
        return int(round(self.duration * self.record_fps))


# ──────────────────────────────────────────────────────────────────────────
#  Presets
# ──────────────────────────────────────────────────────────────────────────

#: Modèle minimal recommandé par la relecture critique du projet
#: (`rapport_evaluation_modele_fourmis.md`, §6.1) : objectif, freinage,
#: évitement court, antennation, murs. Suivi et dépassement désactivés.
PRESET_MINIMAL = dict(use_follow=False, use_overtake=False)

#: Modèle complet : tous les termes actifs (valeurs par défaut).
PRESET_FULL: dict = {}

#: Configuration du rapport de passation, MOINS la non-pénétration positionnelle
#: (retirée du code : jamais utilisée, cf. README §8). Reste donc : la loi de
#: paroi tangentielle avec c_wall = 10, l'équation angulaire du rapport, pas de
#: bruit, c_comm = 12, réflexion spéculaire et dégagement de paroi.
#: Attention : sans la correction positionnelle, cette configuration ne
#: reproduit PAS à l'identique le diagramme fondamental publié.
PRESET_REPORT = dict(
    angular_mode="field_norm",
    wall_law="tangential",
    c_wall=10.0,
    sigma_theta=0.0,
    c_comm=12.0,
    wall_mode="specular",
    x_mode="torus",
    use_wall_escape=True,
    c_wall_escape=20.0,
    l_wall_stuck=1.5,
    wall_escape_asym=0.2,
)

PRESETS = {"full": PRESET_FULL, "minimal": PRESET_MINIMAL, "report": PRESET_REPORT}


def make_model(preset: str = "full", **overrides) -> ModelParams:
    """Construit un `ModelParams` à partir d'un preset, avec surcharges.

    >>> make_model("minimal", height=5.0, c_comm=0.0).use_follow
    False
    """
    if preset not in PRESETS:
        raise ValueError(f"preset inconnu : {preset!r} (disponibles : {sorted(PRESETS)})")
    values = dict(PRESETS[preset])
    values.update(overrides)
    return ModelParams(**values)


# ──────────────────────────────────────────────────────────────────────────
#  Pont vers la couche compilée
# ──────────────────────────────────────────────────────────────────────────

#: Champs convertis en entiers (codes discrets et activations).
_INT_FIELDS = (
    "geom_type", "angular_mode", "wall_mode", "x_mode", "wall_law",
    "use_brake", "use_avoid_steer", "use_comm", "use_repulse",
    "use_overtake", "use_follow", "use_wall_attraction",
    "use_wall_escape", "overtake_smooth_gate",
)

#: Champs flottants, dans l'ordre de construction du NamedTuple.
_FLOAT_FIELDS = (
    "length", "height", "radius",
    "angle_vision", "l_vision",
    "nu_cruise", "xi_cruise", "nu_wall_stop",
    "k_obj", "lambda_theta", "lambda_max", "force_ref", "sigma_theta",
    "l_brake", "eps_brake", "c_brake", "alpha_brake",
    "l_avoid_steer", "eps_avoid_steer", "c_avoid_steer",
    "l_repulse", "eps_repulse", "c_repulse",
    "l_comm_min", "l_comm", "eps_comm", "c_comm",
    "l_overtake", "eps_overtake", "c_overtake", "alpha_overtake",
    "l_follow1", "l_follow2", "eps_follow", "c_follow",
    "l_wall_min", "l_wall_max", "c_wall", "l_wall_standoff",
    "c_wall_escape", "l_wall_stuck", "wall_escape_asym",
    "rho_reg",
)

KERNEL_FIELDS = _INT_FIELDS + _FLOAT_FIELDS

#: Empreinte de la LISTE DES CHAMPS du noyau.
#:
#: Numba nomme un type NamedTuple d'après `cls.__name__` et la liste des types
#: de ses membres — PAS d'après les noms de champs. Comme les champs sont ici
#: homogènes (des int64 puis des float64), deux versions successives de
#: `KERNEL_FIELDS` de même longueur produisent un type de MÊME NOM : le cache
#: disque de Numba (`cache=True`) resservait alors du code machine compilé pour
#: un ORDRE DE CHAMPS PÉRIMÉ, qui lisait chaque paramètre à la mauvaise
#: position. Symptôme observé : freinage et braquage simultanément faux, sans
#: qu'aucun paramètre ni aucune ligne de code ne soit en cause.
#:
#: La parade est dans `__init__.py` : le RÉPERTOIRE de cache de Numba porte
#: cette empreinte, donc une disposition différente écrit dans un répertoire
#: différent. On ne touche PAS au nom de la classe : Numba pickle le type dans
#: son index de cache, et un nom qui change casserait la relecture des index
#: déjà écrits (AttributeError au lieu d'une simple recompilation).
KERNEL_FINGERPRINT = hashlib.blake2s(
    "|".join(KERNEL_FIELDS).encode(), digest_size=4).hexdigest()

#: Le NamedTuple passé en ARGUMENT à chaque fonction compilée.
KernelParams = namedtuple("KernelParams", KERNEL_FIELDS)


def build_kernel_params(model: ModelParams) -> KernelParams:
    """Convertit un `ModelParams` en `KernelParams` (types figés).

    Les types sont strictement homogènes d'un appel à l'autre (int64 pour les
    codes et activations, float64 pour le reste) : Numba compile une fois, puis
    n'importe quelle autre instance passe sans recompilation — c'est ce qui rend
    les balayages de paramètres possibles.
    """
    values = {
        "geom_type": int(GEOMETRY_CODES[model.geometry]),
        "angular_mode": int(ANGULAR_MODES[model.angular_mode]),
        "wall_mode": int(WALL_MODES[model.wall_mode]),
        "x_mode": int(X_MODES[model.x_mode]),
        "wall_law": int(WALL_LAWS[model.wall_law]),
    }
    for name in _INT_FIELDS:
        if name not in values:
            values[name] = int(bool(getattr(model, name)))
    for name in _FLOAT_FIELDS:
        values[name] = float(getattr(model, name))
    return KernelParams(**values)


# ──────────────────────────────────────────────────────────────────────────
#  Sérialisation (un run doit être rejouable depuis son seul JSON)
# ──────────────────────────────────────────────────────────────────────────

def config_dict(model: ModelParams, sim: SimParams) -> dict:
    """Dictionnaire complet du run : TOUS les paramètres, la graine, la version."""
    return {
        "code_version": CODE_VERSION,
        "units": {"length": "mm", "time": "s"},
        "model": asdict(model),
        "sim": asdict(sim),
    }


def save_config(model: ModelParams, sim: SimParams, path: str | Path) -> Path:
    """Écrit la configuration complète du run dans un fichier JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config_dict(model, sim), indent=2, ensure_ascii=False))
    return path


def load_config(path: str | Path) -> tuple[ModelParams, SimParams]:
    """Recharge (model, sim) depuis un JSON écrit par `save_config`.

    Un champ inconnu (JSON écrit par une autre version du code) lève une erreur
    explicite plutôt que d'être ignoré silencieusement : un run rejoué doit
    l'être à l'identique ou pas du tout.
    """
    payload = json.loads(Path(path).read_text())
    model_names = {f.name for f in fields(ModelParams)}
    sim_names = {f.name for f in fields(SimParams)}

    for section, known in (("model", model_names), ("sim", sim_names)):
        unknown = set(payload.get(section, {})) - known
        if unknown:
            raise ValueError(
                f"champs inconnus dans la section {section!r} de {path} : "
                f"{sorted(unknown)} (config écrite par la version "
                f"{payload.get('code_version', '?')}, code actuel {CODE_VERSION})")

    return ModelParams(**payload["model"]), SimParams(**payload["sim"])
