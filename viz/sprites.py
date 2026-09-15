"""
sprites.py
==========
Représentations graphiques d'un agent, de la plus sobre à la plus inutile.

Le rendu par défaut reste l'ellipse orientée : c'est celui des figures et des
vidéos d'analyse, et il ne doit pas bouger. Les autres sprites sont là pour les
présentations et pour le plaisir.

    ellipse      ellipse orientée (défaut, inchangé)
    ant          fourmi : gastre, thorax, tête, antennes, six pattes en
                 trépied alterné
    rabbit       lapin, avec oreilles et bonds
    rally_car    voiture de rallye vue de dessus, roues et aileron
    grandpa      papi vu de dessus, avec sa canne

Principe
--------
Un sprite est décrit dans un repère LOCAL orienté (+x vers l'avant, +y à
gauche), en unités de longueur de corps. Chaque sprite renvoie trois listes :

    fills_body   formes PLEINES de la couleur du groupe (corps, tête, ...)
    fills_shade  formes pleines dans une version ASSOMBRIE de cette couleur,
                 pour donner du relief sans perdre l'identité du groupe
    fills_dark   formes pleines sombres (yeux, roues, chapeau, ...)
    strokes      polylignes fines (pattes, antennes, canne, ...)

Le rendu les assemble en QUATRE artistes matplotlib par agent : trois chemins
composés (un par couleur de remplissage) et une polyligne à segments séparés
par des NaN. Sans cela, 60 agents × 12 membres feraient des centaines
d'artistes à rafraîchir à chaque image.

La marche est pilotée par la PHASE DE DÉMARCHE, elle-même calculée à partir de
la distance parcourue et non du temps : les pattes accélèrent avec l'agent,
et s'arrêtent quand il s'arrête. C'est ce détail qui fait que ça a l'air vivant.
"""

from __future__ import annotations

import numpy as np

#: Longueur de foulée (mm) : distance parcourue pour un cycle de démarche complet.
STRIDE_MM = 2.2

#: Longueur du CORPS d'un agent à l'écran, en mm.
#: C'est la longueur de fourmi MESURÉE sur les données de suivi (boîte
#: englobante médiane : 3.2 x 1.8 mm, cf. `scripts/extract_params_from_data.py`).
#: Attention : le modèle, lui, fait interagir des disques de rayon R = 1 mm.
#: Une fourmi dessinée à sa taille réelle se recouvre donc visuellement plus que
#: les disques du modèle ne le font — c'est un fait sur le modèle, pas un défaut
#: du dessin. Pour dessiner les agents à la taille de leur disque d'interaction,
#: passer `scale=0.63` (soit 2R / 3.2).
AGENT_LENGTH_MM = 3.2


def _swing(phase, offset=0.0):
    """Élongation d'un membre, dans [-1, 1], pour une phase de démarche."""
    return np.sin(2.0 * np.pi * (phase + offset))


#: Fraction du cycle passée en APPUI (le reste est le retour aérien). Chez les
#: insectes elle vaut ~0.6 : le pied recule lentement en poussant, puis revient
#: vite vers l'avant. C'est cette asymétrie qui distingue une vraie démarche
#: d'un balancement sinusoïdal.
DUTY = 0.6


def _stride(phase, offset=0.0, amp=1.0):
    """(avance du pied, levée) pour une patte, démarche à appui/retour asymétrique.

    `avance` va de +amp (pose) à -amp (fin de poussée) pendant l'appui, puis
    revient vite vers l'avant. `levée` est nulle pendant l'appui et forme une
    bosse pendant le retour — la patte se décolle visiblement.
    """
    u = (phase + offset) % 1.0
    if u < DUTY:
        return amp * (1.0 - 2.0 * u / DUTY), 0.0
    v = (u - DUTY) / (1.0 - DUTY)
    return amp * (-1.0 + 2.0 * v), 0.16 * np.sin(np.pi * v)


def _teardrop(cx, cy, rx, ry, point=1.5, n=22):
    """Ovale effilé vers l'arrière — la forme du gastre d'une fourmi."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    stretch = 1.0 + (point - 1.0) * np.clip(-np.cos(a), 0.0, 1.0) ** 2
    return np.column_stack([cx + rx * stretch * np.cos(a), cy + ry * np.sin(a)])


def _ellipse(cx, cy, rx, ry, rot=0.0, n=18):
    """Ellipse fermée, éventuellement inclinée — la brique de base des corps."""
    a = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pts = np.column_stack([rx * np.cos(a), ry * np.sin(a)])
    if rot:
        c, s = np.cos(rot), np.sin(rot)
        pts = pts @ np.array([[c, s], [-s, c]])
    return pts + np.array([cx, cy])


def _box(x0, x1, y0, y1):
    """Rectangle fermé."""
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])


# ──────────────────────────────────────────────────────────────────────────
#  Fourmi
# ──────────────────────────────────────────────────────────────────────────

def _ant(phase):
    fills = [
        _ellipse(-1.05, 0.0, 0.62, 0.50),                # gastre
        _ellipse(-0.15, 0.0, 0.36, 0.26),                # thorax
        _ellipse(0.62, 0.0, 0.44, 0.40),                 # tête
    ]
    dark_fills = []
    dark = [np.array([[-0.48, 0.0], [0.20, 0.0]])]       # pétiole
    # Six pattes, trépied alterné : (avant gauche, milieu droite, arrière
    # gauche) en phase, les trois autres en opposition.
    for side, sgn in ((0, 1.0), (1, -1.0)):
        for idx, (x0, reach, spread) in enumerate(
                ((0.15, 0.55, 0.95), (-0.10, 0.30, 1.05), (-0.35, -0.30, 0.95))):
            offset = 0.5 * ((idx + side) % 2)
            s = _swing(phase, offset)
            knee = np.array([x0 + 0.25 * s, sgn * 0.45])
            foot = np.array([x0 + reach + 0.45 * s, sgn * spread])
            dark.append(np.array([[x0, sgn * 0.12], knee, foot]))
    # Antennes coudées, qui balaient doucement
    for sgn in (1.0, -1.0):
        w = 0.12 * _swing(phase, 0.25 * sgn)
        dark.append(np.array([[0.75, sgn * 0.22],
                              [1.25, sgn * (0.45 + w)],
                              [1.70, sgn * (0.30 + w) + 0.15]]))
    return fills, [], dark_fills, dark


# ──────────────────────────────────────────────────────────────────────────
#  Fourmi réaliste
# ──────────────────────────────────────────────────────────────────────────

#: Attaches des six pattes sur le MÉSOSOME, et non le long du corps : chez un
#: insecte les six pattes partent toutes du thorax. C'est le détail qui, plus
#: que la silhouette, fait qu'on lit « fourmi » plutôt qu'« araignée ».
#: (x de la hanche, portée du genou, portée du pied, ouverture, décalage de phase)
_LEGS = (
    (0.16, (0.38, 0.52), (0.92, 0.98), 0.0),     # avant
    (0.00, (0.08, 0.62), (0.14, 1.30), 0.5),     # milieu
    (-0.24, (-0.32, 0.58), (-0.88, 1.16), 0.0),  # arrière
)


def _ant_realistic(phase):
    fills = [
        _teardrop(-1.18, 0.0, 0.52, 0.44, point=1.55),   # gastre effilé
        _ellipse(-0.50, 0.0, 0.15, 0.13),                # pétiole (un seul noeud)
    ]
    shade = [
        _ellipse(-0.14, 0.0, 0.40, 0.22),                # mésosome, étroit
        _ellipse(0.26, 0.0, 0.10, 0.08),                 # cou, fin : il faut que
                                                         # la tête se détache
        _ellipse(0.72, 0.0, 0.38, 0.41),                 # tête, plus large en arrière
    ]
    dark = [
        _ellipse(0.76, 0.28, 0.10, 0.08, rot=0.4),       # yeux composés
        _ellipse(0.76, -0.28, 0.10, 0.08, rot=-0.4),
    ]
    strokes = []

    # Mandibules, légèrement écartées
    for sgn in (1.0, -1.0):
        strokes.append(np.array([[1.04, sgn * 0.16],
                                 [1.28, sgn * 0.26],
                                 [1.40, sgn * 0.10]]))

    # Antennes coudées : scape droit, puis funicule qui retombe et balaie
    for sgn in (1.0, -1.0):
        w = 0.14 * _swing(phase, 0.25 * sgn)
        elbow = np.array([1.22, sgn * (0.44 + 0.5 * w)])
        strokes.append(np.array([[0.94, sgn * 0.22], elbow,
                                 [1.64, sgn * (0.32 + w)],
                                 [1.92, sgn * (0.12 + w)]]))

    # Six pattes en trépied : avant-gauche, milieu-droite, arrière-gauche en
    # phase, les trois autres en opposition.
    for side, sgn in ((0, 1.0), (1, -1.0)):
        for (hip_x, knee, foot, base) in _LEGS:
            fore, lift = _stride(phase, base + 0.5 * side, amp=0.30)
            kx, ky = knee
            fx, fy = foot
            strokes.append(np.array([
                [hip_x, sgn * 0.16],
                [kx + 0.45 * fore, sgn * (ky + 0.6 * lift)],
                [fx + fore, sgn * (fy + lift)],
            ]))
    return fills, shade, dark, strokes


# ──────────────────────────────────────────────────────────────────────────
#  Lapin
# ──────────────────────────────────────────────────────────────────────────

def _rabbit(phase):
    hop = 0.14 * max(0.0, _swing(phase))           # le lapin avance par bonds
    fills = [
        _ellipse(-0.30 + hop, 0.0, 0.82, 0.54),    # corps
        _ellipse(0.62 + hop, 0.0, 0.40, 0.34),     # tête
        _ellipse(1.22 + hop, 0.34, 0.38, 0.14, rot=0.45),    # oreille gauche
        _ellipse(1.22 + hop, -0.34, 0.38, 0.14, rot=-0.45),  # oreille droite
        _ellipse(-1.18 + hop, 0.0, 0.20, 0.20),    # queue
    ]
    dark_fills = [_ellipse(0.82 + hop, 0.20, 0.07, 0.07),
                  _ellipse(0.82 + hop, -0.20, 0.07, 0.07)]  # yeux
    s = _swing(phase)
    dark = []
    for sgn in (1.0, -1.0):
        dark.append(np.array([[-0.40 + hop, sgn * 0.48],
                              [-0.66 + hop - 0.18 * s, sgn * 0.80]]))
        dark.append(np.array([[0.28 + hop, sgn * 0.38],
                              [0.48 + hop + 0.18 * s, sgn * 0.62]]))
    return fills, [], dark_fills, dark


# ──────────────────────────────────────────────────────────────────────────
#  Voiture de rallye
# ──────────────────────────────────────────────────────────────────────────

def _rally_car(phase):
    fills = [np.array([[-1.20, -0.52], [0.80, -0.52], [1.28, -0.28],
                       [1.28, 0.28], [0.80, 0.52], [-1.20, 0.52]]),
             _box(-1.45, -1.28, -0.62, 0.62)]      # châssis + aileron
    dark_fills = [_box(x - 0.28, x + 0.28, sgn * 0.52, sgn * 0.78)
                  for x in (0.70, -0.70) for sgn in (1.0, -1.0)]   # roues
    dark = [np.array([[0.18, -0.42], [0.52, -0.24], [0.52, 0.24], [0.18, 0.42]]),
            np.array([[-1.28, 0.0], [-1.52, 0.0]])]   # pare-brise, échappement
    if _swing(phase) > 0.5:                        # gravillons projetés
        dark.append(np.array([[-1.62, 0.28], [-1.88, 0.40]]))
        dark.append(np.array([[-1.62, -0.22], [-1.84, -0.38]]))
    return fills, [], dark_fills, dark


# ──────────────────────────────────────────────────────────────────────────
#  Papi et sa canne
# ──────────────────────────────────────────────────────────────────────────

def _grandpa(phase):
    s = _swing(phase)
    fills = [
        _ellipse(-0.12, 0.0, 0.52, 0.44),                    # épaules, vues de dessus
        _ellipse(0.30, 0.0, 0.46, 0.44),                     # bord du chapeau
    ]
    dark_fills = [_ellipse(0.32, 0.0, 0.26, 0.25)]           # calotte du chapeau
    dark = [
        np.array([[-0.16, 0.42], [0.18, 0.66 + 0.12 * s]]),  # bras gauche
        np.array([[-0.16, -0.42], [0.34, -0.72]]),           # bras droit
        np.array([[0.34, -0.72], [0.86, -0.84]]),            # canne
        np.array([[-0.38, 0.18], [-0.66 + 0.34 * s, 0.32]]), # jambes
        np.array([[-0.38, -0.18], [-0.66 - 0.34 * s, -0.32]]),
    ]
    return fills, [], dark_fills, dark


# ──────────────────────────────────────────────────────────────────────────
#  Registre
# ──────────────────────────────────────────────────────────────────────────

#: Couleur des traits fins. Les pattes et antennes d'une fourmi sont en gris
#: moyen plutôt qu'en noir : à cette taille, du noir pur donne un enchevêtrement
#: illisible dès que les agents se touchent.
STROKE_GREY = "#6b7078"
STROKE_DARK = "#20242b"

#: nom -> construction, épaisseur et couleur des traits fins.
#: L'échelle n'est PAS réglée à la main : elle est calculée pour que le corps
#: dessiné mesure `AGENT_LENGTH_MM`, quel que soit le repère local du sprite.
#: Tous les rendus font donc exactement la même taille à l'écran.
SPRITES = {
    "ant":       dict(build=_ant,            lw_dark=0.9,  stroke=STROKE_GREY),
    "ant_real":  dict(build=_ant_realistic,  lw_dark=0.85, stroke=STROKE_GREY),
    "rabbit":    dict(build=_rabbit,         lw_dark=1.1,  stroke=STROKE_GREY),
    "rally_car": dict(build=_rally_car,      lw_dark=1.1,  stroke=STROKE_DARK),
    "grandpa":   dict(build=_grandpa,        lw_dark=1.2,  stroke=STROKE_DARK),
}

#: Le rendu par défaut, qui n'utilise pas ce module.
DEFAULT = "ellipse"

#: Nombre de phases distinctes pré-calculées. Les membres n'ont pas besoin
#: d'une résolution continue, et mettre en cache évite de reconstruire la
#: géométrie de chaque agent à chaque image.
N_PHASES = 16


def available():
    """Noms de rendu acceptés, `ellipse` compris."""
    return [DEFAULT] + sorted(SPRITES)


def _body_span(groups):
    """Extension avant-arrière des formes PLEINES, en unités locales.

    Mesurée sur les remplissages seuls : les pattes, antennes et cannes
    dépassent du corps, comme dans la nature, et n'entrent pas dans la
    définition de la longueur du corps.
    """
    pts = [np.asarray(a, dtype=float) for group in groups[:3] for a in group]
    if not pts:
        return 1.0
    xs = np.concatenate([a[:, 0] for a in pts])
    return float(xs.max() - xs.min())


def build_cache(name, scale=1.0):
    """Pré-calcule les polylignes pour `N_PHASES` phases de démarche.

    L'échelle est DÉDUITE du sprite : on mesure l'extension de ses formes
    pleines et on la ramène à `AGENT_LENGTH_MM`. Tous les rendus sortent donc à
    la même taille, et celle-ci est la taille réelle mesurée d'une fourmi — pas
    un réglage à l'oeil. `scale` est un multiplicateur pour ajuster.

    Renvoie (cache, spec) où `cache[p]` est un quadruplet
    (corps, ombre, détails sombres, traits) en repère local, mis à l'échelle.
    """
    if name not in SPRITES:
        raise ValueError(f"sprite inconnu : {name!r} (disponibles : {available()})")
    spec = SPRITES[name]
    k = scale * AGENT_LENGTH_MM / _body_span(spec["build"](0.0))
    cache = []
    for p in range(N_PHASES):
        groups = spec["build"](p / N_PHASES)
        cache.append(tuple([k * np.asarray(a, dtype=float) for a in group]
                           for group in groups))
    return cache, spec


def compound_path(polygons, x, y, theta):
    """Chemin matplotlib fermé, réunissant plusieurs polygones en UN artiste."""
    from matplotlib.path import Path as MplPath

    if not polygons:
        return MplPath(np.zeros((1, 2)), [MplPath.MOVETO])
    verts, codes = [], []
    for poly in polygons:
        pts = place(poly, x, y, theta)
        verts.append(pts)
        verts.append(pts[:1])
        codes.append(MplPath.MOVETO)
        codes.extend([MplPath.LINETO] * (len(pts) - 1))
        codes.append(MplPath.CLOSEPOLY)
    return MplPath(np.vstack(verts), codes)


def stitch(polylines):
    """Concatène des polylignes en un seul tableau, séparées par des NaN.

    C'est ce qui permet de dessiner tous les membres d'un agent avec UN seul
    artiste matplotlib au lieu d'un par segment.
    """
    if not polylines:
        return np.zeros((0, 2))
    out = []
    for i, p in enumerate(polylines):
        if i:
            out.append(np.array([[np.nan, np.nan]]))
        out.append(p)
    return np.vstack(out)


def place(local, x, y, theta):
    """Applique la rotation du cap et la translation à des coordonnées locales."""
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def gait_phase(xs, ys, length, stride=STRIDE_MM):
    """Phase de démarche de chaque agent à chaque image.

    Intègre la distance réellement parcourue (en ignorant les sauts dus à la
    périodicité) et la divise par la longueur de foulée. Les membres bougent
    donc au rythme de l'avancée, et se figent quand l'agent est arrêté.
    """
    dx = np.diff(xs, axis=0)
    dy = np.diff(ys, axis=0)
    wrapped = np.abs(dx) > 0.5 * length
    step = np.where(wrapped, 0.0, np.hypot(dx, dy))
    travelled = np.vstack([np.zeros((1, xs.shape[1])), np.cumsum(step, axis=0)])
    return (travelled / stride) % 1.0
