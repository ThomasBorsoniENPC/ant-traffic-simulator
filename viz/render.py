"""
render.py
=========
Images fixes et vidéos d'un run.

La simulation et le rendu sont DÉCOUPLÉS : on rend depuis un `RunResult`
(éventuellement rechargé depuis le disque avec `antsim.load_run`), jamais
pendant la simulation. On peut donc re-rendre autrement — autre fenêtre, autre
cadence, autre colorisation — sans resimuler.

Convention de couleurs (imposée par le cahier des charges) :
    groupe +1, vers la source (+x)  ->  BLEU
    groupe -1, vers le nid   (-x)   ->  ROUGE

Les fourmis sont dessinées en ellipses orientées selon le cap : la forme rend
l'orientation lisible même à haute densité, là où des points ne montrent plus
rien. Les demi-axes sont un choix de RENDU (l'agent reste un disque de rayon R
dans le modèle) ; ils valent par défaut ceux utilisés dans les vidéos
antérieures du projet.

Le paramètre `sprite` permet de remplacer l'ellipse par une silhouette animée
(`ant`, `rabbit`, `rally_car`, `grandpa` — cf. `viz/sprites.py`). C'est
strictement cosmétique : la dynamique et les mesures ne changent pas, et
`ellipse` reste le défaut pour toutes les figures d'analyse. Le rendu par
sprite est en revanche nettement plus lent, comptez-le pour les présentations
et pas pour un balayage.

Sortie mp4 si ffmpeg est disponible, gif sinon (l'extension du chemin est
ajustée automatiquement).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import sprites

#: Demi-axes de rendu d'une fourmi (mm) : longueur selon le cap, largeur perpendiculaire.
ELLIPSE_SEMI_AXES = (1.4, 0.7)

COLOR_PLUS = "#2c6fbb"    # groupe +1 -> source
COLOR_MINUS = "#c0392b"   # groupe -1 -> nid


#: Teinte d'une ouvrière de Linepithema humile : brun sombre.
NATURAL_BODY = "#6d4a30"

#: Modes de coloration.
#:   group   : bleu/rouge selon le sens de marche — lisible, c'est le défaut,
#:             et c'est indispensable dès qu'on veut suivre le mélange.
#:   natural : toutes les fourmis de la même teinte brune, légèrement variée
#:             d'un individu à l'autre. Plus proche de l'oeil, mais on perd
#:             l'information de groupe : à réserver à l'illustration.
COLOR_MODES = ("group", "natural")


def _darken(hex_color, factor):
    """Éclaircit (facteur > 1) ou assombrit (facteur < 1) une couleur.

    Les sprites détaillés ont besoin de relief (tête et thorax plus sombres que
    le gastre) sans perdre la teinte qui porte l'information : on module donc la
    couleur de base au lieu de passer au gris.
    """
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(min(255, max(0, int(c * factor)))
                                   for c in (r, g, b))


def _agent_look(result, color_mode, size_jitter):
    """(couleur, facteur de taille) de chaque agent.

    La variabilité — teinte et gabarit — est tirée d'un générateur initialisé
    par la graine du run : deux rendus du même run donnent exactement les mêmes
    fourmis, et deux runs différents donnent des fourmis différentes.
    """
    rng = np.random.default_rng(result.sim.seed + 9161)
    n = result.n_agents
    sizes = 1.0 + size_jitter * (2.0 * rng.random(n) - 1.0)
    if color_mode == "natural":
        tints = 0.82 + 0.36 * rng.random(n)
        colors = [_darken(NATURAL_BODY, f) for f in tints]
    else:
        colors = [COLOR_PLUS if g > 0 else COLOR_MINUS for g in result.group]
    return colors, sizes


def _import_matplotlib(interactive: bool):
    import matplotlib
    if not interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _setup_axes(plt, result, figure_width: float):
    """Crée la figure, trace les parois, renvoie (fig, ax)."""
    model = result.model
    height_ratio = model.height / model.length
    fig, ax = plt.subplots(figsize=(figure_width, figure_width * height_ratio + 0.9))
    ax.set_xlim(0.0, model.length)
    ax.set_ylim(-0.5, model.height + 0.5)
    ax.set_aspect("equal")
    ax.plot([0, model.length], [0, 0], "k-", lw=1.2)
    ax.plot([0, model.length], [model.height, model.height], "k-", lw=1.2)
    ax.set_yticks([])
    # Seules les parois doivent se lire comme des bords : on retire le cadre.
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("x (mm)")
    return fig, ax


def _make_ellipses(ax, result, frame: int, semi_axes):
    """Ajoute une ellipse par agent. Renvoie (artistes, fonction de mise à jour)."""
    from matplotlib.patches import Ellipse

    a, b = semi_axes
    patches = []
    for i in range(result.n_agents):
        color = COLOR_PLUS if result.group[i] > 0 else COLOR_MINUS
        e = Ellipse((result.x[frame, i], result.y[frame, i]),
                    width=2 * a, height=2 * b,
                    angle=np.degrees(result.theta[frame, i]),
                    facecolor=color, edgecolor="k", lw=0.3, alpha=0.85)
        ax.add_patch(e)
        patches.append(e)

    def update(k):
        for i, patch in enumerate(patches):
            patch.set_center((result.x[k, i], result.y[k, i]))
            patch.angle = np.degrees(result.theta[k, i])
        return patches

    return patches, update


def _make_sprites(ax, result, frame: int, sprite: str, scale: float = 1.0,
                  color_mode: str = "group", size_jitter: float = 0.0):
    """Ajoute une silhouette animée par agent.

    Quatre artistes par agent seulement : trois chemins composés (corps, ombre,
    détails sombres) et une polyligne dont les segments sont séparés par des
    NaN. Sans cela, six pattes par fourmi feraient des centaines d'artistes à
    rafraîchir à chaque image.
    """
    from matplotlib.patches import PathPatch

    cache, spec = sprites.build_cache(sprite, scale)
    phases = sprites.gait_phase(result.x, result.y, result.model.length)
    colors, sizes = _agent_look(result, color_mode, size_jitter)
    artists, groups = [], []
    empty = sprites.compound_path([], 0, 0, 0)

    for i in range(result.n_agents):
        color = colors[i]
        body = PathPatch(empty, facecolor=color, edgecolor="#20242b",
                         lw=0.45, zorder=3)
        shade = PathPatch(empty, facecolor=_darken(color, 0.74),
                          edgecolor="#20242b", lw=0.45, zorder=4)
        dark = PathPatch(empty, facecolor="#20242b", edgecolor="none", zorder=5)
        for patch in (body, shade, dark):
            ax.add_patch(patch)
        line, = ax.plot([], [], lw=spec["lw_dark"], color=spec["stroke"],
                        solid_capstyle="round", solid_joinstyle="round", zorder=2)
        groups.append((body, shade, dark, line))
        artists += [body, shade, dark, line]

    def update(k):
        for i, (body, shade, dark, line) in enumerate(groups):
            bucket = int(phases[k, i] * sprites.N_PHASES) % sprites.N_PHASES
            f_body, f_shade, f_dark, strokes = cache[bucket]
            x, y, th, s = (result.x[k, i], result.y[k, i],
                           result.theta[k, i], sizes[i])
            body.set_path(sprites.compound_path([s * a for a in f_body], x, y, th))
            shade.set_path(sprites.compound_path([s * a for a in f_shade], x, y, th))
            dark.set_path(sprites.compound_path([s * a for a in f_dark], x, y, th))
            pts = sprites.place(s * sprites.stitch(strokes), x, y, th)
            line.set_data(pts[:, 0], pts[:, 1])
        return artists

    update(frame)
    return artists, update


def _make_agents(ax, result, frame, sprite, semi_axes, scale=1.0,
                 color_mode="group", size_jitter=0.0):
    """Aiguille vers le rendu demandé."""
    if sprite == sprites.DEFAULT:
        return _make_ellipses(ax, result, frame, semi_axes)
    return _make_sprites(ax, result, frame, sprite, scale, color_mode, size_jitter)


def _legend(ax, color_mode="group"):
    """Légende des groupes. Sans objet en coloration naturelle, où toutes les
    fourmis ont la même teinte : on ne l'affiche alors pas."""
    if color_mode != "group":
        return
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=COLOR_PLUS, edgecolor="k",
                             label="g = +1  (towards the food)"),
                       Patch(facecolor=COLOR_MINUS, edgecolor="k",
                             label="g = -1  (towards the nest)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.35),
              ncol=2, frameon=False, fontsize=9)


# ──────────────────────────────────────────────────────────────────────────
#  Image fixe
# ──────────────────────────────────────────────────────────────────────────

def save_snapshot(result, path: str | Path, frame: int = -1,
                  figure_width: float = 13.0, semi_axes=ELLIPSE_SEMI_AXES,
                  dpi: int = 140, sprite: str = sprites.DEFAULT,
                  sprite_scale: float = 1.0, color_mode: str = "group",
                  size_jitter: float = 0.0) -> Path:
    """Enregistre une image d'une image donnée du run (par défaut la dernière)."""
    plt = _import_matplotlib(interactive=False)
    if frame < 0:
        frame = result.n_frames + frame

    fig, ax = _setup_axes(plt, result, figure_width)
    _make_agents(ax, result, frame, sprite, semi_axes, sprite_scale,
                 color_mode, size_jitter)
    ax.set_title(f"t = {result.times[frame]:.2f} s   "
                 f"({result.n_agents} agents, "
                 f"{result.model.length:.0f} x {result.model.height:.0f} mm bridge)")
    _legend(ax, color_mode)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────
#  Vidéo
# ──────────────────────────────────────────────────────────────────────────

def save_video(result, path: str | Path, fps: int = 20, stride: int = 1,
               figure_width: float = 13.0, semi_axes=ELLIPSE_SEMI_AXES,
               dpi: int = 110, bitrate: int = 2400, progress: bool = True,
               sprite: str = sprites.DEFAULT, sprite_scale: float = 1.0,
               color_mode: str = "group", size_jitter: float = 0.0) -> Path:
    """Écrit une vidéo du run.

    `stride` permet de n'utiliser qu'une image enregistrée sur n (accélère la
    lecture et le rendu). Le chemin reçoit l'extension `.mp4` si ffmpeg est
    disponible, `.gif` sinon.
    """
    plt = _import_matplotlib(interactive=False)
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

    frames = np.arange(0, result.n_frames, stride)
    fig, ax = _setup_axes(plt, result, figure_width)
    artists, draw = _make_agents(ax, result, int(frames[0]), sprite,
                                 semi_axes, sprite_scale, color_mode, size_jitter)
    _legend(ax, color_mode)
    label = ax.text(0.01, 1.02, "", transform=ax.transAxes, fontsize=10, va="bottom")
    fig.tight_layout()

    bar = None
    if progress:
        try:
            from tqdm import tqdm
            bar = tqdm(total=len(frames), desc="rendu", unit="f")
        except ImportError:
            pass

    def update(k):
        drawn = draw(k)
        label.set_text(f"t = {result.times[k]:6.2f} s")
        if bar is not None:
            bar.update(1)
        return list(drawn) + [label]

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if FFMpegWriter.isAvailable():
        path = path.with_suffix(".mp4")
        writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    else:
        path = path.with_suffix(".gif")
        writer = PillowWriter(fps=fps)

    anim.save(str(path), writer=writer, dpi=dpi)
    if bar is not None:
        bar.close()
    plt.close(fig)
    return path
