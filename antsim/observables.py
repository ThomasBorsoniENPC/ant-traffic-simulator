"""
observables.py
==============
Mesures quantitatives sur un run, définies pour être comparables à
Poissonnier et al. 2019 (eLife).

Trois familles :

  * FLUX ET DENSITÉ — `fundamental_point`, avec les trois estimateurs du flux
    décrits dans le rapport de passation (ligne centrale seule / moyenne sur
    plusieurs sections / domaine entier) ;
  * CONTACTS — `contact_stats`, qui reconstruit les épisodes de contact
    antennaire entre sens opposés et en tire les deux lois de l'article :
    C(k) (contacts par fourmi) et T = T0 + C.dT (temps de traversée) ;
  * STRUCTURE — `density_spacetime` (densité le long du pont au cours du temps)
    et `segregation_index` (mélange transverse).

Conventions d'unités (identiques à l'article) :
    k [fourmis/cm²]      densité surfacique
    q [fourmis/cm/s]     flux par unité de largeur, LES DEUX SENS CONFONDUS
    U [mm/s]             vitesse
Le pont mesure `length` x `height` en mm ; 1 cm = 10 mm.
"""

from __future__ import annotations

import numpy as np

MM_PER_CM = 10.0


def _stationary_slice(result, warmup_frac):
    """Indice de la première image du régime considéré comme stationnaire."""
    return int(round(warmup_frac * result.n_frames))


# ──────────────────────────────────────────────────────────────────────────
#  Flux et densité
# ──────────────────────────────────────────────────────────────────────────

def line_crossings(result, x_line, start):
    """Nombre de franchissements de l'abscisse `x_line`, les deux sens confondus.

    Compte les changements de signe de (x - x_line) entre deux images
    consécutives, en ignorant les sauts dus à la périodicité (|dx| > L/2).
    """
    length = result.model.length
    x = result.x[start:].astype(np.float64)
    dx = np.diff(x, axis=0)
    wrapped = np.abs(dx) > 0.5 * length
    before = x[:-1] - x_line
    after = x[1:] - x_line
    crossed = (np.sign(before) != np.sign(after)) & ~wrapped
    return int(np.sum(crossed))


def window_density(result, x_center, start, window_mm=MM_PER_CM):
    """Densité moyenne (fourmis/cm²) dans une fenêtre centrée en `x_center`.

    La fenêtre fait `window_mm` de long sur toute la largeur du pont : sur le
    pont standard de 10 mm de large c'est exactement le 1 cm² de l'article.
    """
    half = 0.5 * window_mm
    x = result.x[start:]
    inside = (x >= x_center - half) & (x <= x_center + half)
    area_cm2 = (window_mm / MM_PER_CM) * (result.model.height / MM_PER_CM)
    return float(inside.sum(axis=1).mean() / area_cm2)


def fundamental_point(result, warmup_frac=0.4, n_lines=7, window_mm=MM_PER_CM):
    """Réduit un run à un point (k, q) du diagramme fondamental.

    Renvoie les TROIS estimateurs, mesurés sur les mêmes images :

      `_line`   ligne centrale seule et fenêtre de 1 cm² — le protocole eLife,
                et le plus bruité ;
      `_lines`  moyenne sur `n_lines` sections réparties sur le pont. En régime
                stationnaire le flux moyen est le même à toute section (le
                nombre d'agents se conserve), donc moyenner réduit la variance
                sans biaiser ;
      `_domain` domaine entier : k est exact (N / aire) et q est le flux intégré
                sur toutes les sections, q = <somme des |v_x|> / (L . W).
    """
    model = result.model
    start = _stationary_slice(result, warmup_frac)
    dt_rec = float(result.times[1] - result.times[0])
    duration = (result.n_frames - 1 - start) * dt_rec
    width_cm = model.height / MM_PER_CM

    x_center = 0.5 * model.length
    k_line = window_density(result, x_center, start, window_mm)
    q_line = line_crossings(result, x_center, start) / (duration * width_cm)

    xs = np.linspace(0.2 * model.length, 0.8 * model.length, n_lines)
    k_lines = float(np.mean([window_density(result, xc, start, window_mm) for xc in xs]))
    q_lines = float(np.mean([line_crossings(result, xc, start) / (duration * width_cm)
                             for xc in xs]))

    area_cm2 = (model.length / MM_PER_CM) * width_cm
    k_domain = result.n_agents / area_cm2
    speed_x = np.abs(result.u[start:] * np.cos(result.theta[start:]))
    q_domain = float(speed_x.mean() * result.n_agents / (model.length * width_cm))

    u_mean = float(result.u[start:].mean())
    return {
        "n_agents": result.n_agents, "seed": result.sim.seed,
        "height": model.height,
        "k_line": k_line, "q_line": q_line,
        "k_lines": k_lines, "q_lines": q_lines,
        "k_domain": k_domain, "q_domain": q_domain,
        "u_mean": u_mean,
        "u_from_flux": q_domain * MM_PER_CM / k_domain if k_domain > 0 else np.nan,
        "stopped_frac": float(np.mean(result.u[start:] < 1.0)),
    }


def monod_flux(k, xi, beta_bar, nu_cruise):
    """Loi de Monod q = xi.nu_c.k / (nu_c + beta_bar.k) — la fermeture mono-cinétique.

    Pente xi à densité nulle (la vitesse libre est PRÉDITE, pas ajustée),
    asymptote xi.nu_c/beta_bar. Ne décroît jamais : c'est la signature d'un
    freinage placé comme taux de relaxation et non comme capacité soustraite.
    """
    return xi * nu_cruise * k / (nu_cruise + beta_bar * k)


# ──────────────────────────────────────────────────────────────────────────
#  Contacts antennaires
# ──────────────────────────────────────────────────────────────────────────

def _contact_pairs(x, y, theta, group, d_contact, cos_min):
    """Paires (i, j), i < j, en contact antennaire à cette image.

    Contact = sens opposés, distance <= d_contact, et CHACUN voit l'autre dans
    son cône frontal (cos >= cos_min). La distance seuil représente la portée
    des antennes, un peu plus large que le corps (2R = 2 mm).
    """
    n = len(x)
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    dist = np.hypot(dx, dy)
    opposite = (group[:, None] * group[None, :]) < 0
    close = (dist <= d_contact) & (dist > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        ux, uy = dx / dist, dy / dist
    faces_i = np.cos(theta)[:, None] * ux + np.sin(theta)[:, None] * uy >= cos_min
    faces_j = np.cos(theta)[None, :] * (-ux) + np.sin(theta)[None, :] * (-uy) >= cos_min
    mask = np.triu(opposite & close & faces_i & faces_j, k=1)
    i, j = np.nonzero(mask)
    return i * n + j


def contact_stats(result, d_contact=2.5, cos_min=0.5, warmup_frac=0.4,
                  section_mm=20.0):
    """Épisodes de contact antennaire, et les deux lois de l'article.

    `d_contact` : portée antennaire (mm). Le corps fait 2R = 2 mm ; 2.5 place le
    contact juste au-delà, ce qui correspond à des antennes qui se touchent.
    `section_mm` : longueur de la section centrale sur laquelle on mesure le
    temps de traversée T (l'article mesure sur une section centrale, pas sur le
    pont entier).

    Renvoie notamment :
      `contacts_per_ant_per_s`  taux de contacts par fourmi et par seconde ;
      `contacts_per_transit`    C de l'article : contacts subis pendant une
                                traversée de la section ;
      `mean_duration`           durée moyenne d'un contact (s) ;
      `T0`, `delta_T`           ordonnée à l'origine et pente de T = T0 + C.dT,
                                ajustées sur les traversées individuelles.
    """
    start = _stationary_slice(result, warmup_frac)
    dt_rec = float(result.times[1] - result.times[0])
    n_frames = result.n_frames - start
    group = result.group

    # --- épisodes de contact ------------------------------------------------
    # `new_per_frame[f, i]` = nombre de contacts DÉBUTANT à l'image f pour
    # l'agent i. C'est ce tableau qui permet ensuite de compter les contacts
    # subis pendant une traversée donnée, sans recalculer les contacts.
    n_agents = result.n_agents
    new_per_frame = np.zeros((n_frames, n_agents), dtype=np.int32)
    durations = []
    active = {}
    previous = set()

    for f in range(n_frames):
        g = start + f
        pairs = set(_contact_pairs(result.x[g].astype(np.float64),
                                   result.y[g].astype(np.float64),
                                   result.theta[g].astype(np.float64),
                                   group, d_contact, cos_min).tolist())
        for code in pairs - previous:
            active[code] = f
            new_per_frame[f, code // n_agents] += 1
            new_per_frame[f, code % n_agents] += 1
        for code in previous - pairs:
            durations.append((f - active.pop(code)) * dt_rec)
        previous = pairs
    for code, f0 in active.items():          # contacts encore ouverts à la fin
        durations.append((n_frames - f0) * dt_rec)

    total_time = n_frames * dt_rec
    rate = float(new_per_frame.sum() / 2.0 / (n_agents * total_time))

    # --- traversées de la section centrale ----------------------------------
    transits = _section_transits(result, start, section_mm, dt_rec, new_per_frame)
    per_transit = np.nan
    T0 = delta_T = np.nan
    if len(transits) >= 5:
        T = np.array([t for t, _ in transits], dtype=float)
        C = np.array([c for _, c in transits], dtype=float)
        per_transit = float(C.mean())
        if C.max() > C.min():
            slope, intercept = np.polyfit(C, T, 1)
            T0, delta_T = float(intercept), float(slope)

    area_cm2 = (result.model.length / MM_PER_CM) * (result.model.height / MM_PER_CM)
    return {
        "n_agents": n_agents, "seed": result.sim.seed,
        "k": n_agents / area_cm2,
        "contacts_per_ant_per_s": rate,
        "contacts_per_transit": per_transit,
        "n_transits": len(transits),
        "mean_duration": float(np.mean(durations)) if durations else np.nan,
        "median_duration": float(np.median(durations)) if durations else np.nan,
        "T0": T0, "delta_T": delta_T,
        "d_contact": d_contact,
    }


def _section_transits(result, start, section_mm, dt_rec, new_per_frame):
    """(durée, nombre de contacts) de chaque traversée complète de la section.

    Une traversée est un séjour continu de l'agent dans la bande centrale,
    entrée et sortie observées, sans repliement périodique au milieu. Les
    contacts comptés sont ceux qui DÉBUTENT pendant le séjour.
    """
    model = result.model
    lo = 0.5 * model.length - 0.5 * section_mm
    hi = 0.5 * model.length + 0.5 * section_mm
    out = []

    for i in range(result.n_agents):
        xi = result.x[start:, i].astype(np.float64)
        inside = (xi >= lo) & (xi <= hi)
        wrapped = np.abs(np.diff(xi)) > 0.5 * model.length
        n = len(xi)
        f = 1                                    # on veut voir l'ENTRÉE
        while f < n:
            if not inside[f] or inside[f - 1]:
                f += 1
                continue
            f0 = f
            broken = False
            while f + 1 < n and inside[f + 1]:
                if wrapped[f]:
                    broken = True
                    break
                f += 1
            if not broken and f + 1 < n:         # la SORTIE est observée
                duration = (f - f0 + 1) * dt_rec
                contacts = int(new_per_frame[f0:f + 1, i].sum())
                out.append((duration, contacts))
            f += 1
    return out


# ──────────────────────────────────────────────────────────────────────────
#  Structure spatio-temporelle et mélange
# ──────────────────────────────────────────────────────────────────────────

def density_spacetime(result, n_bins=50, warmup_frac=0.0):
    """Densité (fourmis/cm²) par tranche de `x` et par image.

    Renvoie (densites, x_centres, temps) avec `densites` de forme
    (n_images, n_bins). C'est le diagramme qui montre, s'il y en a, les ondes
    d'arrêt-redémarrage et la séparation de phases invoquée pour expliquer le
    plateau au-delà de la densité critique.
    """
    model = result.model
    start = _stationary_slice(result, warmup_frac)
    edges = np.linspace(0.0, model.length, n_bins + 1)
    bin_mm = model.length / n_bins
    area_cm2 = (bin_mm / MM_PER_CM) * (model.height / MM_PER_CM)

    x = result.x[start:]
    dens = np.empty((x.shape[0], n_bins))
    for f in range(x.shape[0]):
        dens[f] = np.histogram(x[f], bins=edges)[0] / area_cm2
    return dens, 0.5 * (edges[:-1] + edges[1:]), result.times[start:]


def _raw_segregation(y_frame, group, n_strips, height):
    """Polarisation moyenne, pondérée par l'occupation, sur des bandes en y."""
    idx = np.clip((y_frame / height * n_strips).astype(int), 0, n_strips - 1)
    num = den = 0.0
    for s in range(n_strips):
        m = idx == s
        npos = float(np.sum(group[m] > 0))
        nneg = float(np.sum(group[m] < 0))
        if npos + nneg:
            num += abs(npos - nneg)
            den += npos + nneg
    return num / den if den else np.nan


def segregation_index(result, n_strips=8, warmup_frac=0.4, stride=10,
                      n_shuffles=5, rng=None):
    """Mélange transverse, CORRIGÉ DU PLANCHER STATISTIQUE.

    L'indice brut est la moyenne, pondérée par l'occupation, de
    |n+ - n-| / (n+ + n-) sur des bandes horizontales : 0 si chaque bande est
    équilibrée, 1 si chaque bande est pure (voies nettes).

    Piège : cet indice brut n'est PAS nul pour un mélange parfait. Avec n agents
    dans une bande et des groupes tirés au hasard, |n+ - n-| / n vaut en moyenne
    ~sqrt(2/(pi.n)), soit 0.29 pour 7 agents par bande et 0.14 pour 30. L'indice
    brut décroît donc mécaniquement quand la densité monte, sans qu'aucun
    mélange supplémentaire n'ait lieu.

    On soustrait donc un PLANCHER mesuré : à chaque image, on rebat au hasard
    les étiquettes de groupe entre les agents présents (les positions, elles,
    sont inchangées) et on recalcule l'indice. C'est le null exact « positions
    observées, groupes indépendants de la position ».

    Renvoie un dictionnaire :
      `raw`      indice brut ;
      `null`     plancher (groupes rebattus) ;
      `excess`   raw - null, la seule quantité interprétable : 0 = mélange
                 indiscernable du hasard, > 0 = ségrégation réelle.
    """
    start = _stationary_slice(result, warmup_frac)
    h = result.model.height
    group = result.group
    if rng is None:
        rng = np.random.default_rng(0)

    raw, null = [], []
    for f in range(start, result.n_frames, stride):
        y = result.y[f]
        raw.append(_raw_segregation(y, group, n_strips, h))
        for _ in range(n_shuffles):
            null.append(_raw_segregation(y, rng.permutation(group), n_strips, h))

    raw_m = float(np.nanmean(raw)) if raw else np.nan
    null_m = float(np.nanmean(null)) if null else np.nan
    return {"raw": raw_m, "null": null_m, "excess": raw_m - null_m}
