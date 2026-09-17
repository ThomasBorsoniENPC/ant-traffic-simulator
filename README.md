# `simulateur/` — modèle d'agents pour le trafic bidirectionnel de fourmis

Simulateur repensé du modèle microscopique de trafic bidirectionnel de fourmis
d'Argentine (*Linepithema humile*) sur un pont étroit, d'après les expériences
de Poissonnier, Motsch, Gautrais, Buhl & Dussutour (2019, *eLife*).

La référence faisant autorité pour les équations est le **rapport de passation**
`Wrap Up Projet Fourmis/Wrap up bidirectional ant trafic flow/Rapport_passation/rapport_passation.tex`
(noyaux : annexe A ; paramètres : annexe B ; schéma numérique : annexe C).
Le rapport `Rapport existant LLM/rapport_LLM.tex` en donne la limite de champ
moyen (McKean–Vlasov) et les fermetures macroscopiques ; il est cité ici pour
les formes explicites des noyaux (§ « Noyaux d'interaction sans priorité »).

Ce dossier est **autonome** : il ne modifie aucun fichier existant du projet.

---

## 1. Ce que le modèle doit reproduire

Deux faits expérimentaux, à faire émerger **mécaniquement** :

1. le diagramme fondamental n'est pas en cloche : le flux `q(k)` croît puis
   atteint un **plateau** (« two-phase flow »), sans déclin jusqu'à 80 %
   d'occupation ;
2. l'écoulement reste **bien mélangé** : pas de formation de voies, contrairement
   aux piétons et aux fourmis légionnaires.

Les deux mécanismes revendiqués sont isolables dans le code :

| Mécanisme | Où | Effet revendiqué |
|---|---|---|
| séparation radiale / angulaire, freinage placé comme **taux** de relaxation | `braking.py` | plateau : `q = ξν_c k/(ν_c + β̄k)`, loi de Monod, jamais décroissante |
| **antennation** (attraction entre sens opposés à moyenne portée) | `forces.py`, terme `COMM` | mélange : `c_comm = 0` fait apparaître des voies |

Le rapport de passation rappelle une condition indispensable au plateau : le
freinage doit voir la densité **physique**. Si les agents se recouvrent sans
borne, la réponse linéaire `B ∝ ρ` se rompt et le flux cesse de saturer. C'est
le rôle de `contacts.py` — **désactivé par défaut ici** (voir §7).

---

## 2. Ce que contient (et ne contient pas) ce dépôt

Le dépôt contient le **code**, la **documentation**, le **rapport** et les
**figures** — environ 5 Mo. Il ne contient pas :

- **les vidéos et les trajectoires** (`*.mp4`, `*.npz`, 140 Mo). Git stocke
  chaque version d'un binaire en entier et pour toujours ; les versionner
  alourdirait le dépôt d'un facteur 30 pour rien. Ce n'est pas une perte :
  chaque run archivé conserve son `config.json`, qui suffit à le rejouer à
  l'identique (`python scripts/replay.py <config.json>`), après quoi les vidéos
  se régénèrent avec `viz.save_video`.
- **les trajectoires expérimentales**. Elles appartiennent à l'équipe de
  biologie et sortent du périmètre de ce dépôt. Les scripts qui les lisent
  (`analyze_real_fd.py`, `extract_params_from_data.py`,
  `fit_speed_distribution.py`) les cherchent via `--data-root`, la variable
  d'environnement `ANTSIM_DATA`, ou `../Data` — voir `scripts/_data_root.py`.

`donnees_publiees/elife2019_figure4.csv` est en revanche versionné : ce sont des
points relevés à la main sur une figure publiée, pas des données brutes.

---

## 3. Installation et lancement

```bash
cd simulateur
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Un environnement est déjà en place dans `.venv/` (Python 3.11, numba, numpy,
matplotlib, pandas, scipy, tqdm). `ffmpeg` sur le système donne des vidéos
`.mp4` ; sans lui, le rendu bascule automatiquement en `.gif`.

```bash
# un run + une image + une vidéo
.venv/bin/python scripts/quickstart.py

# rendu illustratif : vraies fourmis avec les pattes qui bougent
.venv/bin/python scripts/quickstart.py --sprite ant_real
# ... en teinte naturelle et gabarits variés, pour une image « comme en vrai »
.venv/bin/python scripts/quickstart.py --sprite ant_real --natural --size-jitter 0.16
# ... ou lapins, voitures de rallye, papis à canne
.venv/bin/python scripts/quickstart.py --sprite rabbit --agents 40

# variantes
.venv/bin/python scripts/quickstart.py --agents 120 --duration 45 --width 20 --seed 3
.venv/bin/python scripts/quickstart.py --preset report      # configuration du rapport
.venv/bin/python scripts/quickstart.py --preset minimal --no-video

# rejouer un run depuis son seul JSON, et vérifier qu'il est identique
.venv/bin/python scripts/replay.py out/quickstart/config.json --check

# tests
.venv/bin/python tests/run_tests.py
```

En bibliothèque :

```python
from antsim import ModelParams, SimParams, run, save_run
from viz import save_video

result = run(ModelParams(height=20.0, c_comm=0.0),        # pont large, sans antennation
             SimParams(num_agents=120, duration=60.0, seed=1))
save_run(result, "out/sans_antennation")
save_video(result, "out/sans_antennation/dynamics")
```

Ordre de grandeur : **30 s simulées, 60 agents ≈ 0,4 s** de calcul (après la
compilation Numba du premier appel, ~5 s, mise en cache ensuite). Le rendu vidéo
est plus lent que la simulation (~35 images/s).

### Rendu des agents

`--sprite` change uniquement l'apparence, jamais la dynamique ni les mesures :

| valeur | rendu |
|---|---|
| `ellipse` (**défaut**) | ellipse orientée — celui de toutes les figures d'analyse |
| `ant` | fourmi stylisée : gastre, thorax, tête, antennes, six pattes en trépied |
| `ant_real` | fourmi détaillée : gastre effilé, pétiole, mésosome et tête ombrés, yeux, mandibules, antennes coudées, six pattes attachées au thorax avec appui/retour asymétrique |
| `rabbit` | lapin, oreilles et bonds |
| `rally_car` | voiture de rallye vue de dessus, roues, aileron, gravillons |
| `grandpa` | papi vu de dessus, avec sa canne |

Deux options pour l'illustration, **toutes deux désactivées par défaut** :

- `--natural` donne à toutes les fourmis la même teinte brune (`#6d4a30`,
  celle d'une ouvrière de *Linepithema humile*), légèrement variée d'un
  individu à l'autre. On y gagne en réalisme et **on y perd l'information de
  groupe** : le bleu/rouge reste indispensable dès qu'on veut lire le mélange,
  donc la légende disparaît d'elle-même en mode naturel ;
- `--size-jitter 0.16` fait varier le gabarit de ±16 % d'un individu à l'autre.

Les deux sont **cosmétiques** : la dynamique ne change pas, tous les agents
restent des disques de rayon R identiques. Teinte et gabarit sont tirés d'un
générateur initialisé par la graine du run, donc deux rendus du même run
donnent exactement les mêmes fourmis.

**Taille à l'échelle** : l'échelle n'est pas réglée à l'oeil, elle est déduite
du sprite pour que le corps dessiné mesure `AGENT_LENGTH_MM = 3.2` mm — la
longueur de fourmi **mesurée sur les données de suivi** (boîte englobante
médiane 3.2 × 1.8 mm). Tous les rendus sortent donc à la même taille. À noter :
le modèle fait interagir des disques de rayon R = 1 mm, donc une fourmi dessinée
à sa taille réelle se recouvre visuellement plus que les disques du modèle — un
fait sur le modèle, pas un défaut du dessin. `--sprite-scale 0.63` dessine les
agents à la taille de leur disque d'interaction.

Les membres sont animés par la **distance parcourue**, pas par le temps : ils
accélèrent avec l'agent et se figent quand il s'arrête. `ant_real` va plus loin
et utilise une démarche à **appui/retour asymétrique** (60 % du cycle en appui,
le pied reculant en poussant, puis un retour aérien rapide avec levée) : c'est
cette asymétrie, et non la silhouette, qui distingue une vraie démarche d'un
balancement. Le rendu par sprite
coûte environ 40 % de temps de plus que l'ellipse (16 s contre 12 s pour 400
images à 50 agents) — c'est pour les présentations, pas pour les balayages.

---

## 4. Carte des modules

```
antsim/
  params.py      ModelParams / SimParams (dataclasses, JSON) + presets
                 + build_kernel_params -> KernelParams (NamedTuple Numba)
  kernels.py     profils scalaires : rampe, profil singulier, facteur de cône
  geometry.py    parois du canal (rectangle, analytique), normales, occlusion
  neighbors.py   cell list O(N) + champ de vision (cône, portée, occlusions)
  forces.py      champ d'ORIENTATION : objectif, paroi, évitement, antennation,
                 répulsion, dépassement, suivi, dégagement de paroi
  braking.py     dynamique RADIALE : nu_brake + relaxation exacte de u
  contacts.py    non-pénétration semi-dure (projection de position)  [option]
  boundaries.py  parois (spéculaire | glissement), extrémités (tore | injection)
  step.py        LE noyau compilé : un pas dt pour tous les agents
  engine.py      initialisation, boucle, RNG unique, RunResult
  io.py          sauvegarde/rechargement (config.json + trajectories.npz)
viz/
  render.py      image fixe et vidéo (bleu = +1, rouge = -1)
  sprites.py     silhouettes animées optionnelles (fourmi, lapin, voiture, papi)
  plots.py       figures d'analyse
scripts/
  quickstart.py     un run + une image + une vidéo
  replay.py         rejoue un run depuis son JSON et vérifie l'identité
  diagnose_wall.py  départage les lois de thigmotactisme sur trois critères
tests/
  run_tests.py   lanceur sans dépendance (compatible pytest également)
```

Dépendances : `params → kernels → geometry → neighbors → forces/braking/contacts
→ boundaries → step → engine → io`. Aucun cycle, aucune variable globale de
paramètre.

---

## 5. Le modèle

Unités internes : **1 unité de longueur = 1 mm, 1 unité de temps = 1 s**, fourmi
de rayon `R = 1`. Conversions vers l'article :
`k[fourmis/cm²] = ρ[fourmis/mm²] × 100` et `q[fourmis/cm/s] = 10 ρ U`.

Chaque agent porte une position `X_i`, une vitesse **scalaire** `u_i ≥ 0`
(composante radiale), un cap `θ_i` (composante angulaire) et un groupe
`g_i = ±1` (`+1` vers la source, en **bleu** ; `−1` vers le nid, en **rouge**).

**Radial** — équation (speed) du rapport, intégrée exactement sur le pas :

```
du/dt = (ν_c + ν_brake)(u* − u),     u* = ν_c ξ_i / (ν_c + ν_brake)
u ← u e^{−r dt} + u*(1 − e^{−r dt}),  r = ν_c + ν_brake
```

Inconditionnellement stable, sans dépassement, et `u ≥ 0` est préservé
exactement. `ν_brake` est un **taux** : il augmente la vitesse de relaxation
*et* abaisse la cible, au lieu de soustraire une capacité — c'est là que se joue
le plateau (`braking.py`).

**Angulaire** — équation (heading) du rapport, intégrée exactement :

```
ω = arg(F),   F = k_obj·(g,0) + W(paroi) + Σ_j K(i,j)
θ̇ = λ sin(ω − θ)      ⇒   tan((θ−ω)/2) ← tan((θ−ω)/2) e^{−λ dt}
```

Trois choix de `λ`, voir §6.

**Noyaux** (annexe A du rapport ; formes explicites dans `rapport_LLM.tex`).
Chaque terme est (magnitude) × (direction), avec deux profils de distance :
*singulier* `(ℓ−ρ)/ρ` pour les comportements de contact, *rampe*
`(ρ−ℓ_min)/(ℓ_max−ℓ_min)` pour les attractions à distance, multipliés par un
facteur de cône `[(ε − (1−cos α))/ε]_+`.

| Terme | Groupes | Support | Direction |
|---|---|---|---|
| freinage (→ `ν_brake`) | tous | `ρ ≤ ℓ_b`, cône avant | — (radial) |
| évitement | opposés | `ρ ≤ ℓ_avoid`, cône avant | `±σ⊥` (désengageant) |
| antennation | opposés | `ℓ_avoid < ρ ≤ ℓ_comm`, cône avant | `+σ` (attraction) |
| répulsion latérale | même | `ρ ≤ ℓ'_avoid`, latéral | `−σ` |
| dépassement | même | `ρ ≤ ℓ_over`, cône avant, `u_j ≤ αξ_i` | `σ⊥` vers l'objectif |
| suivi | même | `ℓ_f < ρ ≤ ℓ'_f`, cône avant | `+σ` |
| paroi (thigmotactisme) | — | `ℓ_min < d ≤ ℓ_max`, cap vers la paroi | `σ_wall` |

Le terme de paroi est pondéré par la composante **tangentielle** du cap
(`|e_θ⊥·σ_wall| = √(1−(e_θ·σ_wall)²)`) : maximal quand l'agent longe la paroi,
nul quand il la vise de face. C'est la forme corrigée du rapport ; le premier
jet pondérait par la composante normale.

---

## 6. Paramètres

Tous dans `ModelParams` / `SimParams`, sérialisés intégralement avec la graine.
Les valeurs par défaut sont celles de la **configuration finale** du rapport
(annexe B), et non celles, antérieures à la calibration, du `main.py` d'origine.

| Groupe | Paramètre | Défaut | Valeur historique | Unité |
|---|---|---|---|---|
| Géométrie | `length`, `height`, `radius` | 100, 10, 1 | idem | mm |
| Perception | `angle_vision`, `l_vision` | π, 8 | idem | rad, mm |
| Radial | `nu_cruise` | 20 | 15 | 1/s |
| | `xi_cruise` | 17 | 10 | mm/s |
| | `xi_spread` | 0.2 | idem | — |
| | `nu_wall_stop` | 1e10 | idem | 1/s |
| Angulaire | `angular_mode` | `field_norm` | — | — |
| | `k_obj` | 10 | idem | — |
| | `lambda_theta` | 10 | — | 1/s |
| | `lambda_max` | **8** | 50 | 1/s |
| | `force_ref` | 10 | — | — |
| | `sigma_theta` | 0.3 | absent | rad·s^(−1/2) |
| Freinage | `l_brake, eps_brake, c_brake, alpha_brake` | 3, 0.7, **300**, 4 | …, **1e3**, … | mm, —, —, — |
| Évitement (opp.) | `l_avoid_steer, eps_avoid_steer, c_avoid_steer` | 2.3, 0.7, 50 | idem | |
| Répulsion (même) | `l_repulse, eps_repulse, c_repulse` | 1.5, 2.0, **40** | …, **1e5** | |
| Antennation | `l_comm_min, l_comm, eps_comm, c_comm` | `l_avoid_steer`, 5, 1.0, **12** | …, **20** | |
| Dépassement | `l_overtake, eps_overtake, c_overtake, alpha_overtake` | 4, 0.7, 15, 10 | idem | |
| Suivi | `l_follow1, l_follow2, eps_follow, c_follow` | 3, 8, 0.7, 30 | idem | |
| Paroi | `wall_law` | **`standoff`** | `tangential` | — |
| | `c_wall` | **2** | 10 | — |
| | `l_wall_standoff` | 1.0 | — | mm |
| | `l_wall_min, l_wall_max` | 0.5, 4 | idem | mm |
| Non-pénétration | `contact_gap, contact_hardness, …` | 1.0, 0.4 | — | mm, — |
| Dégagement paroi | `c_wall_escape, l_wall_stuck, wall_escape_asym` | 20, 1.5, 0.2 | — | |
| Simulation | `dt, record_fps, num_agents, seed` | 0.01, 20, 50, explicite | idem | s, 1/s |

**Ce qui est calibrable en premier** (d'après le rapport) : `ξ` et `ν_c` sont
fixés par la vitesse libre ; `β̄` — l'intégrale effective du noyau de freinage —
est le **seul** nombre qui contrôle le plateau, et se calibre sur lui. Les
intensités `c_*` ne sont pas identifiables séparément à partir des seules
données agrégées.

**Deux écarts connus au rapport, laissés tels quels et signalés dans le code** :
`alpha_overtake = 10` neutralise la porte de vitesse du dépassement (une valeur
dans `(0,1)` la restaure) ; la porte lisse de l'annexe A est désactivée
(`overtake_smooth_gate = False`).

### Presets

| Preset | Contenu |
|---|---|
| `full` (défaut) | tous les termes actifs, non-pénétration et dégagement de paroi **désactivés**, bruit angulaire 0.3 |
| `minimal` | sans suivi ni dépassement — le modèle minimal recommandé par la relecture critique du projet (§6.1) |
| `report` | **la configuration finale du rapport de passation**, celle qui produit le diagramme fondamental publié : `wall_law="tangential"` et `c_wall=10`, `field_norm`, `sigma_theta=0`, `c_comm=12`, réflexion spéculaire, non-pénétration semi-dure (`g=1.0`, `κ=0.4`) et dégagement de paroi actifs |

---

## 7. Dynamique angulaire : trois modes

L'équation du rapport est `θ̇ = |F| sin(ω − θ)`. Écrite ainsi, `|F|` joue **deux
rôles** : elle fixe la direction visée *et* la vitesse à laquelle on y va. Un
terme de répulsion intense rend donc le virage quasi instantané, ce qui oblige à
plafonner `|θ̇|` — plafond qui mélange alors des échelles hétérogènes (c'est le
défaut identifié dans le code de référence, où la constante de non-pénétration
`c = 1e5` saturait systématiquement le plafond ; elle vaut 40 ici).

| `angular_mode` | `λ` | Remarque |
|---|---|---|
| `field_norm` (**défaut**) | `\|F\|`, plafond `\|Δθ\| ≤ lambda_max·dt` | équation du rapport : le simulateur reproduit le modèle de référence |
| `fixed` | `lambda_theta` | les `c_*` ne fixent plus qu'une **priorité relative** (la direction) ; `lambda_theta` est un taux de virage calibrable sur la distribution des vitesses angulaires mesurées, et **est** le plafond. Extension propre du Persistent Turning Walker (Gautrais et al. 2009) — variante recommandée pour la phase de calibration |
| `saturating` | `lambda_max·\|F\|/(\|F\| + force_ref)` | réponse graduée à l'urgence, bornée par construction |

Une logique de **priorité stricte** (le comportement le plus urgent l'emporte)
n'a pas été retenue : elle introduit des discontinuités dans le champ de dérive,
ce qui casse l'hypothèse de noyaux lipschitziens de la limite champ moyen
(`rapport_LLM.tex`) — donc le pont vers la partie macroscopique.

---

## 8. Choix techniques

### Paramètres et couche compilée

Aucune fonction `@njit` ne lit un paramètre comme variable globale : Numba le
capturerait à la compilation, interdisant tout balayage sans recompiler. Tout
circule via `KernelParams`, un `NamedTuple` construit par `build_kernel_params`
et passé **en argument**. Les choix discrets (géométrie, mode angulaire,
condition au bord) sont des **entiers branchés dans les fonctions**, jamais des
pointeurs de fonction globaux. `dt` est également un argument.

### Reproductibilité : un seul générateur

Tout l'aléa vient de `np.random.default_rng(sim.seed)`. Le code de référence
tirait des nombres aléatoires *à l'intérieur* des fonctions `@njit` ; or Numba
possède en mode nopython un état aléatoire **propre**, distinct de celui de
NumPy, d'où l'obligation de seeder deux générateurs. Le problème a été supprimé
plutôt que géré :

- les départages de directions dégénérées utilisent `tie_sign`, un signe ±1 tiré
  une fois par agent à l'initialisation ;
- le tore conserve `y` (le code de référence le rééchantillonnait à chaque
  ré-entrée — voir §8) ;
- le bruit angulaire et les tirages de réinjection sont **pré-tirés par bloc**
  côté NumPy et passés au noyau.

`tests/test_reproducibility.py` vérifie qu'aucun `np.random` ne subsiste dans du
code compilé, que deux runs de même graine sont identiques au bit près, et qu'un
run se rejoue depuis son seul `config.json`. `seed_numba` est conservé et appelé
comme filet de sécurité.

Les tirages sont consommés à chaque sous-pas, que le bruit soit actif ou non :
la séquence aléatoire ne dépend donc pas des options du modèle, ce qui rend les
**ablations comparables à graine égale**.

### Sauvegarde

`config.json` (tous les paramètres + graine + version du code) et
`trajectories.npz` (float32 compressé). Le CSV du code de référence pesait une
vingtaine de mégaoctets par run ; `antsim.to_dataframe` reste disponible pour
qui veut un format tabulaire.

### Conditions aux bords

- **Parois** : `specular` (défaut, la composante normale du cap est *renversée*)
  ou `slide` (elle est *retirée*, l'agent longe la paroi). Les deux annulent le
  flux normal. Le rapport retient la réflexion spéculaire, qui « évite le
  collage aux parois qu'induit une condition de glissement à haute densité ».

  La correction de cap est **stabilisée** par deux réglages, actifs par défaut.
  Sans eux, la réflexion est un saut instantané — jusqu'à π en un pas, mesuré à
  0.195 rad en médiane et 2.65 rad au maximum, contre 0.034 rad pour le braquage
  libre — qui échappait au plafond angulaire (72 % des réflexions le
  dépassaient) et entretenait un chattering : 44 % des contacts successifs d'un
  même agent survenaient en moins de 0.05 s, le dixième à deux pas d'intervalle.

  - `cap_wall_turn` : la correction de cap passe sous le même `lambda_max` que
    la dynamique angulaire. Deux vitesses de rotation maximales différentes dans
    un même modèle n'auraient pas de sens.
  - `wall_cut_normal_speed` : la composante normale *sortante* de la vitesse est
    retirée, `u ← u·√(1 − (e_θ·n)²)`. Un agent face à la paroi s'arrête et tourne
    sur place ; un agent quasi tangent garde sa vitesse. La relaxation ramène `u`
    vers `ξ` dès le cap dégagé : c'est un amortissement transitoire.

  Les deux agissent sur des grandeurs distinctes — le plafond écrase l'amplitude
  du saut (2.65 → 0.65 rad), la coupure réduit la *fréquence* des contacts
  (0.36 → 0.30 par agent et par seconde). Le non-flux reste garanti sans
  condition : il vient du repositionnement de `y` à chaque pas, pas du cap. Ni
  collage ni condensation (temps à moins de 0.5 mm du mur 0.089 → 0.088, écart-
  type transverse inchangé). Voir `resultats/paroi_stabilisation/`.
- **Extrémités** : `torus` (défaut, `x mod L`, `y` conservé — la vision, elle,
  reste bloquée aux extrémités : semi-périodicité) ou `inflow` (l'agent sortant
  est réinjecté à l'entrée de son groupe, `N` constant).

### Performance

- parois **analytiques** pour le rectangle : le code de référence balayait 250
  points de paroi par agent et par pas pour trouver le point le plus proche,
  c'est `min(y, H−y)` ;
- occlusion visuelle par les parois **exacte** (le rectangle est convexe), au
  lieu d'un échantillonnage en 20 points ;
- `cell list` O(N) de maille `l_vision`, équivalente au balayage O(N²) — vérifié
  voisin par voisin, indices triés compris, par `tests/test_neighbors.py` ;
- aucune allocation dans la boucle par agent (tampons alloués une fois par pas).

---

## 8. Écarts au code de référence, et validation

Le simulateur n'est pas une transcription : quatre écarts sont assumés.

1. **Le tore conserve `y`.** Le code de référence rééchantillonnait `y`
   uniformément à chaque ré-entrée périodique, ce qui détruit les corrélations
   transverses et injecte de l'aléa dans la boucle compilée. Conserver `y` est
   plus fidèle au rapport (« positions taken mod L »)  — mais **révèle une
   condensation contre une paroi** que le rééchantillonnage masquait (voir
   ci-dessous). C'est précisément le blocage décrit par le rapport (« a queue
   builds up along the wall ») et que le **dégagement de paroi** corrige.
2. **Départages déterministes** au lieu de tirages à pile ou face (§7).
3. **Mode angulaire** exposé (défaut `field_norm` = équation du rapport).
4. **Non-pénétration et dégagement de paroi désactivés par défaut**, alors que
   la configuration finale du rapport les active. Choix délibéré : ces deux
   mécanismes sont arrivés tard dans le projet et doivent pouvoir être évalués,
   pas subis. `--preset report` les réactive.

**Validation chiffrée** (vitesse scalaire moyenne sur la seconde moitié du run,
moyenne sur 3 graines, 20 s, pont 100 × 10, `c_comm = 20`, `sigma_theta = 0`,
`angular_mode = field_norm`, `wall_law = "tangential"` — c'est-à-dire à
configuration identique à celle du code de référence) :

| N | k (cm⁻²) | référence | ce code, sans dégagement | ce code, avec dégagement de paroi |
|---|---|---|---|---|
| 5 | 0.5 | 15.79 | 15.68 (−0.7 %) | — |
| 40 | 4 | 11.54 | 9.56 (−17 %) | 10.53 (−9 %) |
| 80 | 8 | 9.68 | 7.72 (−20 %) | 9.33 (−3.6 %) |
| 120 | 12 | 8.60 | 6.95 (−19 %) | 8.57 (−0.3 %) |

En écoulement libre les deux codes coïncident à 0,7 % près. L'écart intermédiaire
est entièrement imputable à l'écart n°1 : sans dégagement, 39 % des agents se
trouvent à moins d'1 mm d'une paroi (contre 27 % dans la référence, et 18 % avec
dégagement), et cette accumulation freine l'écoulement. Avec le dégagement de
paroi du rapport, l'accord redevient de quelques pour cent.

**Ce que cela dit du modèle : le thigmotactisme est un piège absorbant.**
La condensation contre une paroi n'est ni un effet de foule ni un artefact
numérique — c'est le terme de paroi lui-même. Trois mesures l'établissent :

- un agent **isolé**, sans aucun voisin, parti du centre, passe **98 %** de son
  temps à moins d'1 mm d'une paroi (`c_wall = 0` : 0 %) ;
- la fraction au mur **ne se stabilise pas** : 24 % à 2 s, 38 % à 10 s, 40–46 %
  au-delà de 40 s. Sans thigmotactisme elle reste à 3–6 % (une distribution
  uniforme en donnerait 20 %) ;
- elle dépend de la largeur du pont à densité linéique égale : **47 / 36 / 23 %**
  pour W = 5 / 10 / 20 mm — la portée du terme est fixe quand la demi-largeur
  varie. C'est une **prédiction falsifiable** : l'eLife fait varier la largeur et
  mesure les profils transverses.

Mécanisme. La force de paroi n'est **jamais répulsive** : elle est dirigée vers
la paroi si `e_θ·σ_wall ≥ 0`, nulle sinon. Rien ne ramène jamais vers le centre.
Son profil radial **croît avec la distance** : nul à `ℓ_min = 0.5`, maximal à
`ℓ_max = 4`. Sur un pont de 10 mm cela fait :

- une bande active `[0.5, 4] ∪ [6, 9.5]` qui couvre **70 % de la largeur**, ne
  laissant qu'un couloir central neutre de 2 mm ;
- une intensité maximale de 10, soit **exactement `k_obj`** : le cap visé est à
  45° vers la paroi ;
- un maximum atteint **juste à côté de l'axe médian**, qui devient une
  séparatrice instable ;
- une pondération par la composante **tangentielle**, donc maximale pour un
  agent qui va tout droit le long du pont — c'est-à-dire pour tout le monde.

Autrement dit la bande de paroi est un **puits dont la force de rappel augmente
avec la distance de fuite** : plus l'agent tente de s'en éloigner, plus il est
tiré en arrière, jusqu'à `ℓ_max`. C'est l'inverse d'un suivi de bord de portée
de contact. Une fois collé, la réflexion spéculaire rend son cap tangentiel —
soit précisément l'état de traction maximale vers la paroi.

L'interpénétration et les oscillations de cap que l'on observe au bord en sont
la **conséquence**, pas la cause : une fois 40 % des agents dans une bande de
1 mm, les sens opposés se croisent frontalement là où seuls des termes *soft*
s'opposent au recouvrement (distance minimale entre agents : 0.02 mm avec
`ℓ_max = 4`, 0.98 mm sans thigmotactisme). Le dégagement de paroi et la
projection de non-pénétration agissent sur cette conséquence.

Le mode angulaire n'est pas en cause : `field_norm` / `fixed` / `saturating`
donnent 36 / 40 / 42 % au mur. (`fixed` réduit d'un tiers l'oscillation du cap,
mais ne change pas l'entassement.)

### Trois lois de thigmotactisme

Le paramètre `wall_law` sélectionne la loi ; `scripts/diagnose_wall.py` les
départage sur le cahier des charges du comportement voulu : *un agent seul longe
la paroi sans s'y écraser ; deux agents de sens opposés qui se rencontrent
contre la paroi se désengagent ; pas de condensation en foule* — le tout sans
casser le mélange, qui est le résultat principal du modèle.

| `wall_law` | Force | Comportement |
|---|---|---|
| `tangential` | attraction pondérée par la composante tangentielle du cap, nulle sous `l_wall_min` | loi du rapport. Maximale quand l'agent longe déjà la paroi ⇒ état absorbant |
| `outward_damping` | ne retient que l'agent qui s'éloigne | corrige la pondération, mais reste unidirectionnelle : ne provoque jamais le suivi de bord, se contente de geler `y` |
| `standoff` (**défaut**, `c_wall = 2`) | rappel vers `l_wall_standoff` : **répulsif en deçà**, attractif au-delà | produit une **préférence de bord sans écrasement** : l'agent n'est plus plaqué contre la paroi et rien ne l'y retient si un voisin l'écarte |

Mesures (N = 80, 60 s, 3 graines ; `%écrasé` = à moins de 0.5 mm de la paroi,
soit enfoncé dedans puisque `R = 1` ; ségrégation : 0 = mélangé, 1 = voies) :

| Configuration | agent seul : `d` finale | ségrégation | %écrasé | vitesse |
|---|---|---|---|---|
| `tangential`, `c_wall=10` (rapport) | **0.47** (écrasé) | 0.29 | 0.30 | 8.63 |
| `outward_damping` | 1.00 mais **aucune capture** | 0.23 | 0.15 | 8.50 |
| `standoff`, `c_wall=10` | 1.00 | **0.92** (voies !) | 0.12 | 13.42 |
| `standoff`, `c_wall=5` | 1.00 | 0.69 | 0.20 | 11.84 |
| **`standoff`, `c_wall=2`** | **1.00** | **0.32** | **0.11** | **9.61** |
| aucun thigmotactisme | — | 0.29 | 0.02 | 10.03 |
| `tangential`, `c_comm=0` | — | 0.96 | 0.34 | 14.13 |

Trois lectures :

1. **L'antennation tient son rôle, quelle que soit la loi de paroi** : la
   couper fait passer la ségrégation de 0.29 à 0.96. Le résultat principal du
   modèle est robuste au choix débattu ici.
2. **`standoff` à `c_wall = 10` casse le mélange** (0.92) : deux distances de
   consigne stables, une par paroi, font deux voies qui l'emportent sur
   l'antennation. `c_wall = 10` avait été calibré pour la loi `tangential`, où
   la force est fortement atténuée (rampe faible près de la paroi × facteur
   tangentiel) ; la loi `standoff` applique `c_wall` plein à l'écart de
   consigne, et doit donc être réétalonnée.
3. **`standoff` à `c_wall = 2` satisfait les trois critères, et c'est la
   configuration RETENUE PAR DÉFAUT** : suivi de bord à une distance stable de
   1 mm (le flanc touche la paroi, le corps ne la traverse pas), mélange
   identique au rapport (0.32 contre 0.29), trois fois moins d'agents écrasés,
   plus de superposition (distance minimale entre agents 0.88 mm contre 0.01),
   et un écoulement légèrement plus rapide.

**Pourquoi il a fallu changer de signe, et pas seulement annuler.** Le cahier
des charges était : « une fois arrivé suffisamment près du mur, celui-ci
n'attire plus, pour laisser l'agent libre ». Annuler la force près de la paroi
ne suffisait pas — c'était **déjà** le cas sous `l_wall_min = 0.5` — parce que
tout autour la force pointait encore vers le mur : la zone morte était un fond
de puits, pas une sortie. Il fallait que la force devienne **répulsive** en
deçà de la distance de consigne. De même, supprimer la pondération tangentielle
était nécessaire (un agent qui longe déjà la paroi n'a aucune raison d'être
tiré vers elle) mais pas suffisant : la loi `outward_damping`, qui ne fait que
cela, gèle `y` sans jamais produire de suivi de bord.

`--preset report` restaure `wall_law = "tangential"` et `c_wall = 10` pour
reproduire les figures du rapport de passation.

Limite connue, commune aux trois lois : dans une configuration **exactement**
symétrique (deux agents à la même ordonnée, de même signe de départage) les
braquages d'évitement s'annulent et la paire se bloque. C'est un cas de mesure
nulle — dès 0.2 mm de décalage les trois lois croisent proprement — mais c'est
lui que le *dégagement de paroi* traite.

## 10. Ce qui n'est pas encore là

**Phase 2 — observables** : densité / vitesse / flux locaux, diagramme
fondamental mesuré comme dans l'eLife (comptage de passages de ligne + fenêtre
de 1 cm², moyenné sur plusieurs sections), ajustement de la loi de Monod, indice
de ségrégation latérale, taux de contacts inter-groupes.

**Phase 3 — extensions**, dans l'ordre de priorité indiqué par la relecture
critique du projet :

- **table d'ablations** systématique (le livrable le plus important) ;
- **antennation à coût temporel** (point ⭐ B.1 du backlog) : le terme `COMM`
  actuel est une attraction angulaire *sans coût temporel*, alors que l'eLife
  mesure `T = T₀ + C·ΔT` avec `T₀ ≈ 0.95 s`, `ΔT ≈ 0.24 s` et `C = 0.61 k`.
  Variante recommandée : une minuterie par agent (`t_ant`), posée à `ΔT` à
  chaque *nouveau* contact frontal avec un opposé, forçant `u → 0` tant qu'elle
  court. Point d'insertion : un état supplémentaire par agent dans `engine.py`,
  décrémenté et lu dans `braking.py` ;
- balayages (densité, largeurs 5 / 10 / 20 mm, intensité d'antennation) ;
- géométries alternatives (goulots, bosse) — la marche à suivre est en tête de
  `geometry.py` ;
- règles de priorité (manœuvrabilité + agent fantôme), anticipation balistique
  des collisions, pont vers la version macroscopique.

---

## 11. Tests

```bash
.venv/bin/python tests/run_tests.py            # 29 tests, ~2 s
.venv/bin/python tests/run_tests.py kernels    # un sous-ensemble
```

- `test_kernels` : supports, bornes et monotonie des profils de force ;
- `test_neighbors` : cell list ≡ référence O(N²), voisin par voisin ;
- `test_reproducibility` : même graine ⇒ identité bit à bit ; rejouabilité
  depuis le JSON ; absence de tirage aléatoire dans le code compilé ;
- `test_invariants` : `u ≥ 0`, agents dans le domaine, effectifs conservés,
  absence de NaN, convergence de la relaxation radiale vers l'équilibre de
  Monod, absence de dépassement de la relaxation angulaire, et deux contrôles
  de bon sens (un agent isolé garde sa vitesse de croisière ; la foule ralentit
  l'écoulement).
