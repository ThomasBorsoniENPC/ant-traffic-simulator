# Rendus illustratifs

Vidéos produites avec `--sprite` (cf. README principal, section « Rendu des
agents »). Même run à chaque fois : 60 agents, 30 s, graine 0, configuration par
défaut.

| fichier | rendu |
|---|---|
| `ant_60.mp4` | fourmi stylisée, pattes en gris |
| `ant_real_60.mp4` | fourmi détaillée, à la taille mesurée (corps 3,2 mm) |
| `ant_natural_60.mp4` | idem, teinte brune naturelle et gabarits variés (±16 %) |
| `ant.mp4`, `rabbit.mp4`, `rally_car.mp4`, `grandpa.mp4` | les cinq rendus sur un run de 50 agents |

Ces rendus sont **cosmétiques** : ils ne changent ni la dynamique ni les
mesures. Le mode `natural` fait perdre l'information de sens de marche
(plus de bleu/rouge), donc il est à réserver à l'illustration.

> **Note.** Ces runs datent d'avant le retrait du terme d'amortissement de
> paroi (`c_wall_damping`). Leurs `config.json` ne se rechargent donc plus
> (`load_config` refuse les champs inconnus, par construction). Les vidéos
> restent valables comme illustration ; pour des runs rejouables, relancer
> le script de la section « Reproduction ».
