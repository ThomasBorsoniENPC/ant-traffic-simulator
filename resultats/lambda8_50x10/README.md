# Plafond angulaire à 8 : diagramme fondamental, domaine 50 x 10 mm

```bash
python scripts/run_fd.py --densities 75 80 85 90 95 100 105 110 115 120 125 130 \
    135 140 145 150 --seeds 3 --duration 40 --height 10 \
    --set length=50 --set lambda_max=8 --out resultats/lambda8_50x10
# référence, identique sauf --set lambda_max=50 --out resultats/lambda50_50x10
```

Domaine 50 x 10 mm = 5 cm², donc **k = N/5** : k = 15 à 30 par pas de 1, soit
N = 75 à 150 par pas de 5. 3 graines par point, 40 s par run, 48 runs par
balayage (~2.5 s/run).

## Résultat : le plafond ne change presque rien

| k | q (λ=8) | q (λ=50) | Δ |
|---|---|---|---|
| 20 | 10.43 | 10.33 | +0.10 |
| 22 | **10.74 (max)** | 10.50 | +0.24 |
| 25 | 10.67 | 10.35 | +0.33 |
| 28 | 10.54 | 10.48 | +0.06 |
| 30 | 10.58 | 10.33 | +0.25 |

Sur k >= 20, q vaut 10.60 en moyenne contre 10.35 : **+2.4 %**, à peine plus
que la dispersion entre graines. Les deux courbes plafonnent au même endroit,
avec un maximum à k = 22 (λ=8) contre k = 23 (λ=50), puis une décroissance de
1.5 % seulement jusqu'à k = 30. Aucune cloche, dans aucun des deux cas.

Le plateau tombe sur `q ~ 10.5`, à comparer au `q_j = 10.2` de Poissonnier
et al. — cohérent avec ce que donnait déjà la géométrie 100 x 10.

## Pourquoi : le plafond est presque inerte

En mode `field_norm`, `lambda = |F|` et `lambda_max` n'est qu'un PLAFOND sur
`|dtheta|`. Or le taux de virage reste la plupart du temps bien en dessous de 8 :

| `lambda_max` | \|dθ/dt\| médian | p90 | p99 | pas au plafond |
|---|---|---|---|---|
| 8 | 2.26 | 5.93 | 9.34 | **3.6 %** |
| 50 | 2.20 | 6.17 | 11.83 | 0.0 % |

(mesuré à k = 25, entre images à 20 Hz)

Passer de 50 à 8 ne fait donc mordre le plafond que sur 3.6 % des pas, et
seulement dans la queue de la distribution. C'est trop peu pour déplacer le
diagramme.

Les maxima (~40 rad/s) ne viennent PAS de la loi de braquage : ils viennent des
réflexions spéculaires sur la paroi et des ré-injections aux extrémités, qui
sont des sauts de cap non soumis au plafond.

## Portée

`lambda_max` n'a pas été changé dans `ModelParams` : il est passé en surcharge
et figure donc dans le `config.json` de chaque run. Le défaut reste 50.
