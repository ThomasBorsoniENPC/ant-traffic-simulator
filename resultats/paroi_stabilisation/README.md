# Stabilisation de la paroi : plafonner la réflexion et couper la vitesse sortante

```bash
python scripts/run_fd.py --densities 75 80 85 90 95 100 105 110 115 120 125 130 \
    135 140 145 150 --seeds 3 --duration 40 --height 10 \
    --set length=50 --set lambda_max=8 --out resultats/lambda8_stab_50x10
```

## Le problème

`apply_wall` renversait la composante normale du cap d'un seul coup. C'est un
SAUT instantané, jusqu'à pi, qui échappait entièrement au plafond angulaire
`lambda_max` — lequel ne s'applique qu'à la loi de braquage. Mesuré à 100 Hz
(un enregistrement par pas), pont 50 x 10, N = 100 :

| | valeur |
|---|---|
| \|Δθ\| médian au contact, en UN pas | 0.195 rad |
| \|Δθ\| max au contact | 2.647 rad (152°) |
| pour comparaison, \|Δθ\| médian hors contact | 0.034 rad |
| réflexions dépassant le plafond `lambda_max = 8` | **72 %** |
| contacts successifs d'un même agent en < 0.05 s | **44 %** (p10 = 0.02 s, deux pas) |

Un agent rebondissait donc contre la paroi en quelques pas de temps, avec des
sauts de cap dix fois plus grands que son braquage libre.

## La correction, en deux morceaux indépendants

`cap_wall_turn` (défaut `True`) — la correction de cap est soumise au même
`lambda_max` que la dynamique angulaire. Avoir deux vitesses de rotation
maximales différentes dans le même modèle n'aurait pas de sens.

`wall_cut_normal_speed` (défaut `True`) — la composante normale SORTANTE de la
vitesse est retirée : `u <- u * sqrt(1 - (e_theta . n)^2)`. Un agent face à la
paroi s'arrête et tourne sur place ; un agent quasi tangent garde sa vitesse.
Sans cela, le plafond seul laisse l'agent pousser dans la paroi pendant tout
son demi-tour. La relaxation ramène `u` vers `xi` dès le cap dégagé : c'est un
amortissement transitoire, pas une pénalité.

Le non-flux reste garanti sans condition — il vient du repositionnement de `y`
dans le canal à chaque pas, indépendamment du cap.

## Effet (λ_max = 50, N = 100, 30 s, 100 Hz)

| | brut | plafond | coupure | les deux |
|---|---|---|---|---|
| \|Δθ\| max au contact | 2.647 | 0.630 | 2.551 | **0.649** |
| contacts /agent/s | 0.36 | 0.35 | 0.30 | **0.30** |
| enchaînés < 0.05 s | 0.436 | 0.358 | 0.414 | **0.358** |
| temps à moins de 0.5 mm du mur | 0.089 | 0.098 | 0.088 | **0.088** |
| écart-type de y (mm) | 2.73 | 2.78 | 2.73 | **2.73** |
| vitesse u | 5.50 | 5.56 | 5.51 | 5.51 |

Les deux morceaux agissent sur des grandeurs différentes : le plafond écrase
l'amplitude du saut, la coupure réduit la FRÉQUENCE des contacts. Avec
`lambda_max = 8`, les contacts enchaînés tombent à **0.153** (contre 0.436) et
\|Δθ\| max à 0.238 rad.

**Pas de condensation** : le temps passé à moins de 0.5 mm du mur et l'écart-type
transverse sont inchangés. C'était le risque principal — un agent qui tourne
lentement reste plus longtemps au contact — et il ne se matérialise pas.

## Effet sur le diagramme fondamental (λ_max = 8)

Sur k >= 20, le flux moyen passe de 10.60 à 10.95, soit **+3.3 %**, et le
maximum glisse de k = 22 à k = 29 : le plateau s'aplatit encore. Le modèle reste
en régime de saturation, sans cloche.
