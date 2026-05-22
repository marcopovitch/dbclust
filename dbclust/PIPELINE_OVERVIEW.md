# DBClust Pipeline Overview

## Architecture générale

DBClust détecte et localise les séismes à partir d'un catalogue de picks (arrivées P et S) en appliquant une chaîne de traitements par fenêtre temporelle glissante.

## Workflow par fenêtre temporelle

```
┌─────────────────────────────────────────────────────────────────┐
│  Fenêtre temporelle (10 min + overlap 250 s)                    │
│                                                                 │
│  1. LEIDEN (clustering principal)                               │
│     - Graphe de picks : arêtes pondérées par exp(-TT/σ)·p_i·p_j│
│     - PS-boost : poids × ps_boost_factor sur les paires P-S    │
│       de même station pour forcer leur cohésion                 │
│     - Optimisation CPM → communautés de picks cohérents        │
│     - Picks isolés (noise) → pool séparé                       │
│                                                                 │
│  2. PYOCTO (association fine, par cluster Leiden)               │
│     - Association des picks du cluster avec un modèle 1D        │
│     - Picks non-assignés → sous-clustering Leiden               │
│     - Noise Leiden → cluster supplémentaire soumis à PyOcto    │
│                                                                 │
│  3. PRE-NLL (filtres qualité pre-localisation)                  │
│     - station_score ≥ 5.0  (P:1.0, S:0.5, P+S:2.0 par station)│
│     - ps_ratio ≥ 0.2       (stations avec P et S / total)      │
│     - min_stations ≥ 5                                         │
│                                                                 │
│  4. NONLINLOC (localisation double pass)                        │
│     - Pass 1 : preloc PyOcto → NLL sur modèle de vitesse zonal │
│     - Nettoyage des picks (résidus, poids)                      │
│     - Pass 2 : relocalisation sur zone détectée                 │
│                                                                 │
│  5. POST-NLL (filtres qualité post-localisation)                │
│     - score ≥ 5.0                                              │
│     - ps_ratio ≥ 0.2                                           │
│                                                                 │
│  6. GESTION OVERLAP                                             │
│     - Events dont tous les picks sont avant la zone overlap     │
│       → sauvegarde dans le catalogue                            │
│     - Events straddling (picks dans la zone overlap)            │
│       → reportés à la fenêtre suivante pour complétion          │
└─────────────────────────────────────────────────────────────────┘
```

## Paramètres clés (Leiden)

| Paramètre | Valeur | Rôle |
|-----------|--------|------|
| `leiden_resolution` | 0.01 | Granularité CPM — faible = grandes communautés |
| `leiden_edge_weight_scale` | 35 s | Échelle σ de décroissance exponentielle des poids |
| `leiden_ps_boost_factor` | 2000 | Boost des arêtes P-S intra-station |
| `leiden_min_edge_weight` | 0.08 | Seuil de coupure des arêtes faibles |
| `max_search_dist` | 70 s | Distance TT maximale pour créer une arête |
| `time_window` | 10 min | Durée de la fenêtre de clustering |
| `overlap_window` | 250 s | Zone de chevauchement entre fenêtres |

## Notes

- **Leiden remplace HDBSCAN** pour toutes les étapes de clustering : fenêtre principale, zone de chevauchement arrière, et sous-clustering PyOcto.
- Le **cap dt_SP_max** (contrainte physique sur le délai S-P) est désactivé pour Leiden car le PS-boost gère la cohésion des paires P-S.
- La **stabilité CPM** (densité pondérée des arêtes internes) est calculée pour chaque cluster mais les filtres pre-NLL (station_score, ps_ratio) sont plus discriminants.
