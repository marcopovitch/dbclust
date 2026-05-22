# Leiden clustering — expériences et conclusions (session 2026-05-22)

## Problème de 10:36:44 (non résolu)

Événement Massif Central (45.6°N, 4.3°E), Ml~2, 6 picks manuels.

### Diagnostic précis
- Leiden (resolution=0.03, sigma=35s) fragmente les picks en 4 clusters géographiques :
  C14 (Nice/Côte d'Azur), C15 (Pyrénées), C17 (Massif Central/Alpes), C18 (Alsace)
- PyOcto trouve l'événement dans C17 (17-18 picks) mais associe 10 picks sur 9 stations
- ps_ratio = 1/9 = 0.11 < seuil 0.20 → rejeté par NLL
- Cause racine : C17 couvre tout le réseau France, PyOcto utilise le modèle haslach
  alors que l'épicentre est dans le Massif Central (modèle auvergne serait correct)
- Résidus trop grands avec haslach → picks incohérents retenus → mauvais ps_ratio

### Tests pick_match_tolerance sur le vrai cluster
| tol  | events | picks | ps_ratio | verdict       |
|------|--------|-------|----------|---------------|
| 5.0  | 1      | 11    | 0.11     | échec         |
| 1.5  | 1      | 9     | 0.14     | échec         |
| 0.8  | 1      | 7     | **0.20** | passe ✓       |
| 0.5  | 1      | 6     | 0.00     | trop strict   |
| 0.1  | 0      | —     | —        | rien trouvé   |

### Mécanismes tentés et abandonnés
1. **leiden_sigma_km** (poids factorisé exp(-dt/σ_t)*exp(-dist/σ_km)) — n'aide pas,
   crée plus de faux positifs avec sigma élevé
2. **leiden_ps_dt_max** (limite délai S-P pour le boost) — fragmente les clusters légitimes
3. **failed_cluster_merge_window** (fusion clusters PyOcto échoués) — pool trop hétérogène,
   PyOcto trouve 0 events même après fusion
4. **retry_low_ps_ratio_clusters** (relance PyOcto avec tolérance stricte) — PyOcto trouve
   0 events sur le pool de 60 picks, rien à améliorer
5. **leiden_min_stability=0.12** (dissolve vers HDBSCAN) — crée un méga-cluster HDBSCAN
   de 96 picks encore plus hétérogène
6. **moveout PP/SS actif** — supprime 1000+ edges, fragmente davantage

### Conclusion
Le cas 10:36 nécessiterait une sélection de modèle PyOcto adaptée à l'épicentre
estimé du cluster, pas à son centroïd de stations. Non faisable sans refonte architecturale.

## Paramètres stables (config dbclust-alceste-mac-test.yml)

```yaml
leiden_resolution: 0.03
leiden_edge_weight_scale: 35
leiden_ps_boost_factor: 500.0
leiden_min_edge_weight: 0.05
leiden_vp: 0.0   # moveout désactivé
leiden_vs: 0.0
leiden_hdbscan_fallback: true
leiden_min_stability: 0.0
min_cluster_size: 3
min_ps_ratio: 0.20
```

Résultat jobs 9+10 : 14 accepted / 4 rejected
Manquants : 09:02, 09:57, 10:11, **10:36**
Faux positifs : 08:59, 09:43
