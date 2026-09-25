# Plan: Consolidation de picks de réserve — complétion S uniquement (v6)

## Contexte

La v1 validait les picks de réserve contre le foyer préloc PyOcto/Leiden
(circularité, modèle TT grossier), laissait `min_station_with_P_and_S` être
satisfait sans distinction par des picks sous le seuil d'import, et n'avait
pas de règle anti-ambiguïté. Les itérations v2/v3 ont exploré une complétion
symétrique P+S avant de converger sur la v4 : **on ne complète jamais un P
manquant, uniquement un S manquant**.

La v5 intègre la revue de code du 2026-09-25 (vérifiée ligne par ligne
contre le code courant). La v6 intègre deux décisions utilisateur
supplémentaires. Changements structurants vs v4 :

1. **Gate pre-NLL : bypass, pas de comptage provisionnel** (nouvelle Phase
   3) : `min_station_with_P_and_S` est appliqué DEUX fois — pre-NLL dans
   `Clusterize` (clusterize.py:1702) et post-NLL dans
   `get_catalog_from_results` (localization.py:1418). La v4 ne modifiant
   que le comptage post-NLL, le cas cible (« event à P nombreux mais S
   sous seuil ») était rejeté AVANT Pass 1 et jamais sauvé — l'event n'a
   jamais la chance de s'enrichir en S. Décision v6 : quand
   `reserve_pick.enable=true` et que le cluster a au moins une station
   P-only avec un candidat S de réserve plausible dans le pool (fenêtre
   temporelle dérivée de `tt_tolerance_s` et de
   `max_search_dist * (1/leiden_vs - 1/leiden_vp)`), le gate pre-NLL
   **laisse passer le cluster** sans essayer de trancher ; la décision
   définitive reste le gate post-NLL pondéré. Pas de duplication de la
   logique d'acceptation, pas de dépendance au préloc PyOcto.
2. **Tag sur le Pick, pas sur l'Arrival** : au point d'accroche, un pick de
   réserve ajouté à `event2.picks` n'a PAS d'Arrival (celles de l'origin
   Pass 1 référencent les picks Pass 1 ; les arrivals de Pass 2 sont des
   objets neufs créés par NLL). Le tag Arrival spécifié en v4 était donc
   perdu. Les commentaires de Pick survivent au remap
   (`_remap_arrivals_to_picks`, deepcopy, localization.py:1621).
3. **Sentinelle `ps_with_both < 1` → `== 0`** : la valeur de
   `ps_with_both` devient pondérée (float) et est utilisée PARTOUT
   (sentinelle, `min_station_with_P_and_S`, `min_ps_ratio`/Wilson). La
   sentinelle « aucune station P&S du tout » (localization.py:1432) passe
   de `< 1` à `== 0` : une station réserve (0.5) n'est pas « aucune
   station P&S » et ne tombe donc pas à tort dans le chemin
   `p_only_max_rms`. Pas de dualité natif/pondéré à maintenir — le comptage
   natif n'apparaît que dans le logging de décomposition.
4. **Plomberie par fenêtre** : le locator est construit une fois par job
   (core.py:509) mais le pool de réserve est par fenêtre temporelle —
   `reserve_picks` passe par `get_localisations_from_nllobs_dir(...)`
   (bundles per-cluster, comme `picks`), pas par `__init__`.
5. **Filtres d'ingestion appliqués au pool** : renommage de stations,
   blacklist, fréquence, dédup — sinon le matching station échoue et la règle
   d'unicité rejette systématiquement les doublons de composantes.
6. **`event_id` des picks de réserve écrasé** — sinon ils contaminent
   `event_ids_in_picks` et activent le override « known event_id support ».
7. **Dépendance `double_pass` validée en config** (le hook n'existe que dans
   la branche `if self.double_pass and pass_count == 0`).
8. **Garde-fou dégradation RMS décrit honnêtement** : actif seulement si
   `pass2_degradation_factor` est défini ET `pass2_fallback: true`.
9. Références code ré-ancrées (v4 citait `locate()` — inexistant — et des
   lignes décalées d'environ 300 lignes).

Décisions de fond conservées de la v4 :

- Point d'accroche : entre Pass 1 et Pass 2, dans le flux `event2` /
  `cleanup_pick_phase` de `nll_localisation()` — aucun appel NLL
  supplémentaire, pas de circularité (foyer = origin Pass 1 NLL, pas le
  préloc PyOcto).
- Modèle TT de prédiction : Vp/Vs homogène (`leiden_vp`/`leiden_vs`,
  config.py:542-543 — utilisés par le filtre de moveout compatibility).
  `ΔP-S` attendu = d/Vs − d/Vp, dérivé de la même distance foyer-station.
- Complétion S uniquement ; le chemin « compléter un P » est absent du
  design, pas pondéré à 0.
- Test d'acceptation du candidat S : TT absolu + ΔP-S systématiques
  (le P existant sert d'ancre gratuite).
- Règle d'unicité : exactement 1 candidat dans la fenêtre après dédup
  (0 ou ≥2 → ignoré, pas de sélection du « meilleur »).
- Station complétée par S de réserve : poids configurable
  (`ps_count_weight`, défaut 0.5) dans `min_station_with_P_and_S`.
- Aucun forçage de `time_weight` : les picks consolidés entrent dans Pass 2
  à parts égales, NLL pondère selon la qualité d'ajustement.
- Feature opt-in (`enable: false`), validation offline obligatoire avant
  production.

## Steps

### Phase 1 — Config (*base de tout le reste*)

1. Nouveau bloc `ReservePickConfig` dans `dbclust/config.py`, section
   **optionnelle** (pattern `slurm` : annotation
   `reserve_pick: Optional[ReservePickConfig] = None` + entrée dans
   `optional_keys`) :
   - `enable: bool = False`
   - `S_proba_threshold_low: float` — borne basse du pool de réserve S
     (pas de bande basse P — le pool ne contient que des S).
   - `tt_tolerance_s: float` — tolérance TT absolue (s) sur le temps prédit
     du candidat S. Doit absorber l'erreur de profondeur de l'origin Pass 1
     (modèle générique) — à calibrer en Phase 6.
   - `ps_diff_tolerance_s: float` — tolérance sur l'écart ΔP-S prédit
     (d/Vs − d/Vp) vs observé (candidat S − P existant à la station).
   - `ps_count_weight: float = 0.5` — poids d'une station complétée par S
     de réserve dans le comptage P+S.
2. Validations **croisées** dans `DBClustConfig.__init__` (config.py:1192,
   à la fin, au niveau de `zones.load_zones()` — pas dans
   `ReservePickConfig.__post_init__` qui n'a pas accès à `cfg.pick`) :
   - erreur si `S_proba_threshold_low >= pick.S_proba_threshold` ;
   - erreur si `enable=true` et `relocation.double_pass` est falsy (le hook
     de consolidation n'existe que dans la branche double-pass ;
     localization.py:1030) ;
   - erreur si `enable=true` et `config_type == "reloc"` → en pratique :
     **warning + désactivation** (plutôt qu'une erreur) pour qu'un même
     fichier de config serve le pipeline et la CLI relocate ;
   - erreur si `0 < ps_count_weight <= 1` n'est pas respecté, ou tolérance
     ≤ 0.

### Phase 2 — Ingestion du pool de réserve S (*depends on Phase 1*)

3. Dans `dbclust/core.py`, refactorer la requête SQL de lecture des phases
   en une fonction paramétrée par bornes de proba (sites : la requête
   fenêtre ~L567-601, et `get_cross_partition_picks` ~L149-185). La requête
   principale reste inchangée (P ≥ `P_proba_threshold`, S ≥
   `S_proba_threshold`) ; une seconde requête lit la bande basse S
   (`S_proba_threshold_low` ≤ score < `S_proba_threshold`), uniquement si
   `reserve_pick.enable`. Pas de bande basse P.
4. Le DataFrame de réserve passe par les **mêmes filtres** que les picks
   actifs, dans le même ordre (core.py:666-700) :
   - `apply_station_filters` (blacklist, frequency_threshold, **rename** —
     sans le renommage, le matching station échoue pour toute station
     renommée) ;
   - `deduplicate_picks_by_time` avec les mêmes
     `S_proximity_threshold`/`P_proximity_threshold` — c'est cette dédup qui
     fusionne les picks S dupliqués sur plusieurs composantes (N+E), ce qui
     remplace la règle « dédup channel » sous-spécifiée de la v4 ;
   - `import_phases` avec les bornes de la bande basse → objets `Phase`.
5. Pas de champ `reserve` sur `Phase` : le pool vit dans une liste séparée
   (suffisant pour l'isolation), et le seul tag utile — `reserve_consolidated`
   — est posé sur le **Pick** au moment de la consolidation (étape 20), pas
   à l'ingestion.
6. **Écraser `event_id`** des picks de réserve (`Phase.event_id = None`, et
   pas de `creation_info.author` porteur d'un event_id) : un pick de réserve
   issu du catalog ne doit pas contaminer `event_ids_in_picks`
   (localization.py:1339-1346) et activer le override « known event_id
   support » (localization.py:1475).
7. Le pool vit dans une liste séparée `reserve_picks`, **jamais injectée**
   dans `df_subset`, `previous_myclust`, les `Phase` de clusters, ou
   `find_cluster_phases_for_event`. Il est transmis en **lecture seule** à
   `Clusterize` (Phase 3, pour le gate uniquement) puis à `NllLoc` par
   fenêtre (Phase 4). Vérifier qu'aucun chemin de `clusterize.py` ne
   l'ajoute aux pick sets de clusters (`picks_bundles`, obs files).
8. Mode parallèle : étendre `get_cross_partition_picks` pour lire aussi la
   bande basse S de la zone d'overlap backward (core.py:149), et faire
   passer le pool de réserve correspondant avec `df_backward_overlap` —
   sinon les events à cheval sur deux partitions n'ont pas de candidats.
   Les pools suivent aussi les `deferred_phases_next_round`.
9. Côté import fichier (`dbclust/phase.py`, `import_phases` ~L642-700) :
   même mécanisme si le pipeline CSV/parquet doit alimenter la réserve
   (bornes paramétrées déjà présentes dans la signature).

### Phase 3 — Gate pre-NLL : bypass (*depends on Phase 2, NOUVEAU*)

10. `Clusterize` reçoit le pool de réserve en paramètre lecture seule (par
    fenêtre, avec les `Phase` de la fenêtre), jamais comme picks de
    cluster.
11. Dans la boucle de finalisation des clusters (clusterize.py:1690-1750,
    filtre `_check_pre_nll_filter` à :1702) :
    - compter `stations_with_both` natif comme aujourd'hui ;
    - si `stations_with_both < min_station_with_P_and_S` ET réserve
      activée : chercher si le cluster a au moins une station P-only (P
      dans le cluster, pas de S) pour laquelle le pool contient un pick S
      de réserve dans la fenêtre généreuse
      `[t_min_cluster − tt_tolerance_s,
        t_max_cluster + tt_tolerance_s + max_search_dist × (1/leiden_vs − 1/leiden_vp)]`
      (recherche d'existence uniquement — pas de test TT/ΔP-S/unicité à ce
      stade, pas de préloc) ;
    - si oui : **bypass** — le cluster est envoyé à NLL (log
      `Passed (reserve candidates)`), la décision définitive revient au
      gate post-NLL pondéré (Phase 5). Si non : comportement actuel
      (rejet ou overrides existants).
12. Sans candidat plausible, rien ne change : pas de surcoût NLL pour les
    clusters P-only sans aucune chance d'enrichissement. Un cluster
    bypassé dont aucun candidat ne survit à la consolidation est rejeté
    par le gate post-NLL — pas de régression possible.
13. Le pool entier (petit : bande basse S d'une fenêtre) est passé à
    `locator.get_localisations_from_nllobs_dir(..., reserve_picks=pool)`
    — pas de shortlist per-cluster à maintenir ; la consolidation filtre
    par station + fenêtre temps autour de l'origin Pass 1.

### Phase 4 — Consolidation entre Pass 1 et Pass 2 (*depends on Phase 3*)

14. `NllLoc` : paramètres de config via `nllloc_kwargs_from_config`
    (core.py:236 — tolérances, poids, flag ; `getattr` avec défauts
    désactivés pour le chemin reloc). Le pool **ne passe pas par
    `__init__`** (locator construit une seule fois par job, core.py:509) :
    il est threadé par appel — `get_localisations_from_nllobs_dir(...
    reserve_picks=...)` → `nll_localisation(..., reserve_picks=...)`
    (localization.py:1525, signature alignée sur `picks`).
15. Nouvelle méthode `NllLoc._consolidate_reserve_s_picks(event2,
    origin_pass1, reserve_candidates)`, appelée dans `nll_localisation()`
    dans la branche `if self.double_pass and pass_count == 0:` (localization.py:1030),
    **après** `cleanup_picks_and_relabel_picks`/`cleanup_pick_phase`
    (localization.py:1047-1054) et **avant** `cat2.write(new_nll_obs_file,
    format="NLLOC_OBS")` (localization.py:1075). Garde : no-op si
    `reserve_picks` vide ou `enable=false`.
16. Candidature d'une station (via `_station_phase_sets`, localization.py:187,
    sur les arrivals de l'origin Pass 1 de `event2` — les arrivals
    désactivées par le cleanup ont `time_weight=0`) :
    - un P **actif** (`time_weight != 0`) ;
    - **aucun pick S du tout dans `event2.picks`** — actif OU désactivé.
      Cette condition protège le mode `use_deactivated_arrivals=true`, où un
      S désactivé reste dans `event2.picks` et serait réévalué par Pass 2 :
      y ajouter un S de réserve donnerait deux picks S à la même station
      dans le fichier obs.
17. Temps prédit du S : distance **hypocentrale** recalculée depuis
    `origin_pass1` (lat/lon/depth) et les coordonnées de la station (objets
    `Phase` avec `coord`, phase.py `_fetch_coordinates`) — PAS
    `arrival.distance` (épicentrale, en degrés). `t_pred_S =
    origin_time + d / leiden_vs`.
18. Acceptation d'un candidat (toutes les conditions, dans cet ordre) :
    - fenêtre TT : `|t_cand − t_pred_S| ≤ tt_tolerance_s` ;
    - ΔP-S : `|ΔP-S_observé − (d/vs − d/vp)| ≤ ps_diff_tolerance_s` ;
    - **unicité** : exactement 1 candidat dans la fenêtre après dédup
      (0 ou ≥2 → station ignorée, pas de sélection du « meilleur »).
19. **Usage unique** : marquer les picks acceptés comme consommés (set
    per-fenêtre, clé station+temps) — deux events voisins d'une même fenêtre
    ne peuvent pas consolider le même pick de réserve. Les picks consolidés
    d'un event accepté suivent le circuit normal (catalog, unload
    `picks_to_remove` de la zone overlap, core.py:1120).
20. Si accepté : ajouter le `Pick` à `event2.picks`, avec **commentaire de
    Pick** JSON `{"reserve_consolidated": true, "predicted_tt": ...,
    "residual_at_accept": ...}`. Pas d'Arrival à ce stade (voir Contexte,
    point 2) ; le commentaire de Pick survit au remap
    (`_remap_arrivals_to_picks`, deepcopy, localization.py:1621). Le
    commentaire d'Arrival pour la QC est posé en Phase 5, sur l'origin
    final.
21. Le reste du flux est inchangé : fichier obs écrit depuis `event2.picks`,
    Pass 2 pondère normalement, filtre MAD Pass 3
    (`_detect_mad_outliers`, localization.py:1192) s'applique. **Garde-fou
    dégradation** (localization.py:1233-1252) : ce n'est un vrai filet de
    sécurité que si `pass2_degradation_factor` est défini ET
    `pass2_fallback: true` (défaut : warn seulement). Recommander cette
    combinaison dans la doc de `reserve_pick`. Si le fallback revert vers
    l'origin Pass 1 : le comptage pondéré de la Phase 5 se calcule
    naturellement sur l'origin préféré (les arrivals de réserve n'y sont
    pas → comptage natif), et les picks de réserve restent dans `e.picks`
    (picks sans arrival sur l'origin préféré — comportement documenté,
    visible en QC).

### Phase 5 — Comptage pondéré post-NLL (*depends on Phase 4*)

22. Étendre `_station_phase_sets` (localization.py:187) — ou ajouter une
    fonction sœur — pour exposer, par station, si le S de la paire P+S
    provient d'un pick taggé `reserve_consolidated` (résolution
    `arrival.pick_id` → pick → commentaire JSON).
23. `_compute_ps_ratio` (localization.py:1284) renvoie un `ps_with_both`
    **pondéré** (float) utilisé partout — plus de dualité natif/pondéré :
    - la sentinelle `ps_with_both < 1` (localization.py:1432) devient
      `ps_with_both == 0` : une station réserve (0.5) n'est pas « aucune
      station P&S du tout » et ne tombe pas à tort dans le chemin
      `p_only_max_rms` ;
    - `min_station_with_P_and_S` (localization.py:1418) compare la valeur
      pondérée ;
    - `_ps_ratio_ok` (min_ps_ratio / Wilson, localization.py:1309) utilise
      la valeur pondérée — sémantique cohérente avec le gate ;
    - `o.quality.ps_station_count` reçoit la valeur pondérée ; la
      décomposition natif/réserve part dans un **commentaire d'event**
      JSON `{"reserve_ps": {"native": ..., "reserve": ..., "weighted": ...}}`
      (même convention que `cluster_stability`/`nll_obs_file`,
      localization.py:1362 — sérialisé en QuakeML).
24. `get_origin_station_score` (localization.py:366) : une station complétée
    par réserve compte `1.0 + ps_count_weight` (P plein + S pondéré) au lieu
    de 2.0 — sinon la même station vaut 0.5 dans le gate ps et 2.0 dans
    `min_station_score`/`score_override`, deux portes de la même décision.
    Cohérence des overrides : `min_station_with_P_and_S_stability_override`
    et `..._score_override` inchangés, appliqués en plus, sur la même base
    pondérée.
25. Tag final d'Arrival : dans `get_catalog_from_results`, une fois
    l'origin final choisi (couvre Pass 2, Pass 3 et le fallback Pass 1),
    ajouter le commentaire d'arrival `{"reserve_consolidated": ...}` par
    matching `arrival.pick_id` → commentaire de Pick. Un seul site, tous
    chemins couverts, alimente la QC et la traçabilité fdsnws.
26. Logging : le `summary` (localization.py:1378) gagne la décomposition,
    ex. `ps=2.5/3(native=2, reserve=1)`.

### Phase 6 — Validation offline avant activation (*depends on Phase 5*)

27. Harnais de comparaison : rejouer une période de référence (config de
    test locale à définir — aucun fichier du type
    `dbclust-alceste-mac-test.yml` n'existe dans le repo) deux fois,
    `reserve_pick.enable=false` puis `true`, et comparer via
    `match_catalogs.py` / `compare_catalogs.py` (**à la racine du repo**,
    pas dans `Utils/`) :
    - nombre d'events promus en plus/en moins (séparer : sauvés par le gate
      pre-NLL vs sauvés par le gate post-NLL) ;
    - distribution `ps_with_both` (natif et pondéré), RMS, profondeur
      avant/après ;
    - taux d'acceptation à chaque étage : candidats provisionnels pre-NLL,
      candidats consolidés Pass 1, survivants MAD Pass 3 ;
    - faux positifs : QC visuel waveform vs pick sur un échantillon de
      `reserve_consolidated`, et contrôle des events sauvés SEULEMENT par
      réserve (0 station P&S native) ;
    - rejet correct de la sentinelle : un event 0 native + 1 réserve avec
      mauvais RMS doit être rejeté par `p_only_max_rms`, message correct.
28. Ne passer `reserve_pick.enable: true` en prod qu'après calibration de
    `tt_tolerance_s`, `ps_diff_tolerance_s`, `ps_count_weight` (et
    vérification de `pass2_degradation_factor`/`pass2_fallback`).

## Relevant files (références vérifiées sur le code courant, branche moveout)

- `dbclust/config.py` — `ReservePickConfig` (nouvelle section optionnelle),
  validations croisées dans `DBClustConfig.__init__` (L1192+).
- `dbclust/core.py` — requête SQL paramétrée par bornes (L149-185, L567-601),
  filtres d'ingestion du pool (L666-700), plomberie per-fenêtre vers
  `get_localisations_from_nllobs_dir` (L962), overlap backward (L149),
  `nllloc_kwargs_from_config` (L236).
- `dbclust/phase.py` — champ `Phase.reserve` (L34), `to_pick` (L206),
  `import_phases` bornes paramétrées (L642-700).
- `dbclust/clusterize.py` — pool lecture seule, comptage provisionnel et
  gate pre-NLL (`_check_pre_nll_filter`, L1702), `generate_nllobs` →
  `reserve_shortlists`, `preloc[i]`.
- `dbclust/localization.py` — `nll_localisation` hook (branche double-pass
  L1030, cleanup L1047-1054, `cat2.write` L1075),
  `_consolidate_reserve_s_picks` (nouveau), `_station_phase_sets` (L187),
  `get_origin_station_score` (L366), `_remap_arrivals_to_picks` (L1621),
  `_compute_ps_ratio` (L1284), `_ps_ratio_ok` (L1309),
  `get_catalog_from_results` (L1322), sentinelle `ps_with_both < 1`
  (L1432), gate pondéré (L1418), dégradation Pass 2 (L1233-1252).
- `match_catalogs.py`, `compare_catalogs.py` (racine du repo) — Phase 6.
- Nouveau test (style `test_mad_outlier.py`).

## Verification

1. Test unitaire : règle d'unicité (0 candidat → ignoré, 1 → accepté,
   ≥2 → ignoré), sur pool dédupé.
2. Test unitaire : dédup du pool fusionne deux picks S de composantes
   différentes à ~même temps (une seule candidature).
3. Test unitaire : comptage pondéré — station native = 1.0, station réserve
   = `ps_count_weight` ; sentinelle : event 0 station P&S → `p_only_max_rms`
   ; event 1 station réserve uniquement (0.5) → PAS dans la branche
   « no P+S station at all ».
4. Test unitaire : `min_station_with_P_and_S` compare la valeur pondérée
   (1 native + 1 réserve à 0.5 = 1.5 < seuil 2 → rejet, sauf overrides).
5. Test unitaire : aucune tentative de complétion P (station avec S actif
   et P manquant inchangée).
6. Test unitaire : station avec S **désactivé** présent dans `event2.picks`
   (`use_deactivated_arrivals=true`) → pas de consolidation (pas de double
   S).
7. Test unitaire : le commentaire `reserve_consolidated` du Pick survit au
   remap (`_remap_arrivals_to_picks`) et le tag d'Arrival est posé sur
   l'origin final (Pass 2, Pass 3 et fallback Pass 1).
8. Test unitaire : `event_id` d'un pick de réserve n'apparaît pas dans
   `event_ids_in_picks` de l'event final.
9. Test unitaire : usage unique — le même pick de réserve n'est pas
   consolidé dans deux events de la même fenêtre.
10. Test unitaire : gate pre-NLL — cluster natif < seuil avec candidat S de
    réserve plausible dans la fenêtre généreuse → passe ; sans candidat →
    comportement inchangé.
11. Test config : erreurs attendues pour `S_proba_threshold_low >=
    S_proba_threshold`, `enable=true` sans `double_pass`, `enable=true` en
    mode reloc, `ps_count_weight` hors ]0, 1].
12. Non-régression : suite existante (`test_clustering.py`,
    `test_mad_outlier.py`, `test_force_double_pass.py`, ...) avec
    `reserve_pick.enable=false` → sorties identiques.
13. Validation offline Phase 6 avant tout passage en prod.
14. QC manuel : commentaires `reserve_consolidated` visibles dans le
    QuakeML de sortie et dans `injectdb`/`fdsnws-server`.

## Decisions

- Hook entre Pass 1 et Pass 2 (pas de nouvel appel NLL, pas de circularité).
- Modèle TT : Vp/Vs homogène (`leiden_vp`/`leiden_vs`), distance
  hypocentrale recalculée depuis l'origin Pass 1 et les coordonnées
  station.
- Complétion S uniquement — pas de complétion P du tout.
- Acceptation du candidat : TT absolu + ΔP-S + unicité, systématiques.
- **Gate pre-NLL en bypass** : quand la réserve est activée, un cluster
  sous le seuil natif mais avec un candidat S de réserve plausible est
  envoyé à NLL sans trancher — c'est le seul moyen d'atteindre le cas
  d'usage déclaré, et ça donne à l'event la chance de s'enrichir en S.
  Aucun pick n'est accepté à ce stade ; le gate post-NLL pondéré décide.
- Tag sur le **Pick** au moment de la consolidation ; tag d'Arrival posé
  sur l'origin final uniquement (single site).
- Sentinelle `ps_with_both == 0` : une station réserve n'est pas « aucune
  station P&S » ; comptage pondéré utilisé partout (gate, ratio, Wilson),
  décomposition natif/réserve seulement en logging.
- `get_origin_station_score` : station réserve = `1.0 + ps_count_weight`
  (cohérence avec le gate ps).
- Feature opt-in (`enable: false`), exige `double_pass`, désactivée en mode
  reloc, validation offline obligatoire.
- Règle d'unicité non configurable.
- Garde-fou dégradation : recommander `pass2_degradation_factor` +
  `pass2_fallback: true` avec la feature ; fallback → comptage natif.

## Further Considerations

1. Calibration empirique de `ps_count_weight` (0.5 proposé), de
   `tt_tolerance_s` et `ps_diff_tolerance_s` en Phase 6 — pas de valeurs
   figées.
2. La requête SQL refactorée par bornes réduit la dette technique et le
   risque de désynchronisation entre bande principale et bande de réserve.
3. Si un besoin de compléter des P apparaît : extension séparée, analyse de
   risque dédiée (résidus P dominants pour OT/profondeur), et il faudra
   aussi revisiter le gate pre-NLL et le station score — pas de symétrie
   générique.
4. Le bypass pre-NLL repose sur une fenêtre temporelle généreuse (pas de
   préloc) : mesurer en Phase 6 le surcoût NLL (clusters P-only envoyés à
   NLL puis rejetés) vs le gain d'events sauvés.
5. `pass2_fallback` revert vers l'origin Pass 1 : les picks de réserve
   restent alors dans `e.picks` sans arrival sur l'origin préféré. Si ça
   pose problème aux consommateurs aval (injectdb), ajouter un nettoyage
   des picks de réserve orphelins au fallback.
