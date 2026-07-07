# DBClust — Revue de bugs (branche `leiden` vs `main`)

**Date** : 2026-07-07
**Reviewer** : Claude (review assistée par LLM, 8 angles + vérification indépendante)
**Scope** : diff `main...HEAD` sur la branche `leiden` (~29 000 lignes, refactor complet du pipeline : `dbclust.py`/`dask-dbclust.py` → `dbclust/core.py` + `dbclust/executors/`)
**Branche de correction** : `bugfix/leiden-review-2026-07-07` (créée depuis `leiden`)

## Méthodologie

Review en deux phases :
1. **Recherche** — 8 angles indépendants (scan ligne à ligne du diff, audit des comportements supprimés lors du refactoring, traçage des appelants/appelés inter-fichiers, réutilisation, simplification, efficacité, altitude architecturale, conventions CLAUDE.md) → 37 candidats.
2. **Vérification** — chaque candidat relu contre le code réel (signature exacte, callers, chemins d'exécution atteignables) et classé CONFIRMED / PLAUSIBLE / REFUTED. Deux candidats ont été écartés en vérification (voir Annexe).

**Mise à jour (correction appliquée le 2026-07-08, branche `bugfix/leiden-review-2026-07-07`)** : chacun des 17 findings a été retraité individuellement avant correction — plusieurs se sont révélés être des faux positifs après vérification approfondie de l'historique git et/ou des données réelles (bugs 3, 9, 17-partiel). Voir statut par bug ci-dessous.

## Récapitulatif

| # | Sévérité | Fichier | Symptôme court | Statut final |
|---|----------|---------|----------------|--------------|
| 1 | Critique | `dbclust/core.py` | Picks backward-overlap contournent blacklist/rename/frequency/unload | ✅ Corrigé |
| 2 | Critique | `dbclust/dbclust2pyocto.py` | Tolérance PyOcto dégradée en cliquet inter-fenêtres | ✅ Corrigé |
| 3 | Élevée | `dbclust/leiden.py` | Fallback mega-cluster ignore vp/vs configurés | ❌ Faux positif — comportement intentionnel (voir §3) |
| 4 | Élevée | `dbclust/dbclust2pyocto.py` | 2ᵉ passe Leiden : cap P-S même station force des arêtes distance 0 | ✅ Corrigé |
| 5 | Élevée | `dbclust/dbclust2pyocto.py` | Déduplication de picks no-op (rebind au lieu de mutation) | ✅ Corrigé |
| 6 | Élevée | `dbclust/clusterize.py` | Absorption de picks S bruit sans contrainte de causalité/distance | ✅ Corrigé |
| 7 | Moyenne | `dbclust/quakeml.py` | Event ID dépendant du fuseau horaire de la machine | ✅ Corrigé |
| 8 | Moyenne | `dbclust/locate.py` | TypeError garanti au lancement du CLI `locate` | ✅ Corrigé (conséquence du bug 12) |
| 9 | Moyenne | `dbclust/core.py` | Filtre anti-faux-picks `phase_index != 1` perdu (CSV) | ❌ Faux positif — format de fichier obsolète (voir §9) |
| 10 | Moyenne | `dbclust/localization.py` | `isclose(None, 0)` → TypeError sur arrivals d'agence | ✅ Corrigé (8 sites) |
| 11 | Moyenne | `fdsnws/server.py` | `OFFSET` sans `LIMIT` → HTTP 500 | ✅ Corrigé |
| 12 | À trancher | `dbclust/localization.py` + `samples/config.yml` | Check post-loc `min_station_with_P_and_S` supprimé mais documenté | ✅ Corrigé — check restauré dans `NllLoc` |
| 13 | À trancher | `dbclust/config.py` | Invariant `nll.min_phase` désactivé, défaut abaissé | 📝 Documenté seulement (défaut=4 conservé, décision auteur) |
| 14 | Cohérence | `dbclust/inject_spatialite.py` / `localization_error.py` | `get_erh_erz` dupliqué et divergent | ✅ Corrigé — copie supprimée, canonisé |
| 15 | Cohérence | 4 fichiers | Décompte "stations P+S" dupliqué 4× | ✅ Corrigé partiellement — tally interne factorisé (voir §15) |
| 16 | Cohérence | `dbclust/executors/parsl_hte.py` | Perte du load-balancing longest-first sur les backends HPC | ✅ Corrigé |
| 17 | Mineure | `dbclust/executors/base.py` | Fallback mort, sans impact sur les exécuteurs de production | ✅ Corrigé — hook explicite `_inject_future` |

---

## 1. Picks d'overlap backward contournent les filtres de station

**Sévérité** : Critique
**Statut** : CONFIRMED

### Symptôme

En mode parallèle, une station blacklistée ou renommée peut réapparaître dans le clustering au premier window de chaque job (sauf le premier), via les picks injectés depuis la zone d'overlap backward.

### Cause

`get_cross_partition_picks()` (`dbclust/core.py:116-156`) construit `df_backward_overlap` directement depuis DuckDB avec uniquement des seuils de probabilité P/S :

```python
rqt = f"""
    SELECT DISTINCT station_id, channel, phase_type, phase_time, ...
    FROM PICKS
    WHERE phase_time >= '{backward_start}' AND phase_time < '{start}'
    AND phase_type IN ('P', 'Pg', 'Pn', 'S', 'Sg', 'Sn')
    AND ( ... phase_score >= seuils ... )
"""
```

Ce DataFrame est ensuite importé directement (`dbclust/core.py:650`, appel à `import_phases(df_backward_overlap, ...)`) **sans passer par** les filtres appliqués à `df_subset` juste avant :

- blacklist de stations — `dbclust/core.py:571-576`
- frequency_threshold — `dbclust/core.py:606-616`
- rename de stations — `dbclust/core.py:619-620`
- `unload_picks_list` (retrait des picks déjà associés à un événement) — `dbclust/core.py:594-602`

### Localisation

- Point d'injection : [dbclust/core.py:645-660](dbclust/core.py#L645-L660)
- Filtres appliqués à `df_subset` mais pas à `df_backward_overlap` : [dbclust/core.py:571](dbclust/core.py#L571), [:594](dbclust/core.py#L594), [:606](dbclust/core.py#L606), [:619](dbclust/core.py#L619)
- Source : `get_cross_partition_picks()`, [dbclust/core.py:116-156](dbclust/core.py#L116-L156)

### Correctif proposé

Extraire le bloc de filtrage (blacklist → dedup → unload → frequency → rename, lignes 571-620) en une fonction `apply_station_filters(df, cfg, picks_to_remove)` et l'appeler sur `df_backward_overlap` juste après `get_cross_partition_picks()`, avant `import_phases`.

```python
# Avant (core.py:645)
if df_backward_overlap is not None and not df_backward_overlap.empty:
    backward_phases = import_phases(df_backward_overlap, ...)

# Après
if df_backward_overlap is not None and not df_backward_overlap.empty:
    df_backward_overlap = apply_station_filters(df_backward_overlap, cfg, picks_to_remove=[])
    backward_phases = import_phases(df_backward_overlap, ...)
```

---

## 2. La tolérance PyOcto se dégrade en cliquet à travers les fenêtres

**Sévérité** : Critique
**Statut** : CONFIRMED

### Symptôme

Après qu'une fenêtre temporelle ait déclenché un `MultipleEventIDsWithSameAgencyError` et forcé une décroissance de tolérance, toutes les fenêtres suivantes du même run démarrent avec la tolérance dégradée au lieu de la valeur configurée.

### Cause

`adjust_associator_tolerance()` lit et mute directement l'objet de config partagé :

```python
# dbclust/dbclust2pyocto.py:101-109
associator = cfg.pyocto.current_model.associator
tolerance = associator.pick_match_tolerance   # relit la valeur courante, pas la valeur configurée
...
while tolerance >= min_tolerance:
    ...
    associator.pick_match_tolerance = tolerance   # mutation en place, jamais restaurée
```

`cfg.pyocto.current_model.associator` est lié une seule fois pour tout le run (`config.py:993-995`). Aucune restauration n'a lieu après le `while`, ni dans l'appelant (`dbclust/core.py:~796`).

### Localisation

[dbclust/dbclust2pyocto.py:101-115](dbclust/dbclust2pyocto.py#L101-L115)

### Correctif proposé

Sauvegarder/restaurer la tolérance d'origine, ou ne plus muter l'objet partagé :

```python
def adjust_associator_tolerance(cfg, myclust, ..., min_tolerance=0.1):
    associator = cfg.pyocto.current_model.associator
    original_tolerance = associator.pick_match_tolerance
    tolerance = original_tolerance
    try:
        while tolerance >= min_tolerance:
            associator.pick_match_tolerance = tolerance
            ...
    finally:
        associator.pick_match_tolerance = original_tolerance
    return best_result
```

---

## 3. ~~Le fallback mega-cluster Leiden ignore vp/vs configurés~~ — Faux positif, comportement intentionnel

**Statut** : REFUTED (après question de l'auteur : « est-ce que cela n'était pas fait intentionnellement ? »)

### Ce qui a été vérifié

`recluster_mega_with_leiden()` (`dbclust/leiden.py:456-489`) n'accepte effectivement pas `vp`/`vs` et utilise donc toujours les défauts 6.0/3.5 de `leiden_cluster()`. Mais ce chemin **n'est appelé que depuis la branche HDBSCAN** de `Clusterize.get_clusters()` (`clusterize.py:1282-1293`, sous `if mega_cluster_fallback_leiden:`), jamais depuis la branche `clustering_method == "leiden"`.

Le message du commit qui a introduit ce code (`31158b0`, *"Hybrid Leiden+HDBSCAN clustering with PS-boost and noise recovery"*) est explicite sur l'intention :

> - Disable dt_SP_max cap for Leiden (PS-boost handles P-S cohesion instead)
> - **Enable dt_SP_max cap for HDBSCAN (vp=6.0, vs=3.5) to prevent spurious cross-event P-S links**

`mega_cluster_fallback_leiden` re-cluster un **mega-cluster produit par HDBSCAN** (potentiellement pollué de faux liens P-S inter-événements que le cap moveout est censé prévenir) — l'algorithme Leiden y est utilisé uniquement comme méthode de partitionnement, mais la protection moveout de HDBSCAN doit rester active sur ce pool de picks. Transmettre `leiden_vp=0`/`leiden_vs=0` (qui n'a de sens que pour désactiver le cap sur le chemin Leiden *primaire*) casserait cette protection.

**Conclusion** : pas de correctif — le comportement actuel est le design voulu. Retiré du plan de correction.

---

## 4. Seconde passe PyOcto : cap P-S incompatible avec Leiden

**Sévérité** : Élevée
**Statut** : CONFIRMED

### Symptôme

Quand `clustering_method=leiden` et qu'une passe résiduelle re-cluster >20 picks non associés par PyOcto, des picks P et S de **deux événements distincts** à la même station peuvent être fusionnés dans le même cluster.

### Cause

La seconde passe construit la matrice TT avec `vp`/`vs` toujours définis :

```python
# dbclust/dbclust2pyocto.py:407-411
pseudo_tt2 = myclust.numpy_compute_tt_matrix_vectorized(
    unassigned, myclust.average_velocity,
    vp=myclust.apparent_vp,
    vs=myclust.apparent_vs,
)
```

alors que les sites primaires désactivent explicitement ce cap pour Leiden car il "altère la composition des clusters de façon défavorable" (docstring `clusterize.py:888-889`) :

```python
# clusterize.py:755 (site primaire)
vp=self.apparent_vp if self.clustering_method != "leiden" else None,
```

Pour une paire P-S à la même station, `dist_km=0` ⇒ `dt_SP_max=0` ⇒ la distance pseudo-TT calculée vaut 0, ce qui colle artificiellement deux picks qui devraient être loin dans le graphe.

### Localisation

[dbclust/dbclust2pyocto.py:407-411](dbclust/dbclust2pyocto.py#L407-L411) vs le pattern correct en [dbclust/clusterize.py:755](dbclust/clusterize.py#L755)

### Correctif proposé

```python
pseudo_tt2 = myclust.numpy_compute_tt_matrix_vectorized(
    unassigned, myclust.average_velocity,
    vp=myclust.apparent_vp if myclust.clustering_method != "leiden" else None,
    vs=myclust.apparent_vs if myclust.clustering_method != "leiden" else None,
)
```

---

## 5. Déduplication de picks no-op dans `aggregate_pick_to_cluster_with_common_event_id`

**Sévérité** : Élevée
**Statut** : CONFIRMED

### Symptôme

Quand un cluster PyOcto dépasse `pick_count_threshold` picks partageant un `event_id`, ses propres picks sont ré-ajoutés en double dans `newclust.clusters`, gonflant les comptages en aval (Counter de picks communs dans `cluster_merge`, comptages d'`event_id`, totaux utilisés par les filtres pré-NLL).

### Cause

```python
# dbclust/dbclust2pyocto.py:720-732
cluster.extend(picks_to_add)          # mutation EN PLACE de la liste dans `clusters`
added = set(id(p) for p in picks_to_add)
picks = [p for p in picks if id(p) not in added]

# Remove duplicates in the cluster
cluster = list(set(cluster))          # REBIND local — la version dédupliquée est jetée
```

`picks_to_add` est filtré depuis `all_picks_list`, qui contient déjà les phases propres du cluster (`chain(*myclust.clusters)`), donc `extend()` réintroduit des picks déjà membres. La ligne suivante était censée dédupliquer via `set()`, mais elle ne fait que réassigner la variable locale `cluster` — la liste réellement stockée dans `clusters` (mutée par `.extend()`) n'est jamais remplacée.

### Localisation

[dbclust/dbclust2pyocto.py:720-732](dbclust/dbclust2pyocto.py#L720-L732)

### Correctif proposé

Réassigner dans la structure conteneur, pas dans la variable de boucle. Si `clusters` est une liste de listes indexée :

```python
clusters[i] = list(set(cluster.extend(picks_to_add) or cluster))
```

Plus proprement, filtrer `picks_to_add` par appartenance avant l'`extend` (évite de dépendre du hachage de `Phase` pour corriger après coup) :

```python
existing_ids = set(id(p) for p in cluster)
picks_to_add = [p for p in picks_to_add if id(p) not in existing_ids]
cluster.extend(picks_to_add)
```

---

## 6. Absorption de picks S bruit sans contrainte de causalité ni de distance

**Sévérité** : Élevée
**Statut** : CONFIRMED

### Symptôme

Deux événements distincts survenant à quelques minutes d'écart, tous deux enregistrés à la station X : le pick S du second événement (isolé comme bruit par HDBSCAN) peut être absorbé dans le cluster du premier événement simplement parce que celui-ci contient un pick P de la station X — même si le S précède le P ou en est distant de plusieurs minutes.

### Cause

```python
# dbclust/clusterize.py:1105-1116
for cluster in clusters:
    for p_pick in cluster:
        if (
            p_pick.is_p()
            and p_pick.network == s_pick.network
            and p_pick.station == s_pick.station
        ):
            cluster.append(s_pick)
            recovered += 1
            absorbed = True
            break
```

Aucune vérification de causalité (`s_pick.time > p_pick.time`), ni de proximité temporelle, ni de borne `max_search_dist` — contrairement à la logique miroir côté Leiden qui exige explicitement `times[cols] > times[rows]` et une distance pseudo-TT bornée (`dbclust/leiden.py:71,115`).

### Localisation

[dbclust/clusterize.py:1100-1119](dbclust/clusterize.py#L1100-L1119) vs le pattern correct en [dbclust/leiden.py:71-115](dbclust/leiden.py#L71-L115)

### Correctif proposé

Ajouter les trois gardes manquantes avant absorption :

```python
if (
    p_pick.is_p()
    and p_pick.network == s_pick.network
    and p_pick.station == s_pick.station
    and s_pick.time > p_pick.time
    and (s_pick.time - p_pick.time) <= max_search_dist / apparent_vs  # borne physique raisonnable
):
    cluster.append(s_pick)
    ...
```

---

## 7. Event ID dépendant du fuseau horaire de la machine

**Sévérité** : Moyenne
**Statut** : CONFIRMED

### Symptôme

Le même catalogue produit des `event_id` différents selon que le pipeline tourne sur une machine `TZ=Europe/Paris` ou sur un nœud HPC en UTC. Pendant l'heure doublée du passage à l'heure d'hiver, deux événements distants d'une heure peuvent produire une collision d'ID.

### Cause

```python
# dbclust/quakeml.py:37-39
def datetime_to_base64_timestamp(dt, precision="microsecond"):
    timestamp = dt.timestamp()  # dt est naïf (issu de UTCDateTime.datetime) → interprété en heure LOCALE
```

Appelé depuis `make_event_id()` (`dbclust/quakeml.py:77-80`) avec `dt = time.datetime` (naïf, sans tzinfo). Python interprète tout objet `datetime` naïf comme heure locale au moment de `.timestamp()`.

### Localisation

[dbclust/quakeml.py:37-39](dbclust/quakeml.py#L37-L39), appelé depuis [dbclust/quakeml.py:77-80](dbclust/quakeml.py#L77-L80)

### Correctif proposé

```python
from datetime import timezone

def datetime_to_base64_timestamp(dt, precision="microsecond"):
    timestamp = dt.replace(tzinfo=timezone.utc).timestamp()
```

---

## 8. TypeError garanti au lancement du CLI `locate`

**Sévérité** : Moyenne
**Statut** : CONFIRMED

### Symptôme

Toute invocation du CLI `dbclust/locate.py` crashe avant toute localisation.

### Cause

```python
# dbclust/locate.py:335
min_station_with_P_and_S=getattr(cfg_cluster, "min_station_with_P_and_S", 0),
```

`NllLoc.__init__` (`dbclust/localization.py:168-213`) n'a **pas** de paramètre `min_station_with_P_and_S` (ni `**kwargs`) — c'est un fossile du refactoring qui a retiré ce check de `NllLoc` (voir finding 12). L'appel lève immédiatement :

```
TypeError: __init__() got an unexpected keyword argument 'min_station_with_P_and_S'
```

### Localisation

[dbclust/locate.py:335](dbclust/locate.py#L335), signature actuelle de [dbclust/localization.py:168-213](dbclust/localization.py#L168-L213)

### Correctif proposé

Lié à la décision du finding 12. Dans l'immédiat, a minima retirer le kwarg orphelin :

```python
# Retirer purement et simplement la ligne 335 tant que NllLoc n'accepte pas ce paramètre
```

Si le check post-localisation est restauré dans `NllLoc` (cf. finding 12), le rebrancher au lieu de le supprimer.

---

## 9. ~~Filtre anti-faux-picks `phase_index != 1` perdu pour l'entrée CSV~~ — Faux positif, format obsolète

**Statut** : REFUTED (après vérification d'un fichier CSV réellement utilisé aujourd'hui)

### Ce qui a été vérifié

L'ancien filtre `phase_index != 1` visait un ancien format PhaseNetSDS. Inspection d'un fichier CSV de picks actuellement produit par la chaîne (`~/gitlab/renass/acqui_utils/.../work/picks/*.csv`) :

```
station_id,channel,phase_type,phase_time,phase_score,phase_evaluation,phase_method,event_id,agency
FR.AGO,.SH,P,2003-02-22T20:42:27.419200Z,0.94228697,automatic,PHASENET,,RENASS
```

Ce schéma correspond exactement aux colonnes lues par la requête DuckDB actuelle (`dbclust/core.py`) et **ne contient pas de colonne `phase_index`**. Ce format est obsolète et n'est plus produit par la chaîne d'acquisition actuelle.

**Conclusion** : pas de correctif — aucun fichier CSV réellement consommé aujourd'hui n'a cette colonne, donc rien à filtrer. Retiré du plan de correction.

---

## 10. `isclose(None, 0)` — TypeError sur arrivals sans time_weight

**Sévérité** : Moyenne
**Statut** : CONFIRMED

### Symptôme

Le rechargement d'un événement QuakeML provenant d'une agence externe (arrivals sans `time_weight`, courant pour les catalogues tiers) fait planter `reloc_event()`.

### Cause

```python
# dbclust/localization.py:433-436
if not self.use_deactivated_arrivals and isclose(
    arrival.time_weight, 0, abs_tol=time_weight_tolerance
):
    continue
```

`arrival.time_weight` peut être `None` (absent du QuakeML source). `math.isclose(None, 0, ...)` lève `TypeError: must be real number, not NoneType`. L'ancien code tolérait ce cas via `arrival.time_weight == 0` (falsy-safe en Python pour `None == 0` → `False`, pas d'exception). Noter qu'un `hasattr(arrival, "time_weight")` existe ailleurs dans le fichier (`localization.py:295`) mais ne protège pas contre une valeur `None` — seulement contre l'attribut absent.

### Localisation

[dbclust/localization.py:433-436](dbclust/localization.py#L433-L436)

### Correctif proposé

```python
if not self.use_deactivated_arrivals and arrival.time_weight is not None and isclose(
    arrival.time_weight, 0, abs_tol=time_weight_tolerance
):
    continue
```

---

## 11. `OFFSET` sans `LIMIT` — SQL invalide sous SQLite

**Sévérité** : Moyenne
**Statut** : CONFIRMED

### Symptôme

`GET /fdsnws/event/1/query?offset=100` (offset fourni, limit omis) retourne une erreur HTTP 500.

### Cause

```python
# fdsnws/server.py:263-268
if limit:
    sql += " LIMIT ?"
    params.append(limit)
if offset:
    sql += " OFFSET ?"
    params.append(offset)
```

SQLite exige une clause `LIMIT` pour accepter `OFFSET` — `... OFFSET ?` seul lève `sqlite3.OperationalError: near "OFFSET": syntax error`.

### Localisation

[fdsnws/server.py:263-268](fdsnws/server.py#L263-L268)

### Correctif proposé

```python
if limit:
    sql += " LIMIT ?"
    params.append(limit)
elif offset:
    sql += " LIMIT -1"   # SQLite : -1 = illimité, autorise OFFSET
if offset:
    sql += " OFFSET ?"
    params.append(offset)
```

---

## 12. Check post-localisation `min_station_with_P_and_S` supprimé mais toujours documenté

**Sévérité** : À trancher par l'auteur (régression de comportement)
**Statut** : CONFIRMED

### Symptôme

Un cluster passant le filtre pré-NLL (`min_station_with_P_and_S`, appliqué sur les *picks du cluster*) peut aboutir à un événement localisé avec 0 station ayant à la fois P et S *effectivement utilisés par NLL* — l'ancien pipeline le rejetait après coup.

### Cause

L'ancien `localization.py` avait une méthode `check_stations_with_P_and_S(event, origin, min_count)` appelée en post-localisation (`main:localization.py:488-499`) pour vérifier que les arrivals *retenus par NLL* (pas seulement les picks d'entrée) respectent le seuil. Cette méthode et son appel ont disparu de `NllLoc` dans le refactor.

Or `samples/config.yml` recommande encore explicitement ce mécanisme :

```yaml
# samples/config.yml:176-180
# n_p_and_s_picks doesn't seem to work !
# set it to 0 and rely on dbclust's min_station_with_P_and_S
n_p_and_s_picks: 0
```

Et `dbclust/config.py:485` déclare toujours `min_station_with_P_and_S: int` comme champ obligatoire de `ClusterConfig`, avec des overrides de stabilité/score aux lignes 565-570 — mais plus aucun code dans `NllLoc` ne consulte ce paramètre post-localisation. Il n'est utilisé que pré-NLL sur les picks du cluster (`clusterize.py:1417-1443`, `_compute_station_metrics`), ce qui est une garantie plus faible que sur les arrivals réellement utilisés.

### Localisation

Champ config toujours actif : [dbclust/config.py:485](dbclust/config.py#L485)
Documentation utilisateur obsolète : [samples/config.yml:176-180](samples/config.yml#L176-L180)
Check pré-NLL existant (mais pas post-NLL) : [dbclust/clusterize.py:1417-1443](dbclust/clusterize.py#L1417-L1443)
Lié au finding 8 : [dbclust/locate.py:335](dbclust/locate.py#L335)

### Décision (auteur, 2026-07-08) : Option A — restaurer

### Correctif appliqué

- Ajout du paramètre `min_station_with_P_and_S: int = 0` à `NllLoc.__init__` ([dbclust/localization.py](dbclust/localization.py)).
- Nouveau check dans `get_catalog_from_results()`, juste avant le check `min_ps_ratio` existant, réutilisant `ps_with_both` (déjà calculé par `_compute_ps_ratio` sur les arrivals *effectivement utilisés par NLL* — la garantie forte que l'ancien pipeline offrait) : rejette l'origin si `ps_with_both < min_station_with_P_and_S`, avec le même mécanisme d'override `event_ids_in_picks` (accepté avec warning) que les autres critères de rejet.
- Câblage depuis la config : `cfg.cluster.min_station_with_P_and_S` passé à `NllLoc` dans `get_locator_from_config()` ([dbclust/core.py](dbclust/core.py)).
- `locate.py:335` fonctionne maintenant tel quel (le kwarg existait déjà côté appelant, seul `NllLoc` ne l'acceptait pas) — voir finding 8.
- Note : une méthode `check_stations_with_P_and_S` existait déjà dans le code mais n'avait plus aucun appelant (code mort) ; le nouveau check réutilise `_compute_ps_ratio` plutôt que cette méthode, pour éviter de recréer une 5ᵉ implémentation divergente du même tally (cf. finding 15).

---

## 13. Invariant `nll.min_phase` désactivé, défaut abaissé silencieusement

**Sévérité** : À trancher par l'auteur (régression de comportement)
**Statut** : CONFIRMED

### Symptôme

Une configuration qui ne fixe pas explicitement `nll.min_phase` obtient désormais `NLL_MIN_PHASE=4`, alors que l'ancien pipeline le calculait à partir de `min_station_count + min_station_with_P_and_S` (souvent > 4, ex. 5+2=7 avec les valeurs de `samples/config.yml`). NLL peut désormais localiser avec moins de phases que ce que la configuration impliquait sur `main`.

### Cause

```python
# dbclust/config.py:1213-1216 (commenté, désactivé)
# NLL will discard any location with number of phase < min_phase
# take into account cluster parameters to set it accordingly
# use -1 to not set a limit
# self.nll.min_phase = (
#     self.cluster.min_station_count + self.cluster.min_station_with_P_and_S
# )
```

Le défaut de la dataclass est maintenant fixe :

```python
# dbclust/config.py:610
min_phase: Optional[int] = 4
```

### Localisation

[dbclust/config.py:610](dbclust/config.py#L610) (défaut) et [dbclust/config.py:1213-1216](dbclust/config.py#L1213-L1216) (dérivation désactivée)

### Décision (auteur, 2026-07-08) : Option B — garder le défaut fixe, documenter

### Correctif appliqué

Pas de changement de comportement. Ajout d'un commentaire explicatif dans [dbclust/config.py:610](dbclust/config.py#L610) (au-dessus de `min_phase: Optional[int] = 4`) et d'une note dans [samples/config.yml](samples/config.yml) (section `nll:`) expliquant que l'ancienne dérivation automatique n'est plus active et que les utilisateurs qui en dépendaient doivent fixer `nll.min_phase` explicitement. Le check post-localisation `min_station_with_P_and_S` (finding 12, restauré) reste un filet de sécurité indépendant.

---

## 14. `get_erh_erz` dupliqué et divergent

**Sévérité** : Cohérence (bug latent : incohérence de données)
**Statut** : CONFIRMED

### Symptôme

La valeur `erh`/`erz` stockée dans la base SpatiaLite peut différer de celle portée par le QuakeML pour le même origin.

### Cause

Deux implémentations indépendantes de la même fonction :

```python
# dbclust/localization_error.py:10-25 (canonique, utilisée par le pipeline QuakeML)
match = re.search(r"CovXX (\d+\.\d+) .* YY (\d+\.\d+) .* ZZ (\d+\.\d+)", text)
# ... retourne np.nan en cas d'échec

# dbclust/inject_spatialite.py:529-545 (copie locale, utilisée à l'injection DB)
match = re.search(
    r"CovXX (-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) .* "
    r"YY (-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?) .* "
    r"ZZ (-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)",
    text,
)
# ... retourne None en cas d'échec
```

La regex d'`inject_spatialite.py` gère les valeurs négatives et la notation scientifique, celle de `localization_error.py` non — sur un `CovXX`/`YY`/`ZZ` négatif ou en notation scientifique, la version canonique échoue silencieusement (`np.nan`) alors que la version DB réussit.

### Localisation

[dbclust/localization_error.py:10-25](dbclust/localization_error.py#L10-L25) (canonique)
[dbclust/inject_spatialite.py:529-545](dbclust/inject_spatialite.py#L529-L545) (copie divergente)

### Correctif appliqué

Exactement comme proposé : regex robuste adoptée dans `localization_error.py`, copie locale supprimée d'`inject_spatialite.py`, remplacée par `from dbclust.localization_error import get_erh_erz`. Vérifié que le pilote `sqlite3` stocke `np.nan` exactement comme `None` (converti en `NULL` dans les deux cas), donc unifier sur `np.nan` (comportement de la version canonique) ne change rien côté base de données. Import `re` devenu inutile retiré d'`inject_spatialite.py`.

---

## 15. Décompte "stations avec P et S" dupliqué 4× avec filtres divergents

**Sévérité** : Cohérence
**Statut** : CONFIRMED

### Symptôme

`ps_ratio` et `station_score` peuvent classer différemment le même origin selon la fonction consultée, car chaque copie filtre les arrivals différemment (certaines traitent `time_weight is None` comme "présent", d'autres via `isclose` avec tolérance, d'autres pas du tout).

### Cause

Quatre implémentations indépendantes du même calcul "nombre de stations ayant à la fois un pick P et un pick S" :

- `dbclust/localization.py:311` — `get_origin_station_score` (arrivals post-NLL)
- `dbclust/localization.py:1199-1225` — `_compute_ps_ratio` (arrivals post-NLL)
- `dbclust/clusterize.py:1417-1443` — `_compute_station_metrics` (Phase pré-NLL)
- `dbclust/dbclust2pyocto.py:358-382` — filtre inline `min_ps_ratio` (Phase pré-NLL)

### Localisation

Voir les 4 emplacements ci-dessus.

### Correctif appliqué (scope réduit après discussion avec l'auteur)

En creusant, les 4 sites se répartissent en **deux familles structurellement différentes**, pas une seule duplication à 4 :
- Les 2 sites *pré-NLL* (`clusterize.py`, `dbclust2pyocto.py`) opèrent sur des objets `Phase` bruts (`p.is_p()`, pas de `time_weight`) et étaient déjà cohérents entre eux.
- Les 2 sites *post-NLL* (`get_origin_station_score`, `_compute_ps_ratio` dans `localization.py`) opèrent sur des `Arrival` NLL et calculent des **métriques différentes** à partir du même tally (score pondéré 2.0/1.0/0.5 vs ratio simple) — pas une pure duplication non plus.

Une fusion complète en un seul helper générique aurait été un refactor profond sur un pipeline scientifique sensible, pour un risque de changement de comportement numérique difficile à justifier par rapport au gain. Décision : **factoriser uniquement le tally partagé** entre les deux sites post-NLL (qui avaient effectivement la duplication la plus stricte), sans toucher aux sites pré-NLL ni à la méthode déjà-morte `check_stations_with_P_and_S`.

Nouvelle fonction module-level `_station_phase_sets(event, arrivals)` dans [dbclust/localization.py](dbclust/localization.py), appelée par `get_origin_station_score` et `_compute_ps_ratio` (et par le nouveau check du finding 12, qui réutilise `_compute_ps_ratio`). Élimine la seule duplication qui avait un risque réel de divergence silencieuse ; les 2 sites pré-NLL restent séparés (déjà cohérents, structure de données différente).

---

## 16. `ParslHTEExecutor` a perdu le load-balancing longest-first

**Sévérité** : Cohérence (dégradation de performance sur les runs HPC)
**Statut** : CONFIRMED

### Symptôme

Les runs sur backend `parsl_hte`/`parsl_slurm` (les backends HPC réellement utilisés, cf. config Alceste) n'appliquent pas l'optimisation de tri "tâches les plus longues en premier" ajoutée dans `base.run()`, et se rabattent sur un shuffle aléatoire — plus de traînards en fin de run.

### Cause

`base.py` a gagné une optimisation absente de la copie forkée :

```python
# dbclust/executors/base.py:151-153 (chemin générique)
# Falls back to random shuffle if no prior profile exists.
indexed_partitions = self._sort_longest_first(indexed_partitions)
```

Mais `ParslHTEExecutor.run()` (`dbclust/executors/parsl_hte.py:408-...`) **réimplémente entièrement** l'orchestration au lieu d'appeler `base.run()`, et a gardé l'ancien comportement :

```python
# dbclust/executors/parsl_hte.py:479
random.shuffle(indexed_partitions)
```

### Localisation

Optimisation présente : [dbclust/executors/base.py:151-153](dbclust/executors/base.py#L151-L153), implémentation [dbclust/executors/base.py:181-225](dbclust/executors/base.py#L181-L225)
Copie non mise à jour : [dbclust/executors/parsl_hte.py:479](dbclust/executors/parsl_hte.py#L479)

### Correctif appliqué

Exactement comme proposé : `random.shuffle(indexed_partitions)` remplacé par `indexed_partitions = self._sort_longest_first(indexed_partitions)` dans [dbclust/executors/parsl_hte.py](dbclust/executors/parsl_hte.py). Import `random` devenu inutile retiré du fichier.

---

## 17. Fallback mort dans `executors/base.py`

**Sévérité** : Mineure / dette technique
**Statut** : CONFIRMED comme dead-code, mais **sans impact sur les runs de production**

### Contexte de la réévaluation

Ce point a d'abord été classé "critique / perte de données" avant d'être retracé en détail suite à une question de l'auteur ("je n'ai pas constaté de perte"). Le retraçage confirme le défaut de code mais montre qu'il est **inatteignable** avec les exécuteurs réellement utilisés.

### Symptôme (théorique)

Dans le chemin générique `_process_results_windowed` / `_stream_results` de `ExecutorBase`, le refill de la fenêtre glissante duck-type sur des attributs privés d'exécuteurs concrets :

```python
# dbclust/executors/base.py:391-398
ac = getattr(self, "_as_completed", None)
pf = getattr(self, "_pending_futures", None)
if ac is not None:
    ac.add(new_future)          # Dask
elif pf is not None:
    pf.append(new_future)       # Ray
else:
    pending.append(new_future)  # fallback — jamais relu !
```

`_stream_results` (l.377) appelle `wait_for_results(list(pending))` **une seule fois**, sur un snapshot figé. Un exécuteur qui tombe dans la branche `else` verrait ses futures de refill ajoutées à `pending`, mais cette liste n'est plus jamais transmise à un nouvel appel de `wait_for_results`.

### Pourquoi ce n'est pas un bug en pratique

- **Dask** expose `_as_completed` (`dask_executor.py:192`) → branche correcte.
- **Ray** expose `_pending_futures` (`ray_executor.py:187`) → branche correcte.
- **`ParslHTEExecutor`** et **`ParslSlurmExecutor`** — les exécuteurs utilisés en HPC (config Alceste : `executor: "parsl_hte"`) — **overrident entièrement `run()`** (`parsl_hte.py:408`) avec leur propre `_sliding_window_results()` qui refait un `_fill()` correct à chaque itération (`parsl_hte.py:406`). Ils ne passent **jamais** par le chemin générique de `base.py`.
- Seul **`ParslThreadExecutor`** (exécuteur de test local rapide, sans override de `run()`) emprunte le chemin générique — et ne tomberait dans la branche fallback que si un run local soumet plus de tâches que `n_workers × oversubscription_factor` (défaut 5×) d'un coup, ce qui n'arrive pas sur un run de test typique.

### Localisation

[dbclust/executors/base.py:391-398](dbclust/executors/base.py#L391-L398)

### Décision (auteur, 2026-07-08) : ajouter la protection malgré l'absence d'impact actuel

### Correctif appliqué

Duck-typing silencieux remplacé par un hook explicite `_inject_future(pending, new_future)` sur `ExecutorBase` ([dbclust/executors/base.py](dbclust/executors/base.py)) : l'implémentation par défaut lève `NotImplementedError` avec un message explicite au lieu de perdre silencieusement les futures de refill. `DaskExecutor._inject_future` et `RayExecutor._inject_future` implémentent le hook (injection dans `self._as_completed` / `self._pending_futures`, même mécanisme qu'avant, juste explicite). `ParslThreadExecutor` continue de fonctionner tant qu'il ne dépasse pas `max_inflight` (cas d'usage réel : petits runs locaux) ; `ParslHTEExecutor`/`ParslSlurmExecutor` ne sont pas concernés car ils overrident `run()` entièrement.

---

## Annexe — Candidats vérifiés et écartés (faux positifs)

Ces deux candidats issus de la phase de recherche ont été vérifiés et **réfutés** — conservés ici pour éviter de les re-signaler dans une future review.

- **`clusterize.py:1377`** (branche "recovery" pad/truncate sur `clusters`/`clusters_stability`) : la branche existe mais est **prouvée inatteignable**. `stabs` est normalisé à `len(self.clusters)` à l'entrée (l.1318-1322) et chaque mutation de `self.clusters` dans la boucle est faite en lockstep avec `clusters_stability` (l.1329-1330, 1348-1350, 1361-1362). `len(clusters) != len(clusters_stability)` ne peut pas se produire.
- **`config.py: get_time_partitions` / NaT** : le crash théorique sur `pick.start`/`pick.end` non fournis est impossible — `PickConfig.__post_init__` (`config.py:192-195`) dérive `start`/`end` du MIN/MAX réel des données avant que `get_time_partitions` ne soit appelé.
