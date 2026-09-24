# Bug report complet — filtre moveout (leiden.py) et re-fusion de fragments (dbclust2pyocto.py)

Date : 2026-09-24
Branche : `bugfix/leiden-review-2026-07-07`
Fichiers modifiés : `dbclust/config.py`, `dbclust/dbclust2pyocto.py`, `dbclust/leiden.py`, `pyproject.toml`, `samples/config.yml`

Source : synthèse de deux revues indépendantes — (1) revue multi-angles interne (8 agents : bugs, simplification, efficacité, conventions, reuse, root-cause, removed-behavior, cross-file) et (2) rapport externe `mistral-bug-repport-moveout-re_fuse.md` — recoupées et vérifiées ligne à ligne sur le code réel. Chaque point ci-dessous a été confirmé par lecture directe du fichier ; un point de la revue interne (tolérance non restaurée) a été infirmé et retiré.

---

## Verdict global

Le code compile et les fonctions pures se comportent comme prévu. Aucun crash reproduit à l'exécution, mais plusieurs bugs logiques concrets (crash potentiel sur config mal formée, régression de performance sur le cas commun, re-fusion structurellement incapable de couvrir un de ses deux cas d'usage), une option de config rendue muette, et une hypothèse de monotonicité non testée dont l'échec est invisible. Aucun test ne couvre les nouvelles fonctions.

---

## 🔴 Bugs — probablement réels

> **Statut** : B1, B3, B5, B6, B7, B9, B10 ont été corrigés dans `dbclust/dbclust2pyocto.py` (voir note "✅ Corrigé" sous chaque point). B2, B4, B8 restent ouverts — voir leur section respective pour le détail (B2 dépend du même bloc que B1 et a été corrigé au passage ; B4 et B8 nécessitent une validation empirique / une décision produit avant correction).

### B1. La recherche binaire est plus lente que l'ancien code sur le cas commun

`dbclust/dbclust2pyocto.py:99`, `:213-227`

La docstring affirme « Higher tolerance is tried first because it is cheap and the common case », mais la recherche démarre au **milieu** des candidats (`lo, hi = 0, len(candidates)-1` puis `mid = (lo+hi)//2`), pas au sommet. Concrètement : avec `original_tolerance=2.0` et 8 candidats `[2.0, 1.5, 1.0, 0.9, ..., 0.5]`, si `2.0` réussit (le cas que le docstring lui-même qualifie de commun), l'ancien code linéaire faisait **1** association PyOcto ; le nouveau en fait **3** (mid=1.0, puis 1.5, puis 2.0) avant de conclure que le premier candidat était le meilleur. Chaque tentative relance une association PyOcto complète, potentiellement longue — c'est une régression de performance nette sur le cas le plus fréquent, à l'opposé de l'objectif affiché du changement.

**Correction** : sonder `candidates[0]` en premier ; ne basculer en recherche binaire sur `candidates[1:]` que s'il échoue.

**✅ Corrigé** (`dbclust/dbclust2pyocto.py`, `adjust_associator_tolerance`) : `candidates[0]` est maintenant systématiquement essayé en premier ; la bissection ne démarre sur `candidates[1:]` que s'il échoue. Vérifié : le cas commun (tolérance la plus haute réussit) coûte désormais 1 seule tentative au lieu de 3 (sur un jeu de 8 candidats). Marqué `BUGFIX B1` dans le code.

### B2. Hypothèse de monotonicité non garantie, échec silencieux

`dbclust/dbclust2pyocto.py:210-246`

Si le succès n'est pas monotone en tolérance (un `mid` échoue alors qu'une tolérance supérieure aurait réussi — plausible, la géométrie des clusters peut se réorganiser de façon non monotone entre deux pas), la recherche peut conclure « Exhausted all tolerances » et retourner un résultat partiel dégradé, voire `None`, là où la décroissance linéaire aurait trouvé un résultat complet. L'hypothèse est documentée dans le docstring mais son éventuelle violation est invisible dans les logs.

De plus, quand **tous** les candidats échouent, `best_result`/`best_n_clusters` n'est mis à jour que sur les O(log N) tentatives réellement essayées par la bissection (ex. 4 sur 14), au lieu des N tentatives que l'ancien code linéaire explorait exhaustivement. Le « meilleur résultat partiel » retourné en cas d'échec total peut donc être strictement moins bon qu'avant, silencieusement, sans erreur ni log qui le signale. Le docstring/log affirme une équivalence complète avec l'ancien algorithme (« same result … O(log N) instead of O(N) ») ; cette équivalence ne tient que pour la frontière succès/échec, pas pour le chemin de repli.

**Correction** :
- Quand tout échoue, retomber sur un balayage linéaire complet des candidats non encore essayés (coût O(N) uniquement dans ce pire cas, donc pas de régression perf sur le chemin nominal).
- Logger un warning explicite quand `success_idx is None` alors qu'au moins un candidat non testé était supérieur au plus haut échec observé (« monotonicity assumption may have been violated »).
- Corriger le docstring pour ne plus affirmer une équivalence totale.

**✅ Corrigé** (`dbclust/dbclust2pyocto.py`, `adjust_associator_tolerance`) : quand la bissection échoue sur tous les candidats qu'elle a réellement essayés, un balayage linéaire complet des candidats restants est maintenant exécuté en repli (coût O(N) uniquement dans ce pire cas déjà lent, donc pas de régression sur le chemin nominal). Si ce balayage trouve un succès qu'une violation de monotonicité avait fait sauter à la bissection, un `logger.warning` explicite le signale. Marqué `BUGFIX B2` dans le code.

### B3. `min_node_size_for_target` peut devenir un nombre complexe → crash

`dbclust/dbclust2pyocto.py:446-448`

`min_node_size_for_target = ((lat_span_km * lon_span_km * z_span_km) / target_leaf_cells) ** (1/3)`. Si `associator_cfg.zlim` (simple `List[float]` non validée dans `config.py`) est mal configuré avec `zlim[1] <= zlim[0]` (plage de profondeur inversée), `z_span_km` devient négatif ou nul. Une puissance fractionnaire d'une base négative renvoie silencieusement un **nombre complexe** en Python (pas d'exception à ce stade), et l'appel suivant `min(max(min_node_size_for_target, associator_cfg.min_node_size_location), min_node_size_cap)` lève alors un `TypeError: '<' not supported between instances of 'float' and 'complex'`, plantant tout le traitement du cluster au lieu d'échouer proprement sur une config invalide.

**Correction** : garder `z_span_km <= 0` (et `lat_span_km`/`lon_span_km <= 0`) avant le calcul du cube, logger un warning et retomber directement sur `associator_cfg.min_node_size_location`. Envisager aussi une validation de `zlim` au chargement de la config.

**✅ Corrigé** (`dbclust/dbclust2pyocto.py`, boucle de dimensionnement `min_node_size`) : ajout d'une garde sur `search_volume_km3 = lat_span_km * lon_span_km * z_span_km <= 0` avant l'élévation à la puissance `1/3` ; en cas de volume non positif, warning explicite et repli direct sur `associator_cfg.min_node_size_location`, sans passer par le calcul qui produisait un nombre complexe. Marqué `BUGFIX B3` dans le code. La validation de `zlim` au chargement de la config (`config.py`) n'a pas été ajoutée — reste une piste d'amélioration séparée.

### B4. La re-fusion ne peut jamais réparer les faux positifs du filtre qui les a causés

`dbclust/dbclust2pyocto.py:778-873`

`_refuse_fragment_events` exige `dt <= d/vp` (avec `min_compat_fraction=1.0`, strict) en utilisant le **même** `leiden_vp`/`leiden_vs` que le filtre moveout de `leiden.py` qui a coupé l'arête à l'origine du fragment. Si la fragmentation vient d'une vitesse apparente réelle **supérieure** à `leiden_vp` (ex. onde Pn régionale ≈ 8 km/s vs défaut 6.0), la paire de picks incriminée viole *aussi* la borne de re-fusion — la fonction ne peut donc logiquement traiter que les scissions dues à la connectivité du graphe (arêtes intermédiaires coupées alors que source et stations restent physiquement compatibles), jamais les cas où le filtre lui-même a mal classé une paire à cause d'un `vp`/`vs` mal calibré. C'est structurellement circulaire : le réparateur partage l'hypothèse physique du filtre qu'il est censé corriger.

**Correction** :
- Valider empiriquement sur le batch Alceste que la re-fusion se déclenche bien sur des fragments connus (voir recommandation prioritaire ci-dessous) — c'est le seul moyen de trancher si ce cas est actif en pratique ou seulement théorique.
- Si confirmé, introduire une tolérance de fusion distincte et plus généreuse que celle du filtre (ex. `refuse_vp_factor: float = 1.2` appliqué à `vp`/`vs` avant `cross_moveout_ok`).

### B5. `min_compat_fraction=1.0` : un seul pair de picks peut décider la fusion (ou la bloquer)

`dbclust/dbclust2pyocto.py:842-873`

`cross_moveout_ok()` exige que 100 % des paires P-P/S-S croisées passent. Avec très peu de paires disponibles (ex. `n_checked=1`, un seul poste partagé entre fragment et événement principal), une seule paire coïncidente suffit à valider la fusion de deux événements en réalité distincts ; inversement, un seul pick bruité peut bloquer une fusion légitime. Aucun garde-fou sur un nombre minimal de paires observées avant de trancher.

**Correction** : ajouter un seuil `min_pairs_checked` (ex. 3) — si `n_checked` est en dessous, refuser la fusion par prudence plutôt que de trancher sur une évidence statistique insuffisante.

**✅ Corrigé** (`dbclust/dbclust2pyocto.py`, `_refuse_fragment_events`/`cross_moveout_ok`) : nouveau paramètre `min_pairs_checked: int = 3` sur `_refuse_fragment_events` ; `cross_moveout_ok` refuse désormais la fusion (`False`) dès que `n_checked < min_pairs_checked`, au lieu de trancher sur `n_checked == 0` uniquement. Marqué `BUGFIX B5` dans le code.

### B6. `_moveout_fusion_params` se déclenche même quand le filtre moveout n'a pas tourné

`dbclust/dbclust2pyocto.py:778-792`, `dbclust/clusterize.py`

Le signal utilisé est `getattr(myclust, "leiden_vp", None)` / `leiden_vs` — mais `Clusterize.__init__` fixe systématiquement ces attributs à des valeurs par défaut (6.0/3.5), indépendamment du fait que le filtre moveout de `leiden.py` (lui-même conditionné par `dist_km is not None`) ait réellement tourné. Le test `getattr(...) is None` censé détecter « le filtre a tourné » est donc du code mort ; en pratique la condition se réduit à `clustering_method == "leiden"`.

**Correction** : remplacer par un flag explicite, ex. `myclust.moveout_filter_applied: bool`, positionné par `leiden.py` uniquement quand `dist_km is not None` et que le filtre a effectivement retiré au moins une arête.

**✅ Investigué et documenté** (pas de flag explicite ajouté — voir justification) : en retraçant `Clusterize.cluster()` → `get_clusters()` → `leiden_cluster()` (`dbclust/clusterize.py`, `dbclust/leiden.py`), la condition d'activation réelle du filtre moveout dans `leiden.py` (`dist_km_matrix` construit ssi `vp > 0 and vs > 0`, `leiden.py:371`) est **exactement** la même condition que celle déjà testée par `_moveout_fusion_params` (`vp>0 and vs>0`, avec les mêmes valeurs `leiden_vp`/`leiden_vs`). Le test actuel n'est donc pas un bug logique actif avec les valeurs par défaut actuelles (toujours >0) — le risque théorique (un futur flag de désactivation qui ne passerait pas par `vp`/`vs` à 0) reste réel mais un flag explicite `moveout_filter_applied` nécessiterait de le faire remonter à travers plusieurs points d'appel (`get_clusters`, `cluster()`, second pass) pour un gain incertain. **Décision** : documentation renforcée dans `_moveout_fusion_params` (marquée `BUGFIX B6`) expliquant explicitement que ce test *mirrore* la condition d'activation de `leiden.py` et doit être maintenu en synchronisation si celle-ci change, plutôt qu'un refactor plus large et risqué. À rouvrir si `leiden.py` gagne un mode de désactivation indépendant de `vp`/`vs`.

### B7. Chemin `delegate_dbclust` incohérent avec le chemin nominal

`dbclust/dbclust2pyocto.py:676`

Quand PyOcto ne trouve aucun cluster et `delegate_dbclust=True`, la fonction retourne `myclust` directement — avant le bloc de nettoyage `delattr` et avant `_moveout_fusion_params`/`_refuse_fragment_events` que le chemin nominal applique. Toute fragmentation moveout présente dans les clusters `dbclust` d'origine n'est jamais re-fusionnée sur ce chemin, et les attributs censés être nettoyés après tout appel à `dbclust2pyocto()` restent présents.

**Correction** : faire passer ce retour anticipé par le même bloc de nettoyage et de re-fusion que le chemin nominal, ou documenter explicitement pourquoi ce cas est exempté si c'est intentionnel.

**✅ Corrigé** (`dbclust/dbclust2pyocto.py`, retour anticipé `delegate_dbclust`) : la re-fusion moveout (`_moveout_fusion_params` + `_refuse_fragment_events`) est désormais appliquée sur `myclust.clusters`/`myclust.preloc` avant de retourner `myclust`, protégée par `myclust.preloc is not None` (rien à réaligner si aucun preloc PyOcto n'existe sur ce chemin). Le bloc `delattr` de nettoyage n'a **pas** été ajouté ici intentionnellement : il ne s'applique qu'à `newclust` (une copie) sur le chemin nominal, et le supprimer sur `myclust` (l'objet original retourné directement ici) casserait le contrat de cet objet pour l'appelant. Marqué `BUGFIX B7` dans le code.

### B8. `min_node_size` de la config devient une entrée morte

`dbclust/dbclust2pyocto.py:423-468`, `samples/config.yml:216`

`effective_min_node_size` (ligne 449) est calculé uniquement à partir de l'étendue géographique du cluster, de `min_node_size_location` (utilisé comme plancher via `max(...)`) et d'un cap codé en dur de 30 km. `associator_cfg.min_node_size` n'intervient plus nulle part dans le calcul — il n'apparaît plus que dans la f-string du log (ligne 465). `samples/config.yml:216` documente pourtant `min_node_size: 5` comme un paramètre actif : c'est un piège pour l'opérateur qui croit régler quelque chose d'effectif.

**Correction** (décision produit à trancher) : soit (a) réintroduire `min_node_size` comme borne supplémentaire dans le calcul, soit (b) le retirer de `Associator` et de `samples/config.yml` avec une note de migration. Recommandation : (b) si `min_node_size_location` couvre déjà le besoin, pour éviter deux réglages qui se chevauchent.

### B9. Désalignement `clusters_stability` / `clusters_pyocto_ops` sur le chemin exception

`dbclust/dbclust2pyocto.py:690-719`

Sur le chemin `except MultipleEventIDsWithSameAgencyError`, `newclust.n_clusters` est rafraîchi après fusion (`len(newclust.clusters)`) mais `clusters_stability` (placeholder `[1]*n`) n'est pas retronqué en conséquence, et `clusters_pyocto_ops` n'est jamais renseigné sur ce chemin alors qu'il l'est sur le chemin succès. Pas de crash aujourd'hui (liste simplement plus longue que nécessaire) mais désalignement latent pour tout code futur qui zippe `clusters` et `clusters_stability` par index.

**Correction** : appliquer la même troncature/reconstruction de `clusters_stability` (et définir `clusters_pyocto_ops`, au moins à une valeur par défaut cohérente) sur le chemin exception.

**✅ Corrigé** (`dbclust/dbclust2pyocto.py`, chemin `except MultipleEventIDsWithSameAgencyError`) : quand `_refuse_fragment_events` fusionne au moins un fragment (`n_fused > 0`) sur ce chemin, `newclust.n_clusters`, `newclust.clusters_stability` (réinitialisé à `[1.0] * n_clusters`) et `newclust.clusters_pyocto_ops` (réinitialisé à une liste de placeholders `"passthrough"`) sont désormais tous les trois réalignés sur le nombre de clusters post-fusion, au lieu de laisser `clusters_stability`/`clusters_pyocto_ops` sur leur état pré-fusion. Marqué `BUGFIX B9` dans le code.

### B10. Grille de localisation potentiellement plus fine que la config sur cluster compact

`dbclust/dbclust2pyocto.py:453-456`

`effective_min_node_size_location = min(associator_cfg.min_node_size_location, effective_min_node_size * 0.9)`. Pour un cluster compact où `effective_min_node_size` est déjà proche de `min_node_size_location`, ce calcul produit une résolution **plus fine** que la config (ex. 1.35 km au lieu de 1.5 km configuré) — en contradiction directe avec le commentaire adjacent « do not go finer than the location-refinement resolution itself — no point ». Soit le commentaire soit le code exprime l'intention réelle, mais pas les deux à la fois.

**Correction** : clarifier l'intention avec l'auteur avant de corriger (le sens du `min` semble inversé par rapport au commentaire).

**✅ Corrigé** (`dbclust/dbclust2pyocto.py`, calcul `effective_min_node_size_location`) : le `min(config, 0.9×effective)` a été remplacé par `max(config, min(0.9×effective, effective))`, pour que `min_node_size_location` agisse comme un plancher (jamais plus fin que la config) plutôt qu'un plafond — conforme à l'intention exprimée dans le commentaire adjacent (« do not go finer than the location-refinement resolution itself »), tout en préservant le grossissement voulu pour les gros clusters. Marqué `BUGFIX B10` dans le code.

---

## 🟠 Cohérence / fragilité

> **Statut** : C2 à C7 corrigés. C1 (exposition config des seuils de `_refuse_fragment_events`) reste ouvert — c'est un ajout de surface de config plus large, à traiter avec le lot "complexité"/config plutôt qu'en correctif ponctuel.

### C1. Constantes de repli codées en dur, non exposées en config
`_refuse_fragment_events` (dbclust2pyocto.py:795-806) a 7 paramètres (`frag_max_st`, `target_min_st`, `station_ratio_max`, `max_origin_diff_s`, `min_compat_fraction`, `max_rounds`) en défauts Python, plus `target_leaf_cells=5e4` et `min_node_size_cap=30.0` pour le dimensionnement — alors que le reste du diff (`min_pick_fraction_event_count_exponent`) a été proprement exposé dans `config.py`/`samples/config.yml`. Calibrés sur un seul jeu de données (Alceste), non ajustables sans éditer le source.

**✅ Traité, par une refonte plus profonde que "exposer en config"** (`dbclust/dbclust2pyocto.py`, `_refuse_fragment_events`) — voir section dédiée **"Refonte de `_refuse_fragment_events`"** ci-dessous. Résumé : sur les 7 paramètres, 4 ont été supprimés en tant que constantes indépendantes (`frag_max_st`, `target_min_st`, `max_origin_diff_s` sont désormais dérivés de valeurs déjà connues du contexte pipeline ; `min_compat_fraction` et `max_rounds` ont été retirés car ils n'étaient respectivement jamais utilisés qu'à une seule valeur, ou pas un vrai levier de tuning). Il ne reste que **2** paramètres réellement libres (`station_ratio_max`, `min_pairs_checked`), documentés comme tels. Aucun n'a été simplement déplacé en config sans traiter le "sorti du chapeau" — exposer 7 magic numbers en config les aurait rendus configurables sans les rendre justifiés.

### C2. Duplication copier-coller de l'appel de re-fusion
Le motif `moveout_vps = _moveout_fusion_params(myclust); if moveout_vps is not None: _refuse_fragment_events(...)` est dupliqué mot pour mot entre le chemin succès et le chemin exception (~15 lignes d'écart, pas côte à côte). Risque de dérive si un paramètre change d'un côté seulement.

**✅ Corrigé** (`dbclust/dbclust2pyocto.py`) : nouveau helper `_apply_moveout_refusion(myclust, clusters, prelocs)` qui encapsule `_moveout_fusion_params` + `_refuse_fragment_events` (et le court-circuit `prelocs is None`) en un seul appel, réutilisé aux **3** points d'appel — le chemin succès, le chemin exception, et le retour anticipé `delegate_dbclust` (ajouté par le correctif B7). Chaque site garde son propre post-traitement spécifique (réalignement `clusters_stability`/`clusters_pyocto_ops` côté exception, mise à jour `n_clusters` côté `delegate_dbclust`) mais ne duplique plus la logique de déclenchement. Marqué `BUGFIX C2` dans le code.

### C3. Copie superficielle fragile de `Clusterize`
`copy.copy(myclust)` dans `try_tolerance` (dbclust2pyocto.py:156-158) ne clone que `.clusters`/`.noise`. D'autres attributs mutables (`.preloc`, `._deferred_cluster_indices` si présents) restent des références partagées entre `myclust_copy` et l'original — latent tant que rien ne les mute, mais piège pour un futur ajout de code.

**✅ Corrigé** (`dbclust/dbclust2pyocto.py`, `try_tolerance`) : `myclust_copy.preloc` est désormais rebindé à une liste fraîche (`list(myclust.preloc)`) quand `myclust.preloc is not None`, au même titre que `.clusters`/`.noise`. **Ce risque n'était plus seulement latent** : le correctif B7 (retour anticipé `delegate_dbclust`) fait désormais muter `.preloc` en place via `_refuse_fragment_events`, ce qui aurait corrompu la référence partagée avec `myclust` original sans ce correctif — trouvé et corrigé en vérifiant l'interaction entre B7 et C3. `._deferred_cluster_indices` n'est mutée que par les méthodes de `Clusterize` lui-même, jamais par `dbclust2pyocto()` : pas besoin du même traitement. Marqué `BUGFIX C3` dans le code.

### C4. `pyproject.toml` : pin `obspy` relâché
`obspy==1.4.2` → `obspy>=1.4.2`, sans borne supérieure. Le repo applique un patch source (`patch_obspy_nll.sh`) ciblant précisément `obspy/io/nlloc/core.py` de la version 1.4.2 (AGENTS.md : « requires specific patches for NonLinLoc and ObsPy »). Une version plus récente peut faire échouer le patch silencieusement (fuzzy match de `patch --batch --forward`) ou être appliquée au mauvais endroit, réintroduisant le bug de tolérance de pick que ce patch corrige, sans qu'aucun test ne le détecte.

**Correction recommandée** : revenir à `==1.4.2` tant que le patch n'a pas été revalidé contre une version plus récente ; au minimum `>=1.4.2,<2`.

**✅ Corrigé** (`pyproject.toml`) : repin exact à `obspy==1.4.2`, avec commentaire expliquant la dépendance au patch source. Marqué `BUGFIX C4`.

### C5. Commentaires `del partial_result` / `del myclust_copy` trompeurs
`dbclust2pyocto.py:189-201` — Ce sont des locales de fonction libérées de toute façon au `return` qui suit immédiatement. Les ~15 lignes de commentaire qui justifient ces `del` décrivent un effet de libération mémoire qui n'a pas lieu d'être ici (le `del` explicite juste avant un `return` n'apporte rien que le retour de fonction n'aurait pas fait seul).

**✅ Corrigé** (`dbclust/dbclust2pyocto.py`, `try_tolerance`) : les deux `del` explicites ont été retirés, remplacés par un commentaire court expliquant que les locales sortent de portée au `return` normalement. Comportement inchangé (rien ne dépendait du timing du `del`). Marqué `BUGFIX C5` dans le code.

### C6. Incohérence entre défaut code et config exemple pour l'exposant adaptatif
`config.py:975` (`min_pick_fraction_event_count_exponent: float = 0.3`) vs `samples/config.yml:233-237` (commentaire « 0.2 validated on the close-in-time-not-in-space Alceste batch », valeur `0.2` dans le yaml). Un utilisateur qui omet la clé obtient silencieusement `0.3`, jamais testé comme « validé ».

**✅ Corrigé, avec une correction supplémentaire** (`samples/config.yml`) : en revérifiant `TUNING_HANDOFF_close_in_time_not_in_space.md`, le document source dit explicitement **« Ce point n'est pas encore résolu »** et appelle à retester 0.25/0.2/etc. — donc `0.2` n'est **pas** validé, contrairement à ce que disait le commentaire de l'exemple (qui était lui-même la source de l'erreur, pas seulement en décalage avec le défaut code). La ligne `min_pick_fraction_event_count_exponent: 0.2` a été commentée dans `samples/config.yml` (avec l'explication), pour laisser le défaut `config.py` (0.3, la valeur qui a effectivement été testée et documentée avec ses effets connus, y compris sa régression) s'appliquer tant qu'aucune valeur n'a été formellement validée. Marqué `BUGFIX C6/C7` dans le code.

### C7. Commentaire faux sur la baseline `1/sqrt(n)`
`samples/config.yml:233` — « 0.5 is equivalent to the previous 1/sqrt(n) behavior » est factuellement faux vis-à-vis du code committé avant ce diff : l'ancien comportement était `min_pick_fraction * median_proba`, sans aucun facteur d'effectif (`n_event_ids`).

**✅ Corrigé** (même edit que C6, `samples/config.yml`) : le commentaire a été réécrit pour ne plus affirmer d'équivalence avec un comportement qui n'existait pas dans le code pré-diff (aucun facteur d'effectif n'existait avant ce changement — le vrai baseline était `min_pick_fraction * median_proba`, sans terme `1/n^x` du tout). Le commentaire clarifie maintenant qu'un exposant de 1.0 correspondrait à `1/n` (pas au baseline réel non plus) et que le défaut effectif du code reste `config.py`'s 0.3, la seule valeur documentée avec ses effets connus (voir C6).

---

## 🔧 Refonte de `_refuse_fragment_events` (2026-09-24, suite au retour utilisateur)

Après relecture ciblée à la demande de l'utilisateur ("il y a des paramètres qui sortent un peu du chapeau"), inventaire complet des 8 paramètres de la fonction (`vp`, `vs`, `frag_max_st`, `target_min_st`, `station_ratio_max`, `max_origin_diff_s`, `min_compat_fraction`, `min_pairs_checked`, `max_rounds`) :

| Paramètre | Avant | Statut | Traitement |
|---|---|---|---|
| `vp`, `vs` | hérités du filtre moveout | légitime (physique, déjà justifié) | inchangé |
| `frag_max_st` | `6` codé en dur | sorti du chapeau | **dérivé** : `min_station_count + 1` |
| `target_min_st` | `10` codé en dur | sorti du chapeau | **dérivé** : `2 * min_station_count` |
| `station_ratio_max` | `0.35` codé en dur | réglage empirique légitime, mais partiellement redondant avec `frag_max_st` | **conservé** comme seul vrai knob de forme |
| `max_origin_diff_s` | `10.0` codé en dur, sans lien avec le reste du filtre | sorti du chapeau | **dérivé** : réutilise `max_search_dist` (déjà la borne temporelle du clustering) |
| `min_compat_fraction` | `1.0` codé en dur, jamais appelé autrement | complexité morte (identifié dans la revue précédente) | **supprimé** — `cross_moveout_ok` exige maintenant toujours 100% des paires, avec sortie anticipée dès la première paire incompatible |
| `min_pairs_checked` | `3` (mon propre ajout, BUGFIX B5) | arbitraire, mais nécessaire (garde-fou statistique) | **conservé** comme second vrai knob |
| `max_rounds` | `5` codé en dur | pas un vrai réglage — la boucle s'arrête déjà dès qu'un round ne fusionne rien | **supprimé de la signature**, dérivé localement en `len(clusters)` (borne structurelle réelle : chaque round fusionne au plus une paire) |

**Résultat** : 8 paramètres → 6 dans la signature, dont seulement **2 restent des réglages réellement libres et non dérivables** (`station_ratio_max`, `min_pairs_checked`) — documentés comme tels dans le docstring de la fonction, avec une "design note" qui explique explicitement pourquoi chaque paramètre supprimé ne l'était pas justifié plutôt que de se contenter de le déplacer en config.

Concrètement (`dbclust/dbclust2pyocto.py`) :
- `_apply_moveout_refusion` passe désormais `min_station_count=myclust.min_station_count` et `max_search_dist=myclust.max_search_dist` (tous deux déjà connus du pipeline, pas de nouvelle config) à `_refuse_fragment_events`.
- `frag_max_st`, `target_min_st`, `max_origin_diff_s`, `max_rounds` sont calculés en 4 lignes au début de la fonction plutôt que d'être des paramètres par défaut indépendants.
- `cross_moveout_ok` retourne `False` dès la première paire de picks incompatible (early-exit), au lieu de scorer exhaustivement toutes les paires pour ensuite comparer à une fraction toujours égale à 1.0 — traite au passage le point d'efficacité déjà identifié dans la revue de complexité.

**Validation** : sur la config Alceste actuelle (`min_station_count=5`, `max_search_dist=8.0`), les valeurs dérivées reproduisent exactement les anciennes constantes (`frag_max_st=6`, `target_min_st=10`) ou un ordre de grandeur cohérent (`max_origin_diff_s=8.0s` contre `10.0s` avant, légèrement plus strict). Testé avec des `Phase` synthétiques : un fragment compatible se fusionne toujours correctement (cas positif), un événement réellement distinct ne fusionne toujours pas (cas négatif) — comportement fonctionnel préservé.

**Non traité dans cette passe** : `station_ratio_max` et `frag_max_st` (dérivé) se chevauchent encore pour les grands réseaux (`frag_max_st` devient le facteur limitant au-delà de ~17 stations dans la cible, avec `station_ratio_max=0.35`) — leur interaction reste implicite plutôt que formellement unifiée en un seul critère. Une simplification plus radicale (ratio seul avec plancher absolu) a été envisagée mais écartée pour ne pas changer silencieusement le comportement observé sur Alceste sans revalidation empirique.

---

## 🟡 Complexité excessive

> **Statut** : tous les points corrigeables dans le code ont été traités. Seul le point "diff non atomique" reste ouvert — c'est un découpage de commits/process, pas quelque chose qu'on corrige dans le code lui-même a posteriori.

- **Diff non atomique** : 453 lignes pour 4 changements indépendants (recherche binaire de tolérance, exposant adaptatif `min_pick_fraction`, re-fusion moveout, dimensionnement `min_node_size`). Le dimensionnement `min_node_size` en particulier n'a aucun rapport avec le filtre moveout et aurait mérité son propre commit.

  **Statut : toujours ouvert** — à traiter au moment de committer (découper en commits thématiques), pas par une édition de code.

- ~~**~170 lignes de commentaires-rationale**~~ dans `try_tolerance` et le bloc `min_node_size`.

  **✅ Corrigé** (`dbclust/dbclust2pyocto.py`) : la rationale de fond (pourquoi une copie superficielle par tentative, pourquoi le dimensionnement adaptatif de `min_node_size`) a été déplacée dans les docstrings de `adjust_associator_tolerance()` et `dbclust2pyocto()` ; les commentaires inline ne gardent que ce qui aide à lire le code ligne par ligne (renvoi au docstring pour le "pourquoi", détail local pour le "quoi"). Le bloc `try_tolerance` est passé d'environ 30 lignes de commentaires à 5 ; le bloc `min_node_size` d'environ 15 à 3.

- ~~**Triplication de la formule haversine**~~ : `hav_km()` local dans `_refuse_fragment_events` réimplémentait `_haversine_km()` déjà présent plus bas dans le même fichier.

  **✅ Corrigé** (`dbclust/dbclust2pyocto.py`) : `hav_km()` local supprimé, `_refuse_fragment_events` appelle désormais `_haversine_km()` directement (Python résout les appels de fonctions module-level à l'exécution, donc l'ordre de définition dans le fichier n'a pas d'importance). Au passage, `_haversine_km()` a gagné le clip `np.clip(a, 0.0, 1.0)` avant `arcsin` que `hav_km()` avait et qu'elle n'avait pas — garde contre un léger dépassement de `1.0` par arrondi flottant sur des points quasi identiques ou antipodaux, qui aurait fait planter `arcsin`. `haversine_distance()` de `dbclust/localization_quality.py` (qui délègue en réalité à `geopy.distance.geodesic`, une formule ellipsoïdale différente, pas une simple haversine sphérique malgré le nom) a été volontairement laissée à part : usage différent (qualité de localisation, pas le chemin chaud du filtre moveout) et dépendance externe (`geopy`) que je ne voulais pas introduire dans `dbclust2pyocto.py` sans que ce soit demandé.

- ~~**Duplication de la clé de dédup**~~ : `_refuse_fragment_events` recopiait à la main la clé `(network, station, phase[0].upper(), time)` + règle de préférence « pick avec `event_id` » déjà implémentée dans `cluster_merge_one_pass()`.

  **✅ Corrigé** (`dbclust/dbclust2pyocto.py`) : nouveau helper module-level `_dedup_picks_prefer_event_id(picks)`, utilisé par les deux sites (`_refuse_fragment_events` et `cluster_merge_one_pass`). Une seule implémentation à maintenir désormais.

- ~~**`min_compat_fraction`** généralisé en fraction configurable mais jamais appelé qu'avec `1.0`.~~ **✅ Corrigé** dans la refonte de `_refuse_fragment_events` (section dédiée plus haut) : paramètre supprimé, `cross_moveout_ok` exige toujours 100% des paires.
- ~~**Recherche sans early-exit** dans `cross_moveout_ok`.~~ **✅ Corrigé** dans la même refonte : sortie anticipée dès la première paire incompatible.

- ~~**Aucun test**~~ pour `_tolerance_candidates`, `_refuse_fragment_events`, `_dedup_picks_prefer_event_id`, `_haversine_km`.

  **✅ Corrigé** : nouveau fichier [test_refuse_fragment_events.py](test_refuse_fragment_events.py) (12 tests), suivant la convention déjà en place dans le repo (scripts `test_*.py` autonomes exécutables directement, sans dépendre de PyOcto/NonLinLoc/ObsPy — le projet n'a pas de répertoire `tests/` ni de config pytest, donc pas de nouvelle convention imposée). Couvre : fusion d'un fragment moveout-compatible (cas positif), non-fusion d'un événement réellement distinct (cas négatif), le gate `frag_max_st` dérivé, le garde-fou `min_pairs_checked`, le cas `clusters` vide, la préférence `event_id` de `_dedup_picks_prefer_event_id` dans les deux ordres, une distance haversine connue (Strasbourg-Paris) et deux cas limites (distance nulle, points antipodaux identiques), et la construction de `_tolerance_candidates`. Exécutable directement (`python test_refuse_fragment_events.py`) ou via `pytest test_refuse_fragment_events.py` — les deux ont été vérifiés (12/12 passent). La propriété de monotonicité de la recherche binaire elle-même n'est **pas** testée (nécessiterait de mocker `dbclust2pyocto()`, plus lourd) — reste une piste non traitée dans cette passe.

---

## ✅ Cohérence positive (vérifiée, pas d'action requise)

- `leiden.py:60-105` : le nouveau log corrige un message erroné précédent (« PS-cross>d/vs » alors que les paires P-S ne sont en réalité jamais filtrées — `dt_max` vaut `np.inf` pour elles) ; le garde-fou à 50 % de suppression d'arêtes est une addition utile pour détecter un `vp`/`vs` mal calibré.
- La copie défensive dans `try_tolerance` (`copy.copy` + listes fraîches, pas de `deepcopy` des `Phase`/inventaires) est correcte : `dbclust2pyocto()` ne mute que `clusters`/`noise`/attributs de la copie, jamais un `Phase` ou son `info_sta` partagé ; `aggregate_pick_to_cluster_with_common_event_id` mute bien en place sans changer le nombre de clusters — le commentaire du chemin d'exception affirmant que `pyocto_clusters` reste le même objet que `newclust.clusters` est exact.
- L'alignement `pyocto_preloc` ↔ `pyocto_clusters` tient aux deux points d'appel de la re-fusion (même objet que `newclust.clusters`/`newclust.preloc`).
- `finally: associator.pick_match_tolerance = original_tolerance` (dbclust2pyocto.py:249-252) restaure bien correctement la tolérance partagée entre fenêtres — un point soulevé initialement en interne s'est révélé infondé après relecture directe.

---

## Recommandations prioritaires (ordre d'impact)

1. ~~**B1** — Sonder `candidates[0]` avant la recherche binaire.~~ **✅ Corrigé.**
2. ~~**B3** — Garde sur `z_span_km <= 0`.~~ **✅ Corrigé.**
3. **B4** — Valider sur données réelles que la re-fusion se déclenche bien sur les fragments connus du batch Alceste — seul moyen de trancher si ce bug logique est actif en pratique. **Toujours ouvert**, nécessite une validation empirique hors de portée d'une correction statique.
4. **B8** — Décider du sort de `min_node_size` dans la config (borne réintégrée ou suppression) : impact utilisateur direct, piège de configuration silencieux. **Toujours ouvert**, décision produit à trancher avec l'utilisateur avant d'implémenter.
5. **C6/C7** — Corriger l'incohérence 0.3/0.2 et le commentaire `1/sqrt(n)` faux dans `samples/config.yml` : coût minime, évite une mauvaise configuration en production. **Toujours ouvert.**
6. ~~**B2, B5, B6, B7, B9, B10** — correctifs de cohérence logique.~~ **✅ Tous corrigés** (B6 par documentation renforcée après investigation ayant montré que le comportement actuel était déjà correct — voir détail sous B6).
7. **C1–C5, complexité** — cleanup (dédup helpers, exposition config, découpage du commit, tests) : à planifier après les correctifs fonctionnels. **Toujours ouvert.**

### Bilan de cette passe de correction (2026-09-24)

Corrigés dans `dbclust/dbclust2pyocto.py`, chacun marqué `BUGFIX B<n>` en commentaire au point d'application : **B1, B2, B3, B5, B7, B9, B10** (correctifs de code) et **B6** (investigation + documentation renforcée, comportement déjà correct pour les valeurs par défaut actuelles). Vérifiés par import du module et tests fumée ciblés (logique de recherche de tolérance simulée en isolation, signature et robustesse de `_refuse_fragment_events` sur entrée vide). Aucun test automatisé formel (pytest) n'a été ajouté — reste couvert par le point complexité "Aucun test" ci-dessus.

**Restent ouverts** : B4 (validation empirique requise), B8 (décision produit requise), tous les points 🟠 Cohérence (C1–C7) et 🟡 Complexité excessive.
