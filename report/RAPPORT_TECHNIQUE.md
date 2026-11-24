# Rapport de projet — Data Mining
## Optimisation des hyperparamètres
### Détection d’intrusions réseau : comparaison de méthodes d’optimisation sur le jeu de données UNSW-NB15

**Réalisé par :**  
GHOUL Sami  
BELGHACHE Walid  
*4e année d’ingéniorat, Génie Logiciel*  
*Université de Béjaïa, Faculté des Sciences Exactes, Département d’Informatique*  
*Novembre 2026*  

---

### Résumé

Ce rapport présente une étude d’optimisation d’hyperparamètres appliquée à la détection d’intrusions réseau (NIDS), en classification binaire (trafic normal ou attaque), sur le jeu de données UNSW-NB15.

Après un prétraitement sans fuite de données (92 357 flux uniques, 186 variables ramenées à 46), six algorithmes sont évalués avec leurs paramètres par défaut. XGBoost et Random Forest sont ensuite optimisés par Grid Search, Random Search, optimisation bayésienne (Optuna, estimateur TPE) et algorithme génétique (DEAP). Les onze expériences sont comparées par validation croisée à 5 plis, avec le F1-score comme critère.

XGBoost optimisé par Random Search obtient le meilleur F1 (0,9180). Optuna (0,9179) et Grid Search (0,9176) en sont très proches, et XGBoost avec ses paramètres par défaut atteint déjà 0,9154 en 6,7 s. Sur le jeu de test isolé de 18 472 flux, le modèle retenu obtient 91,99 % d’accuracy, un rappel de 94,66 % sur les attaques et un ROC-AUC de 0,9824, avec 10,38 % de faux positifs. Le rapport décrit la méthode, les résultats, l’interprétation des décisions par valeurs SHAP et les limites de l’étude.

**Mots-clés :** détection d’intrusions, NIDS, UNSW-NB15, optimisation d’hyperparamètres, XGBoost, Optuna, Random Search, algorithme génétique, SHAP.

---

### Table des matières

1. Introduction
   - 1.1 Contexte
   - 1.2 Sujet et objectifs
   - 1.3 Organisation du rapport
2. État de l'art
3. Jeu de données et prétraitement
   - 3.1 Le jeu UNSW-NB15
   - 3.2 Nettoyage et découpage
   - 3.3 Encodage, standardisation et sélection
4. Méthodologie expérimentale
   - 4.1 Chaîne de traitement
   - 4.2 Protocole d'évaluation
   - 4.3 Algorithmes de base
   - 4.4 Méthodes d'optimisation
5. Résultats de la comparaison
   - 5.1 Baseline
   - 5.2 Effet de l'optimisation
   - 5.3 Coût de calcul
   - 5.4 Choix du modèle
6. Évaluation du modèle retenu
   - 6.1 Performances sur le jeu de test
   - 6.2 Matrice de confusion et courbe ROC
   - 6.3 Latence d'inférence
   - 6.4 Interprétation par valeurs SHAP
7. Limites et perspectives
   - 7.1 Limites actuelles
   - 7.2 Pistes d'amélioration
8. Conclusion
Bibliographie

---

## 1. Introduction

### 1.1 Contexte
Un système de détection d’intrusion réseau (NIDS, pour *Network Intrusion Detection System*) surveille le trafic d’un réseau et signale les flux qui ressemblent à une attaque. Les NIDS à signatures reconnaissent bien les attaques déjà cataloguées, mais ils laissent passer les variantes modifiées et les attaques inédites. L’apprentissage automatique propose une autre approche : un modèle apprend, à partir de flux étiquetés, ce qui distingue le trafic normal du trafic malveillant, puis il classe les nouveaux flux.

Choisir un algorithme ne suffit pas. Le même algorithme donne des résultats différents selon ses hyperparamètres (nombre d’arbres, profondeur, taux d’apprentissage, etc.), qui ne sont pas appris pendant l’entraînement et doivent être fixés avant. Les régler à la main prend du temps. Des méthodes de recherche automatique existent, avec des coûts de calcul très différents d’une méthode à l’autre [1].

Un NIDS doit en outre satisfaire trois exigences qui tirent dans des sens opposés : un rappel élevé, pour laisser passer peu d’attaques ; un faible taux de faux positifs, pour ne pas noyer les analystes d’un centre opérationnel de sécurité (SOC) sous de fausses alertes ; une latence d’inférence assez basse pour suivre le débit du réseau.

### 1.2 Sujet et objectifs
Le sujet demande de développer un modèle d’apprentissage automatique avec des hyperparamètres optimaux, pour un problème de data mining dans un contexte applicatif au choix, en testant plusieurs méthodes d’optimisation. Nous avons retenu le contexte Sécurité : détection d’intrusions réseau sur UNSW-NB15, en classification binaire. Le travail comprend les étapes suivantes :
1. préparer les données en évitant toute fuite d’information entre l’apprentissage et le test ;
2. évaluer six algorithmes avec leurs hyperparamètres par défaut (baseline) ;
3. optimiser les deux meilleurs avec quatre méthodes : Grid Search, Random Search, optimisation bayésienne (Optuna, TPE) et algorithme génétique (DEAP) ;
4. comparer les onze expériences obtenues sur la même validation croisée et retenir une configuration ;
5. évaluer cette configuration une seule fois sur un jeu de test isolé, puis interpréter ses décisions avec SHAP.

### 1.3 Organisation du rapport
La section 2 résume les travaux sur lesquels nous nous appuyons. La section 3 décrit le jeu de données et son prétraitement, et la section 4 la méthodologie expérimentale. La section 5 présente la comparaison des onze expériences, puis la section 6 l’évaluation du modèle retenu sur le jeu de test. La section 7 regroupe les limites de l’étude et les pistes d’amélioration.

---

## 2. État de l'art

Le jeu UNSW-NB15 a été construit par Moustafa et Slay au laboratoire Cyber Range de l’Australian Centre for Cyber Security. Son trafic mélange des activités normales réelles et des attaques synthétiques de neuf familles, produites avec l’outil IXIA PerfectStorm [2]. Il a été conçu pour remplacer KDD’99 et NSL-KDD, dont le trafic est ancien. Les mêmes auteurs ont ensuite comparé UNSW-NB15 et KDD’99 par une analyse statistique et des classifieurs, et ont montré que UNSW-NB15 est plus difficile à classer [3].

Pour les données tabulaires, le boosting de gradient fait partie des méthodes de référence. XGBoost [4] construit les arbres l’un après l’autre, ajoute des termes de régularisation à l’objectif et exploite efficacement les données creuses. Bhati et al. l’ont utilisé dans une technique de détection d’intrusion à base d’ensemble [5].

Côté optimisation, Yang et Shami [1] passent en revue les méthodes de recherche d’hyperparamètres : grille, recherche aléatoire, optimisation bayésienne et méthodes évolutionnaires. Bergstra et Bengio [6] ont montré qu’à budget égal la recherche aléatoire trouve souvent de meilleures configurations que la grille, parce que quelques hyperparamètres seulement pèsent vraiment sur le résultat. L’estimateur TPE (*Tree-structured Parzen Estimator*) [7] choisit les essais suivants d’après les résultats déjà obtenus ; il est disponible dans Optuna [8]. Les algorithmes génétiques, que nous utilisons avec la bibliothèque DEAP [9], font évoluer une population de configurations par sélection, croisement et mutation.

Pour expliquer les décisions d’un modèle, Lundberg et Lee [10] ont proposé SHAP, qui attribue à chaque variable une contribution à chaque prédiction à partir des valeurs de Shapley. SHAP dispose d’une variante pour les modèles à base d’arbres (TreeExplainer), applicable à XGBoost. Notre étude réunit ces éléments dans un même protocole : mêmes données, même validation croisée et même métrique pour toutes les méthodes d’optimisation.

---

## 3. Jeu de données et prétraitement

### 3.1 Le jeu UNSW-NB15
UNSW-NB15 décrit chaque connexion réseau par des attributs de flux, des attributs de base, de contenu et de temps, ainsi que par des compteurs calculés sur une fenêtre de 100 connexions. Les attaques appartiennent à neuf familles : Fuzzers, Analysis, Backdoors, DoS, Exploits, Generic, Reconnaissance, Shellcode et Worms. Le tableau 1 résume le corpus utilisé.

**Précision sur le fichier utilisé :** Le corpus de départ de notre travail est le fichier `UNSW_NB15_training-set.parquet` (175 341 enregistrements bruts). Le benchmark UNSW-NB15 complet publié par l'ACCS comprend environ 257 673 flux, répartis entre ce fichier et une seconde partition de 82 332 flux (`UNSW_NB15_testing-set.parquet`). Conformément au protocole de l'étude, nous sommes partis du jeu de 175 341 flux, au sein duquel nous avons éliminé les 82 984 enregistrements dupliqués pour obtenir 92 357 flux uniques, puis réalisé notre propre partitionnement étanche 80 % / 20 %.

**Tableau 1 : Caractéristiques du corpus**

| Élément | Valeur |
| :--- | :--- |
| Origine | Cyber Range Lab de l’ACCS, trafic généré avec IXIA PerfectStorm |
| Fichier source utilisé | `UNSW_NB15_training-set.parquet` (175 341 flux bruts) |
| Doublons supprimés | 82 984 |
| Flux uniques après nettoyage | 92 357 |
| Variables d’entrée | 42, dont trois catégorielles (`proto`, `service`, `state`) |
| Variable cible | `label` : 0 = normal (environ 52,9 %), 1 = attaque (environ 47,1 %) |
| Familles d’attaque | 9, regroupées dans la classe Attaque (détection binaire) |
| Jeu d’entraînement | 73 885 flux (80 %) |
| Jeu de test | 18 472 flux (20 %), dont 9 779 normaux et 8 693 attaques |

### 3.2 Nettoyage et découpage
Nous avons supprimé l’identifiant technique `id` et la colonne `attack_cat`, puisque la cible est binaire, puis les 82 984 enregistrements en double. Il reste 92 357 flux uniques. Les deux classes étant proches de l’équilibre (52,9 % normal et 47,1 % attaque), nous n’avons appliqué aucun rééchantillonnage de type SMOTE.

Le jeu est découpé en 80 % d’entraînement (73 885 flux) et 20 % de test (18 472 flux) avec `stratify=y` et `random_state=42`, ce qui conserve les proportions de classes. Le jeu de test n’intervient ni dans le choix des variables, ni dans l’ajustement du scaler, ni dans la recherche d’hyperparamètres. Il sert une seule fois, à la fin (section 6).

### 3.3 Encodage, standardisation et sélection
Le tableau 2 donne la suite des opérations appliquées. Chaque transformation est ajustée sur le jeu d’entraînement seul, puis appliquée telle quelle au jeu de test.

**Tableau 2 : Étapes du prétraitement**

| N° | Étape | Détail | Résultat |
| :---: | :--- | :--- | :--- |
| 1 | Découpage stratifié | 80 % entraînement, 20 % test, `stratify=y` (`seed=42`) | 73 885 / 18 472 flux |
| 2 | Encodage One-Hot | `OneHotEncoder(handle_unknown='ignore')` sur `proto`, `service`, `state` | 186 variables |
| 3 | Standardisation | `StandardScaler`, `fit_transform` sur les variables numériques d’entraînement, `transform` sur le test | moyenne 0, variance 1 |
| 4 | Sélection de variables | Calculée sur le jeu d’entraînement : élimination des corrélations linéaires supérieures à 0,95 et élagage par importance Random Forest inférieure à 0,0001 | 46 variables (environ 75 % de moins) |

Les variables conservées sont surtout des indicateurs de temps et de volume du flux : `tcprtt`, `ackdat`, `synack`, `dload`, `rate`, `sbytes` et `dbytes`, auxquels s’ajoutent `state_INT`, `state_CON` et `proto_tcp`, issues de l’encodage.

---

## 4. Méthodologie expérimentale

### 4.1 Chaîne de traitement
La figure 1 résume le déroulement. Chaque étape produit des mesures calculées de la même façon, ce qui permet de comparer directement les onze expériences : six baselines et cinq optimisations.

![Figure 1 : Chaîne de traitement de l'étude](figures/figure1_pipeline.png)

### 4.2 Protocole d’évaluation
Toutes les expériences utilisent la même validation croisée stratifiée à 5 plis (`StratifiedKFold`, $k=5$, `random_state=42`) sur les 73 885 flux d’entraînement. Les métriques du tableau 3 sont calculées avec la classe Attaque comme classe positive. Nous optimisons le F1-score, parce qu’un NIDS doit à la fois détecter les attaques (rappel) et limiter les fausses alertes (précision). Le temps indiqué pour une expérience est la durée totale de la recherche, validation croisée comprise, mesurée sur notre machine.

**Tableau 3 : Métriques utilisées**

| Métrique | Définition | Lecture en détection d’intrusions |
| :--- | :--- | :--- |
| Accuracy | $(TP+TN)/\text{total}$ | part des flux correctement classés |
| Précision | $TP/(TP+FP)$ | part des alertes qui sont de vraies attaques |
| Rappel | $TP/(TP+FN)$ | part des attaques détectées |
| F1-score | $2 \cdot P \cdot R / (P + R)$ | compromis entre précision et rappel |
| Taux de faux positifs (FPR) | $FP/(FP+TN)$ | part du trafic normal signalée à tort |
| ROC-AUC | aire sous la courbe ROC | capacité à séparer les deux classes, quel que soit le seuil |

### 4.3 Algorithmes de base
Six algorithmes de familles différentes servent de point de départ, avec leurs hyperparamètres par défaut (tableau 4). Les implémentations viennent de scikit-learn [12], sauf XGBoost [4].

**Tableau 4 : Algorithmes de la baseline**

| Algorithme | Famille | Configuration par défaut |
| :--- | :--- | :--- |
| Régression logistique | modèle linéaire | `max_iter=1000` |
| SVM | méthode à noyau | `kernel='rbf'`, `C=1.0` |
| KNN | distances | `n_neighbors=5` |
| Arbre de décision | arbre seul (CART) | `criterion='gini'` |
| Random Forest | ensemble par bagging [11] | `n_estimators=100` |
| XGBoost | boosting de gradient [4] | `n_estimators=100`, `max_depth=6` |

### 4.4 Méthodes d’optimisation
Les deux meilleurs algorithmes de la baseline, XGBoost et Random Forest, sont optimisés. Chaque méthode maximise le F1 moyen de la validation croisée à 5 plis (tableau 5).

**Tableau 5 : Méthodes d’optimisation et budgets**

| Méthode | Modèle | Budget | Espace de recherche |
| :--- | :--- | :--- | :--- |
| Grid Search (`GridSearchCV`) | XGBoost | 18 combinaisons | `n_estimators` $\in \{100, 200, 300\}$, `max_depth` $\in \{6, 8, 10\}$, `learning_rate` $\in \{0.05, 0.1\}$ |
| Random Search (`RandomizedSearchCV`) | XGBoost | 20 tirages | 6 hyperparamètres : `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight` |
| Random Search (`RandomizedSearchCV`) | Random Forest | 20 tirages | 5 hyperparamètres : `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` |
| Optuna (échantillonneur TPE) | XGBoost | 15 essais | 8 hyperparamètres discrets et continus, dont `reg_alpha` (L1) et `reg_lambda` (L2) |
| Algorithme génétique (DEAP) | XGBoost | 18 évaluations (6 individus, 3 générations) | hyperparamètres de XGBoost, dont `n_estimators`, `max_depth`, `learning_rate` et `subsample` |

Dans l’algorithme génétique, une configuration XGBoost est un individu. Les parents sont choisis par tournoi, croisés par un croisement Blend (probabilité 0,6) puis modifiés par une mutation gaussienne (probabilité 0,3). Random Forest n’a été optimisé que par Random Search pour des raisons de coût de calcul : ses 20 tirages ont nécessité 1 772,49 s (environ 88,6 s par tirage en validation croisée 5 plis) ; étendre l'optimisation par grille ou par algorithme bayésien/génétique sur Random Forest aurait représenté un coût de calcul prohibitif face à la rapidité de convergence de XGBoost.

---

## 5. Résultats de la comparaison

### 5.1 Baseline
Les performances de la baseline s’échelonnent de 0,8720 pour la régression logistique à 0,9154 pour XGBoost (tableau 6, lignes 6 à 11). Le F1 progresse régulièrement du linéaire à l’arbre, puis aux ensembles, ce qui indique des frontières de décision fortement non linéaires dans ces données.

Le SVM demande un commentaire. Son rappel est le plus élevé de toute l’étude (0,9934), mais sa précision tombe à 0,8220 : il classe beaucoup de trafic normal comme attaque. Sa validation croisée dure aussi 1 121,75 s, contre 6,67 s pour XGBoost. Son F1 (0,8996) le place derrière les deux méthodes d’ensemble.

**Tableau 6 : Résultats des onze expériences (validation croisée à 5 plis, classés par F1)**

| N° | Modèle | Méthode | Hyperparamètres retenus | Acc. | Préc. | Rappel | F1 | Temps (s) |
|:---:|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | XGBoost | Random Search | `n_est=150, depth=7, lr=0.1, sub=1.0, col=0.9, mcw=1` | 0,9205 | 0,8920 | 0,9455 | 0,9180 | 168,99 |
| 2 | XGBoost | Optuna (TPE) | `n_est=191, depth=9, lr=0.0414, sub=0.854, col=0.878` | 0,9202 | 0,8896 | 0,9480 | 0,9179 | 103,80 |
| 3 | XGBoost | Grid Search | `n_est=200, depth=8, lr=0.05` | 0,9199 | 0,8895 | 0,9475 | 0,9176 | 115,26 |
| 4 | Random Forest | Random Search | `n_est=150, split=5, leaf=4, max_feat=0.5, depth=None` | 0,9194 | 0,8943 | 0,9399 | 0,9165 | 1 772,49 |
| 5 | XGBoost | Algo. génétique | `n_est=239, depth=5, lr=0.1009, sub=0.7` | 0,9187 | 0,8899 | 0,9441 | 0,9162 | 124,95 |
| 6 | XGBoost | Baseline | `n_est=100, depth=6` | 0,9185 | 0,8947 | 0,9370 | 0,9154 | 6,67 |
| 7 | Random Forest | Baseline | `n_est=100` | 0,9142 | 0,8885 | 0,9349 | 0,9111 | 29,98 |
| 8 | SVM (RBF) | Baseline | `C=1.0, noyau RBF` | 0,8956 | 0,8220 | 0,9934 | 0,8996 | 1 121,75 |
| 9 | Arbre de décision | Baseline | `criterion=gini` | 0,8972 | 0,8933 | 0,8875 | 0,8904 | 7,33 |
| 10 | KNN | Baseline | `k=5` | 0,8839 | 0,8498 | 0,9151 | 0,8812 | 23,47 |
| 11 | Régression log. | Baseline | `max_iter=1000` | 0,8701 | 0,8126 | 0,9409 | 0,8720 | 22,57 |

*Note : n_est = n_estimators ; depth = max_depth ; lr = learning_rate ; sub = subsample ; col = colsample_bytree ; mcw = min_child_weight ; split = min_samples_split ; leaf = min_samples_leaf. Pour Optuna, seuls les cinq hyperparamètres principaux sont listés. Temps mesurés sur notre machine.*

![Figure 2 : F1-score moyen des onze expériences](figures/figure2_f1_scores.png)

### 5.2 Effet de l’optimisation
Pour XGBoost, les quatre méthodes améliorent la baseline, mais de peu. Le F1 passe de 0,9154 à 0,9180 avec Random Search, soit 0,26 point. Les trois meilleures méthodes sont à moins de 0,0005 les unes des autres (0,9180, 0,9179 et 0,9176). Nous n’avons pas relevé l’écart type entre plis. Nous ne pouvons donc pas dire si ces différences sont significatives, et nous les traitons comme une égalité pratique. L’algorithme génétique (0,9162) fait un peu moins bien. Son budget, 6 individus sur 3 générations, laisse peu de place à l’évolution, et nous n’avons pas testé de budget plus grand.

Le gain porte surtout sur le rappel. Avec Random Search, il passe de 0,9370 à 0,9455 (+0,85 point), alors que la précision recule de 0,8947 à 0,8920. Les configurations optimisées détectent davantage d’attaques et déclenchent un peu plus de fausses alertes. Pour Random Forest, Random Search fait monter le F1 de 0,9111 à 0,9165 (+0,54 point), un gain plus net que pour XGBoost, mais le modèle reste légèrement derrière XGBoost optimisé.

La figure 3 montre la progression d’Optuna lors de sa recherche séquentielle. Le premier essai donne déjà un F1 d’environ 0,9168, le meilleur score au cours de la recherche séquentielle est atteint au quatrième essai (0,9177) et les onze essais suivants ne l’améliorent pas. Lors de la réévaluation standardisée du tableau 6 avec les hyperparamètres retenus sous `cross_validate`, XGBoost Optuna atteint un F1 de 0,9179. Au bout de ses 15 essais, Optuna a pris 103,80 s, soit 38 % de moins que Random Search (168,99 s), pour un F1 équivalent.

![Figure 3 : Progression d'Optuna sur ses 15 essais](figures/figure3_optuna_progression.png)

### 5.3 Coût de calcul
Le temps de recherche varie beaucoup plus que le F1 (figure 4). XGBoost avec ses paramètres par défaut demande 6,67 s. Random Search sur le même modèle demande 168,99 s, soit 25 fois plus, pour un gain de 0,26 point de F1. Par configuration évaluée (une validation croisée complète), Grid Search, Optuna et l’algorithme génétique coûtent entre 6,4 et 6,9 s, et Random Search 8,5 s. Random Forest est d’un autre ordre de grandeur, avec 88,6 s par tirage (1 772,49 s au total).

![Figure 4 : Temps de recherche et F1-score moyen](figures/figure4_search_time_vs_f1.png)

Dans notre expérience, le coût dépend donc davantage du choix du modèle que de la méthode de recherche. Parmi les méthodes appliquées à XGBoost, Optuna est la moins coûteuse en temps total (103,80 s).

### 5.4 Choix du modèle
Nous retenons XGBoost optimisé par Random Search. Il a le meilleur F1 (0,9180) et la meilleure accuracy (0,9205) de l’étude. Optuna donne presque le même résultat en moins de temps et aurait été un choix tout aussi défendable. La configuration retenue est :  
`n_estimators = 150`, `max_depth = 7`, `learning_rate = 0.1`, `subsample = 1.0`, `colsample_bytree = 0.9` et `min_child_weight = 1`.  
Ce choix repose sur la validation croisée du jeu d’entraînement, pas sur le jeu de test.

---

## 6. Évaluation du modèle retenu

### 6.1 Performances sur le jeu de test
Le modèle retenu est réentraîné sur les 73 885 flux d’entraînement (1,37 s), puis évalué une seule fois sur les 18 472 flux du jeu de test (tableau 7).

**Tableau 7 : Résultats sur le jeu de test (18 472 flux)**

| Métrique | Valeur sur le test | Valeur en validation croisée |
| :--- | :---: | :---: |
| Accuracy | 91,99 % | 92,05 % |
| ROC-AUC | 0,9824 | non calculé |
| Précision (classe Attaque) | 0,8902 | 0,8920 |
| Rappel (classe Attaque) | 0,9466 | 0,9455 |
| F1-score (classe Attaque) | 0,9175 | 0,9180 |
| Macro : précision / rappel / F1 | 0,9200 / 0,9214 / 0,9199 | non calculé |
| Pondérée : précision / rappel / F1 | 0,9217 / 0,9199 / 0,9200 | non calculé |
| Taux de faux positifs (FPR) | 10,38 % | non calculé |
| Taux de faux négatifs (FNR) | 5,34 % | non calculé |

Les valeurs obtenues sur le test sont proches de celles de la validation croisée (F1 de 0,9175 contre 0,9180, rappel de 0,9466 contre 0,9455). La recherche d’hyperparamètres ne semble donc pas avoir provoqué de surapprentissage.

### 6.2 Matrice de confusion et courbe ROC
Sur 9 779 flux normaux, 8 764 sont bien classés et 1 015 déclenchent une fausse alerte, soit un FPR de 10,38 %. Sur 8 693 attaques, 8 229 sont détectées et 464 sont manquées, soit un taux de faux négatifs de 5,34 %. Pour un SOC, 1 015 fausses alertes sur 18 472 flux représentent un volume élevé. Le seuil de décision peut être déplacé pour réduire ce chiffre (section 7).

![Figure 5 : Matrice de confusion sur le jeu de test](figures/figure5_confusion_matrix.png)

La courbe ROC monte presque verticalement au départ, pour une aire de 0,9824. Elle atteint un taux de vrais positifs de 1,0 pour un taux de faux positifs d’environ 0,19. Un seuil plus tolérant détecterait donc toutes les attaques, mais au prix de beaucoup plus de fausses alertes. Le seuil actuel se situe sur la même courbe, à un FPR de 10,38 % pour un rappel de 94,66 %.

![Figure 6 : Courbe ROC du modèle retenu sur le jeu de test](figures/figure6_roc_curve.png)

### 6.3 Latence d’inférence
La prédiction sur les 18 472 flux du jeu de test prend entre 42,15 et 48,50 ms, soit 2,28 à 2,63 µs par flux, ce qui correspond à plus de 380 000 flux par seconde sur un processeur standard. Cette mesure porte sur le modèle seul : elle n’inclut ni la capture du trafic, ni le calcul des attributs de flux.

### 6.4 Interprétation par valeurs SHAP
Nous avons croisé l’importance par gain interne de XGBoost avec le diagramme SHAP (calculé avec TreeExplainer). Le gain place `state_INT` en tête, alors que le diagramme SHAP, qui classe les variables par impact moyen, met `ackdat`, `dload` et `tcprtt` aux trois premières places. Les deux classements ne coïncident pas exactement, car ils ne mesurent pas la même chose. `state_INT` correspond à des connexions interrompues, un comportement attendu pour des scans de ports ou certains dénis de service.

![Figure 7 : Diagramme SHAP (summary plot) du modèle retenu](figures/figure7_shap_summary.png)

Un SHAP positif pousse la prédiction vers la classe Attaque et un SHAP négatif vers le trafic normal :
- Des valeurs élevées de `ackdat` donnent un SHAP positif, et des valeurs faibles un SHAP négatif. Il en va de même pour `tcprtt`, dont les valeurs faibles se regroupent autour de -1,4 et les valeurs élevées autour de +0,5. Ces deux variables mesurent la durée de la poignée de main TCP : dans ce jeu, des échanges anormalement lents sont associés aux attaques.
- La variable `dload` a l’effet le plus marqué dans l’autre sens. Ses valeurs élevées ont un SHAP très négatif (environ -4) et poussent fortement vers le trafic normal, alors que ses valeurs faibles ont des effets plus dispersés.
- Des tailles moyennes de paquets élevées (`dmean`, `smean`) vont vers la classe Attaque.
- À l’inverse, des valeurs élevées de `service_-` (service non identifié), de `ct_src_dport_ltm` et de `swin` vont vers le trafic normal.

Ces effets ont un sens réseau plausible, mais SHAP décrit le comportement du modèle et ne prouve aucune cause.

---

## 7. Limites et perspectives

### 7.1 Limites actuelles
- Le taux de faux positifs de 10,38 % reste trop élevé pour un réseau à fort trafic.
- Les gains de l’optimisation sont faibles (0,26 point de F1 pour XGBoost) et nous n’avons pas calculé d’écart type entre plis. Les différences entre les trois meilleures méthodes ne sont donc pas testées statistiquement.
- Les budgets d’optimisation sont petits et inégaux (de 15 à 20 configurations), et Random Forest n’a été optimisé que par Random Search.
- La sélection des variables est faite sur l’ensemble d’entraînement avant la validation croisée. Les scores de validation croisée peuvent en être légèrement optimistes. Le jeu de test n’est pas concerné.
- L’étude porte sur un seul jeu de données, fait de trafic généré en laboratoire, et sur un seul découpage entraînement et test. La classification binaire ne distingue pas les familles d’attaque. Le jeu est statique, sans dérive de trafic dans le temps.

### 7.2 Pistes d’amélioration
1. Régler le seuil de décision à partir de la courbe précision-rappel, selon la politique du SOC : un seuil plus haut réduit les fausses alertes au prix d’un rappel plus faible, et un seuil plus bas fait l’inverse.
2. Répéter la validation croisée avec plusieurs graines et relever l’écart type, pour savoir si les écarts entre méthodes d’optimisation sont réels, avec des budgets égaux et plus larges.
3. Passer à une classification hiérarchique : un second modèle identifierait la famille d’attaque (`attack_cat`) des flux déjà détectés.
4. Tester le modèle sur un autre jeu, par exemple CIC-IDS2017, pour mesurer sa capacité à se généraliser, et utiliser un autoencodeur non supervisé pour compléter XGBoost et repérer des attaques jamais vues.

---

## 8. Conclusion

Nous avons construit un pipeline de détection d’intrusions sur UNSW-NB15, de la préparation des données à l’interprétation des décisions, et comparé onze expériences : six algorithmes avec leurs paramètres par défaut et cinq optimisations d’hyperparamètres. XGBoost optimisé par Random Search arrive en tête (F1 de 0,9180 en validation croisée), à égalité pratique avec Optuna et Grid Search.

Le passage de la régression logistique (F1 de 0,8720) à XGBoost avec ses paramètres par défaut (0,9154) apporte 4,3 points. L’optimisation des hyperparamètres en ajoute 0,26 pour XGBoost, au prix d’un temps de recherche 25 fois plus long. Sur les 18 472 flux du jeu de test, le modèle retenu obtient 91,99 % d’accuracy, un F1 macro de 0,9199 et un ROC-AUC de 0,9824. Il détecte 94,66 % des attaques, avec 10,38 % de faux positifs et 2,3 µs de prédiction par flux.

---

## Bibliographie

[1] L. Yang et A. Shami, « On hyperparameter optimization of machine learning algorithms: theory and practice », *Neurocomputing*, vol. 415, 2020, p. 295-316.  
[2] N. Moustafa et J. Slay, « UNSW-NB15: a comprehensive dataset for network intrusion detection systems (UNSW-NB15 network dataset) », *Military Communications and Information Systems Conference (MilCIS)*, IEEE, 2015, p. 1-6.  
[3] N. Moustafa et J. Slay, « The evaluation of network anomaly detection systems: statistical analysis of the UNSW-NB15 dataset and the comparison with the KDD99 dataset », *Information Security Journal: A Global Perspective*, vol. 25, n° 1-3, 2016, p. 18-31.  
[4] T. Chen et C. Guestrin, « XGBoost: a scalable tree boosting system », *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 2016, p. 785-794.  
[5] B. S. Bhati, G. Chugh, F. Al-Turjman et N. S. Bhati, « An improved ensemble based intrusion detection technique using XGBoost », *Transactions on Emerging Telecommunications Technologies*, vol. 32, n° 6, 2021, e4076.  
[6] J. Bergstra et Y. Bengio, « Random search for hyper-parameter optimization », *Journal of Machine Learning Research*, vol. 13, 2012, p. 281-305.  
[7] J. Bergstra, R. Bardenet, Y. Bengio et B. Kégl, « Algorithms for hyper-parameter optimization », *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 24, 2011, p. 2546-2554.  
[8] T. Akiba, S. Sano, T. Yanase, T. Ohta et M. Koyama, « Optuna: a next-generation hyperparameter optimization framework », *Proceedings of the 25th ACM SIGKDD*, 2019, p. 2623-2631.  
[9] F.-A. Fortin, F.-M. De Rainville, M.-A. Gardner, M. Parizeau et C. Gagné, « DEAP: evolutionary algorithms made easy », *Journal of Machine Learning Research*, vol. 13, 2012, p. 2171-2175.  
[10] S. M. Lundberg et S.-I. Lee, « A unified approach to interpreting model predictions », *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, p. 4765-4774.  
[11] L. Breiman, « Random forests », *Machine Learning*, vol. 45, n° 1, 2001, p. 5-32.  
[12] F. Pedregosa et al., « Scikit-learn: machine learning in Python », *Journal of Machine Learning Research*, vol. 12, 2011, p. 2825-2830.
