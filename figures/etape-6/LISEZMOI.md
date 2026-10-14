# Étape 6 — Décrire la réponse : effet de chaque seuil sur données réelles

Chaque image compare plusieurs valeurs d'un seul réglage. Tout le reste garde la valeur proposée.
En rouge : la valeur proposée aujourd'hui. En gris : les autres possibilités.

Données utilisées (déjà publiques, déjà sur le disque) :

- **CCEP** : ds004080 (6 patients, 7 enregistrements) et ds004977 (4 patients). On prend la moyenne des essais. Dans chaque enregistrement, on garde les 10 voies qui répondent le plus et 10 voies tirées au hasard (graine 0). Cela fait 220 voies. Le preset est `ccep` 1.1.
- **ERP CORE** : les ondes de différence publiées des 7 composantes, pour 40 sujets. Cela fait 279 ondes, car une onde n'existe pas chez un sujet. On les compare à la latence publiée : le moment où 50 % de l'aire du signal brut est atteinte.
- **TEP** : ds002094, 20 sujets, avec le nettoyage publié (TESA). Pour chaque sujet, on prend la voie qui répond le plus entre 38 et 140 ms. Le preset est `tep` 1.1.

Pour refaire les images, lancer `python make_etape6.py`. Le cache des ondes ERP se reconstruit avec `build_erp.py`.

---

## 6-01-formes.png — Formes de bosse permises

On compare trois choix : gaussienne seule, asymétrique seule, ou les deux (proposé). Test sur 220 voies CCEP de 10 patients.
Avec la gaussienne seule, il faut plus de bosses (environ 2,4 au lieu de 2,3), et l'ajustement est moins bon sur une partie des voies (R² plus bas).

## 6-02-fond.png — Forme du fond sous les bosses

On compare quatre fonds : aucun, constant, ligne droite, exponentiel (proposé). Test sur 220 voies CCEP de 10 patients.
Sans fond, le modèle ajoute des bosses pour imiter la dérive : environ 3 bosses par voie au lieu de 2,3. Avec le fond exponentiel, il en faut le moins, et le R² est le plus élevé.

## 6-03-largeurs-max-bosses.png — Bornes : nombre maximum de bosses et largeur maximale

On fait varier le maximum de bosses (2 à 10) et la largeur maximale (50 à 500 ms). Test sur 220 voies CCEP de 10 patients.
Avec un maximum de 6 (proposé), moins de 2 % des voies l'atteignent : la borne ne bloque presque jamais. Une largeur maximale de 50 ou 100 ms découpe les bosses lentes en plusieurs morceaux (3,2 bosses au lieu de 2,3).

## 6-04-delta-bic.png — Règle pour garder une bosse de plus (ΔBIC)

On fait varier le seuil ΔBIC : 2, 5, 10 (proposé), 20, 40, et « calibré » (seuil réglé sur des simulations). Test sur 220 voies CCEP de 10 patients.
Plus le seuil est sévère, moins on garde de bosses. Sur les voies tirées au hasard, on passe de 0,66 bosse (seuil 2) à 0,26 (seuil 40). Une N1 apparaît dans 5,5 % des voies au hasard avec le seuil 2, contre 2 % avec le seuil 10. Sur les voies qui répondent, la part des N1 bouge peu jusqu'au seuil 20, puis chute au seuil 40 (28 %).

## 6-05-seuil-candidat.png — Seuil des candidats de départ

On fait varier le seuil de départ d'une bosse candidate, de 1 à 4 σ. Test sur 220 voies CCEP de 10 patients.
L'effet est faible sur le nombre de bosses (de 2,4 à 2,1) et sur les N1 (de 42 % à 38 %). L'effet est surtout sur le temps de calcul : il est divisé par plus de 2 entre 1 σ et 4 σ.

## 6-06-fusion.png — Fusion de deux bosses proches

On fait varier le facteur de fusion c, de 0 à 2. Deux bosses sont fusionnées si leur écart est plus petit que c × (σ₁ + σ₂). Test sur 220 voies CCEP de 10 patients.
Sur ces données, ce réglage ne change presque rien : environ 2,3 bosses quelle que soit la valeur, et le même écart entre bosses. C'est un choix peu sensible.

## 6-07-paires-opposees.png — Paires de bosses opposées qui s'annulent

On compare deux choix : garder ou rejeter deux bosses proches de signes opposés (+ et −). Test sur 220 voies CCEP, puis sur les ondes ERP CORE calées sur le stimulus (preset `erp` complet).
Rejeter ces paires ne change pas le nombre moyen de bosses. Mais la part des ondes qui contiennent une telle paire baisse : de 14,5 % à 9,5 % en CCEP, et de 6,5 % à 3,5 % en ERP.

## 6-08-definition-latence.png — Définition de la latence

On compare trois définitions de la latence : le pic brut du signal, le sommet de la bosse du modèle, et le moment où 50 % de l'aire du modèle est atteinte (proposé pour les ERP). Test sur 279 ondes ERP CORE (40 sujets × 7 composantes).
La définition « 50 % de l'aire » donne 92 à 100 % des sujets à ± 10 ms de la valeur publiée. Le pic brut et le sommet du modèle s'en écartent beaucoup pour P3 et N400 : seulement 7 à 17 % des sujets à ± 10 ms. La définition peut donc déplacer la latence de plusieurs dizaines de ms.

## 6-09-debut-fin.png — Début et fin d'une bosse

On fait varier le seuil de début et de fin : 5, 10 (proposé), 20 ou 50 % de la hauteur. Test sur 279 ondes ERP CORE (40 sujets × 7 composantes).
La durée rapportée dépend fortement de ce seuil. Pour la P3, elle va d'environ 430 ms (seuil 5 %) à 195 ms (seuil 50 %). Il faut fixer ce seuil avant de comparer des durées entre études.

## 6-10-mesure-ciblee.png — Mesure ciblée d'une composante ERP

On fait varier la marge autour de la fenêtre publiée (0 à 50 ms) et le nombre de parties de la bosse (1 ou 3). Ensuite on fait varier la tolérance de la comparaison. Test sur 279 ondes ERP CORE (40 sujets).
Avec 3 parties, l'accord est meilleur qu'avec 1 partie (P3 : 92 % contre 75 %). La marge de ± 25 ms (proposée) donne le meilleur accord global. La courbe de droite montre qu'au-delà de 10 ms, élargir la tolérance n'ajoute presque plus de sujets.

## 6-11-fenetres-tep.png — Fenêtres des composantes TEP N45 et N100

On compare trois jeux de fenêtres :
- proposé : N45 de 38 à 55 ms, N100 de 80 à 140 ms ;
- étroit : N45 de 40 à 50 ms, N100 de 90 à 120 ms ;
- large : N45 de 30 à 60 ms, N100 de 70 à 160 ms.

Test sur 20 sujets TMS-EEG (ds002094, nettoyage TESA).
La N45 n'a une bosse négative dans sa fenêtre que chez 10 à 20 % des sujets, quelle que soit la fenêtre. Ce n'est donc pas surtout un problème de fenêtre. Pour la N100, on passe de 45 % (fenêtre proposée) à 55 % (fenêtre large).

## 6-12-essai-par-essai.png — Décalage maximal essai par essai

On fait varier le décalage maximal permis à chaque essai, de 0,1 à 2 fois la largeur de la bosse. Test sur la première bosse de 89 voies CCEP qui répondent (10 patients, 10 à 12 essais par voie).
Avec 0,25 × la largeur (proposé pour l'intracrânien), 17 % des essais sont bloqués à la limite. Avec 0,5 × la largeur, c'est 9 %. Mais la variabilité trouvée reste proche de celle que produit le bruit seul, quelle que soit la limite. Avec 10 à 12 essais, on ne peut donc pas mesurer une vraie variabilité de latence.
