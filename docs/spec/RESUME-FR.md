# evoparam — Résumé de la spécification en français simple

*Ce document résume `SPEC.md` (en anglais) avec des mots simples. Rien n'est enlevé sur le fond : chaque choix, chaque limite et chaque exception importante est gardé. Chaque mot technique est gardé tel quel (pour pouvoir le chercher), puis expliqué tout de suite.*

Auteur : Yann V. Bellec. Date : 7 octobre 2026. Statut : privé, rien n'est publié avant la décision de HIIE (le bureau de transfert de technologie de l'UAB).

---

## 0. Le vocabulaire de base

- **Réponse évoquée** (*evoked response*) : la petite onde électrique que le cerveau (ou un nerf) produit juste après un stimulus. Exemple : on envoie un petit courant électrique dans le cerveau, et 20 millisecondes plus tard, une électrode voisine voit une vague. Cette vague est la réponse évoquée.
- **ms** : milliseconde, un millième de seconde.
- **µV** : microvolt, un millionième de volt. Les réponses du cerveau font quelques µV à quelques centaines de µV.
- **Composante** (*component*) : une « bosse » ou un « creux » nommé dans la réponse. Exemple : N1 = le premier creux négatif, N2 = le deuxième.
- **Latence** : le moment où la bosse est la plus haute (en secondes après le stimulus).
- **Amplitude** : la hauteur de la bosse (en µV), avec un signe (+ ou −).
- **Largeur / FWHM** (*full width at half maximum*) : la largeur de la bosse mesurée à mi-hauteur. Comme mesurer la largeur d'une colline à mi-pente.
- **Essai** (*trial*) : une répétition du stimulus. On répète souvent 10 à 100 fois, puis on fait la moyenne pour enlever le bruit.
- **Canal** (*channel*) : une électrode (un point de mesure).
- **Bruit** : tout ce qui n'est pas la réponse (activité du cerveau en fond, électronique, secteur 50/60 Hz).
- **Ligne de base** (*baseline*) : le morceau de signal juste avant le stimulus. Il sert de référence : « à quoi ressemble le signal quand rien ne se passe ».

Les modalités (types d'enregistrement) couvertes :
- **ERP** (*event-related potential*) : réponse mesurée sur le cuir chevelu (EEG) après un son, une image, etc.
- **SPES / CCEP** (*single pulse electrical stimulation / cortico-cortical evoked potential*) : on envoie une impulsion électrique dans le cerveau par une électrode implantée (chez des patients épileptiques avant chirurgie), et on mesure la réponse sur les autres électrodes. N1 arrive vers 10–50 ms, N2 vers 50–300 ms.
- **ECoG** : électrodes posées à la surface du cerveau. **sEEG** : électrodes plantées en profondeur, comme des aiguilles.
- **TEP** (*TMS-evoked potential*) : réponse EEG après une impulsion magnétique sur le crâne (TMS = stimulation magnétique transcrânienne).
- **ERNA** (*evoked resonant neural activity*) : une petite oscillation rapide (200–450 Hz) qui suit chaque impulsion de stimulation cérébrale profonde (**DBS**, *deep brain stimulation*, utilisée dans la maladie de Parkinson).
- **ECAP** (*evoked compound action potential*) : la réponse d'un faisceau de fibres nerveuses. Utilisée dans les stimulateurs de moelle épinière (**SCS**) et les implants cochléaires.
- **SEP** : réponse somatosensorielle (après stimulation d'un nerf du bras ou de la jambe).
- **LFP** (*local field potential*) : le signal électrique « lent » mesuré dans le cerveau des animaux.

---

## 1. Vision : le problème et la promesse

### Le problème
Tout le monde mesure des réponses évoquées, mais chaque labo a ses propres règles.
- En CCEP, une revue de 187 études montre que **49,7 %** n'expliquent pas comment elles décident qu'il y a une réponse. Parmi celles qui l'expliquent, **68,1 %** utilisent un simple seuil d'amplitude.
- Les seuils vont de **2,6 à 6 écarts-types** (*SD*, *standard deviation* : la taille typique du bruit). Les fenêtres commencent entre **5 et 16 ms**.
- Deux experts humains qui regardent les mêmes courbes ne sont d'accord qu'avec un **κ ≈ 0,60**. Le **κ de Cohen** mesure l'accord entre deux juges : 1 = toujours d'accord, 0 = d'accord seulement par hasard. 0,60 = accord moyen.
- Pour les ERP sur le scalp, les bonnes méthodes de mesure existent surtout en **MATLAB** (un logiciel payant). En Python, la bibliothèque principale (**MNE-Python**) n'a qu'un chercheur de maximum. Une demande d'outil de mesure ERP y est ouverte depuis 2020.
- En Python, il manque aussi : la mesure essai par essai, le nettoyage des artefacts de stimulation, un test « y a-t-il une réponse ? », et une incertitude sur chaque nombre.

### Le cas concret qui a tout déclenché : le N1 perdu
Le code de départ (le fork d'ERPparam) trouve le N1 sur **seulement 2 canaux sur 46** dans l'exemple public (jeu de données **ds004080**, sujet `ccepAgeUMCU01`, stimulation entre les contacts PT01 et PT02). Les causes, trouvées dans le code :
1. une règle qui jette une bosse si elle touche le bord de la fenêtre (le N1 est souvent tout au début de la fenêtre, donc au bord) ;
2. une règle qui jette une bosse si elle est trop étroite (le N1 est étroit : 4 à 8 ms) ;
3. une règle qui fusionne puis supprime deux bosses proches (le N1 est souvent suivi de près par le N2, du même signe) ;
4. un seuil d'acceptation qui laisse passer **93 %** des fenêtres de pur bruit, ce qui oblige ensuite à élaguer trop fort.
La spec remplace toute cette logique (voir §5.5 plus bas) et rend la correction **mesurable** (REQ-113 : un critère chiffré à atteindre).

### La promesse
**evoparam transforme n'importe quelle réponse évoquée, chez n'importe quelle espèce et avec n'importe quel type d'enregistrement, en une courte liste de composantes nommées (latence, amplitude, largeur, forme, probabilité d'existence, incertitude), en un seul appel de fonction, avec un paragraphe « Méthodes » prêt à coller dans l'article.**

Cinq engagements :
1. **Un seul modèle pour toutes les modalités.** Les presets (réglages tout prêts) changent les chiffres, pas le code.
2. **D'abord prouver qu'il y a une réponse, ensuite la décrire.** Avec un taux de fausses alarmes contrôlé.
3. **Chaque nombre a une incertitude et une traçabilité** (*provenance* : l'historique complet de ce qui a été fait, qui permet de refaire exactement le calcul).
4. **Essais, canaux et sujets sont des niveaux d'un même modèle**, pas des scripts séparés.
5. **Léger par défaut.** Seulement **NumPy** et **SciPy** (les deux bibliothèques de calcul de base en Python) sont obligatoires. Le reste est optionnel (*extras*).

### La phrase que les utilisateurs citeront
Le logiciel génère lui-même un paragraphe du type : « Les réponses évoquées ont été paramétrées avec evoparam (version X.Y.Z ; DOI …) avec le preset `ccep` : les réponses ont été testées contre la ligne de base par permutation… ». Le **DOI** (*digital object identifier*) est l'identifiant permanent qui permet de citer un article ou un logiciel. Ce paragraphe imprimé à la fin de chaque analyse est le moteur principal des citations (c'est ce qui a fait le succès de FOOOF).

### Positionnement face aux outils existants
- **ERPLAB** (MATLAB) : mesures ERP classiques complètes, environ 300 citations par an ; mais pas de modèle de composantes, pas d'essai par essai, pas d'intracrânien.
- **MNE-Python** : la base Python ; mais pas de modèle de composantes, artefacts de stimulation seulement par interpolation. evoparam lit et rend des objets MNE.
- **FieldTrip, EEGLAB, Brainstorm** : laissent l'extraction des paramètres à l'utilisateur.
- **ER-detect** (Python, licence GPL) : détecte le N1 en CCEP, mais seulement la détection, seulement l'ECoG, validé à ~0,2 Hz.
- **CRP / BPC** (méthodes du labo Hermes/Miller) : formes apprises des données, mais une seule forme par site, sous forme de scripts.
- **TESA, TEPpy** (TMS-EEG) : listes de pics fixes, pas d'incertitude.
- **RIDE, Unfold, mTRF** : séparent ou déconvoluent, mais sans composantes paramétriques nommées ni test d'existence.
- **ERPparam** (point de départ du fork) : jamais publié (pas sur PyPI, 4 étoiles), N1 supprimé par heuristiques.
- **FOOOF / specparam** : paramètre les **spectres** (pas les réponses dans le temps), donc pas concurrent direct. Mais c'est le modèle de citations à égaler : **2 533 citations**, environ **600 par an** (OpenAlex, 7/10/2026).

---

## 2. Périmètre (ce qui est dedans, ce qui est dehors)

### Dedans
- **Signaux** : ERP/SEP/P300 sur le scalp ; CCEP sur ECoG et sEEG (réponses précoces, tardives et spectrales) ; TEP ; ERNA ; ECAP de moelle et d'implant cochléaire ; potentiels évoqués d'animaux (souris, rat, macaque).
- **Fonctions** (un module chacune) : lecture des données ; détection des impulsions de stimulation dans le signal ; nettoyage des artefacts ; préparation légère ; ajustement du modèle de composantes ; mesures ERP classiques ; test d'existence ; essai par essai ; modèle hiérarchique ; formes partagées entre canaux et cartes ; temps-fréquence ; statistiques ; simulateurs avec vérité connue ; rapports ; figures ; ligne de commande ; mode en ligne ; benchmark public.

### Dehors (avec la raison)
- **Interface graphique (GUI)** : c'est un autre produit, et c'est le module payant principal.
- **Contrôle en boucle fermée d'un stimulateur** : agir sur un patient fait du logiciel un **dispositif médical** (réglementé). Le paquet gratuit donne seulement les nombres.
- **Sorties de décision clinique** (« la zone épileptogène est ici ») : la validité clinique n'est pas prouvée par cette spec, et ce serait aussi un dispositif médical.
- **Localisation de sources, graphes de connectivité** : d'autres outils le font très bien (MNE, Brainstorm). evoparam exporte des matrices.
- **Tri de potentiels d'action (spike sorting), imagerie calcique** : autres signaux.
- **Prétraitement EEG général (ICA, filtres, détection de mauvais canaux)** : MNE le fait déjà.
- **Spectres de puissance au repos** : c'est un autre problème ; on garde le vocabulaire et le code séparés.
- **Classification / décodage (BCI)** : d'autres cadres le font ; evoparam fournit des paramètres utilisables comme entrées.
- **Moteur complet de modèles mixtes** : statsmodels, lme4, brms existent ; on exporte vers eux.
- **Détecteurs par apprentissage profond par défaut** : pas calibrés, difficiles à expliquer, pas de données d'entraînement pour la plupart des modalités. Ils peuvent concourir dans le benchmark.
- **Code copié depuis des logiciels GPL ou sans licence** : incompatible avec la licence Apache-2.0 et le modèle open-core. On ré-implémente à partir des articles (« salle blanche » : on lit l'article, pas le code).
- **Temps réel garanti (« dur »)** : Python ne peut pas le garantir ; le mode en ligne vise un temps réel « souple ».

---

## 3. Exigences (REQ)

Chaque exigence a un numéro stable `REQ-nnn`, un critère d'acceptation **testable** (on peut écrire un test automatique qui dit oui ou non) et une source. Exemple : REQ-113 dit précisément ce que veut dire « le N1 est corrigé ».

Règles de produit (REQ-001 à REQ-019), en résumé :
- un seul modèle et un seul appel (`evoparam.fit(data, preset=...)`) ;
- aucun filtre par le signe d'une bosse (le signe dépend du montage et de la position de l'électrode) ;
- aucune suppression de composante par heuristique : seulement par comparaison de modèles ;
- jamais d'échec silencieux (pas de « NaN » sans explication ; NaN = *not a number*, une valeur vide) ;
- une incertitude et une traçabilité partout ; un paragraphe Méthodes partout ;
- unités explicites ; noyau léger ; pas de code GPL ; pas de vocabulaire spectral hérité ;
- résultats identiques à chaque relance avec la même graine aléatoire (*random_state*) ;
- pas d'accès réseau caché ; mention « usage de recherche seulement, pas un dispositif médical » ;
- le correctif du N1 est testé à chaque construction ;
- chaque lacune du tableau GAPS est reliée à une exigence, un module, un test et une validation ;
- rien de public avant la décision de HIIE.

---

## 4. Architecture

### Les modules et leurs dépendances
Les données passent dans une chaîne : lecture (`io`) → événements (`events`) → artefacts (`artifact`) → préparation (`prep`) → formes (`shapes`) et modèle (`models`) → puis existence, essais, hiérarchique, spatial, temps-fréquence, statistiques. À côté : simulateurs (`sim`), presets, rapports, figures, ligne de commande, mode en ligne, benchmark.

Une règle d'**import** (quel module a le droit d'utiliser quel autre) est vérifiée par un test automatique. Exemple important : le simulateur n'a **pas** le droit d'utiliser le module d'ajustement, sinon la « vérité » pourrait être influencée par l'outil qu'on veut tester.

### Le modèle de données
- **TrialSet** : un tableau 3D (essais × canaux × temps) avec les temps, la fréquence d'échantillonnage (*sfreq*, le nombre de points par seconde), les noms de canaux, les unités, un tableau d'informations par essai (*metadata* : site de stimulation, courant en mA, largeur d'impulsion, polarité, état de veille/sommeil, etc.), un tableau par canal (type, coordonnées, région), des **masques** (les points remplacés lors du nettoyage, pour que l'ajustement les ignore), et la provenance.
- **Evoked** : la moyenne (canaux × temps).
- **Provenance** : version du paquet, date, source des données, chaque étape avec ses réglages, environnement, graine aléatoire, empreinte des données (*hash*).
- **Résultats** : `ResponseFit` (une courbe), `GroupFit` (plusieurs canaux), `Component` (une bosse avec toutes ses mesures), `GoodnessOfFit` (qualité de l'ajustement), `TrialFit`, `HierarchicalFit`, `ResponseTest`, `ArtifactResult`, `ShapeBasis`, `TFFit`, `Preset`.
- Chaque résultat a un **statut** : `ok`, `no_response` (aucune réponse : c'est un résultat valable), `failed` (l'optimiseur a échoué, avec message), `skipped` (canal ignoré, avec raison).

### Les renommages du code hérité
La classe `evoparam` (même nom que le paquet, ce qui crée des bugs d'import) devient `ResponseModel` (l'outil qui ajuste) + `ResponseFit` (le résultat). `peak_width_limits` devient `min_fwhm`/`max_fwhm` en secondes. `max_n_peaks` → `max_components`. `peak_threshold` → `detection_threshold`, **utilisé seulement pour proposer des candidats, jamais pour accepter**. `Bands` → `ComponentWindows` (des fenêtres de temps, pas des bandes de fréquence). Etc.

### Le sort de chaque morceau du code hérité
- **Gardés** : lecture BrainVision/BIDS, époques, téléchargement OpenNeuro, détection d'onsets, interpolation, et leurs tests.
- **Réécrits** : le moteur d'ajustement (pour toutes les raisons du N1), les groupes, l'écriture des résultats, les rapports, le simulateur (il dépendait d'un paquet externe de bruit).
- **Supprimés** : utilitaires morts, corrections, figures spécifiques aux spectres, outils de debug.
- **Erreurs** : une hiérarchie claire (`DataError`, `FitError`, `SettingsError`, etc.). Jamais d'échec silencieux.

---

## 5. Les modules, un par un

### 5.1 `io` — lire et écrire
Lit BIDS (le format standard de rangement des données de neuroimagerie), MNE, EDF, BrainVision, NWB (format standard pour la neurophysiologie animale), Neo, et des tableaux NumPy. **Lecture partielle à distance** : on ne télécharge que les morceaux utiles d'un gros fichier OpenNeuro (requêtes HTTP « Range »). Exemple : 10 essais de ds004080 en moins de 60 Mo et 30 s. Harmonise les colonnes de stimulation (site, courant, largeur d'impulsion) qui ne sont pas standardisées dans BIDS. Un registre de 14 jeux de données publics avec leur licence.

### 5.2 `events` — retrouver les impulsions dans le signal
Quand les marqueurs de stimulation manquent, on les retrouve grâce à l'artefact (le pic énorme de l'impulsion). Méthodes : détecteur robuste sur la dérivée (normalisée par la **MAD**, *median absolute deviation*, une mesure de dispersion qui n'est pas perturbée par les valeurs extrêmes), puis confirmation par **filtre adapté** (*matched filter* : on compare chaque pic à un modèle d'artefact ; corrélation > 0,8). Détecte aussi la **saturation** (l'amplificateur bloqué à son maximum) et identifie le **site stimulé** (les deux contacts avec les plus gros artefacts de signes opposés). Objectif : rappel et précision ≥ 0,99 sur ds004080 (rappel = part des vraies impulsions trouvées ; précision = part des détections qui sont vraies).

### 5.3 `artifact` — enlever l'artefact de stimulation sans abîmer la réponse
Onze méthodes, chacune écrite à partir des équations publiées (pas de code copié) :
1. **interpolate** : on remplace la fenêtre de l'artefact par une ligne (ou une courbe) entre avant et après.
2. **template** : on fait un « gabarit » de l'artefact (la médiane de tous les artefacts du même site, même courant, même polarité), on l'ajuste en taille pour chaque essai, et on le soustrait.
3. **tail** : on modélise la « queue » de l'artefact (la lente remontée après l'impulsion) par deux exponentielles décroissantes. Option `joint` : la queue est ajustée **en même temps** que les composantes, parce qu'une queue exponentielle et une composante large ressemblent l'une à l'autre au début.
4. **period_template** : pour les trains rapides (DBS) ; on estime l'artefact à partir des échantillons situés au même endroit dans les périodes voisines. ⚠ Méthode couverte par une **demande de brevet** de Brown University (US 2022/0338786 A1).
5. **lowrank** : on enlève ce qui est commun à beaucoup de canaux, d'impulsions et d'essais (régression « de bas rang »).
6. **subspace** : on « blanchit » les données avec le bruit de base, puis on projette hors des directions de l'artefact.
7. **polynomial** : à chaque instant, on ajuste un petit polynôme local et on le soustrait (pour les enregistrements très rapides).
8. **matching_pursuit** : on décompose chaque impulsion en briques (secteur, réponse, artefact) et on enlève les briques d'artefact. ⚠ Couverte par une **demande de brevet** de Washington University (US 2023/0240618 A1) ; avertissement affiché.
9. **paradigm** : on utilise le protocole. Polarité alternée : l'artefact s'inverse quand la polarité s'inverse, pas la réponse, donc la moyenne des deux annule l'artefact. Masquage avant (*forward masking*, implants cochléaires) : A − B + C − D.
10. **learned** : on apprend une petite base de formes d'artefact par **ACP** (analyse en composantes principales) et on l'enlève.
11. **Score de qualité** par essai (entre 0 et 1) et **masques** : les points remplacés sont marqués pour que l'ajustement ne s'y fie pas.

### 5.4 `prep` — préparation légère
Correction de ligne de base (avec vérification qu'elle ne chevauche pas une réponse tardive) ; **références** : bipolaire (différence entre deux contacts voisins), moyenne commune (**CAR**), **CAR ajustée** (méthode CARLA : on retire de la moyenne les canaux qui répondent, sinon ils contaminent tous les autres) ; rapport de **sensibilité à la référence** (le montage seul peut faire passer la latence du N1 de 47,4 ms à 70,7 ms dans une étude publiée) ; marquage des mauvais essais (jamais supprimés, seulement marqués) ; moyenne (moyenne, moyenne tronquée, médiane, pondérée).

### 5.5 `shapes` et `models` — le cœur : décrire la courbe
**Le modèle** : la courbe = une ligne de fond lente + une somme de quelques composantes + du bruit. Chaque composante = une **forme** × une **amplitude** (avec son signe).

**Les formes disponibles** (toutes avec une formule et une dérivée exacte) :
- gaussienne (cloche symétrique) ;
- gaussienne asymétrique (*skewed Gaussian* : montée rapide, descente lente ; typique du N1) ;
- ex-gaussienne (cloche + traîne exponentielle) ;
- gamma (forme « synaptique », départ net) ;
- bi-exponentielle (montée et descente exponentielles ; typique des ECAP) ;
- oscillation amortie (une sinusoïde qui s'éteint ; c'est l'ERNA) ;
- Gabor (une sinusoïde dans une cloche ; bouffées oscillantes en TMS) ;
- gabarit (*template*) appris des données.

**Les fichiers de données livrés avec le paquet** (presets, registre des jeux de données, règles publiées, zones) sont en **JSON**, lisible par Python sans rien installer, pour garder le cœur léger.

**Les fonds** : aucun, constant, linéaire, exponentiel (la queue d'artefact), double exponentiel, **P-spline** (une courbe souple, mais avec des nœuds espacés d'au moins deux largeurs de composante, pour qu'elle ne puisse pas « avaler » une vraie bosse), base d'artefact apprise.

**Le moteur en 10 étapes** (c'est la correction du N1) :
1. **Bruit estimé sur la ligne de base** : taille du bruit par la MAD, et son **autocorrélation** (à quel point un point ressemble au point suivant ; le bruit EEG/iEEG est très « collant »). On en tire un **nombre effectif d'échantillons** `n_eff = n(1−ρ)/(1+ρ)`. Exemple : avec ρ = 0,9, 2 000 points ne valent qu'environ 105 points indépendants. Sans cette correction, l'outil croit avoir beaucoup plus de preuves qu'il n'en a, et invente des bosses.
2. **Projection variable** (*variable projection*) : les amplitudes et le fond sont calculés exactement (moindres carrés linéaires) ; seuls les paramètres de forme sont cherchés par l'optimiseur. Avantages : moitié moins de paramètres, et un décalage constant (*DC offset*) ne change rien.
3. **Candidats** : on lisse le résidu (ce qui reste à expliquer), on cherche les extrema au-dessus de 2 écarts-types. **Les extrema au bord de la fenêtre sont gardés** (c'était la règle 1 qui tuait le N1). La largeur de départ est mesurée, et si la bosse est trop étroite, on la met à la largeur minimale au lieu de la jeter (c'était la règle 2). Le seuil de 2 sert **seulement** à proposer des candidats.
4. **Ajout pas à pas** : on part de zéro composante, on ajoute le meilleur candidat, on ré-optimise tout (avec bornes, dérivées exactes, 3 points de départ), et on **garde la composante seulement si le BIC baisse d'au moins 10**. Le **BIC** (*Bayesian information criterion*) est un score qui récompense un bon ajustement mais pénalise chaque paramètre ajouté : il faut que la nouvelle bosse « paie » sa complexité. Une baisse de 10 = preuve « très forte ».
5. **Ordre et séparation au lieu de suppression** : les composantes restent rangées dans le temps. Si deux bosses sont trop proches, on essaie de les **fusionner** et on garde la version (une ou deux bosses) qui a le meilleur score. Rien n'est supprimé juste parce que deux bosses se chevauchent (c'était la règle 3).
6. **Réajustement final et diagnostics** : incertitudes, **nombre de conditionnement** (si énorme, certains paramètres ne sont pas séparables), corrélations entre paramètres. On ne rejette pas, on **prévient** quels nombres ne pas croire.
7. **Mesures calculées sur la courbe ajustée**, pas sur les données brutes : latence, amplitude, largeur, début et fin (à 10 % du pic), aire, temps de montée, de descente, symétrie, « pointu ». Une latence peut tomber hors fenêtre : elle est gardée avec un avertissement.
8. **Qualité** : **R²** (coefficient de détermination : part de la variance expliquée ; l'ancien code utilisait un carré de corrélation, ce qui n'est pas la même chose), erreurs, AIC/BIC.
9. **Mesures ERP classiques** dans la même fenêtre (pour comparer avec la littérature).
10. **Statut** : `ok`, `no_response` (avec le meilleur candidat refusé et son score, pour inspection), `failed` (avec le message d'erreur et un avertissement dans le journal).

**Choix du nombre de composantes** : BIC (par défaut), AICc, validation croisée sur les essais, ou nombre fixe (pour reproduire une définition publiée). Le seuil de 10 est **calibré par simulation** : moins de 5 % de fausses bosses sur du bruit pur.

**Quatre niveaux d'incertitude** :
- `jacobian` (gratuit, par défaut) : à partir de la courbure du problème, corrigée pour l'autocorrélation ;
- `profile` : vraisemblance profilée (on fixe un paramètre et on ré-optimise les autres ; plus fiable quand les bosses se chevauchent) ;
- `bootstrap` : on ré-échantillonne les essais 200 fois et on refait l'ajustement ;
- `posterior` : modèle bayésien complet.
Les quatre rendent les mêmes champs.

**Mesures classiques** (comme ERPLAB, mais en Python) : pic local, amplitude moyenne, aire (4 variantes), **latence à 50 % de l'aire** (*fractional area latency* : le moment où la moitié de l'aire est atteinte), latence à fraction du pic, **SME** (*standardized measurement error* : l'incertitude de chaque score), outils **jackknife** (on refait le calcul en retirant un participant à la fois ; corrections t/(n−1) et F/(n−1)²).

**Critères chiffrés principaux** :
- sur simulations, rappel et précision ≥ 0,95 si le rapport signal/bruit (**SNR**) ≥ 3 ;
- erreur de latence ≤ 0,5 échantillon et biais d'amplitude ≤ 5 % si SNR ≥ 5 ;
- avec bruit autocorrélé (ρ = 0,9) : ≤ 5 % de fausses bosses (contre 93 % pour l'ancien code) ;
- **REQ-113, le N1** : sur ds004080 sujet 01, site PT01-PT02, N1 trouvé partout où la règle publiée (3,4 × SD, 9–100 ms) le trouve, à un canal près, avec κ ≥ 0,8 sur les 46 canaux, et reproduit sur au moins 5 sujets ;
- latences à ±2 ms de la règle publiée sur ≥ 90 % des canaux ;
- ≤ 50 ms par courbe ;
- intervalles à 95 % qui contiennent la vraie valeur dans 90–97 % des cas.

### 5.6 `existence` — y a-t-il une réponse ?
- **Méthode par défaut : permutation par inversion de signe.** Idée : s'il n'y a pas de réponse, inverser le signe d'un essai au hasard ne change rien en moyenne. On fait cela 1 000 fois pour voir ce que donne le « hasard », et on compare avec la vraie donnée. Le résultat est une **valeur p** (la probabilité d'obtenir un effet aussi fort par hasard). Deux statistiques : le maximum dans le temps (pour les réponses courtes) ou la « masse de cluster » (pour les réponses longues). Donne aussi la durée et le début de la réponse.
- **Projection croisée** (ré-implémentation de la méthode CRP d'après l'article) : on projette chaque essai sur les autres ; si les essais se ressemblent, il y a une réponse.
- **Comparaison de modèles** : le gain de BIC, transformé en valeur p par simulation.
- **Registre des règles publiées**, reproduites à la lettre : van Blooijs 2023 (3,4 × SD, 9–100 ms), Hays 2021 (z > 6, 10–50 ms), Keller 2014 (z > 6, < 50 ms, ligne de base −500 à −5 ms), Silverstein 2020 (z > 5, 10–50 ms), F-TRACT (Z entre 3 et 7 sur la moyenne des 800 ms après le stimulus), Prime 2020 (règle sur la RMS, *root mean square*, la « puissance moyenne » du signal, entre 10 et 100 ms). Plusieurs articles ne donnent pas la ligne de base ou le seuil exact : le registre marque ces champs « non précisé », refuse de deviner et demande la valeur à l'utilisateur.
- **Corrections pour comparaisons multiples** : **FDR** (*false discovery rate*, méthode de Benjamini–Hochberg : contrôle la proportion de fausses découvertes parmi les découvertes), Holm, Bonferroni.
- **Discordance entre trains** : si certains trains de stimulation donnent une réponse et d'autres non.
- Critères : taux de fausses alarmes à ±20 % de α (le seuil choisi, 5 %) ; puissance ≥ 0,9 à SNR 2 avec 10 essais ; accord κ ≥ 0,7 avec la règle publiée sur ds004080.

### 5.7 `trials` — essai par essai
Chaque essai = la forme moyenne, **décalée** dans le temps et **mise à l'échelle**. On estime pour chaque essai le décalage (*jitter*, la variation de latence) et l'amplitude. Niveaux : décalage+échelle (par défaut, rapide) ; paramètres libres ; gabarit ; régularisé (tiré vers la moyenne quand les essais sont bruités). Mesures dérivées :
- **écart-type du jitter**, comparé à un « jitter du hasard » mesuré sur la ligne de base (pour ne pas inventer du jitter) ;
- **amplitude corrigée du jitter** (le jitter aplatit la moyenne ; on calcule ce que serait le pic sans jitter) ;
- **moyenne réalignée** (refusée si le jitter n'est pas réel) ;
- **covariables** : tendance le long du train (adaptation), **dose–réponse** (amplitude selon le courant, courbe de Hill avec un courant seuil), **état avant le stimulus** (phase et puissance de l'alpha, 8–13 Hz, avec l'incertitude de l'estimation de phase).

### 5.8 `hierarchical` — essais dans canaux dans sujets
Un seul modèle où les essais sont regroupés par canal et les canaux par sujet. Avantages : l'incertitude passe d'un niveau à l'autre, les canaux pauvres empruntent de la force au groupe, et la **probabilité qu'un canal réponde** devient une vraie probabilité. Par défaut **Bayes empirique** (rapide, quelques secondes, sans dépendance). En option, **bayésien complet** avec **NUTS** (un algorithme d'échantillonnage) via NumPyro ou PyMC, avec diagnostics (r̂ ≤ 1,01, taille d'échantillon effective ≥ 400). Objectif : ≤ 5 min par sujet (contre ~10 h et pas de code pour la méthode publiée la plus proche). Formules comme « latence ~ âge » pour les effets de groupe ; on doit retrouver l'effet de l'âge sur la latence du N1 publié pour ds004080.

### 5.9 `spatial` — formes partagées et cartes
- **Formes partagées** (ré-implémentation de la méthode BPC d'après l'article) : on cherche les quelques formes de réponse communes à plusieurs sites de stimulation (ou canaux), par **NMF** (factorisation en matrices non négatives) sur une matrice de ressemblances.
- **Ajustement multi-canaux** : la même forme pour plusieurs canaux, avec option de lissage entre contacts voisins.
- **Cartes** des paramètres sur le cerveau.
- **Dépistage de la conduction volumique** (un signal qui se propage passivement et n'est pas une vraie réponse locale) : inversion de polarité entre contacts voisins, décroissance avec la distance, sensibilité à la référence. Rien n'est enlevé automatiquement.
- **Paradigme convergent** (beaucoup de sites → un canal) et divergent (un site → tous les canaux).

### 5.10 `tf` — temps-fréquence
Décompose la réponse en fréquences au cours du temps (ondelettes **Morse** γ = 3, P² = 30, comme dans les travaux du labo ; ou Morlet, multitaper). Points clés :
- l'artefact est masqué dans le plan temps-fréquence, élargi selon chaque fréquence ;
- **trois cartes** : totale, **empreinte évoquée** (ce que les bosses ajustées produisent à elles seules) et **résidu**. Pourquoi : un N1 très pointu crée à lui seul de l'énergie « gamma » ; sans cette séparation, on le prend pour du « gamma induit ». La décomposition est exacte ;
- statistiques : FDR par pixel, permutation par clusters, **TFCE** ;
- **zones trouvées par les données** (NMF ou décomposition tensorielle) au lieu de boîtes dessinées à la main ;
- **couche de compatibilité** pour les zones publiées : la zone « tardive 50–500 ms × gamma haut 50–250 Hz » est vérifiée ; les 10 autres zones de la thèse de Brinyark sont marquées **non vérifiées** et refusées sans confirmation explicite ;
- ECoG et sEEG traités pareil, avec la modalité comme covariable.

### 5.11 `stats` — statistiques
Bootstrap (intervalles BCa), jackknife sur participants, comparaisons de conditions par permutation au **bon niveau** (on ne permute pas des essais pour comparer des sujets), **fiabilité** (split-half avec correction de Spearman–Brown ; **dépendabilité Φ** et courbe « combien d'essais pour Φ ≥ 0,70 »), planification du nombre d'essais, export vers les modèles mixtes (statsmodels, lme4, brms), résumés de groupe qui comptent toujours les canaux sans réponse.

### 5.12 `sim` — simulateurs avec vérité connue
Générateurs pour ERP, CCEP (avec artefact, saturation, trains), ERNA, TEP (avec artefacts TMS et confusion auditive), ECAP, potentiels animaux. Bruits : blanc, autocorrélé, « rose », **bruit réel** (morceaux de lignes de base réelles de ds004080 et ERP CORE, avec licence vérifiée). Le simulateur n'utilise jamais le module d'ajustement.

### 5.13 Rapports, figures, ligne de commande, mode en ligne
- **Paragraphe Méthodes** automatique, JSON validé par schéma, **dérivés BIDS** (pas encore de standard officiel : la proposition BEP021 est bloquée), rapport HTML autonome et PDF.
- **Figures** (fonctionnent sans écran).
- **Ligne de commande** (détaillée au §10.5) : 13 sous-commandes : `fit`, `detect-events`, `clean`, `report`, `test`, `benchmark`, `simulate`, `validate-bids`, `reproduce`, `migrate`, `info`, `presets`, `cite`. `evoparam --version` affiche aussi la mention « usage de recherche seulement ».
- **Mode en ligne** (`StreamFitter`, détaillé au §10.6) : reçoit les époques en direct (via **LSL**, *Lab Streaming Layer*, un protocole standard de diffusion de signaux), ajuste chaque impulsion avec la forme fixée (seulement décalage et échelle), avec un budget de temps de calcul par impulsion (moins de 20 ms au 99e centile à 2 048 Hz et 64 canaux ; ce chiffre est un choix de la spec, pas une valeur publiée), et un test d'existence séquentiel qui ne gonfle pas les fausses alarmes quand on regarde souvent. Pas de contrôle de stimulateur.

---

## 6. Presets (réglages tout prêts)
Onze presets : `erp`, `ccep`, `ccep_seeg`, `sep`, `tep`, `erna`, `ecap` (moelle), `ecap_ci` (implant cochléaire), `rodent_ep`, `primate_ep`, `p300_bci`. Chacun fixe : époque, ligne de base, fenêtre d'artefact et méthode, référence, fenêtre d'ajustement, formes, fond, nombre maximum de composantes, largeurs min/max, étiquettes de composantes avec leurs fenêtres, essai par essai ou non, et les sources. Exemples :
- `ccep` : époque −1 à 2 s, ligne de base −0,5 à −0,02 s (comme ER-detect), artefact −2 à 8 ms par gabarit + queue jointe, référence CAR ajustée, ajustement de 9 ms à 500 ms, N1 9–50 ms, N2 50–300 ms, tardif 100 ms–1 s, largeur min 3 ms.
- `ccep_seeg` : pareil mais référence bipolaire (le sEEG est plus sensible au montage).
- `tep` : N15, P30, N45, P60, N100, P180 ; prévient que N100/P180 peuvent être contaminés par le clic de la bobine.
- `erna` : oscillation amortie 200–450 Hz, une époque par impulsion (7,7 ms à 130 Hz).
Les valeurs qui sont un **choix** de la spec (et pas une convention publiée) sont marquées et testées dans le benchmark. Les presets validés seulement en simulation (ECAP, ERNA : pas de données publiques) le disent dans le paragraphe Méthodes.

---

## 7. Validation scientifique (4 couches)
1. **Mathématique** : les formules sont justes (comparées à un calcul de haute précision).
2. **Simulations avec vérité connue** : un plan complet (modalité × SNR × nombre de composantes × séparation × largeur × nombre d'essais × jitter × artefact), 1 000 répétitions par case ; cartes de **où chaque méthode échoue**.
3. **Données réelles vs méthodes publiées et experts** : ds004080 (sujets choisis par une règle fixée **avant** de voir les résultats), ds003708 (BPC), ds004977 (CARLA), ERP CORE (40 participants, 7 composantes), ds002094 (TMS-EEG), Allen Neuropixels (souris), Localize-MI.
4. **Robustesse** : montage, presets (±50 %), fréquence d'échantillonnage, nombre d'essais, système d'exploitation.
Ce qui n'est **pas** prétendu : la validité clinique (ex. que la carte N1 localise la zone épileptogène). Être d'accord avec une règle publiée, c'est être d'accord avec une convention, pas avec la vérité ; c'est pour ça que les simulations existent aussi.

---

## 8. Benchmark public
Pourquoi : aucun benchmark commun n'existe ; un benchmark avec classement fait que chaque nouvelle méthode concurrente cite le benchmark. Sept tâches : détection de réponse, latence/amplitude du N1, décomposition en composantes, retrait d'artefact, essai par essai, mesures ERP, temps-fréquence. Données : suites simulées (déposées sur **Zenodo** avec DOI) + données réelles publiques ; une partie cachée (graines publiées seulement après acceptation de l'article, avec empreinte déposée avant). Méthodes de référence : toutes les règles publiées, CRP, mesures classiques, gabarit, evoparam. Les outils GPL sont comparés via des **adaptateurs** qui les lancent dans un processus séparé (sans les intégrer au paquet). Classement généré automatiquement.

---

## 9. Ingénierie
Performances (ex. ajustement d'une courbe, mémoire < 2 × la taille des données) mesurées en CI (**intégration continue** : tests automatiques à chaque modification). Dépendances : NumPy + SciPy seulement, le reste en *extras*. Jamais d'import de code GPL. Versions sémantiques, dépréciation sur deux versions mineures. Résultats identiques (à 1e-9 près sur CPU). CI sur Python 3.10–3.13 et trois systèmes ; couverture de tests ≥ 90 % du cœur, 100 % du module des formes, ≥ 80 % pour figures, ligne de commande, mode en ligne et benchmark. Typage strict. Documentation avec erreurs bloquantes. Pas de réseau caché, pas de télémétrie. Conteneur de résultats `.evoparam` avec sommes de contrôle, JSON valide, réglages enregistrés, fonction `evoparam.reproduce` qui refait tout le calcul.

## 10. Écosystème
- **MNE** : lit et rend des objets MNE. Ce qui serait proposé à MNE est écrit à neuf sous licence BSD-3.
- **BIDS** : table d'harmonisation des colonnes de stimulation, validateur, dérivés.
- **NWB / Neo / DANDI** : lecture et écriture.
- **scikit-learn** : estimateurs compatibles (mêmes conventions `fit`, `get_params`).
- Ligne de commande détaillée, mode en ligne détaillé (« recherche seulement, pas un dispositif médical »).
- Documentation : 17 tutoriels ; gouvernance : l'auteur décide seul pendant les versions 0.x, puis un comité de trois à la 1.0.

## 11. Citations et publications
- **Mécanisme de citation** : le paragraphe Méthodes automatique, `CITATION.cff` (fichier standard qui dit comment citer), un DOI Zenodo par version, un benchmark public, des presets nommés.
- **Leçons des courbes de citations** : un outil cité comme **méthode nommée** monte vite (FOOOF : 1 000 citations en ~3,2 ans) ; une boîte à outils générale monte lentement ; il faut le code et une prépublication ~2 ans avant le grand article ; maintenance sur dix ans.
- **Trois articles** : article logiciel (JOSS ; exige 6 mois de développement public) ; article de méthode (Nature Methods, sinon Nature Biomedical Engineering / eLife / PLOS Comp Bio / J Neurosci Methods) ; article benchmark (Scientific Data, avec le labo Smith).
- **Ordre des étapes publiques** : M0 déclaration d'invention → M1 décision écrite de HIIE → M2 sortie publique (GitHub, PyPI, Zenodo le même jour) → … → M8 article benchmark. **Rien de public avant M1.** Aux États-Unis, l'inventeur a un an de grâce après sa propre divulgation ; presque partout ailleurs, toute divulgation avant dépôt détruit la nouveauté (« nouveauté absolue »).
- **Labo Voytek** : rester indépendant jusqu'à la première sortie publique, puis proposer un benchmark commun (pas une co-signature de la méthode).
- **Cibles** : environ un tiers de la courbe de FOOOF aux mêmes années après sortie.

## 12. Ordre de construction
21 étapes (S0 à S20) dans l'ordre des dépendances : S0 squelette (fait) → S1 modèle de données → S2 lecture → S3 formes → S4 événements → S5 artefacts → S6 préparation → **S7 moteur d'ajustement (correction du N1)** → S8 existence → S9 simulateurs (en parallèle) → S10 essais → S11 hiérarchique → S12 spatial → S13 temps-fréquence → S14 stats → S15 presets et validation → S16 rapports/CLI → S17 écosystème → S18 mode en ligne → S19 benchmark → S20 robustesse et publication. L'ancien moteur reste disponible jusqu'à ce que S7 soit vert. À partir de S7, le test du N1 tourne à chaque construction.

## 13. Valorisation (modèle commercial)
- **Open-core** : le moteur, les presets, la ligne de commande, la lecture, le benchmark et même la statistique d'alarme sont **gratuits**. **Payants** : GUI, intégration temps réel, rapports cliniques, produit de monitorage peropératoire (**IONM**), cloud, versions validées. Règle : rien de payant n'est nécessaire pour faire la science.
- **Marchés** (sources et dates dans la spec, rapports de marché souvent lus seulement en résumé) : IONM 3,65 milliards $ (2025) → 4,93 milliards $ (2031) ; implants cochléaires 2,80 milliards $ (2024) → 4,73 milliards $ (2030) ; BCI non invasives 397,59 millions $ (2025) → 773,82 millions $ (2033) ; systèmes TMS environ 1,9 milliard $ en 2030 (chiffre d'une annonce, rapport non lu) ; SCS en boucle fermée pilotée par l'ECAP (Evoke de Saluda, Inceptiv de Medtronic).
- **Réglementation** : un logiciel de recherche n'est pas un dispositif médical. Dès qu'il sert à un diagnostic ou un traitement, il devient un **SaMD** (*software as a medical device*) : aux États-Unis via la FDA (ex. 21 CFR 882.1870 pour les potentiels évoqués), en Europe via le **MDR** (règle 11 : classe IIa ou plus). Le développement doit alors suivre **IEC 62304** (norme du cycle de vie des logiciels médicaux, classes A/B/C selon le risque ; le texte de la norme n'a pas été lu, c'est signalé).
- **Propriété intellectuelle** : marque, six éléments de méthode potentiellement brevetables (à juger par HIIE), question du brevet implicite d'Apache §3, deux demandes de brevet tierces à vérifier (**FTO**, *freedom to operate* : la liberté d'exploiter sans violer un brevet), propriété de l'UAB, partage des revenus.
- **21 questions pour HIIE** (§13.5), dont : l'UAB possède-t-elle le logiciel ? quelle licence ? brevet ou pas avant la sortie ? marque ? FTO sur les deux brevets ? des modules payants construits sur le moteur seraient-ils à l'UAB ? rôle d'un doctorant dans une startup ? propriété des annotations cliniques pour le benchmark ?

## 14. Matrice de traçabilité
Chaque lacune de `GAPS.md` → exigence(s) → module → fichier de test → élément de validation. Une lacune « traitée » sans cette chaîne complète est une erreur.

## 15. Risques (30, en 7 familles)
Exemples : le labo Voytek publie ERPparam avant nous ; les demandes de brevet tierces sont accordées ; le plafond d'accord humain (κ 0,60) limite toute validation contre des experts ; les critères AIC/BIC avec bruit coloré ; la latence du N1 qui dépend du montage ; pas de données publiques ERNA/ECAP/IONM ; dispersion du projet sur trop de modalités. Chacun a une probabilité, un impact, un signal d'alerte, une parade et la section qui l'implémente.

## 16. Questions que toi seul peux trancher (avec ma recommandation)
1. **Nom** : evoparam (reco) ou evoresponse.
2. **Licence à proposer à HIIE** : Apache-2.0 partout (reco).
3. **Formes par défaut du preset ccep** : gaussienne asymétrique + gaussienne (reco) jusqu'à ce que la validation compare avec ex-gaussienne et gamma.
4. **Seuil de sélection** : ΔBIC −10 (reco), calibré pour ≤ 5 % de fausses bosses.
5. **Méthodes sous brevet** (period_template, matching_pursuit) : les coder dans une branche privée maintenant, les publier seulement après l'avis FTO de HIIE (reco).
6. **Premier moteur bayésien** : NumPyro (reco) puis PyMC.
7. **Contact labo Voytek** : après la première sortie publique, en proposant un benchmark commun (reco).
8. **Données cliniques UAB dans le benchmark** : données publiques seulement au début (reco), UAB après IRB et HIIE.
9. **Contenu de la première sortie publique** : étapes S0 à S16 (reco) ; le reste suit en versions mineures (c'est un rythme de sortie, pas une réduction du périmètre).
10. **Gouvernance** : décideur unique en 0.x, comité de trois à la 1.0 (reco).
11. **Mode en ligne gratuit** : oui, en temps réel souple (reco) ; le temps réel dur reste payant.
12. **Python minimum** : 3.10 (reco, la CI teste déjà 3.10 à 3.13). Attention : les versions actuelles de MNE-Python et scikit-learn exigent Python 3.11 ; garder 3.10 oblige à utiliser d'anciennes versions de ces options, et il faudra passer à 3.11 quand elles ne suivront plus.
13. **Unité d'affichage** : µV par défaut, unité native sur demande (reco).
14. **Signatures des articles** : premier et correspondant sur les articles logiciel et méthode ; benchmark avec les labos qui donnent des données en co-auteurs (reco).
