# Annexe : tous les seuils, réglages et formules

Sources : la spec (§ indiqués), le code actuel du paquet (version 0.1.0.dev0) et docs/spec/DECISIONS.md.

Notations : `W = [t_0, t_1]` est la fenêtre d'ajustement, `|W|` sa durée, `σ_n` le SD (écart-type) du bruit de la ligne de base, `F = 2.3548` le nombre de FWHM (largeur à mi-hauteur) par σ. « DEC §n » renvoie à la section n de docs/spec/DECISIONS.md.

## 1. Formes des bosses

Les formes possibles d'une bosse (une « composante » du signal), leurs formules et les limites de leurs paramètres.

| Élément | Ce que dit la spec (§) | Dans le code, si différent | Origine |
| --- | --- | --- | --- |
| Normalisation | `max_t \|s(t)\| = 1`, `A_k` = amplitude linéaire à part (§5.5.1) |  | choix de la spec |
| `gaussian` | `s = exp(-(t-μ)²/(2σ²))` ; `μ ∈ [t_0 - σ_max/2, t_1 + σ_max/2]`, `σ ∈ [min_fwhm, max_fwhm]/F` ; latence `μ` (§5.5.1) | la borne de latence est aussi coupée à `[t_0 - 0.05·\|W\|, t_1 + 0.05·\|W\|]` par `edge_extension` (`engine.py` `_shape_bounds`) | publié (LIT-fitting-5, LIT-fitting-14) |
| `skewed_gaussian` | `s = g(z)·(1 + erf(αz/√2))/c(α)`, `z = (t-ξ)/ω` ; `ω` comme `σ` ; `α ∈ [-10, 10]` ; latence = mode (sommet de la courbe) (§5.5.1) | `alpha_max = 10.0` (`shapes/components.py`) | publié (LIT-fitting-14) |
| `exgaussian` | `h = (1/2τ)·exp(σ²/(2τ²) - (t-μ)/τ)·erfc((σ/τ - (t-μ)/σ)/√2)`, `s = h/max h` ; `τ ∈ [1/sfreq, max_fwhm]` ; latence = mode |  | publié (LIT-fitting-14, LIT-fitting-27) |
| `gamma` | `s = ((t-t_0)/((κ-1)θ))^(κ-1)·exp(κ-1-(t-t_0)/θ)`, `t ≥ t_0` ; `κ ∈ [1.5, 20]` ; `θ` tel que la FWHM soit dans `[min_fwhm, max_fwhm]` ; latence `t_0 + (κ-1)θ` | boîte (simples bornes min et max) `θ ∈ [min_fwhm/F_γ(κ_max), max_fwhm/F_γ(κ_min)]`, `t_0 ∈ [t_0 - max_fwhm, t_1]` (`components.py`) | choix de la spec (classique) |
| `biexponential` | `s = [exp(-(t-t_0)/τ_d) - exp(-(t-t_0)/τ_r)]/m` ; `τ_r ≥ 1/sfreq`, `τ_d ≥ 1.5·τ_r`, `τ_d ≤ \|W\|` ; latence `t_0 + τ_rτ_d/(τ_d-τ_r)·ln(τ_d/τ_r)` | boîte `τ_r ∈ [1/sfreq, \|W\|/1.5]`, `τ_d ∈ [1.5/sfreq, \|W\|]` ; le rapport 1,5 est une contrainte, pas une boîte (`components.py`) | choix de la spec (classique) |
| `damped_oscillation` | `s = exp(-(t-t_0)/τ)·cos(2πf(t-t_0)+φ)`, `t ≥ t_0` ; `f ∈ [f_min, f_max]` (donné par le préréglage ; ERNA (réponse neurale évoquée résonante) 200–450 Hz) ; `τ ∈ [1/f, \|W\|]` ; latence `t_0` + `peak_times` | `τ ∈ [1/f_max, \|W\|]` ; `t_0 ∈ [t_0 - max_fwhm, t_1]` ; `φ ∈ [-π, π]` ; `f ∈ [2/\|W\|, sfreq/4]` par défaut quand le préréglage n'en donne pas ; `extremum` donné à part | publié (LIT-fitting-4, LIT-fitting-24, LIT-fitting-26) |
| `gabor` | `s = exp(-(t-t_0)²/(2σ²))·cos(2πf(t-t_0)+φ)` ; bornes comme `gaussian` + `f` | même plage de `f` par défaut que ci-dessus | publié (LIT-fitting-23) |
| `template` | `s = T((t-δ)/γ)/max\|T\|` ; `δ ∈ ±max_shift` (valeur non donnée) ; `γ ∈ [0.8, 1.25]` | `max_shift` vaut `\|W\|/4` par défaut (`components.py` `Template`) | publié (LIT-fitting-1, LIT-fitting-2, LIT-erpmeas-7) |
| Limites de largeur par défaut d'une forme seule | non dit | `min_fwhm = 2/sfreq`, `max_fwhm = \|W\|/2` quand rien n'est réglé (`components.py` `_WidthShape`) | code seulement |
| Valeurs de départ | `α_0 = 0`, `κ_0 = 2`, `τ_r0 = σ_0/2`, `τ_d0 = 2σ_0`, `f_0` = centre des bornes, `φ_0 = 0` (§5.5.3 étape 3) | oscillation amortie : `f_0` = pic du périodogramme (spectre simple) dans les bornes, `φ_0 = -π/2`, `t_0 = peak - 1/(4f_0)`, `τ_0 = 2σ_0` (`components.py`) | choix de la spec |
| Grille des mesures | mesures prises sur une grille 10× plus fine que `1/sfreq` (§5.5.1) | `FEATURE_OVERSAMPLING = 10` | choix de la spec |
| Fond (courbe lente sous les bosses) `none` / `constant` / `linear` | `b = 0` ; `b = c_0` ; `b = c_0 + c_1(t - t_0)` (§5.5.2) | coefficients linéaires bornés de façon large à 10× l'échelle des données (`COEF_BOUND`, `backgrounds.py`) | classique |
| Fond `exponential` | `b = c_0 + c_1·exp(-(t-t_0)/τ)`, `τ ∈ [0.5 ms, \|W\|]` (§5.5.2) | `TAU_MIN = 5e-4` ; `background_tau_max = 0.05` s dans `ccep` et `ccep_seeg` (préréglage `fit_extras`) | publié (LIT-fitting-15, LIT-fitting-16, LIT-artifact-1) |
| Fond `double_exponential` | `τ_2 ≥ 3·τ_1` | boîte `τ_1 ∈ [τ_min, τ_max/3]`, `τ_2 ∈ [3τ_min, τ_max]` ; le rapport est une contrainte (`backgrounds.py`) | publié (LIT-artifact-1) |
| Fond `pspline` | B-splines (morceaux de courbe raccordés) cubiques, écart entre nœuds `≥ 2·max_fwhm`, pénalité `λ` sur la différence seconde, choisie par GCV (validation croisée généralisée) sur la ligne de base | écart par défaut `2·max_fwhm` | choix de la spec (P-spline standard) |
| Fond `artifact_basis` | `b = Σ c_j u_j(t)`, base tirée de la méthode `learned` du §5.3 |  | choix de la spec |

## 2. Moteur d'ajustement

Comment le moteur ajuste les bosses et décide combien de bosses garder.

| Élément | Ce que dit la spec (§) | Dans le code, si différent | Origine |
| --- | --- | --- | --- |
| Modèle | `y(t) = b(t; β_b) + Σ_k A_k·s_k(t; θ_k) + ε(t)` (§5.5) |  | choix de la spec |
| Objectif | moindres carrés pondérés, projection variable (paramètres linéaires résolus à part) `β̂(Θ) = argmin ‖W^½(y - M(Θ)β)‖²`, jacobienne (tableau des dérivées) de Kaufman `-W^½(I - MM⁺)∂M/∂Θ β̂` (§5.5.3 étape 2) | solveur TRF (région de confiance avec bornes) maison (`models/_trf.py`), voie compilée en option (`_fast.py`, tolérance 1e-6) | classique (Golub–Pereyra) ; choix de la spec |
| Échelle du bruit | `σ_n = 1.4826·median(\|y_B - median(y_B)\|)` (§5.5.3 étape 1) | `noise_source="auto"` : la moyenne plus-moins (essais pris avec signes alternés) remplace la ligne de base quand la ligne de base a `n(1-ρ)/(1+ρ) < 30` et qu'il y a ≥ 10 essais (`engine.py` `MIN_EFFECTIVE_BASELINE`, `MIN_PLUS_MINUS`) | classique (MAD, écart médian absolu) ; plus-moins : publié (Schimmel 1967), code seulement |
| Modèle de bruit / blanchiment (rendre le bruit sans mémoire) | `ρ = corr(y_B[1:], y_B[:-1])`, `n_eff = n_W·(1-ρ)/(1+ρ)` (nombre de points vraiment indépendants) dans chaque critère et chaque covariance (§5.5.3 étape 1) | par défaut `noise_model="ar"` : pré-blanchiment AR(p) (bruit qui dépend du passé), `p ≤ 10` choisi par AIC, ≥ 20 points de ligne de base par ordre, moindres carrés ; le critère compte les lignes blanchies. `"neff"` = la spec, `"white"` = `n` brut (`engine.py` `MAX_AR_ORDER`, `SAMPLES_PER_ORDER`) | publié (LIT-fitting-3) pour `n_eff` ; AR : code seulement |
| Bruit limité en bande | pas dans la spec | saut `k ∈ 2..4` (on garde 1 point sur k) quand la bande au-dessus de `1/k` Nyquist (moitié de la fréquence d'échantillonnage) est sous `1e-4` de l'octave juste en dessous (`EMPTY_BAND`, `MAX_STRIDE = 4`, `MIN_BAND_POINTS = 8`) | code seulement |
| Ligne de base courte | `< 20` points → `σ_n = MAD(diff(y_W))/√2`, avertissement `short_baseline` | `MIN_BASELINE = 20` | choix de la spec |
| Poids | `w_i = 1/max(sem_i, 0.1·σ_n)` avec `sem` (erreur type de la moyenne) ; sinon `1/σ_n` ; points masqués `w_i = 0` | `_SEM_FLOOR = 0.1` | choix de la spec |
| A priori doux (légère attirance vers une valeur) | pas dans la spec | `background_prior = 3` : `c_0 ~ N(baseline median, (3σ_n)²)` ; `amplitude_prior = 3` : les autres coefficients linéaires ont un SD de `3·max(range(y_W), σ_n)` ; `edge_extension = 0.05·\|W\|` (`_settings.py`, `engine.py` `prior_rows`) | code seulement (CHANGELOG « Soft priors ») |
| Lissage des candidats | noyau gaussien de SD `min_fwhm/F/2` |  | choix de la spec |
| `detection_threshold` | 2,0 σ\_n ; les extrêmes `≥` seuil sont candidats ; ne change que l'ordre et le nombre (§5.5.3 étape 3, REQ-104) | `threshold_role="order"` par défaut : chaque extrême est candidat, le seuil sert seulement à les classer ; `"select"` (texte de la spec) dans le préréglage `ccep` (`engine.py` `candidates`) | choix de la spec |
| Extrême au bord | le plus grand `\|r̃\|` à moins de `min_fwhm` du bord |  | choix de la spec (correction d'une règle héritée) |
| Largeur de départ | `σ_0` = HWHM (demi-largeur à mi-hauteur) de `\|r̃\|`, coupée à `[min_fwhm, max_fwhm]/F` ; repli `min_fwhm/F` |  | choix de la spec |
| `n_starts` | 3 : le candidat + des variations `μ_0 ± σ_0/2`, `σ_0 × {0.5, 2}` (§5.5.3 étape 4) | voie « beam » (garder plusieurs candidats en parallèle) : latence `± max(\|W\|/50, 2 samples)` seulement ; départs de la spec avec `candidate_search="spec"` (`engine.py` `_starts`) | choix de la spec |
| Optimiseur | `least_squares(trf, max_nfev = max_iter = 2000, loss="linear", f_scale=None)` | TRF maison, `tol = 1e-8` | choix de la spec |
| Candidats par étape | `max_candidates = 2·max_components`, le premier accepté est gardé | `candidate_search="beam"` : 3 candidats valables comparés (`CANDIDATE_BEAM`), arrêt après 6 extrêmes sans gain (`CANDIDATE_SCAN`), propositions de couper en deux les 2 plus grandes composantes | choix de la spec |
| Critère | `BIC = n_eff·ln(RSS_w/n_eff) + p·ln n_eff` ; `AIC = … + 2p` ; `AICc = AIC + 2p(p+1)/(n_eff - p - 1)` (§5.5.4) ; BIC (critère qui punit la complexité), AICc (AIC corrigé pour petits échantillons), `RSS_w` (somme pondérée des erreurs au carré) | `n_eff` gardé au moins à `p + 2` (`engine.py` `criterion`) | classique ; publié (LIT-fitting-3) |
| Δ d'acceptation | `ΔBIC ≤ -10` (`delta = 10`) pour chaque nouvelle composante (§5.5.4) | 10 par défaut (`FIXED_DELTA`) ; option `delta="calibrated"`, `alpha = 0.04` : niveau de balayage par caractéristique d'Euler (seuil pour cartes bruitées) sur latence × log-largeur, élargi pour l'erreur du spectre AR et la dérive de ligne de base (3 segments, 48 largeurs) ; il faut `c + (p-3)·ln n` (`_calibration.py`) | choix de la spec (échelle de preuve « très forte ») |
| Autres choix | `selection ∈ {bic, aicc, cv, fixed}` ; cv (validation croisée) = k-fold (k parts tour à tour), minimum à 1 SE (erreur type) près | `n_folds = 5` | classique |
| Choix de forme | la meilleure forme selon le critère pendant la recherche pas à pas | par défaut du paquet `shape_choice="aic"` : les composantes finales de FWHM ≥ 8 points sont réajustées avec chaque forme permise (asymétrie de départ ±2), l'AIC le plus bas est gardé ; les préréglages gardent `"bic"` (`engine.py` `SHAPE_MIN_SAMPLES`, `SKEW_STARTS`) | choix de la spec |
| `max_components` | 6 par défaut dans le paquet (§5.5.8) |  | choix de la spec |
| `min_fwhm` / `max_fwhm` | défauts du paquet 0,002 / 0,300 s |  | choix de la spec |
| Fréquence d'échantillonnage basse | `sfreq < 2/min_fwhm` → `min_fwhm = 2/sfreq` (§5.5.9) | `2·stride/sfreq`, et `max_fwhm ≥ 2·min_fwhm` (`engine.py` `_noise_band`) | choix de la spec |
| Taille de fenêtre | fenêtre `< 3·min_fwhm` → SettingsError ; `< p + 5` points → composante non essayée | `MIN_WINDOW_FWHMS = 3`, `MIN_VALID_SAMPLES = 5` | choix de la spec |
| Part masquée | `> max_masked_fraction = 0.5` → `skipped` |  | choix de la spec |
| Ordre des composantes | `μ_1 = u_1`, `μ_{k+1} = μ_k + exp(v_k)` (§5.5.3 étape 5) | par défaut `latency_order="parameterised"` | choix de la spec |
| Règle de chevauchement | `\|μ_{k+1} - μ_k\| < overlap_c·(σ_k + σ_{k+1})`, `overlap_c = 0.5` → fusionner et comparer, garder le meilleur critère (§5.5.3 étape 5) | le modèle fusionné est gardé si `crit_merged ≤ crit` ; pas de fusion en `fixed`/`cv` (DEC §1 D7) | choix de la spec ; publié (LIT-fitting-34) |
| Paires qui s'annulent | pas dans la spec | `reseed=True` : nouvel ajustement à partir des 6 plus forts extrêmes des données quand une paire de signes opposés se chevauche ou que le conditionnement (sensibilité aux petites erreurs) de la matrice du modèle `> 1e5` ; puis retrait des composantes qui ratent le Δ ; `reject_cancelling=False` (`engine.py` `RESEED_EXTREMA`, `RESEED_COND`) | code seulement |
| Test de résolution | pas dans la spec (REQ-111 demande `merged:k,k+1`) | par défaut du paquet `resolution_check=True` : signale quand une paire égale espacée de `overlap_c(σ_1+σ_2)` n'est pas distinguable ; voisins à moins de 2 FWHM ignorés ; préréglages `False` | code seulement |
| Conditionnement | `cond(JᵀJ) > 1e8` ou `\|corr\| > 0.95` → `ill_conditioned`, jamais rejeté (§5.5.3 étape 6) | `COND_LIMIT`, `CORR_LIMIT` | publié (LIT-fitting-34, LIT-fitting-35) |
| À la borne | à moins de `1e-3` d'une borne → `at_bound:<param>` (§5.5.5, §5.5.9) | mesuré par rapport à la largeur de l'intervalle de bornes (`BOUND_TOL`) | choix de la spec |
| Covariance | `Cov(Θ̂) = s²(JᵀJ)⁻¹`, `s² = RSS_w/(n_eff - p)` | `n_eff` = lignes blanchies avec `"ar"` | classique |
| `uncertainty="jacobian"` | Wald (valeur ± z erreurs types) `θ̂ ± z_{1-α/2}·se`, méthode delta (propager l'erreur par dérivée) pour les mesures ; `ci_level = 0.95` |  | classique |
| `uncertainty="profile"` | `2[ln L(Θ̂) - ln PL(θ_i)] ≤ χ²_{1,1-α}` (3,84) ; environ `2×20×K` réajustements | grille de 21 points sur `±4·se`, interpolation linéaire du point de passage (`api.py` `_profile`) | publié (LIT-fitting-35, LIT-fitting-48) |
| `uncertainty="bootstrap"` | `B = 200` tirages d'essais (bootstrap : rééchantillonnage avec remise), percentile (quantiles des tirages) et BCa (bootstrap corrigé du biais) | percentile élargi seulement : `q = √(n/(n-1))·t_{n-1}` (Hesterberg 2015), pas de BCa ; ≥ 10 tirages valides (`api.py` `_bootstrap`) | publié (LIT-fitting-28, LIT-fitting-7) |
| IC gonflé par le masque | IC (intervalle de confiance) gonflé selon la part masquée du support de la composante (§5.3 point 12) | aucune mise en œuvre trouvée | choix de la spec |
| Fenêtres longues | `> 10 s` à ≥ 1 kHz → recherche pas à pas décimée (sous-échantillonnée) à ≤ 20 000 points | aucune mise en œuvre trouvée | choix de la spec |
| Statut | `ok` (K ≥ 1), `no_response` (K = 0, avec `best_rejected_delta`), `failed`, `skipped` |  | choix de la spec |

## 3. Mesures sur chaque bosse

Les mesures qu'on lit sur chaque bosse ajustée.

| Élément | Ce que dit la spec (§) | Dans le code, si différent | Origine |
| --- | --- | --- | --- |
| Latence | moment de l'extrême signé de `A_k·s_k` (mode) ; selon la forme : `μ` (gaussian), mode (skewed, exgaussian), `t_0 + (κ-1)θ` (gamma), début `t_0` (oscillation amortie), centre de l'enveloppe (gabor) (§5.5.1) |  | choix de la spec |
| Amplitude | `A_k`, avec son signe ; polarité `sign(A_k)` |  | choix de la spec |
| FWHM | largeur totale à la moitié de l'extrême de la courbe de la composante |  | classique |
| Début / fin | premier / dernier `t` avec `\|A_k s_k\| ≥ onset_fraction·\|A_k\|`, `onset_fraction = 0.10` | `ONSET_FRACTION = 0.10` | publié (LIT-erpmeas-3) |
| Aire | intégrale signée par la méthode des trapèzes sur `[onset, offset]` |  | choix de la spec |
| Montée / descente | `latency - onset` ; `offset - latency` |  | classique |
| Symétrie | `(decay - rise)/(decay + rise)` |  | choix de la spec |
| Finesse du pic | `\|s''(latency)\|·fwhm²` |  | choix de la spec |
| Latence hors fenêtre | donnée avec l'avertissement `latency_outside_window` |  | choix de la spec |
| Pic classique | pic local plus grand que `n_neighbors = 3` points de chaque côté (§5.5.6) |  | publié (LIT-erpmeas-1, ERPLAB) |
| Latences fractionnelles classiques | aire fractionnelle (moment où X % de l'aire est passé) 50 % ; pic fractionnel 50 % ; début à 10 % |  | publié (LIT-erpmeas-3, LIT-erpmeas-45) |
| Aire classique | `total`, `positive`, `negative`, `rectified` ; ligne de base `"zero"` par défaut ou `"fraction_of_peak"` |  | publié (LIT-erpmeas-3) |
| SME (erreur de mesure standardisée) | `SD/√n` (scores linéaires) ; bootstrap `B = 1000` (scores non linéaires) |  | publié (LIT-erpmeas-41) |
| Étiquettes | fenêtre d'étiquette + polarité ; le plus grand `\|A\|` prend l'étiquette, l'autre reçoit `<label>_b` + `multiple_in_window` ; sinon `C{k}` (§5.5.7) | les fenêtres prennent leur polarité dans `label_polarity` du JSON du préréglage | choix de la spec |

## 4. Test d'existence d'une réponse

Comment l'outil décide s'il y a une vraie réponse ou seulement du bruit.

| Élément | Ce que dit la spec (§) | Dans le code, si différent | Origine |
| --- | --- | --- | --- |
| Méthode par défaut | `baseline_permutation` (§5.6, §6) |  | choix de la spec |
| Statistique | `t(t) = mean_k x_k(t)/(SD_k x_k(t)/√n)` ; `max_t` par défaut, `cluster` (groupe de points voisins) = masse au-dessus de `t_crit` |  | classique |
| Seuil de cluster | `t_{0.975, n-1}` |  | classique |
| Permutations (tirages au hasard des signes) | 1 000 inversions de signe ; toutes les combinaisons quand `> 2^n` ; `p = (1 + count)/(1 + N)` | queue de Pareto généralisée (modèle des valeurs extrêmes) quand `< 10` valeurs de la loi nulle (valeurs sous pur hasard) dépassent et que la loi nulle compte `≥ 50` valeurs, ajustée au-dessus du quantile 0,9 (`permutation.py` `TAIL_*`) | classique ; queue : publié (Knijnenburg 2009), code seulement |
| Fenêtre / ligne de base | (0,009 ; 0,5) s / (-0,5 ; -0,05) s |  | choix de la spec (pensé pour `ccep`) |
| α / correction | `α = 0.05` ; `fdr_bh` par défaut (FDR : part de fausses découvertes ; BH : méthode Benjamini–Hochberg) ; `holm`, `bonferroni`, `none` |  | publié (BH) ; choix de la spec |
| Nombre minimal d'essais | `< 5` → DataError | `MIN_TRIALS = 5` | choix de la spec |
| Loi nulle pour une moyenne déjà faite (Evoked) | fenêtres de ligne de base mélangées (plus faible, avec avertissement) | intervalles par bootstrap de la ligne de base, 200 tirages (DEC §4 REQ-006) | choix de la spec |
| Durée / début | premier / dernier point significatif ; IC par bootstrap sur les essais | `n_boot = 200` | choix de la spec |
| SNR (rapport signal sur bruit) | `10·log10(mean_t ŷ² / σ_n²)` sur `W` |  | classique |
| Durée `projection` | `τ_R = argmax_t P̄(t)/√(t - t_0)` | par défaut `projection_profile="published"` : `argmax_t P̄(t)` (DEC §3) | publié (LIT-ccep-15, LIT-fitting-1) |
| Significativité `projection` | test t à un échantillon sur la moitié des paires sans chevauchement | par défaut permutation `projection_pairs="signflip"` ; `"disjoint"` et `"half"` disponibles (DEC §3, §6 REQ-134) | publié (LIT-ccep-15) |
| `model_comparison` | ΔBIC contre fond seul ; bootstrap paramétrique `p = P(ΔBIC_null ≤ ΔBIC_obs)` | utilise les défauts du paquet pour le moteur (version 1.3) | choix de la spec |
| Règle `vanblooijs2023` | `\|x\| > 3.4·SD_baseline`, pic négatif, ≥ 1 point, 9–100 ms ; ligne de base non précisée | `peak="extremum"` par défaut ; lecture retenue pour les tests de validation `peak="strict"` (minimum local strict le plus profond) ; ligne de base suggérée (-1 ; -0,1) s (`rules.json` ; DEC §5.8) | publié (LIT-ccep-7, LIT-ccep-8, LIT-ccep-30) |
| Règle `vanblooijs2023_code` | pas dans la spec | ligne de base (-2 ; -0,1) s, plancher de SD 50 µV, sélectivité 20 µV, fin de recherche 0,5 s, saturation 3000 µV | lecture du code publié, code seulement |
| Règle `hays2021` | `\|z\| > 6`, 10–50 ms, ligne de base non précisée |  | publié (LIT-ccep-5) |
| Règle `keller2014` | `z > 6` sur A1 sans tenir compte de la polarité, < 50 ms, ligne de base de -500 à -5 ms | fenêtre (0 ; 0,05) | publié (LIT-ccep-3) |
| Règle `silverstein2020` | `z > 5` sur la N1 négative, 10–50 ms |  | publié (LIT-ccep-6) |
| Règle `ftract` | `Z` entre 3 et 7, 0–800 ms, ligne de base de -400 à -10 ms, 1–45 Hz, coupe optionnelle à un délai de 200 ms | `k` non précisé (l'utilisateur doit le donner) | publié (LIT-ccep-14, LIT-ccep-31) |
| Règle `prime2020` | RMS (moyenne quadratique) sur 10–100 ms, seuil non précisé |  | publié (LIT-ccep-47) |
| Désaccord dans un train (suite d'impulsions) | `d = 1 - 2·\|P(sig) - 0.5\|` |  | choix de la spec |
| Polarité | test bilatéral (les deux signes) par défaut ; unilatéral avec `polarity` |  | choix de la spec (REQ-003) |
| Séquentiel (en direct) | règle de dépense d'alpha (risque réparti entre plusieurs regards), fonction non nommée (§5.13) | dépense Lan–DeMets de type O'Brien–Fleming sur les trajectoires de permutation (`sequential.py`) | code seulement |
| κ humain cité (accord entre experts) | 0,60 (§5.6, REQ-135) | `HUMAN_KAPPA = 0.60` | publié (LIT-ccep-8) |

## 5. Ajustement essai par essai

Comment l'outil suit chaque bosse dans chaque essai.

| Élément | Ce que dit la spec (§) | Dans le code, si différent | Origine |
| --- | --- | --- | --- |
| Modèle par essai (§5.7) | `x_i(t) = b_i(t) + Σ_k a_ik·s_k(t - δ_ik; θ̂_k ⊕ γ_ik) + ε_i(t)` | `trial_model="anchored"` par défaut (toutes les composantes du canal sont présentes, seules celles choisies bougent) ; la spec correspond à `"selected"` (`trials/api.py`) | publié (LIT-erpmeas-7, LIT-erpmeas-8, LIT-fitting-3) |
| `max_shift` | `0.25·FWHM_k` en intracrânien, `0.5·FWHM_k` sur le scalp ; coupé à la moitié de la distance à la composante voisine | scalp = `erp`, `p300_bci`, `tep`, `sep` (`SCALP_PRESETS`) | choix de la spec |
| Étirement | `γ ∈ [0.8, 1.25]` |  | choix de la spec |
| Point de départ | `a = 1`, `δ = 0` |  | choix de la spec |
| Mode `free` | paramètres non linéaires à ±25 % des valeurs du canal | `FREE_RANGE = 0.25` | choix de la spec |
| Mode `regularized` | ridge (pénalité qui tire vers une valeur) vers les valeurs du canal, `λ` choisi en laissant un essai de côté à tour de rôle | grille `λ ∈ {0, 0.01, 0.1, 1, 10}` ; `ridge_cv="loto"` | choix de la spec |
| Confiance | `converged` et `r2 ≥ r2_min = 0.2` et décalage pas à la borne |  | choix de la spec |
| Gigue (variation de latence entre essais) | `jitter_sd_k = SD_i(δ_ik)` sur les essais fiables, IC par bootstrap |  | classique |
| Amplitude corrigée de la gigue | `Â_k/a_k ≈ σ_k/√(σ_k² + jitter_sd_k²)` |  | publié (LIT-erpmeas-13) |
| Refus de réaligner | refuser quand `jitter_sd ≤ 1.2·jitter_sd_null` (un segment de ligne de base) | par défaut `null_rule="segments"` : 5 segments de ligne de base par essai, et aussi le quantile 95 % d'une loi F (`JITTER_REFUSE = 1.2`, `NULL_REPEATS = 5`) | choix de la spec |
| Tendance au fil des essais | `a_ik = a_k(1 + c_k·i)` ou `a_k·exp(-i/τ_k)` |  | publié (LIT-variability-4, LIT-variability-6) |
| Dose (Hill : courbe dose-réponse en S) | `a(I) = a_max·I^h/(I_50^h + I^h)`, `h ∈ [0.5, 6]`, linéaire si < 4 courants, `I_thr = I_50·(0.1/0.9)^(1/h)` | `_HILL_MIN_DOSES = 4` | publié (LIT-variability-5, -6, -36, -39) |
| Avant le stimulus | bande 8–13 Hz, fenêtre (-0,5 ; -0,01) s, filtre causal (n'utilise que le passé) + Hilbert (donne la phase instantanée), 3 estimateurs ; `a = c_0 + c_1·cos(φ - φ_0)`, 1 000 permutations ; ligne de base ≥ 3 cycles |  | publié (LIT-variability-16..23) |

## 6. Modèle hiérarchique et bayésien

Comment l'outil met en commun les essais, les canaux et les sujets.

| Élément | Ce que dit la spec (§) | Dans le code, si différent | Origine |
| --- | --- | --- | --- |
| Loi de la gigue de latence (§5.8) | `δ ~ exGaussian(0, σ_δ, τ_δ)` (défaut `latency_dist="exgaussian"`) |  | publié (LIT-fitting-27, LIT-erpmeas-27) |
| Niveaux | `a ~ N(α_sck, σ²_a)`, `α_sck ~ N(α_sk, σ²_α)`, `θ_sck ~ N(θ_sk, Σ_θ)`, `α_sk ~ N(α_k, σ²_group)`, bruit AR(1) |  | publié (LIT-fitting-6, LIT-fitting-30) |
| A priori d'existence | spike-and-slab (soit zéro, soit vraie valeur) `α = z·α̃`, `z ~ Bernoulli(π)` | EB (Bayes empirique) : mélange à 2 composantes, `π` par EM (algorithme espérance-maximisation), `π ∈ [1e-4, 1 - 1e-4]`, tolérance 1e-9, ≤ 1 000 itérations (`eb.py`) | choix de la spec |
| Moteur EB | modèle REML (vraisemblance restreinte) à deux niveaux, rétrécissement (tirer vers la moyenne du groupe) en formule fermée | SE de latence du canal = Wald + gigue commune/√n, coupée à la fenêtre d'étiquette (DEC §5.6) | choix de la spec |
| NUTS (échantillonneur bayésien automatique) | 4 chaînes, 1 000 tirages de chauffe, 1 000 tirages, `target_accept = 0.9` | avertissement de divergence à 1 % | publié (LIT-fitting-38, -40, -44) |
| Canal dominé par l'a priori | `< 3` essais fiables → rétréci à fond, signalé | `MIN_TRUSTED = 3` | choix de la spec |
| Constantes du code seul | pas dans la spec | décalage d'essai ≤ 0,75 FWHM (`JITTER_RANGE`) ; gabarit ±1,5 FWHM ; gabarit de groupe sous 3 canaux ; départ EB à partir des canaux au-delà de 3 SD | code seulement |

## 7. Artefacts de stimulation

Comment l'outil enlève le gros signal laissé par l'impulsion électrique.

| Élément | Ce que dit la spec (§) | Dans le code, si différent | Origine |
| --- | --- | --- | --- |
| Méthode / fenêtre par défaut (§5.3) | `template`, `W_a = (-0.002, 0.008)` s |  | choix de la spec (`ccep`) |
| `interpolate` | linéaire / cubique / PCHIP (cubique sans dépassement) entre les moyennes de `buffer = 3` points |  | publié (LIT-artifact-1) |
| `template` | `T_g = median_{k∈g} x_k` sur `W_a` ; `α_k = ⟨x_k,T_g⟩/⟨T_g,T_g⟩` ; groupes `(stim_site, current, polarity)` ; local `n_neighbors = 10` ; avertissement quand la corrélation gabarit–réponse `> 0.5` ; `< 3` impulsions → erreur ou repli | `fallback=False` par défaut | publié (LIT-artifact-2, LIT-artifact-13) |
| `tail` | `r(t) = c_0 + c_1 e^{-(t-b)/τ_1} + c_2 e^{-(t-b)/τ_2}`, `τ ∈ [0.5, 50] ms`, `τ_2 ≥ 3τ_1`, `tail_length = 0.05` s |  | publié (LIT-fitting-15, LIT-fitting-34) |
| `period_template` | `d_period = 0.01·T`, `n_skip = round(0.05·f_s)`, `n_bins` le plus petit avec `\|B_t\| ≥ 50` | `MIN_BIN_SAMPLES = 50`, 401 candidats par étape, 3 étapes, recherche ±0,02 | publié (LIT-artifact-1 ; brevet LIT-artifact-4) |
| `lowrank` | `(K_C, K_P, K_R) = (4, 2, 4)` ; ≥ 8 canaux, ≥ 4 impulsions, ≥ 4 essais ; fenêtre après l'impulsion de 30 ms |  | publié (LIT-artifact-6) |
| `subspace` | `k` tel que l'énergie retirée ≥ 95 %, au plus 6 | option `n_components="noise"` (2× le bord de Marchenko–Pastur, limite des valeurs propres du bruit) | publié (LIT-artifact-14) |
| `polynomial` | cubique sur `[t - τ, t + τ]`, `τ = 3 ms`, test de confiance sur `0.5τ` | confiance à 3 SD du bruit (`TRUST_SD`) | publié (LIT-artifact-27) |
| `matching_pursuit` | dictionnaire non précisé au-delà des types d'atomes (petites formes de base) | 3 harmoniques du secteur à 50 Hz, atomes gaussiens de SD 2–50 ms par pas de 5 ms, 12 décroissances de 0,2–50 ms, 20 atomes, longueur 0,1 s | publié (LIT-artifact-5 ; brevet LIT-artifact-25) |
| `learned` | base PCA (analyse en composantes principales), variance expliquée ≥ 95 %, au plus 8, ridge `λ` | `ridge = 1e-3` | choix de la spec |
| `paradigm` | `x̂ = (x_cath + x_anod)/2` ; masquage avant (forward masking) `A - B + (C - D)` |  | publié (LIT-artifact-36, LIT-artifact-37) |
| Score de qualité | `q = 1 - clip(mean(p_1..p_4), 0, 1)` ; `p_1 = 0.5·min(RMS(x̂ on [b, b+10 ms])/RMS(baseline), 1)` ; `p_2 = min(\|Δx̂ at edges\|/(3·MAD_baseline), 1)` ; `p_3` = part saturée dans `[a, b + 20 ms]` ; `p_4 = 1 - r²` | `p_2` utilise la MAD des pas d'un point au suivant dans la ligne de base (`quality.py`) | choix de la spec (aucun outil ne note les essais, LIT-artifact-63) |
| Saturation | rail (limite de l'amplificateur) `0.999·max\|x_c\|`, plateau plat de `≥ 3` points sur ±50 ms, récupération 5 ms |  | publié (LIT-artifact-27, LIT-artifact-6) |

## 8. Référence et préparation

Comment l'outil prépare les signaux avant l'ajustement.

| Élément | Ce que dit la spec (§) | Dans le code, si différent | Origine |
| --- | --- | --- | --- |
| `bipolar` (§5.4) | contacts voisins dans chaque groupe, `"A1-A2"` |  | classique (convention MNE) |
| `car` (référence moyenne commune) | moyenne des canaux non mauvais, hors contact de stimulation, non saturés |  | classique |
| `car_adaptive` | classer par covariance entre essais sur 10–300 ms ; `n* = argmax(-ζ_n)` ou premier maximum local significatif (`n_boot = 100`, IC 95 %) ; `< 8` canaux → `car` | options en plus `sensitive`, `line_freq = 60` (`reference.py`) | publié (LIT-ccep-18) |
| `laplacian` (signal moins ses voisins) | voisins dans un rayon `radius` | `radius = 12.0` (en unités des coordonnées) | choix de la spec |
| Sensibilité à la référence | schémas bipolar, car, car\_adaptive ; κ de Cohen | registre des détecteurs (DEC §5.1) | publié (LIT-ccep-27, LIT-ccep-45) |
| Correction de ligne de base | moyenne (ou médiane, ou linéaire) ; `delayed_margin = 1.0` s après l'impulsion précédente ; < 50 % valides → DataError |  | publié (LIT-ccep-19, LIT-tf-2) |
| Rejet d'essais | `z = (v - median)/(1.4826·MAD)` ; `ptp_z = rms_z = 5` (ptp : crête à crête) ; plat si `MAD < 1e-3` ; `quality_min = 0.5` ; masqué `> 0.2` ; plafond `max_fraction = 0.3` | en plus `all_channels_fraction = 0.5` | choix de la spec |
| Moyenne | `trimmed` = moyenne tronquée à 20 % ; poids = qualité |  | classique |
| Filtres | phase nulle `sosfiltfilt`, ordre 4 ; coupe-bande 60 Hz, `q = 30`, 3 harmoniques |  | classique |
| Références des préréglages | `ccep` car\_adaptive ; `ccep_seeg` bipolar ; `tep` average ; `erna` bipolar ; les autres comme enregistrés (§6.1) |  | publié (LIT-ccep-18, LIT-ccep-46, -47, -27) |

## 9. Zones temps-fréquence

Comment l'outil cherche des zones d'activité dans le plan temps-fréquence.

| Élément | Ce que dit la spec (§) | Dans le code, si différent | Origine |
| --- | --- | --- | --- |
| Ondelette de Morse (§5.10) | `γ = 3`, `β = P²/γ`, `P² = 30` → `β = 10` |  | publié (LIT-tf-2, LIT-tf-24) |
| Autres transformées | Morlet (sinus × cloche) `n_cycles = 7` ; multitaper (plusieurs fenêtres moyennées) `NW = 4` ; STFT (Fourier sur fenêtre glissante) | fenêtre STFT de 0,2 s (défaut du code) | classique |
| Fréquences | grille log 2–250 Hz, 40 cases ; retirer `> sfreq/2.5` | `NYQUIST_FACTOR = 2.5` | choix de la spec |
| Fenêtre / ligne de base | (0,0 ; 1,0) s / (-0,5 ; -0,1) s |  | publié (LIT-tf-2) |
| Normalisation | `db = 10·log10(P/P̄_B)` par défaut ; niveau `single_trial` par défaut | cartes de puissance et cartes `*_norm` rendues toutes les deux (DEC §1 D12) | publié (LIT-tf-12, LIT-tf-36) |
| Statistiques | t par pixel contre 0 ; `fdr` par défaut, `cluster`, `tfce` (renforcement des amas sans seuil) (`E = 0.5`, `H = 2`) ; 1 000 permutations ; `α = 0.05` ; `< 10` essais avec cluster → avertissement | pas d'intégration (taille d'une marche) du TFCE `dh = 0.2` ; en plus `baseline_normal` | publié (LIT-tf-2, LIT-tf-10, LIT-tf-38, LIT-tf-39) |
| Zones tirées des données | NMF (découpage en parts positives) ou CP (découpage d'un tableau en facteurs) ; rang choisi par concordance entre moitiés ≥ 0,8 sur 20 redémarrages ; `max_zones = 8` | `STABILITY_MIN = 0.8`, `N_SPLITS = 20`, `DIP_FRACTION = 0.5` | publié (LIT-tf-42..45) |
| Zones publiées | zone tardive 50–500 ms × 50–250 Hz vérifiée ; 10 autres non vérifiées, il faut `allow_unverified` | `zones.json` ; certains champs de la zone vérifiée (mode de ligne de base, niveau, correction des canaux) marqués non vérifiés | publié (LIT-tf-3, LIT-tf-4) |
| Masque d'artefact en temps-fréquence | élargi par le support de l'ondelette à chaque fréquence | énergie du support 99 % (`SUPPORT_ENERGY`) | choix de la spec |

## 10. Statistiques

Les outils pour comparer, mesurer la fidélité et compter les essais nécessaires.

| Élément | Ce que dit la spec (§) | Dans le code, si différent | Origine |
| --- | --- | --- | --- |
| Bootstrap (§5.11) | `B = 1 000`, BCa par défaut, percentile ; stratifié par `by` ; BCa → percentile sous 20 valeurs uniques | par défaut `expand=True` : quantile `√(n/(n-1))·t_{n-1}` (Hesterberg 2015) (`stats/bootstrap.py`) | publié (LIT-fitting-28) |
| Jackknife (retirer un sujet à la fois) | `t_c = t/(n-1)`, `F_c = F/(n-1)²`, `o_i = n·J̄ - (n-1)·J_i` ; `< 3` participants → DataError |  | publié (LIT-erpmeas-2, -4, -5, -6) |
| Comparaison de conditions | permutation, 5 000 ; `g` de Hedges (taille d'effet corrigée) ; `fdr_bh` |  | classique |
| Fidélité | deux moitiés (pair/impair + 1 000 coupes au hasard) avec Spearman–Brown (correction pour la longueur) ; `Φ = σ²_u/(σ²_u + σ²_t/n)`, cible `Φ ≥ 0.70` |  | publié (LIT-erpmeas-36, LIT-erpmeas-38) |
| Essais nécessaires | rééchantillonner `n = 5, 10, …, N`, 200 tirages, extrapolation en `1/√n` |  | publié (LIT-erpmeas-43) |
| Export vers modèle mixte (effets fixes et aléatoires) | poids `1/se²` |  | publié (LIT-fitting-36, LIT-fitting-37) |

## 11. Préréglages

Les réglages tout prêts par type d'expérience, et leurs règles générales.

| Élément | Ce que dit la spec (§) | Dans le code, si différent | Origine |
| --- | --- | --- | --- |
| Règle `min_fwhm` (§6) | `max(narrowest published width, 3/sfreq_typical)` |  | choix de la spec |
| Règle `max_fwhm` | environ la moitié de la fenêtre d'ajustement |  | choix de la spec |
| Règle de polarité | `"both"` partout |  | publié (LIT-ccep-8, LIT-ccep-27) |
| Règle d'existence | `baseline_permutation`, `α = 0.05`, FDR entre canaux |  | choix de la spec |
| Règle de sélection | `bic`, `delta = 10`, `jacobian` | les préréglages imposent aussi `shape_choice="bic"`, `resolution_check=False` (préréglages `fit_fields`) | choix de la spec |
| `erp` `min_fwhm` *(choix)* | 0,010 s (¼ de la composante la plus étroite) |  | choix de la spec |
| Fenêtre d'artefact `ccep` *(choix)* | (-0,002 ; 0,008) s, finit 1 ms avant la fenêtre d'ajustement |  | choix de la spec |
| `sep` `max_fwhm` *(choix)* | 0,025 s |  | choix de la spec |
| `erna` `max_components` *(choix)* | 2 |  | choix de la spec |
| Fenêtres `ecap` (potentiel d'action composé évoqué) *(choix)* | époque (-0,0005 ; 0,003), ligne de base (-0,0005 ; -0,00005), artefact (-0,00005 ; 0,0003), ajustement (0,0003 ; 0,003), étiquettes, `min_fwhm` 0,0001, `max_fwhm` 0,0015 |  | choix de la spec (pas de données publiques) |
| Changements des préréglages dans le code | — | `ccep` `threshold_role="select"`, `background_tau_max = 0.05` ; `ccep_seeg` `background_tau_max = 0.05` ; `erp` `reject_cancelling=True` ; `sep` fond `constant` (spec `linear`) ; `tep` fond `constant` (spec `double_exponential`), formes sans `gabor`, `reject_cancelling=True` ; `ecap`, `ecap_ci` `reseed="always"` ; `rodent_ep` `noise_source="baseline"` (`presets/*.json`) | code seulement / mesuré |
| Règle de sensibilité | chaque *(choix)* changé de ±50 % ; signaler un écart > 10 % de rappel (part des vraies bosses trouvées) ou > 2 ms de latence médiane (§7.4) | lu comme ×0,5 / ×1,5, les entiers arrondis vers le bas sans descendre sous 1 (DEC §6) | choix de la spec |

## 12. Seuils de validation

Les seuils que les tests doivent atteindre pour dire que l'outil marche, une ligne par famille d'exigences.

| Famille | Seuils (spec) | Origine |
| --- | --- | --- |
| Événements REQ-040–045 | rappel, précision (part des trouvailles justes) ≥ 0,99 ; erreur médiane sur le début ≤ 1 point ; accord sur le site ≥ 95 %, exactitude ≥ 0,98 à confiance ≥ 0,5 ; saturation : 100 % des plateaux de ≥ 3 points, ≥ 95 % de ceux de 2 points, ≤ 1 faux positif par 10 min ; ≤ 1 s par 10 min × 100 canaux | choix de la spec |
| Artefact REQ-060–068 | changement de la ligne de base ≤ 2 % du RMS ; ≥ 20 dB de mieux que `interpolate` ; biais de latence ≤ 0,5 ms, biais d'amplitude ≤ 5 % ; Wilcoxon (test sur les rangs) p > 0,05, SD de la période ≤ 1e-4 point ; corrélation ≥ 0,95, RMS des canaux silencieux à 10 % près ; ≥ 30 dB de SIR (rapport signal sur interférence) ; récupération en 2 ms ; AUC (aire sous la courbe ROC) de qualité ≥ 0,9 ; résidu de `paradigm` ≤ 1 % ; ≤ 2 s (≤ 20 s pour lowrank/subspace) | publié (LIT-artifact-1, -6, -14, -27) ; choix de la spec |
| Préparation REQ-080–085 | car\_adaptive : faux négatifs médian 0 jusqu'à 84 % de canaux qui répondent, faux positifs ≤ 2,5 ± 1, Wilcoxon p < 0,01 ; rejet : ≥ 95 % signalés, ≤ 2 % à tort ; sensibilité < 2 min ; latence après rééchantillonnage à `1/(2·sfreq_new)` près | publié (LIT-ccep-18) ; choix de la spec |
| Formes REQ-100–103 | valeurs contre mpmath (calcul en haute précision) 1e-10 ; jacobiennes 1e-6 en relatif ; projection variable 1e-10 | choix de la spec |
| Détection du moteur REQ-104–105 | invariance au seuil ≥ 99 % ; rappel, précision ≥ 0,95 à SNR ≥ 3, FWHM ≥ 3 points (appariement si `\|Δlat\| ≤ FWHM/2`) | choix de la spec |
| Exactitude du moteur REQ-106, 109, 110 | MAE (erreur absolue moyenne) de latence ≤ 0,5 point, biais d'amplitude ≤ 5 % à SNR ≥ 5 ; invariance à un décalage constant 1e-6 ; rappel au bord ≥ 0,95 | choix de la spec |
| Bruit et moteur REQ-107, 112 | fausses composantes ≤ 5 % (AR(1) 0,9) contre ≥ 30 % avec le `n` brut ; ≥ 95 % de `no_response` sur bruit pur et sur les lignes de base de ds004080 | choix de la spec |
| Masques / résolution REQ-108, 111, 121 | rappel ≥ 0,9 avec ≥ 60 % de la FWHM non masquée ; deux composantes à 1,0 FWHM dans ≥ 90 % des cas, une seule + `merged` à 0,5 FWHM dans ≥ 90 % ; mal conditionné à 0,3 FWHM dans 100 % | choix de la spec |
| Cas N1 REQ-113–114 | κ ≥ 0,8, ≤ 1 canal de la règle manqué, ≥ 5 sujets ; latence à 2 ms près sur ≥ 90 % | publié (LIT-ccep-7, LIT-ccep-8) |
| Vitesse du moteur REQ-115 | ≤ 50 ms par courbe de 2 048 points ; ≤ 10 s pour 100 canaux | choix de la spec |
| Couverture REQ-116 (part des IC qui contiennent la vraie valeur) | jacobian 90–97 % (bruit blanc), ≥ 85 % (AR 0,9) ; bootstrap/profile 92–97 % | choix de la spec |
| Classique / sélection REQ-117, 122 | définitions ERPLAB à 1e-9 près ; la validation croisée choisit le vrai K dans ≥ 85 % des cas (40 essais, SNR ≥ 5) | publié (LIT-erpmeas-39) |
| Existence REQ-130–138 | fausse alarme (erreur de type I) dans `[0.8α, 1.2α]` ; puissance (chance de voir un vrai effet) ≥ 0,9 (10 essais, FWHM 10 ms, SNR 2) ; cluster ≥ 0,9 contre max\_t < 0,7 (100 ms, SNR 1) ; durée ±10 %, début 2 ms ; la règle bascule à `k ± 1e-6` ; projection ±5 ms et ≥ 95 % des décisions ; κ ≥ 0,7 contre vanblooijs2023 ; FDR ≤ 5 %, Bonferroni ≤ 1 fausse alarme dans ≥ 95 % ; ≤ 20 s | publié (LIT-ccep-8, LIT-ccep-15) ; choix de la spec |
| Essais REQ-150–158 | MAE de latence ≤ 1 point, amplitude ≤ 15 %, gigue ±20 % (SNR ≥ 2) ; nul ≥ 0,8·gigue, refus ≥ 95 % ; amplitude corrigée ±10 % ; regularized meilleur sur ≥ 90 % ; `I_50` ±15 %, `h` ±30 % ; avant stimulus : fausse alarme ≤ 0,06, puissance ≥ 0,8 ; ≤ 2 s par sujet, essais fiables ≥ 0,8 ; accord de signe dans les trains ≥ 80 % | choix de la spec |
| Hiérarchique REQ-170–176 | AUC ≥ 0,95, pente de calibration 0,8–1,2 ; RMSE (erreur quadratique moyenne) meilleure sur ≥ 90 % ; variances ±20 % ; `r_hat ≤ 1.01` (accord entre chaînes), ESS (nombre de tirages utiles) ≥ 400, < 1 % de divergences ; moteurs à 1 SD près ; effet couvert ≥ 90 % ; EB ≤ 10 s, NUTS ≤ 5 min par sujet | publié (LIT-fitting-38) ; choix de la spec |
| Spatial REQ-190–194 | `n_shapes` juste ≥ 90 %, corrélation ≥ 0,95 ; RMSE −30 % ; pénalité jamais > +1 % ; tri par volume ≥ 90 % ; affectation des formes de base ≥ 85 % | publié (LIT-ccep-16) ; choix de la spec |
| Temps-fréquence REQ-210–218 | Morse 1e-6 ; décomposition 1e-9 ; résidu high-gamma (activité très rapide) ≤ 5 % contre total ≥ 80 % ; fréquence de zone ±10 %, temps ±20 ms, ≥ 90 % ; fausse alarme ≤ 0,06 ; influence d'un seul essai −50 % ; Spearman `\|ρ\| ≥ 0.3` sur la moitié des sujets ; ≤ 60 s | publié (LIT-tf-2) ; choix de la spec |
| Statistiques REQ-230–234 | couverture BCa/percentile 93–97 % (30 essais) ; jackknife 1e-9 ; fausse alarme dans \[0,04 ; 0,06\] ; fidélité ±0,01 ; `trials_needed` ±15 % | publié (LIT-erpmeas-2, -6, -39) |
| Simulation REQ-251–253 | exposant du bruit rose ±0,1, ρ de l'AR ±0,02 ; SNR ±5 % ; résidu du gabarit ≤ 10 % | choix de la spec |
| Préréglages REQ-274–278 | ERP CORE ≥ 90 % des participants, ±10 ms ; TEP ≥ 80 % des sujets ; ERNA ±5 Hz, ±0,1 ms (SNR ≥ 3) ; SEP N20 ±0,5 ms (SNR ≥ 5) ; rongeur ≥ 90 % des canaux | publié (LIT-erpmeas-39, LIT-datasets-17, -50) |
| Flux en direct REQ-347–348 | < 20 ms au 99e centile (64 canaux, 2 048 Hz) ; 1 % d'amplitude, 1 point ; fausses alarmes ≤ α + 0,01 sur 50 regards | choix de la spec |
| Données réelles REQ-371–376 | κ contre les annotateurs ≥ κ entre annotateurs (0,60–0,89) ; Jaccard (recouvrement de deux ensembles) ≥ 0,8 ; Localize-MI ≥ 80 %, faux positifs ≤ α + 0,02 ; garde-fou de régression : κ −0,05, latence +1 ms, rappel −3 points | publié (LIT-ccep-8, LIT-ccep-18) |
| Robustesse §7.4 | signalement des perturbations à > 10 % de rappel ou > 2 ms ; rappel de la N1 à 5 points près à 1 024 Hz ; entre plateformes 1e-9 | choix de la spec |
| Banc d'essai / grille REQ-350, 370 | suite de base ≤ 30 min ; grille de couche 2 ≤ 6 h (une fraction par semaine, DEC §1 D2) ; partie cachée 20 % ; 1 000 / 200 / 50 répétitions | choix de la spec |
| Performance REQ-290 | P1 < 50 ms ; P2 < 60 s sur 4 cœurs ; P3 < 2× les données (< 0,25× + 512 MB en lecture à la demande) ; P4 1e-6 ; P5 < 1,5× ; P6 < 300 ms ; tests automatiques en échec si > 1,25× la référence, alerte si > 1,10× | choix de la spec |

## 13. Autres réglages

Les autres réglages : détection des impulsions, formes partagées, mémoire et reproductibilité.

| Élément | Ce que dit la spec (§) | Dans le code, si différent | Origine |
| --- | --- | --- | --- |
| Détecteur d'impulsions | `z = (d - median d)/(1.4826·MAD d)`, `θ = threshold_sd = 20` ; affinage ±2 ms ; `min_interval = 0.5` s (§5.2) | par défaut `procedure="evoparam"` (fusion 20 ms, pic 3 ms, début à 25 % du score du pic dans les 2 ms) ; procédure de la spec = `"spec"` ; `min_interval=None` (`events/pulses.py`) | code hérité (CODE §1.1) ; code seulement |
| Détecteur par gabarit | `K = 5` premières détections, −1 à +3 ms, `ρ_min = 0.8` | `min_relative = 0.25` | publié (LIT-artifact-13) |
| Règles de cadence | `min_interval = 0.5/rate` ; signale `< 0.5/rate` ou `> 2/rate` ; trains coupés à `> 3/rate` ou 2 s |  | choix de la spec |
| Repérage du site stimulé | `a_c = max_{\|t\|≤1 ms} \|x_c(t) - x_c(-2 ms)\|` ; confiance `(a_1 - a_3)/a_1`, ×0,5 si les contrôles échouent | par défaut `site_method="evidence"` (−5 à +1 ms, avance ≥ 10 SD et ≥ 30 %, maintien 5–100 ms ; DEC §3 K6) | choix de la spec |
| Appariement des débuts | tolérance 10 ms |  | code hérité |
| Formes partagées | NMF sur la matrice t positive ; ARI (accord entre deux regroupements) ≥ 0,8 sur 20 redémarrages ; `max_shapes = 6` ; < 3 groupes → erreur (§5.9) |  | publié (LIT-ccep-16, LIT-tf-45) |
| `fit_shared` | `max_shift = 0.005` s ; rayon laplacien 10 mm ; λ choisi en laissant un canal de côté à tour de rôle |  | choix de la spec |
| SNR du simulateur | amplitude du pic / SD du bruit par essai sur la ligne de base simulée (§5.12) |  | choix de la spec |
| Découpage en blocs (P3) | `chunk × n_trials × n_times × 8 × 3 < max_memory_gb = 0.25 × RAM` (sinon 2 GB) | pas d'argument `chunk` dans `evoparam.fit` (DEC §6 P3) | choix de la spec |
| Reproduire | `rtol = 1e-9`, `atol = 0` (REQ-333) | 1e-9 entre plateformes, identique à l'octet près sur la même machine (DEC §1 D1) | choix de la spec |
| Version du paquet | — | `0.1.0.dev0` ; versions des algorithmes dans `version.py` (fit 1.3, test\_response 1.3, fit\_trials 1.2) | code seulement |
