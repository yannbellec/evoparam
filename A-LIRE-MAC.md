# Coordination cloud → Mac (13/10/2026 ~17h UTC)

- REQ-298 (shapes 100 %, cœur 90,5 %), REQ-007 (reproduce de fit_trials / tf_fit), REQ-009
  (unité explicite des amplitudes), REQ-006, 010, 292, 294, 306, 307, 333, 395 : DÉJÀ FAITS,
  dans main = f626c68 (fusion de la branche ingenierie-spec). Ne pas refaire.
- REQ-290 P1 à P6 (dont P6 import < 300 ms) : pris par le cloud (branche vitesse-p1).
- REQ-105, 106, 111 : cloud (branche moteur-sim-2). REQ-301/305 : cloud (qualite-zero).
- Libres pour le Mac (simulations, rien en cours ailleurs) : REQ-116 (couverture des
  intervalles), REQ-150 / 151 (essais shift_scale, jitter_sd_null), REQ-171 (EB), REQ-213
  (zones gamma NMF), REQ-234 (trials_needed), REQ-347 (StreamFitter). Régler sur des graines
  de développement, rapporter sur les graines des tests.
- Windows (REQ-375 (5)) : facturation GitHub, dit à gg ; Wine essayé, NumPy plante.
