# À lire avant de commencer une étape (9/10/2026)

- **S18 (mode temps réel) est déjà finie** : branche `s18-en-ligne` (055f3a9), déjà réunie dans `integration-2`.
  `evoparam.stream` (StreamFitter, replay, LSL), `trials.FixedShapeModel`, `existence.OnlineExistence`,
  `events.OnlinePulseDetector`, tutoriel `integration_streaming`. REQ-347 atteint ; p99 ≤ 11,5 ms. Ne pas la refaire.
- **Branche de travail commune : `integration-2`** (S17 + s17-audit 6cfcd7a + S18 + corrections architecture,
  artefact, stats/TF, rapports ; moteur et essais en cours de fusion). Partir de là.
- **Tests réseau impossibles sur le cloud** (OSF et DANDI bloqués) : à lancer sur une machine avec accès,
  `EVOPARAM_NETWORK_TESTS=1 python tools/run_all_tests.py`, surtout ERP CORE P3 (REQ-340), N170 (REQ-233), Allen.
- **Ordre de travail décidé (9/10, 12h26 UTC) : la stimulation d'abord** (CCEP, ECAP, DBS), puis les ERP.

## Mise à jour (9/10, 17h UTC) : tests réseau ciblés, à lancer en premier

La batterie complète avec réseau prend plusieurs heures. Ce que le cloud ne peut PAS lancer (OSF, DANDI bloqués) tient en une commande courte, à lancer sur `integration-2` :

    EVOPARAM_NETWORK_TESTS=1 python -m pytest -p no:cacheprovider \
      evoparam/tests/validation/test_erpcore.py \
      evoparam/tests/reports/test_reports_text.py \
      evoparam/tests/stats/test_stats_erpcore.py \
      evoparam/tests/stats/test_stats_jackknife.py \
      evoparam/tests/io/test_allen_lfp.py \
      evoparam/tests/io/test_registry.py

Points à regarder : REQ-340 (ERP CORE P3 réel, texte figé), REQ-233 (N170), REQ-231 (Kiesel 2008 : l'article est aussi inaccessible depuis le cloud), Allen (DANDI). Les tests OpenNeuro (ds004080, ds002094, ds004977) passent déjà depuis le cloud.
