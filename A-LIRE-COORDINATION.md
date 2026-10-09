# À lire avant de commencer une étape (9/10/2026)

- **S18 (mode temps réel) est déjà finie** : branche `s18-en-ligne` (055f3a9), déjà réunie dans `integration-2`.
  `evoparam.stream` (StreamFitter, replay, LSL), `trials.FixedShapeModel`, `existence.OnlineExistence`,
  `events.OnlinePulseDetector`, tutoriel `integration_streaming`. REQ-347 atteint ; p99 ≤ 11,5 ms. Ne pas la refaire.
- **Branche de travail commune : `integration-2`** (S17 + s17-audit 6cfcd7a + S18 + corrections architecture,
  artefact, stats/TF, rapports ; moteur et essais en cours de fusion). Partir de là.
- **Tests réseau impossibles sur le cloud** (OSF et DANDI bloqués) : à lancer sur une machine avec accès,
  `EVOPARAM_NETWORK_TESTS=1 python tools/run_all_tests.py`, surtout ERP CORE P3 (REQ-340), N170 (REQ-233), Allen.
- **Ordre de travail décidé (9/10, 12h26 UTC) : la stimulation d'abord** (CCEP, ECAP, DBS), puis les ERP.
