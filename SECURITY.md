# Security policy

## Supported versions

During 0.x only the latest minor release receives fixes.

## Reporting a vulnerability

Please do not open a public issue. Use GitHub's private vulnerability reporting
(*Security* tab of the repository, *Report a vulnerability*), with the version
(`evoparam info`), a description and, if possible, a minimal reproduction. You will get an
answer within 7 days; a fix and an advisory follow as soon as the problem is confirmed.

## Design rules that limit the attack surface

* No network access except the explicit `evoparam.io.fetch_*` functions and readers given
  an `http(s)://` address; `EVOPARAM_OFFLINE=1` forbids both.
* Every download is checked against the sha256 of `evoparam/io/registry.json`.
* No telemetry, no update check.
* No `pickle` in public formats (`.evoparam` containers use JSON, `npz` with
  `allow_pickle=False`, parquet or CSV).
* Plugins are ordinary installed packages: install only plugins you trust;
  `EVOPARAM_DISABLE_PLUGINS=1` turns them all off.

evoparam is research software, not a medical device.
