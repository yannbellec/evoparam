"""One command for the step-6 threshold benchmarks (version 2).

    python benchmarks/seuils/etape6_all.py

1. fetch the 30 ds004080 patients (etape6_fetch_ccep.py; skipped when cached);
2. build the per-subject caches (etape6_data.py);
3. run the CCEP and ERP CORE sweeps (etape6_ccep.py, etape6_erp.py; results cached);
4. compute the statistics, the figures and the CSV (etape6_figures.py) into figures/v2/etape-6/.

Needs the ERP CORE files under $EVOPARAM_ERPCORE (default ~/.cache/evoparam/erpcore) and the
TEP averages of the TEP validation (evoparam/tests/validation/test_tep_ds002094.py).
"""

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

for script, args in (
    ("etape6_fetch_ccep.py", ["30"]),
    ("etape6_data.py", []),
    ("etape6_ccep.py", []),
    ("etape6_erp.py", []),
    ("etape6_figures.py", []),
):
    print(">>", script, flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, script), *args], check=True)
