"""Cache the ERP CORE difference waves, their plus-minus noise references and trial counts."""

import sys, glob, os
import numpy as np

sys.path.insert(0, os.path.expanduser("~/evoparam/validation"))
import erpcore

out = {}
for comp in erpcore.DESIGNS:
    folders = sorted(
        int(os.path.basename(p))
        for p in glob.glob(os.path.join(erpcore.ROOT, comp, "*"))
        if os.path.basename(p).isdigit()
    )
    for part in folders:
        try:
            times, data, names, bins = erpcore.load(comp, part)
        except (FileNotFoundError, ValueError):
            continue
        wave, n = erpcore.difference_wave(comp, times, data, names, bins)
        if not np.all(np.isfinite(wave)):
            continue
        ref = erpcore.noise_reference(comp, data, names, bins)
        key = f"{comp}_{part}"
        out[key + "_wave"] = wave
        out[key + "_n"] = n
        if ref is not None:
            out[key + "_ref"] = ref
        out[comp + "_times"] = times
        print(key, flush=True)
np.savez(os.path.expanduser("~/evoparam-figures-seuils/_cache/erp_waves.npz"), **out)
print("done")
