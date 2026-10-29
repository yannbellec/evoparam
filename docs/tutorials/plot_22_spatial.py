# SPDX-License-Identifier: Apache-2.0
"""
22: Shapes and maps across space
================================

Share shapes across channels, map a parameter on the electrodes, and screen volume conduction.
"""  # noqa: D205, D400  (sphinx-gallery: the docstring opens with an RST title)

###################################################################################################
# Introduction
# ------------
#
# The earlier tutorials fitted each channel on its own. But channels sit in space, and a
# stimulation drives many at once, so there is structure across them. ``evoparam.spatial``
# uses it:
#
# - **shared shapes**: find a few waveform shapes that, combined, explain all the channels
#   of one stimulation, instead of a free fit per channel (the "basis profile curves"
#   idea);
# - **parameter maps**: put one parameter (amplitude, latency) on the electrode
#   coordinates, ready to plot on the brain;
# - **connectivity**: a site-by-channel matrix of a parameter, the input to a connectivity
#   analysis;
# - **a volume-conduction screen**: tell a true local response from the passive spread of a
#   big response nearby.
#
# We simulate one stimulation on a 12-electrode grid, most of them responding.

###################################################################################################

# Import what we need
import warnings

import numpy as np

from evoparam import models, spatial
from evoparam.sim import simulate_ccep

trials, _ = simulate_ccep(
    20, n_channels=12, p_response=0.7, p_n2=1.0, artifact=False, layout="grid",
    sfreq=512.0, seed=0,
)  # fmt: skip
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    group = models.fit(trials, "ccep", random_state=0)

###################################################################################################
# Shared shapes
# -------------
#
# ``shared_shapes`` asks: how many different waveform shapes are really there? It factorises
# all the channels of the stimulation into a few shared shapes and says which channel uses
# which. A handful of shapes that explain many channels is both a cleaner fit and a finding:
# the channels that share a shape are driven in the same way.

###################################################################################################

basis = spatial.shared_shapes(trials, by="channel", max_shapes=3, random_state=0)
print(f"{basis.n_shapes} shared shapes explain the {trials.n_channels} channels "
      f"(median channel r2 = {np.nanmedian(basis.r2):.2f}, "
      f"shape stability = {np.mean(basis.stability):.2f})")

###################################################################################################
# A parameter map
# ---------------
#
# ``parameter_map`` returns one row per channel with the chosen parameter and the electrode
# coordinates, the table you hand to a brain plot. Here the N1 amplitude of each channel,
# with its position.

###################################################################################################

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    amplitude_map = spatial.parameter_map(group, parameter="amplitude", component="N1")
columns = ["channel", "amplitude", "marker"]
print(amplitude_map[columns].round(2).head(8).to_string(index=False))

###################################################################################################
#
# Each row also carries the electrode coordinates (``x``, ``y``, ``z``) when the montage has
# them. ``plots.plot_parameter_map`` draws the map; with a full montage it goes on a brain
# surface through MNE (tutorial on MNE). Our simulated grid has only partial coordinates, so
# here it stays a table.

###################################################################################################
# A connectivity matrix
# ---------------------
#
# ``connectivity_matrix`` lays a parameter out as **stimulated site by recorded channel**:
# with several stimulations it becomes the matrix a connectivity analysis works on (who
# drives whom, and how strongly). Here there is one site, so it is one row.

###################################################################################################

matrix = spatial.connectivity_matrix(group, "amplitude", component="N1")
print(matrix.round(1).to_string())

###################################################################################################
# Screening for volume conduction
# -------------------------------
#
# A big response on one contact spreads passively to its neighbours (volume conduction);
# that spread is not a local response and should not be read as one. ``screen_volume_conduction``
# flags it: it checks for a polarity reversal across the contact, how fast the response
# decays with distance, and whether a different reference removes it. ``volume_conduction_risk``
# is the verdict per channel.

###################################################################################################

screen = spatial.screen_volume_conduction(trials, group)
columns = ["channel", "distance_mm", "reversal", "decay", "reference", "volume_conduction_risk"]
print(screen[columns].head(8).to_string(index=False))

###################################################################################################
#
# A channel with no polarity reversal, a slow decay and a response that survives a change of
# reference is a real local response (low risk); one that reverses, decays fast and vanishes
# under another reference is likely passive spread (high risk). Use the screen to drop the
# passive channels before reading a map as connectivity.

###################################################################################################
# Conclusion
# ----------
#
# The spatial module reads the structure across channels that a per-channel fit ignores:
# shared shapes group the channels driven alike, parameter maps put the numbers on the
# anatomy, the connectivity matrix feeds a network analysis, and the volume-conduction
# screen keeps passive spread out of it. It is the bridge from one stimulation's fit to a
# map of the brain.
