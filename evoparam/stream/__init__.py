# SPDX-License-Identifier: Apache-2.0
"""Online mode: per-pulse evoked parameters from a stream of samples (spec §5.13, §10.6).

For research use only: evoparam is not a medical device and is not validated for
clinical decisions. The online mode exposes numbers (per pulse and channel, the shift and
scale of a frozen template, the derived amplitude and latency, a per-pulse existence score
and a running existence test); it ships no closed-loop control, device driver or safety
interlock, and acting on its output is the user's responsibility.

- :class:`StreamFitter` receives samples (``push``), finds the pulses (threshold detector,
  external triggers through ``push_pulse``, or a callable) and returns one
  :class:`PulseEstimate` per pulse, channel and component once a pulse's window is complete.
  The estimator is :class:`evoparam.trials.FixedShapeModel`, the running existence test
  :class:`evoparam.existence.OnlineExistence` and the detector
  :class:`evoparam.events.OnlinePulseDetector`.
- :class:`LSLInlet` and :class:`LSLOutlet` (extra ``stream``: ``pylsl``) connect it to Lab
  Streaming Layer.
- :func:`replay` pushes recorded data through a fitter, for validation.
"""

from ._fitter import INTENDED_USE, STATUSES, PulseEstimate, StreamFitter, replay
from .lsl import FIELDS, LSLInlet, LSLOutlet

__all__ = [
    "StreamFitter",
    "PulseEstimate",
    "replay",
    "LSLInlet",
    "LSLOutlet",
    "INTENDED_USE",
    "STATUSES",
    "FIELDS",
]
