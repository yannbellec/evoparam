# SPDX-License-Identifier: Apache-2.0
"""The estimator: ResponseModel.fit_waveform and fit_group."""

import numpy as np
import numpy.typing as npt

from evoparam.data import Evoked, GroupFit, ResponseFit
from evoparam.models import ResponseModel


def one(times: npt.NDArray[np.float64], waveform: npt.NDArray[np.float64]) -> ResponseFit:
    model = ResponseModel("ccep", max_components=4, random_state=0)
    fit: ResponseFit = model.fit_waveform(times, waveform)
    return fit


def many(evoked: Evoked) -> list[float]:
    group: GroupFit = ResponseModel("erp").fit_group(evoked)
    return [fit.gof.r2 for fit in group.fits]
