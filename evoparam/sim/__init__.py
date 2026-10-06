""""Simulation sub-module for evoparam."""

# Link the Sim Params object into `sim`, so it can be imported from here
from evoparam.data import SimParams

from .gen import gen_time_vector, simulate_erp, simulate_erps
