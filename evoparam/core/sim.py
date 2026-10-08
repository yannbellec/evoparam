"""Inherited simulator of the skeleton (``sim.gen``, ``sim.params``, ``sim.utils``).

Kept only for the inherited fitter (``evoparam.objs``) and its tests until build
step S7, when both are removed. New code uses :mod:`evoparam.sim`. The power-law
noise of ``sim_noise`` was drawn by ``neurodsp.sim.sim_powerlaw``; it is now
generated here with the same algorithm and the same calls to numpy's global
random state, so the inherited tests see the same numbers without ``neurodsp``.
"""

import numpy as np

from evoparam.core.utils import check_iter, group_three, group_four, check_flat
from evoparam.core.funcs import get_pe_func
from evoparam.data import SimParams

###################################################################################################
###################################################################################################

def gen_time_vector(time_range, fs):
    """Generate a time vector.

    Parameters
    ----------
    time_range : list of [float, float]
        Time range to create time vector across, as [start_time, end_time], inclusive.
    fs : float
        Sampling frequency for desired time vector.

    Returns
    -------
    time : 1d array
        Time values.

    Examples
    --------
    Generate a vector of time values from -1 to 1:

    >>> time = gen_time_vector([-1, 1], fs=2000)
    """

    # The end value has something added to it, to make sure the last value is included
    #   It adds a fraction to not accidentally include points beyond range
    #   due to rounding / or uneven division of the fs into range to simulate
    time = np.arange(time_range[0], time_range[1] + (0.5 / fs), (1/fs))

    return time


def simulate_erp(time_range,  params, nlv=0.005, fs=1000, 
                  peak_mode='gaussian', return_params=False):
    """Generate a simulated ERP.

    Parameters
    ----------
    time_range : list of [float, float]
        Time range to create time vector across, as [start_time, end_time], inclusive.
    params : list of float or generator
        Parameters for the ERP components.
        Length of n_components * 3.
    nlvs : float or list of float or generator, optional, default: 0.005
        Noise level to add to generated ERP.
    fs : float
        Sampling frequency for desired time vector.
    peak_mode : {'gaussian'}, optional
        Which kind of component to generate.
    return_params : bool, optional, default: False
        Whether to return the parameters for the simulated ERP.

    Returns
    -------
    time : 1d array
        Time vector.
    signals : 2d array
        Matrix of signals [n_signals, n_times-1].
    sim_params : list of SimParams
        Definitions of parameters used for each signal. Has length of n_signals.
        Only returned if `return_params` is True.
    """

    time = gen_time_vector(time_range, fs)[:-1]
    signal, time = gen_signal(time, time_range, params, fs, nlv,
                                peak_mode=peak_mode)

    if return_params:
        sim_params = collect_sim_params(params, nlv, peak_mode=peak_mode)
        return time, signal, sim_params
    else:
        return time, signal



def simulate_erps(n_signals, time_range, params, nlvs=0.005, fs=1000, 
                  peak_mode='gaussian', return_params=False):
    """Generate a group of simulated ERPs.

    Parameters
    ----------
    n_signals : int
        The number of ERPs to generate.
    time_range : list of [float, float]
        Time range to create time vector across, as [start_time, end_time], inclusive.
    params : list of float or generator
        Parameters for the ERP components.
        Length of n_peaks * 3.
    nlvs : float or list of float or generator, optional, default: 0.005
        Noise level to add to generated ERPs.
    fs : float
        Sampling frequency for desired time vector.
    peak_mode : {'gaussian'}, optional
        Which kind of component to generate.
    return_params : bool, optional, default: False
        Whether to return the parameters for the simulated ERPs.

    Returns
    -------
    time : 1d array
        Time vector.
    signals : 2d array
        Matrix of signals [n_signals, n_times-1].
    sim_params : list of SimParams
        Definitions of parameters used for each signal. Has length of n_signals.
        Only returned if `return_params` is True.
    """

    # Initialize things
    time = gen_time_vector(time_range, fs)[:-1]
    signals = np.zeros([n_signals, (len(time))])
    sim_params = [None] * n_signals

    # Check if inputs are generators, if not, make them into repeat generators
    pe_params = check_iter(params, n_signals)
    nlvs = check_iter(nlvs, n_signals)

    # Simulate ERPs
    for ind, pe, nlv in zip(range(n_signals), pe_params, nlvs):
        signals[ind, :], time = gen_signal(time, time_range, pe, fs, nlv,
                                            peak_mode=peak_mode)
        sim_params[ind] = collect_sim_params(pe, nlv, peak_mode=peak_mode)

    if return_params:
        return time, signals, sim_params
    else:
        return time, signals


def sim_erp(time, params, peak_mode='gaussian'):
    """Generate signal values.

    Parameters
    ----------
    time : 1d array
        Time vector to create peak values for.
    params : list of float
        Parameters to create the component.
    peak_mode : {'gaussian'}, optional
        Which kind of component to generate.

    Returns
    -------
    peak_vals : 1d array
        Peak values.
    """

    pe_func = get_pe_func(peak_mode)

    pe_vals = pe_func(time, *params)

    return pe_vals


def sim_noise(time_range, params, fs, nlv):
    """Generate noise values for a simulated ERP.

    Parameters
    ----------
    time_range : list of [float, float]
        Times to create noise values.
    params : list of float
        Parameters of ERP components to scale noise to
    fs : float
        Sampling frequency for desired time vector.
    nlv : float
        Noise level to generate.

    Returns
    -------
    noise_vals : 1d vector
        Noise values.

    Notes
    -----
    This approach generates noise as randomly distributed white noise.
    The 'level' of noise is controlled as the scale of the normal distribution.
    """
    noise_vals = _powerlaw_noise(time_range[1]-time_range[0], fs) * (nlv*params[1])

    return noise_vals


def gen_signal(time, time_range, params, fs, nlv, peak_mode='gaussian'):
    """Generate simulated ERP.

    Parameters
    ----------
    time : 1d array
        Time vector to create values for.
    params : list of float
        Parameters to create the ERP components.
    fs : float
        Sampling frequency for desired time vector.
    nlv : float
        Noise level to add to generated signal.

    Returns
    -------
    signal : 1d vector
        simulated signal.

    """

    pe_vals = sim_erp(time, params, peak_mode=peak_mode)
    if nlv != 0:
        noise = sim_noise(time_range, params, fs, nlv)
    else:
        noise = np.zeros(len(pe_vals))

    if len(noise) == (len(pe_vals)-1):
        pe_vals = pe_vals[:-1]
        time = time[:-1]
    signal = pe_vals + noise 

    return signal, time


def _powerlaw_noise(n_seconds, fs, exponent=-2.0):
    """Power-law noise with unit variance, as ``neurodsp.sim.sim_powerlaw`` made it.

    White noise from numpy's global random state is rotated in the frequency
    domain to the power-law ``exponent`` (Timmer & Koenig 1995), then z-scored.
    """

    n_samples = int(np.ceil(n_seconds * fs))
    sig = np.random.standard_normal(n_samples)
    fft_vals = np.fft.fft(sig)
    freqs = np.fft.fftfreq(n_samples, 1. / fs)
    mask = np.ones(n_samples)
    mask[1:] = np.abs(freqs[1:]) ** (exponent / 2)
    sig = np.real(np.fft.ifft(fft_vals * mask))
    sig = (sig - sig.mean()) / sig.std()
    sig = sig / sig.std()
    return sig - sig.mean()


def collect_sim_params(peak_params, nlv, peak_mode='gaussian'):
    """Collect simulation parameters into a SimParams object.

    Parameters
    ----------
    peak_params : list of float or list of list of float
        Parameters of the peak parameters of the ERP.
    nlv : float
        Noise level of the signal.
    peak_mode : {'gaussian'}, optional
        Which kind of component to generate.

    Returns
    -------
    SimParams
        Object containing the simulation parameters.
    """

    if peak_mode == 'gaussian':
        grouped_params = group_three(check_flat(peak_params))
        sorted_grouped_params = sorted(grouped_params, key=lambda x: x[0])
    elif peak_mode == 'skewed_gaussian':
        grouped_params = group_four(check_flat(peak_params))
        sorted_grouped_params = sorted(grouped_params, key=lambda x: x[0])
    else:
        raise ValueError("Peak mode not recognized.")
        
    return SimParams(sorted_grouped_params,
                     nlv)

class Stepper():
    """Object for stepping across parameter values.

    Parameters
    ----------
    start : float
        Start value to iterate from.
    stop : float
        End value to iterate to.
    step : float
        Increment of each iteration.

    Attributes
    ----------
    len : int
        Length of generator range.
    data : iterator
        Set of specified parameters to iterate across.

    Examples
    --------
    Define a stepper object for center time values of a peak, from 10 to 50 ms:

    >>> ct_steps = Stepper(0.010, 0.050, 0.005)
    """

    def __init__(self, start, stop, step):
        """Initialize a Stepper object."""

        self._check_values(start, stop, step)

        self.start = start
        self.stop = stop
        self.step = step
        self.len = round((stop-start)/step)
        self.data = iter(np.arange(start, stop, step))

    def __len__(self):

        return self.len

    def __next__(self):

        return round(next(self.data), 4)

    def __iter__(self):

        return self.data

    @staticmethod
    def _check_values(start, stop, step):
        """Checks if provided values are valid.

        Parameters
        ----------
        start, stop, step : float
            Definition of the parameter range to iterate over.

        Raises
        ------
        ValueError
            If the given values for defining the iteration range are inconsistent.
        """

        if any(ii < 0 for ii in [start, stop, step]):
            raise ValueError("Inputs 'start', 'stop', and 'step' should all be positive values.")

        if not start < stop:
            raise ValueError("Input 'start' should be less than 'stop'.")

        if not step < (stop - start):
            raise ValueError("Input 'step' is too large given values for 'start' and 'stop'.")


def param_iter(params):
    """Create a generator to iterate across parameter ranges.

    Parameters
    ----------
    params : list of floats and Stepper
        Parameters over which to iterate, including a Stepper object.
        The Stepper defines the iterated parameter and its range.
        Floats define the other parameters, that will be held constant.

    Yields
    ------
    list of floats
        Next generated list of parameters.

    Raises
    ------
    ValueError
        If the number of Stepper objects given is greater than one.

    Examples
    --------
    Iterate over center time values from 10 to 50 ms, in increments of 5 ms:

    >>> peaks = param_iter([Stepper(0.010, 0.050, 0.005), 0.5, 0.01])
    """

    # If input is a list of lists, check each element, and flatten if needed
    if isinstance(params[0], list):
        params = [item for sublist in params for item in sublist]

    # Finds where Stepper object is. If there is more than one, raise an error
    iter_ind = 0
    num_iters = 0
    for cur_ind, param in enumerate(params):

        if isinstance(param, Stepper):
            num_iters += 1
            iter_ind = cur_ind

        if num_iters > 1:
            raise ValueError("Iteration is only supported across one parameter at a time.")

    # Generate and yield next set of parameters
    gen = params[iter_ind]
    while True:
        try:
            params[iter_ind] = next(gen)
            yield params
        except StopIteration:
            return


def param_sampler(params, probs=None):
    """Create a generator to sample randomly from possible parameters.

    Parameters
    ----------
    params : list of lists or list of float
        Possible parameter values.
    probs : list of float, optional
        Probabilities with which to sample each parameter option.
        If None, each parameter option is sampled uniformly.

    Yields
    ------
    list of float
        A randomly sampled set of parameters.

    Examples
    --------
    Sample from peak definitions of an N1 alone, or an N1 & N2, with 75% chance of sampling just N1:

    >>> peaks = param_sampler([[0.03, -1, 0.01], [[0.03, -1, 0.01], [0.2, -0.5, 0.05]]],
    ...                       probs=[0.75, 0.25])
    """

    # If input is a list of lists, check each element, and flatten if needed
    if isinstance(params[0], list):
        params = [check_flat(lst) for lst in params]

    # In order to use numpy's choice, with probabilities, choices are made on indices
    # This is because the params can be a messy-sized list, that numpy choice does not like
    inds = np.array(range(len(params)))

    # Check that length of options is same as length of probs, if provided
    if np.any(probs):
        if len(inds) != len(probs):
            raise ValueError("The number of options must match the number of probabilities.")

    # While loop allows the generator to be called an arbitrary number of times
    while True:
        yield params[np.random.choice(inds, p=probs)]


def param_jitter(params, jitters):
    """Create a generator that adds jitter to parameter definitions.

    Parameters
    ----------
    params : list of lists or list of float
        Possible parameter values.
    jitters : list of lists or list of float
        The scale of the jitter for each parameter.
        Must be the same shape and organization as `params`.

    Yields
    ------
    list of float
        A jittered set of parameters.

    Notes
    -----
    - Jitter is added as random samples from a normal (gaussian) distribution.

        - The jitter specified corresponds to the standard deviation of the normal distribution.
    - For any parameter for which there should be no jitter, set the corresponding value to zero.

    Examples
    --------
    Jitter center time of peak definitions, by different amounts for N1 & N2:

    >>> peaks = param_jitter([[0.03, -1, 0.01], [0.2, -0.5, 0.05]], [[0.002, 0, 0], [0.01, 0, 0]])
    """

    # Check if inputs are list of lists, and flatten if so
    if isinstance(params[0], list):
        params = check_flat(params)
        jitters = check_flat(jitters)

    # While loop allows the generator to be called an arbitrary number of times
    while True:

        out_params = [None] * len(params)
        for ind, (param, jitter) in enumerate(zip(params, jitters)):
            out_params[ind] = param + np.random.normal(0, jitter)

        yield out_params


def set_random_seed(seed_value=0):
    """Set the random seed value.

    Parameters
    ----------
    seed_value : int
        Seed value for the random generator.
    """

    np.random.seed(seed_value)
