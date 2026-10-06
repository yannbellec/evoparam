"""Utilities for input / output for data and models."""

###################################################################################################
###################################################################################################

def load_evoparam(file_name, file_path=None, regenerate=True):
    """Load a evoparam file into a evoparam object.

    Parameters
    ----------
    file_name : str or FileObject
        File to load the data from.
    file_path : str or None, optional
        Path to directory to load from. If None, loads from current directory.
    regenerate : bool, optional, default: True
        Whether to regenerate the model fit from the loaded data, if data is available.

    Returns
    -------
    fm : evoparam
        Object with the loaded data.
    """

    # Initialize a evoparam object (imported locally to avoid circular imports)
    from evoparam.objs import evoparam
    fm = evoparam()

    # Load data into object
    fm.load(file_name, file_path, regenerate)

    return fm


def load_evoparamGroup(file_name, file_path=None):
    """Load data from file into a evoparamGroup object.

    Parameters
    ----------
    file_name : str
        File to load data data.
    file_path : str, optional
        Path to directory to load from. If None, loads from current directory.

    Returns
    -------
    fg : evoparamGroup
        Object with the loaded data.
    """

    # Initialize a evoparamGroup object (imported locally to avoid circular imports)
    from evoparam.objs import evoparamGroup
    fg = evoparamGroup()

    # Load data into object
    fg.load(file_name, file_path)

    return fg
