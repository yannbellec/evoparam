"""Utilities to create reports and useful print outs."""

from evoparam.core.strings import (gen_version_str, gen_settings_str, gen_time_range_str,
                                gen_methods_report_str, gen_methods_text_str)

###################################################################################################
###################################################################################################

def methods_report_info(evoparam_obj=None, concise=False):
    """Prints out a report of information required for methods reporting.

    Parameters
    ----------
    evoparam_obj : evoparam or evoparamGroup, optional
        An object with setting information available.
        If provided, is used to collect and print information to be reported.
    concise : bool, optional, default: False
        Whether to print the report in concise mode.

    Notes
    -----
    Any missing values (information that is not available) will be printed out as 'XX'.
    """

    print(gen_methods_report_str(concise))

    if evoparam_obj:
        print(gen_version_str(concise))
        print(gen_settings_str(evoparam_obj, concise=concise))
        print(gen_time_range_str(evoparam_obj, concise=concise))

def methods_report_text(evoparam_obj=None):
    """Prints out a text template of methods reporting information.

    Parameters
    ----------
    evoparam_obj : evoparam or evoparamGroup, optional
        An object with setting information available.
        If None, the text is returned as a template, without values.

    Notes
    -----
    Any missing values (information that is not available) will be printed out as 'XX'.
    """

    print(gen_methods_text_str(evoparam_obj))
