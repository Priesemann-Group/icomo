# mypy: disable-error-code="attr-defined"
"""Inference of Compartmental Models Toolbox."""
from importlib import metadata as importlib_metadata


from . import comp_model, examples, pytensor_op, slow_modulation, tools
from .comp_model import (
    CompModel,
    ODEIntegrator,
    interpolate_pytensor,
    interpolation_func,
    erlang_kernel,
    delayed_copy,
    SIR,
    Erlang_SEIRS,
)

from .slow_modulation import sigmoidal_changepoints, priors_for_cps
from .tools import hierarchical_priors

import importlib_metadata


def _get_version():
    try:
        return importlib_metadata.version("icomo")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


__author__ = "Jonas Dehning"
__email__ = "jonas.dehning@ds.mpg.de"
__version__ = _get_version()
