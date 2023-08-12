# mypy: disable-error-code="attr-defined"
"""Compartmental Models Inference Toolbox."""
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


def _get_version() -> str:
    """Return the program version."""
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "0.1.0"  # Default version


version: str = _get_version()

__author__ = "Jonas Dehning"
__email__ = "jonas.dehning@ds.mpg.de"
__version__: str = version
