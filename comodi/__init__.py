# mypy: disable-error-code="attr-defined"
"""Compartmental Models Inference Toolbox."""
from importlib import metadata as importlib_metadata


def get_version() -> str:
    """Return the program version."""
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "0.1.0"  # semantic-release


version: str = get_version()

__author__ = "Jonas Dehning"
__email__ = "jonas.dehning@ds.mpg.de"
__version__: str = version


from . import pytensor_op
from . import comp_model
from . import slow_modulation
from . import examples
from . import tools


from .comp_model import (
    ODEIntegrator,
    interpolate,
    interpolation_func,
    SIR,
    Erlang_SEIR,
    Erlang_SEIRS,
    erlang_kernel,
    delayed_copy,
    CompModel,
)

from .tools import hierarchical_priors
from .slow_modulation import priors_for_cps, sigmoidal_changepoints
