# mypy: disable-error-code="attr-defined"
# isort: skip_file
"""Inference of Compartmental Models Toolbox."""

from importlib import metadata as importlib_metadata

from . import comp_model
from .comp_model import (
    CompModel,
    erlang_kernel,
)

from .diffrax_wrapper import diffeqsolve, interpolate_func

from .jax2pytensor import jax2pytensor, jax2pyfunc


def _get_version():
    try:
        return importlib_metadata.version("icomo")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


__author__ = "Jonas Dehning"
__email__ = "jonas.dehning@ds.mpg.de"
__version__ = _get_version()
