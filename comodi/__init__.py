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
