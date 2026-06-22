from importlib.metadata import version as _pkg_version, PackageNotFoundError as _PackageNotFoundError
try:
    __version__ = _pkg_version("organism")
except _PackageNotFoundError:
    __version__ = "0.0.0"  # running from source without install

from .config import OrganismConfig

__all__ = ["OrganismConfig"]