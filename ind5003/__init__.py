"""ind5003 — helper utilities for the IND5003 data analytics course at NUS."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("ind5003")
except PackageNotFoundError:
    __version__ = "unknown"

from . import clust, inference, nlp, ts

__all__ = ["clust", "inference", "nlp", "ts", "__version__"]
