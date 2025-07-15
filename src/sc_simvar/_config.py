"""Configurations for the SCSimVar package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__: str = version(__name__.rsplit(".", 1)[0])  # type: ignore
except PackageNotFoundError:  # pragma: no cover
    __version__: str = "unknown"
finally:
    del version, PackageNotFoundError
