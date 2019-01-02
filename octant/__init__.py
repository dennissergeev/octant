"""octant package."""
import threading

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__all__ = ["RuntimeOpts", "RUNTIME"]


class RuntimeOpts(threading.local):
    """Run-time configuration controller."""

    def __init__(self, enable_progress_bar=False):
        """Initialise run-time configuration."""
        self.__dict__["enable_progress_bar"] = enable_progress_bar

    def __repr__(self):
        """Repr of run-time configuration."""
        msg = f"RuntimeOpts(enable_progress_bar={self.enable_progress_bar})"
        return msg

    def __setattr__(self, name, value):
        """Set attributes."""
        if name not in self.__dict__:
            msg = "'RuntimeOpts' object has no attribute {!r}".format(name)
            raise AttributeError(msg)
        self.__dict__[name] = value


RUNTIME = RuntimeOpts()
