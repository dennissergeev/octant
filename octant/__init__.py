"""octant package."""
import threading

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

__all__ = ["RuntimeOpts", "RUNTIME"]


class RuntimeOpts(threading.local):
    """Run-time configuration controller."""

    def __init__(self, enable_progress_bar=False, progress_bar_lib=None):
        """Initialise run-time configuration."""
        self.__dict__["enable_progress_bar"] = enable_progress_bar
        self.__dict__["progress_bar_lib"] = progress_bar_lib

    def __repr__(self):
        """Repr of run-time configuration."""
        keyval_pairs = [f"{key}={self.__dict__[key]}" for key in self.__dict__]
        msg = f"RuntimeOpts({', '.join(keyval_pairs)})"
        return msg

    def __setattr__(self, name, value):
        """Set attributes."""
        if name not in self.__dict__:
            msg = "'RuntimeOpts' object has no attribute {!r}".format(name)
            raise AttributeError(msg)
        self.__dict__[name] = value


RUNTIME = RuntimeOpts()
