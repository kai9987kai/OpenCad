"""OpenCad - an experimental Python CAD playground.

The package is layered so that geometry can be used without a desktop:

- :mod:`src.kernel` is numpy-only geometry with no UI dependencies.
- :mod:`src.core` holds the scene, project I/O, and the PyVista bridge.
- :mod:`src.ui` holds the PySide6 desktop application.
"""

__version__ = "0.3.0"
