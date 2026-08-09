"""Scene management, project persistence, and the bridge to the PyVista viewport.

Modules here may import ``pyvista``; :mod:`src.kernel` may not.  Anything in
this package that does *not* need the viewport (for example
:mod:`src.core.history`) is kept import-light so it stays unit testable.
"""
