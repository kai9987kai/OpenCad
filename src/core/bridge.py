"""The single seam between the numpy geometry kernel and the PyVista viewport.

Everything that converts a :class:`~src.kernel.mesh.Mesh` into something VTK can
render - or reads geometry back out of an actor - lives here.  Keeping it in one
module is what lets :mod:`src.kernel` stay import-light: the kernel never has to
know that VTK exists, and the UI never has to hand-roll matrix maths.

VTK transform convention
------------------------
A ``vtkProp3D`` composes its matrix as ``user_matrix @ T(position) @ R @ S``,
where ``R`` applies the ``orientation`` triple as Z, then X, then Y.  Rather
than re-deriving that by hand, :func:`actor_matrix` asks VTK for the composed
matrix, so the two can never drift apart.  :func:`compose_matrix` reproduces the
same convention for code paths that have no actor to ask - project loading,
tests, and the headless CLI.
"""

from __future__ import annotations

import numpy as np

from src.kernel.mesh import EPS, Mesh, rotation_matrix

__all__ = [
    "actor_matrix",
    "actor_world_mesh",
    "bake_actor_transform",
    "combined_bounds",
    "compose_matrix",
    "decompose_matrix",
    "from_polydata",
    "normalize_color",
    "reset_actor_transform",
    "set_actor_mesh",
    "to_polydata",
    "world_mesh_from_parts",
]


# ----------------------------------------------------------------------
# Mesh <-> PolyData
# ----------------------------------------------------------------------
def to_polydata(mesh):
    """Convert a kernel mesh into ``pyvista.PolyData`` for rendering."""
    if isinstance(mesh, Mesh):
        return mesh.to_polydata()
    return mesh  # already a PyVista dataset


def from_polydata(dataset):
    """Convert any PyVista dataset into a triangulated kernel mesh."""
    if isinstance(dataset, Mesh):
        return dataset
    return Mesh.from_polydata(dataset)


# ----------------------------------------------------------------------
# Actor transforms
# ----------------------------------------------------------------------
def actor_matrix(actor):
    """The actor's full local-to-world 4x4 matrix, as a numpy array."""
    import pyvista as pv

    return np.asarray(pv.array_from_vtkmatrix(actor.GetMatrix()), dtype=float)


def compose_matrix(position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), user_matrix=None):
    """Build the same matrix VTK would, without needing an actor.

    ``orientation`` is degrees applied in VTK's Z, X, Y order.  When
    ``user_matrix`` is supplied it pre-multiplies the result, matching how VTK
    treats ``vtkProp3D.SetUserMatrix``.
    """
    rx, ry, rz = np.asarray(orientation, dtype=float).reshape(3)
    rotation = rotation_matrix(rz, "z") @ rotation_matrix(rx, "x") @ rotation_matrix(ry, "y")

    matrix = np.eye(4)
    matrix[:3, :3] = rotation @ np.diag(
        np.broadcast_to(np.asarray(scale, dtype=float), (3,))
    )
    matrix[:3, 3] = np.asarray(position, dtype=float).reshape(3)

    if user_matrix is not None:
        matrix = np.asarray(user_matrix, dtype=float).reshape(4, 4) @ matrix
    return matrix


def decompose_matrix(matrix):
    """Split a 4x4 into ``(position, orientation_degrees, scale)``.

    Shear is not representable in an actor's position/orientation/scale triple,
    so a sheared matrix is approximated by orthonormalising its basis.  Callers
    that need exactness should keep the matrix as a ``user_matrix`` instead of
    round-tripping through this function.
    """
    matrix = np.asarray(matrix, dtype=float).reshape(4, 4)
    position = matrix[:3, 3].copy()
    basis = matrix[:3, :3].copy()

    scale = np.linalg.norm(basis, axis=0)
    scale = np.where(scale > EPS, scale, 1.0)
    rotation = basis / scale

    # A left-handed basis means one axis is mirrored; fold that into the scale.
    if np.linalg.det(rotation) < 0:
        scale[0] = -scale[0]
        rotation[:, 0] = -rotation[:, 0]

    # Invert VTK's Z-X-Y order: rotation == Rz @ Rx @ Ry.
    sx = np.clip(rotation[2, 1], -1.0, 1.0)
    x = np.arcsin(sx)
    if np.abs(sx) < 1.0 - 1e-9:
        y = np.arctan2(-rotation[2, 0], rotation[2, 2])
        z = np.arctan2(-rotation[0, 1], rotation[1, 1])
    else:  # gimbal lock: fold Y into Z
        y = 0.0
        z = np.arctan2(rotation[1, 0], rotation[0, 0])

    orientation = np.degrees([x, y, z])
    return position, orientation, scale


def actor_world_mesh(actor):
    """Read an actor's geometry back as a kernel mesh in world coordinates.

    This is the operation booleans, exports, arrays, and measurements all need:
    the displayed geometry, with the actor's interactive transform baked in.
    """
    mesh = from_polydata(actor.mapper.dataset)
    return mesh.transform(actor_matrix(actor))


def world_mesh_from_parts(mesh, position=(0.0, 0.0, 0.0), orientation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0), user_matrix=None):
    """Bake a stored transform into a mesh without constructing an actor.

    Used by project loading and the headless CLI, where the same geometry must
    be reproduced exactly but there is no renderer.
    """
    return from_polydata(mesh).transform(
        compose_matrix(position, orientation, scale, user_matrix)
    )


def set_actor_mesh(actor, mesh):
    """Swap the geometry an actor draws, leaving its transform alone."""
    actor.mapper.dataset = to_polydata(mesh)
    return actor


def reset_actor_transform(actor):
    """Return the actor to identity - position 0, no rotation, unit scale."""
    actor.position = (0.0, 0.0, 0.0)
    actor.scale = (1.0, 1.0, 1.0)
    actor.orientation = (0.0, 0.0, 0.0)
    actor.user_matrix = None
    return actor


def bake_actor_transform(actor):
    """Freeze the actor's transform into its vertices and reset it to identity.

    Returns the baked kernel mesh.  Freezing matters before booleans and export
    because a non-uniform scale on the actor is otherwise invisible to any code
    that reads the mapper's dataset directly.
    """
    baked = actor_world_mesh(actor)
    set_actor_mesh(actor, baked)
    reset_actor_transform(actor)
    return baked


def combined_bounds(actors):
    """Axis-aligned bounds spanning several actors, in PyVista's tuple order."""
    boxes = [actor.GetBounds() for actor in actors]
    if not boxes:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        min(box[0] for box in boxes),
        max(box[1] for box in boxes),
        min(box[2] for box in boxes),
        max(box[3] for box in boxes),
        min(box[4] for box in boxes),
        max(box[5] for box in boxes),
    )


# ----------------------------------------------------------------------
# Colour
# ----------------------------------------------------------------------
def normalize_color(value, default="#89b4fa"):
    """Coerce the many colour shapes floating around the app into ``#rrggbb``.

    VTK hands back floats in 0-1, Qt hands back ``#rrggbb``, project files may
    hold either, and the toolbar uses names like ``"cyan"``.  Everything that
    persists colour goes through here so a saved project reloads identically.
    """
    if value is None:
        return default

    if isinstance(value, str):
        text = value.strip()
        if text.startswith("#") and len(text) in (4, 7):
            if len(text) == 4:  # expand #abc
                return "#" + "".join(channel * 2 for channel in text[1:])
            return text.lower()
        named = _NAMED_COLORS.get(text.lower())
        if named:
            return named
        return default

    try:
        channels = np.asarray(value, dtype=float).ravel()[:3]
    except (TypeError, ValueError):
        return default
    if channels.size < 3 or not np.all(np.isfinite(channels)):
        return default

    # Distinguishing 0-1 floats from 0-255 bytes has to be a heuristic, so make
    # it a careful one: values within 0-1 are floats, whole numbers above 1 are
    # bytes, and anything else is an out-of-range float that we simply clamp.
    # Getting this wrong turns a saved colour black on reload, which is why it
    # is not just ``max > 1``.
    if channels.max() > 1.0 and np.all(np.mod(channels, 1.0) == 0.0):
        channels = channels / 255.0
    channels = np.clip(channels, 0.0, 1.0)
    return "#{:02x}{:02x}{:02x}".format(*(np.round(channels * 255).astype(int)))


_NAMED_COLORS = {
    "black": "#000000",
    "white": "#ffffff",
    "red": "#ff0000",
    "green": "#00ff00",
    "blue": "#0000ff",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
    "yellow": "#ffff00",
    "gray": "#808080",
    "grey": "#808080",
    "orange": "#ffa500",
    "purple": "#800080",
}
