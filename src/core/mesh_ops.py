import numpy as np
import pyvista as pv


class MeshOps:
    TPMS_TYPES = {
        "gyroid": "Gyroid",
        "schwarz_p": "Schwarz P",
        "diamond": "Diamond",
    }

    @staticmethod
    def subdivide(mesh, levels=1):
        """Subdivides the mesh to increase resolution."""
        return mesh.subdivide(levels, subfilter='linear')

    @staticmethod
    def extrude(mesh, vector=(0, 0, 1.0)):
        """Extrudes a mesh (usually 2D) along a vector."""
        return mesh.extrude(vector, capping=True)

    @staticmethod
    def bevel(mesh, amount=0.1):
        """
        Approximates a bevel by smoothing the mesh.
        True beveling requires complex topology changes not easily available in standard VTK filters
        without custom algorithms. We use a smooth filter here as a proxy.
        """
        # smooth_taubin is generally better for preserving volume than standard smooth
        return mesh.smooth_taubin(n_iter=20, pass_band=0.1)

    @staticmethod
    def clean(mesh):
        """Repair common mesh issues and normalize topology for later operations."""
        return mesh.clean().triangulate()

    @staticmethod
    def decimate(mesh, reduction=0.5):
        """Reduce triangle count while preserving the overall surface."""
        reduction = float(np.clip(reduction, 0.05, 0.9))
        return mesh.triangulate().decimate_pro(reduction).clean()

    @staticmethod
    def boolean(mesh_a, mesh_b, operation="union"):
        """Run a boolean operation on two closed, triangulated surface meshes."""
        a = mesh_a.extract_surface(algorithm="dataset_surface").triangulate().clean()
        b = mesh_b.extract_surface(algorithm="dataset_surface").triangulate().clean()
        operation = str(operation).lower()

        if operation == "union":
            result = a.boolean_union(b)
        elif operation == "difference":
            result = a.boolean_difference(b)
        elif operation == "intersection":
            result = a.boolean_intersection(b)
        else:
            raise ValueError(f"Unsupported boolean operation '{operation}'.")

        if result.n_points == 0:
            raise ValueError("The boolean operation produced an empty mesh.")
        return result.clean().triangulate()

    @staticmethod
    def mirror(mesh, axis="x"):
        """Mirror mesh geometry across a global origin plane."""
        mirrored = mesh.copy()
        axis_index = MeshOps._axis_index(axis)
        mirrored.points[:, axis_index] *= -1.0
        return MeshOps._repair_normals(mirrored)

    @staticmethod
    def translate(mesh, vector):
        translated = mesh.copy()
        translated.points += np.asarray(vector, dtype=float)
        return translated

    @staticmethod
    def rotate_around_origin(mesh, angle_degrees, axis="z"):
        rotated = mesh.copy()
        axis = str(axis).lower()
        angle = np.deg2rad(float(angle_degrees))
        c, s = np.cos(angle), np.sin(angle)

        if axis == "x":
            matrix = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == "y":
            matrix = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == "z":
            matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise ValueError("Axis must be X, Y, or Z.")

        rotated.points = rotated.points @ matrix.T
        return MeshOps._repair_normals(rotated)

    @staticmethod
    def offset_surface(mesh, distance=0.1):
        """Create an experimental shell-like surface offset along vertex normals."""
        distance = float(distance)
        surface = mesh.extract_surface(algorithm="dataset_surface").triangulate().compute_normals(
            point_normals=True,
            cell_normals=False,
            inplace=False,
        )
        normals = surface.point_data.get("Normals")
        if normals is None:
            raise ValueError("Could not compute normals for offset surface.")

        offset = surface.copy()
        offset.points = offset.points + normals * distance
        return MeshOps._repair_normals(offset)

    @staticmethod
    def clip(mesh, axis="z", origin=0.0, keep_positive=True):
        """Clip a mesh with an axis-aligned plane."""
        axis = str(axis).lower()
        MeshOps._axis_index(axis)
        origin = float(origin)
        origin_point = {
            "x": (origin, 0.0, 0.0),
            "y": (0.0, origin, 0.0),
            "z": (0.0, 0.0, origin),
        }[axis]
        clipped = mesh.clip(
            normal=axis,
            origin=origin_point,
            invert=not bool(keep_positive),
        )
        if clipped.n_points == 0:
            raise ValueError("The clip operation produced an empty mesh.")
        return clipped.extract_surface(algorithm="dataset_surface").triangulate().clean()

    @staticmethod
    def largest_component(mesh):
        """Keep only the largest connected mesh island."""
        largest = mesh.extract_surface(algorithm="dataset_surface").extract_largest()
        if largest.n_points == 0:
            raise ValueError("No connected component could be extracted.")
        return largest.triangulate().clean()

    @staticmethod
    def snap_points(mesh, spacing=0.25):
        """Quantize mesh vertices to a regular grid."""
        spacing = max(float(spacing), 1e-6)
        snapped = mesh.copy()
        snapped.points = np.round(snapped.points / spacing) * spacing
        return snapped.clean(tolerance=spacing * 1e-6).triangulate()

    @staticmethod
    def recompute_normals(mesh):
        """Rebuild consistent surface normals."""
        return MeshOps._repair_normals(mesh)

    @staticmethod
    def flip_normals(mesh):
        """Reverse surface orientation and normals."""
        flipped = mesh.extract_surface(algorithm="dataset_surface").triangulate().flip_faces()
        return MeshOps._repair_normals(flipped)

    @staticmethod
    def create_tpms_lattice(
        kind="gyroid",
        size=4.0,
        cells=3,
        resolution=56,
        level=0.0,
        gradient=0.0,
    ):
        """
        Generate a triply periodic minimal surface (TPMS) iso-surface.

        TPMS geometry is useful for additive-manufacturing lattices because the
        implicit field can be sampled at different densities without manually
        building repeated cell topology.
        """
        kind = MeshOps._normalize_tpms_kind(kind)
        size = max(float(size), 0.1)
        cells = max(float(cells), 0.25)
        resolution = int(np.clip(resolution, 16, 140))
        level = float(level)
        gradient = float(np.clip(gradient, -1.5, 1.5))

        axis = np.linspace(-size / 2.0, size / 2.0, resolution)
        x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")

        frequency = (2.0 * np.pi * cells) / size
        field = MeshOps._tpms_field(kind, x * frequency, y * frequency, z * frequency)

        contour_level = level
        if gradient:
            z_normalized = z / (size / 2.0)
            field = field - (level + gradient * z_normalized)
            contour_level = 0.0

        spacing = (size / (resolution - 1),) * 3
        grid = pv.ImageData(
            dimensions=(resolution, resolution, resolution),
            spacing=spacing,
            origin=(-size / 2.0, -size / 2.0, -size / 2.0),
        )
        grid.point_data["implicit"] = np.ascontiguousarray(field.ravel(order="F"))

        try:
            mesh = grid.contour([contour_level], scalars="implicit", method="flying_edges")
        except TypeError:
            mesh = grid.contour([contour_level], scalars="implicit")

        if mesh.n_points == 0:
            raise ValueError("The TPMS settings produced an empty mesh. Try reducing level or gradient.")

        mesh = mesh.triangulate().clean(tolerance=size * 1e-6)
        try:
            mesh = mesh.compute_normals(
                auto_orient_normals=True,
                consistent_normals=True,
                inplace=False,
            )
        except TypeError:
            mesh = mesh.compute_normals(inplace=False)

        mesh.field_data["generator"] = ["tpms"]
        mesh.field_data["tpms_kind"] = [kind]
        mesh.field_data["tpms_cells"] = [cells]
        mesh.field_data["tpms_level"] = [level]
        mesh.field_data["tpms_gradient"] = [gradient]
        return mesh

    @staticmethod
    def _normalize_tpms_kind(kind):
        normalized = str(kind).lower().replace(" ", "_").replace("-", "_")
        if normalized in {"schwarz", "schwarzp", "primitive"}:
            normalized = "schwarz_p"
        if normalized not in MeshOps.TPMS_TYPES:
            supported = ", ".join(MeshOps.TPMS_TYPES.values())
            raise ValueError(f"Unsupported TPMS type '{kind}'. Supported types: {supported}.")
        return normalized

    @staticmethod
    def _tpms_field(kind, x, y, z):
        if kind == "gyroid":
            return np.sin(x) * np.cos(y) + np.sin(y) * np.cos(z) + np.sin(z) * np.cos(x)
        if kind == "schwarz_p":
            return np.cos(x) + np.cos(y) + np.cos(z)
        if kind == "diamond":
            return (
                np.sin(x) * np.sin(y) * np.sin(z)
                + np.sin(x) * np.cos(y) * np.cos(z)
                + np.cos(x) * np.sin(y) * np.cos(z)
                + np.cos(x) * np.cos(y) * np.sin(z)
            )
        raise ValueError(f"Unsupported TPMS type '{kind}'.")

    @staticmethod
    def _axis_index(axis):
        normalized = str(axis).lower()
        if normalized == "x":
            return 0
        if normalized == "y":
            return 1
        if normalized == "z":
            return 2
        raise ValueError("Axis must be X, Y, or Z.")

    @staticmethod
    def _repair_normals(mesh):
        try:
            return mesh.triangulate().clean().compute_normals(
                auto_orient_normals=True,
                consistent_normals=True,
                inplace=False,
            )
        except TypeError:
            return mesh.triangulate().clean().compute_normals(inplace=False)
