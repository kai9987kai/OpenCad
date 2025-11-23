import pyvista as pv
import numpy as np

class MeshOps:
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
