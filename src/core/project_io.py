import json
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pyvista as pv

PROJECT_VERSION = 1


class ProjectIO:
    @staticmethod
    def save(path, objects):
        path = Path(path)
        manifest = {
            "version": PROJECT_VERSION,
            "objects": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            for item in objects:
                mesh_name = f"{item['oid']}.vtp"
                mesh_path = tmpdir / mesh_name
                item["mesh"].save(mesh_path)

                manifest["objects"].append({
                    "oid": item["oid"],
                    "mesh": f"meshes/{mesh_name}",
                    "metadata": item["metadata"],
                    "color": item["color"],
                    "style": item["style"],
                    "opacity": item.get("opacity", 1.0),
                    "show_edges": item.get("show_edges", True),
                    "visible": item["visible"],
                    "position": item["position"],
                    "scale": item["scale"],
                    "orientation": item["orientation"],
                    "user_matrix": ProjectIO._matrix_to_json(item["user_matrix"]),
                })

            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, indent=2))
                for item in manifest["objects"]:
                    archive.write(tmpdir / Path(item["mesh"]).name, item["mesh"])

    @staticmethod
    def load(path):
        path = Path(path)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            with zipfile.ZipFile(path, "r") as archive:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                if manifest.get("version") != PROJECT_VERSION:
                    raise ValueError(f"Unsupported project version: {manifest.get('version')}")

                objects = []
                for item in manifest["objects"]:
                    mesh_path = tmpdir / Path(item["mesh"]).name
                    mesh_path.write_bytes(archive.read(item["mesh"]))
                    mesh = pv.read(mesh_path)

                    loaded = dict(item)
                    loaded["mesh"] = mesh
                    loaded["user_matrix"] = (
                        None if item["user_matrix"] is None else np.array(item["user_matrix"], dtype=float)
                    )
                    objects.append(loaded)

        return objects

    @staticmethod
    def _matrix_to_json(matrix):
        if matrix is None:
            return None
        if hasattr(matrix, "tolist"):
            return matrix.tolist()
        return matrix
