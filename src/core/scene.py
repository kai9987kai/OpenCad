import pyvista as pv

class Scene:
    def __init__(self):
        self.objects = {}  # ID -> Actor
        self.selected_id = None
        self._id_counter = 0

    def add_object(self, mesh, name=None):
        oid = f"obj_{self._id_counter}"
        self._id_counter += 1
        if name is None:
            name = f"Object {self._id_counter}"
        
        # Store metadata on the mesh if needed, or keep a separate dict
        mesh.field_data["id"] = oid
        mesh.field_data["name"] = name
        
        return oid, mesh

    def remove_object(self, oid):
        if oid in self.objects:
            actor = self.objects[oid]
            del self.objects[oid]
            if self.selected_id == oid:
                self.selected_id = None
            return actor
        return None

    def select_object(self, oid):
        if oid in self.objects:
            self.selected_id = oid
            return self.objects[oid]
        self.selected_id = None
        return None
