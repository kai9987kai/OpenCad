class Scene:
    def __init__(self):
        self.objects = {}  # ID -> Actor
        self.metadata = {}  # ID -> object metadata
        self.selected_id = None
        self._id_counter = 0

    def add_object(self, mesh, name=None, metadata=None):
        oid = f"obj_{self._id_counter}"
        self._id_counter += 1
        if name is None:
            name = f"Object {self._id_counter}"

        self.metadata[oid] = {"name": name}
        if metadata:
            self.metadata[oid].update(metadata)

        self.apply_mesh_metadata(mesh, oid)
        return oid, mesh

    def apply_mesh_metadata(self, mesh, oid):
        mesh.field_data["id"] = [oid]
        mesh.field_data["name"] = [self.get_name(oid)]

    def get_name(self, oid):
        return self.metadata.get(oid, {}).get("name", oid)

    def rename_object(self, oid, name):
        if oid not in self.metadata:
            return
        clean_name = str(name).strip()
        if clean_name:
            self.metadata[oid]["name"] = clean_name
            actor = self.objects.get(oid)
            if actor:
                self.apply_mesh_metadata(actor.mapper.dataset, oid)

    def remove_object(self, oid):
        if oid in self.objects:
            actor = self.objects[oid]
            del self.objects[oid]
            self.metadata.pop(oid, None)
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
