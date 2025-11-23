import pyvista as pv

class TransformManager:
    def __init__(self, viewport, scene, on_update=None):
        self.viewport = viewport
        self.scene = scene
        self.active_widget = None
        self.current_mode = None  # 'translate', 'scale', None
        self.on_update = on_update

    def set_mode(self, mode):
        self.disable_widget()
        self.current_mode = mode
        self.update_widget()

    def update_widget(self):
        oid = self.scene.selected_id
        if not oid or not self.current_mode:
            self.disable_widget()
            return

        actor = self.scene.objects[oid]
        mesh = actor.mapper.dataset

        if self.current_mode == 'translate':
            self.active_widget = self.viewport.plotter.add_box_widget(
                self._on_transform,
                bounds=mesh.bounds,
                color="blue",
                rotation_enabled=False
            )
        elif self.current_mode == 'scale':
            self.active_widget = self.viewport.plotter.add_box_widget(
                self._on_transform,
                bounds=mesh.bounds,
                color="green"
            )
    
    def disable_widget(self):
        if self.active_widget:
            self.viewport.plotter.clear_box_widgets()
            self.active_widget = None

    def _on_transform(self, box_widget):
        oid = self.scene.selected_id
        if not oid: 
            return
            
        actor = self.scene.objects[oid]
        # Get the transform matrix from the widget
        t = box_widget.GetTransform()
        # Apply to actor
        actor.user_matrix = t
        
        if self.on_update:
            self.on_update(oid)
