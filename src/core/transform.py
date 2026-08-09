import numpy as np
import pyvista as pv
import vtk


class TransformManager:
    def __init__(self, viewport, scene, on_update=None):
        self.viewport = viewport
        self.scene = scene
        self.active_widget = None
        self.current_mode = None  # 'translate', 'scale', None
        self.on_update = on_update
        self._base_user_matrix = np.eye(4)

    def set_mode(self, mode):
        self.disable_widget()
        self.current_mode = mode
        self.update_widget()

    def update_widget(self):
        self.disable_widget()
        oid = self.scene.selected_id
        if not oid or not self.current_mode:
            return

        actor = self.scene.objects[oid]
        if self.scene.metadata.get(oid, {}).get("locked") or not actor.GetVisibility():
            return

        bounds = actor.GetBounds()
        self._base_user_matrix = self._actor_user_matrix(actor)

        if self.current_mode == 'translate':
            self.active_widget = self.viewport.plotter.add_box_widget(
                self._on_transform,
                bounds=bounds,
                color="blue",
                rotation_enabled=False,
                outline_translation=True,
                pass_widget=True,
                interaction_event="always",
            )
        elif self.current_mode == 'scale':
            self.active_widget = self.viewport.plotter.add_box_widget(
                self._on_transform,
                bounds=bounds,
                color="green",
                rotation_enabled=False,
                outline_translation=True,
                pass_widget=True,
                interaction_event="always",
            )
        elif self.current_mode == 'rotate':
            self.active_widget = self.viewport.plotter.add_box_widget(
                self._on_transform,
                bounds=bounds,
                color="orange",
                rotation_enabled=True,
                outline_translation=True,
                pass_widget=True,
                interaction_event="always",
            )

    def disable_widget(self):
        if self.active_widget:
            self.viewport.plotter.clear_box_widgets()
            self.active_widget = None

    def _on_transform(self, _box_mesh, box_widget):
        oid = self.scene.selected_id
        if not oid:
            return

        actor = self.scene.objects[oid]
        transform = vtk.vtkTransform()
        box_widget.GetTransform(transform)
        widget_matrix = pv.array_from_vtkmatrix(transform.GetMatrix())
        actor.user_matrix = widget_matrix @ self._base_user_matrix

        if self.on_update:
            self.on_update(oid)

    def _actor_user_matrix(self, actor):
        matrix = actor.user_matrix
        if matrix is None:
            return np.eye(4)
        return np.array(matrix, dtype=float)
