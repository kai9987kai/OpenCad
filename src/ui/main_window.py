from pathlib import Path

import pyvista as pv
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QAbstractItemView,
    QDockWidget,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QToolBar,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from src.core.viewport import CADViewport
from src.core.scene import Scene
from src.ui.properties_panel import PropertiesPanel
from src.ui.generative_panel import GenerativePanel

from src.core.transform import TransformManager

from src.core.mesh_ops import MeshOps

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced CAD")
        self.resize(1200, 800)
        
        self.scene = Scene()
        
        # Central Widget - 3D Viewport
        self.viewport = CADViewport(self)
        self.setCentralWidget(self.viewport)
        
        self.transform_mgr = TransformManager(self.viewport, self.scene, self.on_transform_update)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Menubar
        menubar = self.menuBar()
        
        # File Menu
        file_menu = menubar.addMenu("File")
        import_act = QAction("Import Mesh...", self)
        import_act.triggered.connect(self.import_mesh)
        file_menu.addAction(import_act)

        export_act = QAction("Export STL...", self)
        export_act.triggered.connect(self.export_object)
        file_menu.addAction(export_act)

        # Edit Menu
        edit_menu = menubar.addMenu("Edit")
        duplicate_act = QAction("Duplicate Selected", self)
        duplicate_act.triggered.connect(self.duplicate_selected)
        edit_menu.addAction(duplicate_act)

        delete_menu_act = QAction("Delete Selected", self)
        delete_menu_act.triggered.connect(self.delete_selected)
        edit_menu.addAction(delete_menu_act)
        
        # View Menu
        view_menu = menubar.addMenu("View")
        
        wireframe_act = QAction("Toggle Wireframe", self)
        wireframe_act.setCheckable(True)
        wireframe_act.triggered.connect(self.toggle_wireframe)
        view_menu.addAction(wireframe_act)
        
        reset_cam_act = QAction("Reset Camera", self)
        reset_cam_act.triggered.connect(self.reset_camera)
        view_menu.addAction(reset_cam_act)

        view_menu.addSeparator()

        top_view_act = QAction("Top", self)
        top_view_act.triggered.connect(lambda: self.set_camera_view("top"))
        view_menu.addAction(top_view_act)

        front_view_act = QAction("Front", self)
        front_view_act.triggered.connect(lambda: self.set_camera_view("front"))
        view_menu.addAction(front_view_act)

        right_view_act = QAction("Right", self)
        right_view_act.triggered.connect(lambda: self.set_camera_view("right"))
        view_menu.addAction(right_view_act)

        iso_view_act = QAction("Isometric", self)
        iso_view_act.triggered.connect(lambda: self.set_camera_view("iso"))
        view_menu.addAction(iso_view_act)

        # Mesh Menu
        mesh_menu = menubar.addMenu("Mesh")
        
        subdivide_act = QAction("Subdivide", self)
        subdivide_act.triggered.connect(self.op_subdivide)
        mesh_menu.addAction(subdivide_act)
        
        extrude_act = QAction("Extrude (Z)", self)
        extrude_act.triggered.connect(self.op_extrude)
        mesh_menu.addAction(extrude_act)
        
        bevel_act = QAction("Bevel/Smooth", self)
        bevel_act.triggered.connect(self.op_bevel)
        mesh_menu.addAction(bevel_act)

        mesh_menu.addSeparator()

        clean_act = QAction("Clean/Repair Mesh", self)
        clean_act.triggered.connect(self.op_clean)
        mesh_menu.addAction(clean_act)

        decimate_act = QAction("Decimate 50%", self)
        decimate_act.triggered.connect(self.op_decimate)
        mesh_menu.addAction(decimate_act)

        mesh_menu.addSeparator()

        union_act = QAction("Boolean Union", self)
        union_act.triggered.connect(lambda: self.op_boolean("union"))
        mesh_menu.addAction(union_act)

        difference_act = QAction("Boolean Difference", self)
        difference_act.triggered.connect(lambda: self.op_boolean("difference"))
        mesh_menu.addAction(difference_act)

        intersection_act = QAction("Boolean Intersection", self)
        intersection_act.triggered.connect(lambda: self.op_boolean("intersection"))
        mesh_menu.addAction(intersection_act)

        mesh_menu.addSeparator()

        tpms_act = QAction("Generate TPMS Lattice", self)
        tpms_act.triggered.connect(lambda: self.add_tpms_lattice())
        mesh_menu.addAction(tpms_act)

        # Toolbar
        toolbar = QToolBar("Tools")
        self.addToolBar(toolbar)
        
        # Shapes
        add_cube_act = QAction("Add Cube", self)
        add_cube_act.triggered.connect(self.add_cube)
        toolbar.addAction(add_cube_act)
        
        add_sphere_act = QAction("Add Sphere", self)
        add_sphere_act.triggered.connect(self.add_sphere)
        toolbar.addAction(add_sphere_act)

        add_cylinder_act = QAction("Add Cylinder", self)
        add_cylinder_act.triggered.connect(self.add_cylinder)
        toolbar.addAction(add_cylinder_act)

        add_cone_act = QAction("Add Cone", self)
        add_cone_act.triggered.connect(self.add_cone)
        toolbar.addAction(add_cone_act)

        add_torus_act = QAction("Add Torus", self)
        add_torus_act.triggered.connect(self.add_torus)
        toolbar.addAction(add_torus_act)
        
        add_plane_act = QAction("Add Plane", self)
        add_plane_act.triggered.connect(self.add_plane)
        toolbar.addAction(add_plane_act)
        
        add_circle_act = QAction("Add Circle", self)
        add_circle_act.triggered.connect(self.add_circle)
        toolbar.addAction(add_circle_act)
        
        toolbar.addSeparator()

        add_tpms_act = QAction("TPMS Lattice", self)
        add_tpms_act.triggered.connect(lambda: self.add_tpms_lattice())
        toolbar.addAction(add_tpms_act)
        
        toolbar.addSeparator()
        
        delete_act = QAction("Delete", self)
        delete_act.triggered.connect(self.delete_selected)
        toolbar.addAction(delete_act)

        duplicate_toolbar_act = QAction("Duplicate", self)
        duplicate_toolbar_act.triggered.connect(self.duplicate_selected)
        toolbar.addAction(duplicate_toolbar_act)
        
        toolbar.addSeparator()
        
        # Transform Tools
        self.select_act = QAction("Select", self)
        self.select_act.setCheckable(True)
        self.select_act.triggered.connect(lambda: self.set_tool(None))
        toolbar.addAction(self.select_act)
        
        self.move_act = QAction("Move", self)
        self.move_act.setCheckable(True)
        self.move_act.triggered.connect(lambda: self.set_tool("translate"))
        toolbar.addAction(self.move_act)
        
        self.scale_act = QAction("Scale", self)
        self.scale_act.setCheckable(True)
        self.scale_act.triggered.connect(lambda: self.set_tool("scale"))
        toolbar.addAction(self.scale_act)

        self.rotate_act = QAction("Rotate", self)
        self.rotate_act.setCheckable(True)
        self.rotate_act.triggered.connect(lambda: self.set_tool("rotate"))
        toolbar.addAction(self.rotate_act)
        
        # Group for exclusive checking
        self.tools_group = [self.select_act, self.move_act, self.scale_act, self.rotate_act]
        self.select_act.setChecked(True)

        # Dock - Scene Graph
        self.scene_dock = QDockWidget("Scene Graph", self)
        self.scene_list = QListWidget()
        self.scene_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.scene_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.scene_list.customContextMenuRequested.connect(self.show_context_menu)
        self.scene_list.itemClicked.connect(self.on_list_selection)
        self.scene_list.itemChanged.connect(self.on_item_renamed)
        self.scene_dock.setWidget(self.scene_list)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.scene_dock)
        
        # Dock - Properties
        self.prop_dock = QDockWidget("Properties", self)
        self.properties = PropertiesPanel()
        self.properties.property_changed.connect(self.on_property_changed)
        self.prop_dock.setWidget(self.properties)
        self.addDockWidget(Qt.RightDockWidgetArea, self.prop_dock)

        # Dock - Generative Lab
        self.generative_dock = QDockWidget("Generative Lab", self)
        self.generative_panel = GenerativePanel()
        self.generative_panel.generate_requested.connect(self.add_tpms_lattice)
        self.generative_dock.setWidget(self.generative_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.generative_dock)
        self.tabifyDockWidget(self.prop_dock, self.generative_dock)
        self.prop_dock.raise_()

        # Shortcuts
        from PySide6.QtGui import QKeySequence, QShortcut
        self.del_shortcut = QShortcut(QKeySequence.Delete, self)
        self.del_shortcut.activated.connect(self.delete_selected)

    def delete_selected(self):
        items = self.scene_list.selectedItems()
        if not items and self.scene_list.currentItem():
            items = [self.scene_list.currentItem()]

        for item in items:
            self.delete_object(item)

    def show_context_menu(self, pos):
        item = self.scene_list.itemAt(pos)
        if not item:
            return
            
        menu = QMenu(self)
        delete_act = QAction("Delete", self)
        delete_act.triggered.connect(lambda: self.delete_object(item))
        menu.addAction(delete_act)

        duplicate_act = QAction("Duplicate", self)
        duplicate_act.triggered.connect(self.duplicate_selected)
        menu.addAction(duplicate_act)
        
        rename_act = QAction("Rename", self)
        rename_act.triggered.connect(lambda: self.edit_item(item))
        menu.addAction(rename_act)
        
        menu.exec(self.scene_list.mapToGlobal(pos))

    def edit_item(self, item):
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.scene_list.editItem(item)

    def on_item_renamed(self, item):
        oid = self._item_oid(item)
        if not oid:
            return

        name = item.text().strip()
        suffix = f"({oid})"
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()

        if not name:
            name = self.scene.get_name(oid)

        self.scene.rename_object(oid, name)
        desired_text = self._scene_item_text(oid)
        if item.text() != desired_text:
            self.scene_list.blockSignals(True)
            item.setText(desired_text)
            self.scene_list.blockSignals(False)

    def delete_object(self, item):
        oid = self._item_oid(item)
        if not oid:
            return

        actor = self.scene.remove_object(oid)
        if actor:
            self.viewport.plotter.remove_actor(actor)
            self.scene_list.takeItem(self.scene_list.row(item))
            self.properties.clear_selection()
            self.transform_mgr.disable_widget()
            self.viewport.plotter.render()

    def set_tool(self, mode):
        for act in self.tools_group:
            act.setChecked(False)
        
        if mode == "translate":
            self.move_act.setChecked(True)
        elif mode == "scale":
            self.scale_act.setChecked(True)
        elif mode == "rotate":
            self.rotate_act.setChecked(True)
        else:
            self.select_act.setChecked(True)
            
        self.transform_mgr.set_mode(mode)

    def on_object_picked(self, mesh):
        # Find object ID from mesh field data
        if mesh and "id" in mesh.field_data:
            oid = self._field_value(mesh.field_data["id"])
            self.select_object(oid)

    def on_list_selection(self, item):
        oid = self._item_oid(item)
        if oid:
            self.select_object(oid)

    def select_object(self, oid):
        actor = self.scene.select_object(oid)
        if actor:
            self._sync_scene_selection(oid)
            self.properties.set_object(oid, actor)
            self.transform_mgr.update_widget()
            
    def on_property_changed(self, oid, changes):
        if oid in self.scene.objects:
            actor = self.scene.objects[oid]
            if "position" in changes:
                actor.position = changes["position"]
            if "scale" in changes:
                actor.scale = changes["scale"]
            if "orientation" in changes:
                actor.orientation = changes["orientation"]
            if "color" in changes:
                actor.prop.color = changes["color"]
            
            self.viewport.plotter.render()

    def on_transform_update(self, oid):
        if oid in self.scene.objects:
            actor = self.scene.objects[oid]
            self.properties.set_object(oid, actor)

    def import_mesh(self):
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Import Mesh",
            "",
            "Mesh Files (*.stl *.ply *.obj *.vtp *.vtk);;All Files (*)",
        )
        if not fname:
            return

        try:
            mesh = pv.read(fname).extract_surface().triangulate().clean()
        except Exception as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            return

        name = Path(fname).stem or "Imported Mesh"
        self._add_scene_mesh(mesh, name, "#89b4fa")

    def duplicate_selected(self):
        oid = self.scene.selected_id
        if not oid:
            return

        actor = self.scene.objects[oid]
        mesh = actor.mapper.dataset.copy()
        name = f"{self.scene.get_name(oid)} Copy"
        new_oid = self._add_scene_mesh(mesh, name, actor.prop.color)
        new_actor = self.scene.objects[new_oid]
        new_actor.position = (
            actor.position[0] + 0.35,
            actor.position[1] + 0.35,
            actor.position[2],
        )
        new_actor.scale = actor.scale
        new_actor.orientation = actor.orientation
        if actor.user_matrix is not None:
            new_actor.user_matrix = actor.user_matrix.copy()
        self.properties.set_object(new_oid, new_actor)
        self.viewport.plotter.render()

    def add_cube(self):
        self._add_scene_mesh(pv.Cube(), "Cube", "cyan")

    def add_sphere(self):
        self._add_scene_mesh(pv.Sphere(), "Sphere", "magenta")

    def add_cylinder(self):
        mesh = pv.Cylinder(radius=0.5, height=2.0, resolution=64).triangulate()
        self._add_scene_mesh(mesh, "Cylinder", "#94e2d5")

    def add_cone(self):
        mesh = pv.Cone(radius=0.75, height=1.8, resolution=64).triangulate()
        self._add_scene_mesh(mesh, "Cone", "#f9e2af")

    def add_torus(self):
        mesh = pv.ParametricTorus(ringradius=1.0, crosssectionradius=0.25).triangulate()
        self._add_scene_mesh(mesh, "Torus", "#cba6f7")

    def add_plane(self):
        self._add_scene_mesh(pv.Plane(direction=(0, 0, 1)), "Plane", "gray")

    def add_circle(self):
        self._add_scene_mesh(pv.Circle(radius=1.0), "Circle", "yellow")

    def add_tpms_lattice(self, params=None):
        if params is None:
            params = self.generative_panel.parameters()

        self.generative_dock.raise_()
        self.statusBar().showMessage("Generating TPMS lattice...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            mesh = MeshOps.create_tpms_lattice(**params)
        except Exception as exc:
            QMessageBox.warning(self, "TPMS Generation Failed", str(exc))
            self.statusBar().showMessage("TPMS generation failed", 5000)
            return
        finally:
            QApplication.restoreOverrideCursor()

        kind = params["kind"].replace("_", " ").title()
        oid = self._add_scene_mesh(mesh, f"{kind} TPMS", "#8bd5ca", metadata={"generator": params})
        self.statusBar().showMessage(f"Generated {self.scene.get_name(oid)}", 5000)

    def op_subdivide(self):
        oid = self.scene.selected_id
        if not oid: return
        actor = self.scene.objects[oid]
        mesh = actor.mapper.dataset
        
        new_mesh = MeshOps.subdivide(mesh)
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_extrude(self):
        oid = self.scene.selected_id
        if not oid: return
        actor = self.scene.objects[oid]
        mesh = actor.mapper.dataset
        
        new_mesh = MeshOps.extrude(mesh, vector=(0, 0, 1.0))
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_bevel(self):
        oid = self.scene.selected_id
        if not oid: return
        actor = self.scene.objects[oid]
        mesh = actor.mapper.dataset
        
        new_mesh = MeshOps.bevel(mesh)
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_clean(self):
        oid = self.scene.selected_id
        if not oid: return
        actor = self.scene.objects[oid]
        new_mesh = MeshOps.clean(actor.mapper.dataset)
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_decimate(self):
        oid = self.scene.selected_id
        if not oid: return
        actor = self.scene.objects[oid]
        try:
            new_mesh = MeshOps.decimate(actor.mapper.dataset, reduction=0.5)
        except Exception as exc:
            QMessageBox.warning(self, "Decimate Failed", str(exc))
            return
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_boolean(self, operation):
        selected_oids = self._selected_oids()
        if len(selected_oids) < 2:
            QMessageBox.information(
                self,
                "Select Two Objects",
                "Select two scene objects for a boolean operation.",
            )
            return

        first_oid, second_oid = selected_oids[:2]
        first = self._world_mesh(self.scene.objects[first_oid])
        second = self._world_mesh(self.scene.objects[second_oid])

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = MeshOps.boolean(first, second, operation=operation)
        except Exception as exc:
            QMessageBox.warning(self, "Boolean Failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        label = operation.replace("_", " ").title()
        self._add_scene_mesh(result, f"{label} Result", "#fab387")

    def export_object(self):
        oid = self.scene.selected_id
        if not oid:
            return
            
        fname, _ = QFileDialog.getSaveFileName(self, "Export Mesh", "", "STL Files (*.stl);;PLY Files (*.ply);;All Files (*)")
        if fname:
            actor = self.scene.objects[oid]
            save_mesh = self._world_mesh(actor)
            save_mesh.save(fname)

    def toggle_wireframe(self, checked):
        for actor in self.scene.objects.values():
            actor.prop.style = 'wireframe' if checked else 'surface'
        self.viewport.plotter.render()

    def reset_camera(self):
        self.viewport.plotter.reset_camera()

    def set_camera_view(self, view):
        if view == "top":
            self.viewport.plotter.view_xy()
        elif view == "front":
            self.viewport.plotter.view_xz()
        elif view == "right":
            self.viewport.plotter.view_yz()
        elif view == "iso":
            self.viewport.plotter.view_isometric()
        self.viewport.plotter.reset_camera()

    def _selected_oids(self):
        oids = []
        for item in self.scene_list.selectedItems():
            oid = self._item_oid(item)
            if oid and oid in self.scene.objects and oid not in oids:
                oids.append(oid)

        if not oids and self.scene.selected_id:
            oids.append(self.scene.selected_id)
        return oids

    def _world_mesh(self, actor):
        mesh = actor.mapper.dataset.copy()
        matrix = pv.array_from_vtkmatrix(actor.GetMatrix())
        mesh.transform(matrix, inplace=True)
        return mesh.extract_surface().triangulate().clean()

    def _update_actor_mesh(self, oid, actor, new_mesh):
        self.scene.apply_mesh_metadata(new_mesh, oid)
        actor.mapper.dataset = new_mesh
        self.properties.set_object(oid, actor)
        self.transform_mgr.update_widget()
        self.viewport.plotter.render()

    def _add_scene_mesh(self, mesh, name, color, metadata=None):
        oid, mesh = self.scene.add_object(mesh, name, metadata=metadata)
        actor = self.viewport.add_mesh(mesh, color=color, show_edges=True, pickable=True)
        self.scene.apply_mesh_metadata(actor.mapper.dataset, oid)
        self.scene.objects[oid] = actor
        self._add_scene_item(oid)
        self.select_object(oid)
        self.viewport.plotter.reset_camera()
        return oid

    def _add_scene_item(self, oid):
        item = QListWidgetItem(self._scene_item_text(oid))
        item.setData(Qt.UserRole, oid)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.scene_list.addItem(item)

    def _scene_item_text(self, oid):
        return f"{self.scene.get_name(oid)} ({oid})"

    def _item_oid(self, item):
        oid = item.data(Qt.UserRole)
        if oid:
            return oid

        text = item.text()
        if "(" in text and ")" in text:
            return text.split("(")[-1].strip(")")
        return None

    def _sync_scene_selection(self, oid):
        for row in range(self.scene_list.count()):
            item = self.scene_list.item(row)
            if self._item_oid(item) == oid:
                if self.scene_list.currentItem() is not item:
                    self.scene_list.blockSignals(True)
                    self.scene_list.setCurrentItem(item)
                    self.scene_list.blockSignals(False)
                return

    def _field_value(self, value):
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value
