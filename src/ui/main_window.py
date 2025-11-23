from PySide6.QtWidgets import QMainWindow, QDockWidget, QToolBar, QListWidget, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from src.core.viewport import CADViewport
from src.core.scene import Scene
import pyvista as pv
from src.ui.properties_panel import PropertiesPanel

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
        export_act = QAction("Export STL...", self)
        export_act.triggered.connect(self.export_object)
        file_menu.addAction(export_act)
        
        # View Menu
        view_menu = menubar.addMenu("View")
        
        wireframe_act = QAction("Toggle Wireframe", self)
        wireframe_act.setCheckable(True)
        wireframe_act.triggered.connect(self.toggle_wireframe)
        view_menu.addAction(wireframe_act)
        
        reset_cam_act = QAction("Reset Camera", self)
        reset_cam_act.triggered.connect(self.reset_camera)
        view_menu.addAction(reset_cam_act)

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
        
        add_plane_act = QAction("Add Plane", self)
        add_plane_act.triggered.connect(self.add_plane)
        toolbar.addAction(add_plane_act)
        
        add_circle_act = QAction("Add Circle", self)
        add_circle_act.triggered.connect(self.add_circle)
        toolbar.addAction(add_circle_act)
        
        toolbar.addSeparator()
        
        toolbar.addSeparator()
        
        delete_act = QAction("Delete", self)
        delete_act.triggered.connect(self.delete_selected)
        toolbar.addAction(delete_act)
        
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
        
        # Group for exclusive checking
        self.tools_group = [self.select_act, self.move_act, self.scale_act]
        self.select_act.setChecked(True)

        # Dock - Scene Graph
        self.scene_dock = QDockWidget("Scene Graph", self)
        self.scene_list = QListWidget()
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

        # Shortcuts
        from PySide6.QtGui import QKeySequence, QShortcut
        self.del_shortcut = QShortcut(QKeySequence.Delete, self)
        self.del_shortcut.activated.connect(self.delete_selected)

    def delete_selected(self):
        item = self.scene_list.currentItem()
        if item:
            self.delete_object(item)

    def show_context_menu(self, pos):
        item = self.scene_list.itemAt(pos)
        if not item:
            return
            
        menu = QMenu(self)
        delete_act = QAction("Delete", self)
        delete_act.triggered.connect(lambda: self.delete_object(item))
        menu.addAction(delete_act)
        
        rename_act = QAction("Rename", self)
        rename_act.triggered.connect(lambda: self.edit_item(item))
        menu.addAction(rename_act)
        
        menu.exec(self.scene_list.mapToGlobal(pos))

    def edit_item(self, item):
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.scene_list.editItem(item)

    def on_item_renamed(self, item):
        # Handle renaming if needed (update internal name map)
        # For now, we rely on ID which is in the text, but we should parse it carefully
        pass

    def delete_object(self, item):
        text = item.text()
        if "(" in text and ")" in text:
            oid = text.split("(")[-1].strip(")")
            actor = self.scene.remove_object(oid)
            if actor:
                # Remove from viewport
                self.viewport.plotter.remove_actor(actor)
                
                # Remove from list
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
        else:
            self.select_act.setChecked(True)
            
        self.transform_mgr.set_mode(mode)

    def on_object_picked(self, mesh):
        # Find object ID from mesh field data
        if mesh and "id" in mesh.field_data:
            oid = mesh.field_data["id"][0] if isinstance(mesh.field_data["id"], list) else mesh.field_data["id"]
            self.select_object(oid)

    def on_list_selection(self, item):
        # Extract ID from text "Name (ID)"
        text = item.text()
        if "(" in text and ")" in text:
            oid = text.split("(")[-1].strip(")")
            self.select_object(oid)

    def select_object(self, oid):
        actor = self.scene.select_object(oid)
        if actor:
            self.properties.set_object(oid, actor)
            self.transform_mgr.update_widget()
            
    def on_property_changed(self, oid, changes):
        if oid in self.scene.objects:
            actor = self.scene.objects[oid]
            if "position" in changes:
                actor.position = changes["position"]
            if "scale" in changes:
                actor.scale = changes["scale"]
            if "color" in changes:
                actor.prop.color = changes["color"]
            
            self.viewport.plotter.render()

    def on_transform_update(self, oid):
        if oid in self.scene.objects:
            actor = self.scene.objects[oid]
            self.properties.set_object(oid, actor)

    def add_cube(self):
        mesh = pv.Cube()
        oid, _ = self.scene.add_object(mesh, "Cube")
        actor = self.viewport.add_mesh(mesh, color="cyan", show_edges=True, pickable=True)
        # Ensure field data is preserved in the actor's mapper input if needed, 
        # but PyVista actors usually wrap the mesh. 
        # We need to make sure picking returns the mesh with data.
        actor.mapper.dataset.field_data["id"] = oid
        self.scene.objects[oid] = actor
        self.scene_list.addItem(f"Cube ({oid})")

    def add_sphere(self):
        mesh = pv.Sphere()
        oid, _ = self.scene.add_object(mesh, "Sphere")
        actor = self.viewport.add_mesh(mesh, color="magenta", show_edges=True, pickable=True)
        actor.mapper.dataset.field_data["id"] = oid
        self.scene.objects[oid] = actor
        self.scene_list.addItem(f"Sphere ({oid})")

    def add_plane(self):
        mesh = pv.Plane(direction=(0, 0, 1))
        oid, _ = self.scene.add_object(mesh, "Plane")
        actor = self.viewport.add_mesh(mesh, color="gray", show_edges=True, pickable=True)
        actor.mapper.dataset.field_data["id"] = oid
        self.scene.objects[oid] = actor
        self.scene_list.addItem(f"Plane ({oid})")

    def add_circle(self):
        mesh = pv.Circle(radius=1.0)
        oid, _ = self.scene.add_object(mesh, "Circle")
        actor = self.viewport.add_mesh(mesh, color="yellow", show_edges=True, pickable=True)
        actor.mapper.dataset.field_data["id"] = oid
        self.scene.objects[oid] = actor
        self.scene_list.addItem(f"Circle ({oid})")

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

    def export_object(self):
        oid = self.scene.selected_id
        if not oid:
            return
            
        from PySide6.QtWidgets import QFileDialog
        fname, _ = QFileDialog.getSaveFileName(self, "Export Mesh", "", "STL Files (*.stl);;PLY Files (*.ply);;All Files (*)")
        if fname:
            actor = self.scene.objects[oid]
            mesh = actor.mapper.dataset
            
            save_mesh = mesh.copy()
            if actor.user_matrix is not None:
                save_mesh.transform(actor.user_matrix, inplace=True)
                
            save_mesh.save(fname)

    def toggle_wireframe(self, checked):
        for actor in self.scene.objects.values():
            actor.prop.style = 'wireframe' if checked else 'surface'
        self.viewport.plotter.render()

    def reset_camera(self):
        self.viewport.plotter.reset_camera()

    def _update_actor_mesh(self, oid, actor, new_mesh):
        # Preserve ID
        new_mesh.field_data["id"] = oid
        # Update the mapper
        actor.mapper.dataset = new_mesh
        # Force render
        self.viewport.plotter.render()
