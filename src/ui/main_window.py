import contextlib
from pathlib import Path

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QToolBar,
)

from src.core.bridge import actor_world_mesh, to_polydata
from src.core.mesh_ops import MeshOps
from src.core.project_io import ProjectIO
from src.core.scene import Scene
from src.core.transform import TransformManager
from src.core.viewport import CADViewport
from src.kernel import io_mesh
from src.kernel.analysis import mesh_report, printability, volume_fraction
from src.kernel.mesh import Mesh
from src.kernel.meshing import surface_nets
from src.ui.analysis_panel import AnalysisPanel
from src.ui.appearance_panel import AppearancePanel
from src.ui.generative_panel import GenerativePanel
from src.ui.grid_panel import GridPanel
from src.ui.implicit_panel import ImplicitPanel, build_lattice_field
from src.ui.measurements_panel import MeasurementsPanel
from src.ui.primitive_panel import PrimitivePanel
from src.ui.properties_panel import PropertiesPanel
from src.ui.tasks import TaskRunner
from src.ui.tool_dialogs import (
    CircularArrayDialog,
    ClipDialog,
    DecimateDialog,
    LinearArrayDialog,
    OffsetSurfaceDialog,
)

# ----------------------------------------------------------------------
# Worker functions.
#
# These run on a thread pool, so they must touch nothing but plain data - no
# actors, no widgets, no plotter. They return kernel meshes, which are numpy
# arrays and safe to hand back to the GUI thread.
# ----------------------------------------------------------------------


def _lattice_worker(context, params, bounds):
    field, sampling_box, _ = build_lattice_field(params, bounds)

    def report(fraction, message):
        context.raise_if_cancelled()
        context.progress(fraction, message)

    return surface_nets(
        field, bounds=sampling_box, resolution=int(params["resolution"]), progress=report
    )


def _density_worker(context, params, bounds):
    field, _, region = build_lattice_field(params, bounds)
    context.progress(0.2, "Sampling volume")
    # Measure against the region the user asked to fill, not the padded sampling
    # box - padding is empty space and would understate the density.
    # A coarse grid is plenty for a density estimate and keeps it interactive.
    return volume_fraction(field, bounds=region, resolution=min(int(params["resolution"]), 96))


def _analysis_worker(context, mesh, params):
    context.progress(0.2, "Measuring geometry")
    report = mesh_report(mesh)
    context.raise_if_cancelled()
    context.progress(0.6, "Checking printability")
    return report, printability(mesh, **params)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenCad")
        self.resize(1200, 800)

        self.scene = Scene()
        self.undo_stack = []
        self.redo_stack = []
        self.history_limit = 40
        self._restoring_history = False

        # Heavy geometry runs here so the viewport keeps repainting.
        self.tasks = TaskRunner(self)
        self.tasks.progress.connect(self._on_task_progress)
        self.tasks.busy_changed.connect(self._on_busy_changed)

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
        new_project_act = QAction("New Project", self)
        new_project_act.triggered.connect(self.new_project)
        file_menu.addAction(new_project_act)

        open_project_act = QAction("Open Project...", self)
        open_project_act.triggered.connect(self.open_project)
        file_menu.addAction(open_project_act)

        save_project_act = QAction("Save Project...", self)
        save_project_act.triggered.connect(self.save_project)
        file_menu.addAction(save_project_act)

        file_menu.addSeparator()

        import_act = QAction("Import Mesh...", self)
        import_act.triggered.connect(self.import_mesh)
        file_menu.addAction(import_act)

        export_act = QAction("Export STL...", self)
        export_act.triggered.connect(self.export_object)
        file_menu.addAction(export_act)

        export_scene_act = QAction("Export Visible Scene...", self)
        export_scene_act.triggered.connect(self.export_visible_scene)
        file_menu.addAction(export_scene_act)

        # Edit Menu
        edit_menu = menubar.addMenu("Edit")
        undo_act = QAction("Undo", self)
        undo_act.setShortcut("Ctrl+Z")
        undo_act.triggered.connect(self.undo)
        edit_menu.addAction(undo_act)

        redo_act = QAction("Redo", self)
        redo_act.setShortcut("Ctrl+Y")
        redo_act.triggered.connect(self.redo)
        edit_menu.addAction(redo_act)

        edit_menu.addSeparator()

        duplicate_act = QAction("Duplicate Selected", self)
        duplicate_act.triggered.connect(self.duplicate_selected)
        edit_menu.addAction(duplicate_act)

        delete_menu_act = QAction("Delete Selected", self)
        delete_menu_act.triggered.connect(self.delete_selected)
        edit_menu.addAction(delete_menu_act)

        edit_menu.addSeparator()

        lock_act = QAction("Lock Selected", self)
        lock_act.triggered.connect(lambda: self.lock_selected(True))
        edit_menu.addAction(lock_act)

        unlock_act = QAction("Unlock Selected", self)
        unlock_act.triggered.connect(lambda: self.lock_selected(False))
        edit_menu.addAction(unlock_act)

        # Modify Menu
        modify_menu = menubar.addMenu("Modify")

        linear_array_act = QAction("Linear Array...", self)
        linear_array_act.triggered.connect(self.linear_array_selected)
        modify_menu.addAction(linear_array_act)

        circular_array_act = QAction("Circular Array...", self)
        circular_array_act.triggered.connect(self.circular_array_selected)
        modify_menu.addAction(circular_array_act)

        modify_menu.addSeparator()

        for axis in ("X", "Y", "Z"):
            mirror_act = QAction(f"Mirror Across {axis}=0", self)
            mirror_act.triggered.connect(lambda checked=False, axis=axis.lower(): self.mirror_selected(axis))
            modify_menu.addAction(mirror_act)

        modify_menu.addSeparator()

        align_origin_act = QAction("Center to Origin", self)
        align_origin_act.triggered.connect(lambda: self.align_selected("origin"))
        modify_menu.addAction(align_origin_act)

        align_floor_act = QAction("Drop to Floor", self)
        align_floor_act.triggered.connect(lambda: self.align_selected("floor"))
        modify_menu.addAction(align_floor_act)

        freeze_act = QAction("Freeze Transforms", self)
        freeze_act.triggered.connect(self.freeze_selected_transforms)
        modify_menu.addAction(freeze_act)

        # View Menu
        view_menu = menubar.addMenu("View")

        wireframe_act = QAction("Toggle Wireframe", self)
        wireframe_act.setCheckable(True)
        wireframe_act.triggered.connect(self.toggle_wireframe)
        view_menu.addAction(wireframe_act)

        reset_cam_act = QAction("Reset Camera", self)
        reset_cam_act.triggered.connect(self.reset_camera)
        view_menu.addAction(reset_cam_act)

        fit_selection_act = QAction("Fit Selection", self)
        fit_selection_act.triggered.connect(self.fit_selection)
        view_menu.addAction(fit_selection_act)

        view_menu.addSeparator()

        orthographic_act = QAction("Orthographic Projection", self)
        orthographic_act.triggered.connect(lambda: self.set_projection(True))
        view_menu.addAction(orthographic_act)

        perspective_act = QAction("Perspective Projection", self)
        perspective_act.triggered.connect(lambda: self.set_projection(False))
        view_menu.addAction(perspective_act)

        view_menu.addSeparator()

        hide_act = QAction("Hide Selected", self)
        hide_act.triggered.connect(lambda: self.set_selected_visibility(False))
        view_menu.addAction(hide_act)

        show_selected_act = QAction("Show Selected", self)
        show_selected_act.triggered.connect(lambda: self.set_selected_visibility(True))
        view_menu.addAction(show_selected_act)

        isolate_act = QAction("Isolate Selected", self)
        isolate_act.triggered.connect(self.isolate_selected)
        view_menu.addAction(isolate_act)

        show_all_act = QAction("Show All", self)
        show_all_act.triggered.connect(self.show_all)
        view_menu.addAction(show_all_act)

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

        decimate_act = QAction("Decimate...", self)
        decimate_act.triggered.connect(self.op_decimate)
        mesh_menu.addAction(decimate_act)

        offset_act = QAction("Offset Surface...", self)
        offset_act.triggered.connect(self.op_offset_surface)
        mesh_menu.addAction(offset_act)

        clip_act = QAction("Clip Mesh...", self)
        clip_act.triggered.connect(self.op_clip)
        mesh_menu.addAction(clip_act)

        largest_act = QAction("Keep Largest Component", self)
        largest_act.triggered.connect(self.op_largest_component)
        mesh_menu.addAction(largest_act)

        recompute_normals_act = QAction("Recompute Normals", self)
        recompute_normals_act.triggered.connect(self.op_recompute_normals)
        mesh_menu.addAction(recompute_normals_act)

        flip_normals_act = QAction("Flip Normals", self)
        flip_normals_act.triggered.connect(self.op_flip_normals)
        mesh_menu.addAction(flip_normals_act)

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

        # Analyse Menu
        analyse_menu = menubar.addMenu("Analyse")

        printability_act = QAction("Printability Report", self)
        printability_act.setShortcut("Ctrl+R")
        printability_act.triggered.connect(
            lambda: self.run_analysis(self.analysis_panel.parameters())
        )
        analyse_menu.addAction(printability_act)

        # The docks are built further down setup_ui, so these must be lambdas -
        # a bound method reference would be resolved now, before they exist.
        show_analysis_act = QAction("Show Analysis Dock", self)
        show_analysis_act.triggered.connect(lambda: self.analysis_dock.raise_())
        analyse_menu.addAction(show_analysis_act)

        analyse_menu.addSeparator()

        implicit_act = QAction("Implicit Lab", self)
        implicit_act.setShortcut("Ctrl+L")
        implicit_act.triggered.connect(lambda: self.implicit_dock.raise_())
        analyse_menu.addAction(implicit_act)

        lattice_act = QAction("Generate Graded Lattice", self)
        lattice_act.triggered.connect(
            lambda: self.generate_implicit_lattice(self.implicit_panel.parameters())
        )
        analyse_menu.addAction(lattice_act)

        density_act = QAction("Estimate Relative Density", self)
        density_act.triggered.connect(
            lambda: self.estimate_lattice_density(self.implicit_panel.parameters())
        )
        analyse_menu.addAction(density_act)

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

        hide_toolbar_act = QAction("Hide", self)
        hide_toolbar_act.triggered.connect(lambda: self.set_selected_visibility(False))
        toolbar.addAction(hide_toolbar_act)

        show_all_toolbar_act = QAction("Show All", self)
        show_all_toolbar_act.triggered.connect(self.show_all)
        toolbar.addAction(show_all_toolbar_act)

        linear_array_toolbar_act = QAction("Array", self)
        linear_array_toolbar_act.triggered.connect(self.linear_array_selected)
        toolbar.addAction(linear_array_toolbar_act)

        mirror_toolbar_act = QAction("Mirror X", self)
        mirror_toolbar_act.triggered.connect(lambda: self.mirror_selected("x"))
        toolbar.addAction(mirror_toolbar_act)

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

        # Dock - Generative Lab.
        # Superseded by the Implicit Lab, which has more surfaces, metric wall
        # thickness, grading, and threaded generation. Kept because it still
        # works and existing habits should not break on an upgrade.
        self.generative_dock = QDockWidget("Generative Lab (legacy)", self)
        self.generative_panel = GenerativePanel()
        self.generative_panel.setToolTip(
            "Superseded by the Implicit Lab (Ctrl+L), which offers seven surface "
            "families, wall thickness in millimetres, graded lattices, and "
            "generation on a background thread."
        )
        self.generative_panel.generate_requested.connect(self.add_tpms_lattice)
        self.generative_dock.setWidget(self.generative_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.generative_dock)
        self.tabifyDockWidget(self.prop_dock, self.generative_dock)
        self.prop_dock.raise_()

        # Dock - Measurements
        self.measure_dock = QDockWidget("Measurements", self)
        self.measurements = MeasurementsPanel()
        self.measure_dock.setWidget(self.measurements)
        self.addDockWidget(Qt.RightDockWidgetArea, self.measure_dock)
        self.tabifyDockWidget(self.prop_dock, self.measure_dock)

        # Dock - Appearance
        self.appearance_dock = QDockWidget("Appearance", self)
        self.appearance = AppearancePanel()
        self.appearance.appearance_changed.connect(self.on_appearance_changed)
        self.appearance_dock.setWidget(self.appearance)
        self.addDockWidget(Qt.RightDockWidgetArea, self.appearance_dock)
        self.tabifyDockWidget(self.prop_dock, self.appearance_dock)

        # Dock - Primitive Lab
        self.primitive_dock = QDockWidget("Primitive Lab", self)
        self.primitive_panel = PrimitivePanel()
        self.primitive_panel.primitive_requested.connect(self.add_parametric_primitive)
        self.primitive_dock.setWidget(self.primitive_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.primitive_dock)
        self.tabifyDockWidget(self.prop_dock, self.primitive_dock)

        # Dock - Grid/Snap
        self.grid_dock = QDockWidget("Grid / Snap", self)
        self.grid_panel = GridPanel()
        self.grid_panel.snap_position_requested.connect(self.snap_selected_position)
        self.grid_panel.snap_vertices_requested.connect(self.snap_selected_vertices)
        self.grid_dock.setWidget(self.grid_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.grid_dock)
        self.tabifyDockWidget(self.prop_dock, self.grid_dock)

        # Dock - Implicit Lab (signed distance fields and graded lattices)
        self.implicit_dock = QDockWidget("Implicit Lab", self)
        self.implicit_panel = ImplicitPanel()
        self.implicit_panel.generate_requested.connect(self.generate_implicit_lattice)
        self.implicit_panel.estimate_requested.connect(self.estimate_lattice_density)
        self.implicit_dock.setWidget(self.implicit_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.implicit_dock)
        self.tabifyDockWidget(self.prop_dock, self.implicit_dock)

        # Dock - Analysis
        self.analysis_dock = QDockWidget("Analysis", self)
        self.analysis_panel = AnalysisPanel()
        self.analysis_panel.analyze_requested.connect(self.run_analysis)
        self.analysis_dock.setWidget(self.analysis_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.analysis_dock)
        self.tabifyDockWidget(self.prop_dock, self.analysis_dock)

        # Shortcuts
        from PySide6.QtGui import QKeySequence, QShortcut
        self.del_shortcut = QShortcut(QKeySequence.Delete, self)
        self.del_shortcut.activated.connect(self.delete_selected)
        self.update_status()

    # ------------------------------------------------------------------
    # Background work
    # ------------------------------------------------------------------
    def _on_task_progress(self, fraction, message):
        self.statusBar().showMessage(f"{message} ({fraction * 100:.0f}%)")

    def _on_busy_changed(self, busy):
        self.implicit_panel.set_busy(busy)
        self.analysis_panel.set_busy(busy)
        if not busy:
            self.update_status()

    def _on_task_failed(self, message, traceback_text, title="Operation Failed"):
        # Show the message, keep the traceback where a bug report can find it.
        print(traceback_text)
        QMessageBox.warning(self, title, message)
        self.statusBar().showMessage(message, 8000)

    def _selection_bounds(self):
        """World-space ``(min_xyz, max_xyz)`` of the selection, or None."""
        oids = self._selected_oids()
        if not oids:
            return None
        bounds = self._combined_actor_bounds([self.scene.objects[oid] for oid in oids])
        return (
            np.array([bounds[0], bounds[2], bounds[4]], dtype=float),
            np.array([bounds[1], bounds[3], bounds[5]], dtype=float),
        )

    def generate_implicit_lattice(self, params):
        """Build a TPMS lattice on a worker thread and add it to the scene."""
        bounds = None
        if params.get("trim") == "bounds":
            bounds = self._selection_bounds()
            if bounds is None:
                QMessageBox.information(
                    self,
                    "Select an Object",
                    "Choose an object first so the lattice knows what to fill, "
                    "or switch the extent back to a free-standing cube.",
                )
                return

        label = f"{params['kind'].replace('_', ' ').title()} Lattice"
        self.statusBar().showMessage(f"Generating {label}...")
        self.tasks.submit(
            _lattice_worker,
            params,
            bounds,
            label=label,
            on_result=lambda mesh, label=label: self._on_lattice_ready(mesh, label),
            on_error=lambda message, tb: self._on_task_failed(
                message, tb, "Lattice Generation Failed"
            ),
            on_cancelled=lambda: self.statusBar().showMessage("Lattice cancelled", 4000),
        )

    def _on_lattice_ready(self, mesh, label):
        if mesh is None or mesh.is_empty:
            QMessageBox.warning(
                self,
                "Empty Lattice",
                "Those settings produced no geometry. Try a thicker wall, a "
                "smaller cell size, or a higher resolution.",
            )
            return

        oid = self._add_kernel_mesh(mesh, label, "#8bd5ca")
        self.implicit_panel.set_status(
            f"{mesh.n_faces:,} triangles, volume {mesh.volume:.1f} mm³"
        )
        self.statusBar().showMessage(
            f"Generated {self.scene.get_name(oid)} - {mesh.n_faces:,} triangles", 6000
        )

    def estimate_lattice_density(self, params):
        """Report what fraction of the volume the lattice would actually fill."""
        bounds = self._selection_bounds() if params.get("trim") == "bounds" else None
        if params.get("trim") == "bounds" and bounds is None:
            QMessageBox.information(
                self, "Select an Object", "Choose an object to measure the lattice inside."
            )
            return

        self.tasks.submit(
            _density_worker,
            params,
            bounds,
            label="Relative density",
            on_result=self._on_density_ready,
            on_error=lambda message, tb: self._on_task_failed(
                message, tb, "Density Estimate Failed"
            ),
        )

    def _on_density_ready(self, result):
        fraction = result["fraction"]
        self.implicit_panel.set_status(
            f"Relative density {fraction * 100:.1f}% - "
            f"{result['solid_volume']:.0f} mm³ of {result['sampled_volume']:.0f} mm³. "
            f"A part printed this way weighs about {fraction * 100:.0f}% of the solid."
        )
        self.statusBar().showMessage(f"Relative density {fraction * 100:.1f}%", 6000)

    def run_analysis(self, params):
        """Run the printability report on the selected object."""
        oids = self._selected_oids()
        if not oids:
            QMessageBox.information(
                self, "Select an Object", "Choose an object to analyse."
            )
            return

        oid = oids[0]
        name = self.scene.get_name(oid)
        mesh = actor_world_mesh(self.scene.objects[oid])
        self.analysis_dock.raise_()
        self.tasks.submit(
            _analysis_worker,
            mesh,
            params,
            label=f"Analysing {name}",
            on_result=lambda result, name=name: self.analysis_panel.show_report(
                name, result[0], result[1]
            ),
            on_error=lambda message, tb: self._on_task_failed(
                message, tb, "Analysis Failed"
            ),
        )

    def _add_kernel_mesh(self, mesh, name, color, metadata=None):
        """Add a numpy kernel mesh to the scene, converting at the boundary."""
        return self._add_scene_mesh(to_polydata(mesh), name, color, metadata=metadata)

    def closeEvent(self, event):
        """Stop background work before the window and its plotter go away."""
        self.tasks.shutdown()
        super().closeEvent(event)

    def undo(self):
        if not self.undo_stack:
            return
        self.redo_stack.append(self._snapshot_scene())
        snapshot = self.undo_stack.pop()
        self._restore_scene_snapshot(snapshot)

    def redo(self):
        if not self.redo_stack:
            return
        self.undo_stack.append(self._snapshot_scene())
        snapshot = self.redo_stack.pop()
        self._restore_scene_snapshot(snapshot)

    def new_project(self):
        if self.scene.objects:
            self._record_undo()
        self._restore_scene_snapshot({
            "objects": [],
            "selected_id": None,
            "id_counter": 0,
        })

    def open_project(self):
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Open OpenCad Project",
            "",
            "OpenCad Project (*.ocad);;All Files (*)",
        )
        if fname:
            self.open_project_file(fname)

    def open_project_file(self, path):
        """Load a project from a path, with no dialog.

        Split out from :meth:`open_project` so a file can be opened from the
        command line - which is what makes double-clicking an ``.ocad`` file in
        Explorer work once the installer registers the association.

        Returns True when the project loaded.
        """
        path = str(path)
        try:
            objects = ProjectIO.load(path)
        except Exception as exc:
            QMessageBox.warning(self, "Open Project Failed", str(exc))
            return False

        self._record_undo()
        max_id = self._max_object_index(objects)
        self._restore_scene_snapshot({
            "objects": objects,
            "selected_id": objects[0]["oid"] if objects else None,
            "id_counter": max_id + 1,
        })
        self.statusBar().showMessage(f"Opened {Path(path).name}", 5000)
        return True

    def open_path(self, path):
        """Open a project or import a mesh, chosen by the file's extension.

        The single door the command line and the file association both use.
        """
        path = Path(path)
        if not path.exists():
            QMessageBox.warning(
                self, "File Not Found", f"There is no file at:\n{path}"
            )
            return False

        if path.suffix.lower() == ".ocad":
            return self.open_project_file(path)

        try:
            mesh = self._read_mesh_file(str(path))
        except Exception as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            return False

        self._add_scene_mesh(mesh, path.stem or "Imported Mesh", "#89b4fa")
        self.statusBar().showMessage(f"Imported {path.name}", 5000)
        return True

    def save_project(self):
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Save OpenCad Project",
            "",
            "OpenCad Project (*.ocad)",
        )
        if not fname:
            return

        if not fname.lower().endswith(".ocad"):
            fname = f"{fname}.ocad"

        try:
            ProjectIO.save(fname, self._snapshot_scene()["objects"])
        except Exception as exc:
            QMessageBox.warning(self, "Save Project Failed", str(exc))
            return

        self.statusBar().showMessage(f"Saved {Path(fname).name}", 5000)

    def delete_selected(self):
        items = self.scene_list.selectedItems()
        if not items and self.scene_list.currentItem():
            items = [self.scene_list.currentItem()]

        if items:
            self._record_undo()
        for item in items:
            self.delete_object(item, record_history=False)

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

        menu.addSeparator()

        hide_act = QAction("Hide", self)
        hide_act.triggered.connect(lambda: self.set_selected_visibility(False))
        menu.addAction(hide_act)

        show_act = QAction("Show", self)
        show_act.triggered.connect(lambda: self.set_selected_visibility(True))
        menu.addAction(show_act)

        isolate_act = QAction("Isolate", self)
        isolate_act.triggered.connect(self.isolate_selected)
        menu.addAction(isolate_act)

        menu.addSeparator()

        if self._is_locked(self._item_oid(item)):
            lock_act = QAction("Unlock", self)
            lock_act.triggered.connect(lambda: self.lock_selected(False))
        else:
            lock_act = QAction("Lock", self)
            lock_act.triggered.connect(lambda: self.lock_selected(True))
        menu.addAction(lock_act)

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
        name = self._strip_item_markers(name)

        if not name:
            name = self.scene.get_name(oid)

        if name != self.scene.get_name(oid):
            self._record_undo()
            self.scene.rename_object(oid, name)
        desired_text = self._scene_item_text(oid)
        if item.text() != desired_text:
            self.scene_list.blockSignals(True)
            item.setText(desired_text)
            self.scene_list.blockSignals(False)

    def delete_object(self, item, record_history=True):
        oid = self._item_oid(item)
        if not oid:
            return
        if not self._ensure_editable(oid, "delete"):
            return

        if record_history:
            self._record_undo()
        actor = self.scene.remove_object(oid)
        if actor:
            self.viewport.plotter.remove_actor(actor)
            self.scene_list.takeItem(self.scene_list.row(item))
            self.properties.clear_selection()
            self.appearance.clear_selection()
            self.transform_mgr.disable_widget()
            self._refresh_measurements()
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
            self.appearance.set_actor(oid, actor)
            self.transform_mgr.update_widget()
            self._refresh_measurements()

    def on_property_changed(self, oid, changes):
        if oid in self.scene.objects:
            if not self._ensure_editable(oid, "edit"):
                return
            self._record_undo()
            actor = self.scene.objects[oid]
            if "position" in changes:
                actor.position = changes["position"]
            if "scale" in changes:
                actor.scale = changes["scale"]
            if "orientation" in changes:
                actor.orientation = changes["orientation"]
            if "color" in changes:
                actor.prop.color = changes["color"]

            self._refresh_measurements()
            self.viewport.plotter.render()

    def on_appearance_changed(self, changes):
        oid = self.scene.selected_id
        if not oid or oid not in self.scene.objects:
            return
        if not self._ensure_editable(oid, "edit appearance"):
            return

        self._record_undo()
        actor = self.scene.objects[oid]
        if "color" in changes:
            actor.prop.color = changes["color"]
        if "opacity" in changes:
            actor.prop.opacity = changes["opacity"]
        if "style" in changes:
            actor.prop.style = changes["style"]
        if "show_edges" in changes:
            actor.prop.show_edges = bool(changes["show_edges"])

        self.properties.set_object(oid, actor)
        self._refresh_measurements()
        self.viewport.plotter.render()

    def on_transform_update(self, oid):
        if oid in self.scene.objects:
            actor = self.scene.objects[oid]
            self.properties.set_object(oid, actor)

    def set_selected_visibility(self, visible):
        selected_oids = self._selected_oids()
        if not selected_oids:
            return

        self._record_undo()
        for oid in selected_oids:
            self._set_visibility(oid, visible)
        self._refresh_measurements()
        self.viewport.plotter.render()

    def show_all(self):
        if not self.scene.objects:
            return

        self._record_undo()
        for oid in self.scene.objects:
            self._set_visibility(oid, True)
        self._refresh_measurements()
        self.viewport.plotter.render()

    def isolate_selected(self):
        selected_oids = set(self._selected_oids())
        if not selected_oids:
            return

        self._record_undo()
        for oid in self.scene.objects:
            self._set_visibility(oid, oid in selected_oids)
        self._refresh_measurements()
        self.viewport.plotter.render()

    def lock_selected(self, locked):
        selected_oids = self._selected_oids()
        if not selected_oids:
            return

        self._record_undo()
        for oid in selected_oids:
            self._set_locked(oid, locked)
        self._refresh_measurements()
        self.viewport.plotter.render()

    def import_mesh(self):
        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Import Mesh",
            "",
            "Mesh Files (*.stl *.obj *.ply *.off *.3mf *.vtp *.vtk);;"
            "STL (*.stl);;OBJ (*.obj);;PLY (*.ply);;OFF (*.off);;3MF (*.3mf);;"
            "VTK (*.vtp *.vtk);;All Files (*)",
        )
        if not fname:
            return

        try:
            mesh = self._read_mesh_file(fname)
        except Exception as exc:
            QMessageBox.warning(self, "Import Failed", str(exc))
            return

        name = Path(fname).stem or "Imported Mesh"
        self._add_scene_mesh(mesh, name, "#89b4fa")
        self.statusBar().showMessage(f"Imported {Path(fname).name}", 5000)

    def _read_mesh_file(self, path):
        """Read through the kernel where possible, falling back to VTK.

        The kernel readers cover STL, OBJ, PLY, OFF, and 3MF without needing
        VTK, which is what lets the same formats work from the headless CLI.
        VTK's own formats still go through pyvista.
        """
        suffix = Path(path).suffix.lower()
        if suffix in io_mesh.SUPPORTED_READ_FORMATS:
            return to_polydata(io_mesh.read_mesh(path))
        return pv.read(path).extract_surface().triangulate().clean()

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
        self._refresh_measurements()
        self.viewport.plotter.render()

    def linear_array_selected(self):
        oid = self.scene.selected_id
        if not oid:
            return

        dialog = LinearArrayDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        params = dialog.parameters()
        spacing = params["spacing"]
        base_mesh = self._world_mesh(self.scene.objects[oid])
        base_name = self.scene.get_name(oid)

        self._record_undo()
        for index in range(1, params["count"]):
            offset = tuple(component * index for component in spacing)
            mesh = MeshOps.translate(base_mesh, offset)
            self._add_scene_mesh(
                mesh,
                f"{base_name} Array {index + 1}",
                "#a6e3a1",
                record_history=False,
            )

    def circular_array_selected(self):
        oid = self.scene.selected_id
        if not oid:
            return

        dialog = CircularArrayDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return

        params = dialog.parameters()
        count = params["count"]
        step = params["angle"] / (count if params["angle"] >= 360.0 else count - 1)
        base_mesh = self._world_mesh(self.scene.objects[oid])
        base_name = self.scene.get_name(oid)

        self._record_undo()
        for index in range(1, count):
            mesh = MeshOps.rotate_around_origin(base_mesh, step * index, params["axis"])
            self._add_scene_mesh(
                mesh,
                f"{base_name} Radial {index + 1}",
                "#f38ba8",
                record_history=False,
            )

    def mirror_selected(self, axis):
        oid = self.scene.selected_id
        if not oid:
            return

        actor = self.scene.objects[oid]
        mesh = MeshOps.mirror(self._world_mesh(actor), axis)
        self._add_scene_mesh(mesh, f"{self.scene.get_name(oid)} Mirror {axis.upper()}", "#89dceb")

    def align_selected(self, mode):
        selected_oids = self._selected_oids()
        if not selected_oids:
            return

        self._record_undo()
        for oid in selected_oids:
            actor = self.scene.objects[oid]
            mesh = self._world_mesh(actor)
            bounds = mesh.bounds

            if mode == "origin":
                delta = (
                    -((bounds[0] + bounds[1]) / 2.0),
                    -((bounds[2] + bounds[3]) / 2.0),
                    -((bounds[4] + bounds[5]) / 2.0),
                )
            elif mode == "floor":
                delta = (0.0, 0.0, -bounds[4])
            else:
                return

            self._replace_actor_with_world_mesh(oid, MeshOps.translate(mesh, delta), record_history=False)

        self._refresh_measurements()
        self.viewport.plotter.render()

    def freeze_selected_transforms(self):
        selected_oids = self._selected_oids()
        if not selected_oids:
            return

        self._record_undo()
        for oid in selected_oids:
            self._replace_actor_with_world_mesh(oid, self._world_mesh(self.scene.objects[oid]), record_history=False)

        self._refresh_measurements()
        self.viewport.plotter.render()

    def snap_selected_position(self, spacing):
        selected_oids = self._selected_oids()
        if not selected_oids:
            return

        spacing = max(float(spacing), 1e-6)
        self._record_undo()
        for oid in selected_oids:
            if not self._ensure_editable(oid, "snap"):
                continue
            actor = self.scene.objects[oid]
            actor.position = tuple(np.round(np.asarray(actor.position, dtype=float) / spacing) * spacing)
            if self.scene.selected_id == oid:
                self.properties.set_object(oid, actor)

        self._refresh_measurements()
        self.viewport.plotter.render()

    def snap_selected_vertices(self, spacing):
        selected_oids = self._selected_oids()
        if not selected_oids:
            return

        self._record_undo()
        for oid in selected_oids:
            if not self._ensure_editable(oid, "snap"):
                continue
            mesh = MeshOps.snap_points(self._world_mesh(self.scene.objects[oid]), spacing)
            self._replace_actor_with_world_mesh(oid, mesh, record_history=False)

        self._refresh_measurements()
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

    def add_parametric_primitive(self, params):
        mesh, name, color = self._create_parametric_primitive(params)
        self._add_scene_mesh(mesh, name, color, metadata={"parameters": params})

    def _create_parametric_primitive(self, params):
        kind = params["kind"]
        resolution = int(params["resolution"])
        size_x, size_y, size_z = params["size"]

        if kind == "box":
            bounds = (
                -size_x / 2.0,
                size_x / 2.0,
                -size_y / 2.0,
                size_y / 2.0,
                -size_z / 2.0,
                size_z / 2.0,
            )
            return pv.Box(bounds=bounds).triangulate(), "Param Box", "#89b4fa"

        if kind == "sphere":
            mesh = pv.Sphere(
                radius=params["radius"],
                theta_resolution=resolution,
                phi_resolution=max(8, resolution // 2),
            ).triangulate()
            return mesh, "Param Sphere", "#f5c2e7"

        if kind == "cylinder":
            mesh = pv.Cylinder(
                radius=params["radius"],
                height=params["height"],
                resolution=resolution,
            ).triangulate()
            return mesh, "Param Cylinder", "#94e2d5"

        if kind == "cone":
            mesh = pv.Cone(
                radius=params["radius"],
                height=params["height"],
                resolution=resolution,
            ).triangulate()
            return mesh, "Param Cone", "#f9e2af"

        if kind == "torus":
            mesh = pv.ParametricTorus(
                ringradius=params["major_radius"],
                crosssectionradius=params["minor_radius"],
            ).triangulate()
            return mesh, "Param Torus", "#cba6f7"

        if kind == "plane":
            mesh = pv.Plane(
                direction=(0, 0, 1),
                i_size=size_x,
                j_size=size_y,
                i_resolution=resolution,
                j_resolution=resolution,
            ).triangulate()
            return mesh, "Param Plane", "#a6adc8"

        raise ValueError(f"Unsupported primitive type: {kind}")

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
        oid = self._live_selection()
        if oid is None:
            return
        actor = self.scene.objects[oid]
        mesh = actor.mapper.dataset

        new_mesh = MeshOps.subdivide(mesh)
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_extrude(self):
        oid = self._live_selection()
        if oid is None:
            return
        actor = self.scene.objects[oid]
        mesh = actor.mapper.dataset

        new_mesh = MeshOps.extrude(mesh, vector=(0, 0, 1.0))
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_bevel(self):
        oid = self._live_selection()
        if oid is None:
            return
        actor = self.scene.objects[oid]
        mesh = actor.mapper.dataset

        new_mesh = MeshOps.bevel(mesh)
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_clean(self):
        oid = self._live_selection()
        if oid is None:
            return
        actor = self.scene.objects[oid]
        new_mesh = MeshOps.clean(actor.mapper.dataset)
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_decimate(self):
        oid = self._live_selection()
        if oid is None:
            return
        dialog = DecimateDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        actor = self.scene.objects[oid]
        try:
            new_mesh = MeshOps.decimate(actor.mapper.dataset, reduction=dialog.parameters()["reduction"])
        except Exception as exc:
            QMessageBox.warning(self, "Decimate Failed", str(exc))
            return
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_offset_surface(self):
        oid = self._live_selection()
        if oid is None:
            return
        dialog = OffsetSurfaceDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        actor = self.scene.objects[oid]
        try:
            new_mesh = MeshOps.offset_surface(actor.mapper.dataset, distance=dialog.parameters()["distance"])
        except Exception as exc:
            QMessageBox.warning(self, "Offset Failed", str(exc))
            return
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_clip(self):
        oid = self._live_selection()
        if oid is None:
            return
        dialog = ClipDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        actor = self.scene.objects[oid]
        try:
            new_mesh = MeshOps.clip(actor.mapper.dataset, **dialog.parameters())
        except Exception as exc:
            QMessageBox.warning(self, "Clip Failed", str(exc))
            return
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_largest_component(self):
        oid = self._live_selection()
        if oid is None:
            return
        actor = self.scene.objects[oid]
        try:
            new_mesh = MeshOps.largest_component(actor.mapper.dataset)
        except Exception as exc:
            QMessageBox.warning(self, "Largest Component Failed", str(exc))
            return
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_recompute_normals(self):
        oid = self._live_selection()
        if oid is None:
            return
        actor = self.scene.objects[oid]
        try:
            new_mesh = MeshOps.recompute_normals(actor.mapper.dataset)
        except Exception as exc:
            QMessageBox.warning(self, "Recompute Normals Failed", str(exc))
            return
        self._update_actor_mesh(oid, actor, new_mesh)

    def op_flip_normals(self):
        oid = self._live_selection()
        if oid is None:
            return
        actor = self.scene.objects[oid]
        try:
            new_mesh = MeshOps.flip_normals(actor.mapper.dataset)
        except Exception as exc:
            QMessageBox.warning(self, "Flip Normals Failed", str(exc))
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
            QMessageBox.information(self, "Export", "Select an object to export.")
            return

        fname, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Mesh",
            f"{self.scene.get_name(oid)}.stl",
            "STL (*.stl);;OBJ (*.obj);;PLY (*.ply);;OFF (*.off);;3MF (*.3mf);;All Files (*)",
        )
        if not fname:
            return

        fname = self._ensure_extension(fname, selected_filter, ".stl")
        try:
            io_mesh.write_mesh(fname, actor_world_mesh(self.scene.objects[oid]))
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported {Path(fname).name}", 5000)

    @staticmethod
    def _ensure_extension(path, selected_filter="", default=".stl"):
        """Append the chosen filter's extension when the user typed none.

        Without this a filename with no suffix reaches the writer, which then
        fails on a format it cannot infer - after the user thinks they saved.
        """
        if Path(path).suffix:
            return path
        suffix = default
        if "(*." in selected_filter:
            candidate = selected_filter.split("(*.", 1)[1].split(")", 1)[0]
            candidate = candidate.split()[0].strip()
            if candidate and candidate != "*":
                suffix = f".{candidate}"
        return f"{path}{suffix}"

    def export_visible_scene(self):
        meshes = [
            self._world_mesh(actor)
            for actor in self.scene.objects.values()
            if actor.GetVisibility()
        ]
        if not meshes:
            QMessageBox.information(self, "Export Scene", "There are no visible objects to export.")
            return

        fname, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Visible Scene",
            "scene.stl",
            "STL (*.stl);;OBJ (*.obj);;PLY (*.ply);;OFF (*.off);;3MF (*.3mf);;All Files (*)",
        )
        if not fname:
            return

        fname = self._ensure_extension(fname, selected_filter, ".stl")
        try:
            combined = Mesh.concatenate(
                [
                    actor_world_mesh(actor)
                    for actor in self.scene.objects.values()
                    if actor.GetVisibility()
                ]
            )
            io_mesh.write_mesh(fname, combined)
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))
            return
        self.statusBar().showMessage(f"Exported visible scene to {Path(fname).name}", 5000)

    def toggle_wireframe(self, checked):
        if self.scene.objects:
            self._record_undo()
        for actor in self.scene.objects.values():
            actor.prop.style = 'wireframe' if checked else 'surface'
        if self.scene.selected_id in self.scene.objects:
            self.appearance.set_actor(self.scene.selected_id, self.scene.objects[self.scene.selected_id])
        self.viewport.plotter.render()

    def reset_camera(self):
        self.viewport.plotter.reset_camera()

    def set_projection(self, orthographic):
        if orthographic:
            self.viewport.plotter.enable_parallel_projection()
            self.statusBar().showMessage("Orthographic projection enabled", 3000)
        else:
            self.viewport.plotter.disable_parallel_projection()
            self.statusBar().showMessage("Perspective projection enabled", 3000)
        self.viewport.plotter.render()

    def fit_selection(self):
        selected_oids = self._selected_oids()
        if not selected_oids:
            self.reset_camera()
            return

        bounds = self._combined_actor_bounds([self.scene.objects[oid] for oid in selected_oids])
        self.viewport.plotter.reset_camera(bounds=bounds)

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

    def _live_selection(self):
        """The selected object id, but only if it still exists in the scene.

        ``Scene.selected_id`` can outlive the object it names - undo, delete,
        and project loading all replace the object table - so every mesh
        operation goes through this rather than indexing the table directly and
        raising ``KeyError`` at the user.
        """
        oid = self.scene.selected_id
        return oid if oid in self.scene.objects else None

    def _selected_oids(self):
        oids = []
        for item in self.scene_list.selectedItems():
            oid = self._item_oid(item)
            if oid and oid in self.scene.objects and oid not in oids:
                oids.append(oid)

        if not oids and self.scene.selected_id:
            oids.append(self.scene.selected_id)
        return oids

    def _combined_actor_bounds(self, actors):
        bounds = [actor.GetBounds() for actor in actors]
        return (
            min(item[0] for item in bounds),
            max(item[1] for item in bounds),
            min(item[2] for item in bounds),
            max(item[3] for item in bounds),
            min(item[4] for item in bounds),
            max(item[5] for item in bounds),
        )

    def _world_mesh(self, actor):
        mesh = actor.mapper.dataset.copy()
        matrix = pv.array_from_vtkmatrix(actor.GetMatrix())
        mesh.transform(matrix, inplace=True)
        return mesh.extract_surface(algorithm="dataset_surface").triangulate().clean()

    def _replace_actor_with_world_mesh(self, oid, mesh, record_history=True):
        if not self._ensure_editable(oid, "edit"):
            return
        if record_history:
            self._record_undo()
        actor = self.scene.objects[oid]
        self.scene.apply_mesh_metadata(mesh, oid)
        actor.mapper.dataset = mesh
        actor.position = (0.0, 0.0, 0.0)
        actor.scale = (1.0, 1.0, 1.0)
        actor.orientation = (0.0, 0.0, 0.0)
        actor.user_matrix = None
        if self.scene.selected_id == oid:
            self.properties.set_object(oid, actor)
            self.transform_mgr.update_widget()

    def _record_undo(self):
        if self._restoring_history:
            return
        self.undo_stack.append(self._snapshot_scene())
        if len(self.undo_stack) > self.history_limit:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _snapshot_scene(self):
        ordered_oids = []
        for row in range(self.scene_list.count()):
            oid = self._item_oid(self.scene_list.item(row))
            if oid in self.scene.objects:
                ordered_oids.append(oid)

        for oid in self.scene.objects:
            if oid not in ordered_oids:
                ordered_oids.append(oid)

        objects = []
        for oid in ordered_oids:
            actor = self.scene.objects[oid]
            user_matrix = actor.user_matrix
            objects.append({
                "oid": oid,
                "mesh": actor.mapper.dataset.copy(),
                "metadata": dict(self.scene.metadata.get(oid, {})),
                "color": tuple(actor.prop.color),
                "style": actor.prop.style,
                "opacity": float(actor.prop.opacity),
                "show_edges": bool(actor.prop.show_edges),
                "visible": bool(actor.GetVisibility()),
                "position": tuple(actor.position),
                "scale": tuple(actor.scale),
                "orientation": tuple(actor.orientation),
                "user_matrix": None if user_matrix is None else np.array(user_matrix, dtype=float),
            })

        return {
            "objects": objects,
            "selected_id": self.scene.selected_id,
            "id_counter": self.scene._id_counter,
        }

    def _restore_scene_snapshot(self, snapshot):
        self._restoring_history = True
        try:
            self.transform_mgr.disable_widget()
            for actor in list(self.scene.objects.values()):
                self.viewport.plotter.remove_actor(actor)

            self.scene.objects.clear()
            self.scene.metadata.clear()
            self.scene.selected_id = None
            self.scene._id_counter = snapshot.get("id_counter", 0)
            self.scene_list.clear()
            self.properties.clear_selection()
            self.appearance.clear_selection()

            for item in snapshot["objects"]:
                oid = item["oid"]
                mesh = item["mesh"].copy()
                self.scene.metadata[oid] = dict(item["metadata"])
                self.scene.apply_mesh_metadata(mesh, oid)
                actor = self.viewport.add_mesh(
                    mesh,
                    color=item["color"],
                    show_edges=True,
                    pickable=True,
                )
                actor.prop.style = item["style"]
                actor.prop.opacity = item.get("opacity", 1.0)
                actor.prop.show_edges = item.get("show_edges", True)
                actor.SetVisibility(bool(item.get("visible", True)))
                actor.position = item["position"]
                actor.scale = item["scale"]
                actor.orientation = item["orientation"]
                actor.user_matrix = item["user_matrix"]
                actor.SetPickable(self._is_actor_pickable(oid, actor))
                self.scene.objects[oid] = actor
                self._add_scene_item(oid)

            selected_id = snapshot.get("selected_id")
            if selected_id in self.scene.objects:
                self.select_object(selected_id)
            else:
                self._refresh_measurements()

            self.viewport.plotter.render()
        finally:
            self._restoring_history = False

    def _update_actor_mesh(self, oid, actor, new_mesh, record_history=True):
        if not self._ensure_editable(oid, "edit"):
            return
        if record_history:
            self._record_undo()
        self.scene.apply_mesh_metadata(new_mesh, oid)
        actor.mapper.dataset = new_mesh
        self.properties.set_object(oid, actor)
        self.transform_mgr.update_widget()
        self._refresh_measurements()
        self.viewport.plotter.render()

    def _add_scene_mesh(self, mesh, name, color, metadata=None, record_history=True):
        if record_history:
            self._record_undo()
        oid, mesh = self.scene.add_object(mesh, name, metadata=metadata)
        actor = self.viewport.add_mesh(mesh, color=color, show_edges=True, pickable=True)
        self.scene.apply_mesh_metadata(actor.mapper.dataset, oid)
        self.scene.objects[oid] = actor
        self._add_scene_item(oid)
        self.select_object(oid)
        self.viewport.plotter.reset_camera()
        self._refresh_measurements()
        return oid

    def _add_scene_item(self, oid):
        item = QListWidgetItem(self._scene_item_text(oid))
        item.setData(Qt.UserRole, oid)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        item.setToolTip(self._scene_item_tooltip(oid))
        self.scene_list.addItem(item)

    def _scene_item_text(self, oid):
        markers = []
        actor = self.scene.objects.get(oid)
        if actor and not actor.GetVisibility():
            markers.append("[H]")
        if self._is_locked(oid):
            markers.append("[L]")
        prefix = " ".join(markers)
        if prefix:
            prefix = f"{prefix} "
        return f"{prefix}{self.scene.get_name(oid)} ({oid})"

    def _scene_item_tooltip(self, oid):
        states = []
        actor = self.scene.objects.get(oid)
        states.append("hidden" if actor and not actor.GetVisibility() else "visible")
        states.append("locked" if self._is_locked(oid) else "editable")
        return ", ".join(states)

    def _strip_item_markers(self, text):
        text = text.strip()
        changed = True
        while changed:
            changed = False
            for marker in ("[H]", "[L]"):
                if text.startswith(marker):
                    text = text[len(marker):].strip()
                    changed = True
        return text

    def _item_oid(self, item):
        oid = item.data(Qt.UserRole)
        if oid:
            return oid

        text = item.text()
        if "(" in text and ")" in text:
            return text.split("(")[-1].strip(")")
        return None

    def _update_scene_item(self, oid):
        for row in range(self.scene_list.count()):
            item = self.scene_list.item(row)
            if self._item_oid(item) == oid:
                self.scene_list.blockSignals(True)
                item.setText(self._scene_item_text(oid))
                item.setToolTip(self._scene_item_tooltip(oid))
                self.scene_list.blockSignals(False)
                return

    def _set_visibility(self, oid, visible):
        actor = self.scene.objects.get(oid)
        if not actor:
            return
        actor.SetVisibility(bool(visible))
        actor.SetPickable(self._is_actor_pickable(oid, actor))
        self._update_scene_item(oid)

    def _set_locked(self, oid, locked):
        if oid not in self.scene.metadata:
            return
        self.scene.metadata[oid]["locked"] = bool(locked)
        actor = self.scene.objects.get(oid)
        if actor:
            actor.SetPickable(self._is_actor_pickable(oid, actor))
        self._update_scene_item(oid)

    def _is_locked(self, oid):
        if not oid:
            return False
        return bool(self.scene.metadata.get(oid, {}).get("locked", False))

    def _is_actor_pickable(self, oid, actor):
        return bool(actor.GetVisibility()) and not self._is_locked(oid)

    def _ensure_editable(self, oid, action):
        if not self._is_locked(oid):
            return True
        self.statusBar().showMessage(f"{self.scene.get_name(oid)} is locked; unlock it to {action}.", 5000)
        return False

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

    def _refresh_measurements(self):
        selected = self._selected_oids()
        if hasattr(self, "measurements"):
            self.measurements.set_measurements(selected, self.scene)
        if hasattr(self, "implicit_panel"):
            # The Implicit Lab can only fill a shape once one is selected.
            self.implicit_panel.set_has_selection(bool(selected))
        self.update_status()

    def update_status(self):
        total = len(self.scene.objects)
        visible = sum(1 for actor in self.scene.objects.values() if actor.GetVisibility())
        locked = sum(1 for oid in self.scene.objects if self._is_locked(oid))
        selected = self.scene.selected_id
        selected_text = self.scene.get_name(selected) if selected else "None"
        self.statusBar().showMessage(
            f"Objects: {total} | Visible: {visible} | Locked: {locked} | Selected: {selected_text}"
        )

    def _max_object_index(self, objects):
        max_id = -1
        for item in objects:
            oid = str(item.get("oid", ""))
            if oid.startswith("obj_"):
                # A hand-edited project may carry a non-numeric id; skip it
                # rather than refusing to open the file.
                with contextlib.suppress(ValueError):
                    max_id = max(max_id, int(oid.split("_", 1)[1]))
        return max_id
