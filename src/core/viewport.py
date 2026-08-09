from PySide6.QtWidgets import QVBoxLayout, QWidget
from pyvistaqt import QtInteractor


class CADViewport(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Initialize PyVista Qt Interactor
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)

        self.setup_scene()

    def setup_scene(self):
        self.plotter.set_background("#1e1e2e")
        self.plotter.show_grid()
        self.plotter.add_axes()
        self.plotter.enable_trackball_style()
        self.plotter.enable_element_picking(callback=self.on_pick, show_message=False, mode='mesh')

    def on_pick(self, mesh):
        if mesh:
            self.parent().on_object_picked(mesh)

    def add_mesh(self, mesh, **kwargs):
        return self.plotter.add_mesh(mesh, **kwargs)

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)
