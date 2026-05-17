from PySide6.QtWidgets import QWidget, QVBoxLayout, QFormLayout, QDoubleSpinBox, QLabel, QColorDialog, QPushButton
from PySide6.QtCore import Signal

class PropertiesPanel(QWidget):
    property_changed = Signal(str, object)  # oid, changes

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_oid = None
        self.layout = QVBoxLayout(self)
        
        self.form_layout = QFormLayout()
        self.layout.addLayout(self.form_layout)
        
        # Position Inputs
        self.pos_x = self._create_spinbox()
        self.pos_y = self._create_spinbox()
        self.pos_z = self._create_spinbox()
        
        self.form_layout.addRow("Pos X:", self.pos_x)
        self.form_layout.addRow("Pos Y:", self.pos_y)
        self.form_layout.addRow("Pos Z:", self.pos_z)
        
        # Scale Inputs
        self.scale_x = self._create_spinbox(1.0)
        self.scale_y = self._create_spinbox(1.0)
        self.scale_z = self._create_spinbox(1.0)
        
        self.form_layout.addRow("Scale X:", self.scale_x)
        self.form_layout.addRow("Scale Y:", self.scale_y)
        self.form_layout.addRow("Scale Z:", self.scale_z)

        # Rotation Inputs
        self.rot_x = self._create_spinbox(0.0, -360.0, 360.0, 5.0)
        self.rot_y = self._create_spinbox(0.0, -360.0, 360.0, 5.0)
        self.rot_z = self._create_spinbox(0.0, -360.0, 360.0, 5.0)

        self.form_layout.addRow("Rot X:", self.rot_x)
        self.form_layout.addRow("Rot Y:", self.rot_y)
        self.form_layout.addRow("Rot Z:", self.rot_z)

        # Color
        self.color_btn = QPushButton("Change Color")
        self.color_btn.clicked.connect(self.pick_color)
        self.form_layout.addRow("Color:", self.color_btn)

        self.points_lbl = QLabel("-")
        self.cells_lbl = QLabel("-")
        self.area_lbl = QLabel("-")
        self.volume_lbl = QLabel("-")
        self.form_layout.addRow("Points:", self.points_lbl)
        self.form_layout.addRow("Cells:", self.cells_lbl)
        self.form_layout.addRow("Area:", self.area_lbl)
        self.form_layout.addRow("Volume:", self.volume_lbl)

        self.layout.addStretch()
        self.setEnabled(False)

    def _create_spinbox(self, val=0.0, minimum=-1000.0, maximum=1000.0, step=0.1):
        sb = QDoubleSpinBox()
        sb.setRange(minimum, maximum)
        sb.setSingleStep(step)
        sb.setValue(val)
        sb.valueChanged.connect(self.on_value_changed)
        return sb

    def set_object(self, oid, actor):
        self._block_value_signals(True)
        self.current_oid = oid
        
        # Update UI from actor
        pos = actor.position
        self.pos_x.setValue(pos[0])
        self.pos_y.setValue(pos[1])
        self.pos_z.setValue(pos[2])
        
        scale = actor.scale
        self.scale_x.setValue(scale[0])
        self.scale_y.setValue(scale[1])
        self.scale_z.setValue(scale[2])

        rotation = actor.orientation
        self.rot_x.setValue(rotation[0])
        self.rot_y.setValue(rotation[1])
        self.rot_z.setValue(rotation[2])

        mesh = actor.mapper.dataset
        self.points_lbl.setText(str(getattr(mesh, "n_points", "-")))
        self.cells_lbl.setText(str(getattr(mesh, "n_cells", "-")))
        self.area_lbl.setText(self._format_mesh_value(mesh, "area"))
        self.volume_lbl.setText(self._format_mesh_value(mesh, "volume"))

        self._block_value_signals(False)
        self.setEnabled(True)

    def clear_selection(self):
        self.current_oid = None
        for label in (self.points_lbl, self.cells_lbl, self.area_lbl, self.volume_lbl):
            label.setText("-")
        self.setEnabled(False)

    def on_value_changed(self):
        if not self.current_oid:
            return
            
        changes = {
            "position": (self.pos_x.value(), self.pos_y.value(), self.pos_z.value()),
            "scale": (self.scale_x.value(), self.scale_y.value(), self.scale_z.value()),
            "orientation": (self.rot_x.value(), self.rot_y.value(), self.rot_z.value()),
        }
        self.property_changed.emit(self.current_oid, changes)

    def pick_color(self):
        if not self.current_oid:
            return
        color = QColorDialog.getColor()
        if color.isValid():
            self.property_changed.emit(self.current_oid, {"color": color.name()})

    def _block_value_signals(self, blocked):
        for control in (
            self.pos_x,
            self.pos_y,
            self.pos_z,
            self.scale_x,
            self.scale_y,
            self.scale_z,
            self.rot_x,
            self.rot_y,
            self.rot_z,
        ):
            control.blockSignals(blocked)

    def _format_mesh_value(self, mesh, attr):
        try:
            value = getattr(mesh, attr)
        except Exception:
            return "-"
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return "-"
