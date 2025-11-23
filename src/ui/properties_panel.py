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

        # Color
        self.color_btn = QPushButton("Change Color")
        self.color_btn.clicked.connect(self.pick_color)
        self.form_layout.addRow("Color:", self.color_btn)

        self.layout.addStretch()

    def _create_spinbox(self, val=0.0):
        sb = QDoubleSpinBox()
        sb.setRange(-1000.0, 1000.0)
        sb.setSingleStep(0.1)
        sb.setValue(val)
        sb.valueChanged.connect(self.on_value_changed)
        return sb

    def set_object(self, oid, actor):
        self.blockSignals(True)
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
        
        self.blockSignals(False)
        self.setEnabled(True)

    def clear_selection(self):
        self.current_oid = None
        self.setEnabled(False)

    def on_value_changed(self):
        if not self.current_oid:
            return
            
        changes = {
            "position": (self.pos_x.value(), self.pos_y.value(), self.pos_z.value()),
            "scale": (self.scale_x.value(), self.scale_y.value(), self.scale_z.value())
        }
        self.property_changed.emit(self.current_oid, changes)

    def pick_color(self):
        if not self.current_oid:
            return
        color = QColorDialog.getColor()
        if color.isValid():
            self.property_changed.emit(self.current_oid, {"color": color.name()})
