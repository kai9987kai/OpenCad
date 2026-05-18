from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class PrimitivePanel(QWidget):
    primitive_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.kind = QComboBox()
        self.kind.addItems(["Box", "Sphere", "Cylinder", "Cone", "Torus", "Plane"])
        form.addRow("Type:", self.kind)

        self.size_x = self._double_spinbox(1.0, 0.01, 1000.0, 0.1)
        self.size_y = self._double_spinbox(1.0, 0.01, 1000.0, 0.1)
        self.size_z = self._double_spinbox(1.0, 0.01, 1000.0, 0.1)
        form.addRow("Size X:", self.size_x)
        form.addRow("Size Y:", self.size_y)
        form.addRow("Size Z:", self.size_z)

        self.radius = self._double_spinbox(0.5, 0.01, 1000.0, 0.05)
        self.height = self._double_spinbox(1.0, 0.01, 1000.0, 0.1)
        form.addRow("Radius:", self.radius)
        form.addRow("Height:", self.height)

        self.major_radius = self._double_spinbox(1.0, 0.01, 1000.0, 0.05)
        self.minor_radius = self._double_spinbox(0.25, 0.01, 1000.0, 0.05)
        form.addRow("Major Radius:", self.major_radius)
        form.addRow("Minor Radius:", self.minor_radius)

        self.resolution = QSpinBox()
        self.resolution.setRange(8, 192)
        self.resolution.setSingleStep(4)
        self.resolution.setValue(48)
        form.addRow("Resolution:", self.resolution)

        self.create_btn = QPushButton("Create Primitive")
        self.create_btn.clicked.connect(self._emit_create)
        layout.addWidget(self.create_btn)
        layout.addStretch()

    def parameters(self):
        return {
            "kind": self.kind.currentText().lower(),
            "size": (self.size_x.value(), self.size_y.value(), self.size_z.value()),
            "radius": self.radius.value(),
            "height": self.height.value(),
            "major_radius": self.major_radius.value(),
            "minor_radius": self.minor_radius.value(),
            "resolution": self.resolution.value(),
        }

    def _emit_create(self):
        self.primitive_requested.emit(self.parameters())

    def _double_spinbox(self, value, minimum, maximum, step):
        spinbox = QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(3)
        spinbox.setValue(value)
        return spinbox
