from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QSpinBox,
    QVBoxLayout,
)


class LinearArrayDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Linear Array")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.count = QSpinBox()
        self.count.setRange(2, 50)
        self.count.setValue(5)
        form.addRow("Count:", self.count)

        self.spacing_x = self._spacing_spinbox(1.25)
        self.spacing_y = self._spacing_spinbox(0.0)
        self.spacing_z = self._spacing_spinbox(0.0)
        form.addRow("Spacing X:", self.spacing_x)
        form.addRow("Spacing Y:", self.spacing_y)
        form.addRow("Spacing Z:", self.spacing_z)

        self._add_buttons(layout)

    def parameters(self):
        return {
            "count": self.count.value(),
            "spacing": (
                self.spacing_x.value(),
                self.spacing_y.value(),
                self.spacing_z.value(),
            ),
        }

    def _spacing_spinbox(self, value):
        spinbox = QDoubleSpinBox()
        spinbox.setRange(-1000.0, 1000.0)
        spinbox.setSingleStep(0.1)
        spinbox.setDecimals(3)
        spinbox.setValue(value)
        return spinbox

    def _add_buttons(self, layout):
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class CircularArrayDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Circular Array")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.count = QSpinBox()
        self.count.setRange(3, 96)
        self.count.setValue(8)
        form.addRow("Count:", self.count)

        self.axis = QComboBox()
        self.axis.addItems(["Z", "X", "Y"])
        form.addRow("Axis:", self.axis)

        self.angle = QDoubleSpinBox()
        self.angle.setRange(1.0, 360.0)
        self.angle.setSingleStep(15.0)
        self.angle.setValue(360.0)
        form.addRow("Total Angle:", self.angle)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def parameters(self):
        return {
            "count": self.count.value(),
            "axis": self.axis.currentText().lower(),
            "angle": self.angle.value(),
        }


class DecimateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Decimate Mesh")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.reduction = QDoubleSpinBox()
        self.reduction.setRange(0.05, 0.9)
        self.reduction.setSingleStep(0.05)
        self.reduction.setDecimals(2)
        self.reduction.setValue(0.5)
        form.addRow("Reduction:", self.reduction)

        self._add_buttons(layout)

    def parameters(self):
        return {"reduction": self.reduction.value()}

    def _add_buttons(self, layout):
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class OffsetSurfaceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Offset Surface")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.distance = QDoubleSpinBox()
        self.distance.setRange(-10.0, 10.0)
        self.distance.setSingleStep(0.05)
        self.distance.setDecimals(3)
        self.distance.setValue(0.1)
        form.addRow("Distance:", self.distance)

        self._add_buttons(layout)

    def parameters(self):
        return {"distance": self.distance.value()}

    def _add_buttons(self, layout):
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class ClipDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Clip Mesh")

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.axis = QComboBox()
        self.axis.addItems(["Z", "X", "Y"])
        form.addRow("Axis:", self.axis)

        self.origin = QDoubleSpinBox()
        self.origin.setRange(-1000.0, 1000.0)
        self.origin.setSingleStep(0.1)
        self.origin.setDecimals(3)
        self.origin.setValue(0.0)
        form.addRow("Plane Origin:", self.origin)

        self.keep_positive = QCheckBox()
        self.keep_positive.setChecked(True)
        form.addRow("Keep Positive:", self.keep_positive)

        self._add_buttons(layout)

    def parameters(self):
        return {
            "axis": self.axis.currentText().lower(),
            "origin": self.origin.value(),
            "keep_positive": self.keep_positive.isChecked(),
        }

    def _add_buttons(self, layout):
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
