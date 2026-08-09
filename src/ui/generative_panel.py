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


class GenerativePanel(QWidget):
    generate_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.kind = QComboBox()
        self.kind.addItems(["Gyroid", "Schwarz P", "Diamond"])
        form.addRow("Surface:", self.kind)

        self.size = self._double_spinbox(4.0, 0.5, 20.0, 0.25)
        form.addRow("Size:", self.size)

        self.cells = QSpinBox()
        self.cells.setRange(1, 10)
        self.cells.setValue(3)
        form.addRow("Cells:", self.cells)

        self.resolution = QSpinBox()
        self.resolution.setRange(16, 120)
        self.resolution.setSingleStep(4)
        self.resolution.setValue(56)
        form.addRow("Resolution:", self.resolution)

        self.level = self._double_spinbox(0.0, -1.5, 1.5, 0.05)
        form.addRow("Iso Level:", self.level)

        self.gradient = self._double_spinbox(0.0, -1.5, 1.5, 0.05)
        form.addRow("Z Gradient:", self.gradient)

        self.generate_btn = QPushButton("Generate TPMS")
        self.generate_btn.clicked.connect(self._emit_generate)
        layout.addWidget(self.generate_btn)
        layout.addStretch()

    def parameters(self):
        return {
            "kind": self.kind.currentText(),
            "size": self.size.value(),
            "cells": self.cells.value(),
            "resolution": self.resolution.value(),
            "level": self.level.value(),
            "gradient": self.gradient.value(),
        }

    def _emit_generate(self):
        self.generate_requested.emit(self.parameters())

    def _double_spinbox(self, value, minimum, maximum, step):
        spinbox = QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(3)
        spinbox.setValue(value)
        return spinbox
