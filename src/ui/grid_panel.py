from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class GridPanel(QWidget):
    snap_position_requested = Signal(float)
    snap_vertices_requested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.spacing = QDoubleSpinBox()
        self.spacing.setRange(0.001, 1000.0)
        self.spacing.setSingleStep(0.05)
        self.spacing.setDecimals(3)
        self.spacing.setValue(0.25)
        form.addRow("Spacing:", self.spacing)

        self.snap_position_btn = QPushButton("Snap Position")
        self.snap_position_btn.clicked.connect(
            lambda: self.snap_position_requested.emit(self.spacing.value())
        )
        layout.addWidget(self.snap_position_btn)

        self.snap_vertices_btn = QPushButton("Snap Vertices")
        self.snap_vertices_btn.clicked.connect(
            lambda: self.snap_vertices_requested.emit(self.spacing.value())
        )
        layout.addWidget(self.snap_vertices_btn)

        layout.addStretch()
