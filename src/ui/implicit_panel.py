"""The Implicit Lab - lattice generation from signed distance fields.

This replaces the old Generative Lab's three hard-coded surfaces with the full
implicit pipeline:

- seven TPMS families rather than three;
- **sheet** and **network** solids, not just an iso-contour of the raw field;
- wall thickness in millimetres instead of unitless field levels;
- **functionally graded** lattices, where thickness or cell size varies through
  space - the actual research direction in ``docs/research_roadmap.md``;
- trimming to the selected object's bounds, so a lattice fills a part rather
  than floating in a cube of its own.

The panel only *describes* what to build.  Building it happens on a worker
thread in :mod:`src.ui.tasks`, because a 160^3 grid is four million field
evaluations and would otherwise freeze the window.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.kernel.lattice import MIN_CELLS_PER_WALL, build_lattice_field, cells_per_wall
from src.kernel.sdf import TPMS_KINDS

# ``build_lattice_field`` is re-exported so the dock and the CLI share one
# definition of what a lattice specification means.
__all__ = ["ImplicitPanel", "build_lattice_field"]

GRADE_TARGETS = {
    "None (uniform)": "none",
    "Wall thickness": "thickness",
    "Cell size": "period",
}

TRIM_MODES = {
    "Free-standing cube": "cube",
    "Selected object bounds": "bounds",
}


class ImplicitPanel(QWidget):
    """Parameters for one lattice build."""

    generate_requested = Signal(object)
    estimate_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        # ---------------- Surface ----------------
        surface_box = QGroupBox("Surface")
        surface_form = QFormLayout(surface_box)
        layout.addWidget(surface_box)

        self.kind = QComboBox()
        for key, label in TPMS_KINDS.items():
            self.kind.addItem(label, key)
        surface_form.addRow("Type:", self.kind)

        self.mode = QComboBox()
        self.mode.addItem("Sheet (walls)", "sheet")
        self.mode.addItem("Network (struts)", "solid")
        self.mode.currentIndexChanged.connect(self._sync_mode)
        surface_form.addRow("Solid:", self.mode)

        self.period = self._length_spinbox(6.0, 0.5, 200.0, 0.5)
        self.period.setToolTip("Edge length of one unit cell, in millimetres.")
        surface_form.addRow("Cell size:", self.period)

        self.thickness = self._length_spinbox(0.8, 0.05, 50.0, 0.05)
        self.thickness.valueChanged.connect(self._sync_cost)
        self.thickness.setToolTip(
            "Wall thickness in millimetres. Accurate to a few percent for "
            "Gyroid, Diamond and I-WP; less so for Neovius and Lidinoid."
        )
        surface_form.addRow("Wall thickness:", self.thickness)

        self.level = self._length_spinbox(0.0, -5.0, 5.0, 0.05)
        self.level.setToolTip(
            "Shifts the surface for a network solid: positive thickens the struts."
        )
        surface_form.addRow("Level:", self.level)

        # ---------------- Grading ----------------
        grade_box = QGroupBox("Grading")
        grade_form = QFormLayout(grade_box)
        layout.addWidget(grade_box)

        self.grade_target = QComboBox()
        for label, key in GRADE_TARGETS.items():
            self.grade_target.addItem(label, key)
        self.grade_target.currentIndexChanged.connect(self._sync_grading)
        grade_form.addRow("Vary:", self.grade_target)

        self.grade_axis = QComboBox()
        self.grade_axis.addItems(["Z", "X", "Y", "Radial"])
        grade_form.addRow("Along:", self.grade_axis)

        self.grade_amount = self._length_spinbox(0.5, -10.0, 10.0, 0.05)
        self.grade_amount.setToolTip(
            "Total change from one end of the part to the other, in millimetres."
        )
        grade_form.addRow("Change:", self.grade_amount)

        # ---------------- Extent ----------------
        extent_box = QGroupBox("Extent")
        extent_form = QFormLayout(extent_box)
        layout.addWidget(extent_box)

        self.trim = QComboBox()
        for label, key in TRIM_MODES.items():
            self.trim.addItem(label, key)
        self.trim.currentIndexChanged.connect(self._sync_trim)
        extent_form.addRow("Fill:", self.trim)

        self.size = self._length_spinbox(30.0, 1.0, 1000.0, 1.0)
        self.size.valueChanged.connect(self._sync_cost)
        extent_form.addRow("Cube size:", self.size)

        self.resolution = QSpinBox()
        self.resolution.setRange(24, 320)
        self.resolution.setSingleStep(8)
        self.resolution.setValue(96)
        self.resolution.setToolTip(
            "Samples per axis. Cost grows with the cube of this number: 96 is a "
            "good preview, 192 a good export."
        )
        self.resolution.valueChanged.connect(self._sync_cost)
        extent_form.addRow("Resolution:", self.resolution)

        self.cost = QLabel()
        self.cost.setWordWrap(True)
        extent_form.addRow("Cost:", self.cost)

        # ---------------- Actions ----------------
        self.generate_btn = QPushButton("Generate Lattice")
        self.generate_btn.clicked.connect(
            lambda: self.generate_requested.emit(self.parameters())
        )
        layout.addWidget(self.generate_btn)

        self.estimate_btn = QPushButton("Estimate Relative Density")
        self.estimate_btn.setToolTip(
            "Fraction of the volume that is solid - what the lattice weighs "
            "compared with the same shape printed solid."
        )
        self.estimate_btn.clicked.connect(
            lambda: self.estimate_requested.emit(self.parameters())
        )
        layout.addWidget(self.estimate_btn)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        layout.addStretch()

        self._has_selection = False
        self._sync_mode()
        self._sync_grading()
        self._sync_trim()
        self._sync_cost()

    # ------------------------------------------------------------------
    def parameters(self):
        """Everything the worker needs, as plain data safe to cross threads."""
        return {
            "kind": self.kind.currentData(),
            "mode": self.mode.currentData(),
            "period": self.period.value(),
            "thickness": self.thickness.value(),
            "level": self.level.value(),
            "grade_target": self.grade_target.currentData(),
            "grade_axis": self.grade_axis.currentText().lower(),
            "grade_amount": self.grade_amount.value(),
            "trim": self.trim.currentData(),
            "size": self.size.value(),
            "resolution": self.resolution.value(),
        }

    def set_has_selection(self, has_selection):
        """Enable trimming only when there is something to trim to."""
        self._has_selection = bool(has_selection)
        self._sync_trim()

    def set_status(self, message):
        self.status.setText(message or "")

    def set_busy(self, busy):
        self.generate_btn.setEnabled(not busy)
        self.estimate_btn.setEnabled(not busy)
        self.generate_btn.setText("Generating..." if busy else "Generate Lattice")

    # ------------------------------------------------------------------
    def _sync_mode(self):
        is_sheet = self.mode.currentData() == "sheet"
        self.thickness.setEnabled(is_sheet)
        self.level.setEnabled(not is_sheet)
        if self.grade_target.currentData() == "thickness" and not is_sheet:
            self.grade_target.setCurrentIndex(0)

    def _sync_grading(self):
        graded = self.grade_target.currentData() != "none"
        self.grade_axis.setEnabled(graded)
        self.grade_amount.setEnabled(graded)

    def _sync_trim(self):
        mode = self.trim.currentData()
        self.size.setEnabled(mode == "cube")
        if mode == "bounds" and not self._has_selection:
            self.status.setText("Select an object first to fill its bounds.")
        else:
            self.status.setText("")

    def _sync_cost(self):
        """Report the grid pitch, and warn when it cannot resolve the wall.

        Surface Nets places one vertex per cell, so a wall thinner than roughly
        two cells cannot be represented and material is quietly lost - a 0.8 mm
        wall on a 30 mm cube at resolution 48 comes out 11% light. Saying so
        here is far better than letting someone export a lattice that weighs
        less than they think.
        """
        resolution = self.resolution.value()
        text = f"{resolution**3 / 1e6:.1f}M samples"

        if self.trim.currentData() == "cube":
            span = self.size.value()
            pitch = span * 1.06 / max(resolution - 1, 1)
            text += f", {pitch:.3f} mm/cell"
            if self.mode.currentData() == "sheet":
                per_wall = cells_per_wall(self.parameters(), resolution, span)
                text += f" - {per_wall:.1f} cells across the wall"
                if per_wall < MIN_CELLS_PER_WALL:
                    text += " (too coarse: raise the resolution)"
        self.cost.setText(text)

    @staticmethod
    def _length_spinbox(value, minimum, maximum, step):
        spinbox = QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(3)
        spinbox.setValue(value)
        return spinbox
