"""The Analysis dock - will this part actually print?

OpenCad could already report area and volume.  What it could not say was whether
a model was closed, how much of it overhangs, whether it fits the printer, or
where its centre of mass sits - the questions worth asking before starting an
eight-hour print.

Findings are colour-coded by severity and each one states what was measured and
against which threshold, so a warning can be argued with rather than obeyed.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.kernel.analysis import ERROR, INFO, WARNING

__all__ = ["AnalysisPanel"]

SEVERITY_COLOR = {
    ERROR: "#f38ba8",
    WARNING: "#f9e2af",
    INFO: "#89b4fa",
}

SEVERITY_ICON = {ERROR: "✖", WARNING: "⚠", INFO: "•"}


class AnalysisPanel(QWidget):
    """Shows a printability report for the current selection."""

    analyze_requested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        settings = QGroupBox("Process")
        form = QFormLayout(settings)
        layout.addWidget(settings)

        self.overhang_angle = self._spinbox(45.0, 1.0, 89.0, 1.0, decimals=0)
        self.overhang_angle.setToolTip(
            "Faces steeper than this from horizontal need support. 45 degrees is "
            "the usual FDM rule of thumb."
        )
        form.addRow("Max overhang:", self.overhang_angle)

        self.min_feature = self._spinbox(0.8, 0.05, 20.0, 0.05)
        self.min_feature.setToolTip("Smallest printable wall - about two nozzle widths.")
        form.addRow("Min feature:", self.min_feature)

        self.build_x = self._spinbox(220.0, 10.0, 2000.0, 10.0, decimals=0)
        self.build_y = self._spinbox(220.0, 10.0, 2000.0, 10.0, decimals=0)
        self.build_z = self._spinbox(250.0, 10.0, 2000.0, 10.0, decimals=0)
        form.addRow("Build X:", self.build_x)
        form.addRow("Build Y:", self.build_y)
        form.addRow("Build Z:", self.build_z)

        self.analyze_btn = QPushButton("Analyse Selection")
        self.analyze_btn.clicked.connect(
            lambda: self.analyze_requested.emit(self.parameters())
        )
        layout.addWidget(self.analyze_btn)

        self.summary = QLabel("Select an object and run the analysis.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Check", "Result"])
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.tree.header().setStretchLastSection(True)
        layout.addWidget(self.tree, 1)

    # ------------------------------------------------------------------
    def parameters(self):
        return {
            "max_overhang_angle": self.overhang_angle.value(),
            "min_feature_size": self.min_feature.value(),
            "build_volume": (
                self.build_x.value(),
                self.build_y.value(),
                self.build_z.value(),
            ),
        }

    def set_busy(self, busy):
        self.analyze_btn.setEnabled(not busy)
        self.analyze_btn.setText("Analysing..." if busy else "Analyse Selection")

    def clear(self):
        self.tree.clear()
        self.summary.setText("Select an object and run the analysis.")

    def show_report(self, name, report, findings):
        """Populate the tree from :func:`mesh_report` and :func:`printability`."""
        self.tree.clear()

        problems = [finding for finding in findings if finding.is_problem]
        if problems:
            worst = problems[0].severity
            colour = SEVERITY_COLOR.get(worst, "#cdd6f4")
            self.summary.setText(
                f"<b>{name}</b>: <span style='color:{colour}'>"
                f"{len(problems)} issue(s) found</span>"
            )
        else:
            self.summary.setText(
                f"<b>{name}</b>: <span style='color:#a6e3a1'>no problems found</span>"
            )

        findings_root = self._section("Findings")
        for finding in findings:
            item = QTreeWidgetItem(
                [
                    f"{SEVERITY_ICON.get(finding.severity, '')} {finding.title}",
                    finding.detail,
                ]
            )
            colour = SEVERITY_COLOR.get(finding.severity)
            if colour:
                item.setForeground(0, _brush(colour))
            item.setToolTip(1, finding.detail)
            findings_root.addChild(item)

        geometry = report.get("geometry", {})
        geometry_root = self._section("Geometry")
        for label, value in [
            ("Size", geometry.get("size_text", "-")),
            ("Volume", geometry.get("volume_text", "-")),
            ("Surface area", geometry.get("area_text", "-")),
            ("Centre of mass", geometry.get("center_of_mass_text", "-")),
            ("Triangles", f"{geometry.get('triangles', 0):,}"),
            ("Vertices", f"{geometry.get('vertices', 0):,}"),
        ]:
            geometry_root.addChild(QTreeWidgetItem([label, str(value)]))

        topology = report.get("topology", {})
        topology_root = self._section("Topology")
        for label, key in [
            ("Watertight", "watertight"),
            ("Manifold edges", "edge_manifold"),
            ("Consistent winding", "oriented"),
            ("Separate bodies", "components"),
            ("Boundary edges", "boundary_edges"),
            ("Non-manifold edges", "non_manifold_edges"),
            ("Genus", "genus"),
        ]:
            value = topology.get(key)
            item = QTreeWidgetItem([label, _format(value)])
            if isinstance(value, bool):
                item.setForeground(1, _brush("#a6e3a1" if value else "#f38ba8"))
            elif key in ("boundary_edges", "non_manifold_edges") and value:
                item.setForeground(1, _brush("#f9e2af"))
            topology_root.addChild(item)

        quality = report.get("quality", {})
        quality_root = self._section("Mesh quality")
        for label, key, fmt in [
            ("Smallest angle", "min_angle_deg", "{:.2f}°"),
            ("Worst aspect ratio", "max_aspect_ratio", "{:.1f}"),
            ("Sliver triangles", "slivers", "{:d}"),
            ("Degenerate triangles", "degenerate", "{:d}"),
        ]:
            value = quality.get(key)
            try:
                text = fmt.format(value)
            except (TypeError, ValueError):
                text = _format(value)
            quality_root.addChild(QTreeWidgetItem([label, text]))

        self.tree.expandItem(findings_root)
        self.tree.expandItem(geometry_root)

    def _section(self, title):
        root = QTreeWidgetItem([title, ""])
        font = root.font(0)
        font.setBold(True)
        root.setFont(0, font)
        self.tree.addTopLevelItem(root)
        root.setExpanded(True)
        return root

    @staticmethod
    def _spinbox(value, minimum, maximum, step, decimals=2):
        spinbox = QDoubleSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setSingleStep(step)
        spinbox.setDecimals(decimals)
        spinbox.setValue(value)
        return spinbox


def _brush(colour):
    from PySide6.QtGui import QBrush, QColor

    return QBrush(QColor(colour))


def _format(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)
