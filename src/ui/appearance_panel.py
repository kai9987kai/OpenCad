from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class AppearancePanel(QWidget):
    appearance_changed = Signal(object)

    PRESETS = {
        "Cyan": "#00ffff",
        "Magenta": "#ff00ff",
        "Yellow": "#ffff00",
        "Steel": "#89b4fa",
        "Mint": "#94e2d5",
        "Amber": "#f9e2af",
        "Coral": "#fab387",
        "Lavender": "#cba6f7",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_oid = None
        self._loading = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.color = QComboBox()
        self.color.addItems(list(self.PRESETS))
        self.color.currentTextChanged.connect(self._emit_color)
        form.addRow("Preset:", self.color)

        self.custom_color = QPushButton("Custom Color")
        self.custom_color.clicked.connect(self._pick_color)
        form.addRow("Color:", self.custom_color)

        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0.05, 1.0)
        self.opacity.setSingleStep(0.05)
        self.opacity.setDecimals(2)
        self.opacity.setValue(1.0)
        self.opacity.valueChanged.connect(lambda value: self._emit({"opacity": value}))
        form.addRow("Opacity:", self.opacity)

        self.style = QComboBox()
        self.style.addItems(["Surface", "Wireframe", "Points"])
        self.style.currentTextChanged.connect(lambda value: self._emit({"style": value.lower()}))
        form.addRow("Style:", self.style)

        self.show_edges = QCheckBox()
        self.show_edges.setChecked(True)
        self.show_edges.stateChanged.connect(
            lambda _state: self._emit({"show_edges": self.show_edges.isChecked()})
        )
        form.addRow("Show Edges:", self.show_edges)

        layout.addStretch()
        self.setEnabled(False)

    def set_actor(self, oid, actor):
        self._loading = True
        self.current_oid = oid
        self.opacity.setValue(float(actor.prop.opacity))
        style = str(actor.prop.style).lower()
        for index in range(self.style.count()):
            if self.style.itemText(index).lower() == style:
                self.style.setCurrentIndex(index)
                break
        self.show_edges.setChecked(bool(actor.prop.show_edges))
        self._loading = False
        self.setEnabled(True)

    def clear_selection(self):
        self.current_oid = None
        self.setEnabled(False)

    def _emit_color(self, name):
        color = self.PRESETS.get(name)
        if color:
            self._emit({"color": color})

    def _pick_color(self):
        if not self.current_oid:
            return
        color = QColorDialog.getColor()
        if color.isValid():
            self._emit({"color": color.name()})

    def _emit(self, changes):
        if self._loading or not self.current_oid:
            return
        self.appearance_changed.emit(changes)
