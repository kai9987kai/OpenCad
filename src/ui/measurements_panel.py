import math

from PySide6.QtWidgets import QFormLayout, QLabel, QVBoxLayout, QWidget


class MeasurementsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.selection = QLabel("-")
        self.dimensions = QLabel("-")
        self.center = QLabel("-")
        self.area = QLabel("-")
        self.volume = QLabel("-")
        self.distance = QLabel("-")

        form.addRow("Selection:", self.selection)
        form.addRow("Dimensions:", self.dimensions)
        form.addRow("Center:", self.center)
        form.addRow("Area:", self.area)
        form.addRow("Volume:", self.volume)
        form.addRow("Center Distance:", self.distance)
        layout.addStretch()

    def set_measurements(self, selected_oids, scene):
        actors = [scene.objects[oid] for oid in selected_oids if oid in scene.objects]
        if not actors:
            self.clear()
            return

        self.selection.setText(", ".join(scene.get_name(oid) for oid in selected_oids))

        bounds = self._combined_bounds(actors)
        dims = (
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4],
        )
        center = (
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        )

        self.dimensions.setText(self._fmt_tuple(dims))
        self.center.setText(self._fmt_tuple(center))
        self.area.setText(self._fmt_number(sum(self._mesh_value(actor, "area") for actor in actors)))
        self.volume.setText(self._fmt_number(sum(self._mesh_value(actor, "volume") for actor in actors)))

        if len(actors) == 2:
            c0 = self._bounds_center(actors[0].GetBounds())
            c1 = self._bounds_center(actors[1].GetBounds())
            dist = math.dist(c0, c1)
            self.distance.setText(self._fmt_number(dist))
        else:
            self.distance.setText("-")

    def clear(self):
        for label in (
            self.selection,
            self.dimensions,
            self.center,
            self.area,
            self.volume,
            self.distance,
        ):
            label.setText("-")

    def _combined_bounds(self, actors):
        bounds = [actor.GetBounds() for actor in actors]
        return (
            min(item[0] for item in bounds),
            max(item[1] for item in bounds),
            min(item[2] for item in bounds),
            max(item[3] for item in bounds),
            min(item[4] for item in bounds),
            max(item[5] for item in bounds),
        )

    def _bounds_center(self, bounds):
        return (
            (bounds[0] + bounds[1]) / 2.0,
            (bounds[2] + bounds[3]) / 2.0,
            (bounds[4] + bounds[5]) / 2.0,
        )

    def _mesh_value(self, actor, attr):
        try:
            return float(getattr(actor.mapper.dataset, attr))
        except Exception:
            return 0.0

    def _fmt_tuple(self, values):
        return ", ".join(self._fmt_number(value) for value in values)

    def _fmt_number(self, value):
        return f"{float(value):.4g}"
