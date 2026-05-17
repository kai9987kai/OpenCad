DARK_THEME = """
QMainWindow {
    background-color: #1e1e2e;
    color: #cdd6f4;
}

QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', sans-serif;
    font-size: 10pt;
}

QDockWidget {
    titlebar-close-icon: url(close.png);
    titlebar-normal-icon: url(float.png);
    border: 1px solid #313244;
}

QDockWidget::title {
    background: #181825;
    padding-left: 5px;
    padding-top: 2px;
}

QListWidget {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 4px;
    outline: none;
}

QListWidget::item {
    padding: 5px;
    border-radius: 4px;
}

QListWidget::item:selected {
    background-color: #313244;
    color: #89b4fa;
}

QListWidget::item:hover {
    background-color: #313244;
}

QToolBar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
    spacing: 5px;
    padding: 5px;
}

QToolButton {
    background-color: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 4px;
    color: #cdd6f4;
}

QToolButton:hover {
    background-color: #313244;
    border: 1px solid #45475a;
}

QToolButton:checked {
    background-color: #45475a;
    border: 1px solid #585b70;
}

QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 5px 10px;
    color: #cdd6f4;
}

QPushButton:hover {
    background-color: #45475a;
}

QPushButton:pressed {
    background-color: #585b70;
}

QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 4px;
    padding: 2px;
    color: #cdd6f4;
}

QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
QSpinBox::up-button, QSpinBox::down-button {
    background-color: #313244;
    border: none;
    border-radius: 2px;
    margin: 1px;
}

QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #45475a;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QLabel {
    color: #a6adc8;
}

QMenu {
    background-color: #1e1e2e;
    border: 1px solid #313244;
}

QMenu::item {
    padding: 5px 20px;
}

QMenu::item:selected {
    background-color: #313244;
}
"""
