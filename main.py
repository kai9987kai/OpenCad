import sys

from PySide6.QtWidgets import QApplication

from src.ui.main_window import MainWindow
from src.ui.styles import DARK_THEME


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_THEME)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
