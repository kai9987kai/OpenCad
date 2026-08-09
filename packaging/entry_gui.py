"""Frozen-application entry point for the OpenCad desktop app.

This is deliberately not ``main.py``. A packaged Windows app is built with no
console, so anything written to stderr - including the traceback of a crash on
startup - disappears silently and the user sees an icon that does nothing. This
entry point installs a handler that writes a log and shows a dialog instead.

It also sets the application identity, which Windows uses to group taskbar
windows and which ``QSettings`` uses to decide where preferences live.
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

APP_NAME = "OpenCad"
ORGANISATION = "OpenCad"


def _log_directory():
    """Somewhere writable: never next to the exe, which may be in Program Files."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    directory = Path(base) / APP_NAME / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _write_crash_log(text):
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        path = _log_directory() / f"crash-{stamp}.log"
        path.write_text(text, encoding="utf-8")
        return path
    except Exception:  # pragma: no cover - logging must never mask the crash
        return None


def _report(title, message, details):
    """Show the failure to the user, falling back to a console if Qt is gone."""
    path = _write_crash_log(details)
    if path:
        message = f"{message}\n\nA log was written to:\n{path}"

    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication(sys.argv)
        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle(title)
        box.setText(message)
        box.setDetailedText(details)
        box.exec()
        del app
    except Exception:
        print(f"{title}: {message}\n\n{details}", file=sys.stderr)


def _install_excepthook():
    def handler(kind, value, tb):
        details = "".join(traceback.format_exception(kind, value, tb))
        _report(
            f"{APP_NAME} - Unexpected Error",
            f"{APP_NAME} hit a problem it could not recover from:\n\n{value}",
            details,
        )
        sys.__excepthook__(kind, value, tb)

    sys.excepthook = handler


def main():
    _install_excepthook()

    # PyInstaller's --windowed build gives the process no stdout or stderr at
    # all. Anything that writes to them - a warning from VTK, a stray print -
    # would raise on a null stream, so give them somewhere harmless to go.
    for name in ("stdout", "stderr"):
        if getattr(sys, name, None) is None:
            # Deliberately not a context manager: these stay open for the whole
            # life of the process and are closed by the interpreter at exit.
            setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))  # noqa: SIM115

    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication

        from src.ui.main_window import MainWindow
        from src.ui.styles import DARK_THEME
    except Exception:
        _report(
            f"{APP_NAME} - Startup Failed",
            f"{APP_NAME} could not load its components. The installation may be "
            "incomplete or damaged; reinstalling usually fixes this.",
            traceback.format_exc(),
        )
        return 1

    # Tell Windows this is its own application rather than a generic Python
    # process, so the taskbar shows our icon and groups our windows together.
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                f"{ORGANISATION}.{APP_NAME}.Desktop"
            )
        except Exception:
            pass

    QApplication.setAttribute(Qt.AA_ShareOpenGLContexts, True)
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(ORGANISATION)
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_THEME)

    icon_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parents[1]))
    icon_file = icon_path / "assets" / "opencad.ico"
    if icon_file.exists():
        app.setWindowIcon(QIcon(str(icon_file)))

    try:
        window = MainWindow()
        window.show()
    except Exception:
        _report(
            f"{APP_NAME} - Startup Failed",
            f"{APP_NAME} could not open its main window. This is usually a "
            "graphics driver problem: the 3D viewport needs working OpenGL.",
            traceback.format_exc(),
        )
        return 1

    # Explorer passes the double-clicked file as the first argument, which is
    # how the .ocad association registered by the installer reaches us.
    for argument in app.arguments()[1:]:
        if argument.startswith("-"):
            continue
        window.open_path(argument)
        break

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
