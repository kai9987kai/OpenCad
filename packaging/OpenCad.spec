# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for OpenCad: the desktop app plus the headless CLI.

Two executables share one ``dist/OpenCad`` folder:

- ``OpenCad.exe``  - windowed, no console, the desktop application.
- ``opencad.exe``  - console, the headless geometry tools.

This is a **one-folder** build, not one-file. VTK ships a few hundred megabytes
of DLLs and loads several of them dynamically; a one-file build has to unpack
all of that to a temporary directory on every launch, which costs seconds of
startup and breaks whenever an antivirus scanner holds a lock on the temp copy.
The installer hides the folder anyway, so there is nothing to gain.

Build with::

    .venv\\Scripts\\pyinstaller --noconfirm --clean packaging/OpenCad.spec
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
ICON = ROOT / "assets" / "opencad.ico"

block_cipher = None

# ----------------------------------------------------------------------
# Dependencies that need help
# ----------------------------------------------------------------------
# VTK loads its Python modules through vtkmodules.all at runtime and keeps its
# native libraries beside the package, so nothing short of collecting the whole
# thing is reliable. pyvista and pyvistaqt both resolve names dynamically too.
hidden = []
datas = []
binaries = []

for package in ("vtkmodules", "pyvista", "pyvistaqt"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hidden += package_hidden

# pyvista reaches for matplotlib's colormaps; scooby is its dependency reporter.
hidden += collect_submodules("scooby")
datas += collect_data_files("matplotlib", subdir="mpl-data")

# Our own packages are imported through strings in a few places (the primitive
# registry, the CLI dispatch table), so name them explicitly.
hidden += collect_submodules("src")

datas += [(str(ICON), "assets")]
for document in ("README.md", "LICENSE"):
    source = ROOT / document
    if source.exists():
        datas += [(str(source), ".")]

# ----------------------------------------------------------------------
# Ballast
# ----------------------------------------------------------------------
# QtWebEngine alone is roughly 150 MB and nothing in OpenCad uses a browser.
# The test and notebook stacks are development-only.
excludes = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtSerialPort",
    "PySide6.QtDesigner",
    "tkinter",
    "pytest",
    "_pytest",
    "ruff",
    "IPython",
    "jupyter",
    "notebook",
    "jedi",
    "sphinx",
    "trame",
    "trame_vtk",
    "trame_vuetify",
]

gui_analysis = Analysis(
    [str(ROOT / "packaging" / "entry_gui.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

cli_analysis = Analysis(
    [str(ROOT / "packaging" / "entry_cli.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=[],
    # The CLI needs the kernel only - no Qt, no VTK - but it shares this
    # folder, so the heavy pieces are already present and must not be
    # duplicated. MERGE below assigns each dependency to exactly one binary.
    hiddenimports=collect_submodules("src.kernel") + ["src.cli"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Share every common dependency between the two executables rather than
# shipping two copies of VTK.
MERGE((gui_analysis, "OpenCad", "OpenCad"), (cli_analysis, "opencad", "opencad"))

gui_pyz = PYZ(gui_analysis.pure, gui_analysis.zipped_data, cipher=block_cipher)
cli_pyz = PYZ(cli_analysis.pure, cli_analysis.zipped_data, cipher=block_cipher)

version_file = ROOT / "packaging" / "version_info.txt"
version_argument = str(version_file) if version_file.exists() else None

gui_exe = EXE(
    gui_pyz,
    gui_analysis.scripts,
    [],
    exclude_binaries=True,
    name="OpenCad",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX and VTK's DLLs do not get along
    console=False,  # windowed: see packaging/entry_gui.py for the crash handler
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.exists() else None,
    version=version_argument,
)

cli_exe = EXE(
    cli_pyz,
    cli_analysis.scripts,
    [],
    exclude_binaries=True,
    name="opencad",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.exists() else None,
    version=version_argument,
)

collection = COLLECT(
    gui_exe,
    gui_analysis.binaries,
    gui_analysis.zipfiles,
    gui_analysis.datas,
    cli_exe,
    cli_analysis.binaries,
    cli_analysis.zipfiles,
    cli_analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="OpenCad",
)
