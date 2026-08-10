# Packaging OpenCad for Windows

Produces two things:

- `dist\OpenCad\` — a self-contained application folder holding `OpenCad.exe`
  (the desktop app) and `opencad-cli.exe` (the headless CLI). It runs on a machine
  with no Python installed.
- `dist\installer\OpenCad-<version>-windows-<arch>-setup.exe` — a single-file
  installer with shortcuts, `.ocad` file association, an optional PATH entry,
  and an uninstaller.

## Build it

```powershell
.\packaging\build.ps1
```

That runs the tests, regenerates the icon, builds both executables, smoke-tests
the CLI, and compiles the installer. To skip the installer step:

```powershell
.\packaging\build.ps1 -SkipInstaller
```

### Prerequisites

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[gui,accel,dev]" pyinstaller
winget install JRSoftware.InnoSetup
```

## Architecture matters

PyInstaller freezes the interpreter that runs it, so **the build is native to
the machine that produces it**. An ARM64 build does not run on x64 Windows, and
an x64 build does not run natively on ARM64 (it runs under emulation, slowly, and
only if the whole dependency chain has x64 wheels).

The installer declares its architecture through `ArchitecturesAllowed`, so
Windows refuses a mismatched install with a clear message rather than letting it
fail later in a confusing way.

To ship both, build on both — a GitHub Actions matrix with `windows-latest` and
`windows-11-arm` runners is the straightforward route.

## The pieces

| File | What it does |
| --- | --- |
| `make_icon.py` | Renders `assets/opencad.ico` from signed distance functions, at all seven sizes Windows asks for |
| `entry_gui.py` | Frozen entry point for the desktop app |
| `entry_cli.py` | Frozen entry point for the CLI |
| `OpenCad.spec` | PyInstaller recipe: what to collect, what to leave out |
| `version_info.txt` | The Windows version resource shown in file properties |
| `installer.iss` | Inno Setup script |
| `build.ps1` | Runs all of the above in order |

## Decisions worth knowing about

**One folder, not one file.** VTK ships hundreds of megabytes of DLLs and loads
several dynamically. A `--onefile` build unpacks all of that to a temp directory
on every launch, which costs seconds of startup and breaks whenever an antivirus
scanner holds a lock on the extracted copy. The installer hides the folder from
the user anyway, so there is nothing to gain.

**No UPX.** Compressing VTK's DLLs saves space but trips heuristic antivirus
detection and slows the first load.

**The windowed build has a crash handler.** `--windowed` gives the process no
stderr, so an exception during startup would otherwise show the user nothing at
all. `entry_gui.py` installs a handler that writes a log to
`%LOCALAPPDATA%\OpenCad\logs` and shows a dialog.

**Per-user install by default.** `PrivilegesRequired=lowest` means no UAC prompt
for the common case; the wizard still offers a machine-wide install for anyone
who wants one.

**Mesh formats get an "Open with" entry, not the association.** Taking over
`.stl` would be rude — plenty of people already have a viewer they prefer. Only
`.ocad`, which is our own format, is claimed outright.

## Size

Expect roughly 400–700 MB unpacked and 150–250 MB for the installer. VTK and Qt
dominate; the OpenCad code itself is under a megabyte. The spec already excludes
the largest unused pieces, `QtWebEngine` chief among them.

## Code signing

The build is unsigned, so Windows SmartScreen will warn on first run
("Windows protected your PC" → *More info* → *Run anyway*). Signing needs a
certificate from a CA. Once you have one, add to `[Setup]` in `installer.iss`:

```
SignTool=signtool sign /f "C:\path\cert.pfx" /p $p /fd sha256 /tr http://timestamp.digicert.com /td sha256 $f
```

and sign the two executables before the installer is compiled.
