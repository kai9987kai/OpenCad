<#
.SYNOPSIS
    Build the OpenCad executables and the Windows installer.

.DESCRIPTION
    Runs the whole release pipeline: icon, PyInstaller one-folder build, a smoke
    test of the result, and the Inno Setup installer.

    The build is architecture-native. PyInstaller produces a binary for the
    Python that runs it, so an ARM64 Python yields an ARM64 application which
    will not run on x64 Windows, and vice versa. Build on the architecture you
    intend to ship, or run this twice on two machines.

.PARAMETER SkipInstaller
    Build the executables but not the setup file. Useful when Inno Setup is not
    installed.

.PARAMETER SkipTests
    Skip the test suite. The default is to refuse to package code that fails.

.EXAMPLE
    .\packaging\build.ps1
    .\packaging\build.ps1 -SkipInstaller
#>
[CmdletBinding()]
param(
    [switch]$SkipInstaller,
    [switch]$SkipTests,
    [switch]$SkipClean
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Write-Step($message) {
    Write-Host ""
    Write-Host "==> $message" -ForegroundColor Cyan
}

function Fail($message) {
    Write-Host "ERROR: $message" -ForegroundColor Red
    exit 1
}

# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Fail "No virtual environment at .venv. Create one first:`n" +
         "  python -m venv .venv`n" +
         "  .venv\Scripts\python -m pip install -e `".[gui,accel,dev]`" pyinstaller"
}

$arch = & $python -c "import platform; print(platform.machine().lower())"
switch -Wildcard ($arch) {
    'arm64' { $targetArch = 'arm64' }
    'aarch64' { $targetArch = 'arm64' }
    'amd64' { $targetArch = 'x64' }
    'x86_64' { $targetArch = 'x64' }
    default { Fail "Unsupported build architecture '$arch'." }
}
Write-Host "Building OpenCad for windows-$targetArch" -ForegroundColor Green

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
if (-not $SkipTests) {
    Write-Step "Running the test suite"
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) { Fail "Tests failed; refusing to package." }
}

# ----------------------------------------------------------------------
# Icon
# ----------------------------------------------------------------------
Write-Step "Generating the application icon"
& $python (Join-Path $PSScriptRoot 'make_icon.py')
if ($LASTEXITCODE -ne 0) { Fail "Icon generation failed." }

# ----------------------------------------------------------------------
# Executables
# ----------------------------------------------------------------------
Write-Step "Building the executables with PyInstaller"
$pyinstaller = Join-Path $root '.venv\Scripts\pyinstaller.exe'
if (-not (Test-Path $pyinstaller)) {
    Fail "PyInstaller is not installed in .venv. Run:`n  .venv\Scripts\python -m pip install pyinstaller"
}

$arguments = @('--noconfirm', '--distpath', 'dist', '--workpath', 'build')
if (-not $SkipClean) { $arguments += '--clean' }
$arguments += (Join-Path $PSScriptRoot 'OpenCad.spec')

& $pyinstaller @arguments
if ($LASTEXITCODE -ne 0) { Fail "PyInstaller failed." }

$appDir = Join-Path $root 'dist\OpenCad'
$guiExe = Join-Path $appDir 'OpenCad.exe'
$cliExe = Join-Path $appDir 'opencad.exe'
foreach ($exe in @($guiExe, $cliExe)) {
    if (-not (Test-Path $exe)) { Fail "Expected $exe but it was not produced." }
}

$size = (Get-ChildItem $appDir -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("Built {0:N0} files, {1:N1} MB" -f `
    (Get-ChildItem $appDir -Recurse -File).Count, ($size / 1MB)) -ForegroundColor Green

# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------
Write-Step "Smoke testing the built CLI"
& $cliExe --version
if ($LASTEXITCODE -ne 0) { Fail "The built CLI does not run." }

$probe = Join-Path ([System.IO.Path]::GetTempPath()) 'opencad-smoke.stl'
& $cliExe primitive icosphere --radius 10 -o $probe
if ($LASTEXITCODE -ne 0) { Fail "The built CLI could not generate geometry." }
& $cliExe info $probe
if ($LASTEXITCODE -ne 0) { Fail "The built CLI could not read geometry back." }
Remove-Item $probe -ErrorAction SilentlyContinue

# ----------------------------------------------------------------------
# Installer
# ----------------------------------------------------------------------
if ($SkipInstaller) {
    Write-Step "Skipping the installer (-SkipInstaller)"
    Write-Host "Application folder: $appDir"
    exit 0
}

Write-Step "Building the installer with Inno Setup"
$iscc = $null
foreach ($candidate in @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)) {
    if (Test-Path $candidate) { $iscc = $candidate; break }
}
if (-not $iscc) {
    $command = Get-Command iscc -ErrorAction SilentlyContinue
    if ($command) { $iscc = $command.Source }
}
if (-not $iscc) {
    Fail "Inno Setup was not found. Install it with:`n" +
         "  winget install JRSoftware.InnoSetup`n" +
         "or re-run with -SkipInstaller to produce just the application folder."
}

& $iscc "/DTargetArch=$targetArch" (Join-Path $PSScriptRoot 'installer.iss')
if ($LASTEXITCODE -ne 0) { Fail "Inno Setup failed." }

Write-Step "Done"
Get-ChildItem (Join-Path $root 'dist\installer') -Filter *.exe |
    ForEach-Object { Write-Host ("  {0}  ({1:N1} MB)" -f $_.Name, ($_.Length / 1MB)) -ForegroundColor Green }
