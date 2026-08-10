; Inno Setup script for OpenCad.
;
; Packages the one-folder PyInstaller output in dist\OpenCad into a single
; setup executable. Build the application first:
;
;   .venv\Scripts\pyinstaller --noconfirm --clean packaging\OpenCad.spec
;   iscc packaging\installer.iss
;
; Or run packaging\build.ps1, which does both.

#define AppName        "OpenCad"
#define AppVersion     "0.3.0"
#define AppPublisher   "OpenCad"
#define AppURL         "https://github.com/kai9987kai/OpenCad"
#define AppExeName     "OpenCad.exe"
#define CliExeName     "opencad-cli.exe"
#define SourceDir      "..\dist\OpenCad"

; The architecture of the build. PyInstaller produces a binary for whatever
; Python built it, so the installer must refuse machines that cannot run the
; result rather than failing confusingly after installing.
#ifndef TargetArch
  #define TargetArch "x64"
#endif

; "compatible" rather than "os" is deliberate for x64: ARM64 Windows runs x64
; binaries under emulation perfectly well, and an installer that refuses to
; install on an ARM64 machine would be wrong. An ARM64 build, by contrast,
; genuinely only runs on ARM64.
#if TargetArch == "arm64"
  #define ArchAllowed "arm64"
#else
  #define ArchAllowed "x64compatible"
#endif

[Setup]
AppId={{7B3F2A46-1C5D-4E8B-9A21-6F0C4D8E5B31}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
InfoBeforeFile=
OutputDir=..\dist\installer
OutputBaseFilename={#AppName}-{#AppVersion}-windows-{#TargetArch}-setup
SetupIconFile=..\assets\opencad.ico
UninstallDisplayIcon={app}\{#AppExeName}
UninstallDisplayName={#AppName} {#AppVersion}
WizardStyle=modern
Compression=lzma2/max
SolidCompression=yes

; Install per-user by default so no administrator prompt appears, but let
; someone who wants a machine-wide install choose it in the wizard.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline

; Refuse to install a binary the processor cannot run.
ArchitecturesAllowed={#ArchAllowed}
ArchitecturesInstallIn64BitMode={#ArchAllowed}

; VTK and Qt make this a large payload; warn early rather than half way in.
DirExistsWarning=no
ShowLanguageDialog=no
CloseApplications=yes
CloseApplicationsFilter=*.exe,*.dll

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"
Name: "associate";   Description: "Associate &.ocad project files with {#AppName}"; GroupDescription: "File associations:"
Name: "addtopath";   Description: "Add the &command line tool to PATH"; GroupDescription: "Command line:"; Flags: unchecked

[Files]
; The whole PyInstaller folder. recursesubdirs picks up the Qt plugins, the VTK
; libraries, and the bundled data files.
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}";              Filename: "{app}\{#AppExeName}"
Name: "{group}\{#AppName} on the web";   Filename: "{#AppURL}"
Name: "{group}\Uninstall {#AppName}";    Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";        Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; HKA is Inno's "auto" root: HKLM for a machine-wide install, HKCU for a
; per-user one. Writing under Software\Classes there means the association
; lands wherever this particular install has the right to put it.

; ---- .ocad association ------------------------------------------------
Root: HKA; Subkey: "Software\Classes\.ocad"; \
    ValueType: string; ValueName: ""; ValueData: "OpenCad.Project"; \
    Flags: uninsdeletevalue; Tasks: associate
Root: HKA; Subkey: "Software\Classes\OpenCad.Project"; \
    ValueType: string; ValueName: ""; ValueData: "OpenCad Project"; \
    Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\Classes\OpenCad.Project\DefaultIcon"; \
    ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExeName},0"; \
    Tasks: associate
Root: HKA; Subkey: "Software\Classes\OpenCad.Project\shell\open\command"; \
    ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" ""%1"""; \
    Tasks: associate

; ---- "Open with OpenCad" on mesh files --------------------------------
; Registered as an application that supports these types rather than taking
; over the extensions, so OpenCad appears in the Open With menu without
; stealing .stl from whatever viewer the user already prefers.
Root: HKA; Subkey: "Software\Classes\Applications\{#AppExeName}\shell\open\command"; \
    ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExeName}"" ""%1"""; \
    Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Applications\{#AppExeName}\SupportedTypes"; \
    ValueType: string; ValueName: ".stl";  ValueData: ""; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Applications\{#AppExeName}\SupportedTypes"; \
    ValueType: string; ValueName: ".obj";  ValueData: ""; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Applications\{#AppExeName}\SupportedTypes"; \
    ValueType: string; ValueName: ".ply";  ValueData: ""; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Applications\{#AppExeName}\SupportedTypes"; \
    ValueType: string; ValueName: ".3mf";  ValueData: ""; Tasks: associate
Root: HKA; Subkey: "Software\Classes\Applications\{#AppExeName}\SupportedTypes"; \
    ValueType: string; ValueName: ".off";  ValueData: ""; Tasks: associate

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; \
    Flags: nowait postinstall skipifsilent

[Code]
const
  EnvironmentKeyUser = 'Environment';
  EnvironmentKeyMachine = 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';

function EnvironmentRoot: Integer;
begin
  if IsAdminInstallMode then
    Result := HKEY_LOCAL_MACHINE
  else
    Result := HKEY_CURRENT_USER;
end;

function EnvironmentKey: String;
begin
  if IsAdminInstallMode then
    Result := EnvironmentKeyMachine
  else
    Result := EnvironmentKeyUser;
end;

{ Append the install directory to PATH, taking care not to add it twice and not
  to lose whatever is already there. }
procedure AddToPath();
var
  Existing: String;
  Target: String;
begin
  Target := ExpandConstant('{app}');
  if not RegQueryStringValue(EnvironmentRoot, EnvironmentKey, 'Path', Existing) then
    Existing := '';

  if Pos(';' + Uppercase(Target) + ';', ';' + Uppercase(Existing) + ';') > 0 then
    Exit;

  if (Existing <> '') and (Existing[Length(Existing)] <> ';') then
    Existing := Existing + ';';

  RegWriteExpandStringValue(EnvironmentRoot, EnvironmentKey, 'Path', Existing + Target);
end;

procedure RemoveFromPath();
var
  Existing: String;
  Target: String;
  Position: Integer;
begin
  Target := ExpandConstant('{app}');
  if not RegQueryStringValue(EnvironmentRoot, EnvironmentKey, 'Path', Existing) then
    Exit;

  Position := Pos(';' + Uppercase(Target), ';' + Uppercase(Existing));
  if Position = 0 then
    Exit;

  Delete(Existing, Position, Length(Target) + 1);
  RegWriteExpandStringValue(EnvironmentRoot, EnvironmentKey, 'Path', Existing);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
    AddToPath();
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath();
end;
