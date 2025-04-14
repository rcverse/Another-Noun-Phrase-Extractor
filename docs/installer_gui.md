# ANPE Installer GUI Design Discussion Summary

## 1. Problem Definition

The main ANPE GUI application (`anpe_gui`) has significant dependencies (NLTK, SpaCy, Benepar, PyTorch) which make the final packaged executable (e.g., using PyInstaller) very large (potentially hundreds of MBs). This makes distribution difficult and the initial download cumbersome for end-users, especially those without programming knowledge.

## 2. Initial Idea (Rejected)

The initial idea was to modify the `anpe_gui` application or its `SetupWizard` to download and install these large Python dependencies *after* the main application starts. This was deemed impractical and fragile due to:
*   Difficulties installing packages into a running, frozen executable's environment.
*   Potential dependency conflicts and installation errors (`pip` complexity).
*   Poor user experience (long waits, complex errors, potential for broken state).
*   Need to bundle `pip` within the main application.

## 3. Proposed Solution: Dedicated Installer Application

The recommended approach is to create a separate installer application. This installer handles the entire setup process before the main `anpe_gui` is ever run.

**Installer Responsibilities:**
1.  **Environment Provisioning:** (Recommended) Create/unpack a dedicated, isolated Python environment on the user's machine (e.g., using a bundled embeddable Python for Windows, requires different strategy for macOS/Linux).
2.  **Package Installation:** Use the target Python environment's `pip` to install all necessary Python packages:
    *   `anpe` (from PyPI, as requested)
    *   `nltk`
    *   `spacy`
    *   `benepar`
    *   `torch` (CPU version recommended: `--index-url https://download.pytorch.org/whl/cpu`)
    *   Any other runtime dependencies.
    *   (Optionally install from bundled wheel files for speed/offline capability).
3.  **Model/Data Download:** Download the required language models/data files for SpaCy, NLTK, and Benepar (adapting logic from the existing `SetupWizard`).
4.  **Shortcut Creation:** Create desktop/start menu shortcuts that point to the installed Python executable running the `anpe_gui` application entry point (e.g., `path/to/python.exe -m anpe_gui`). Use `pyshortcuts` or platform-specific code for cross-platform compatibility.

**Main Application (`anpe_gui`):**
*   The `anpe_gui` Python code itself is *not* packaged into a standalone `.exe` by the developer beforehand.
*   It is treated as Python code to be installed into the environment by the installer via `pip install anpe`.
*   It relies on the installer having successfully set up the environment and installed all dependencies.

## 4. Installer GUI Requirements (User Request)

*   **Framework:** Use **PyQt6** for the installer's GUI to maintain visual consistency with the main `anpe_gui` application, even though this will increase the installer's own file size compared to using Tkinter.
*   **Functionality:**
    *   Provide clear visual feedback during the installation process.
    *   Display status updates (e.g., "Installing NLTK...", "Downloading SpaCy model...").
    *   Show a log of actions (capturing stdout/stderr from the underlying installation script).
    *   Include an indeterminate progress indicator.
    *   Have "Start" and "Close" buttons.
*   **Implementation:** The GUI should run a separate Python script (`installer_core.py`) in the background using `subprocess` to perform the actual installation steps and display its output.

## 5. Development and Packaging Workflow

1.  Develop `anpe` core library (already done, on PyPI).
2.  Develop `anpe_gui` application (as part of `anpe` package or separate, runnable via `python -m anpe_gui` or a console script).
3.  Develop `installer_core.py` script (handles pip installs, model downloads, shortcut creation, prints status to stdout).
4.  Develop `installer_gui.pyw` (PyQt6 GUI, runs `installer_core.py`, displays output).
5.  Package the *installer* using PyInstaller:
    *   Input: `installer_gui.pyw`.
    *   Include: PyQt6, `installer_core.py`.
    *   Bundle: Embeddable Python (Windows), strategy for macOS/Linux Python, optional wheels.
    *   Output: `Installer.exe` (or platform equivalent).
6.  Distribute the `Installer.exe`.

## 6. Cross-Platform Considerations

*   Python scripts, `pip`, and PyQt6/Tkinter are generally cross-platform.
*   **Shortcut Creation:** Requires careful handling (`pyshortcuts` or platform-specific logic).
*   **Python Distribution:** Providing a self-contained Python environment is straightforward on Windows (embeddable package) but significantly more complex on macOS/Linux if aiming for zero user prerequisites. Relying on system Python or requiring a manual Miniconda install are alternatives but compromise the "fool-proof" goal.

## Detailed Implementation Plan (Final)

This plan outlines the steps and components required to build the dedicated ANPE installer using the **Standalone Python** distribution method.

### 1. Installer Directory Structure

A new top-level directory `installer/` will be created:

```
ANPE_public/
├── anpe/
├── anpe_gui/
├── docs/
│   └── installer_gui.md
├── installer/                 <-- New Directory
│   ├── installer_gui.pyw      # PyQt6 Wizard GUI frontend
│   ├── installer_core.py      # Core installation logic script
│   ├── installer.spec         # PyInstaller spec file for building the installer
│   ├── LICENSE.installer.txt  # License file to display in wizard
│   └── assets/                # Bundled resources
│       ├── python-embed-amd64/  # (Windows) Bundled Python Embeddable Package
│       ├── python-standalone/   # (macOS/Linux) Bundled standalone Python builds
│       └── wheels/              # (Optional) Bundled .whl files
│           └── ...
└── ... # Other project files (README, LICENSE, .gitignore, etc.)
```

### 2. Core Components

*   **`installer_gui.pyw` (PyQt6 Wizard Frontend):**
    *   **Purpose:** Provides a step-by-step user interface for the installation process.
    *   **Framework:** PyQt6 `QWizard`.
    *   **Design:** 
        *   Mirror the visual style of `anpe_gui` using `anpe_gui/theme.py` constants.
        *   Implement a standard wizard flow:
            *   **Welcome Page:** Title, brief introduction to ANPE and the installer.
            *   **License Page:** Display license text (from `installer/LICENSE.installer.txt`) in a `QTextEdit`. Require acceptance via `QCheckBox` to enable "Next".
            *   **Path Page:** Allow user to select installation directory. Suggest default path:
                *   Windows: `C:\Users\<User>\AppData\Local\ANPE`
                *   macOS: `/Applications/ANPE` (Note: Installing here typically requires admin privileges, handled by the `.pkg` installer).
                *   Linux: `$HOME/.local/share/ANPE`
                Validate path writability.
            *   **Options Page:** 
                *   Windows: `QCheckBox` "Create Desktop Shortcut", `QCheckBox` "Create Start Menu Shortcut".
                *   macOS: `QCheckBox` "Create alias in Applications folder" (default checked, usually redundant if installing there), `QCheckBox` "Add alias to Dock".
                *   Linux: `QCheckBox` "Create Desktop Entry (.desktop file)", `QCheckBox` "Add to Application Menu".
            *   **Progress Page:** 
                *   This page initiates the actual installation.
                *   Make it a `CommitPage`.
                *   Upon entry (`initializePage`), it retrieves settings (path, options) from previous pages.
                *   Displays Status Label, Log Area (`QTextEdit`), and indeterminate Progress Bar (`QProgressBar`).
                *   Launches `installer_core.py` using `subprocess.Popen` (passing install path and options as arguments).
                *   Uses `QThread` to monitor `installer_core.py` output asynchronously, updating the Status Label and Log Area.
                *   The wizard's "Next" button remains disabled until `installer_core.py` signals completion (success or failure).
            *   **Finish Page:** 
                *   Displays a summary message: "Installation completed successfully." or "Installation failed. Please check the log."
                *   If successful, optionally include a `QCheckBox` "Launch ANPE now".
                *   The "Finish" button closes the wizard.
    *   **Logic:** Use wizard fields (`registerField`, `field`) to pass information (install path, options) between pages.

*   **`installer_core.py` (Core Logic Script):**
    *   **Purpose:** Performs the actual installation tasks sequentially. Contains no GUI code.
    *   **Execution:** Run via `subprocess` by `installer_gui.pyw`'s Progress Page.
    *   **Key Functionality:**
        1.  **Parse Arguments:** Receive target installation path and options (e.g., shortcut creation flags) from command-line arguments passed by `installer_gui.pyw`.
        2.  **Establish Python Environment:** Locate and unpack the appropriate bundled Python distribution (Embeddable for Windows, Standalone for macOS/Linux) from `installer/assets/` into the target installation path. Determine the path to the unpacked `python` executable.
        3.  **Install Packages:** 
            *   Use the unpacked `<python_exe>` to run `pip install ...`.
            *   Iterate through required packages: `anpe` (from PyPI/wheel), `nltk`, `spacy`, `benepar`, `PyQt6` (runtime requirement for anpe_gui), `pyshortcuts` (if used), etc. SpaCy/Benepar will pull in Torch.
            *   For each package: `print(f"STEP: Installing {package_name}...")`, run `<python_exe> -m pip install ...`, check errors, `print(f"STEP: Finished installing {package_name}.")`.
        4.  **Download Models:** **REMOVED.** This step is now handled by `anpe_gui`'s `SetupWizard` on first launch.
        5.  **Create Shortcuts/Launchers:**
            *   Check options flags received from arguments.
            *   If requested, use `pyshortcuts` or platform-specific code. `print("STEP: Creating launchers/shortcuts...")`.
            *   Target: `<python_exe> -m anpe_gui`.
            *   **Windows:** Create `.lnk` files for Desktop/Start Menu.
            *   **macOS:** Create a simple launcher `.app` bundle (e.g., using a template script or `platypus`) in `/Applications/` pointing to the target command, and potentially use AppleScript or `dockutil` to add to Dock.
            *   **Linux:** Create a `.desktop` file in appropriate locations (e.g., `~/.local/share/applications/` or `/usr/share/applications/`).
        6.  **Final Report:** `print("SUCCESS: ...")` or `print("FAILURE: ...")`.

### 3. Python Distribution Strategy (Final)

The installer will provide an isolated Python environment by bundling pre-compiled Python distributions and unpacking them into the user's chosen installation directory. This avoids interfering with the user's system Python.

*   **Windows:** Bundle the official Python **Embeddable Package** (`.zip` file) from python.org for the target architecture (e.g., amd64) in `installer/assets/python-embed-amd64/`. `installer_core.py` will unpack this.
*   **macOS/Linux:** Bundle **standalone Python builds** (e.g., from the [python-build-standalone](https://github.com/indygreg/python-build-standalone) project, `.tar.gz` files) for target architectures (e.g., x86_64, arm64) in `installer/assets/python-standalone/`. `installer_core.py` will unpack the appropriate archive.

This approach ensures a high degree of isolation and control, using standard `pip` for package installation within the dedicated environment.


### 4. Bundled Assets (`installer/assets/`)

*   Python Embeddable Package (Windows).
*   Standalone Python Builds (macOS/Linux).
*   Optional Wheels (`installer/assets/wheels/`): Including `.whl` files for `anpe` and potentially other dependencies can speed up installation and enable offline installation.

### 5. Packaging (`installer/installer.spec`)

*   Use PyInstaller to package the installer components (GUI, core script, assets).
*   **Windows:** Package the PyInstaller output as `Installer.exe`.
*   **macOS:** Package the PyInstaller output and necessary scripts into a **`.pkg` installer** using tools like `pkgbuild` and `productbuild`. The `.pkg` will handle permissions and execute the installation logic (which runs `installer_core.py`).
*   **Linux:** Package the PyInstaller output as an archive (`.tar.gz`, `.AppImage`) or native package (`.deb`, `.rpm`).

### 6. User Experience Flow (Revised for Wizard)

1.  User runs `ANPE_Installer.exe` (Windows) or `ANPE_Installer.pkg` (macOS) or equivalent (Linux).
2.  Installer wizard (PyQt6 GUI) appears (on macOS, launched by the `.pkg` process).
3.  User clicks Next -> License page, accepts license.
4.  User clicks Next -> Path page, confirms/chooses install location.
5.  User clicks Next -> Options page, confirms shortcut options.
6.  User clicks Install (on Progress Page) -> Installation begins.
7.  Progress Page shows activity, status updates, and logs.
8.  `installer_core.py` runs in background (unpacking Python, pip installs, creating shortcuts).
9.  Wizard automatically proceeds to Finish Page upon completion.
10. Finish Page shows success/failure message.
11. User clicks Finish to close.
12. User launches ANPE via the created shortcut/launcher/`.app`. `anpe_gui` starts and triggers its own `SetupWizard` for model downloads if needed. 