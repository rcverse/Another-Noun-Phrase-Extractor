# ANPE Setup Application Design (v2)

**Document Status:** Final Draft for Implementation

## 1. Goal

To create a user-friendly, modern, self-contained setup application for ANPE. This application acts as a **first-run wizard** rather than a traditional system installer. Its primary goal is to establish a fully functional, isolated ANPE environment on the user's machine by performing the following actions **before** the main `anpe_gui` application is ever launched:

1.  **Provision Python Environment:** Unpack a specific, bundled Python distribution (Embeddable for Windows, Standalone for macOS/Linux) into a user-chosen (Windows) or predefined (macOS) location.
2.  **Install Core Packages:** Use the provisioned Python's `pip` to install `anpe` itself and all its runtime dependencies (e.g., `PyQt6`, `nltk`, `spacy`, `benepar`, `torch`, `pyshortcuts`). This avoids the need to bundle these large dependencies directly into the final application executable.
3.  **Deploy Application Code:** Copy the `anpe_gui` source code (bundled within the setup application) to the target installation location.
4.  **Setup Language Models:** Use the provisioned Python environment (with packages now installed) to download and set up the required language models for `nltk`, `spacy`, and `benepar`.

The result is a ready-to-use ANPE installation accessible via standard application launch mechanisms (e.g., shortcut, `.app` bundle), running the deployed `anpe_gui` code directly using the provisioned Python interpreter.

## 2. UI Paradigm

*   **Framework:** PyQt6
*   **Style:** Modern, single-window application. Clean, minimalist aesthetic following mockups (`docs/ui_design/`). Uses theme constants from `anpe_gui` (colors, fonts where applicable) for visual consistency.
*   **Structure:**
    *   Main Container: `QMainWindow` (fixed size).
    *   View Management: `QStackedWidget` manages the different views presented during setup.
    *   Interaction: Primarily driven by button clicks ("Setup", "Complete"/"Close"). Progress is displayed automatically.
*   **Views/Screens (Indices for `QStackedWidget`):**
    *   `VIEW_WELCOME` (0): Welcome text, logo, configuration (Path/License on Win, License only on Mac).
    *   `VIEW_ENV_PROGRESS` (1): Progress display for Python unpacking and package installation.
    *   `VIEW_MODEL_PROGRESS` (2): Progress display for language model download/setup.
    *   `VIEW_COMPLETION` (3): Final summary (Success/Failure), options (Shortcut/Launch on Win, Launch only on Mac).

## 3. Platform-Specific Entry Points

Two distinct GUI scripts provide tailored experiences:

*   **`installer/setup_windows.pyw`:**
    *   **Target:** Windows.
    *   **Welcome View:** Includes installation path `QLineEdit` + "Browse" `QPushButton`, and "Agree to License" `QCheckBox`.
    *   **Completion View:** If successful, offers "Create Desktop/Start Menu Shortcut" `QCheckBox` and "Launch ANPE now" `QCheckBox`.
    *   **Packaging:** PyInstaller bundles this script, `installer_core.py`, assets (Python embeddable zip, icon, license), and necessary PyQt6 components into `ANPE_Setup.exe`.

*   **`installer/setup_macos.pyw`:**
    *   **Target:** macOS.
    *   **Welcome View:** Simpler welcome text and "Agree to License" `QCheckBox`. Path selection is omitted; the setup determines the installation path relative to the `.app` bundle itself (e.g., inside `ANPESetup.app/Contents/Resources/anpe_env`).
    *   **Completion View:** If successful, offers "Launch ANPE now" `QCheckBox`. Shortcut creation is omitted (handled by user via Dock/Aliases).
    *   **Packaging:** PyInstaller or `py2app` bundles this script, `installer_core.py`, assets (Python standalone tar.gz, icon, license), and PyQt6 into `ANPESetup.app`. Distributed via `.dmg`.

## 4. Underlying Logic & Core Components

The setup process involves the GUI application orchestrating two distinct background stages using separate worker threads (`QThread`) and process execution (`QProcess`).

**Stage 1: Environment Setup**

*   **Triggered by:** User clicking "Setup" on the Welcome view.
*   **Component:** `installer/installer_core.py`
*   **Purpose:** Create the isolated Python environment and install core packages.
*   **Execution:**
    *   The GUI's `EnvironmentSetupWorker` (running in a `QThread`) launches `installer_core.py` using the *system's* Python (`sys.executable`) via `QProcess`:
        ```bash
        <system_python_path> installer_core.py <target_install_path>
        ```
    *   `installer_core.py` performs these steps:
        1.  Parses the `<target_install_path>` argument.
        2.  Detects OS/Arch, locates the appropriate bundled Python archive in `../assets/`.
        3.  Extracts the archive (e.g., `.zip` or `.tar.gz`) to `<target_install_path>/python`.
        4.  Determines the path to the newly unpacked Python executable (e.g., `<target_install_path>/python/python.exe`). Prints this path clearly to stdout: `Python executable found: <path_to_new_python>`. **CRITICAL for Stage 2.**
        5.  Uses the *newly unpacked* Python executable to run `pip install --upgrade pip`.
        6.  Uses the *newly unpacked* Python executable to run `pip install <package>` for each required package (`anpe`, `PyQt6`, `nltk`, `spacy`, `benepar`, `torch`, `pyshortcuts`).
    *   **Communication:** `installer_core.py` prints status updates prefixed with `STEP: ` and a final `SUCCESS: ` or `FAILURE: ` message to stdout. The `EnvironmentSetupWorker` reads this output via `readyReadStandardOutput`, parses it, and emits signals (`log_update`, `status_update`) to the GUI's `ProgressViewWidget`.
*   **Output:** If successful, the `EnvironmentSetupWorker` emits a `finished(True, path_to_new_python)` signal. If failed, it emits `finished(False, None)`.

**Stage 2: Model Setup**

*   **Triggered by:** Successful completion of Stage 1 (`finished(True, ...)` signal received by GUI).
*   **Component:** Logic from `anpe/utils/setup_models.py`
*   **Purpose:** Download and install necessary language models using the environment created in Stage 1.
*   **Execution:**
    *   The GUI's `ModelSetupWorker` (running in a `QThread`) launches the model setup logic using the *Python executable identified in Stage 1* via `QProcess`:
        ```bash
        <path_to_new_python> -m anpe.utils.setup_models
        ```
        *(Note: Assumes `anpe` package installed in Stage 1 provides this module)*
    *   The `anpe.utils.setup_models` script (when run as `__main__`) performs the checks and downloads for spaCy, Benepar, and NLTK models.
    *   **Communication:** The `setup_models` script prints its progress and status messages to stdout (ideally using a consistent format or leveraging its existing logging, which the worker captures). The `ModelSetupWorker` reads this output and emits signals (`log_update`, `status_update`) to the GUI's *second* `ProgressViewWidget`.
*   **Output:** The `setup_models` script should exit with code 0 on success and non-zero on failure. The `ModelSetupWorker` detects this and emits a `finished(True)` or `finished(False)` signal.

## 5. Detailed Setup Flow (Procedural)

1.  **Launch:** User starts `ANPE_Setup.exe` (Win) or `ANPESetup.app` (Mac).
2.  **GUI Init:** `SetupMainWindow` initializes, creates view widgets (`WelcomeViewWidget`, `ProgressViewWidget` x2, `CompletionViewWidget`), adds them to `QStackedWidget`, sets window title/icon, applies styles.
3.  **Show Welcome:** `stacked_widget.setCurrentIndex(VIEW_WELCOME)`.
4.  **User Interaction (Welcome):**
    *   (Win) User potentially modifies install path via `QLineEdit` or "Browse" (`QFileDialog`).
    *   User checks "Agree to License" `QCheckBox`.
    *   User clicks "Setup" `QPushButton`.
5.  **GUI -> Start Env Setup:**
    *   `WelcomeViewWidget` emits `setup_requested(install_path, license_accepted)`.
    *   `SetupMainWindow` slot (`start_environment_setup`) receives signal.
    *   `install_path` is stored.
    *   `stacked_widget.setCurrentIndex(VIEW_ENV_PROGRESS)`.
    *   `env_progress_view.clear_log()`, `env_progress_view.update_status(...)`.
    *   `EnvironmentSetupWorker` instance created (`env_worker`).
    *   `QThread` instance created (`env_thread`).
    *   Worker moved to thread; signals (`log_update`, `status_update`, `finished`) connected to GUI slots; `thread.started` connected to `worker.run`.
    *   `env_thread.start()`.
6.  **Backend (Env Setup):**
    *   `env_worker.run()` executes.
    *   `QProcess` starts `installer_core.py`.
    *   `installer_core.py` runs (unpacking, pip installs).
    *   `installer_core.py` prints logs/status/python_path to stdout.
7.  **GUI Updates (Env Progress):**
    *   `env_worker.handle_output()` reads stdout, emits `log_update` and `status_update`.
    *   `env_progress_view` slots update `log_area` and `status_label`.
8.  **Backend (Env Finish):**
    *   `installer_core.py` exits.
    *   `QProcess` emits `finished`.
    *   `env_worker.handle_finish()` determines success, finds python path, emits `finished(success, python_exe_path)`.
9.  **GUI -> Start Model Setup / Show Completion:**
    *   `SetupMainWindow` slot (`environment_setup_finished`) receives signal.
    *   Env thread is cleaned up.
    *   **If `success is True`:**
        *   `python_exe_path` is stored.
        *   `stacked_widget.setCurrentIndex(VIEW_MODEL_PROGRESS)`.
        *   `model_progress_view.clear_log()`, `model_progress_view.update_status(...)`.
        *   `ModelSetupWorker` instance created (`model_worker`) with `python_exe_path`.
        *   `QThread` instance created (`model_thread`).
        *   Worker moved to thread; signals connected (similar to env worker).
        *   `model_thread.start()`.
    *   **If `success is False`:**
        *   `SetupMainWindow.show_completion_view(success=False)` is called.
        *   Go to step 13.
10. **Backend (Model Setup):**
    *   `model_worker.run()` executes.
    *   `QProcess` starts `<python_exe_path> -m anpe.utils.setup_models`.
    *   `setup_models` runs (checking, downloading models).
    *   `setup_models` prints logs/status to stdout.
11. **GUI Updates (Model Progress):**
    *   `model_worker.handle_output()` reads stdout, emits `log_update` and `status_update`.
    *   `model_progress_view` slots update `log_area` and `status_label`.
12. **Backend (Model Finish):**
    *   `setup_models` script exits (0 for success, non-zero for failure).
    *   `QProcess` emits `finished`.
    *   `model_worker.handle_finish()` determines success based on exit code, emits `finished(success)`.
13. **GUI -> Show Completion:**
    *   `SetupMainWindow` slot (`model_setup_finished` or direct call from step 9) receives signal.
    *   Model thread is cleaned up.
    *   `SetupMainWindow.show_completion_view(success)` is called.
    *   `stacked_widget.setCurrentIndex(VIEW_COMPLETION)`.
    *   `completion_view.set_success_state(success)` updates title and shows/hides options (shortcut checkbox visibility depends on platform).
14. **User Interaction (Completion):**
    *   User reviews status.
    *   (Win) User potentially checks/unchecks "Create Shortcut".
    *   User potentially checks "Launch ANPE now".
    *   User clicks "Complete" / "Close".
15. **GUI -> Final Actions:**
    *   `CompletionViewWidget` slot (`_handle_complete`) emits signals based on checkbox states (`shortcut_requested`, `launch_requested`) and always emits `close_requested`.
    *   `SetupMainWindow` slots receive these signals:
        *   `create_shortcut()`: (Win only) Executes `pyshortcuts` using `self.python_exe_path` to create a shortcut targeting `<self.python_exe_path> <install_path>/app/run.py` (or appropriate entry point).
        *   `launch_anpe()`: Executes `<self.python_exe_path> <install_path>/app/run.py` (or appropriate entry point) via `subprocess.Popen`.
        *   `close()`: Closes the setup application window.

## 6. Progress Display

*   **Components:** `QProgressBar`, `QLabel` (for status text), `QPushButton` ("Show/Hide Details"), `QTextEdit` (for logs).
*   **Behavior:**
    *   Progress bar is initially indeterminate (`setRange(0,0)`).
    *   Status label updated based on `STEP:` messages or other heuristics parsed from script output.
    *   Log `QTextEdit` is hidden by default.
    *   "Show/Hide Details" button toggles the visibility of the `QTextEdit`.
    *   `QTextEdit` receives all captured stdout/stderr from the relevant backend process and auto-scrolls.
*   **Error Handling:** Failures (`FAILURE:` messages or non-zero exit codes) should stop the process, potentially display an error icon/color, update the status label, and proceed directly to the failure state on the Completion view. The logs remain accessible via the toggle button. The Completion view should indicate failure clearly.

## 7. Assets (`installer/assets/`)

This directory remains crucial and must contain:
*   Bundled Python distributions (`python-embed-amd64/`, `python-standalone/`).
*   The `anpe_gui` source code folder (to be bundled by PyInstaller/py2app).
*   Application icon (`app_icon_logo.png`).
*   License file (`LICENSE.installer.txt`).
*   (Optional) Bundled wheels (`wheels/`).

## 8. Dependencies

*   **Setup Application:** PyQt6.
*   **Packaged Environment:** `anpe` (and its dependencies), `PyQt6`, `nltk`, `spacy`, `benepar`, `torch`, `pyshortcuts`.

## 9. Open Questions / TODOs (Updated)

*   Implement robust parsing of `Python executable found:` message from `installer_core.py`.
*   Implement dynamic import/alternative execution strategy if running `anpe.utils.setup_models` via `-m` proves problematic within `QProcess`.
*   Implement the expandable log view UI mechanism smoothly (potentially with animation).
*   Implement path writability check in `WelcomeViewWidget` (Windows).
*   **Implement `anpe_gui` code copying logic in `installer_core.py`.**
*   Implement `create_shortcut()` using `subprocess.run` with error handling **and correct target script path**.
*   Implement `launch_anpe()` using `subprocess.Popen` **with correct entry script path**.
*   Ensure graceful handling of premature window close (`closeEvent`).
*   Refine PyInstaller (`.spec`) / `py2app` (`setup.py`) packaging scripts for both platforms, **ensuring `anpe_gui` folder is included as data**, assets are correctly bundled, and paths (especially for macOS `.app` relative install) are handled.
*   Add mechanism to display license text (e.g., popup dialog when checkbox label is clicked).
*   Consider adding log saving functionality (e.g., button on completion page). 