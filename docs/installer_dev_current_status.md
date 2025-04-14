# ANPE Installer Development Status


## Goal

Create a user-friendly, self-contained setup application (first-run wizard) for ANPE that provisions a Python environment, installs dependencies, deploys application code, and sets up language models before the main `anpe_gui` is launched.

## Current Status

*   The core Stage 1 logic (`installer/installer_core.py`) is functionally complete and tested successfully in development mode.
*   The main GUI structure (`installer/setup_windows.pyw`) is substantially implemented, including:
    *   `SetupMainWindow` with custom title bar.
    *   `QStackedWidget` managing view transitions.
    *   Placeholders and signal connections for all required views (`Welcome`, `Progress` x2, `Completion`).
    *   Basic path validation on the Welcome screen.
    *   Implementation structure for running Stage 1 (`EnvironmentSetupWorker`) and Stage 2 (`ModelSetupWorker`) in separate threads using `QProcess`.
    *   Signal handling for progress updates and completion status between workers and the GUI.
    *   Implementation shells for final actions (shortcut creation, launch).

## Completed Tasks

1.  Resolved `pip` detection issue in `installer_core.py`.
2.  Implemented robust `pip` bootstrapping in `installer_core.py`.
3.  Implemented installation of all necessary packages in `installer_core.py`.
4.  Implemented dynamic discovery of `anpe_gui` source code in `installer_core.py`.
5.  Successfully executed the full `installer_core.py` script (Stage 1) standalone.
6.  Implemented the main window structure, view management, worker integration framework, and basic signal handling in `setup_windows.pyw`.

## Next Steps

1.  **View Implementation Review/Completion:**
    *   Verify/complete `views/welcome_view.py` (input gathering, signal emission).
    *   Verify/complete `views/progress_view.py` (display slots).
    *   Verify/complete `views/completion_view.py` (state display, signal emission).

2.  **Worker Implementation Review/Completion:**
    *   Verify/complete `workers/env_setup_worker.py` (`QProcess` launch, stdout parsing including Python path, signal emission).
    *   Verify/complete `workers/model_setup_worker.py` (`QProcess` launch, output parsing, signal emission).

3.  **Integration Testing:**
    *   Run `setup_windows.pyw` end-to-end.
    *   Verify view transitions, progress display, Python path propagation, and final status reporting.

4.  **Refine Final Actions & Entry Point:**
    *   Confirm the ANPE GUI entry point script (e.g., `app/run_anpe_gui.pyw`?) and update `_create_shortcut` and `_launch_anpe`.
    *   Test shortcut creation and application launch.

5.  **Resource Path Handling:**
    *   Ensure assets (icons, etc.) are loaded robustly for both dev and packaged modes.

6.  **Packaging (`PyInstaller`/`py2app`):**
    *   (After core functionality is tested) Create/refine build scripts, ensuring correct bundling of `anpe_gui` and assets.

## Focus for Next Session

Review and refine the implementation of the individual view classes (`WelcomeViewWidget`, `ProgressViewWidget`, `CompletionViewWidget`) and worker classes (`EnvironmentSetupWorker`, `ModelSetupWorker`) to ensure they function correctly according to the design and integrate properly with `SetupMainWindow`. 