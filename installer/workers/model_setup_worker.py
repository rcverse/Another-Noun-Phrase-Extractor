import sys
import platform
import os
from PyQt6.QtCore import QObject, pyqtSignal, QProcess, QIODeviceBase

# Fallback for utils import (less critical here, as python_exe_path is absolute)
try:
    from ..utils import get_resource_path
except ImportError:
    pass # Not strictly needed here unless we need other utils

class ModelSetupWorker(QObject):
    """Worker object to handle Stage 2: Language model setup."""

    # Signals
    log_update = pyqtSignal(str)          # Emits log messages
    status_update = pyqtSignal(str)       # Emits high-level status updates
    finished = pyqtSignal(bool)           # Emits success (bool) on completion

    def __init__(self, python_exe_path: str, parent=None):
        """Initialize the worker.

        Args:
            python_exe_path: The path to the Python executable installed in Stage 1.
            parent: The parent QObject.
        """
        super().__init__(parent)
        self._python_exe_path = python_exe_path
        self._process = None

    def run(self):
        """Execute the model setup process."""
        if not self._python_exe_path or not os.path.isfile(self._python_exe_path):
            error_msg = f"Error: Invalid Python executable path provided for model setup: {self._python_exe_path}"
            self.log_update.emit(error_msg)
            self.status_update.emit("Error: Cannot start model setup - invalid Python path.")
            self.finished.emit(False)
            return

        self.status_update.emit("Starting language model setup...")
        self.log_update.emit(f"Using Python: {self._python_exe_path}")

        self._process = QProcess()
        self._process.readyReadStandardOutput.connect(self._handle_stdout)
        self._process.readyReadStandardError.connect(self._handle_stderr)
        self._process.errorOccurred.connect(self._handle_process_error)
        self._process.finished.connect(self._handle_finish)

        # Construct the command to run the model setup script using the Stage 1 Python
        # Assumes 'anpe' package is installed and provides 'anpe.utils.setup_models'
        module_to_run = "anpe.utils.setup_models"
        command = [self._python_exe_path, "-m", module_to_run]

        self.log_update.emit(f"Attempting to execute: {' '.join(command)}")

        try:
            self._process.start(command[0], command[1:])
        except Exception as e:
            self.log_update.emit(f"CRITICAL ERROR: Failed to start QProcess for model setup: {e}")
            self.status_update.emit("Error: Failed to launch model setup script (QProcess start failed).")
            self.finished.emit(False)

    def _handle_stdout(self):
        """Process standard output from the script."""
        try:
            data = self._process.readAllStandardOutput().data().decode(errors='replace').strip()
            if data:
                self.log_update.emit(data)
                # Improve status update: look for specific markers or just show last line
                lines = data.splitlines()
                if lines:
                    # Maybe filter out less relevant lines?
                    last_line = lines[-1].strip()
                    if last_line: # Avoid empty status updates
                         self.status_update.emit(last_line)
        except Exception as e:
             self.log_update.emit(f"Error processing model setup stdout: {e}")

    def _handle_stderr(self):
        """Process standard error from the script."""
        try:
            data = self._process.readAllStandardError().data().decode(errors='replace').strip()
            if data:
                self.log_update.emit(f"MODEL SCRIPT STDERR: {data}")
        except Exception as e:
             self.log_update.emit(f"Error processing model setup stderr: {e}")

    def _handle_process_error(self, error: QProcess.ProcessError):
        """Handle errors reported by QProcess itself."""
        error_string = self._process.errorString()
        self.log_update.emit(f"QPROCESS ERROR ({error}) running model setup: {error_string}")
        self.status_update.emit(f"Error launching model setup: {error_string}")
        # Let _handle_finish determine final outcome

    def _handle_finish(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """Handle the process finishing."""
        self._handle_stdout()
        self._handle_stderr()

        self.log_update.emit(f"----> Model setup script finished. Exit Code: {exit_code}, Exit Status: {exit_status}")

        if exit_status == QProcess.ExitStatus.CrashExit:
            self.log_update.emit("----> ERROR: Model setup script crashed.")
            self.status_update.emit("Error: Model setup script crashed unexpectedly.")
            self.finished.emit(False)
        elif exit_code == 0:
            self.log_update.emit("----> Language model setup successful.")
            self.status_update.emit("Language model setup successful.")
            self.finished.emit(True)
        else:
            self.log_update.emit(f"----> ERROR: Model setup script failed (exit code {exit_code}). Check logs.")
            self.status_update.emit("Error: Language model setup failed. Check logs for details.")
            self.finished.emit(False)

        self._process = None
