import sys
import platform
import os
from PyQt6.QtCore import QObject, pyqtSignal, QProcess, QIODeviceBase

# Use relative import for utils
from ..utils import get_resource_path

class EnvironmentSetupWorker(QObject):
    """Worker object to handle Stage 1: Environment setup (Python unpack, pip install)."""

    # Signals
    log_update = pyqtSignal(str)          # Emits log messages
    status_update = pyqtSignal(str)       # Emits high-level status updates
    # Emits success (bool), python_exe_path (str | None)
    finished = pyqtSignal(bool, object)   # Use object for None compatibility

    def __init__(self, install_path: str, parent=None):
        """Initialize the worker."""
        super().__init__(parent)
        self._install_path = install_path
        self._process = None
        self._python_exe_path = ""

    def run(self):
        """Execute the environment setup process."""
        self.status_update.emit("Starting environment setup script...")
        self.log_update.emit(f"Target installation path: {self._install_path}")

        self._process = QProcess()
        self._process.readyReadStandardOutput.connect(self._handle_stdout)
        self._process.readyReadStandardError.connect(self._handle_stderr)
        # Connect the errorOccurred signal for better diagnostics
        self._process.errorOccurred.connect(self._handle_process_error)
        self._process.finished.connect(self._handle_finish)

        # --- Get paths using resource helper --- 
        current_python = sys.executable
        try:
            # Pass the relative path from the project root perspective
            script_path = get_resource_path("installer_core.py")
            if not os.path.isfile(script_path):
                self.log_update.emit(f"ERROR: Cannot find installer_core.py at expected location: {script_path}")
                self.status_update.emit("Error: Core setup script not found.")
                self.finished.emit(False, None)
                return
        except Exception as e:
            self.log_update.emit(f"ERROR: Failed to resolve path for installer_core.py: {e}")
            self.status_update.emit("Error: Could not locate core setup script.")
            self.finished.emit(False, None)
            return
        # --- End path resolution ---

        command = [current_python, script_path, self._install_path]

        self.log_update.emit(f"Attempting to execute: {' '.join(command)}")
        # Ensure the working directory is set appropriately if installer_core depends on it
        # For now, assume it finds assets relative to its own location via utils.py
        # self._process.setWorkingDirectory(...)

        try:
            self._process.start(command[0], command[1:])
        except Exception as e:
            # Catch potential exceptions during QProcess.start itself
            self.log_update.emit(f"CRITICAL ERROR: Failed to start QProcess: {e}")
            self.status_update.emit("Error: Failed to launch setup script (QProcess start failed).")
            self.finished.emit(False, None)

        # Note: waitForStarted is blocking, consider async handling or rely on errorOccurred/finished
        # if not self._process.waitForStarted(5000):
        #     # Error already handled by _handle_process_error if emitted
        #     # Still useful as a fallback timeout?
        #     if self._process.state() == QProcess.ProcessState.NotRunning:
        #          self.log_update.emit("Error: Process failed to start within timeout and did not report specific error.")
        #          self.status_update.emit("Error: Could not start setup script (timeout).")
        #          # Ensure finished is emitted if errorOccurred wasn't
        #          # self.finished.emit(False, None) # Might double-emit
        #     return

    def _handle_stdout(self):
        """Process standard output from the script."""
        try:
            data = self._process.readAllStandardOutput().data().decode(errors='replace').strip()
            if not data:
                return
            self.log_update.emit(data) # Send raw output to log

            # Parse specific messages
            for line in data.splitlines():
                line = line.strip()
                if line.startswith("STEP:"):
                    self.status_update.emit(line.removeprefix("STEP:").strip())
                elif line.startswith("Python executable found:"):
                    self._python_exe_path = line.removeprefix("Python executable found:").strip()
                    self.log_update.emit(f"---> Detected Python executable: {self._python_exe_path}")
                elif line.startswith("SUCCESS:"):
                    self.status_update.emit(line.removeprefix("SUCCESS:").strip())
                # Don't update status on FAILURE here, wait for finish signal
        except Exception as e:
             self.log_update.emit(f"Error processing stdout: {e}")

    def _handle_stderr(self):
        """Process standard error from the script."""
        try:
            data = self._process.readAllStandardError().data().decode(errors='replace').strip()
            if data:
                self.log_update.emit(f"SCRIPT STDERR: {data}") # Log stderr messages clearly
        except Exception as e:
             self.log_update.emit(f"Error processing stderr: {e}")

    def _handle_process_error(self, error: QProcess.ProcessError):
        """Handle errors reported by QProcess itself (e.g., failed to start)."""
        error_string = self._process.errorString()
        self.log_update.emit(f"QPROCESS ERROR ({error}): {error_string}")
        self.status_update.emit(f"Error launching setup script: {error_string}")
        # Don't emit finished here, let _handle_finish do it based on final state
        # self.finished.emit(False, None)

    def _handle_finish(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """Handle the process finishing."""
        # Read any remaining output/error
        self._handle_stdout() 
        self._handle_stderr()

        self.log_update.emit(f"----> installer_core.py finished. Exit Code: {exit_code}, Exit Status: {exit_status}")

        if exit_status == QProcess.ExitStatus.CrashExit:
            self.log_update.emit("----> ERROR: Setup script crashed.")
            self.status_update.emit("Error: Setup script crashed unexpectedly.")
            self.finished.emit(False, None)
        elif exit_code == 0 and self._python_exe_path:
            self.log_update.emit("----> Environment setup successful.")
            self.status_update.emit("Environment setup successful.")
            self.finished.emit(True, self._python_exe_path)
        else:
            failure_reason = []
            if exit_code != 0:
                failure_reason.append(f"non-zero exit code ({exit_code})")
            if not self._python_exe_path:
                failure_reason.append("Python executable path not found in script output")
            reason_str = " and ".join(failure_reason) if failure_reason else "unknown reason"
            
            self.log_update.emit(f"----> ERROR: Environment setup failed ({reason_str}).")
            self.status_update.emit("Error: Environment setup failed. Check logs for details.")
            self.finished.emit(False, None)

        self._process = None # Clean up process reference
