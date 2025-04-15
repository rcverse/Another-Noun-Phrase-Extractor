import sys
import platform
import os
from PyQt6.QtCore import QObject, pyqtSignal, QProcess, QIODeviceBase

# Use relative import for utils
from ..utils import get_resource_path
from ..widgets.task_list_widget import TaskStatus  # Import TaskStatus

class EnvironmentSetupWorker(QObject):
    """Worker object to handle Stage 1: Environment setup (Python unpack, pip install)."""

    # Signals
    log_update = pyqtSignal(str)          # Emits log messages
    status_update = pyqtSignal(str)       # Emits high-level status updates
    # Emits success (bool), python_exe_path (str | None)
    finished = pyqtSignal(bool, object)   # Use object for None compatibility
    # New signal for task status updates
    task_status_update = pyqtSignal(str, int)  # task_id, status

    def __init__(self, install_path: str, parent=None):
        """Initialize the worker."""
        super().__init__(parent)
        self._install_path = install_path
        self._process = None
        self._python_exe_path = ""
        
        # Simplified task list with fewer, consolidated tasks
        self._tasks = {
            "validate_path": "Validate installation path",
            "setup_python": "Set up Python environment", 
            "install_packages": "Install required packages",
            "copy_files": "Copy application files"
        }
        
        # Track active task for step logging
        self._current_task = None
        # Track completed tasks
        self._completed_tasks = set()

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

        # Set all tasks to pending initially
        for task_id in self._tasks:
            self.task_status_update.emit(task_id, TaskStatus.PENDING)

        try:
            self._process.start(command[0], command[1:])
        except Exception as e:
            # Catch potential exceptions during QProcess.start itself
            self.log_update.emit(f"CRITICAL ERROR: Failed to start QProcess: {e}")
            self.status_update.emit("Error: Failed to launch setup script (QProcess start failed).")
            self.finished.emit(False, None)

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
                    step_msg = line.removeprefix("STEP:").strip()
                    self.status_update.emit(step_msg)
                    
                    # Update task status based on step message
                    self._update_task_from_step(step_msg)
                    
                elif line.startswith("Python executable found:"):
                    self._python_exe_path = line.removeprefix("Python executable found:").strip()
                    self.log_update.emit(f"---> Detected Python executable: {self._python_exe_path}")
                elif line.startswith("SUCCESS:"):
                    self.status_update.emit(line.removeprefix("SUCCESS:").strip())
                elif line.startswith("FAILURE:"):
                    # Mark current task as failed if there is one
                    if self._current_task:
                        self.task_status_update.emit(self._current_task, TaskStatus.FAILED)
                
        except Exception as e:
             self.log_update.emit(f"Error processing stdout: {e}")

    def _update_task_from_step(self, step_msg: str):
        """Update task status based on step message from installer_core.py."""
        # Simplified mapping of step messages to consolidated task IDs
        step_msg_lower = step_msg.lower()
        
        if any(keyword in step_msg_lower for keyword in ["validate", "installation path", "valid"]):
            task_id = "validate_path"
            self._set_task_status(task_id, TaskStatus.PROCESSING)
            
        elif any(keyword in step_msg_lower for keyword in 
                ["python", "unpacking", "locating", "site-packages", "bootstrap", "upgrade pip"]):
            task_id = "setup_python"
            self._set_task_status(task_id, TaskStatus.PROCESSING)
            
            # Mark validation as completed if starting Python setup
            if not self._is_task_completed("validate_path"):
                self._set_task_status("validate_path", TaskStatus.COMPLETED)
            
        elif any(keyword in step_msg_lower for keyword in ["installing package", "package:", "installing", "spacy", "nltk", "benepar", "pyqt"]):
            task_id = "install_packages"
            self._set_task_status(task_id, TaskStatus.PROCESSING)
            
            # Mark Python setup as completed if starting package installation
            if not self._is_task_completed("setup_python"):
                self._set_task_status("setup_python", TaskStatus.COMPLETED)
            
        elif any(keyword in step_msg_lower for keyword in ["copy", "copying", "application code", "documentation"]):
            task_id = "copy_files"
            self._set_task_status(task_id, TaskStatus.PROCESSING)
            
            # Mark package installation as completed if starting file copying
            if not self._is_task_completed("install_packages"):
                self._set_task_status("install_packages", TaskStatus.COMPLETED)
            
        elif "environment setup complete" in step_msg_lower:
            # Mark all tasks as completed
            for task_id in self._tasks:
                if not self._is_task_completed(task_id):
                    self._set_task_status(task_id, TaskStatus.COMPLETED)

    def _set_task_status(self, task_id, status):
        """Set task status and track completion."""
        self.task_status_update.emit(task_id, status)
        self._current_task = task_id
        
        # Track completed tasks
        if status == TaskStatus.COMPLETED:
            self._completed_tasks.add(task_id)
        elif status == TaskStatus.FAILED:
            self._completed_tasks.discard(task_id)

    def _is_task_completed(self, task_id):
        """Check if a task is marked as completed."""
        return task_id in self._completed_tasks

    def _handle_stderr(self):
        """Process standard error from the script."""
        try:
            data = self._process.readAllStandardError().data().decode(errors='replace').strip()
            if data:
                self.log_update.emit(f"SCRIPT STDERR: {data}") # Log stderr messages clearly
                
                # Check for error messages and update task status
                if "ERROR:" in data and self._current_task:
                    self.task_status_update.emit(self._current_task, TaskStatus.FAILED)
                    
        except Exception as e:
             self.log_update.emit(f"Error processing stderr: {e}")

    def _handle_process_error(self, error: QProcess.ProcessError):
        """Handle errors reported by QProcess itself (e.g., failed to start)."""
        error_string = self._process.errorString()
        self.log_update.emit(f"QPROCESS ERROR ({error}): {error_string}")
        self.status_update.emit(f"Error launching setup script: {error_string}")
        
        # Mark all tasks as failed since we couldn't even start the process
        for task_id in self._tasks:
            self.task_status_update.emit(task_id, TaskStatus.FAILED)

    def _handle_finish(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """Handle the process finishing."""
        # Read any remaining output/error
        self._handle_stdout() 
        self._handle_stderr()

        self.log_update.emit(f"----> installer_core.py finished. Exit Code: {exit_code}, Exit Status: {exit_status}")

        if exit_status == QProcess.ExitStatus.CrashExit:
            self.log_update.emit("----> ERROR: Setup script crashed.")
            self.status_update.emit("Error: Setup script crashed unexpectedly.")
            
            # Mark current task (and all remaining tasks) as failed
            if self._current_task:
                self.task_status_update.emit(self._current_task, TaskStatus.FAILED)
                
            for task_id in self._tasks:
                if task_id != self._current_task and not self._is_task_completed(task_id):
                    self.task_status_update.emit(task_id, TaskStatus.FAILED)
                    
            self.finished.emit(False, None)
        elif exit_code == 0 and self._python_exe_path:
            self.log_update.emit("----> Environment setup successful.")
            self.status_update.emit("Environment setup successful.")
            
            # Make sure all tasks are marked as completed
            for task_id in self._tasks:
                if not self._is_task_completed(task_id):
                    self.task_status_update.emit(task_id, TaskStatus.COMPLETED)
                    
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
            
            # Mark current task as failed
            if self._current_task:
                self.task_status_update.emit(self._current_task, TaskStatus.FAILED)
                
            # Mark remaining tasks as failed
            for task_id in self._tasks:
                if task_id != self._current_task and not self._is_task_completed(task_id):
                    self.task_status_update.emit(task_id, TaskStatus.FAILED)
                    
            self.finished.emit(False, None)

        self._process = None # Clean up process reference
