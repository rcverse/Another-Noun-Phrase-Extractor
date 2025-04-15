import sys
import platform
import os
import re
import logging
from typing import Dict, List, Optional
from PyQt6.QtCore import QObject, pyqtSignal, QProcess

try:
    from ..utils import get_resource_path
    from ..widgets.task_list_widget import TaskStatus  # Import existing TaskStatus
except ImportError:
    pass  # Not critical if we can't import this

logger = logging.getLogger(__name__)

class ModelSetupWorker(QObject):
    """Worker for handling language model setup process."""

    # Signal definitions
    log_update = pyqtSignal(str)  # For detailed logs
    status_update = pyqtSignal(str)  # For user-friendly status updates
    task_status_update = pyqtSignal(str, int)  # task_id, status
    finished = pyqtSignal(bool)  # Success/failure signal

    def __init__(self, python_exe_path: str, parent=None):
        """Initialize the worker.

        Args:
            python_exe_path: The path to the Python executable to use.
            parent: The parent QObject.
        """
        super().__init__(parent)
        self._python_exe_path = python_exe_path
        self._process = None
        self._current_task = None
        self._completed_tasks = set()

        # Define tasks to match the actual process in setup_models.py
        self._tasks = {
            "check_models": "Checking model presence",
            "install_spacy": "Installing spaCy model",
            "install_benepar": "Installing Benepar model",
            "install_nltk": "Installing NLTK components"
        }

        # Task phase mapping to detect which task is active based on log messages
        self._phase_patterns = {
            "check_models": [
                "checking spacy model", "checking benepar model", "checking nltk", 
                "check_spacy_model", "check_benepar_model", "check_nltk_models",
                "checking for presence", "starting model setup"
            ],
            "install_spacy": [
                "spacy model not found", "downloading spacy model", "install_spacy_model",
                "installing spacy", "en_core_web_md"
            ],
            "install_benepar": [
                "benepar model not found", "download benepar", "install_benepar_model",
                "benepar_en3", "berkeley neural parser", "constituency parsing"
            ],
            "install_nltk": [
                "nltk model", "punkt", "nltk resource", "tokenizers", "nltk data",
                "download/setup nltk", "download nltk"
            ]
        }

    def run(self):
        """Run the model setup process."""
        if not self._python_exe_path or not os.path.isfile(self._python_exe_path):
            error_msg = f"Error: Invalid Python executable: {self._python_exe_path}"
            self.log_update.emit(error_msg)
            self.status_update.emit("Error: Cannot start setup - invalid Python path")
            
            # Mark all tasks as failed
            for task_id in self._tasks:
                self.task_status_update.emit(task_id, TaskStatus.FAILED)
                
            self.finished.emit(False)
            return

        # Initialize tasks as pending
        for task_id in self._tasks:
            self.task_status_update.emit(task_id, TaskStatus.PENDING)

        # Log initial status
        self.log_update.emit(f"Starting model setup with Python: {self._python_exe_path}")
        self.status_update.emit("Starting language model check...")
        
        # Start with the check_models task
        self._set_task_status("check_models", TaskStatus.PROCESSING)
        
        # Set up the process
        self._setup_process()
        
        # Run the setup_models.py script
        module_to_run = "anpe.utils.setup_models"
        command = [self._python_exe_path, "-m", module_to_run]
        
        try:
            self._process.start(command[0], command[1:])
        except Exception as e:
            self.log_update.emit(f"CRITICAL ERROR: Failed to start model setup: {e}")
            self.status_update.emit("Error: Failed to launch model setup script")
            
            # Mark all tasks as failed
            for task_id in self._tasks:
                self.task_status_update.emit(task_id, TaskStatus.FAILED)
                
            self.finished.emit(False)

    def _setup_process(self):
        """Set up the QProcess and connect signals."""
        self._process = QProcess()
        self._process.readyReadStandardOutput.connect(self._handle_stdout)
        self._process.readyReadStandardError.connect(self._handle_stderr)
        self._process.errorOccurred.connect(self._handle_process_error)
        self._process.finished.connect(self._handle_finish)

    def _handle_stdout(self):
        """Process standard output from the script."""
        try:
            data = self._process.readAllStandardOutput().data().decode(errors='replace').strip()
            if not data:
                return
                
            # Log the raw output
            self.log_update.emit(data)
            
            # Process line by line
            for line in data.splitlines():
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Check which task this line corresponds to
                self._update_task_status_from_line(line)
                
                # Send a clean status message
                clean_status = self._clean_status_message(line)
                if clean_status:
                    self.status_update.emit(clean_status)
            
        except Exception as e:
            self.log_update.emit(f"Error processing stdout: {e}")

    def _update_task_status_from_line(self, line: str):
        """Update the active task based on log line content."""
        line_lower = line.strip().lower()
        
        # Handle model check results - non-linear task processing
        # Set corresponding task to PROCESSING if checking that specific model
        if "checking spacy model" in line_lower:
            self._set_task_status("install_spacy", TaskStatus.PROCESSING)
        elif "checking benepar model" in line_lower:
            self._set_task_status("install_benepar", TaskStatus.PROCESSING)
        elif "checking nltk models" in line_lower:
            self._set_task_status("install_nltk", TaskStatus.PROCESSING)
            
        # Handle model presence detection
        if "spacy model is already present" in line_lower:
            self._set_task_status("install_spacy", TaskStatus.COMPLETED)
            self.status_update.emit("SpaCy model is already installed")
        elif "benepar model is already present" in line_lower:
            self._set_task_status("install_benepar", TaskStatus.COMPLETED)
            self.status_update.emit("Benepar model is already installed")
        elif "nltk models are already present" in line_lower:
            self._set_task_status("install_nltk", TaskStatus.COMPLETED)
            self.status_update.emit("NLTK components are already installed")
            
        # Handle model download start
        if "downloading spacy model" in line_lower:
            self._set_task_status("install_spacy", TaskStatus.PROCESSING)
            self.status_update.emit("Downloading SpaCy (text processing) model...")
        elif "download benepar" in line_lower:
            self._set_task_status("install_benepar", TaskStatus.PROCESSING)
            self.status_update.emit("Downloading Benepar (parsing) model...")
        elif "download nltk" in line_lower:
            self._set_task_status("install_nltk", TaskStatus.PROCESSING)
            self.status_update.emit("Downloading NLTK components...")
            
        # Special case for model check: Handle 'one or more models are missing' as a normal status, not failure
        if "one or more models are missing" in line_lower and self._current_task == "check_models":
            # Mark check_models as completed (not failed) and we'll track individual model tasks separately
            self._set_task_status("check_models", TaskStatus.COMPLETED)
            self.status_update.emit("Some models need installation")
            return
            
        # Check for error messages first
        if any(err in line_lower for err in ["error", "failed", "exception", "critical"]):
            # Identify which task this error applies to
            if "spacy" in line_lower and "install_spacy" in self._tasks:
                self._set_task_status("install_spacy", TaskStatus.FAILED)
            elif "benepar" in line_lower and "install_benepar" in self._tasks:
                self._set_task_status("install_benepar", TaskStatus.FAILED)
            elif "nltk" in line_lower and "install_nltk" in self._tasks:
                self._set_task_status("install_nltk", TaskStatus.FAILED)
            elif self._current_task:
                # Default fallback if we can't identify the specific model
                self._set_task_status(self._current_task, TaskStatus.FAILED)
            return
            
        # Check for successful installation messages
        if "successfully downloaded and verified spacy model" in line_lower:
            self._set_task_status("install_spacy", TaskStatus.COMPLETED)
            self.status_update.emit("SpaCy model installed successfully")
        elif "successfully downloaded and verified benepar model" in line_lower:
            self._set_task_status("install_benepar", TaskStatus.COMPLETED)
            self.status_update.emit("Benepar model installed successfully")
        elif "successfully downloaded and verified nltk" in line_lower:
            self._set_task_status("install_nltk", TaskStatus.COMPLETED)
            self.status_update.emit("NLTK components installed successfully")
            
        # Check for overall completion
        if "model setup process completed successfully" in line_lower:
            # Complete any remaining tasks
            for task_id in self._tasks:
                if task_id not in self._completed_tasks:
                    self._set_task_status(task_id, TaskStatus.COMPLETED)
            self.status_update.emit("All language models installed successfully")

    def _clean_status_message(self, message: str) -> str:
        """Clean up log messages to be more user-friendly.
        
        Args:
            message: The raw log message
            
        Returns:
            A cleaned user-friendly message
        """
        # Remove timestamp pattern (2023-04-15 14:49:50,246)
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}'
        
        # Remove log level and module patterns
        log_pattern = r'(- anpe\.setup_models)? - (INFO|DEBUG|WARNING|ERROR|CRITICAL) - '
        
        # Combine patterns
        full_pattern = rf'({timestamp_pattern})?\s*{log_pattern}?'
        
        # Remove patterns
        cleaned = re.sub(full_pattern, '', message)
        
        # Replace technical terms with user-friendly ones
        replacements = {
            r'en_core_web_md': 'English language model',
            r'Downloading spaCy model: en_core_web_md': 'Downloading English language model',
            r'spaCy': 'spaCy (text processing)',
            r'Benepar': 'Benepar (parsing)',
            r'NLTK': 'NLTK (tokenization)',
            r'benepar_en3': 'English parsing model'
        }
        
        for pattern, replacement in replacements.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        
        # Remove double spaces and trim
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()
        
        # Capitalize first letter if needed
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]
            
        return cleaned

    def _set_task_status(self, task_id: str, status: int):
        """Set task status and track state.
        
        Args:
            task_id: The identifier of the task
            status: The status code (from TaskStatus enum)
        """
        if task_id not in self._tasks:
            return
            
        # Update task status
        self.task_status_update.emit(task_id, status)
        
        # Track current task
        if status == TaskStatus.PROCESSING:
            self._current_task = task_id
            
            # Update status message when changing tasks
            task_name = self._tasks[task_id]
            if task_id == "check_models":
                self.status_update.emit("Checking for language models...")
            elif task_id == "install_spacy":
                self.status_update.emit("Downloading spaCy (text processing) model...")
            elif task_id == "install_benepar":
                self.status_update.emit("Downloading Benepar (parsing) model...")
            elif task_id == "install_nltk":
                self.status_update.emit("Downloading NLTK (tokenization) components...")
            
        # Track completed tasks
        if status == TaskStatus.COMPLETED:
            self._completed_tasks.add(task_id)
            
            # Update status message for completed tasks
            if task_id == "check_models":
                self.status_update.emit("Language model check completed.")
            elif task_id == "install_spacy":
                self.status_update.emit("SpaCy model installed successfully.")
            elif task_id == "install_benepar":
                self.status_update.emit("Benepar model installed successfully.")
            elif task_id == "install_nltk":
                self.status_update.emit("NLTK components installed successfully.")
                
        elif status == TaskStatus.FAILED:
            self._completed_tasks.discard(task_id)
            
            # Update status message for failed tasks
            self.status_update.emit(f"Error in {self._tasks[task_id].lower()}.")

    def _handle_stderr(self):
        """Process error output from the script."""
        try:
            data = self._process.readAllStandardError().data().decode(errors='replace').strip()
            if not data:
                return
                
            # Log the error
            self.log_update.emit(f"ERROR: {data}")
            
            # Check for error indication in the current task
            if self._current_task:
                self._set_task_status(self._current_task, TaskStatus.FAILED)
                
        except Exception as e:
            self.log_update.emit(f"Error processing stderr: {e}")

    def _handle_process_error(self, error: QProcess.ProcessError):
        """Handle QProcess errors."""
        error_string = self._process.errorString()
        self.log_update.emit(f"PROCESS ERROR ({error}): {error_string}")
        self.status_update.emit(f"Error launching model setup: {error_string}")
        
        # Mark all tasks as failed
        for task_id in self._tasks:
            self.task_status_update.emit(task_id, TaskStatus.FAILED)
            
        self.finished.emit(False)

    def _handle_finish(self, exit_code: int, exit_status: QProcess.ExitStatus):
        """Handle process completion."""
        # Process any remaining output
        self._handle_stdout()
        self._handle_stderr()
        
        # Log completion
        self.log_update.emit(f"Model setup process finished. Exit code: {exit_code}")
        
        if exit_status == QProcess.ExitStatus.CrashExit:
            self.log_update.emit("ERROR: Model setup process crashed")
            self.status_update.emit("Error: Model setup process crashed unexpectedly")
            
            # Mark remaining tasks as failed
            for task_id in self._tasks:
                if task_id not in self._completed_tasks:
                    self.task_status_update.emit(task_id, TaskStatus.FAILED)
                    
            self.finished.emit(False)
        elif exit_code == 0:
            self.log_update.emit("Model setup completed successfully")
            self.status_update.emit("All language models installed successfully")
            
            # Mark all remaining tasks as complete
            for task_id in self._tasks:
                if task_id not in self._completed_tasks:
                    self.task_status_update.emit(task_id, TaskStatus.COMPLETED)
                    
            self.finished.emit(True)
        else:
            self.log_update.emit(f"ERROR: Model setup failed with exit code {exit_code}")
            self.status_update.emit("Error: Model setup failed. Check logs for details")
            
            # Mark remaining tasks as failed
            for task_id in self._tasks:
                if task_id not in self._completed_tasks:
                    self.task_status_update.emit(task_id, TaskStatus.FAILED)
                    
            self.finished.emit(False)
            
        # Clear the process
        self._process = None
