import sys
import subprocess
import os # Added for shortcut/launch paths
import platform # Added for platform-specific checks
import shutil
import winreg  # For Windows registry operations
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QWidget, QMessageBox, QVBoxLayout, QFrame
from PyQt6.QtCore import Qt, QThread, QProcess # Added QThread and QProcess
from PyQt6.QtGui import QPalette, QColor # Added for background

# Import views
from .views.welcome_view import WelcomeViewWidget
from .views.progress_view import ProgressViewWidget
from .views.completion_view import CompletionViewWidget

# Import workers
from .workers.env_setup_worker import EnvironmentSetupWorker # Added
from .workers.model_setup_worker import ModelSetupWorker # Added

# Import Custom Title Bar
from . widgets.custom_title_bar import CustomTitleBar

# Import utility
from .utils import get_resource_path

from pyshortcuts import make_shortcut


# Constants for view indices (as per design doc)
VIEW_WELCOME = 0
VIEW_ENV_PROGRESS = 1
VIEW_MODEL_PROGRESS = 2
VIEW_COMPLETION = 3

# Define a primary color (using value from anpe_gui.theme)
PRIMARY_COLOR = "#005A9C"
BORDER_RADIUS = 10 # Adjust for desired roundness
BORDER_THICKNESS = 2 # Adjust for desired thickness

class SetupMainWindow(QMainWindow):
    """Main window for the ANPE setup application with custom title bar."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        window_title = "ANPE Setup Wizard"
        self.setWindowTitle(window_title)
        # Fixed size might conflict slightly with exact border/radius look
        # Consider setting minimum size and letting layout manage?
        # For now, keep fixed size.
        self.setFixedSize(650 + (BORDER_THICKNESS * 2), 550 + self._title_bar.height() + (BORDER_THICKNESS * 2) if hasattr(self, '_title_bar') else 500 + 35 + (BORDER_THICKNESS * 2)) 

        # --- Custom Window Frame --- 
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        # Make window background transparent to see rounded corners
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Main container widget (needed for transparency and layout)
        self._main_container = QWidget(self)
        self.setCentralWidget(self._main_container)

        # Layout for the main container
        self._container_layout = QVBoxLayout(self._main_container)
        # Add margins to create space for the border effect
        self._container_layout.setContentsMargins(BORDER_THICKNESS, BORDER_THICKNESS, BORDER_THICKNESS, BORDER_THICKNESS)
        self._container_layout.setSpacing(0)

        # Frame for content (background, border radius)
        self._main_frame = QFrame(self._main_container)
        self._main_frame.setObjectName("MainFrame") # Set object name for styling
        self._container_layout.addWidget(self._main_frame)

        # Layout *inside* the main frame
        self._frame_layout = QVBoxLayout(self._main_frame)
        self._frame_layout.setContentsMargins(0, 0, 0, 0) # Frame handles padding via border
        self._frame_layout.setSpacing(0)

        # Custom Title Bar (Add to frame layout)
        self._title_bar = CustomTitleBar(window_title, self._main_frame)
        self._frame_layout.addWidget(self._title_bar)

        # Content Area (Stacked Widget - Add to frame layout)
        self.stacked_widget = QStackedWidget(self._main_frame)
        self._frame_layout.addWidget(self.stacked_widget)

        # --- Apply Styles --- 
        self.setStyleSheet(f"""
            #MainFrame {{ 
                background-color: white; 
                border: {BORDER_THICKNESS}px solid {PRIMARY_COLOR}; 
                border-radius: {BORDER_RADIUS}px; 
            }}
            /* Ensure child widgets don't draw over the rounded corners */
            /* This might need refinement based on specific widgets used */
            #MainFrame QStackedWidget > QWidget {{ 
                background-color: white; 
                border-radius: 0px; /* Prevent children from having radius */
            }}
            /* Style title bar slightly */
            CustomTitleBar {{ 
                background-color: #f0f0f0; 
                /* Apply top-left/right radius to match main frame */
                border-top-left-radius: {BORDER_RADIUS - BORDER_THICKNESS}px; 
                border-top-right-radius: {BORDER_RADIUS - BORDER_THICKNESS}px; 
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }}
        """)
        # --- End Styling --- 

        # Connect title bar signals
        self._title_bar.minimize_requested.connect(self.showMinimized)
        self._title_bar.close_requested.connect(self.close)

        # --- End Custom Window Frame Setup --- 

        # Store view instances
        self.welcome_view = None
        self.env_progress_view = None
        self.model_progress_view = None
        self.completion_view = None

        # Store setup parameters
        self._install_path = None
        self._python_exe_path = None

        # Worker and thread management
        self._env_worker = None
        self._env_thread = None
        self._model_worker = None
        self._model_thread = None
        self._is_running = False # Flag to track if setup is in progress
        self._is_env_setup_stage = True  # Track current stage

        self._create_views()
        self.stacked_widget.setCurrentIndex(VIEW_WELCOME) # Show welcome view initially

        self.show()

    def _print_log_message(self, message: str):
        """Helper slot to print log messages directly to the console."""
        print(f"WORKER LOG: {message}")

    def _create_views(self):
        """Create and add view widgets to the stacked widget."""
        self.welcome_view = WelcomeViewWidget()
        self.stacked_widget.addWidget(self.welcome_view)
        self.welcome_view.setup_requested.connect(self._handle_setup_request)

        # Environment Progress View (Stage 1)
        self.env_progress_view = ProgressViewWidget("Setting up Environment")
        self.stacked_widget.addWidget(self.env_progress_view)

        # Model Progress View (Stage 2)
        self.model_progress_view = ProgressViewWidget("Setting up Language Models")
        self.stacked_widget.addWidget(self.model_progress_view)

        # Completion View
        self.completion_view = CompletionViewWidget()
        self.stacked_widget.addWidget(self.completion_view)
        self.completion_view.shortcut_requested.connect(self._create_shortcut)
        self.completion_view.launch_requested.connect(self._launch_anpe)
        self.completion_view.close_requested.connect(self.close)

    def _handle_setup_request(self, install_path: str):
        """Slot to handle the setup request from the Welcome view."""
        print(f"Main Window received setup request: Path='{install_path}'")
        
        self._install_path = install_path

        # --- Path Validation --- 
        abs_install_path = os.path.abspath(install_path)
        parent_dir = os.path.dirname(abs_install_path)

        # 1. Check if path is empty
        if not install_path.strip():
            QMessageBox.critical(self, "Invalid Path", "The installation path cannot be empty.")
            return

        # 2. Check if parent directory exists or can be created
        if not os.path.isdir(parent_dir):
             try:
                 print(f"Attempting to create parent directory: {parent_dir}")
                 os.makedirs(parent_dir, exist_ok=True)
             except Exception as e:
                 QMessageBox.critical(self, "Invalid Path", f"The installation path's parent directory does not exist and cannot be created:\n{parent_dir}\nError: {e}")
                 return
        
        # 3. Check writability of the target or parent directory
        # Try writing to the target directory first if it exists, else the parent
        write_test_dir = abs_install_path if os.path.isdir(abs_install_path) else parent_dir
        try:
            test_file = os.path.join(write_test_dir, ".installer_permissions_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
        except Exception as e:
            QMessageBox.critical(self, "Path Not Writable", f"The installation path is not writable:\n{abs_install_path}\n(Tested in: {write_test_dir})\nError: {e}")
            return
        # --- End Path Validation ---

        # Start Stage 1
        self._start_environment_setup()

    def _start_environment_setup(self):
        """Instantiate and start the EnvironmentSetupWorker in a QThread."""
        print("Starting environment setup thread...")
        self._is_running = True
        self._is_env_setup_stage = True  # Track current stage
        
        # Prepare progress view
        self.env_progress_view.clear_log()
        self.env_progress_view.update_status("Initializing environment setup...")
        self.env_progress_view.set_progress_range(0, 0) # Indeterminate
        self.stacked_widget.setCurrentIndex(VIEW_ENV_PROGRESS)

        # Create worker and thread
        self._env_worker = EnvironmentSetupWorker(self._install_path)
        
        # Set up task list based on worker's tasks
        self.env_progress_view.setup_tasks_from_worker(self._env_worker)
        
        self._env_thread = QThread()
        self._env_worker.moveToThread(self._env_thread)

        # Connect signals
        self._env_worker.log_update.connect(self.env_progress_view.append_log)
        self._env_worker.log_update.connect(self._print_log_message)
        self._env_worker.status_update.connect(self.env_progress_view.update_status)
        self._env_worker.task_status_update.connect(self.env_progress_view.update_task_status)
        self._env_worker.finished.connect(self._environment_setup_finished)
        self._env_thread.started.connect(self._env_worker.run)
        self._env_worker.finished.connect(self._env_thread.quit)
        self._env_worker.finished.connect(self._env_worker.deleteLater)
        self._env_thread.finished.connect(self._env_thread.deleteLater)

        self._env_thread.start()

    def _environment_setup_finished(self, success: bool, python_exe_path: str):
        """Handle completion of the environment setup worker."""
        print(f"Environment setup finished. Success: {success}, Python Path: {python_exe_path}")
        # self._env_thread = None # REMOVE: Let deleteLater handle cleanup
        # self._env_worker = None # REMOVE: Let deleteLater handle cleanup

        if success and python_exe_path:
            self._python_exe_path = python_exe_path
            # Start Stage 2: Model Setup
            self._start_model_setup()
        else:
            # Setup failed, show completion view in failure state
            self._is_running = False
            self._is_env_setup_stage = True  # Still in env stage when failed
            self._show_completion_view(success=False)

    def _start_model_setup(self):
        """Instantiate and start the ModelSetupWorker in a QThread."""
        print("Starting model setup thread...")
        self._is_running = True # Still running
        self._is_env_setup_stage = False  # Now in model setup stage
        
        # Prepare progress view
        self.model_progress_view.clear_log()
        self.model_progress_view.update_status("Initializing language model setup...")
        self.model_progress_view.set_progress_range(0, 0) # Indeterminate
        self.stacked_widget.setCurrentIndex(VIEW_MODEL_PROGRESS)

        # Create worker and thread
        self._model_worker = ModelSetupWorker(self._python_exe_path)
        
        # Set up task list based on worker's tasks
        self.model_progress_view.setup_tasks_from_worker(self._model_worker)
        
        self._model_thread = QThread()
        self._model_worker.moveToThread(self._model_thread)

        # Connect signals
        self._model_worker.log_update.connect(self.model_progress_view.append_log)
        self._model_worker.log_update.connect(self._print_log_message)
        self._model_worker.status_update.connect(self.model_progress_view.update_status)
        self._model_worker.task_status_update.connect(self.model_progress_view.update_task_status)
        self._model_worker.finished.connect(self._model_setup_finished)
        self._model_thread.started.connect(self._model_worker.run)
        self._model_worker.finished.connect(self._model_thread.quit)
        self._model_worker.finished.connect(self._model_worker.deleteLater)
        self._model_thread.finished.connect(self._model_thread.deleteLater)

        self._model_thread.start()

    def _model_setup_finished(self, success: bool):
        """Handle completion of the model setup worker."""
        print(f"Model setup finished. Success: {success}")
        # self._model_thread = None # REMOVE: Let deleteLater handle cleanup
        # self._model_worker = None # REMOVE: Let deleteLater handle cleanup
        self._is_running = False # Setup process is now fully complete

        # Show completion view (success or failure)
        self._show_completion_view(success=success)

    def _show_completion_view(self, success: bool):
        """Switch to the completion view and set its state."""
        # Collect logs from the appropriate progress view
        log_content = ""
        if hasattr(self, '_is_env_setup_stage') and self._is_env_setup_stage:
            # Environment setup logs
            log_content = self.env_progress_view._log_area.toPlainText()
        else:
            # Model setup logs (default to this if we can't determine stage)
            log_content = self.model_progress_view._log_area.toPlainText()
            
        # Set success state and pass logs
        self.completion_view.set_success_state(success, log_content)
        self.stacked_widget.setCurrentIndex(VIEW_COMPLETION)

    # --- Slots for Completion Actions --- 
    def _create_shortcut(self, create: bool):
        """Handle the request to create shortcuts (using make_shortcut function)."""
        if not create:
            print("Skipping shortcut creation.")
            return

        if not self._python_exe_path or not self._install_path:
            QMessageBox.warning(self, "Cannot Create Shortcut", "Internal error: Missing Python path or installation path.")
            return

        print("Creating shortcut via make_shortcut...")

        # --- Define Shortcut Parameters & Paths ---
        shortcut_name = "ANPE"
        app_dir_abs = os.path.abspath(os.path.join(self._install_path, "anpe_gui"))
        python_exe_abs = os.path.abspath(self._python_exe_path)

        # --- Determine Python(w).exe for direct execution --- 
        pythonw_exe_abs = python_exe_abs.replace("python.exe", "pythonw.exe")
        if not os.path.exists(pythonw_exe_abs):
            pythonw_exe_abs = python_exe_abs # Fallback
        
        # Use absolute path for script instead of relative
        target_script_abs = os.path.join(app_dir_abs, "run.py")
        
        # --- Resolve Icon Path --- 
        icon_path = None
        try:
            icon_filename = "app_icon_logo.ico"
            icon_path = get_resource_path(f"assets/{icon_filename}") 
            if not os.path.isfile(icon_path):
                 print(f"Warning: Icon file not found: {icon_filename}. Shortcut will use default.", file=sys.stderr)
                 icon_path = None
        except Exception as e:
            print(f"Warning: Could not resolve icon path: {e}. Shortcut will use default.", file=sys.stderr)
            icon_path = None
        # -------------------------

        try:
            # Create shortcut directly to pythonw.exe with script as argument
            # This avoids the batch file and window icon issues
            print(f"Calling make_shortcut: name='{shortcut_name}', script='{target_script_abs}', icon='{icon_path}', terminal=False, working_dir='{app_dir_abs}'")
            make_shortcut(
                script=target_script_abs,  # Use absolute path to the Python script
                name=shortcut_name,
                icon=icon_path,
                executable=pythonw_exe_abs,  # Use pythonw.exe directly
                terminal=False, 
                desktop=True,  
                startmenu=True, 
                working_dir=app_dir_abs  # Set working directory to app directory
            )
            
            # Register application in Windows and create uninstaller
            self._register_app_and_create_uninstaller(app_dir_abs, pythonw_exe_abs, icon_path)
            
            # No dialog needed, user already has the completion screen
            print("Desktop and Start Menu shortcuts created successfully.")
        except Exception as e:
            print(f"make_shortcut failed: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Shortcut Failed", f"Failed to create shortcuts using make_shortcut.\nError: {e}")

    def _register_app_and_create_uninstaller(self, app_dir_abs, pythonw_exe_abs, icon_path):
        """Register the application in Windows Add/Remove Programs and create uninstaller."""
        try:
            # 1. Create installation log file to track installed files
            install_log_path = os.path.join(self._install_path, "install_log.txt")
            with open(install_log_path, "w") as log_file:
                # Record installation path and other important info
                log_file.write(f"INSTALL_PATH={self._install_path}\n")
                log_file.write(f"APP_DIR={app_dir_abs}\n")
                log_file.write(f"PYTHON_PATH={self._python_exe_path}\n")
                log_file.write(f"PYTHONW_PATH={pythonw_exe_abs}\n")
                log_file.write(f"INSTALL_DATE={subprocess.check_output('date /t', shell=True).decode().strip()}\n")
                
                # Log the directories and their contents
                for root, dirs, files in os.walk(self._install_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        log_file.write(f"FILE={file_path}\n")
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        log_file.write(f"DIR={dir_path}\n")
            
            # 2. Copy the uninstaller module to the installation directory
            uninstaller_source = get_resource_path("uninstaller")
            uninstaller_dest = os.path.join(self._install_path, "uninstaller")
            if os.path.exists(uninstaller_source):
                if not os.path.exists(uninstaller_dest):
                    os.makedirs(uninstaller_dest, exist_ok=True)
                shutil.copytree(uninstaller_source, uninstaller_dest, dirs_exist_ok=True)
                print(f"Copied uninstaller from {uninstaller_source} to {uninstaller_dest}")
            else:
                print(f"Warning: Uninstaller source not found at {uninstaller_source}")
                
            # 3. Create uninstaller script
            uninstaller_script = os.path.join(self._install_path, "uninstall_ANPE.pyw")
            with open(uninstaller_script, "w") as f:
                f.write(f'''import os
import sys
import subprocess

# Launch the uninstaller GUI
def main():
    uninstaller_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uninstaller", "uninstall.pyw")
    pythonw_exe = "{pythonw_exe_abs}"
    
    # Launch uninstaller GUI
    subprocess.Popen([pythonw_exe, uninstaller_path], 
                    cwd=os.path.dirname(uninstaller_path))

if __name__ == "__main__":
    main()
''')

            # 4. Create registry entries to add the application to Add/Remove Programs
            app_name = "ANPE"
            app_version = "1.0.0"  # You might want to parameterize this
            
            # HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Uninstall\ANPE
            reg_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\ANPE"
            
            # Create/open the key
            reg_key = winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, reg_path, 0, winreg.KEY_WRITE)
            
            # Set values for the key
            winreg.SetValueEx(reg_key, "DisplayName", 0, winreg.REG_SZ, app_name)
            winreg.SetValueEx(reg_key, "DisplayVersion", 0, winreg.REG_SZ, app_version)
            winreg.SetValueEx(reg_key, "Publisher", 0, winreg.REG_SZ, "ANPE Developer")
            winreg.SetValueEx(reg_key, "InstallLocation", 0, winreg.REG_SZ, self._install_path)
            winreg.SetValueEx(reg_key, "UninstallString", 0, winreg.REG_SZ, 
                             f'"{pythonw_exe_abs}" "{uninstaller_script}"')
            winreg.SetValueEx(reg_key, "DisplayIcon", 0, winreg.REG_SZ, icon_path if icon_path else "")
            winreg.SetValueEx(reg_key, "NoModify", 0, winreg.REG_DWORD, 1)
            winreg.SetValueEx(reg_key, "NoRepair", 0, winreg.REG_DWORD, 1)
            
            winreg.CloseKey(reg_key)
            
            print("Application registered in Windows Add/Remove Programs")
            
        except Exception as e:
            print(f"Error registering application: {e}")
            import traceback
            traceback.print_exc()
            # Continue without failing - this is an optional part of the installation

    def _launch_anpe(self, launch: bool):
        """Handle the request to launch ANPE (using subprocess.Popen)."""
        if not launch:
            print("Skipping ANPE launch.")
            return
        
        if not self._python_exe_path or not self._install_path:
            QMessageBox.warning(self, "Cannot Launch ANPE", "Internal error: Missing Python path or installation path.")
            return
            
        print("Launching ANPE...")
        # --- Set the working directory and launch command --- 
        app_dir = os.path.join(self._install_path, "anpe_gui")
        python_exe_abs = os.path.abspath(self._python_exe_path)
        
        # Use pythonw.exe to hide command window
        pythonw_exe_abs = python_exe_abs.replace("python.exe", "pythonw.exe")
        if not os.path.exists(pythonw_exe_abs):
            pythonw_exe_abs = python_exe_abs 

        # Launch run.py from the app_dir using pythonw.exe
        target_script_relative = "run.py"
        launch_command = [pythonw_exe_abs, target_script_relative]

        print(f"Setting working directory for launch: {app_dir}")
        print(f"Running launch command: {' '.join(launch_command)}")

        try:
            # Pass cwd (app_dir) to Popen
            subprocess.Popen(launch_command, cwd=app_dir)
        except FileNotFoundError:
            # Update error message if needed
            QMessageBox.critical(self, "Launch Failed", f"Could not find Python or the application script.\nCommand: {' '.join(launch_command)}\nDirectory: {app_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Launch Error", f"An unexpected error occurred while trying to launch ANPE: {e}")

    def closeEvent(self, event):
        """Handle the main window close event."""
        if self._is_running:
            reply = QMessageBox.question(self, 'Confirm Close',
                                       "Setup is currently in progress. Are you sure you want to cancel and exit?",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                       QMessageBox.StandardButton.No)

            if reply == QMessageBox.StandardButton.Yes:
                print("Terminating running processes and closing...")
                
                # First try to terminate any running processes gracefully
                # Environment setup worker
                if self._env_thread and self._env_thread.isRunning():
                    if hasattr(self._env_worker, "_process") and self._env_worker._process:
                        try:
                            # Try to terminate any running subprocess
                            self._env_worker._process.terminate()
                            # Give it a moment to terminate
                            self._env_worker._process.waitForFinished(1000)
                            # If it's still running, kill it
                            if self._env_worker._process.state() != QProcess.ProcessState.NotRunning:
                                self._env_worker._process.kill()
                        except Exception as e:
                            print(f"Error terminating environment process: {e}")
                    
                    # Quit the thread
                    self._env_thread.quit()
                    # Wait briefly for thread to finish
                    if not self._env_thread.wait(1000):
                        print("Environment setup thread did not terminate in time")
                
                # Model setup worker
                if self._model_thread and self._model_thread.isRunning():
                    if hasattr(self._model_worker, "_process") and self._model_worker._process:
                        try:
                            # Try to terminate any running subprocess
                            self._model_worker._process.terminate()
                            # Give it a moment to terminate
                            self._model_worker._process.waitForFinished(1000)
                            # If it's still running, kill it
                            if self._model_worker._process.state() != QProcess.ProcessState.NotRunning:
                                self._model_worker._process.kill()
                        except Exception as e:
                            print(f"Error terminating model process: {e}")
                    
                    # Quit the thread
                    self._model_thread.quit()
                    # Wait briefly for thread to finish
                    if not self._model_thread.wait(1000):
                        print("Model setup thread did not terminate in time")
                
                # Set flag to not running
                self._is_running = False
                
                # Accept the close event
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    # Apply basic styling later if needed
    # app.setStyle("Fusion")
    window = SetupMainWindow()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
