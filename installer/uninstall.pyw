import sys
import os

# --- Add parent directory to sys.path to allow relative imports ---
# This ensures that imports like '.widgets' work even if the script
# is not run as a module from the parent directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Check if the script's directory is already in the path
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
# Also add the parent of the script dir (installation root) if needed
INSTALL_ROOT_GUESS = os.path.dirname(SCRIPT_DIR)
if INSTALL_ROOT_GUESS not in sys.path:
     sys.path.insert(0, INSTALL_ROOT_GUESS)
# ------------------------------------------------------------------

import shutil
import winreg
import subprocess
import time
import traceback # Import traceback for detailed error logging
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QWidget, QMessageBox, 
    QVBoxLayout, QHBoxLayout, QFrame, QPushButton, QLabel, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QPixmap

# --- Try importing shared components ---
# Now assumes uninstall.pyw is in the installer/ directory, sibling to others
try:
    from .widgets.custom_title_bar import CustomTitleBar
    from .views.progress_view import ProgressViewWidget
    from .utils import get_resource_path
    from .styles import (
        PRIMARY_BUTTON_STYLE, SECONDARY_BUTTON_STYLE, TITLE_LABEL_STYLE, 
        INFO_LABEL_STYLE, LOG_TEXT_AREA_STYLE,
        COMPACT_PRIMARY_BUTTON_STYLE, COMPACT_SECONDARY_BUTTON_STYLE,
        COMPACT_DANGER_BUTTON_STYLE
    )
except ImportError as e:
    print(f"ERROR: Could not import shared components: {e}", file=sys.stderr)
    # Fallbacks might be needed if run standalone, but assume normal execution
    sys.exit("Uninstaller cannot run without shared components.")


# Constants for view indices
VIEW_WELCOME = 0
VIEW_PROGRESS = 1
VIEW_COMPLETION = 2

# Define constants from installer styles
PRIMARY_COLOR = "#005A9C"
BORDER_RADIUS = 10  # Adjust for desired roundness
BORDER_THICKNESS = 2  # Adjust for desired thickness


class WelcomeUninstallWidget(QWidget):
    """Welcome screen for the uninstaller."""
    uninstall_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)
        
        # Logo
        logo_label = QLabel()
        try:
            logo_path = get_resource_path('assets/app_icon_logo.png')
            if os.path.exists(logo_path):
                pixmap = QPixmap(logo_path)
                if not pixmap.isNull():
                    logo_label.setPixmap(pixmap.scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                logo_label.setText("ANPE")
                logo_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #0066b2;")
                logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        except Exception as e:
            logo_label.setText("[Logo Error]")
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            print(f"Error resolving logo path: {e}", file=sys.stderr)
            
        logo_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(logo_label)
        
        # Title and welcome text
        title_label = QLabel("ANPE Uninstaller")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(TITLE_LABEL_STYLE) # Use shared style
        layout.addWidget(title_label)
        
        # Read installation path to display
        install_path = self._get_install_path()
        if not install_path or not os.path.isdir(install_path):
             install_path_display = "Installation path not found. Uninstallation may be incomplete."
             path_style = "color: red;"
        else:
             install_path_display = f"Files will be removed from: {install_path}"
             path_style = INFO_LABEL_STYLE

        welcome_text = QLabel(
            "This will uninstall ANPE and its components from your computer."
        )
        welcome_text.setWordWrap(True)
        welcome_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_text)
        
        path_label = QLabel(install_path_display)
        path_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        path_label.setStyleSheet(path_style)
        layout.addWidget(path_label)
        
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Button layout - Only show Uninstall button, centered
        button_layout = QHBoxLayout()
        button_layout.addStretch() # Center button
        
        # Uninstall button (use compact danger style)
        self.uninstall_button = QPushButton("Uninstall")
        self.uninstall_button.setStyleSheet(COMPACT_DANGER_BUTTON_STYLE) # Use compact danger style
        self.uninstall_button.setEnabled(install_path is not None and os.path.isdir(install_path))
        self.uninstall_button.clicked.connect(self.uninstall_requested.emit)
        button_layout.addWidget(self.uninstall_button)
        
        button_layout.addStretch() # Center button
        layout.addLayout(button_layout)
    
    def _get_install_path(self):
        """Get the installation path from the registry or fallback."""
        try:
            reg_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\ANPE"
            reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path)
            install_path, _ = winreg.QueryValueEx(reg_key, "InstallLocation")
            winreg.CloseKey(reg_key)
            # Verify path exists
            if os.path.isdir(install_path):
                return install_path
            else:
                print(f"Registry path found but invalid: {install_path}", file=sys.stderr)
                return None # Path from registry doesn't exist
        except FileNotFoundError:
             print("Registry key not found.", file=sys.stderr)
             return None # Key not found
        except Exception as e:
            print(f"Error reading registry: {e}", file=sys.stderr)
            # Fallback logic removed - if registry fails, we cannot reliably find the install path
            # without assuming a specific structure relative to the uninstaller, which is less robust.
            return None


class CompletionUninstallWidget(QWidget):
    """Completion screen for the uninstaller."""
    close_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)
        
        # Reuse logo
        logo_label = QLabel()
        try:
            logo_path = get_resource_path('assets/app_icon_logo.png')
            if os.path.exists(logo_path):
                pixmap = QPixmap(logo_path)
                if not pixmap.isNull():
                    logo_label.setPixmap(pixmap.scaled(70, 70, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        except Exception as e:
            print(f"Error resolving logo path: {e}", file=sys.stderr)
        layout.addWidget(logo_label)

        # Completion message
        self.title_label = QLabel("Uninstallation Complete")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet(TITLE_LABEL_STYLE)
        layout.addWidget(self.title_label)
        
        self.message_label = QLabel(
            "ANPE has been successfully uninstalled from your computer."
        )
        self.message_label.setWordWrap(True)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.message_label)
        
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Close button (Compact Primary style)
        close_button = QPushButton("Close")
        close_button.setStyleSheet(COMPACT_PRIMARY_BUTTON_STYLE) # Use compact style
        close_button.clicked.connect(self.close_requested.emit)
        
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        layout.addLayout(button_layout)

    def set_state(self, success: bool, message: str = ""):
         """Set the completion state (success or failure)."""
         if success:
             self.title_label.setText("Uninstallation Complete")
             self.title_label.setStyleSheet(TITLE_LABEL_STYLE)
             self.message_label.setText("ANPE has been successfully uninstalled.")
         else:
             self.title_label.setText("Uninstallation Failed")
             self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #DD3333;") # Use error color
             error_text = "An error occurred during uninstallation."
             if message:
                 error_text += f"\n\nDetails: {message}"
             self.message_label.setText(error_text)


class UninstallWorker(QObject):
    """Worker thread to handle the actual uninstallation process."""
    status_update = pyqtSignal(str)          # For user-friendly status label
    log_update = pyqtSignal(str)             # For detailed log view
    progress_update = pyqtSignal(int, int)   # current step, total steps
    finished = pyqtSignal(bool, str)         # success, final message

    def __init__(self, install_path: str):
        super().__init__()
        if not install_path or not os.path.isdir(install_path):
             raise ValueError("Invalid installation path provided to UninstallWorker.")
        self.install_path = os.path.abspath(install_path) # Use absolute path
        self.install_log_path = os.path.join(self.install_path, "install_log.txt")
        # Uninstaller directory is now the same as the installer directory
        self.uninstaller_script_path = os.path.abspath(__file__)

    def run(self):
        """Main uninstallation process."""
        steps = 5 # Define total number of major steps for progress
        current_step = 0
        final_message = ""
        try:
            # --- Step 1: Initialization ---
            current_step += 1
            self.progress_update.emit(current_step, steps)
            self.status_update.emit("Starting uninstallation...")
            self.log_update.emit(f"Target installation path: {self.install_path}")
            self.log_update.emit(f"Uninstaller script running from: {self.uninstaller_script_path}")

            time.sleep(0.5) # Small delay for user to see message

            # --- Step 2: Remove Shortcuts ---
            current_step += 1
            self.progress_update.emit(current_step, steps)
            self.status_update.emit("Removing shortcuts...")
            self.log_update.emit("Attempting to remove shortcuts...")
            self._remove_shortcuts()
            time.sleep(0.2)

            # --- Step 3: Remove Registry Entries ---
            current_step += 1
            self.progress_update.emit(current_step, steps)
            self.status_update.emit("Removing registry entries...")
            self.log_update.emit("Attempting to remove registry entries...")
            self._remove_registry_entries()
            time.sleep(0.2)

            # --- Step 4: Remove Files and Directories ---
            current_step += 1
            self.progress_update.emit(current_step, steps)
            self.status_update.emit("Removing installed files...")
            self.log_update.emit("Attempting to remove installed files and directories...")
            items_failed = self._remove_installed_files() # Get count of failed removals
            time.sleep(0.5) # Allow time for file operations

            # --- Step 5: Final Cleanup ---
            current_step += 1
            self.progress_update.emit(current_step, steps)
            self.status_update.emit("Finishing uninstallation...")
            self.log_update.emit("Uninstallation process completed logic.")
            
            if items_failed > 0:
                final_message = f"Uninstallation finished with {items_failed} errors. Manual cleanup may be required."
                self.log_update.emit(f"WARNING: {final_message}")
                self.finished.emit(False, final_message) # Indicate partial success/failure
            else:
                final_message = "ANPE components removed successfully."
                self.log_update.emit("SUCCESS: All detected items (excluding uninstaller script) removed.")
                self.finished.emit(True, final_message)

        except Exception as e:
            error_details = traceback.format_exc()
            self.log_update.emit(f"CRITICAL ERROR during uninstallation: {str(e)}\n{error_details}")
            self.status_update.emit("Uninstallation failed critically")
            final_message = f"Uninstallation failed: {str(e)}"
            self.finished.emit(False, final_message)

    def _remove_shortcuts(self):
        """Remove desktop and start menu shortcuts."""
        shortcut_name = "ANPE.lnk"
        locations = [
            os.path.join(os.path.expanduser("~"), "Desktop"),
            os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs")
        ]
        for loc in locations:
            shortcut_path = os.path.join(loc, shortcut_name)
            try:
                if os.path.exists(shortcut_path):
                    os.remove(shortcut_path)
                    self.log_update.emit(f"Removed shortcut: {shortcut_path}")
                else:
                    self.log_update.emit(f"Shortcut not found (already removed?): {shortcut_path}")
            except Exception as e:
                self.log_update.emit(f"WARNING: Error removing shortcut {shortcut_path}: {e}")

    def _remove_registry_entries(self):
        """Remove registry entries."""
        reg_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\ANPE"
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, reg_path)
            self.log_update.emit(f"Removed registry key: HKEY_CURRENT_USER\\{reg_path}")
        except FileNotFoundError:
            self.log_update.emit(f"Registry key not found (already removed?): HKEY_CURRENT_USER\\{reg_path}")
        except Exception as e:
            self.log_update.emit(f"WARNING: Error removing registry key {reg_path}: {e}")

    def _remove_installed_files(self) -> int:
        """Remove installed files and directories, avoiding the uninstaller itself. Returns count of failed removals."""
        # We will remove the *contents* of the install path
        self.log_update.emit(f"Scanning installation directory for removal: {self.install_path}")

        items_removed = 0
        items_failed = 0

        # Robust check if install path still exists
        if not os.path.isdir(self.install_path):
            self.log_update.emit(f"WARNING: Installation directory {self.install_path} does not exist. Skipping file removal.")
            return 0 # No items failed if directory is gone

        try:
            for item_name in os.listdir(self.install_path):
                item_path = os.path.join(self.install_path, item_name)

                # --- CRITICAL CHECK: Do NOT remove the uninstaller script itself ---
                try:
                    is_same_file = os.path.normcase(os.path.abspath(item_path)) == os.path.normcase(self.uninstaller_script_path)
                except OSError:
                    is_same_file = False
                    
                if is_same_file:
                    self.log_update.emit(f"Skipping removal of active uninstaller script: {item_path}")
                    continue
                
                # Also skip the install log itself as it might be open
                try:
                     is_same_log = os.path.normcase(os.path.abspath(item_path)) == os.path.normcase(os.path.abspath(self.install_log_path))
                except OSError:
                    is_same_log = False
                    
                if is_same_log:
                     self.log_update.emit(f"Skipping removal of install log file: {item_path}")
                     continue

                self.log_update.emit(f"Attempting to remove item: {item_path}")
                removed = False
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=False)
                        self.log_update.emit(f"Successfully removed directory tree: {item_path}")
                        removed = True
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                        self.log_update.emit(f"Successfully removed file: {item_path}")
                        removed = True
                    else:
                        self.log_update.emit(f"Skipping item (not file or dir?): {item_path}")
                    
                    if removed:
                        items_removed += 1
                        
                except PermissionError as pe:
                     self.log_update.emit(f"WARNING: Permission denied removing {item_path}. It might be in use. Error: {pe}")
                     items_failed += 1
                except Exception as e:
                    self.log_update.emit(f"WARNING: Failed to remove {item_path}: {e}")
                    items_failed += 1
                time.sleep(0.05) # Small delay between operations
        except Exception as list_e:
             self.log_update.emit(f"ERROR: Could not list installation directory {self.install_path}: {list_e}")
             items_failed += 1 # Consider this a failure

        self.log_update.emit(f"Finished scan. Items removed: {items_removed}, Items failed: {items_failed}")
        
        # Final Note about manual removal
        self.log_update.emit("NOTE: The main installation folder and the uninstaller script itself may need to be manually removed after closing this window.")
        return items_failed


class UninstallMainWindow(QMainWindow):
    """Main window for the ANPE uninstaller application with custom title bar."""

    def __init__(self):
        """Initialize the main window."""
        super().__init__()

        window_title = "ANPE Uninstaller"
        self.setWindowTitle(window_title)
        self.setFixedSize(600 + (BORDER_THICKNESS * 2), 450 + 35 + (BORDER_THICKNESS * 2))

        # --- Custom Window Frame Setup (Identical to Installer) ---
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._main_container = QWidget(self)
        self.setCentralWidget(self._main_container)

        self._container_layout = QVBoxLayout(self._main_container)
        self._container_layout.setContentsMargins(BORDER_THICKNESS, BORDER_THICKNESS, BORDER_THICKNESS, BORDER_THICKNESS)
        self._container_layout.setSpacing(0)

        self._main_frame = QFrame(self._main_container)
        self._main_frame.setObjectName("MainFrame")
        self._container_layout.addWidget(self._main_frame)

        self._frame_layout = QVBoxLayout(self._main_frame)
        self._frame_layout.setContentsMargins(0, 0, 0, 0)
        self._frame_layout.setSpacing(0)

        self._title_bar = CustomTitleBar(window_title, self._main_frame)
        self._frame_layout.addWidget(self._title_bar)

        self.stacked_widget = QStackedWidget(self._main_frame)
        self._frame_layout.addWidget(self.stacked_widget)

        self.setStyleSheet(f"""
            #MainFrame {{
                background-color: white;
                border: {BORDER_THICKNESS}px solid {PRIMARY_COLOR};
                border-radius: {BORDER_RADIUS}px;
            }}
            #MainFrame QStackedWidget > QWidget {{
                background-color: white;
                border-radius: 0px; /* Prevent children from having radius */
            }}
            CustomTitleBar {{
                background-color: #f0f0f0;
                border-top-left-radius: {BORDER_RADIUS - BORDER_THICKNESS}px;
                border-top-right-radius: {BORDER_RADIUS - BORDER_THICKNESS}px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }}
        """)
        # --- End Custom Window Frame Setup ---

        self._title_bar.minimize_requested.connect(self.showMinimized)
        self._title_bar.close_requested.connect(self.close)

        self.welcome_view = None
        self.progress_view = None
        self.completion_view = None

        self._worker = None
        self._thread = None
        self._is_running = False
        self._install_path = None # Store the validated install path

        self._create_views()
        if self._install_path: # Only proceed if path is valid
            self.stacked_widget.setCurrentIndex(VIEW_WELCOME)
        else:
             QMessageBox.critical(self, "Error", "Could not determine valid installation path. Cannot proceed with uninstallation.")
             self.stacked_widget.setCurrentIndex(VIEW_WELCOME) # Show welcome but button will be disabled


        self.show()

    def _create_views(self):
        """Create and add view widgets to the stacked widget."""
        # Welcome View
        self.welcome_view = WelcomeUninstallWidget()
        self._install_path = self.welcome_view._get_install_path() # Get path early
        self.stacked_widget.addWidget(self.welcome_view)
        if self._install_path: # Only connect if path is valid
            self.welcome_view.uninstall_requested.connect(self._start_uninstallation)

        # Progress View (Use shared ProgressViewWidget)
        self.progress_view = ProgressViewWidget("Uninstalling ANPE")
        # --- Hide the task list for the uninstaller --- 
        if hasattr(self.progress_view, '_task_list'):
            self.progress_view._task_list.setVisible(False)
        # Do NOT call setup_tasks for the uninstaller
        # self.progress_view.setup_tasks({"uninstall": "Uninstalling ANPE"}) 
        self.stacked_widget.addWidget(self.progress_view)

        # Completion View
        self.completion_view = CompletionUninstallWidget()
        self.stacked_widget.addWidget(self.completion_view)
        self.completion_view.close_requested.connect(self.close)

    def _start_uninstallation(self):
        """Start the uninstallation process in a separate thread."""
        if not self._install_path:
             QMessageBox.critical(self, "Error", "Invalid installation path. Cannot start uninstallation.")
             return

        reply = QMessageBox.question(
            self, "Confirm Uninstallation",
            "Are you sure you want to uninstall ANPE?\nThis will remove all installed components.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        self._is_running = True

        # Prepare progress view
        self.progress_view.clear_log()
        self.progress_view.update_status("Initializing uninstallation...")
        self.progress_view.set_progress_range(0, 5) # 5 steps in worker
        self.progress_view.set_progress_value(0)
        # self.progress_view.update_task_status("uninstall", 1) # REMOVED - Task list hidden
        # Ensure log area is visible
        if hasattr(self.progress_view, '_log_area') and not self.progress_view._log_area.isVisible():
             if hasattr(self.progress_view, '_toggle_details'):
                 self.progress_view._toggle_details()
             else:
                 self.progress_view._log_area.setVisible(True)
                 if hasattr(self.progress_view, '_details_button'):
                     self.progress_view._details_button.setText("Hide Details")
        
        self.stacked_widget.setCurrentIndex(VIEW_PROGRESS)

        # Create worker and thread
        try:
            self._worker = UninstallWorker(self._install_path)
        except ValueError as e:
             QMessageBox.critical(self, "Worker Error", f"Failed to initialize uninstaller: {e}")
             self._is_running = False
             self.stacked_widget.setCurrentIndex(VIEW_WELCOME) # Go back
             return

        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._worker.log_update.connect(self.progress_view.append_log)
        self._worker.status_update.connect(self.progress_view.update_status)
        self._worker.progress_update.connect(self.progress_view.set_progress_value) # Direct connect
        self._worker.finished.connect(self._uninstall_finished)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    # Removed _update_progress

    def _uninstall_finished(self, success, message):
        """Handle completion of the uninstallation."""
        self._is_running = False

        # Update progress view task status - REMOVED as task list is hidden
        # self.progress_view.update_task_status("uninstall", 2 if success else 3) 

        self.completion_view.set_state(success, message)
        self.stacked_widget.setCurrentIndex(VIEW_COMPLETION)

        # Optionally display a final message box, especially on failure
        if not success:
            QMessageBox.warning(
                self, "Uninstallation Problem",
                f"Uninstallation encountered problems.\nDetails: {message}\n\nPlease check the installation directory for remaining files."
            )

    def closeEvent(self, event):
        """Handle the main window close event."""
        if self._is_running:
            reply = QMessageBox.question(
                self, 'Confirm Close',
                "Uninstallation is currently in progress. Are you sure you want to cancel and exit?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                if self._thread and self._thread.isRunning():
                    print("Attempting to stop uninstallation thread...")
                    self._thread.quit()
                    if not self._thread.wait(1000): # Wait 1 sec
                         print("Warning: Uninstallation thread did not stop gracefully.")
                         self._thread.terminate() # Force terminate if needed
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
            print("Uninstaller closed. Manual removal of the installation folder might be required.")


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = UninstallMainWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 