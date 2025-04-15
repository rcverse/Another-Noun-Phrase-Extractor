import sys
import os
import shutil
import winreg
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QWidget, QMessageBox, 
    QVBoxLayout, QHBoxLayout, QFrame, QPushButton, QLabel, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QColor, QPixmap

# Constants for view indices
VIEW_WELCOME = 0
VIEW_PROGRESS = 1
VIEW_COMPLETION = 2

# Define a primary color (using value from anpe_gui.theme)
PRIMARY_COLOR = "#005A9C"
BORDER_RADIUS = 10  # Adjust for desired roundness
BORDER_THICKNESS = 2  # Adjust for desired thickness

# Import custom title bar - relative import from the installer package
# This might need adjustment based on how you package your installer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from ..widgets.custom_title_bar import CustomTitleBar
    from ..views.progress_view import ProgressViewWidget
    from ..utils import get_resource_path
except ImportError:
    # Fallback imports (if relative imports don't work)
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "installer"))
        from installer.widgets.custom_title_bar import CustomTitleBar
        from installer.views.progress_view import ProgressViewWidget
        from installer.utils import get_resource_path
    except ImportError:
        # If still can't import, create minimal implementations
        class CustomTitleBar(QWidget):
            close_requested = pyqtSignal()
            minimize_requested = pyqtSignal()
            
            def __init__(self, title, parent=None):
                super().__init__(parent)
                layout = QHBoxLayout(self)
                layout.setContentsMargins(10, 5, 10, 5)
                self.title_label = QLabel(title)
                close_button = QPushButton("X")
                close_button.clicked.connect(self.close_requested.emit)
                layout.addWidget(self.title_label)
                layout.addStretch()
                layout.addWidget(close_button)
                
        def get_resource_path(relative_path):
            """Get absolute path to resource from relative path."""
            base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(base_path, relative_path)


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
        
        # Logo (same as installer)
        logo_label = QLabel()
        try:
            logo_path = get_resource_path('assets/app_icon_logo.png')
            if os.path.exists(logo_path):
                pixmap = QPixmap(logo_path)
                if not pixmap.isNull():
                    logo_label.setPixmap(pixmap.scaled(96, 96, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                    logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                else:
                    logo_label.setText("[Logo Load Error]")
                    logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            else:
                logo_label.setText("[Logo Not Found]")
                logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        except Exception as e:
            logo_label.setText("[Logo Error]")
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            
        logo_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(logo_label)
        
        # Title and welcome text
        title_label = QLabel("ANPE Uninstaller")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Read installation path from registry
        install_path = self._get_install_path()
        
        welcome_text = QLabel(
            "This will uninstall ANPE from your computer.\n\n"
            f"Installation location: {install_path}\n\n"
            "Click 'Uninstall' to continue or 'Cancel' to exit."
        )
        welcome_text.setWordWrap(True)
        welcome_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_text)
        
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.window().close)
        
        # Uninstall button
        uninstall_button = QPushButton("Uninstall")
        uninstall_button.setStyleSheet("QPushButton { font-size: 14pt; padding: 10px; }")
        uninstall_button.clicked.connect(self.uninstall_requested.emit)
        
        button_layout.addWidget(cancel_button)
        button_layout.addStretch()
        button_layout.addWidget(uninstall_button)
        
        layout.addLayout(button_layout)
    
    def _get_install_path(self):
        """Get the installation path from the registry."""
        try:
            reg_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\ANPE"
            reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path)
            install_path, _ = winreg.QueryValueEx(reg_key, "InstallLocation")
            winreg.CloseKey(reg_key)
            return install_path
        except Exception as e:
            # Fallback - get path from the uninstaller location
            return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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
        
        # Success icon or checkmark
        # For simplicity, just using a text label here
        icon_label = QLabel("✓")
        icon_label.setStyleSheet("font-size: 48pt; color: green;")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)
        
        # Completion message
        title_label = QLabel("Uninstallation Complete")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(title_label)
        
        message_label = QLabel(
            "ANPE has been successfully uninstalled from your computer."
        )
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(message_label)
        
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Close button
        close_button = QPushButton("Close")
        close_button.setStyleSheet("QPushButton { font-size: 14pt; padding: 10px; }")
        close_button.clicked.connect(self.close_requested.emit)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignCenter)


class UninstallWorker(QObject):
    """Worker thread to handle the actual uninstallation process."""
    status_update = pyqtSignal(str)
    log_update = pyqtSignal(str)
    progress_update = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal(bool)  # success
    
    def __init__(self):
        super().__init__()
        self.install_log_path = None
        self.install_path = None
        self.app_dir = None
        
    def run(self):
        """Main uninstallation process."""
        try:
            self.status_update.emit("Starting uninstallation...")
            self.log_update.emit("Preparing to uninstall ANPE...")
            
            # 1. Get installation info from registry
            self._get_install_info()
            
            # 2. Read the installation log file
            if not self._read_install_log():
                self.log_update.emit("WARNING: Install log not found. Proceeding with basic uninstallation.")
            
            # 3. Remove shortcuts
            self._remove_shortcuts()
            
            # 4. Remove registry entries
            self._remove_registry_entries()
            
            # 5. Remove files and directories
            self._remove_installed_files()
            
            # 6. Final cleanup
            self.status_update.emit("Finishing uninstallation...")
            self.log_update.emit("Uninstallation completed successfully.")
            
            # Signal completion
            self.finished.emit(True)
            
        except Exception as e:
            self.log_update.emit(f"ERROR: {str(e)}")
            import traceback
            self.log_update.emit(traceback.format_exc())
            self.status_update.emit("Uninstallation failed")
            self.finished.emit(False)
    
    def _get_install_info(self):
        """Get installation information from the registry."""
        try:
            self.log_update.emit("Reading installation information from registry...")
            reg_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\ANPE"
            reg_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, reg_path)
            self.install_path, _ = winreg.QueryValueEx(reg_key, "InstallLocation")
            winreg.CloseKey(reg_key)
            
            self.log_update.emit(f"Install path: {self.install_path}")
            self.install_log_path = os.path.join(self.install_path, "install_log.txt")
            
        except Exception as e:
            self.log_update.emit(f"Warning: Could not read registry information: {e}")
            # Fallback to current directory
            self.install_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.log_update.emit(f"Using fallback install path: {self.install_path}")
            self.install_log_path = os.path.join(self.install_path, "install_log.txt")
    
    def _read_install_log(self):
        """Read the installation log file."""
        if not os.path.exists(self.install_log_path):
            self.log_update.emit(f"Install log not found at: {self.install_log_path}")
            return False
        
        try:
            self.log_update.emit("Reading installation log file...")
            with open(self.install_log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("INSTALL_PATH="):
                        self.install_path = line.split("=", 1)[1]
                    elif line.startswith("APP_DIR="):
                        self.app_dir = line.split("=", 1)[1]
            
            self.log_update.emit(f"Using install path from log: {self.install_path}")
            return True
        except Exception as e:
            self.log_update.emit(f"Error reading install log: {e}")
            return False
    
    def _remove_shortcuts(self):
        """Remove desktop and start menu shortcuts."""
        self.status_update.emit("Removing shortcuts...")
        self.log_update.emit("Removing desktop and start menu shortcuts...")
        
        try:
            # Desktop shortcut
            desktop_dir = os.path.join(os.path.expanduser("~"), "Desktop")
            desktop_shortcut = os.path.join(desktop_dir, "ANPE.lnk")
            if os.path.exists(desktop_shortcut):
                os.remove(desktop_shortcut)
                self.log_update.emit(f"Removed desktop shortcut: {desktop_shortcut}")
            
            # Start menu shortcut
            start_menu_dir = os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs")
            start_menu_shortcut = os.path.join(start_menu_dir, "ANPE.lnk")
            if os.path.exists(start_menu_shortcut):
                os.remove(start_menu_shortcut)
                self.log_update.emit(f"Removed start menu shortcut: {start_menu_shortcut}")
        except Exception as e:
            self.log_update.emit(f"Warning: Error removing shortcuts: {e}")
    
    def _remove_registry_entries(self):
        """Remove registry entries."""
        self.status_update.emit("Removing registry entries...")
        self.log_update.emit("Removing registry entries...")
        
        try:
            reg_path = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\ANPE"
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, reg_path)
            self.log_update.emit("Removed registry entries successfully")
        except Exception as e:
            self.log_update.emit(f"Warning: Error removing registry entries: {e}")
    
    def _remove_installed_files(self):
        """Remove installed files and directories."""
        self.status_update.emit("Removing installed files...")
        
        # First, get a list of all files and directories to remove
        files_to_remove = []
        dirs_to_remove = []
        
        # Check if we have an install log to read from
        if os.path.exists(self.install_log_path):
            self.log_update.emit("Reading file list from installation log...")
            with open(self.install_log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("FILE="):
                        files_to_remove.append(line.split("=", 1)[1])
                    elif line.startswith("DIR="):
                        dirs_to_remove.append(line.split("=", 1)[1])
        else:
            # If no install log, remove the entire installation directory
            self.log_update.emit("No detailed file list found. Removing entire installation directory.")
            
            # Don't remove self (uninstaller) until the end
            self_path = os.path.dirname(os.path.abspath(__file__))
            
            # Collect all files and directories except the uninstaller
            for root, dirs, files in os.walk(self.install_path):
                if self_path in root:
                    continue  # Skip uninstaller directory
                
                for file in files:
                    file_path = os.path.join(root, file)
                    files_to_remove.append(file_path)
                
                # Add directories in reverse order (deepest first)
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    if self_path not in dir_path:
                        dirs_to_remove.append(dir_path)
        
        # Sort directories by depth (deepest first) to ensure proper removal
        dirs_to_remove.sort(key=lambda x: x.count(os.path.sep), reverse=True)
        
        # Remove files
        total_files = len(files_to_remove)
        for i, file_path in enumerate(files_to_remove):
            try:
                self.progress_update.emit(i + 1, total_files)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    self.log_update.emit(f"Removed file: {file_path}")
            except Exception as e:
                self.log_update.emit(f"Error removing file {file_path}: {e}")
        
        # Remove directories
        for dir_path in dirs_to_remove:
            try:
                if os.path.exists(dir_path):
                    # Try to remove the directory (will only work if empty)
                    try:
                        os.rmdir(dir_path)
                        self.log_update.emit(f"Removed directory: {dir_path}")
                    except OSError:
                        # If not empty, try to remove content
                        shutil.rmtree(dir_path, ignore_errors=True)
                        self.log_update.emit(f"Removed directory tree: {dir_path}")
            except Exception as e:
                self.log_update.emit(f"Error removing directory {dir_path}: {e}")
        
        # Finally, try to remove the installation directory itself
        try:
            if os.path.exists(self.install_path) and self.install_path != os.path.dirname(os.path.dirname(os.path.abspath(__file__))):
                shutil.rmtree(self.install_path, ignore_errors=True)
                self.log_update.emit(f"Removed installation directory: {self.install_path}")
        except Exception as e:
            self.log_update.emit(f"Warning: Could not remove installation directory: {e}")


class UninstallMainWindow(QMainWindow):
    """Main window for the ANPE uninstaller application with custom title bar."""
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        window_title = "ANPE Uninstaller"
        self.setWindowTitle(window_title)
        self.setFixedSize(600 + (BORDER_THICKNESS * 2), 450 + 35 + (BORDER_THICKNESS * 2))
        
        # --- Custom Window Frame --- 
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        # Make window background transparent to see rounded corners
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Main container widget
        self._main_container = QWidget(self)
        self.setCentralWidget(self._main_container)
        
        # Layout for the main container
        self._container_layout = QVBoxLayout(self._main_container)
        self._container_layout.setContentsMargins(BORDER_THICKNESS, BORDER_THICKNESS, BORDER_THICKNESS, BORDER_THICKNESS)
        self._container_layout.setSpacing(0)
        
        # Frame for content (background, border radius)
        self._main_frame = QFrame(self._main_container)
        self._main_frame.setObjectName("MainFrame")
        self._container_layout.addWidget(self._main_frame)
        
        # Layout inside the main frame
        self._frame_layout = QVBoxLayout(self._main_frame)
        self._frame_layout.setContentsMargins(0, 0, 0, 0)
        self._frame_layout.setSpacing(0)
        
        # Custom Title Bar
        self._title_bar = CustomTitleBar(window_title, self._main_frame)
        self._frame_layout.addWidget(self._title_bar)
        
        # Content Area (Stacked Widget)
        self.stacked_widget = QStackedWidget(self._main_frame)
        self._frame_layout.addWidget(self.stacked_widget)
        
        # Apply styling
        self.setStyleSheet(f"""
            #MainFrame {{ 
                background-color: white; 
                border: {BORDER_THICKNESS}px solid {PRIMARY_COLOR}; 
                border-radius: {BORDER_RADIUS}px; 
            }}
            #MainFrame QStackedWidget > QWidget {{ 
                background-color: white; 
                border-radius: 0px;
            }}
            CustomTitleBar {{ 
                background-color: #f0f0f0; 
                border-top-left-radius: {BORDER_RADIUS - BORDER_THICKNESS}px; 
                border-top-right-radius: {BORDER_RADIUS - BORDER_THICKNESS}px; 
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }}
        """)
        
        # Connect title bar signals
        self._title_bar.minimize_requested.connect(self.showMinimized)
        self._title_bar.close_requested.connect(self.close)
        
        # Store view instances
        self.welcome_view = None
        self.progress_view = None
        self.completion_view = None
        
        # Worker and thread management
        self._worker = None
        self._thread = None
        self._is_running = False
        
        self._create_views()
        self.stacked_widget.setCurrentIndex(VIEW_WELCOME)  # Show welcome view initially
        
        self.show()
    
    def _create_views(self):
        """Create and add view widgets to the stacked widget."""
        # Welcome View
        self.welcome_view = WelcomeUninstallWidget()
        self.stacked_widget.addWidget(self.welcome_view)
        self.welcome_view.uninstall_requested.connect(self._start_uninstallation)
        
        # Progress View
        self.progress_view = ProgressViewWidget("Uninstalling ANPE")
        self.stacked_widget.addWidget(self.progress_view)
        
        # Completion View
        self.completion_view = CompletionUninstallWidget()
        self.stacked_widget.addWidget(self.completion_view)
        self.completion_view.close_requested.connect(self.close)
    
    def _start_uninstallation(self):
        """Start the uninstallation process in a separate thread."""
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
        self.progress_view.set_progress_range(0, 100)
        self.stacked_widget.setCurrentIndex(VIEW_PROGRESS)
        
        # Create worker and thread
        self._worker = UninstallWorker()
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        
        # Connect signals
        self._worker.log_update.connect(self.progress_view.append_log)
        self._worker.status_update.connect(self.progress_view.update_status)
        self._worker.progress_update.connect(self._update_progress)
        self._worker.finished.connect(self._uninstall_finished)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        
        self._thread.start()
    
    def _update_progress(self, current, total):
        """Update the progress bar."""
        if total > 0:
            percentage = int((current / total) * 100)
            self.progress_view.set_progress_value(percentage)
    
    def _uninstall_finished(self, success):
        """Handle completion of the uninstallation."""
        self._is_running = False
        
        if success:
            # Show completion view
            self.stacked_widget.setCurrentIndex(VIEW_COMPLETION)
        else:
            # Show error message and stay on progress view
            QMessageBox.critical(
                self, "Uninstallation Failed",
                "There were errors during the uninstallation process.\n"
                "Please check the log for details."
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
                    self._thread.quit()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    window = UninstallMainWindow()
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 