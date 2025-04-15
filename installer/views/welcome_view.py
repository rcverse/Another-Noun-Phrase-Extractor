import sys
import os
import ctypes
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QHBoxLayout,
    QSizePolicy, QFileDialog, QSpacerItem, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QMouseEvent

# Import the helper function and the dialog
from .license_dialog import LicenseDialog # Relative import for view
# Use relative import for utils
from ..utils import get_resource_path
from ..styles import PRIMARY_BUTTON_STYLE, BROWSE_BUTTON_STYLE, TITLE_LABEL_STYLE, INFO_LABEL_STYLE, LINK_LABEL_STYLE

# Subclass QLabel to make it clickable for the license link
class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        else:
            super().mousePressEvent(event)

class WelcomeViewWidget(QWidget):
    """Widget for the welcome screen of the setup wizard (Windows version)."""
    # Signal arguments: install_path (str)
    setup_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        """Initialize the welcome view."""
        super().__init__(parent)
        self._license_dialog = None # Hold reference to dialog
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20) # Add some padding
        layout.setSpacing(15)

        # --- Logo ---
        logo_label = QLabel()
        # Use the imported get_resource_path directly
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
                    print(f"Warning: Failed to load logo pixmap from {logo_path}", file=sys.stderr)
            else:
                logo_label.setText("[Logo Not Found]")
                logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                print(f"Warning: Logo file not found at {logo_path}", file=sys.stderr)
        except Exception as e:
             logo_label.setText("[Logo Error]")
             logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
             print(f"Error resolving logo path: {e}", file=sys.stderr)

        logo_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout.addWidget(logo_label)

        # --- Welcome Text ---
        title_label = QLabel("Welcome to ANPE Setup")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(TITLE_LABEL_STYLE) # Basic styling
        layout.addWidget(title_label)

        layout.addSpacerItem(QSpacerItem(20, 30, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # --- Installation Path (Windows Specific) ---
        path_label = QLabel("Install ANPE GUI to:")
        layout.addWidget(path_label)

        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        
        # Automatically determine appropriate installation path based on admin privileges
        is_admin = self._is_admin()
        if is_admin:
            default_path = os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), "ANPE")
        else:
            default_path = os.path.join(os.environ.get('LOCALAPPDATA', os.path.expanduser("~\\AppData\\Local")), "ANPE")
        
        self.path_edit.setText(default_path)
        self.path_edit.textChanged.connect(self._validate_inputs) # Validate on change
        path_layout.addWidget(self.path_edit)

        browse_button = QPushButton("Browse...")
        browse_button.setStyleSheet(BROWSE_BUTTON_STYLE)
        browse_button.clicked.connect(self._browse_directory)
        path_layout.addWidget(browse_button)
        layout.addLayout(path_layout)
        
        # Add explanation about installation location
        path_info = QLabel("Please make sure you have enough disk space to install the application")
        path_info.setStyleSheet(INFO_LABEL_STYLE)
        layout.addWidget(path_info)

        # --- License Agreement ---
        license_layout = QHBoxLayout()
        license_layout.addStretch(1)  # Add stretch at the beginning to center
        license_text = QLabel("ANPE is open-source software licensed under")
        license_text.setStyleSheet(INFO_LABEL_STYLE)
        license_layout.addWidget(license_text)

        license_link_label = ClickableLabel("GNU GPL v3")  # Remove the HTML link styling
        license_link_label.setStyleSheet(LINK_LABEL_STYLE)  # Blue color matching theme
        license_link_label.setOpenExternalLinks(False)
        license_link_label.clicked.connect(self._show_license_dialog)
        license_layout.addWidget(license_link_label)
        license_layout.addStretch(1)  # Add stretch at the end to center
        
        layout.addLayout(license_layout)

        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # --- Setup Button ---
        self.setup_button = QPushButton("Setup")
        self.setup_button.setObjectName("SetupButton")  # For potential styling
        # Update button style to use centralized style
        self.setup_button.setStyleSheet(PRIMARY_BUTTON_STYLE)
        self.setup_button.clicked.connect(self._on_setup_clicked)
        layout.addWidget(self.setup_button, alignment=Qt.AlignmentFlag.AlignCenter)

    def _browse_directory(self):
        """Open a directory selection dialog."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Installation Directory",
            self.path_edit.text() # Start browsing from current path
        )
        if directory: # If user selected a directory (didn't cancel)
            self.path_edit.setText(directory)
            self._validate_inputs() # Re-validate after path change

    def _validate_inputs(self):
        """Enable the setup button only if path is non-empty and valid."""
        install_path = self.path_edit.text().strip()
        
        # Basic validation - path must not be empty
        path_valid = bool(install_path)
        
        # Check if parent directory exists or can be created
        if path_valid:
            try:
                parent_dir = os.path.dirname(os.path.abspath(install_path))
                
                # Check if parent directory exists
                if not os.path.exists(parent_dir):
                    try:
                        # Try temporarily creating it to check permissions
                        os.makedirs(parent_dir, exist_ok=True)
                        os.rmdir(parent_dir)  # Clean up if we created it just for testing
                    except (PermissionError, OSError):
                        path_valid = False
                
                # If target exists, check if it's writable
                if os.path.exists(install_path):
                    if not os.access(install_path, os.W_OK):
                        path_valid = False
                else:
                    # Check if we can write to the parent directory
                    if not os.access(parent_dir, os.W_OK):
                        path_valid = False
            except Exception:
                path_valid = False
                
        self.setup_button.setEnabled(path_valid)

    def _on_setup_clicked(self):
        """Emit the setup_requested signal when the setup button is clicked."""
        install_path = self.path_edit.text().strip()
        
        # Final validation before emitting the signal
        try:
            # Convert to absolute path
            abs_path = os.path.abspath(install_path)
            parent_dir = os.path.dirname(abs_path)
            
            # Ensure parent directory exists or can be created
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
                
            # Check write permissions by attempting to create a test file
            if os.path.exists(abs_path):
                test_dir = abs_path
            else:
                test_dir = parent_dir
                
            test_file = os.path.join(test_dir, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            
            # If we got here, the path is valid and writable
            print(f"Setup requested: Path='{abs_path}'")
            self.setup_requested.emit(abs_path)
        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Invalid Installation Path",
                f"Cannot install to the selected location:\n{install_path}\n\nError: {str(e)}"
            )

    def _show_license_dialog(self):
        """Show the license agreement dialog."""
        # Create dialog if it doesn't exist or reuse if already created (optional)
        if self._license_dialog is None:
            self._license_dialog = LicenseDialog(self) # Pass self as parent
        self._license_dialog.exec()

    def _is_admin(self):
        """Check if the current user has administrator privileges on Windows."""
        try:
            return ctypes.windll.shell32.IsUserAnAdmin() != 0
        except Exception:
            return False

