import sys
import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton, QCheckBox, QHBoxLayout,
    QSizePolicy, QFileDialog, QSpacerItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QMouseEvent

# Import the helper function and the dialog
from .license_dialog import LicenseDialog # Relative import for view
# Use relative import for utils
from ..utils import get_resource_path

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
    # Signal arguments: install_path (str), license_accepted (bool)
    setup_requested = pyqtSignal(str, bool)

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
        title_label.setStyleSheet("font-size: 18pt; font-weight: bold;") # Basic styling
        layout.addWidget(title_label)

        welcome_text = QLabel(
            "This wizard will guide you through setting up the ANPE application environment. "
            "It will install a dedicated Python environment and necessary components."
        )
        welcome_text.setWordWrap(True)
        welcome_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_text)

        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # --- Installation Path (Windows Specific) ---
        path_label = QLabel("Select Installation Location:")
        layout.addWidget(path_label)

        path_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        # Suggest a default path
        default_path = os.path.join(os.path.expanduser("~"), "ANPE")
        self.path_edit.setText(default_path)
        self.path_edit.textChanged.connect(self._validate_inputs) # Validate on change
        path_layout.addWidget(self.path_edit)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_directory)
        path_layout.addWidget(browse_button)
        layout.addLayout(path_layout)

        # --- License Agreement ---
        license_layout = QHBoxLayout()
        license_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.license_checkbox = QCheckBox("I agree to the terms of the ")
        self.license_checkbox.stateChanged.connect(self._validate_inputs) # Validate on change
        license_layout.addWidget(self.license_checkbox)

        license_link_label = ClickableLabel("<a href=\"#\">License Agreement</a>")
        license_link_label.setStyleSheet("text-decoration: none;") # Remove default underline if needed
        license_link_label.setOpenExternalLinks(False)
        license_link_label.clicked.connect(self._show_license_dialog)
        license_layout.addWidget(license_link_label)
        license_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        layout.addLayout(license_layout)

        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # --- Setup Button ---
        self.setup_button = QPushButton("Setup")
        self.setup_button.setEnabled(False) # Initially disabled
        self.setup_button.setObjectName("SetupButton") # For potential styling
        self.setup_button.setStyleSheet("QPushButton#SetupButton { font-size: 14pt; padding: 10px; }") # Bigger button
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
        """Enable the setup button only if path is non-empty and license is checked."""
        path_valid = bool(self.path_edit.text().strip())
        license_accepted = self.license_checkbox.isChecked()
        # TODO: Add path writability check
        self.setup_button.setEnabled(path_valid and license_accepted)

    def _on_setup_clicked(self):
        """Emit the setup_requested signal when the setup button is clicked."""
        install_path = self.path_edit.text().strip()
        license_accepted = self.license_checkbox.isChecked()
        # Basic validation again, although button should be disabled if invalid
        if install_path and license_accepted:
            # TODO: Add more robust validation (path exists, is writable?) before emitting
            print(f"Setup requested: Path='{install_path}', License Accepted={license_accepted}") # Debug print
            self.setup_requested.emit(install_path, license_accepted)

    def _show_license_dialog(self):
        """Show the license agreement dialog."""
        # Create dialog if it doesn't exist or reuse if already created (optional)
        if self._license_dialog is None:
            self._license_dialog = LicenseDialog(self) # Pass self as parent
        self._license_dialog.exec()

# Example usage (for testing the view directly)
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication
    # Need to adjust path for fallback get_resource_path if running directly
    # This is tricky, usually test main window which imports correctly
    # For standalone test, ensure assets is findable or skip logo load

    app = QApplication(sys.argv)
    welcome_view = WelcomeViewWidget()
    welcome_view.resize(500, 350) # Give it some size for standalone testing
    welcome_view.show()
    sys.exit(app.exec())
