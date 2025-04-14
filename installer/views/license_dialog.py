import os
import sys
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox, QLabel, QSizePolicy
)
from PyQt6.QtCore import Qt

# Use relative import for utils
from ..utils import get_resource_path

class LicenseDialog(QDialog):
    """A dialog to display the application license (adapted for installer)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("License Agreement (GPLv3)")
        self._setup_ui()
        self.resize(600, 500) # Make it reasonably sized

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)

        # --- Title Label ---
        title_label = QLabel("ANPE Setup - License Agreement")
        # Using standard link color blue instead of theme dependency
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold; color: #007BFF;")
        layout.addWidget(title_label)

        # --- Text Area for License ---
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.text_edit)

        # --- Load License Text ---
        license_text = "Error: Could not resolve license path."
        try:
            license_path = get_resource_path('assets/LICENSE.installer.txt')
            if os.path.exists(license_path):
                try:
                    with open(license_path, 'r', encoding='utf-8') as f:
                        license_text = f.read()
                except Exception as e:
                    license_text = f"Error loading license file: {e}"
                    print(f"Error loading license file {license_path}: {e}", file=sys.stderr)
            else:
                license_text = f"License file not found at expected location: {license_path}"
                print(f"Warning: License file not found at {license_path}", file=sys.stderr)
        except Exception as e:
            print(f"Error resolving license path with get_resource_path: {e}", file=sys.stderr)

        self.text_edit.setText(license_text)
        self.text_edit.verticalScrollBar().setValue(0) # Scroll to top

        # --- OK Button ---
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)

# Example usage (for testing the dialog directly)
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    dialog = LicenseDialog()
    dialog.exec()
