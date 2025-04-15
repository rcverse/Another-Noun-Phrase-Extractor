import os
import sys
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextEdit, QDialogButtonBox, QLabel, QSizePolicy, 
    QHBoxLayout, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices, QPixmap

# Use relative import for utils
from ..utils import get_resource_path

class LicenseDialog(QDialog):
    """A dialog to display the application license (adapted for installer)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ANPE License Information")
        self._setup_ui()
        self.resize(700, 550)  # Make it reasonably sized

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)

        # --- Header with Logo ---
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        logo_path = get_resource_path("assets/app_icon_logo.png")
        
        # Check if the logo file exists
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio))
        else:
            # Create a text label as fallback
            logo_label.setText("ANPE")
            logo_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #0066b2;")
            
        header_layout.addWidget(logo_label)
        
        title_layout = QVBoxLayout()
        title_label = QLabel("ANPE License Information")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #0066b2;")
        title_layout.addWidget(title_label)
        
        subtitle_label = QLabel("ANPE is open-source software licensed under GPLv3")
        subtitle_label.setStyleSheet("font-size: 14px;")
        title_layout.addWidget(subtitle_label)
        
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # --- Description Section ---
        description = QLabel(
            "ANPE is free software that respects your freedom. It is licensed under the "
            "GNU General Public License version 3 (GPLv3), which guarantees your right to "
            "use, study, share, and modify the software."
        )
        description.setWordWrap(True)
        description.setStyleSheet("margin-top: 10px; margin-bottom: 10px;")
        layout.addWidget(description)
        
        # --- Link to full license ---
        link_label = QLabel("<a href='#'>View the full GPLv3 license text</a>")
        link_label.setOpenExternalLinks(False)
        link_label.linkActivated.connect(self._show_full_license)
        layout.addWidget(link_label)

        # --- License Summary Section ---
        summary_label = QLabel("Key Points of the GPLv3 License:")
        summary_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(summary_label)
        
        summary_text = QTextEdit()
        summary_text.setReadOnly(True)
        summary_text.setStyleSheet("background-color: #f8f8f8; border: 1px solid #e0e0e0;")
        summary_text.setHtml("""
            <ul>
                <li><b>Freedom to Use:</b> You can use ANPE for any purpose.</li>
                <li><b>Freedom to Study:</b> You can study how ANPE works and modify it.</li>
                <li><b>Freedom to Share:</b> You can share ANPE with others.</li>
                <li><b>Freedom to Modify:</b> You can distribute modified versions of ANPE.</li>
                <li><b>Copyleft:</b> Modified versions must also be free software.</li>
            </ul>
            <p>ANPE uses various other open-source components, each with their own licenses:</p>
            <ul>
                <li><b>PyQt6:</b> Available under GPL and commercial licenses</li>
                <li><b>spaCy:</b> MIT License</li>
                <li><b>Benepar:</b> Apache License 2.0</li>
                <li><b>NLTK:</b> Apache License 2.0</li>
            </ul>
        """)
        summary_text.setFixedHeight(200)
        layout.addWidget(summary_text)
        
        # --- OK Button ---
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)
        
    def _show_full_license(self):
        """Show the full license text in a separate scrollable area"""
        # Create and show the full license dialog
        full_license_dialog = QDialog(self)
        full_license_dialog.setWindowTitle("GNU General Public License v3")
        full_license_dialog.resize(750, 600)
        
        layout = QVBoxLayout(full_license_dialog)
        layout.setContentsMargins(15, 15, 15, 15)
        
        license_title = QLabel("GNU General Public License v3")
        license_title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(license_title)
        
        # Add text area for the license text
        license_text_edit = QTextEdit()
        license_text_edit.setReadOnly(True)
        
        # Try to load the license text
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
            
        license_text_edit.setText(license_text)
        license_text_edit.verticalScrollBar().setValue(0)
        layout.addWidget(license_text_edit)
        
        # Add close button
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        button_box.rejected.connect(full_license_dialog.close)
        layout.addWidget(button_box)
        
        full_license_dialog.exec()
