"""
Dialog to display the GNU General Public License (GPL) for the ANPE GUI application.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextBrowser, QDialogButtonBox, QWidget
)
from PyQt6.QtGui import QIcon, QPixmap, QDesktopServices
from PyQt6.QtCore import Qt, QUrl
from anpe_gui.theme import PRIMARY_COLOR

class LicenseDialog(QDialog):
    """Dialog window to display the GPL license information."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ANPE License Information")
        self.setMinimumSize(700, 600)
        self.resize(700, 600)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the dialog UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header section
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 10)
        
        # Title
        title_label = QLabel("GNU General Public License (GPL) v3")
        title_label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {PRIMARY_COLOR};")
        header_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel(
            "ANPE is licensed under the GNU General Public License v3.0, a strong copyleft "
            "license that requires anyone who distributes the software or derivative works "
            "to make the source code available under the same license terms."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #555555;")
        header_layout.addWidget(desc_label)
        
        layout.addWidget(header_widget)
        
        # License text in a scrollable text browser
        self.text_browser = QTextBrowser()
        self.text_browser.setReadOnly(True)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setStyleSheet("""
            QTextBrowser {
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
                background-color: #fafafa;
                font-family: monospace;
                font-size: 10pt;
            }
        """)
        
        # Display license summary and full text
        license_html = self._get_license_html()
        self.text_browser.setHtml(license_html)
        
        layout.addWidget(self.text_browser, 1)  # Give the text browser stretch priority
        
        # Bottom button area
        button_layout = QHBoxLayout()
        button_layout.addStretch(1)
        
        # Add link to full license
        gpl_link_button = QPushButton("View Full GPL License Online")
        gpl_link_button.clicked.connect(self._open_gpl_website)
        gpl_link_button.setStyleSheet(f"""
            QPushButton {{
                color: {PRIMARY_COLOR};
                background-color: transparent;
                border: 1px solid {PRIMARY_COLOR};
                border-radius: 3px;
                padding: 5px 10px;
            }}
            QPushButton:hover {{
                background-color: #e0e0e0;
            }}
        """)
        button_layout.addWidget(gpl_link_button)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.setFixedWidth(80)
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)
        
    def _open_gpl_website(self):
        """Open the full GPL license text on the GNU website."""
        QDesktopServices.openUrl(QUrl("https://www.gnu.org/licenses/gpl-3.0.html"))
        
    def _get_license_html(self):
        """Return the license text/summary formatted as HTML."""
        return """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.5; }
                h2 { color: #005A9C; margin-top: 20px; }
                h3 { color: #444; margin-top: 15px; }
                ul { margin-top: 10px; }
                li { margin-bottom: 8px; }
                .section { margin-bottom: 20px; }
                .important { font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="section">
                <h2>GNU General Public License Version 3 (GPL-3.0)</h2>
                <p>This program is free software: you can redistribute it and/or modify
                it under the terms of the GNU General Public License as published by
                the Free Software Foundation, either version 3 of the License, or
                (at your option) any later version.</p>
                
                <p>This program is distributed in the hope that it will be useful,
                but WITHOUT ANY WARRANTY; without even the implied warranty of
                MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
                GNU General Public License for more details.</p>
            </div>
            
            <div class="section">
                <h3>Key Provisions:</h3>
                <ul>
                    <li><span class="important">Source Code Access:</span> Anyone who receives the software can access its source code.</li>
                    <li><span class="important">Modifications:</span> Anyone can modify the software and distribute their modifications.</li>
                    <li><span class="important">Same License:</span> Modified versions must be distributed under the same license.</li>
                    <li><span class="important">No Additional Restrictions:</span> You cannot impose further restrictions on recipients' exercise of the rights granted.</li>
                    <li><span class="important">Attribution:</span> You must retain copyright notices and give credit to the original authors.</li>
                </ul>
            </div>
            
            <div class="section">
                <h3>Third-Party Dependencies:</h3>
                <p>ANPE uses several open-source libraries and frameworks, each with their own licenses:</p>
                <ul>
                    <li><span class="important">PyQt6:</span> Available under GPL and commercial licenses</li>
                    <li><span class="important">Benepar:</span> Apache License 2.0</li>
                    <li><span class="important">spaCy:</span> MIT License</li>
                    <li><span class="important">NLTK:</span> Apache License 2.0</li>
                </ul>
                <p>These third-party components are distributed according to their original licenses.</p>
            </div>
            
            <div class="section">
                <h3>Copyright Notice:</h3>
                <p>Copyright © 2025 Richard Chen (@rcverse)</p>
                <p>For more information about the GNU General Public License, visit:<br>
                <a href="https://www.gnu.org/licenses/gpl-3.0.html">https://www.gnu.org/licenses/gpl-3.0.html</a></p>
            </div>
        </body>
        </html>
        """ 