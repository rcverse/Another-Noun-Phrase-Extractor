"""
Custom QDialog to display help content from a Markdown file
using a QTextBrowser for better formatting and scrolling.
"""

import logging
from pathlib import Path
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextBrowser, QPushButton, QDialogButtonBox, QLabel, QWidget, QHBoxLayout
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices # For opening URLs
from PyQt6.QtWidgets import QMessageBox # For About box
from anpe_gui.theme import PRIMARY_COLOR  # Import theme colors

class HelpDialog(QDialog):
    def __init__(self, help_file_path: Path, app_version: str, parent=None):
        super().__init__(parent)
        self.help_file_path = help_file_path
        self.app_version = app_version
        self.setWindowTitle("ANPE GUI Help")
        self.setMinimumSize(900, 600)  # Made wider
        self.resize(1000, 700)  # Default size larger than minimum

        self.setup_ui()
        self.load_help_content()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)  # Increased margins

        # Header container with left-aligned title
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        
        # Title label (left-aligned)
        title_label = QLabel("ANPE GUI Help")
        title_label.setStyleSheet(f"""
            QLabel {{
                color: {PRIMARY_COLOR};
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                border-bottom: 2px solid {PRIMARY_COLOR};
            }}
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()  # Push title to the left
        layout.addWidget(header)

        self.text_browser = QTextBrowser()
        self.text_browser.setReadOnly(True)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setStyleSheet("""
            QTextBrowser {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 25px;
                font-size: 13px;
            }
            
            /* Increase line spacing in the markdown content */
            QTextBrowser { qproperty-lineWrapMode: 'WidgetWidth'; }
            QTextBrowser { line-height: 200%; }
            
            /* Style the actual content */
            QTextBrowser p {
                margin: 16px 0;
                line-height: 1.8;
            }
            QTextBrowser h1 {
                font-size: 24px;
                margin: 28px 0 20px 0;
                line-height: 1.4;
            }
            QTextBrowser h2 {
                font-size: 20px;
                margin: 24px 0 16px 0;
                line-height: 1.4;
            }
            QTextBrowser h3 {
                font-size: 16px;
                margin: 20px 0 12px 0;
                line-height: 1.4;
            }
            QTextBrowser ul, QTextBrowser ol {
                margin: 16px 0;
                padding-left: 20px;
            }
            QTextBrowser li {
                margin: 10px 0;
                line-height: 1.6;
            }
            QTextBrowser li > ul, QTextBrowser li > ol {
                margin: 8px 0;
            }
        """)
        layout.addWidget(self.text_browser)

        # Button container with right-aligned buttons
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(10)
        
        # Create buttons
        self.project_page_button = QPushButton("Project Page")
        self.about_button = QPushButton("About")
        self.ok_button = QPushButton("OK")
        
        # Add buttons to layout (right-aligned)
        button_layout.addStretch()  # Push buttons to the right
        button_layout.addWidget(self.project_page_button)
        button_layout.addWidget(self.about_button)
        button_layout.addWidget(self.ok_button)
        
        # Connect signals
        self.ok_button.clicked.connect(self.accept)
        self.project_page_button.clicked.connect(self.go_to_project_page)
        self.about_button.clicked.connect(self.show_about_info)
        
        layout.addWidget(button_container)

    def load_help_content(self):
        """Load and display the help content from the markdown file."""
        try:
            with open(self.help_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                self.text_browser.setMarkdown(content)
        except Exception as e:
            self.text_browser.setPlainText(f"Error loading help content: {str(e)}")

    def handle_anchor_click(self, url):
        """Scrolls to the anchor if it's an internal link."""
        # Simple check if it's an internal anchor link
        if url.scheme() == '' and url.hasFragment():
            self.text_browser.scrollToAnchor(url.fragment())
        else:
            # For external links, let the default behavior handle it (setOpenExternalLinks)
            pass # Or could explicitly open using QDesktopServices.openUrl(url)

    # --- Custom Button Handlers ---

    def go_to_project_page(self):
        """Opens the project's GitHub page in the default web browser."""
        project_url_string = "https://github.com/rcverse/Another-Noun-Phrase-Extractor"
        # Convert the string URL to a QUrl object
        project_qurl = QUrl(project_url_string)
        QDesktopServices.openUrl(project_qurl)

    def show_about_info(self):
        """Displays enhanced About information in a styled message box."""
        about_text = f"""
        <div style='padding: 25px; font-size: 14px;'>
            <h2 style='color: #2c3e50; margin-top: 0; font-size: 24px;'>ANPE GUI</h2>
            <p style='color: #7f8c8d; font-style: italic; font-size: 16px;'>Another Noun Phrase Extractor</p>
            
            <p style='color: #34495e; font-size: 14px;'><b>Version:</b> {self.app_version}</p>
            
            <div style='margin: 20px 0;'>
                <h3 style='color: #2c3e50; margin-top: 0; font-size: 18px;'>Description</h3>
                <p style='color: #34495e; font-size: 14px;'>
                    ANPE GUI is a user-friendly graphical interface designed to simplify the extraction of noun phrases from text. 
                    It provides an accessible way to perform NP extraction without requiring programming knowledge.
                </p>
            </div>
            
            <div style='margin: 20px 0;'>
                <h3 style='color: #2c3e50; margin-top: 0; font-size: 18px;'>Built with</h3>
                <ul style='color: #34495e; font-size: 14px; margin-left: 20px;'>
                    <li><b>Benepar</b> - Berkeley Neural Parser, the core constituency parsing engine</li>
                    <li><b>spaCy</b> - Industrial-strength Natural Language Processing</li>
                    <li><b>NLTK</b> - Natural Language Toolkit for text processing</li>
                </ul>
            </div>
            
            <div style='margin: 20px 0;'>
                <h3 style='color: #2c3e50; margin-top: 0; font-size: 18px;'>Acknowledgments</h3>
                <p style='color: #34495e; font-size: 14px;'>
                    ANPE is essentially a specialized wrapper around the Berkeley Neural Parser (Benepar), 
                    focusing specifically on noun phrase extraction with an intuitive graphical interface. 
                    We are grateful to the developers of Benepar, spaCy, and NLTK for their excellent work 
                    that makes this application possible.
                </p>
            </div>
            
            <div style='margin: 20px 0;'>
                <p style='color: #34495e; margin: 0; font-size: 14px;'>
                    <b>Author:</b> Richard Chen (@rcverse)<br>
                    <b>License:</b> MIT License
                </p>
            </div>
            
            <div style='margin-top: 20px;'>
                <a href='https://github.com/rcverse/Another-Noun-Phrase-Extractor' 
                   style='color: #0d6efd; text-decoration: none; font-weight: bold; font-size: 14px;'>
                    Visit Project Page
                </a>
            </div>
        </div>
        """
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("About ANPE GUI")
        msg_box.setText(about_text)
        msg_box.exec() 