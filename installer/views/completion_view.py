from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QCheckBox, QPushButton, QSizePolicy, QSpacerItem, QHBoxLayout, QFrame,
    QTextEdit, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QFont, QColor, QTextCursor
import os
import re
import datetime

from ..utils import get_resource_path
from ..styles import PRIMARY_BUTTON_STYLE, TITLE_LABEL_STYLE, INFO_LABEL_STYLE, SECONDARY_BUTTON_STYLE, LOG_TEXT_AREA_STYLE

class CompletionViewWidget(QWidget):
    """Widget for the completion screen of the setup wizard (Windows version)."""
    # Signals emitted when the final button is clicked
    shortcut_requested = pyqtSignal(bool)
    launch_requested = pyqtSignal(bool)
    close_requested = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the completion view."""
        super().__init__(parent)
        self._setup_ui()
        self._log_content = ""  # Store log content for export
        self._error_message = ""  # Store extracted error message
        # Set initial state (e.g., assuming success until told otherwise)
        # self.set_success_state(True)

    def _setup_ui(self):
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(25, 15, 25, 15)  # Reduced margins
        layout.setSpacing(10)  # Reduced spacing between elements

        # --- Header with Logo and Title ---
        header_layout = QHBoxLayout()
        
        # Logo (reduced size for better space usage)
        logo_label = QLabel()
        logo_path = get_resource_path("assets/app_icon_logo.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            pixmap = pixmap.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            logo_label.setPixmap(pixmap)
        header_layout.addWidget(logo_label)
        
        # Status Title
        self.status_title = QLabel("Setup Status") # Placeholder text
        self.status_title.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.status_title.setStyleSheet(TITLE_LABEL_STYLE)
        header_layout.addWidget(self.status_title, 1)  # Stretch factor 1
        
        layout.addLayout(header_layout)
        
        # Add separator line with better styling
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #e0e0e0; border: none; height: 1px;")
        separator.setFixedHeight(1)
        layout.addWidget(separator)
        
        # Only add a small spacing after separator
        layout.addSpacing(10)

        # --- Info Text ---
        self.info_text = QLabel("Details about the setup process outcome...") # Placeholder
        self.info_text.setWordWrap(True)
        self.info_text.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.info_text.setStyleSheet("font-size: 14px; color: #333333;")
        layout.addWidget(self.info_text)
        
        # --- Error Highlight Label (only shown on failure) ---
        self.error_highlight = QLabel()
        self.error_highlight.setWordWrap(True)
        self.error_highlight.setStyleSheet("font-size: 13px; color: #DD3333; background-color: #FFEEEE; padding: 8px; border-radius: 4px;")
        self.error_highlight.setVisible(False)
        layout.addWidget(self.error_highlight)
        
        # --- Log Viewer (only shown on failure) ---
        self.log_container = QWidget()
        log_layout = QVBoxLayout(self.log_container)
        log_layout.setContentsMargins(0, 8, 0, 0)
        log_layout.setSpacing(5)
        
        # Add log header and Export button on the same line
        log_header_layout = QHBoxLayout()
        
        log_header = QLabel("Setup Log:")
        log_header.setStyleSheet("font-weight: bold; color: #555555;")
        log_header_layout.addWidget(log_header)
        
        log_header_layout.addStretch()
        
        # Export log button - moved to the header line
        export_button = QPushButton("Export Log")
        export_button.setStyleSheet("""
            QPushButton {
                background-color: #005A9C;
                color: white;
                border: none;
                border-radius: 3px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #0066b2;
            }
            QPushButton:pressed {
                background-color: #005299;
            }
        """)
        export_button.setFixedSize(80, 24)  # Make button smaller
        export_button.clicked.connect(self._export_log)
        log_header_layout.addWidget(export_button)
        
        log_layout.addLayout(log_header_layout)
        
        self.log_viewer = QTextEdit()
        self.log_viewer.setReadOnly(True)
        self.log_viewer.setFont(QFont("Consolas", 9))
        self.log_viewer.setStyleSheet(LOG_TEXT_AREA_STYLE)
        self.log_viewer.setMinimumHeight(100)  # Reduced height
        self.log_viewer.setMaximumHeight(150)  # Added maximum height
        log_layout.addWidget(self.log_viewer)
        
        layout.addWidget(self.log_container)
        self.log_container.setVisible(False)
        
        # --- Reporting Instructions (only shown on failure) ---
        self.report_container = QWidget()
        report_layout = QVBoxLayout(self.report_container)
        report_layout.setContentsMargins(0, 5, 0, 5)
        report_layout.setSpacing(3)  # Reduced spacing
        
        report_label = QLabel("Report this issue:")
        report_label.setStyleSheet("font-weight: bold; color: #555555;")
        report_layout.addWidget(report_label)
        
        # Put reporting links and note on the same line
        links_layout = QHBoxLayout()
        
        github_link = QLabel('<a href="https://github.com/rcverse/Another-Noun-Phrase-Extractor/issues">GitHub Issue</a>')
        github_link.setOpenExternalLinks(True)
        github_link.setStyleSheet("font-size: 13px;")
        links_layout.addWidget(github_link)
        
        links_layout.addWidget(QLabel("|"))
        
        email_link = QLabel('<a href="mailto:rcverse6@gmail.com">Email</a>')
        email_link.setOpenExternalLinks(True)
        email_link.setStyleSheet("font-size: 13px;")
        links_layout.addWidget(email_link)
        
        links_layout.addSpacing(15)  # Add some spacing before the note
        
        # Add the note on the same line
        report_note = QLabel("Please include the exported log file with your report.")
        report_note.setStyleSheet("font-size: 12px; font-style: italic; color: #666666;")
        links_layout.addWidget(report_note)
        
        links_layout.addStretch()
        report_layout.addLayout(links_layout)
        
        layout.addWidget(self.report_container)
        self.report_container.setVisible(False)

        # Add flexible space
        layout.addStretch(1)

        # --- Options (Windows Specific, shown on success) ---
        options_layout = QVBoxLayout()
        options_layout.setSpacing(8)
        
        self.shortcut_checkbox = QCheckBox("Create Desktop/Start Menu Shortcut")
        self.shortcut_checkbox.setChecked(True) # Default to checked
        self.shortcut_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        options_layout.addWidget(self.shortcut_checkbox)

        self.launch_checkbox = QCheckBox("Launch ANPE now")
        self.launch_checkbox.setChecked(True) # Default to checked
        self.launch_checkbox.setStyleSheet("""
            QCheckBox {
                font-size: 14px;
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
        """)
        options_layout.addWidget(self.launch_checkbox)
        
        # Add options with spacing
        layout.addLayout(options_layout)
        layout.addSpacing(10)

        # --- Complete/Close Button ---
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.complete_button = QPushButton("Complete") # Text changes based on state
        self.complete_button.setObjectName("CompleteButton")
        self.complete_button.setStyleSheet(PRIMARY_BUTTON_STYLE)
        self.complete_button.setFixedWidth(120)  # Set a fixed width
        self.complete_button.clicked.connect(self._handle_complete)
        
        button_layout.addWidget(self.complete_button)
        button_layout.addStretch()
        
        layout.addLayout(button_layout)
        layout.addSpacing(5)  # Small margin at the bottom

        # Hide options initially, shown by set_success_state
        self.shortcut_checkbox.setVisible(False)
        self.launch_checkbox.setVisible(False)

    def set_success_state(self, success: bool, log_content: str = ""):
        """Configure the view based on the setup outcome."""
        self._log_content = log_content
        
        if success:
            self.status_title.setText("Setup Complete!")
            self.status_title.setStyleSheet(TITLE_LABEL_STYLE)
            self.info_text.setText(
                "ANPE has been successfully installed and is ready to use.\n\n"
                "Feedbacks are welcome! Please report any issues or suggestions to rcverse6@gmail.com."
            )
            self.shortcut_checkbox.setVisible(True)
            self.launch_checkbox.setVisible(True)
            self.complete_button.setText("Complete")
            
            # Hide error components
            self.log_container.setVisible(False)
            self.report_container.setVisible(False)
            self.error_highlight.setVisible(False)
        else:
            self.status_title.setText("Setup Failed")
            self.status_title.setStyleSheet("font-size: 24px; font-weight: bold; color: #DD3333;")
            self.info_text.setText("An error occurred during the installation process.")
            self.shortcut_checkbox.setVisible(False)
            self.launch_checkbox.setVisible(False)
            self.complete_button.setText("Close")
            
            # Extract and display the error
            self._error_message = self._extract_error_from_log(log_content)
            if self._error_message:
                self.error_highlight.setText(f"Error details: {self._error_message}")
                self.error_highlight.setVisible(True)
            
            # Show log components with size constraints
            if log_content:
                # Trim log content if too long
                if len(log_content) > 5000:
                    shortened_log = log_content[:2000] + "\n...[log trimmed]...\n" + log_content[-2000:]
                    self.log_viewer.setText(shortened_log)
                else:
                    self.log_viewer.setText(log_content)
                self.log_viewer.moveCursor(QTextCursor.MoveOperation.Start)
                self.log_container.setVisible(True)
            
            # Show reporting instructions
            self.report_container.setVisible(True)

    def _extract_error_from_log(self, log_content: str) -> str:
        """Extract the most relevant error message from log content."""
        if not log_content:
            return ""
            
        # Look for common error patterns
        error_patterns = [
            # Look for lines with ERROR, CRITICAL, Failed, etc.
            r'ERROR: (.*?)(?:\n|$)',
            r'CRITICAL: (.*?)(?:\n|$)',
            r'CRITICAL ERROR: (.*?)(?:\n|$)',
            r'Error: (.*?)(?:\n|$)',
            r'Failed: (.*?)(?:\n|$)',
            r'Exception: (.*?)(?:\n|$)',
            # Look for Python exceptions
            r'([A-Za-z]+Error: .*?)(?:\n|$)',
            r'([A-Za-z]+Exception: .*?)(?:\n|$)',
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, log_content)
            if matches:
                # Return the first significant error message
                for match in matches:
                    if len(match) > 5:  # Ensure it's not just a short message
                        return match.strip()
        
        return "No specific error details found. Please check the log for more information."

    def _export_log(self):
        """Export the log to a file."""
        if not self._log_content:
            QMessageBox.warning(self, "Export Failed", "No log content available to export.")
            return
            
        # Suggest a filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"anpe_setup_log_{timestamp}.txt"
        
        # Open save dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Log File",
            default_filename,
            "Text Files (*.txt);;All Files (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self._log_content)
                QMessageBox.information(
                    self, 
                    "Log Exported", 
                    f"Log file has been exported to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export log file: {str(e)}"
                )

    def _handle_complete(self):
        """Emit signals based on checkbox states and request closing."""
        # Only emit requests if setup was successful (when checkboxes are visible)
        if self.shortcut_checkbox.isVisible():
            self.shortcut_requested.emit(self.shortcut_checkbox.isChecked())
        if self.launch_checkbox.isVisible():
            self.launch_requested.emit(self.launch_checkbox.isChecked())

        # Always emit close request
        self.close_requested.emit()

# Example usage (for testing the view directly)
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication, QPushButton, QHBoxLayout

    app = QApplication(sys.argv)
    container = QWidget()
    main_layout = QVBoxLayout(container)

    completion_view = CompletionViewWidget()
    main_layout.addWidget(completion_view)

    # Add buttons to test state changes
    button_layout = QHBoxLayout()
    success_button = QPushButton("Set Success")
    fail_button = QPushButton("Set Failure")
    button_layout.addWidget(success_button)
    button_layout.addWidget(fail_button)
    main_layout.addLayout(button_layout)

    # Add a test log for the failure case
    test_log = """
2023-10-15 14:49:32,123 - anpe.setup_models - INFO - Starting model setup
2023-10-15 14:49:35,246 - anpe.setup_models - INFO - Checking spaCy model
2023-10-15 14:49:40,357 - anpe.setup_models - INFO - Downloading spaCy model: en_core_web_md
2023-10-15 14:49:50,489 - anpe.setup_models - ERROR - Failed to download model: Connection error
2023-10-15 14:49:51,123 - anpe.setup_models - CRITICAL - ConnectionError: Failed to establish a connection to the model repository
2023-10-15 14:49:52,246 - anpe.setup_models - INFO - Model setup process failed
    """

    success_button.clicked.connect(lambda: completion_view.set_success_state(True))
    fail_button.clicked.connect(lambda: completion_view.set_success_state(False, test_log))

    # Connect signals to print output
    completion_view.shortcut_requested.connect(lambda checked: print(f"Shortcut requested: {checked}"))
    completion_view.launch_requested.connect(lambda checked: print(f"Launch requested: {checked}"))
    completion_view.close_requested.connect(lambda: print("Close requested"))

    container.resize(500, 300)
    container.show()
    completion_view.set_success_state(True) # Start in success state for demo

    sys.exit(app.exec())
