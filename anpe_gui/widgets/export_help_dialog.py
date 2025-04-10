"""
Dialog to explain the different export formats available in ANPE.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt
# Import theme colors if available, otherwise use fallbacks
try:
    from ..theme import PRIMARY_COLOR, BORDER_COLOR, TEXT_COLOR
except ImportError:
    PRIMARY_COLOR = "#007acc"  # Fallback primary color
    BORDER_COLOR = "#e0e0e0"   # Fallback border color
    TEXT_COLOR = "#333333"     # Fallback text color

class ExportHelpDialog(QDialog):
    """A simple dialog explaining the TXT, CSV, and JSON export formats."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Information")
        self.setMinimumWidth(550) # Slightly wider for better spacing
        # self.setStyleSheet(get_stylesheet()) # Apply global styles if needed

        layout = QVBoxLayout(self)
        layout.setSpacing(15) # Increased base spacing
        layout.setContentsMargins(20, 20, 20, 20) # Increased margins

        # --- Filename Structure Section ---
        filename_title = QLabel("<b>Export Filename Structure</b>")
        filename_title.setStyleSheet("font-size: 12pt; margin-bottom: 5px;") # Make title stand out
        layout.addWidget(filename_title)

        filename_label = QLabel()
        filename_label.setTextFormat(Qt.TextFormat.RichText)
        # Use slightly more padding/margin in HTML for list items
        filename_label.setText(
            f"""
            <div style="color: {TEXT_COLOR}; line-height: 1.4;">
            Exported files are named automatically to ensure uniqueness and provide context:
            <ul style="margin-left: 0px; padding-left: 20px; margin-top: 8px; margin-bottom: 8px;">
                <li><b>Batch Export (Multiple Files):</b><br>
                    <code style='background-color: #f0f0f0; padding: 1px 3px; border-radius: 3px;'>[<i>prefix</i>_]<i>original_filename</i>_anpe_results_<i>YYYYMMDD_HHMMSS</i>.<i>format</i></code>
                </li>
                <li style="margin-top: 5px;"><b>Single Export (Text Input):</b><br>
                    <code style='background-color: #f0f0f0; padding: 1px 3px; border-radius: 3px;'>[<i>prefix</i>_]anpe_text_results_<i>YYYYMMDD_HHMMSS</i>.<i>format</i></code>
                </li>
            </ul>
            Where:
            <ul style="margin-left: 0px; padding-left: 20px; margin-top: 5px; margin-bottom: 0px;">
                <li><code>[<i>prefix</i>_]</code> is the optional prefix you enter.</li>
                <li><code><i>original_filename</i></code> is the name of the input file (without extension).</li>
                <li><code><i>YYYYMMDD_HHMMSS</i></code> is the timestamp of the export.</li>
                <li><code><i>format</i></code> is the selected format (txt, csv, json).</li>
            </ul>
            </div>
            """
        )
        filename_label.setWordWrap(True)
        layout.addWidget(filename_label)

        # Separator - Lighter and thinner
        separator0 = QFrame()
        separator0.setFrameShape(QFrame.Shape.HLine)
        separator0.setStyleSheet(f"border-top: 1px solid {BORDER_COLOR}; margin-top: 5px; margin-bottom: 5px;")
        layout.addWidget(separator0)

        # --- Format Section Title ---
        format_title_label = QLabel("<b>Export Formats</b>")
        format_title_label.setStyleSheet("font-size: 12pt; margin-bottom: 5px;") # Make title stand out
        layout.addWidget(format_title_label)

        # --- TXT Format ---
        txt_label = QLabel()
        txt_label.setTextFormat(Qt.TextFormat.RichText)
        txt_label.setText(
             f"""
             <div style="color: {TEXT_COLOR}; line-height: 1.4;">
             <b style="font-size: 10pt;">TXT Format (.txt)</b><br>
             Outputs noun phrases in a human-readable, plain text format. 
             Nested phrases are shown using indentation.<br><br>
             <b style="color: {PRIMARY_COLOR};">Recommendation:</b> Best for quick reading, simple lists, or when you 
             just need the plain text of the phrases, especially top-level ones.
             </div>
             """
        )
        txt_label.setWordWrap(True)
        layout.addWidget(txt_label)

        # Separator (thin)
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setStyleSheet(f"border-top: 1px solid {BORDER_COLOR}; margin-top: 5px; margin-bottom: 5px;")
        layout.addWidget(separator1)

        # --- CSV Format ---
        csv_label = QLabel()
        csv_label.setTextFormat(Qt.TextFormat.RichText)
        csv_label.setText(
             f"""
             <div style="color: {TEXT_COLOR}; line-height: 1.4;">
             <b style="font-size: 10pt;">CSV Format (.csv)</b><br>
             Outputs results as Comma-Separated Values, suitable for spreadsheets (like Excel). 
             Provides a flat structure where each row is a noun phrase. 
             Hierarchy is represented using <code>ID</code> and <code>Parent_ID</code> columns. 
             Includes columns for <code>Level</code>, <code>Length</code>, and <code>Structures</code> if metadata is enabled.<br><br>
             <b style="color: {PRIMARY_COLOR};">Recommendation:</b> Best for importing into spreadsheets or other data analysis 
             tools that work well with tabular data.
             </div>
             """
        )
        csv_label.setWordWrap(True)
        layout.addWidget(csv_label)

        # Separator (thin)
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet(f"border-top: 1px solid {BORDER_COLOR}; margin-top: 5px; margin-bottom: 5px;")
        layout.addWidget(separator2)

        # --- JSON Format ---
        json_label = QLabel()
        json_label.setTextFormat(Qt.TextFormat.RichText)
        json_label.setText(
             f"""
             <div style="color: {TEXT_COLOR}; line-height: 1.4;">
             <b style="font-size: 10pt;">JSON Format (.json)</b><br>
             Outputs results in JavaScript Object Notation. Maintains the full hierarchical 
             structure using nested objects. Includes all metadata.<br><br>
             <b style="color: {PRIMARY_COLOR};">Recommendation:</b> Best for programmatic use, preserving hierarchy, or detailed inspection.
             </div>
             """
        )
        json_label.setWordWrap(True)
        layout.addWidget(json_label)
        
        layout.addStretch(1) # Add stretch before the button

        # --- Close Button ---
        button_layout = QHBoxLayout()
        button_layout.addStretch() # Push button to the right
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        # Remove fixed size policy, let layout manage it, apply theme style if possible
        close_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        button_layout.addWidget(close_button)
        
        layout.addLayout(button_layout)

        # Attempt to apply parent stylesheet for theme consistency
        if parent and parent.styleSheet():
             self.setStyleSheet(parent.styleSheet()) 