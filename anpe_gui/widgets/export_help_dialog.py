"""
Dialog to explain the different export formats available in ANPE.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt

class ExportHelpDialog(QDialog):
    """A simple dialog explaining the TXT, CSV, and JSON export formats."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Information")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(15, 15, 15, 15)

        # --- Filename Structure Section ---
        filename_label = QLabel()
        filename_label.setTextFormat(Qt.TextFormat.RichText)
        filename_label.setText(
            "<b><u>Export Filename Structure</u></b><br>"
            "Exported files are named automatically to ensure uniqueness and provide context:<br><br>"
            "&#8226; <b>Batch Export (Multiple Files):</b><br>"
            "  <code>[<i>prefix</i>_]<i>original_filename</i>_anpe_results_<i>YYYYMMDD_HHMMSS</i>.<i>format</i></code><br>"
            "&#8226; <b>Single Export (Text Input):</b><br>"
            "  <code>[<i>prefix</i>_]anpe_text_results_<i>YYYYMMDD_HHMMSS</i>.<i>format</i></code><br><br>"
            "Where:<br>"
            " - <code>[<i>prefix</i>_]</code> is the optional prefix you enter.<br>"
            " - <code><i>original_filename</i></code> is the name of the input file (without extension).<br>"
            " - <code><i>YYYYMMDD_HHMMSS</i></code> is the timestamp of the export.<br>"
            " - <code><i>format</i></code> is the selected format (txt, csv, json)."
        )
        filename_label.setWordWrap(True)
        layout.addWidget(filename_label)

        # Separator
        separator0 = QFrame()
        separator0.setFrameShape(QFrame.Shape.HLine)
        separator0.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator0)

        # --- Format Section Title ---
        format_title_label = QLabel("<b><u>Export Formats</u></b>")
        format_title_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(format_title_label)
        # Add a little space after the title
        layout.addSpacing(5)

        # --- TXT Format ---
        txt_label = QLabel()
        txt_label.setTextFormat(Qt.TextFormat.RichText)
        txt_label.setText(
            "<b>TXT Format (.txt)</b><br>"
            "Outputs noun phrases in a human-readable, plain text format. "
            "Nested phrases are shown using indentation.<br><br>"
            "<b>Recommendation:</b> Best for quick reading, simple lists, or when you "
            "just need the plain text of the phrases, especially top-level ones."
        )
        txt_label.setWordWrap(True)
        layout.addWidget(txt_label)

        # Separator (thin, less prominent)
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.HLine)
        separator1.setStyleSheet("border-top: 1px solid #ccc;")
        layout.addWidget(separator1)

        # --- CSV Format ---
        csv_label = QLabel()
        csv_label.setTextFormat(Qt.TextFormat.RichText)
        csv_label.setText(
            "<b><u>CSV Format (.csv)</u></b><br>"
            "Outputs results as Comma-Separated Values, suitable for spreadsheets (like Excel). "
            "Provides a flat structure where each row is a noun phrase. "
            "Hierarchy is represented using 'ID' and 'Parent_ID' columns. "
            "Includes columns for Level, Length, and Structures if metadata is enabled.<br><br>"
            "<b>Recommendation:</b> Best for importing into spreadsheets or other data analysis "
            "tools that work well with tabular data."
        )
        csv_label.setWordWrap(True)
        layout.addWidget(csv_label)

        # Separator (thin)
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet("border-top: 1px solid #ccc;")
        layout.addWidget(separator2)

        # --- JSON Format ---
        json_label = QLabel()
        json_label.setTextFormat(Qt.TextFormat.RichText)
        json_label.setText(
            "<b>JSON Format (.json)</b><br>"
            "Outputs results in JavaScript Object Notation. Maintains the full hierarchical "
            "structure using nested objects. Includes all metadata.<br>"
            "<b>Recommendation:</b> Best for programmatic use or preserving hierarchy."
        )
        json_label.setWordWrap(True)
        layout.addWidget(json_label)

        # --- Close Button ---
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        close_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        button_layout.addWidget(close_button)
        
        layout.addStretch()
        layout.addLayout(button_layout)

        if parent and parent.styleSheet():
             self.setStyleSheet(parent.styleSheet()) 