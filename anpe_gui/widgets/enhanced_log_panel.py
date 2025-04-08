"""
Enhanced log panel widget with filtering, clear, and copy capabilities.
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
                             QComboBox, QPushButton, QLabel, QApplication)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QTextCursor, QColor, QTextCharFormat, QBrush, QFont
import logging # Import logging for level constants
from anpe_gui.theme import get_scroll_bar_style

class EnhancedLogPanel(QWidget):
    """Enhanced log panel with filtering and copy functionality."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._log_entries = [] # Store all entries for filtering
        self._current_filter_level = logging.INFO # Default filter level
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(5)
        
        # Header
        self.header_layout = QHBoxLayout()
        
        self.title_label = QLabel("Log Output")
        self.title_label.setStyleSheet("font-weight: bold;")
        
        self.filter_label = QLabel("Filter Level:")
        
        self.filter_combo = QComboBox()
        # Add levels using standard logging names and map to values
        self.log_levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.filter_combo.addItems(self.log_levels.keys())
        self.filter_combo.setCurrentText("INFO") # Default filter
        self.filter_combo.currentTextChanged.connect(self.update_filter)
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.setToolTip("Clear the log display")
        self.clear_button.clicked.connect(self.clear_log)
        
        self.copy_button = QPushButton("Copy")
        self.copy_button.setToolTip("Copy log content to clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()
        self.header_layout.addWidget(self.filter_label)
        self.header_layout.addWidget(self.filter_combo)
        self.header_layout.addWidget(self.clear_button)
        self.header_layout.addWidget(self.copy_button)
        
        self.layout.addLayout(self.header_layout)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier New", 9))
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        
        # Ensure scroll bars are always visible and styled
        self.log_text.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.log_text.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.log_text.setStyleSheet(get_scroll_bar_style())
        
        self.layout.addWidget(self.log_text, 1) # Give stretch factor
        
        # Log level color mapping
        self.level_colors = {
            logging.DEBUG: QColor(100, 100, 100), # Gray
            logging.INFO: QColor(0, 0, 0),        # Black
            logging.WARNING: QColor(200, 120, 0), # Dark Orange
            logging.ERROR: QColor(200, 0, 0),     # Dark Red
            logging.CRITICAL: QColor(128, 0, 128) # Purple
        }

    @pyqtSlot(str, int) # Slot accepts message (str) and level (int)
    def add_log_entry(self, message, level):
        """Add a log entry with a specific level."""
        entry = {"level": level, "message": message}
        self._log_entries.append(entry)
        
        # Append to display only if it passes the current filter
        if self.should_display(entry):
            self.append_to_display(entry)

    def append_to_display(self, entry):
        """Append formatted entry to the QTextEdit."""
        level = entry["level"]
        message = entry["message"]
        level_name = logging.getLevelName(level)
        
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # Format based on level
        log_format = QTextCharFormat()
        log_format.setForeground(QBrush(self.level_colors.get(level, QColor("black"))))
        if level >= logging.ERROR:
            log_format.setFontWeight(QFont.Weight.Bold)
        
        # Insert level name first (bold/colored)
        level_format = QTextCharFormat()
        level_format.setForeground(QBrush(self.level_colors.get(level, QColor("black"))))
        level_format.setFontWeight(QFont.Weight.Bold)
        cursor.insertText(f"[{level_name}] ", level_format)
        
        # Insert message with standard formatting for the level
        cursor.insertText(message + "\n", log_format)
        
        # Auto-scroll to bottom
        self.log_text.ensureCursorVisible()

    def update_filter(self, level_name):
        """Update the displayed logs based on the selected filter level."""
        self._current_filter_level = self.log_levels.get(level_name, logging.INFO)
        self.log_text.clear()
        for entry in self._log_entries:
            if self.should_display(entry):
                self.append_to_display(entry)

    def should_display(self, entry):
        """Check if the entry's level meets the current filter level."""
        return entry["level"] >= self._current_filter_level

    def clear_log(self):
        """Clear the log display and the stored entries."""
        self._log_entries.clear()
        self.log_text.clear()

    def copy_to_clipboard(self):
        """Copy the entire log content to the system clipboard."""
        QApplication.clipboard().setText(self.log_text.toPlainText()) 