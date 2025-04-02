# ANPE GUI Code Examples

This document provides code examples for implementing the most critical improvements identified in our analysis.

## 1. Module Structure Fix

### Updated `__init__.py` for proper imports

```python
# anpe_gui/__init__.py
"""
ANPE GUI Package
"""

# Make core components available at package level
from .app import main
from .main_window import MainWindow
from .splash_screen import SplashScreen

# Version information
__version__ = "0.1.0"
```

### Fixed import in `__main__.py`

```python
# anpe_gui/__main__.py
"""
Entry point for the ANPE GUI application when run as a module.
"""

import sys
import os

# Add the parent directory to sys.path to allow importing anpe_gui as a package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now we can import from anpe_gui
from anpe_gui.app import main

if __name__ == "__main__":
    main()
```

## 2. Collapsible Configuration Section

```python
# anpe_gui/widgets/collapsible_section.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QFrame, QSizePolicy, QLabel)
from PyQt5.QtCore import Qt, QPropertyAnimation, QSize, pyqtProperty
from PyQt5.QtGui import QIcon, QPixmap

class CollapsibleSection(QWidget):
    """A collapsible section widget that can expand/collapse its content."""
    
    def __init__(self, title="Configuration", parent=None):
        super().__init__(parent)
        
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Header with toggle button
        self.header_widget = QWidget()
        self.header_layout = QHBoxLayout(self.header_widget)
        self.header_layout.setContentsMargins(5, 5, 5, 5)
        
        self.toggle_button = QPushButton()
        self.toggle_button.setFixedSize(20, 20)
        self.toggle_button.setStyleSheet("border: none;")
        self.toggle_button.clicked.connect(self.toggle_expanded)
        
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: bold;")
        
        self.header_layout.addWidget(self.toggle_button)
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()
        
        # Content frame (will be collapsed/expanded)
        self.content_frame = QFrame()
        self.content_layout = QVBoxLayout(self.content_frame)
        
        # Add widgets to main layout
        self.main_layout.addWidget(self.header_widget)
        self.main_layout.addWidget(self.content_frame)
        
        # State and animation
        self._expanded = True
        self._update_toggle_button()
        self._animation = QPropertyAnimation(self, b"collapsedHeight")
        self._animation.setDuration(200)  # Animation duration in ms
        
        # Initialize size policy
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
    def add_widget(self, widget):
        """Add a widget to the content area."""
        self.content_layout.addWidget(widget)
        
    def toggle_expanded(self):
        """Toggle between expanded and collapsed states."""
        self._expanded = not self._expanded
        self._update_toggle_button()
        
        if self._expanded:
            self._animation.setStartValue(self.content_frame.height())
            self._animation.setEndValue(self.content_frame.sizeHint().height())
            self.content_frame.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
        else:
            self._animation.setStartValue(self.content_frame.height())
            self._animation.setEndValue(0)
            
        self._animation.start()
        
    def _update_toggle_button(self):
        """Update the toggle button icon based on current state."""
        icon = "▼" if self._expanded else "►"
        self.toggle_button.setText(icon)
        
    def is_expanded(self):
        """Return the current expanded state."""
        return self._expanded
        
    @pyqtProperty(int)
    def collapsedHeight(self):
        return self.content_frame.maximumHeight()
    
    @collapsedHeight.setter
    def collapsedHeight(self, height):
        self.content_frame.setMaximumHeight(height)
```

## 3. Standardized Input Controls

```python
# anpe_gui/widgets/standardized_inputs.py
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QLabel, QSpinBox, 
                            QCheckBox, QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt, pyqtSignal

class LabeledSpinBox(QWidget):
    """A standardized widget combining a label with a spin box."""
    
    valueChanged = pyqtSignal(int)
    
    def __init__(self, label_text, min_value=0, max_value=100, default_value=0, parent=None):
        super().__init__(parent)
        
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(label_text)
        self.spin_box = QSpinBox()
        self.spin_box.setRange(min_value, max_value)
        self.spin_box.setValue(default_value)
        self.spin_box.valueChanged.connect(self.valueChanged)
        
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.spin_box)
        self.layout.setStretch(0, 1)  # Make label take up more space
        self.layout.setStretch(1, 0)  # Keep spin box at minimum size
    
    def value(self):
        """Get the current value of the spin box."""
        return self.spin_box.value()
    
    def setValue(self, value):
        """Set the value of the spin box."""
        self.spin_box.setValue(value)

class FilteringOptions(QGroupBox):
    """A group of filtering options with standardized layout."""
    
    def __init__(self, title="Filtering Options", parent=None):
        super().__init__(title, parent)
        
        # Main layout
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(10, 15, 10, 10)
        self.layout.setSpacing(8)
        
        # Min length filter
        self.min_length_checkbox = QCheckBox("Apply minimum length:")
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 20)
        self.min_length_spin.setValue(2)
        
        # Max length filter
        self.max_length_checkbox = QCheckBox("Apply maximum length:")
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(1, 100)
        self.max_length_spin.setValue(10)
        
        # Additional options
        self.accept_pronouns = QCheckBox("Accept Pronouns")
        self.newlines_as_boundaries = QCheckBox("Treat Newlines as Sentence Boundaries")
        
        # Add widgets to layout
        self.layout.addWidget(self.min_length_checkbox, 0, 0)
        self.layout.addWidget(self.min_length_spin, 0, 1)
        self.layout.addWidget(self.max_length_checkbox, 1, 0)
        self.layout.addWidget(self.max_length_spin, 1, 1)
        self.layout.addWidget(self.accept_pronouns, 2, 0, 1, 2)
        self.layout.addWidget(self.newlines_as_boundaries, 3, 0, 1, 2)
        
        # Connect signals
        self.min_length_checkbox.toggled.connect(self.min_length_spin.setEnabled)
        self.max_length_checkbox.toggled.connect(self.max_length_spin.setEnabled)
        
        # Initial state
        self.min_length_spin.setEnabled(False)
        self.max_length_spin.setEnabled(False)
        
    def get_config(self):
        """Get the current configuration as a dictionary."""
        return {
            'min_length_enabled': self.min_length_checkbox.isChecked(),
            'min_length': self.min_length_spin.value(),
            'max_length_enabled': self.max_length_checkbox.isChecked(),
            'max_length': self.max_length_spin.value(),
            'accept_pronouns': self.accept_pronouns.isChecked(),
            'newlines_as_boundaries': self.newlines_as_boundaries.isChecked()
        }
        
    def set_config(self, config):
        """Set configuration from a dictionary."""
        self.min_length_checkbox.setChecked(config.get('min_length_enabled', False))
        self.min_length_spin.setValue(config.get('min_length', 2))
        self.max_length_checkbox.setChecked(config.get('max_length_enabled', False))
        self.max_length_spin.setValue(config.get('max_length', 10))
        self.accept_pronouns.setChecked(config.get('accept_pronouns', True))
        self.newlines_as_boundaries.setChecked(config.get('newlines_as_boundaries', True))
```

## 4. Improved Splash Screen

```python
# anpe_gui/splash_screen.py
import os
from PyQt5.QtWidgets import (QSplashScreen, QProgressBar, QVBoxLayout, 
                             QLabel, QWidget, QApplication)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QFont

class SplashScreen(QSplashScreen):
    """
    A splash screen with a progress bar for the ANPE GUI application.
    """
    
    def __init__(self):
        # Create pixmap and parent with it
        banner_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                   "pics", "banner.png")
        pixmap = QPixmap(banner_path)
        
        # Resize to appropriate dimensions (max 400x300)
        if pixmap.width() > 400 or pixmap.height() > 200:
            pixmap = pixmap.scaled(400, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        super().__init__(pixmap)
        
        # Create a widget to hold our layout
        self.content_widget = QWidget(self)
        self.layout = QVBoxLayout(self.content_widget)
        
        # Add version label
        version_label = QLabel("Version 0.1.0")
        version_label.setAlignment(Qt.AlignRight)
        version_label.setStyleSheet("color: #1a5276; font-weight: bold;")
        
        # Add tagline
        tagline = QLabel("Accurate Noun Phrase Extraction, Simplified")
        tagline.setAlignment(Qt.AlignCenter)
        tagline.setStyleSheet("color: #1a5276; font-style: italic;")
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(15)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                background: #f0f0f0;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                border-radius: 5px;
            }
        """)
        
        # Add status message
        self.status_label = QLabel("Loading...")
        self.status_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to layout with proper spacing
        self.layout.addWidget(version_label)
        self.layout.addWidget(tagline)
        self.layout.addSpacing(10)
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.status_label)
        
        # Set layout margins
        self.layout.setContentsMargins(20, pixmap.height() + 10, 20, 20)
        
        # Set fixed size for the splash screen
        self.setFixedSize(pixmap.width(), pixmap.height() + 100)
        
        # Center content widget
        self.content_widget.setGeometry(0, 0, self.width(), self.height())
    
    def set_progress(self, value, status_message=None):
        """Update the progress bar value and optionally the status message."""
        self.progress_bar.setValue(value)
        if status_message:
            self.status_label.setText(status_message)
        
        # Process events to update UI
        QApplication.processEvents()
```

## 5. Two-Tab Main Window Structure

```python
# Example modification for main_window.py (partial)
from PyQt5.QtWidgets import (QTabWidget, QSplitter, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton)
from .widgets.step_indicator import StepIndicator
from .widgets.collapsible_section import CollapsibleSection
from .widgets.file_list_widget import FileListWidget
from .widgets.structure_filter_widget import StructureFilterWidget

class MainWindow(QMainWindow):
    # ... existing code ...
    
    def setup_ui(self):
        """Set up the main UI components."""
        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create banner and header
        self.setup_header()
        
        # Create step indicator
        self.step_indicator = StepIndicator(["Input & Settings", "Results"], self)
        self.main_layout.addWidget(self.step_indicator)
        
        # Create main content area with splitter
        self.content_splitter = QSplitter(Qt.Vertical)
        
        # Create tab widget with only two tabs
        self.tab_widget = QTabWidget()
        self.setup_input_tab()
        self.setup_results_tab()
        
        # Set up log panel
        self.log_panel = self.create_log_panel()
        
        # Add widgets to splitter
        self.content_splitter.addWidget(self.tab_widget)
        self.content_splitter.addWidget(self.log_panel)
        self.content_splitter.setStretchFactor(0, 3)  # Give more space to tabs
        self.content_splitter.setStretchFactor(1, 1)  # Less space for logs
        
        # Add splitter to main layout
        self.main_layout.addWidget(self.content_splitter)
        
    def setup_input_tab(self):
        """Set up the Input & Settings tab with collapsible configuration section."""
        self.input_tab = QWidget()
        self.input_layout = QVBoxLayout(self.input_tab)
        
        # File selection widget
        self.file_list = FileListWidget()
        self.input_layout.addWidget(self.file_list)
        
        # Collapsible configuration section
        self.config_section = CollapsibleSection("Configuration Settings")
        
        # Create filter options widget
        self.filter_options = self.create_filter_options()
        self.config_section.add_widget(self.filter_options)
        
        # Create structure filter widget
        self.structure_filter = StructureFilterWidget(self.structure_types)
        self.config_section.add_widget(self.structure_filter)
        
        # Add configuration section to layout
        self.input_layout.addWidget(self.config_section)
        
        # Add navigation buttons
        self.input_buttons = QHBoxLayout()
        self.input_buttons.addStretch()
        self.process_button = QPushButton("Process Files")
        self.process_button.clicked.connect(self.process_files)
        self.input_buttons.addWidget(self.process_button)
        self.input_layout.addLayout(self.input_buttons)
        
        # Add tab to widget
        self.tab_widget.addTab(self.input_tab, "Input & Settings")
        
    def setup_results_tab(self):
        """Set up the Results tab."""
        self.results_tab = QWidget()
        self.results_layout = QVBoxLayout(self.results_tab)
        
        # Results display widget (will be populated after processing)
        self.results_display = self.create_results_display()
        self.results_layout.addWidget(self.results_display)
        
        # Export options
        self.export_options = self.create_export_options()
        self.results_layout.addWidget(self.export_options)
        
        # Add navigation buttons
        self.results_buttons = QHBoxLayout()
        self.back_to_input_button = QPushButton("Back to Input")
        self.back_to_input_button.clicked.connect(self.go_to_input_tab)
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results)
        self.new_input_button = QPushButton("Process New Input")
        self.new_input_button.clicked.connect(self.reset_and_go_to_input)
        
        self.results_buttons.addWidget(self.back_to_input_button)
        self.results_buttons.addStretch()
        self.results_buttons.addWidget(self.export_button)
        self.results_buttons.addWidget(self.new_input_button)
        
        self.results_layout.addLayout(self.results_buttons)
        
        # Add tab to widget
        self.tab_widget.addTab(self.results_tab, "Results")
```

## 6. Enhanced Log Panel

```python
# anpe_gui/widgets/enhanced_log_panel.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                            QComboBox, QPushButton, QLabel, QToolButton)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QTextCursor, QColor, QTextCharFormat, QBrush

class EnhancedLogPanel(QWidget):
    """
    An enhanced log panel that displays formatted log messages with filtering capabilities.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # Header with controls
        self.header_layout = QHBoxLayout()
        
        self.title_label = QLabel("Log Output")
        self.title_label.setStyleSheet("font-weight: bold;")
        
        self.level_filter = QComboBox()
        self.level_filter.addItems(["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_filter.setCurrentText("INFO")
        self.level_filter.currentTextChanged.connect(self.filter_log)
        
        self.clear_button = QToolButton()
        self.clear_button.setText("Clear")
        self.clear_button.clicked.connect(self.clear_log)
        
        self.copy_button = QToolButton()
        self.copy_button.setText("Copy")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        
        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()
        self.header_layout.addWidget(QLabel("Filter:"))
        self.header_layout.addWidget(self.level_filter)
        self.header_layout.addWidget(self.clear_button)
        self.header_layout.addWidget(self.copy_button)
        
        # Log text display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setLineWrapMode(QTextEdit.NoWrap)
        
        # Add widgets to main layout
        self.layout.addLayout(self.header_layout)
        self.layout.addWidget(self.log_display)
        
        # Log level color mapping
        self.level_colors = {
            "DEBUG": QColor(150, 150, 150),      # Gray
            "INFO": QColor(0, 0, 0),             # Black
            "WARNING": QColor(255, 165, 0),      # Orange
            "ERROR": QColor(255, 0, 0),          # Red
            "CRITICAL": QColor(128, 0, 128)      # Purple
        }
        
        # Store all log entries for filtering
        self.log_entries = []
        
        # Auto-scroll flag
        self.auto_scroll = True
        
    @pyqtSlot(str, str)
    def add_log_entry(self, message, level="INFO"):
        """
        Add a new log entry with the specified message and level.
        """
        # Create the formatted entry
        timestamp = self.get_timestamp()
        entry = {
            "timestamp": timestamp,
            "level": level.upper(),
            "message": message,
            "display_text": f"[{timestamp}] [{level.upper()}] {message}"
        }
        
        # Store the entry
        self.log_entries.append(entry)
        
        # Check if we should display this entry based on current filter
        if self.should_display_entry(entry):
            self.append_entry_to_display(entry)
            
    def append_entry_to_display(self, entry):
        """Append a log entry to the display with proper formatting."""
        # Create text format with appropriate color
        text_format = QTextCharFormat()
        level = entry["level"]
        if level in self.level_colors:
            text_format.setForeground(QBrush(self.level_colors[level]))
            
        # Append text with formatting
        cursor = self.log_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(entry["display_text"] + "\n", text_format)
        
        # Auto-scroll if enabled
        if self.auto_scroll:
            self.log_display.setTextCursor(cursor)
            
    def filter_log(self):
        """Filter log entries based on the selected level."""
        # Clear the display
        self.log_display.clear()
        
        # Get current filter level
        filter_level = self.level_filter.currentText()
        
        # Add entries that match the filter
        for entry in self.log_entries:
            if self.should_display_entry(entry, filter_level):
                self.append_entry_to_display(entry)
                
    def should_display_entry(self, entry, filter_level=None):
        """Check if an entry should be displayed based on the filter level."""
        if filter_level is None:
            filter_level = self.level_filter.currentText()
            
        if filter_level == "All":
            return True
            
        # Map log levels to numeric values for comparison
        level_values = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }
        
        entry_level = entry["level"]
        if entry_level not in level_values or filter_level not in level_values:
            return True
            
        # Show the entry if its level is >= the filter level
        return level_values[entry_level] >= level_values[filter_level]
        
    def clear_log(self):
        """Clear the log display and entries."""
        self.log_display.clear()
        self.log_entries.clear()
        
    def copy_to_clipboard(self):
        """Copy the current log contents to clipboard."""
        self.log_display.selectAll()
        self.log_display.copy()
        # Deselect text
        cursor = self.log_display.textCursor()
        cursor.clearSelection()
        self.log_display.setTextCursor(cursor)
        
    def get_timestamp(self):
        """Get the current timestamp in a formatted string."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

These code examples address the most critical improvements identified in our analysis. When implemented, they will significantly enhance the ANPE GUI's usability, visual appeal, and code organization. 