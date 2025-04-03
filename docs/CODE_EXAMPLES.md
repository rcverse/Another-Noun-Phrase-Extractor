# ANPE GUI Code Examples

## 1. Two-Tab Main Window Structure

```python
# anpe_gui/main_window.py (partial)
from PyQt5.QtWidgets import (QMainWindow, QWidget, QTabWidget, QSplitter, 
                             QVBoxLayout, QHBoxLayout, QLabel)
from PyQt5.QtCore import Qt

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ANPE - Another Noun Phrase Extractor")
        self.setup_ui()
    
    def setup_ui(self):
        # Create central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Header
        self.header = QWidget()
        header_layout = QHBoxLayout(self.header)
        
        self.title_label = QLabel("ANPE")
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #005792;")
        
        self.subtitle_label = QLabel("Another Noun Phrase Extractor")
        self.subtitle_label.setStyleSheet("font-style: italic; color: #444;")
        
        self.version_label = QLabel("Version: 0.1.0")
        self.version_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.subtitle_label)
        header_layout.addStretch()
        header_layout.addWidget(self.version_label)
        
        self.main_layout.addWidget(self.header)
        
        # Create horizontal splitter
        self.main_splitter = QSplitter(Qt.Horizontal)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create input tab
        self.input_tab = QWidget()
        self.setup_input_tab()
        self.tab_widget.addTab(self.input_tab, "Input")
        
        # Create output tab
        self.output_tab = QWidget()
        self.setup_output_tab()
        self.tab_widget.addTab(self.output_tab, "Output")
        
        # Set up log panel
        self.log_panel = self.create_log_panel()
        
        # Add widgets to splitter
        self.main_splitter.addWidget(self.tab_widget)
        self.main_splitter.addWidget(self.log_panel)
        
        # Set splitter proportions
        self.main_splitter.setStretchFactor(0, 7)  # Main content
        self.main_splitter.setStretchFactor(1, 3)  # Log panel
        
        # Add splitter to main layout
        self.main_layout.addWidget(self.main_splitter)
        
        # Add status bar beneath everything
        self.status_bar = self.create_status_bar()
        self.main_layout.addWidget(self.status_bar)
```

## 2. File Input Section

```python
# anpe_gui/widgets/file_list_widget.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QListWidget, QLabel, QFileDialog, QButtonGroup)
from PyQt5.QtCore import Qt, pyqtSignal

class FileListWidget(QWidget):
    """Widget for file selection and management."""
    
    files_changed = pyqtSignal(list)  # Signal emitted when file list changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        self.layout = QVBoxLayout(self)
        
        # Input mode selection
        self.input_mode_layout = QHBoxLayout()
        
        self.file_button = QPushButton("File Input")
        self.file_button.setCheckable(True)
        self.file_button.setChecked(True)
        
        self.text_button = QPushButton("Text Input")
        self.text_button.setCheckable(True)
        
        # Button group to ensure only one is selected
        self.input_group = QButtonGroup()
        self.input_group.addButton(self.file_button)
        self.input_group.addButton(self.text_button)
        self.input_group.setExclusive(True)
        
        self.input_mode_layout.addWidget(self.file_button)
        self.input_mode_layout.addWidget(self.text_button)
        self.input_mode_layout.addStretch()
        
        self.layout.addLayout(self.input_mode_layout)
        
        # Files to process label
        self.files_label = QLabel("Files to Process")
        self.layout.addWidget(self.files_label)
        
        # File list
        self.file_list = QListWidget()
        self.layout.addWidget(self.file_list)
        
        # File action buttons
        self.button_layout = QHBoxLayout()
        
        self.add_files_button = QPushButton("Add Files")
        self.add_dir_button = QPushButton("Add Dir")
        self.remove_button = QPushButton("Remove")
        self.clear_button = QPushButton("Clear All")
        
        self.button_layout.addWidget(self.add_files_button)
        self.button_layout.addWidget(self.add_dir_button)
        self.button_layout.addWidget(self.remove_button)
        self.button_layout.addWidget(self.clear_button)
        
        self.layout.addLayout(self.button_layout)
        
        # Status label
        self.status_label = QLabel("No files selected")
        self.layout.addWidget(self.status_label)
        
        # Connect signals
        self.add_files_button.clicked.connect(self.add_files)
        self.add_dir_button.clicked.connect(self.add_directory)
        self.remove_button.clicked.connect(self.remove_selected_files)
        self.clear_button.clicked.connect(self.clear_files)
        self.file_list.itemSelectionChanged.connect(self.update_status)
        
    def add_files(self):
        """Open file dialog to add files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Files", "", "Text Files (*.txt);;All Files (*)"
        )
        
        if files:
            for file_path in files:
                if self.file_list.findItems(file_path, Qt.MatchExactly) == []:
                    self.file_list.addItem(file_path)
            
            self.update_status()
            self.files_changed.emit(self.get_files())
            
    def add_directory(self):
        """Open directory dialog to add all files in a directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory"
        )
        
        if directory:
            import os
            for file_name in os.listdir(directory):
                file_path = os.path.join(directory, file_name)
                if os.path.isfile(file_path) and file_path.endswith('.txt'):
                    if self.file_list.findItems(file_path, Qt.MatchExactly) == []:
                        self.file_list.addItem(file_path)
            
            self.update_status()
            self.files_changed.emit(self.get_files())
            
    def remove_selected_files(self):
        """Remove selected files from the list."""
        for item in self.file_list.selectedItems():
            self.file_list.takeItem(self.file_list.row(item))
            
        self.update_status()
        self.files_changed.emit(self.get_files())
        
    def clear_files(self):
        """Clear all files from the list."""
        self.file_list.clear()
        self.update_status()
        self.files_changed.emit([])
        
    def update_status(self):
        """Update the status label based on current selection."""
        file_count = self.file_list.count()
        
        if file_count == 0:
            self.status_label.setText("No files selected")
        else:
            self.status_label.setText(f"{file_count} file(s) selected")
            
    def get_files(self):
        """Get the list of files."""
        files = []
        for i in range(self.file_list.count()):
            files.append(self.file_list.item(i).text())
        return files
```

## 3. Structure Filter Widget

```python
# anpe_gui/widgets/structure_filter_widget.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QCheckBox, QPushButton, QLabel)
from PyQt5.QtCore import Qt

class StructureFilterWidget(QWidget):
    """Widget for selecting structure types to filter."""
    
    def __init__(self, structure_types=None, parent=None):
        super().__init__(parent)
        
        if structure_types is None:
            self.structure_types = [
                "Determiner", "Possessive", "Relative Clause",
                "Adjectival Modifier", "Quantified", "Nonfinite Complement",
                "Prepositional Modifier", "Coordinated", "Named Entity",
                "Compound", "Appositive", "Standalone Noun"
            ]
        else:
            self.structure_types = structure_types
            
        self.setup_ui()
    
    def setup_ui(self):
        # Main layout
        self.layout = QVBoxLayout(self)
        
        # Header with enable toggle
        self.header_layout = QHBoxLayout()
        
        self.enable_toggle = QCheckBox("Enable structure filtering")
        self.enable_toggle.toggled.connect(self.toggle_filtering)
        
        self.help_button = QPushButton("?")
        self.help_button.setFixedSize(20, 20)
        self.help_button.setToolTip("Select which structural types to include in extraction")
        
        self.header_layout.addWidget(self.enable_toggle)
        self.header_layout.addStretch()
        self.header_layout.addWidget(self.help_button)
        
        self.layout.addLayout(self.header_layout)
        
        # Structure type grid (4 rows x 3 columns)
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(10)
        
        self.structure_checkboxes = {}
        
        for i, structure_type in enumerate(self.structure_types):
            row = i // 3
            col = i % 3
            
            checkbox = QCheckBox(structure_type)
            self.structure_checkboxes[structure_type] = checkbox
            self.grid_layout.addWidget(checkbox, row, col)
            
        self.layout.addLayout(self.grid_layout)
        
        # Action buttons
        self.button_layout = QHBoxLayout()
        
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.select_all)
        
        self.clear_selection_button = QPushButton("Clear Selection")
        self.clear_selection_button.clicked.connect(self.clear_selection)
        
        self.button_layout.addWidget(self.select_all_button)
        self.button_layout.addWidget(self.clear_selection_button)
        
        self.layout.addLayout(self.button_layout)
        
        # Initial state
        self.toggle_filtering(False)
        
    def toggle_filtering(self, enabled):
        """Enable or disable structure filtering."""
        for checkbox in self.structure_checkboxes.values():
            checkbox.setEnabled(enabled)
            
        self.select_all_button.setEnabled(enabled)
        self.clear_selection_button.setEnabled(enabled)
        
    def select_all(self):
        """Select all structure types."""
        for checkbox in self.structure_checkboxes.values():
            checkbox.setChecked(True)
            
    def clear_selection(self):
        """Clear structure type selection."""
        for checkbox in self.structure_checkboxes.values():
            checkbox.setChecked(False)
            
    def get_selected_structures(self):
        """Get the list of selected structure types."""
        selected = []
        for structure_type, checkbox in self.structure_checkboxes.items():
            if checkbox.isChecked():
                selected.append(structure_type)
        return selected
```

## 4. Enhanced Log Panel

```python
# anpe_gui/widgets/enhanced_log_panel.py
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                            QComboBox, QPushButton, QLabel)
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QTextCursor, QColor, QTextCharFormat, QBrush

class EnhancedLogPanel(QWidget):
    """Enhanced log panel with filtering and copy functionality."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # Header
        self.header_layout = QHBoxLayout()
        
        self.title_label = QLabel("Log Output")
        self.title_label.setStyleSheet("font-weight: bold;")
        
        self.filter_label = QLabel("Filter:")
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"])
        self.filter_combo.setCurrentText("INFO")
        
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_log)
        
        self.copy_button = QPushButton("Copy")
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
        
        self.layout.addWidget(self.log_text)
        
        # Connect signals
        self.filter_combo.currentTextChanged.connect(self.update_filter)
        
        # Initialize log colors
        self.log_colors = {
            "DEBUG": QColor(100, 100, 100),
            "INFO": QColor(0, 0, 0),
            "WARNING": QColor(255, 140, 0),
            "ERROR": QColor(255, 0, 0),
            "CRITICAL": QColor(128, 0, 128)
        }
        
        # Store log entries for filtering
        self.log_entries = []
        
    def add_log_entry(self, message, level="INFO"):
        """Add a log entry to the panel."""
        entry = {
            "level": level.upper(),
            "message": message
        }
        
        self.log_entries.append(entry)
        
        # Check if we should display based on current filter
        if self.should_display(entry):
            self.append_to_display(entry)
    
    def append_to_display(self, entry):
        """Append an entry to the display with proper formatting."""
        level = entry["level"]
        message = entry["message"]
            
        # Format with color
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        
        format = QTextCharFormat()
        if level in self.log_colors:
            format.setForeground(QBrush(self.log_colors[level]))
        
        cursor.insertText(f"[{level}] {message}\n", format)
        
        # Auto-scroll
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()
    
    def update_filter(self, filter_level):
        """Update displayed logs based on filter."""
        self.log_text.clear()
        
        for entry in self.log_entries:
            if self.should_display(entry):
                self.append_to_display(entry)
    
    def should_display(self, entry):
        """Check if entry should be displayed based on current filter."""
        filter_level = self.filter_combo.currentText()
        
        # Map levels to numeric values
        level_values = {
            "DEBUG": 0,
            "INFO": 1,
            "WARNING": 2,
            "ERROR": 3,
            "CRITICAL": 4
        }
        
        # Display if entry level >= filter level
        return level_values.get(entry["level"], 0) >= level_values.get(filter_level, 0)
        
    def clear_log(self):
        """Clear the log panel."""
        self.log_text.clear()
        self.log_entries = []
        
    def copy_to_clipboard(self):
        """Copy log contents to clipboard."""
        self.log_text.selectAll()
        self.log_text.copy()
        
        # Deselect after copy
        cursor = self.log_text.textCursor()
        cursor.clearSelection()
        self.log_text.setTextCursor(cursor)
```

## 5. Status Bar with Progress

```python
# anpe_gui/widgets/status_bar.py
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QProgressBar, 
                             QLabel, QFrame)
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot

class StatusBar(QWidget):
    """
    Global status bar with progress indicator and text status.
    Displays beneath the log panel and provides feedback for all operations.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 2, 5, 2)  # Slim margins for compact design
        
        # Status label for text updates
        self.status_label = QLabel("Ready")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFixedHeight(16)  # Make it compact
        
        # Use a fixed width for the progress bar
        self.progress_bar.setFixedWidth(200)
        
        # Add separator line
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.VLine)
        self.separator.setFrameShadow(QFrame.Sunken)
        
        # Add widgets to layout
        self.layout.addWidget(self.status_label)
        self.layout.addStretch()
        self.layout.addWidget(self.separator)
        self.layout.addWidget(self.progress_bar)
        
        # Set fixed height for the entire status bar
        self.setFixedHeight(24)
        
        # Add bottom border
        self.setStyleSheet("""
            StatusBar {
                border-top: 1px solid #ccc;
                background-color: #f5f5f5;
            }
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 2px;
                background: white;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #3498db;
                width: 10px;
            }
        """)
        
        # Hide progress bar initially
        self.progress_bar.hide()
        
    @pyqtSlot(str)
    def set_status(self, message):
        """Update the status message."""
        self.status_label.setText(message)
        
    @pyqtSlot(int, str)
    def update_progress(self, value, message=None):
        """Update the progress bar and optionally the status message."""
        self.progress_bar.setValue(value)
        self.progress_bar.show()  # Ensure it's visible
        
        if message:
            self.status_label.setText(message)
            
    def start_progress(self, message="Processing..."):
        """Start an indeterminate progress operation."""
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.progress_bar.show()
        self.status_label.setText(message)
        
    def stop_progress(self, message="Complete"):
        """Stop the progress bar and update status."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.status_label.setText(message)
        
        # Hide progress after a delay
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(2000, self.clear_progress)
        
    def clear_progress(self):
        """Clear the progress bar."""
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
```

## 6. Integration in Main Window

```python
# anpe_gui/main_window.py (status bar integration)
def create_status_bar(self):
    """Create the status bar with progress indicator."""
    from anpe_gui.widgets.status_bar import StatusBar
    
    status_bar = StatusBar()
    
    # Make it accessible to all processing methods
    self.status_bar = status_bar
    
    return status_bar
    
def process_files(self):
    """Process the selected files and update the status bar."""
    files = self.file_list.get_files()
    
    if not files:
        self.status_bar.set_status("No files selected")
        return
        
    # Update status bar
    self.status_bar.update_progress(0, f"Processing {len(files)} file(s)...")
    
    # Create worker for background processing
    from anpe_gui.widgets.batch_processor import BatchWorker
    
    # Create worker and thread
    config = self.get_current_config()
    self.worker = BatchWorker(files, config)
    self.thread = QThread()
    
    # Move worker to thread
    self.worker.moveToThread(self.thread)
    
    # Connect signals
    self.thread.started.connect(self.worker.process)
    self.worker.progress_updated.connect(self.status_bar.update_progress)
    self.worker.processing_complete.connect(self.on_processing_complete)
    self.worker.error_occurred.connect(self.on_processing_error)
    
    # Start processing
    self.thread.start()
    
def on_processing_complete(self, results):
    """Handle processing completion."""
    self.status_bar.stop_progress("Processing complete")
    
    # Store results and switch to output tab
    self.results = results
    self.populate_results_tab()
    self.tab_widget.setCurrentIndex(1)  # Switch to Output tab
    
def on_processing_error(self, error_message):
    """Handle processing error."""
    self.status_bar.stop_progress(f"Error: {error_message}")
```

## 7. Progress Bar for Batch Processing

```python
# anpe_gui/widgets/batch_processor.py
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QProgressBar, 
                             QLabel, QPushButton)

class BatchWorker(QObject):
    """Worker thread for batch processing."""
    
    progress_updated = pyqtSignal(int, str)
    processing_complete = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, files, config):
        super().__init__()
        self.files = files
        self.config = config
        self.results = {}
        self.is_cancelled = False
        
    def process(self):
        """Process all files in the batch."""
        try:
            for i, file_path in enumerate(self.files):
                if self.is_cancelled:
                    break
                    
                # Calculate progress percentage
                progress = int((i / len(self.files)) * 100)
                file_name = os.path.basename(file_path)
                self.progress_updated.emit(progress, f"Processing {file_name}... ({i+1}/{len(self.files)})")
                
                # Process the file using the ANPE core functionality
                # IMPORTANT: This preserves the existing core ANPE functionality
                from anpe.extraction import extract_noun_phrases
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # Apply configuration options from the GUI
                min_length = self.config.get('min_length', None)
                max_length = self.config.get('max_length', None)
                structure_types = self.config.get('structure_types', None)
                
                # Extract noun phrases
                noun_phrases = extract_noun_phrases(
                    text, 
                    min_length=min_length,
                    max_length=max_length,
                    structure_types=structure_types
                )
                
                # Store results
                self.results[file_path] = noun_phrases
                
            # Processing complete
            self.progress_updated.emit(100, "Processing complete!")
            self.processing_complete.emit(self.results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def cancel(self):
        """Cancel processing."""
        self.is_cancelled = True
```

These code examples focus on implementing the core UI components based on the screenshot and user requirements, while preserving the existing ANPE core functionality. 