"""
Main window implementation for the ANPE GUI application.
"""

import os
import sys
import logging
import functools # Import functools
from typing import Optional, Dict, Any, List
from pathlib import Path

from PyQt6.QtCore import Qt, QThreadPool, pyqtSignal, QSize, QTimer, QObject, QThread, pyqtSlot
from PyQt6.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QSpinBox,
    QCheckBox, QComboBox, QGroupBox, QFormLayout, QLineEdit,
    QProgressBar, QMessageBox, QSplitter, QFrame, QTabWidget, QGridLayout, QSizePolicy,
    QButtonGroup, QApplication
)
from PyQt6.QtGui import QIcon, QTextCursor, QPixmap

from .theme import PRIMARY_COLOR, SECONDARY_COLOR # Import specific colors needed

try:
    from anpe import ANPEExtractor, __version__ as anpe_version
except ImportError:
    QMessageBox.critical(
        None, 
        "Import Error", 
        "Could not import ANPE library. Please make sure it's installed."
    )
    anpe_version_str = "N/A" # Use a different name to avoid conflict if ANPEExtractor fails
    # Allow GUI to potentially start to show the error, but disable processing
    class ANPEExtractor: # Dummy class if import fails
        def __init__(self, *args, **kwargs): pass
        def extract_noun_phrases(self, *args, **kwargs): return {}
        def export_to_file(self, *args, **kwargs): pass
else:
    anpe_version_str = anpe_version # Use the real version if import succeeds


from anpe_gui.workers import ExtractionWorker, BatchWorker, QtLogHandler
from anpe_gui.widgets import (FileListWidget, StructureFilterWidget, 
                              StatusBar, EnhancedLogPanel) # Ensure StatusBar is imported from widgets
# Removed redundant theme color imports here, they are used in theme.py
# from anpe_gui.theme import PRIMARY_COLOR, SECONDARY_COLOR, ACCENT_COLOR, TEXT_COLOR, BACKGROUND_COLOR 
from anpe_gui.theme import get_stylesheet # Import the function to get the stylesheet


# --- Worker for Background Initialization ---
class ExtractorInitializer(QObject):
    initialized = pyqtSignal(object)  # Emit the extractor instance
    error = pyqtSignal(str)          # Emit error message
    models_missing = pyqtSignal()    # New signal to indicate models need setup

    def run(self):
        """Initialize the ANPEExtractor in the background."""
        logging.info("WORKER: Initializing ANPEExtractor...")
        try:
            from anpe.utils import setup_models
            # First check if models are present
            if not setup_models.check_all_models_present():
                logging.info("WORKER: Required models not found, setup needed.")
                self.models_missing.emit()
                return
                
            # Models are present, initialize extractor
            extractor = ANPEExtractor()
            logging.info("WORKER: ANPEExtractor initialized successfully.")
            self.initialized.emit(extractor)
        except Exception as e:
            logging.error(f"WORKER: Error initializing ANPEExtractor: {e}", exc_info=True)
            self.error.emit(str(e))
# --- End Worker --- 

class MainWindow(QMainWindow):
    """Main window for the ANPE GUI application."""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("ANPE GUI - Another Noun Phrase Extractor")
        self.setGeometry(100, 100, 1200, 800)
        
        # Apply theme stylesheet
        self.setStyleSheet(get_stylesheet()) 
        
        self.thread_pool = QThreadPool() # Keep thread pool if needed for future complex tasks
        self.extractor_ready = False # Flag to indicate if core extractor is ready
        self.anpe_version = anpe_version_str # Store version string
        self.worker: Optional[ExtractionWorker] = None # For single processing
        self.batch_worker: Optional[BatchWorker] = None # For batch processing
        self.results: Optional[Dict[str, Any]] = None # To store last processing results for export

        self.setup_ui()
        
        # Start Background Extractor Initialization only if ANPEExtractor is real
        if 'ANPEExtractor' in globals() and ANPEExtractor.__module__ != 'builtins': # Check it's not the dummy
             self.start_background_initialization()
        else:
             self.status_bar.showMessage("ANPE Library Error! Processing disabled.", 0) # Persistent message
             self.process_button.setEnabled(False) # Disable processing
             logging.error("ANPE library not found or dummy class used. Initialization skipped.")

    def start_background_initialization(self):
        """Starts the background thread for ANPEExtractor initialization."""
        self.init_thread = QThread()
        self.init_worker = ExtractorInitializer()
        self.init_worker.moveToThread(self.init_thread)
        
        self.init_worker.initialized.connect(self.on_extractor_initialized)
        self.init_worker.error.connect(self.on_extractor_init_error)
        self.init_worker.models_missing.connect(self.on_models_missing)  # Connect new signal
        self.init_thread.started.connect(self.init_worker.run)
        
        # Clean up thread when worker finishes
        self.init_worker.initialized.connect(self.init_thread.quit)
        self.init_worker.error.connect(self.init_thread.quit)
        self.init_worker.models_missing.connect(self.init_thread.quit)  # Add quit for new signal

        self.status_bar.showMessage("Checking ANPE models...")
        self.init_thread.start()
    
    @pyqtSlot(object)
    def on_extractor_initialized(self, extractor_instance):
        """Slot to receive the initialized extractor (though not directly used anymore)."""
        # We don't store the instance globally anymore, config is applied per run.
        # Just mark as ready.
        logging.info("MAIN: Extractor initialized signal received.")
        self.extractor_ready = True 
        self.status_bar.showMessage("ANPE Ready", 3000)
        if hasattr(self, 'process_button'): # Check if button exists
             self.process_button.setEnabled(True) # Enable processing now

    @pyqtSlot(str)
    def on_extractor_init_error(self, error_message):
        """Slot to handle initialization errors."""
        logging.error(f"MAIN: Extractor init error signal received: {error_message}")
        QMessageBox.critical(self, "Initialization Error", 
                             f"Failed to initialize ANPE Extractor: {error_message}")
        self.status_bar.showMessage("ANPE Initialization Failed! Processing disabled.", 0) # Persistent
        if hasattr(self, 'process_button'): # Check if button exists
            self.process_button.setEnabled(False) # Disable processing

    @pyqtSlot()
    def on_models_missing(self):
        """Handle case where models are missing during initialization."""
        logging.info("MAIN: Models missing, launching setup wizard...")
        self.status_bar.showMessage("Required models not found. Opening setup wizard...")
        
        # Create and show setup wizard
        from anpe_gui.setup_wizard import SetupWizard
        wizard = SetupWizard(self)
        wizard.setup_complete.connect(self.on_setup_wizard_complete)
        wizard.setup_cancelled.connect(self.on_setup_wizard_cancelled)
        wizard.show()

    def on_setup_wizard_complete(self):
        """Handle successful completion of the setup wizard."""
        self.status_bar.showMessage("Model setup completed successfully. Reinitializing...", 5000)
        # Restart initialization to verify models and initialize extractor
        self.start_background_initialization()

    def on_setup_wizard_cancelled(self):
        """Handle cancellation of the setup wizard."""
        self.status_bar.showMessage("Model setup cancelled. ANPE functionality limited.", 5000)
        self.process_button.setEnabled(False)  # Disable processing since models aren't ready

    def setup_ui(self):
        """Set up the main UI components: Header, Splitter (Tabs | Log), Status Bar."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 0) # Margins for window edges
        self.main_layout.setSpacing(10)
        
        # 1. Header
        self.setup_header()
        
        # 2. Main Content Splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 2a. Tab Widget (Left Pane)
        self.tab_widget = QTabWidget()
        self.input_tab = QWidget() # Create tab page widget
        self.setup_input_tab()     # Populate it
        self.tab_widget.addTab(self.input_tab, "Input")
        
        self.output_tab = QWidget() # Create tab page widget
        self.setup_output_tab()    # Populate it
        self.tab_widget.addTab(self.output_tab, "Output")
        self.main_splitter.addWidget(self.tab_widget) # Add tabs to splitter
        
        # 2b. Log Panel (Right Pane)
        self.log_panel = self.create_log_panel()
        self.main_splitter.addWidget(self.log_panel) # Add log panel to splitter
        
        # Configure Splitter
        self.main_splitter.setSizes([700, 300]) # Initial size ratio
        self.main_splitter.setStretchFactor(0, 7)  # Allow tabs to take more space
        self.main_splitter.setStretchFactor(1, 3)  # Log panel less space
        self.main_layout.addWidget(self.main_splitter, 1) # Add splitter to main layout (stretch)
        
        # 3. Status Bar
        self.status_bar = StatusBar(self) 
        self.main_layout.addWidget(self.status_bar) # Add status bar at the bottom

        # Set flag after all essential UI elements are created
        self.ui_setup_complete = True

    def setup_header(self):
        """Set up the application header with text title and version."""
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 2, 0, 2) 
        
        # Title Area (Horizontal Layout for title and version)
        title_area_widget = QWidget()
        title_layout = QVBoxLayout(title_area_widget)
        title_layout.setContentsMargins(0,0,0,0)
        title_layout.setSpacing(0)
        
        # Title row (ANPE + Version)
        title_row = QHBoxLayout()
        title_row.setSpacing(5)  # Small gap between title and version
        
        # Main Title: ANPE
        title_label = QLabel()
        title_label.setText(f'<b style="color:{PRIMARY_COLOR}; font-size: 18pt;">ANPE</b>')
        title_row.addWidget(title_label)
        
        # Version Label (next to ANPE)
        version_label = QLabel(f"v{self.anpe_version}")
        version_label.setStyleSheet("color: grey; font-size: 10pt;")
        title_row.addWidget(version_label)
        title_row.addStretch()
        
        # Add title row to layout
        title_layout.addLayout(title_row)
        
        # Subtitle
        subtitle_label = QLabel()
        subtitle_label.setText('<i style="color: grey; font-size: 9pt;">Another Noun Phrase Extractor</i>')
        title_layout.addWidget(subtitle_label)
        
        # Add title area to header layout (left-aligned)
        header_layout.addWidget(title_area_widget)
        
        # Add stretch to push setup button to the right
        header_layout.addStretch()
        
        # Setup Models Button (right-aligned)
        settings_button = QPushButton("Setup Models")
        settings_button.setToolTip("Check or reinstall language models")
        settings_button.clicked.connect(self.open_setup_wizard)
        settings_button.setStyleSheet(f"""
            QPushButton {{
                padding: 8px 15px;
                border: none;
                border-radius: 4px;
                background-color: {PRIMARY_COLOR};
                color: white;
                font-size: 10pt;
            }}
            QPushButton:hover {{
                background-color: #005fb8;
            }}
            QPushButton:pressed {{
                background-color: #004a94;
            }}
        """)
        header_layout.addWidget(settings_button)
        
        # Add header to main layout
        self.main_layout.addWidget(header_container)

    def setup_input_tab(self):
        """Set up the Input tab: Mode buttons, Stacked Input Area, Config, Process Button."""
        # Main layout for the input tab page
        self.input_layout = QVBoxLayout(self.input_tab) 
        self.input_layout.setContentsMargins(15, 15, 15, 15) # Padding around tab content
        self.input_layout.setSpacing(10)

        # --- 1. Input Mode Selection ---
        input_mode_layout = QHBoxLayout()
        self.file_input_button = QPushButton("File Input")
        self.file_input_button.setCheckable(True)
        self.text_input_button = QPushButton("Text Input")
        self.text_input_button.setCheckable(True)
        
        self.input_mode_group = QButtonGroup(self)
        self.input_mode_group.addButton(self.file_input_button, 0) # ID 0 = File mode
        self.input_mode_group.addButton(self.text_input_button, 1) # ID 1 = Text mode
        self.input_mode_group.setExclusive(True)
        self.input_mode_group.idClicked.connect(self.switch_input_mode) # Use idClicked signal
        
        input_mode_layout.addWidget(self.file_input_button)
        input_mode_layout.addWidget(self.text_input_button)
        input_mode_layout.addStretch()
        self.input_layout.addLayout(input_mode_layout)

        # --- 2. Input Area (Stacked Widget) ---
        self.input_stack = QStackedWidget()
        
        # Page 0: File Input
        file_input_page = QWidget() # Container widget for this page
        file_page_layout = QVBoxLayout(file_input_page)
        file_page_layout.setContentsMargins(0,0,0,0) # No margins within page
        file_page_layout.setSpacing(5)
        self.file_list_widget = FileListWidget() # The dedicated file list widget
        self.file_list_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        file_page_layout.addWidget(self.file_list_widget)
        self.input_stack.addWidget(file_input_page) # Add page to stack

        # Page 1: Text Input
        text_input_page = QWidget() # Container widget for this page
        text_page_layout = QVBoxLayout(text_input_page)
        text_page_layout.setContentsMargins(0,0,0,0)
        text_page_layout.setSpacing(5)
        self.direct_text_input = QTextEdit() # The main text edit area
        self.direct_text_input.setPlaceholderText("Paste or type text here for processing...")
        self.direct_text_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.direct_text_input.setMinimumHeight(150) 
        text_page_layout.addWidget(self.direct_text_input, 1) # Allow text area to stretch
        
        # Text action buttons (Paste, Clear)
        text_button_layout = QHBoxLayout()
        paste_button = QPushButton("Paste")
        clear_button = QPushButton("Clear Text")
        paste_button.clicked.connect(self.direct_text_input.paste) # Use built-in paste
        clear_button.clicked.connect(self.direct_text_input.clear) # Use built-in clear
        text_button_layout.addStretch()
        text_button_layout.addWidget(paste_button)
        text_button_layout.addWidget(clear_button)
        text_page_layout.addLayout(text_button_layout)
        self.input_stack.addWidget(text_input_page) # Add page to stack

        self.input_layout.addWidget(self.input_stack, 3) # Changed stretch factor from 2 to 3

        # Set initial state: File input selected
        self.file_input_button.setChecked(True)
        self.input_stack.setCurrentIndex(0) 

        # --- 3. Configuration Section ---
        config_container = QWidget() # Container for all config group boxes
        config_layout = QVBoxLayout(config_container)
        config_layout.setContentsMargins(0,0,0,0)
        config_layout.setSpacing(10)

        # --- GroupBox 1: General Settings ---
        settings_group = QGroupBox("General Settings")
        settings_layout = QGridLayout(settings_group)
        # Increased horizontal spacing
        settings_layout.setHorizontalSpacing(25) 
        settings_layout.setVerticalSpacing(5)

        self.include_nested = QCheckBox("Include Nested Phrases")
        self.include_nested.setChecked(True)
        self.include_nested.setToolTip("Whether to include nested noun phrases (maintains parent-child relationships).")
        settings_layout.addWidget(self.include_nested, 0, 0)

        self.include_metadata = QCheckBox("Include Metadata")
        self.include_metadata.setChecked(True)
        self.include_metadata.setToolTip("Whether to include metadata about each noun phrase (length and structural analysis).")
        settings_layout.addWidget(self.include_metadata, 0, 1)

        self.newline_breaks = QCheckBox("Do not treat newlines as boundaries")
        self.newline_breaks.setChecked(False)
        self.newline_breaks.setToolTip("When unchecked (default), newlines are treated as sentence boundaries.\nWhen checked, newlines are ignored (useful for irregular formatting).")
        settings_layout.addWidget(self.newline_breaks, 0, 2)

        # Column stretches for alignment
        settings_layout.setColumnStretch(0, 0)
        settings_layout.setColumnStretch(1, 0)
        settings_layout.setColumnStretch(2, 1)
        config_layout.addWidget(settings_group)

        # --- GroupBox 2: General Filtering Options ---
        filtering_group = QGroupBox("General Filtering Options")
        filtering_layout = QGridLayout(filtering_group)
        # Increased horizontal spacing (must match above)
        filtering_layout.setHorizontalSpacing(25) 
        filtering_layout.setVerticalSpacing(5)

        # Min Length HBox
        min_length_hbox = QHBoxLayout()
        min_length_hbox.setSpacing(5) # Tight spacing within HBox
        self.min_length_checkbox = QCheckBox("Min length")
        self.min_length_checkbox.setToolTip("Filter out noun phrases shorter than this length (in tokens). Check to enable.")
        self.min_length_spinbox = QSpinBox()
        self.min_length_spinbox.setRange(1, 100)
        self.min_length_spinbox.setValue(2)
        self.min_length_spinbox.setEnabled(False)
        self.min_length_spinbox.setMaximumWidth(70) 
        self.min_length_checkbox.toggled.connect(self.min_length_spinbox.setEnabled)
        min_length_hbox.addWidget(self.min_length_checkbox)
        min_length_hbox.addWidget(self.min_length_spinbox)
        min_length_hbox.addStretch(1) # Push label/spinbox together
        filtering_layout.addLayout(min_length_hbox, 0, 0)

        # Max Length HBox
        max_length_hbox = QHBoxLayout()
        max_length_hbox.setSpacing(5) # Tight spacing within HBox
        self.max_length_checkbox = QCheckBox("Max length")
        self.max_length_checkbox.setToolTip("Filter out noun phrases longer than this length (in tokens). Check to enable.")
        self.max_length_spinbox = QSpinBox()
        self.max_length_spinbox.setRange(1, 100)
        self.max_length_spinbox.setValue(10) 
        self.max_length_spinbox.setEnabled(False)
        self.max_length_spinbox.setMaximumWidth(70) 
        self.max_length_checkbox.toggled.connect(self.max_length_spinbox.setEnabled)
        max_length_hbox.addWidget(self.max_length_checkbox)
        max_length_hbox.addWidget(self.max_length_spinbox)
        max_length_hbox.addStretch(1) # Push label/spinbox together
        filtering_layout.addLayout(max_length_hbox, 0, 1)

        # Accept Pronouns Checkbox
        self.accept_pronouns = QCheckBox("Accept Pronouns")
        self.accept_pronouns.setChecked(True)
        self.accept_pronouns.setToolTip("Whether to include single-word pronouns (e.g., 'it', 'they') as valid noun phrases.")
        filtering_layout.addWidget(self.accept_pronouns, 0, 2)

        # IDENTICAL Column stretches for alignment
        filtering_layout.setColumnStretch(0, 0)
        filtering_layout.setColumnStretch(1, 0)
        filtering_layout.setColumnStretch(2, 1)
        config_layout.addWidget(filtering_group)
        
        # Structure Filtering GroupBox (containing the custom widget)
        structure_filtering_container_group = QGroupBox("Structure Filtering") # Visual container
        structure_filtering_layout = QVBoxLayout(structure_filtering_container_group)
        structure_filtering_layout.setContentsMargins(5,5,5,5) # Inner padding
        self.structure_filter_widget = StructureFilterWidget() # The widget with the toggle logic
        structure_filtering_layout.addWidget(self.structure_filter_widget)
        config_layout.addWidget(structure_filtering_container_group)
        
        self.input_layout.addWidget(config_container) # Add config area to input tab layout

        # --- 4. Process Button ---
        process_reset_layout = QHBoxLayout()
        process_reset_layout.addStretch(1)
        
        # Reset Button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet(f"""
            QPushButton {{
                padding: 8px 15px;
                font-size: 10pt;
                border: none;
                border-radius: 4px;
                background-color: {PRIMARY_COLOR};
                color: white;
            }}
            QPushButton:hover {{
                background-color: #005fb8;
            }}
            QPushButton:pressed {{
                background-color: #004a94;
            }}
        """)
        self.reset_button.clicked.connect(self.reset_workflow)
        process_reset_layout.addWidget(self.reset_button)

        # Process Button
        self.process_button = QPushButton("Process")
        self.process_button.setStyleSheet(f"""
            QPushButton {{
                padding: 8px 15px;
                font-size: 10pt;
                border: none;
                border-radius: 4px;
                background-color: {PRIMARY_COLOR};
                color: white;
            }}
            QPushButton:hover {{
                background-color: #005fb8;
            }}
            QPushButton:pressed {{
                background-color: #004a94;
            }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
        """)
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)
        process_reset_layout.addWidget(self.process_button)
        
        self.input_layout.addLayout(process_reset_layout)

    def setup_output_tab(self):
        """Set up the Output tab: Results display, Export options, Navigation."""
        # Main layout for the output tab page
        self.output_layout = QVBoxLayout(self.output_tab) 
        self.output_layout.setContentsMargins(15, 15, 15, 15)
        self.output_layout.setSpacing(15)

        # --- File Selector (only visible for multi-file results) ---
        self.file_selector_layout = QHBoxLayout()
        self.file_selector_label = QLabel("Displaying results for:")
        self.file_selector_combo = QComboBox()
        self.file_selector_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.file_selector_combo.currentIndexChanged.connect(self.display_selected_file_result)
        self.file_selector_layout.addWidget(self.file_selector_label)
        self.file_selector_layout.addWidget(self.file_selector_combo, 1) # Give combo stretch
        self.output_layout.addLayout(self.file_selector_layout)
        self.file_selector_label.hide() # Hidden initially
        self.file_selector_combo.hide() # Hidden initially

        # --- Results Display Area ---
        results_group = QGroupBox("Extraction Results")
        results_group_layout = QVBoxLayout(results_group)
        self.results_text = QTextEdit() # Standardized name
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(300) 
        self.results_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap) # Prevent wrapping
        results_group_layout.addWidget(self.results_text)
        self.output_layout.addWidget(results_group, 1) # Allow results area to stretch

        # --- Export Options ---
        export_group = QGroupBox("Export Options")
        export_layout = QGridLayout(export_group)
        export_layout.setSpacing(10)

        # Export Format
        export_layout.addWidget(QLabel("Format:"), 0, 0)
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["txt", "csv", "json"]) # Add more formats if supported
        export_layout.addWidget(self.export_format_combo, 0, 1)

        # Export Directory
        export_layout.addWidget(QLabel("Directory:"), 1, 0)
        export_dir_layout = QHBoxLayout()
        self.export_dir_edit = QLineEdit() # Standardized name
        self.export_dir_edit.setPlaceholderText("Select export directory...")
        self.export_dir_edit.setReadOnly(True) # Prevent manual editing
        self.browse_export_dir_button = QPushButton("Browse...")
        self.browse_export_dir_button.clicked.connect(self.browse_export_directory)
        export_dir_layout.addWidget(self.export_dir_edit, 1) # Give edit stretch
        export_dir_layout.addWidget(self.browse_export_dir_button)
        export_layout.addLayout(export_dir_layout, 1, 1)

        # Export Button (aligned right)
        export_button_layout = QHBoxLayout()
        export_button_layout.addStretch(1)
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results) 
        self.export_button.setEnabled(False) # Enable only when results are available
        export_button_layout.addWidget(self.export_button)
        # Add the button layout spanning columns in the grid
        export_layout.addLayout(export_button_layout, 2, 0, 1, 2) 

        self.output_layout.addWidget(export_group)

        # --- Navigation Buttons ---
        nav_button_layout = QHBoxLayout()
        # Back button removed for simplicity, users can just click the Input tab
        # self.back_to_input_button = QPushButton("Back to Input") ...
        
        # Process New Button (acts as reset)
        self.process_new_button = QPushButton("Process New Input")
        self.process_new_button.clicked.connect(self.reset_workflow)

        nav_button_layout.addStretch(1) # Push button to the right
        nav_button_layout.addWidget(self.process_new_button)
        self.output_layout.addLayout(nav_button_layout)

    # --- Input/Processing Logic ---

    @pyqtSlot(int)
    def switch_input_mode(self, index):
        """Switch the QStackedWidget index based on the button ID clicked."""
        self.input_stack.setCurrentIndex(index)
        logging.debug(f"Switched input mode to index: {index}")

    def apply_configuration(self) -> Optional[Dict[str, Any]]:
        """Gather configuration from UI elements and return config dict for ANPEExtractor."""
        # Initialize config dictionary at the beginning
        config = {} 

        if not self.ui_setup_complete:
            logging.warning("apply_configuration called before UI setup is complete.")
            return None 

        try:
            # General Filtering Options -> ANPEExtractor config
            if self.min_length_checkbox.isChecked():
                config["min_length"] = self.min_length_spinbox.value()
            if self.max_length_checkbox.isChecked():
                config["max_length"] = self.max_length_spinbox.value()
            config["accept_pronouns"] = self.accept_pronouns.isChecked()
            config["newline_breaks"] = not self.newline_breaks.isChecked()

            # Structure Filtering Options -> ANPEExtractor config
            if hasattr(self, 'structure_filter_widget') and self.structure_filter_widget:
                if self.structure_filter_widget.is_filtering_enabled():
                    selected_structures = self.structure_filter_widget.get_selected_structures()
                    # Pass filters only if enabled and at least one is selected
                    # Otherwise, ANPE core gets no filter key, meaning include all.
                    if selected_structures:
                        config["structure_filters"] = selected_structures
            else:
                logging.warning("Structure filter widget not found/ready during config gathering.")

            logging.info(f"Configuration gathered successfully: {config}")
            return config

        except AttributeError as ae:
             # Catch specific errors related to UI elements not existing yet
             logging.error(f"Error gathering configuration - UI element missing?: {ae}", exc_info=True)
             QMessageBox.warning(self, "Config Error", f"Failed to gather configuration (UI Error): {ae}")
             return None
        except Exception as e:
            logging.error(f"Error gathering configuration: {e}", exc_info=True)
            QMessageBox.warning(self, "Config Error", f"Failed to gather configuration: {e}")
            return None

    def start_processing(self):
        """Main entry point when the 'Process' button is clicked."""
        if not self.extractor_ready:
            QMessageBox.warning(self, "Initialization Error", 
                                "ANPE Extractor is not ready yet. Please wait or check logs.")
            return

        # 1. Get Configuration
        config = self.apply_configuration()
        if config is None:
            # Error message already shown by apply_configuration
            return
            
        # 2. Get Input Data based on mode
        input_mode_index = self.input_stack.currentIndex()
        if input_mode_index == 0: # File Input Mode
            files = self.file_list_widget.get_files()
            if not files:
                QMessageBox.warning(self, "No Input", "Please select at least one file.")
                return
            # 3. Run Batch Processing
            self.log_panel.clear_log() # Clear log for new run
            self.log(f"Starting batch processing for {len(files)} files...")
            self.status_bar.showMessage(f"Processing {len(files)} files...")
            self.run_batch_processing(files, config)

        elif input_mode_index == 1: # Text Input Mode
            text_content = self.direct_text_input.toPlainText().strip()
            if not text_content:
                QMessageBox.warning(self, "No Input", "Please enter text to process.")
                return
            # 3. Run Single Processing
            self.log_panel.clear_log() # Clear log for new run
            self.log("Starting single text processing...")
            self.status_bar.showMessage("Processing text...")
            self.run_single_processing(text_content, config)

    def run_single_processing(self, text_content: str, config: Dict[str, Any]):
        """Starts the background worker for processing a single text block."""
        if hasattr(self, 'single_thread') and self.single_thread is not None and self.single_thread.isRunning():
            QMessageBox.warning(self, "Processing Busy", 
                                "Another process is already running. Please wait.")
            return
            
        self.results = None 
        self.results_text.clear() 
        self.export_button.setEnabled(False) 
        self.process_button.setEnabled(False) 
        self.status_bar.start_progress("Processing text...") 
        self.processing_error_occurred = False # Initialize error flag

        # 1. Create Worker
        self.worker = ExtractionWorker(
            text_content=text_content, 
            config=config, 
            anpe_version=self.anpe_version,
            include_nested=self.include_nested.isChecked(),
            include_metadata=self.include_metadata.isChecked()
        ) 
        
        # 2. Create Thread and Move Worker
        self.single_thread = QThread() 
        self.worker.moveToThread(self.single_thread)

        # 3. Connect Signals
        self.single_thread.started.connect(self.worker.run)
        # --- Worker Signals ---
        self.worker.signals.result.connect(self.handle_single_result)
        self.worker.signals.error.connect(self.handle_error)
        # Use partial to pass worker type identifier
        finish_slot_single = functools.partial(self.processing_finished, worker_type='single')
        self.worker.signals.finished.connect(finish_slot_single)
        self.worker.signals.finished.connect(self.worker.deleteLater)
        self.worker.signals.finished.connect(self.single_thread.quit)
        # --- Thread Signals ---
        self.single_thread.finished.connect(self.clear_single_thread_reference)
        self.single_thread.finished.connect(self.single_thread.deleteLater)
        
        # 4. Start the Thread
        self.single_thread.start() 

    def run_batch_processing(self, file_paths: List[str], config: Dict[str, Any]):
        """Starts the background worker for processing multiple files."""
        if hasattr(self, 'batch_thread') and self.batch_thread is not None and self.batch_thread.isRunning():
             QMessageBox.warning(self, "Processing Busy", 
                                 "Another process is already running. Please wait.")
             return

        self.results = {} 
        self.results_text.clear() 
        self.file_selector_combo.clear() 
        self.file_selector_label.hide()
        self.file_selector_combo.hide()
        self.export_button.setEnabled(False) 
        self.process_button.setEnabled(False) 
        self.status_bar.update_progress(0, f"Starting batch processing (0/{len(file_paths)} files)..." )
        self.processing_error_occurred = False # Initialize error flag

        # 1. Create Worker
        self.batch_worker = BatchWorker(
            file_paths=file_paths, 
            config=config, 
            anpe_version=self.anpe_version,
            include_nested=self.include_nested.isChecked(),
            include_metadata=self.include_metadata.isChecked()
        )
        
        # 2. Create Thread and Move Worker
        self.batch_thread = QThread() 
        self.batch_worker.moveToThread(self.batch_thread)

        # 3. Connect Signals
        self.batch_thread.started.connect(self.batch_worker.run)
        # --- Worker Signals ---
        self.batch_worker.signals.progress.connect(self.update_batch_progress)
        self.batch_worker.signals.file_result.connect(self.handle_batch_file_result)
        self.batch_worker.signals.error.connect(self.handle_error)
        # Use partial to pass worker type identifier
        finish_slot_batch = functools.partial(self.processing_finished, worker_type='batch')
        self.batch_worker.signals.finished.connect(finish_slot_batch)
        self.batch_worker.signals.finished.connect(self.batch_worker.deleteLater)
        self.batch_worker.signals.finished.connect(self.batch_thread.quit)
        # --- Thread Signals ---
        self.batch_thread.finished.connect(self.clear_batch_thread_reference)
        self.batch_thread.finished.connect(self.batch_thread.deleteLater)

        # 4. Start the Thread
        self.batch_thread.start() 

    # --- Worker Signal Handlers ---

    @pyqtSlot(int, str) # Receives percentage and message
    def update_batch_progress(self, percentage: int, message: str):
        """Update status bar for batch processing progress."""
        self.status_bar.update_progress(percentage, message)

    @pyqtSlot(str) # Receives message
    def update_progress(self, message: str):
         """Update status bar for indeterminate progress (single text)."""
         # For indeterminate, we just update the message
         self.status_bar.showMessage(message)

    @pyqtSlot(dict) # Receives result dictionary for one run
    def handle_single_result(self, result_data: Dict[str, Any]):
        """Handle the result from the ExtractionWorker (single text)."""
        self.results = result_data # Store the complete result
        self.display_results(result_data) # Display it
        self.export_button.setEnabled(True) # Enable export
        self.log("Single text processing completed.")
        # processing_finished will handle status bar final message

    @pyqtSlot(str, dict) # Receives file path and its result dictionary
    def handle_batch_file_result(self, file_path: str, result_data: Dict[str, Any]):
        """Handle the result for a single file from the BatchWorker."""
        if self.results is None: self.results = {} # Ensure results dict exists
        
        self.results[file_path] = result_data # Store result for this file
        
        # Populate combo box as results come in
        base_name = os.path.basename(file_path)
        self.file_selector_combo.addItem(base_name, file_path) # Display name, store full path
        
        # If this is the first result, display it and show the combo box
        if len(self.results) == 1:
             self.display_results(result_data)
             self.file_selector_label.show()
             self.file_selector_combo.show()
             
        self.log(f"Processed file: {base_name}")
        # Status bar updated via update_batch_progress signal

    @pyqtSlot(str) # Receives error message string
    def handle_error(self, error_message: str):
        """Handle errors reported by workers."""
        logging.error(f"Processing Error: {error_message}") 
        self.status_bar.stop_progress(f"Error: {error_message}") # Show error in status bar
        QMessageBox.warning(self, "Processing Error", f"An error occurred: {error_message}")
        self.processing_error_occurred = True # Set error flag
        # Re-enable process button even on error? (Keep current behavior)
        # self.process_button.setEnabled(True) 

    @pyqtSlot(str)
    def processing_finished(self, worker_type: str):
        """Handle UI updates when a worker finishes, identified by worker_type."""
        logging.info(f"Worker finished signal received for worker type: {worker_type}")
        
        worker_cleared = False

        # Clear the appropriate worker reference based on the argument
        if worker_type == 'single' and self.worker is not None:
            logging.debug("Clearing reference for Single worker.")
            self.worker = None
            worker_cleared = True
        elif worker_type == 'batch' and self.batch_worker is not None:
            logging.debug("Clearing reference for Batch worker.")
            self.batch_worker = None
            worker_cleared = True
        else:
            # This can happen if signal arrives after reference is already cleared
            logging.debug(f"processing_finished ({worker_type}): Worker reference already None. Ignoring UI update.")
            return

        # --- Proceed with UI updates only if we cleared a worker reference --- 
        if worker_cleared:
            logging.debug("Updating UI after processing finished.")
            
            # Determine final message using the flag
            final_message = "Processing complete"
            if self.processing_error_occurred: 
                 final_message = "Processing finished with errors"
            elif self.results is None or (isinstance(self.results, dict) and not self.results):
                final_message = "Processing finished (No results)"
                
            self.status_bar.clear_progress() 
            self.status_bar.showMessage(final_message, 5000) 
            
            # Enable export only if results exist AND no error occurred
            can_export = self.results is not None and not self.processing_error_occurred
            self.export_button.setEnabled(can_export)
            
            # Switch to output tab only on successful completion with results
            if can_export: # Use can_export flag which implies success and results exist
                 self.switch_to_output_tab()
            
            # Disable process button after run, enable reset
            if hasattr(self, 'process_button'):
                self.process_button.setEnabled(False) 
            if hasattr(self, 'reset_button'):
                 self.reset_button.setEnabled(True) # Ensure Reset is enabled
            
            logging.debug("Processing finished UI updates complete.")

    @pyqtSlot()
    def clear_single_thread_reference(self):
        """Slot called when the single processing thread finishes."""
        logging.debug("Single processing thread finished. Clearing reference.")
        if hasattr(self, 'single_thread'):
            self.single_thread = None
            
    @pyqtSlot()
    def clear_batch_thread_reference(self):
        """Slot called when the batch processing thread finishes."""
        logging.debug("Batch processing thread finished. Clearing reference.")
        if hasattr(self, 'batch_thread'):
            self.batch_thread = None

    # --- Output Tab Logic ---

    @pyqtSlot()
    def display_selected_file_result(self):
        """Display the results for the file selected in the combo box (batch mode)."""
        selected_file_path = self.file_selector_combo.currentData() # Get stored full path
        if selected_file_path and isinstance(self.results, dict) and selected_file_path in self.results:
            self.display_results(self.results[selected_file_path])
            logging.debug(f"Displayed results for selected file: {selected_file_path}")
        else:
            # This might happen if combo index changes before results are fully populated
            logging.warning(f"Could not find results for selected file: {selected_file_path}")
            # self.results_text.setText("Could not load results for selected file.") # Avoid clearing potentially valid results

    def display_results(self, data: Dict[str, Any]):
        """Populate the results text area with formatted data."""
        self.results_text.clear()
        if not data or not isinstance(data, dict) or 'metadata' not in data or 'results' not in data:
            self.results_text.setText("No valid results data to display.")
            logging.warning(f"Invalid or empty data passed to display_results: {data}")
            return
            
        try:
            metadata = data.get('metadata', {})
            np_results = data.get('results', [])

            self.results_text.append("<b>ANPE Noun Phrase Extraction Results</b>")
            self.results_text.append("==================================")
            self.results_text.append(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
            # These flags now come from the result metadata itself
            self.results_text.append(f"Includes Nested NPs: {metadata.get('includes_nested', 'N/A')}")
            self.results_text.append(f"Includes Metadata: {metadata.get('includes_metadata', 'N/A')}")
            self.results_text.append("")
            
            if not np_results:
                self.results_text.append("--- No noun phrases extracted matching the criteria. ---")
            else:
                self.format_np_for_display(np_results) # Start recursive formatting
                
            self.results_text.moveCursor(QTextCursor.MoveOperation.Start) # Scroll to top
            logging.debug("Results displayed successfully.")
        except Exception as e:
             logging.error(f"Error formatting results for display: {e}", exc_info=True)
             self.results_text.setText(f"Error displaying results: {e}")

    def format_np_for_display(self, np_items: List[Dict[str, Any]], level: int = 0):
        """Recursively format noun phrases for the QTextEdit, using simple text."""
        # Simple text formatting, no HTML here for performance with large results
        for np_item in np_items:
            bullet = "*" if level == 0 else "-" # Use simple bullets
            indent = "  " * level
            
            np_text = np_item.get('noun_phrase', 'N/A')
            np_id = np_item.get('id', 'N/A')
            self.results_text.append(f"{indent}{bullet} [{np_id}] {np_text}")
            
            metadata = np_item.get("metadata", {})
            if metadata:
                meta_parts = []
                if "length" in metadata:
                    meta_parts.append(f"Len: {metadata['length']}")
                if "structures" in metadata:
                    structures = metadata['structures']
                    # Ensure structures is a list or tuple before joining
                    if isinstance(structures, (list, tuple)):
                        structures_str = ", ".join(structures)
                    elif isinstance(structures, str): # Handle case where it might be a single string
                        structures_str = structures
                    else:
                        structures_str = str(structures) # Fallback
                    meta_parts.append(f"Struct: [{structures_str}]")
                
                if meta_parts:
                     self.results_text.append(f"{indent}  ({', '.join(meta_parts)})") # Combine metadata on one line
            
            # Recursive call for children
            children = np_item.get("children")
            if children and isinstance(children, list): # Check if children exist and is a list
                self.format_np_for_display(children, level + 1)

    def export_results(self):
        """Export the currently stored results to a file."""
        if self.results is None:
            QMessageBox.warning(self, "Export Error", "No extraction results available to export.")
            return
            
        export_dir = self.export_dir_edit.text()
        if not export_dir or not os.path.isdir(export_dir): # Check if dir exists
            QMessageBox.warning(self, "Export Error", "Please select a valid export directory.")
            # Try browsing for directory automatically if none selected
            self.browse_export_directory() 
            export_dir = self.export_dir_edit.text() # Get potentially updated dir
            if not export_dir or not os.path.isdir(export_dir): # Check again
                return
            
        export_format = self.export_format_combo.currentText()
        logging.info(f"Attempting export. Format: {export_format}, Dir: {export_dir}")

        try:
            from anpe.utils.export import ANPEExporter # Import here to avoid top-level dependency if export not used
            exporter = ANPEExporter()
            
            export_successful = False
            # Check if results are from batch processing (dict of file_path: result_data)
            # or single processing (single result_data dict)
            if isinstance(self.results, dict) and all(isinstance(k, str) and isinstance(v, dict) for k, v in self.results.items()):
                # Likely Batch results: keys are file paths
                num_files = len(self.results)
                logging.info(f"Exporting batch results for {num_files} files.")
                for file_path, result_data in self.results.items():
                    # Create a base filename from the original input file path
                    base_name = Path(file_path).stem 
                    # Handle potential collisions or default naming if needed
                    output_filename = f"{base_name}_anpe_results.{export_format}" 
                    full_export_path = os.path.join(export_dir, output_filename)
                    
                    logging.debug(f"Exporting '{file_path}' results to '{full_export_path}'")
                    exporter.export(result_data, format=export_format, export_dir=export_dir)
                
                export_successful = True
                message = f"Results for {num_files} files exported successfully to {export_dir}"
                
            elif isinstance(self.results, dict) and 'results' in self.results:
                # Likely Single text result (has 'results' key)
                logging.info("Exporting single text results.")
                 # Use a default filename for text input results
                output_filename = f"anpe_text_results.{export_format}" 
                full_export_path = os.path.join(export_dir, output_filename)
                
                logging.debug(f"Exporting single text result to '{full_export_path}'")
                exporter.export(self.results, format=export_format, export_dir=export_dir)
                
                export_successful = True
                message = f"Results exported successfully to {full_export_path}"
            else:
                # Unknown results format
                 logging.error(f"Cannot export. Unknown results format: {type(self.results)}")
                 QMessageBox.warning(self, "Export Error", "Cannot export results. Unknown data format.")
                 return

            if export_successful:
                logging.info(message)
                QMessageBox.information(self, "Export Successful", message)
            
        except ImportError:
             logging.error("Export failed: ANPEExporter not found. Is 'anpe' installed correctly?")
             QMessageBox.critical(self, "Export Error", "Export functionality requires the full 'anpe' library to be installed.")
        except Exception as e:
            logging.error(f"Error exporting results: {e}", exc_info=True)
            QMessageBox.warning(self, "Export Error", f"Error exporting results: {e}")

    def reset_workflow(self):
        """Reset the UI to start a new processing task."""
        logging.info("Resetting workflow...")
        # Clear input areas
        self.file_list_widget.clear_files()
        self.direct_text_input.clear()
        
        # Clear results area and stored results
        self.results_text.clear()
        self.results = None
        
        # Hide file selector, disable export
        self.file_selector_label.hide()
        self.file_selector_combo.hide()
        self.file_selector_combo.clear()
        self.export_button.setEnabled(False)
        
        # Reset tabs and input mode to default (File Input on Input tab)
        self.tab_widget.setCurrentIndex(0) 
        self.file_input_button.setChecked(True) # Also triggers switch_input_mode via button group
        self.input_stack.setCurrentIndex(0)
        
        # Reset status bar
        self.status_bar.showMessage("Ready")
        self.status_bar.clear_progress()
        
        # Re-enable process button if extractor is ready, disable reset again?
        if hasattr(self, 'process_button'):
            self.process_button.setEnabled(self.extractor_ready)
        if hasattr(self, 'reset_button'):
             self.reset_button.setEnabled(False) # Optional: Disable reset until next run finishes?
             # Or keep it enabled: self.reset_button.setEnabled(True)
        
        logging.info("Workflow reset.")

    # --- Utility and Helper Methods ---

    def create_log_panel(self) -> EnhancedLogPanel:
        """Create the log panel and configure the QtLogHandler."""
        log_widget = EnhancedLogPanel()
        
        # Store handler instance for removal on close
        self.qt_log_handler_instance = QtLogHandler() 
        self.qt_log_handler_instance.log_signal.connect(log_widget.add_log_entry)
        
        # Configure Python's root logger
        logger = logging.getLogger() 
        # Remove existing handlers if any (optional, prevents duplicate logs on reload)
        # for handler in logger.handlers[:]: logger.removeHandler(handler) 
        logger.addHandler(self.qt_log_handler_instance) 
        logger.setLevel(logging.DEBUG) # Set desired logging level
        
        logging.info("Logging initialized.")
        return log_widget
    
    def log(self, message: str, level: int = logging.INFO):
        """Helper method to log messages via Python's logging."""
        logging.log(level, message)

    def browse_export_directory(self):
        """Open directory dialog and set the export directory line edit."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Export Directory", self.export_dir_edit.text() # Start in current dir if set
        )
        if dir_path:
            self.export_dir_edit.setText(dir_path)
            logging.debug(f"Export directory selected: {dir_path}")

    def switch_to_output_tab(self):
         """Switch focus to the Output tab."""
         self.tab_widget.setCurrentIndex(1)

    def open_setup_wizard(self):
        """Open the setup wizard for model management."""
        from anpe_gui.setup_wizard import SetupWizard
        
        wizard = SetupWizard(self)
        wizard.setup_complete.connect(self.on_setup_wizard_complete)
        wizard.setup_cancelled.connect(self.on_setup_wizard_cancelled)
        wizard.show()

    def closeEvent(self, event):
        """Handle the window close event gracefully."""
        logging.info("Close event triggered")
        
        # 1. Stop background threads (Initialization, Workers, Worker Threads)
        logging.debug("Stopping background threads...")
        try:
            # Stop Initializer Thread
            if hasattr(self, 'init_thread') and self.init_thread and self.init_thread.isRunning():
                logging.debug("Quitting Initializer thread...")
                self.init_thread.quit()
                if not self.init_thread.wait(500): 
                    logging.warning("Initializer thread did not finish cleanly.")
                else:
                    logging.debug("Initializer thread finished.")

            # Stop Single Processing Worker Thread (and worker if it exists)
            if hasattr(self, 'single_thread') and self.single_thread is not None:
                if self.single_thread.isRunning():
                    logging.debug("Quitting Single Processing thread...")
                    # If worker still exists, try asking it to stop first? (May not be necessary)
                    # if self.worker: self.worker.cancel() # Example 
                    self.single_thread.quit()
                    if not self.single_thread.wait(500):
                        logging.warning("Single Processing thread did not finish cleanly.")
                    else:
                        logging.debug("Single Processing thread finished.")
                # Regardless of running, clear reference if it exists
                self.single_thread = None 
            # Also clear worker ref just in case
            if hasattr(self, 'worker') and self.worker is not None:
                 logging.debug("Clearing dangling worker reference on close.")
                 self.worker = None

            # Stop Batch Processing Worker Thread (and worker if it exists)
            if hasattr(self, 'batch_thread') and self.batch_thread is not None:
                if self.batch_thread.isRunning():
                    logging.debug("Quitting Batch Processing thread...")
                    if self.batch_worker: # Try to signal worker cancellation
                        try: self.batch_worker.cancel() 
                        except RuntimeError: pass # Ignore if worker already deleted
                    self.batch_thread.quit()
                    if not self.batch_thread.wait(500):
                        logging.warning("Batch Processing thread did not finish cleanly.")
                    else:
                        logging.debug("Batch Processing thread finished.")
                self.batch_thread = None
            if hasattr(self, 'batch_worker') and self.batch_worker is not None:
                 logging.debug("Clearing dangling batch_worker reference on close.")
                 self.batch_worker = None

        except RuntimeError as e:
             logging.error(f"RuntimeError stopping threads during close (likely already deleted): {e}")
        except Exception as e:
            logging.error(f"General error stopping threads during close: {e}", exc_info=True)

        # 2. Remove the log handler 
        if hasattr(self, 'qt_log_handler_instance') and self.qt_log_handler_instance:
            logging.debug("Removing log handler...")
            try:
                logging.getLogger().removeHandler(self.qt_log_handler_instance)
                self.qt_log_handler_instance = None # Clear the reference
                logging.info("Log handler removed.")
            except Exception as e:
                logging.error(f"Error removing log handler: {e}", exc_info=True)
        else:
            logging.debug("Log handler not found or already removed.")

        # 3. Accept the event to close the window
        logging.info("Accepting close event.")
        event.accept()

# Note: The main execution block should be in __main__.py or app.py
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     # Apply stylesheet etc here if needed globally
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec())

# Note: The main execution block should be in __main__.py or app.py
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     # Apply stylesheet etc here if needed globally
#     main_window = MainWindow()
#     main_window.show()
#     sys.exit(app.exec()) 