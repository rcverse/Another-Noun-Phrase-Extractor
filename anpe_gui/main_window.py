"""
Main window implementation for the ANPE GUI application.
"""

import os
import sys # Add this import
import logging
import functools # Import functools
from typing import Optional, Dict, Any, List, Union # Added Union
from pathlib import Path
from datetime import datetime # Added datetime
import platform # Added platform
import subprocess # Added subprocess

# Import version from dedicated module
from anpe_gui.version import __version__ as GUI_VERSION

# Corrected imports: QCoreApplication moved to QtCore
from PyQt6.QtCore import Qt, QThreadPool, pyqtSignal, QSize, QTimer, QObject, QThread, pyqtSlot, QPropertyAnimation, QEasingCurve, pyqtProperty, QUrl, QCoreApplication, QSettings # Added QSettings
from PyQt6.QtWidgets import (
    QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QFileDialog, QSpinBox,
    QCheckBox, QComboBox, QGroupBox, QFormLayout, QLineEdit,
    QProgressBar, QMessageBox, QSplitter, QFrame, QTabWidget, QGridLayout, QSizePolicy,
    QButtonGroup, QApplication, QToolButton
)
from PyQt6.QtGui import QIcon, QTextCursor, QScreen, QPixmap # Added QScreen and QPixmap

from .theme import PRIMARY_COLOR, SECONDARY_COLOR, SUCCESS_COLOR, ERROR_COLOR, WARNING_COLOR, INFO_COLOR, BORDER_COLOR, get_scroll_bar_style, LIGHT_HOVER_BLUE, TEXT_COLOR # Update the import
from .widgets.help_dialog import HelpDialog # Import the new dialog
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
                              StatusBar, EnhancedLogPanel, ResultDisplayWidget) # Ensure StatusBar is imported from widgets
from anpe_gui.theme import get_stylesheet # Import the function to get the stylesheet
from anpe_gui.widgets.settings_dialog import SettingsDialog # Import the new dialog

# --- Worker for Background Initialization ---
class ExtractorInitializer(QObject):
    # Emit dictionary with detailed status on success, or None if check fails before completion
    initialized = pyqtSignal(object) 
    error = pyqtSignal(str)          # Emit error message string

    def run(self):
        """Checks specific model availability and emits status or error."""
        logging.info("Performing initial detailed model status check...")
        MISSING_MODELS_PREFIX = "Missing required models: "
        status_dict = {
            'spacy_models': [],
            'benepar_models': [],
            'nltk_present': False,
            'error': None # Add error field to status dict
        }
        try:
            # Always try to import, even if models might be missing
            from anpe.utils.setup_models import check_nltk_models
            from anpe.utils.model_finder import find_installed_spacy_models, find_installed_benepar_models
            
            # Perform the checks
            status_dict['spacy_models'] = find_installed_spacy_models()
            status_dict['benepar_models'] = find_installed_benepar_models()
            status_dict['nltk_present'] = check_nltk_models(['punkt', 'punkt_tab'])

            logging.info(f"Initial detailed model status: {status_dict}")

            # Check if *minimum* requirements are met (any spacy, any benepar, nltk)
            has_spacy = len(status_dict['spacy_models']) > 0
            has_benepar = len(status_dict['benepar_models']) > 0
            has_nltk = status_dict['nltk_present']

            if has_spacy and has_benepar and has_nltk:
                logging.info("Required model types found. Initialization check successful.")
                self.initialized.emit(status_dict) # Emit detailed status
            else:
                # Minimum requirements not met, construct error message
                missing = []
                if not has_spacy: missing.append("spaCy model(s)")
                if not has_benepar: missing.append("Benepar model(s)")
                if not has_nltk: missing.append("NLTK data")
                error_msg = MISSING_MODELS_PREFIX + ", ".join(missing)
                logging.warning(error_msg + ". Setup needed.")
                status_dict['error'] = error_msg # Store error in status dict as well
                self.error.emit(error_msg) # Emit specific error message
                # Do NOT emit initialized signal here
                
        except ImportError as ie:
             error_msg = f"Initialization check failed: Could not import ANPE utilities ({ie}). Is ANPE installed?"
             logging.error(error_msg, exc_info=True)
             status_dict['error'] = error_msg
             # Emit None on initialized if check fails this way?
             # Let's stick to only emitting error signal for consistency
             # self.initialized.emit(None)
             self.error.emit(error_msg)
        except Exception as e:
            error_msg = f"Error during initialization check: {e}"
            logging.error(error_msg, exc_info=True)
            status_dict['error'] = error_msg
            # self.initialized.emit(None) 
            self.error.emit(error_msg)

# --- End Worker --- 

# Helper function to get the base path
def get_base_path():
    """ Gets the path relative to the executable or script """
    if getattr(sys, 'frozen', False):
        # If the application is run as a bundle/frozen executable, the base path is the directory containing the executable
        base_path = Path(sys.executable).parent
    else:
        # If run from source, assume main_window.py is in anpe_gui and we need the parent (project root)
        base_path = Path(__file__).resolve().parent.parent
    return base_path

class MainWindow(QMainWindow):
    """Main window for the ANPE GUI application."""
    
    # Status bar styles (remains here for now, could be moved to theme.py later)
    STATUS_BAR_STYLE = f"""
        StatusBar {{
            background-color: #f5f5f5;
            border-top: 1px solid #e0e0e0;
            padding: 2px;
            min-height: 22px; /* Ensure minimum height */
        }}
        StatusBar QLabel {{ /* Base style for labels in custom StatusBar widget */
            padding: 3px 10px; /* Increased padding */
            border-radius: 4px;
            font-size: 11pt; /* Increased font size */
            min-height: 22px; /* Increased min height */
            alignment: 'AlignVCenter'; /* Vertically center text */
        }}
        /* Specific styles based on 'status' property */
        StatusBar QLabel[status="ready"] {{
            /* background-color: {SUCCESS_COLOR}20; Lighter green background */
            /* color: {SUCCESS_COLOR}; */
            background-color: #E7F3FF; /* Morandi Light Blue background */
            color: #005A9C; /* Primary Blue text */
            font-weight: bold;
        }}
        StatusBar QLabel[status="error"] {{
            /* background-color: {ERROR_COLOR}20; Lighter red background */
            /* color: {ERROR_COLOR}; */
            background-color: #FAE3E2; /* Morandi Light Red background */
            color: #C04A44; /* Morandi Red text */
            font-weight: bold;
        }}
        StatusBar QLabel[status="warning"] {{
            /* background-color: {WARNING_COLOR}20; Lighter amber background */
            /* color: {WARNING_COLOR}; */
            background-color: #FDF2E8; /* Morandi Light Ochre background */
            color: #E5B17A; /* Morandi Ochre text */
            font-weight: bold;
        }}
        StatusBar QLabel[status="info"] {{
            /* background-color: {INFO_COLOR}20; Lighter info background */
            /* color: {INFO_COLOR}; */
            background-color: #E7F3FF; /* Morandi Light Blue background */
            color: #005A9C; /* Primary Blue text */
        }}
        StatusBar QLabel[status="busy"] {{
            /* background-color: {PRIMARY_COLOR}20; Lighter busy background */
            /* color: {PRIMARY_COLOR}; */
            font-style: bold; /* Keep bold as per user edit */
            background-color: #f0f0f0; /* Light grey background */
            color: #333333; /* Dark grey text */
        }}
        StatusBar QLabel[status="success"] {{
            /* background-color: {SUCCESS_COLOR}20; */
            /* color: {SUCCESS_COLOR}; */
            /* background-color: #e6f9e6; Light Green background */
            /* color: #1e7e34; Deeper Green text */
            background-color: #F1F8E8; /* Morandi Light Green background */
            color: #73A942; /* Morandi Green text */
            font-weight: bold;
        }}
         StatusBar QLabel[status="failed"] {{
            /* background-color: {ERROR_COLOR}20; Lighter red background */
            /* color: {ERROR_COLOR}; */
            background-color: #FAE3E2; /* Morandi Light Red background */
            color: #C04A44; /* Morandi Red text */
            font-weight: bold;
        }}
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle(f"ANPE GUI v{GUI_VERSION}")
        # Set default size first (Enlarged)
        self.initial_width = 1300
        self.initial_height = 950
        self.setGeometry(0, 0, self.initial_width, self.initial_height)  
        
        # Center the window
        self._center_on_screen()
        
        # Set window icon
        from anpe_gui.resource_manager import ResourceManager
        self.setWindowIcon(ResourceManager.get_icon("app_icon.png"))
        
        # Apply theme stylesheet
        self.setStyleSheet(get_stylesheet())
        
        # self.thread_pool = QThreadPool() # Removed unused thread pool
        self.extractor_ready = False # Flag to indicate if core extractor is ready
        self.anpe_version = anpe_version_str # Store version string
        self.worker: Optional[ExtractionWorker] = None # For single processing
        self.batch_worker: Optional[BatchWorker] = None # For batch processing
        self.results: Optional[Dict[str, Any]] = None # To store last processing results for export

        # Animation setup
        self._fade_animation = None
        self.setWindowOpacity(0.0) # Start transparent for fade-in

        # Store initial model status details
        self.initial_model_status = {
            'spacy_models': [],
            'benepar_models': [],
            'nltk_present': False,
            'error': None # Store potential init error
        } 

        self.setup_ui()
        
        # Start Background Extractor Initialization only if ANPEExtractor is real
        if 'ANPEExtractor' in globals() and ANPEExtractor.__module__ != 'builtins': # Check it's not the dummy
             self.start_background_initialization()
        else:
             self.status_bar.showMessage("ANPE Library Error! Processing disabled.", 0) # Persistent message
             self.process_button.setEnabled(False) # Disable processing
             logging.error("ANPE library not found or dummy class used. Initialization skipped.")

    def _center_on_screen(self):
        """Centers the window on the primary screen."""
        try:
            screen = QApplication.primaryScreen()
            if not screen:
                logging.warning("Could not get primary screen. Centering skipped.")
                return
            screen_geometry = screen.availableGeometry() # Use available geometry to avoid taskbars etc.
            window_rect = self.frameGeometry()
            center_point = screen_geometry.center()
            window_rect.moveCenter(center_point)
            # Ensure the window doesn't go off-screen (e.g., if screen is smaller than window)
            final_pos = window_rect.topLeft()
            final_pos.setX(max(0, final_pos.x()))
            final_pos.setY(max(0, final_pos.y()))
            self.move(final_pos)
            logging.debug(f"Centered window on screen at {final_pos}")
        except Exception as e:
            # Log error but don't prevent startup
            logging.error(f"Error centering window: {e}", exc_info=True)

    # Property for animation
    @pyqtProperty(float)
    def windowOpacity(self):
        return super().windowOpacity()

    @windowOpacity.setter
    def windowOpacity(self, value):
        super().setWindowOpacity(value)

    def _fade(self, start_value, end_value, duration, on_finish=None):
        """Helper function to create and run fade animation."""
        if self._fade_animation:
            self._fade_animation.stop() # Stop existing animation

        self._fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self._fade_animation.setDuration(duration)
        self._fade_animation.setStartValue(float(start_value)) # Ensure float
        self._fade_animation.setEndValue(float(end_value))     # Ensure float
        self._fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        if on_finish:
            # Disconnect previous connections to avoid multiple calls
            try:
                self._fade_animation.finished.disconnect() 
            except TypeError: # No connection exists
                pass 
            self._fade_animation.finished.connect(on_finish)
            
        self._fade_animation.start()

    def fade_in(self, duration=300):
        """Fade the main window in."""
        self.setWindowOpacity(0.0) # Ensure starting point
        self.show() # Make window visible before animating opacity
        self._fade(0.0, 1.0, duration) # Add this line back

    def start_background_initialization(self):
        """Starts the background thread for ANPEExtractor initialization."""
        # Clear previous status before starting check
        self.initial_model_status = {
            'spacy_models': [], 'benepar_models': [], 'nltk_present': False, 'error': None
        }
        self.extractor_ready = False # Reset ready state
        if hasattr(self, 'process_button'): self.process_button.setEnabled(False)
        
        self.init_thread = QThread()
        self.init_worker = ExtractorInitializer()
        self.init_worker.moveToThread(self.init_thread)
        
        self.init_worker.initialized.connect(self.on_extractor_initialized)
        self.init_worker.error.connect(self.on_extractor_init_error)
        self.init_thread.started.connect(self.init_worker.run)
        
        # Clean up thread when worker finishes
        self.init_worker.initialized.connect(self.init_thread.quit)
        self.init_worker.error.connect(self.init_thread.quit)

        status_type = 'busy'
        message = "Checking ANPE models..."
        self.status_bar.showMessage(message, status_type=status_type)
        self.init_thread.start()
    
    @pyqtSlot(object) # Receives the status dictionary 
    def on_extractor_initialized(self, status_dict):
        """Slot called when the background initialization check completes successfully."""
        logging.debug(f"MAIN: Initialization check successful. Status received: {status_dict}")
        self.initial_model_status = status_dict # Store detailed status
        self.extractor_ready = True # Mark as ready
        status_type = 'ready'
        message = "ANPE Ready"
        self.status_bar.showMessage(message, 3000, status_type=status_type)
        if hasattr(self, 'process_button'):
             self.process_button.setEnabled(True)

    @pyqtSlot(str)
    def on_extractor_init_error(self, error_message):
        """Slot to handle initialization errors, including missing models."""
        logging.error(f"MAIN: Extractor init error signal received: {error_message}")
        self.extractor_ready = False 
        # Store default/error state in status dict
        self.initial_model_status = {
            'spacy_models': [],
            'benepar_models': [],
            'nltk_present': False,
            'error': error_message 
        }
        if hasattr(self, 'process_button'): 
            self.process_button.setEnabled(False)

        MISSING_MODELS_PREFIX = "Missing required models: "
        if error_message.startswith(MISSING_MODELS_PREFIX):
            status_type = 'warning'
            status_message = f"{error_message}. Use 'Manage Models' (gear icon) to install." 
            self.status_bar.showMessage(status_message, 0, status_type=status_type) 
            QMessageBox.warning(self, "Models Missing", 
                             f"{status_message}\n\nProcessing functionality will be disabled until models are installed.")
        else:
            status_type = 'error'
            status_message = f"ANPE Initialization Failed! {error_message}"
            self.status_bar.showMessage(status_message, 0, status_type=status_type) 
            QMessageBox.critical(self, "Initialization Error", 
                                 f"Failed to initialize ANPE: {error_message}")

    def setup_ui(self):
        """Set up the main UI components: Header, Splitter (Tabs | Log), Status Bar."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 12) # Increased bottom margin
        self.main_layout.setSpacing(10)
        
        # 1. Header
        self.setup_header()
        
        # 2. Main Content Splitter
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 2a. Tab Widget (Left Pane)
        self.main_tabs = QTabWidget()
        self.main_tabs.setDocumentMode(True) # Ensure tabs are not centered on macOS
        self.main_tabs.setStyleSheet("""
            QTabWidget::pane {
                border-top: 1px solid #C2C7CB;
            }
        """)
        self.input_tab = QWidget() # Create tab page widget
        self.setup_input_tab()     # Populate it
        self.main_tabs.addTab(self.input_tab, "Input")
        
        self.output_tab = QWidget() # Create tab page widget
        self.setup_output_tab()    # Populate it
        self.main_tabs.addTab(self.output_tab, "Output")
        self.main_splitter.addWidget(self.main_tabs) # Add tabs to splitter
        
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
        # Apply the specific status bar styles directly
        self.status_bar.setStyleSheet(self.STATUS_BAR_STYLE)
        self.main_layout.addWidget(self.status_bar) # Add status bar at the bottom

        # Set flag after all essential UI elements are created
        self.ui_setup_complete = True

    def setup_header(self):
        """Set up the application header with text title and version."""
        from anpe_gui.resource_manager import ResourceManager
        
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 2, 0, 2) 
        header_layout.setSpacing(5)  # Reduced spacing between header elements
        
        # Title Area (Horizontal Layout for title and subtitle together)
        title_area_widget = QWidget()
        title_layout = QHBoxLayout(title_area_widget)
        title_layout.setContentsMargins(0,0,0,0)
        title_layout.setSpacing(8) # Adjusted spacing

        # Icon Label (Added)
        icon_label = QLabel()
        pixmap = ResourceManager.get_pixmap("app_icon.png")
        pixmap = pixmap.scaled(60, 60, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation) # Scaled icon
        icon_label.setPixmap(pixmap)
        icon_label.setFixedSize(60, 60) # Set fixed size for alignment
        title_layout.addWidget(icon_label)
        
        # Main Title: ANPE
        title_label = QLabel()
        title_label.setText(f'<b style="color:{PRIMARY_COLOR}; font-size: 22pt;">ANPE</b>')
        title_layout.addWidget(title_label)
        
        # Subtitle (two lines with version and creator)
        subtitle_label = QLabel()
        subtitle_label.setText(f'''
            <div style="color: #666666; font-size: 9pt; line-height: 0.8; margin-top: 3px;">
                GUI v{GUI_VERSION} | Core v{anpe_version_str}<br>
                Created by @rcverse
            </div>
        ''')
        title_layout.addWidget(subtitle_label)
        
        # Add stretch to push everything to the left
        title_layout.addStretch()
        
        # Add title area to header layout (left-aligned)
        header_layout.addWidget(title_area_widget)
        
        # Add stretch to push setup button to the right
        header_layout.addStretch()
        
        # Create a container for the icons to keep them close together
        icon_container = QWidget()
        icon_layout = QHBoxLayout(icon_container)
        icon_layout.setContentsMargins(0, 0, 0, 0)
        icon_layout.setSpacing(15)  # Increased spacing between icons (was 2)
        
        # Help Button (using local SVG)
        help_button = QToolButton()
        help_button.setIcon(ResourceManager.get_icon("help.svg"))
        help_button.setToolTip("Show application help")
        help_button.setIconSize(QSize(20, 20))
        help_button.setStyleSheet("""
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 0px;
            }
            QToolButton:hover {
                background-color: #e0e0e0;
                border-radius: 3px;
            }
            QToolButton:pressed {
                background-color: #cccccc;
            }
        """)
        help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        help_button.setFixedSize(30, 30)
        self.help_button = help_button
        icon_layout.addWidget(self.help_button)
        self.help_button.clicked.connect(self.show_help)

        # Setup/Model Management Button (using local SVG)
        settings_button = QToolButton()
        settings_button.setIcon(ResourceManager.get_icon("setting.svg"))
        settings_button.setIconSize(QSize(20, 20))
        settings_button.setStyleSheet("""
            QToolButton {
                background-color: transparent;
                border: none;
                padding: 0px;
            }
            QToolButton:hover {
                background-color: #e0e0e0;
                border-radius: 3px;
            }
            QToolButton:pressed {
                background-color: #cccccc;
            }
        """)
        settings_button.setCursor(Qt.CursorShape.PointingHandCursor)
        settings_button.setFixedSize(30, 30)
        self.model_manage_button = settings_button
        self.model_manage_button.setToolTip("Open Model Management (Setup/Clean)")
        icon_layout.addWidget(self.model_manage_button)
        self.model_manage_button.clicked.connect(self.open_settings)
        
        # Add the icon container to the header layout
        header_layout.addWidget(icon_container)
        
        # Add header to main layout
        self.main_layout.addWidget(header_container)
        
        # Add aesthetic separator line between header and content
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet(f"background-color: {BORDER_COLOR}; max-height: 1px;")
        self.main_layout.addWidget(separator)

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
        paste_button.setProperty("secondary", True)
        clear_button = QPushButton("Clear Text")
        clear_button.setProperty("secondary", True)
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
        self.include_nested.setChecked(False)
        self.include_nested.setToolTip("Whether to include nested noun phrases (maintains parent-child relationships).")
        settings_layout.addWidget(self.include_nested, 0, 0)

        self.include_metadata = QCheckBox("Include Metadata")
        self.include_metadata.setChecked(False)
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

        # Min Length Widgets (placed directly in grid)
        self.min_length_check = QCheckBox("Min length")
        self.min_length_check.setToolTip("Filter out noun phrases shorter than this length (in tokens). Check to enable.")
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(1, 100)
        self.min_length_spin.setValue(2)
        self.min_length_spin.setEnabled(False)
        # self.min_length_spinbox.setMaximumWidth(70) # Let layout handle width
        self.min_length_check.toggled.connect(self.min_length_spin.setEnabled)
        filtering_layout.addWidget(self.min_length_check, 0, 0)
        filtering_layout.addWidget(self.min_length_spin, 0, 1)

        # Max Length Widgets (placed directly in grid)
        self.max_length_check = QCheckBox("Max length")
        self.max_length_check.setToolTip("Filter out noun phrases longer than this length (in tokens). Check to enable.")
        self.max_length_spin = QSpinBox()
        self.max_length_spin.setRange(1, 100)
        self.max_length_spin.setValue(10)
        self.max_length_spin.setEnabled(False)
        # self.max_length_spinbox.setMaximumWidth(70) # Let layout handle width
        self.max_length_check.toggled.connect(self.max_length_spin.setEnabled)
        filtering_layout.addWidget(self.max_length_check, 0, 2)
        filtering_layout.addWidget(self.max_length_spin, 0, 3)

        # Accept Pronouns Checkbox (moved to the next column)
        self.accept_pronouns = QCheckBox("Do not accept pronouns")
        self.accept_pronouns.setChecked(False)
        self.accept_pronouns.setToolTip("When checked, excludes single-word pronouns (e.g., 'it', 'they') from noun phrases.")
        filtering_layout.addWidget(self.accept_pronouns, 0, 4)

        # Adjust Column stretches for the new layout
        filtering_layout.setColumnStretch(0, 0) # Min Checkbox
        filtering_layout.setColumnStretch(1, 0) # Min Spinbox
        filtering_layout.setColumnStretch(2, 0) # Max Checkbox
        filtering_layout.setColumnStretch(3, 0) # Max Spinbox
        filtering_layout.setColumnStretch(4, 1) # Accept Pronouns (takes remaining space)
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
        
        # Default Button (left-aligned)
        self.default_button = QPushButton("Default")
        self.default_button.setStyleSheet(f"""
            QPushButton {{
                padding: 8px 15px;
                font-size: 11pt; /* Increased font size */
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
        self.default_button.setToolTip("Restore filtering options to default state")
        self.default_button.clicked.connect(self.restore_default_filters)
        process_reset_layout.addWidget(self.default_button)
        
        # Add stretch to push other buttons to the right
        process_reset_layout.addStretch(1)

        # Reset Button
        self.reset_button = QPushButton("Reset")
        self.reset_button.setStyleSheet(f"""
            QPushButton {{
                padding: 8px 15px;
                font-size: 11pt; /* Increased font size */
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
                font-size: 11pt; /* Increased font size */
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
        # Add tooltip to explain disabled state
        self.process_button.setToolTip("Process the input text or files. \n(Disabled if initializing, processing, or models are missing - check Status Bar)")
        process_reset_layout.addWidget(self.process_button)
        
        self.input_layout.addLayout(process_reset_layout)

    def setup_output_tab(self):
        """Set up the Output tab: Results display, Export options, Navigation."""
        self.output_tab = QWidget()
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
        self.file_selector_layout.addWidget(self.file_selector_combo, 1)  # Give combo stretch
        self.output_layout.addLayout(self.file_selector_layout)
        self.file_selector_label.hide()  # Hidden initially
        self.file_selector_combo.hide()  # Hidden initially

        # --- Results Display Area ---
        results_group = QGroupBox("Extraction Results")
        results_group_layout = QVBoxLayout(results_group)
        self.results_display_widget = ResultDisplayWidget() # Use the new custom widget
        results_group_layout.addWidget(self.results_display_widget) # Add the new widget
        self.output_layout.addWidget(results_group, 1)  # Allow results area to stretch

        # --- Export Options --- (Using QFormLayout for better alignment)
        export_group = QGroupBox("Export Options")
        # export_layout = QGridLayout(export_group) # Old layout
        export_layout = QFormLayout(export_group) # New layout (QFormLayout)
        export_layout.setSpacing(10)
        export_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow) # Ensure fields expand
        export_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft) # Left-align the labels

        # Row 0: Format (ComboBox + Help Button)
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["txt", "csv", "json"])
        self.export_format_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        # Use the combo box directly
        export_layout.addRow("Format:", self.export_format_combo)

        # Row 1: Filename Prefix (Optional)
        self.export_filename_prefix_edit = QLineEdit()
        self.export_filename_prefix_edit.setPlaceholderText("Optional prefix for exported files")
        # Rename the label instance variable as well for clarity if needed elsewhere, although it's only used here
        self.export_filename_prefix_label = QLabel("Filename Prefix:") 
        export_layout.addRow(self.export_filename_prefix_label, self.export_filename_prefix_edit)

        # Row 2: Directory (LineEdit + Browse Button)
        self.export_dir_edit = QLineEdit()
        self.export_dir_edit.setPlaceholderText("Select export directory...")
        self.export_dir_edit.setReadOnly(True)
        self.browse_export_dir_button = QPushButton("Browse...")
        self.browse_export_dir_button.clicked.connect(self.browse_export_directory)
        
        export_dir_widget_layout = QHBoxLayout()
        export_dir_widget_layout.setContentsMargins(0,0,0,0)
        export_dir_widget_layout.setSpacing(5)
        export_dir_widget_layout.addWidget(self.export_dir_edit, 1) # LineEdit takes stretch
        export_dir_widget_layout.addWidget(self.browse_export_dir_button)
        export_dir_widget_container = QWidget()
        export_dir_widget_container.setLayout(export_dir_widget_layout)
        export_layout.addRow("Directory:", export_dir_widget_container)

        # Row 3: Export Button and Help Button (spans columns)
        bottom_button_layout = QHBoxLayout()
        bottom_button_layout.setContentsMargins(0,0,0,0) # No margins for this layout
        
        # Help button setup (moved from Format row)
        self.export_format_help_button = QPushButton("?") # Changed text to "?"
        self.export_format_help_button.setToolTip("Click for information about export formats and filename structure") # Tooltip remains helpful
        self.export_format_help_button.setCursor(Qt.CursorShape.PointingHandCursor)
        # Connect to the new handler function
        self.export_format_help_button.clicked.connect(self.show_help_at_export_section) 
        
        # Style the button to be round with a "?"
        help_button_size = 26 # Define button size (width and height)
        # Force the size using min/max and fixed policy
        self.export_format_help_button.setMinimumSize(help_button_size, help_button_size)
        self.export_format_help_button.setMaximumSize(help_button_size, help_button_size)
        self.export_format_help_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # self.export_format_help_button.setFixedSize(help_button_size, help_button_size) # Keep this too for belt-and-braces?
        
        self.export_format_help_button.setStyleSheet(f'''
            QPushButton {{
                border: 1px solid #bbb; /* Slightly darker border */
                border-radius: {help_button_size // 2}px; /* Half of size for roundness */
                background-color: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #fefefe, stop:1 #f0f0f0); /* Lighter gradient */
                color: #444; /* Darker grey for '?' */
                font-weight: bold;
                font-size: 10pt; /* Reduced font size */
                padding: 0px; /* Explicitly zero padding */
            }}
            QPushButton:hover {{
                background-color: #e8e8e8; /* Slightly darker hover */
            }}
            QPushButton:pressed {{
                background-color: #d8d8d8; /* Pressed color */
                border-color: #999;
            }}
        ''')

        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self.export_results) 
        self.export_button.setEnabled(False)
        # Optionally make export button slightly bigger/more prominent
        # self.export_button.setMinimumHeight(30)

        bottom_button_layout.addWidget(self.export_format_help_button) # Add help button (left aligned by default)
        bottom_button_layout.addStretch(1) # Add stretch between buttons
        bottom_button_layout.addWidget(self.export_button) # Add export button (right aligned due to stretch)
        
        export_layout.addRow(bottom_button_layout) # Add the combined layout spanning columns

        # Ensure the export group expands horizontally
        export_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.output_layout.addWidget(export_group)

        # --- Navigation Buttons ---
        nav_button_layout = QHBoxLayout()
        # Back button removed for simplicity, users can just click the Input tab
        # self.back_to_input_button = QPushButton("Back to Input") ...
        
        # Process New Button (Connects to new handler)
        self.process_new_button = QPushButton("Process New Input")
        # self.process_new_button.clicked.connect(self.reset_workflow) # Old connection
        self.process_new_button.clicked.connect(self.handle_process_new_input_click) # New connection

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
            if self.min_length_check.isChecked():
                config["min_length"] = self.min_length_spin.value()
            if self.max_length_check.isChecked():
                config["max_length"] = self.max_length_spin.value()
            config["accept_pronouns"] = not self.accept_pronouns.isChecked()
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

            # --- Read Model Usage Preferences from QSettings ---
            settings = QSettings("rcverse", "ANPE_GUI") # Use the same org/app name as in ModelsPage
            spacy_pref = settings.value("modelUsage/spacyModel", "(Auto-detect)")
            benepar_pref = settings.value("modelUsage/beneparModel", "(Auto-detect)")
            
            # Add preferences to config, handle "(Auto-detect)" by passing None (or the string itself if core handles it)
            # Assuming core ANPEExtractor interprets None as auto-detect for model selection
            config["spacy_model_preference"] = None if spacy_pref == "(Auto-detect)" else spacy_pref
            config["benepar_model_preference"] = None if benepar_pref == "(Auto-detect)" else benepar_pref
            logging.debug(f"Read model usage preferences: spaCy='{spacy_pref}', Benepar='{benepar_pref}'")
            # -----------------------------------------------------

            # Config gathering is important, keep INFO level for success
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
        """Start the ANPE processing based on current UI settings and input."""
        # --- Check and clear previous results first ---
        if not self._confirm_and_clear_results():
            return # User cancelled clearing previous results

        # --- Gather configuration ---
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
            log_message = f"Starting batch processing for {len(files)} files..."
            self.log(log_message)
            status_type = 'busy'
            status_message = f"Processing {len(files)} files..."
            # Pass message directly to run_batch_processing which will call status_bar.update_progress
            # Get preferences from the config dict (apply_configuration already handled None for Auto-detect)
            spacy_pref = config.pop('spacy_model_preference', None)
            benepar_pref = config.pop('benepar_model_preference', None)
            
            self.run_batch_processing(files, config, spacy_pref, benepar_pref, initial_status_message=status_message)

        elif input_mode_index == 1: # Text Input Mode
            text_content = self.direct_text_input.toPlainText().strip()
            if not text_content:
                QMessageBox.warning(self, "No Input", "Please enter text to process.")
                return
            # 3. Run Single Processing
            self.log_panel.clear_log() # Clear log for new run
            self.log("Starting single text processing...")
            status_type = 'busy'
            status_message = "Processing text..."
            # Pass message directly to run_single_processing which will call status_bar.start_progress
            # Get preferences from the config dict
            spacy_pref = config.pop('spacy_model_preference', None)
            benepar_pref = config.pop('benepar_model_preference', None)
            self.run_single_processing(text_content, config, spacy_pref, benepar_pref, initial_status_message=status_message)

    def run_single_processing(self, text_content: str, config: Dict[str, Any], 
                              spacy_pref: Optional[str], benepar_pref: Optional[str], # Added prefs
                              initial_status_message: str):
        """Starts the background worker for processing a single text input."""
        if hasattr(self, 'single_thread') and self.single_thread is not None and self.single_thread.isRunning():
             QMessageBox.warning(self, "Processing Busy", 
                                 "Another process is already running. Please wait.")
             return
            
        self.results = None 
        self.results_display_widget.clear_display() 
        self.export_button.setEnabled(False) 
        self.process_button.setEnabled(False) 
        # Use the activity indicator for indeterminate processing
        self.status_bar.update_progress(0, initial_status_message)
        self.processing_error_occurred = False # Initialize error flag

        # 1. Create Worker
        self.worker = ExtractionWorker(
            text_content=text_content, 
            config=config, 
            anpe_version=self.anpe_version,
            include_nested=self.include_nested.isChecked(),
            include_metadata=self.include_metadata.isChecked(),
            spacy_model_preference=spacy_pref, # Pass preference
            benepar_model_preference=benepar_pref # Pass preference
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
        logging.debug("MAIN: Starting single processing thread.")
        self.single_thread.start()

    def run_batch_processing(self, file_paths: List[str], config: Dict[str, Any], 
                             spacy_pref: Optional[str], benepar_pref: Optional[str], # Added prefs
                             initial_status_message: str):
        """Starts the background worker for processing multiple files."""
        if hasattr(self, 'batch_thread') and self.batch_thread is not None and self.batch_thread.isRunning():
             QMessageBox.warning(self, "Processing Busy", 
                                 "Another process is already running. Please wait.")
             return

        self.results = {} 
        self.results_display_widget.clear_display() 
        self.file_selector_combo.clear() 
        self.file_selector_label.hide()
        self.file_selector_combo.hide()
        self.export_button.setEnabled(False) 
        self.process_button.setEnabled(False) 
        # Use the pre-formatted message from start_processing for the initial update
        self.status_bar.update_progress(0, initial_status_message )
        self.processing_error_occurred = False # Initialize error flag

        # 1. Create Worker
        self.batch_worker = BatchWorker(
            file_paths=file_paths, 
            config=config, 
            anpe_version=self.anpe_version,
            include_nested=self.include_nested.isChecked(),
            include_metadata=self.include_metadata.isChecked(),
            spacy_model_preference=spacy_pref, # Pass preference
            benepar_model_preference=benepar_pref # Pass preference
        )
        
        # 2. Create Thread and Move Worker
        self.batch_thread = QThread() 
        self.batch_worker.moveToThread(self.batch_thread)

        # 3. Connect Signals
        self.batch_thread.started.connect(self.batch_worker.run)
        # --- Worker Signals ---
        self.batch_worker.signals.status_update.connect(self.update_status_message)
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
        logging.debug("MAIN: Starting batch processing thread.")
        self.batch_thread.start() 

    # --- Worker Signal Handlers ---

    @pyqtSlot(str) # NEW Slot for status updates
    def update_status_message(self, message: str):
        """Update the status bar's text message."""
        # Only update the text, keep existing progress bar state
        self.status_bar.showMessage(message, status_type='busy')

    @pyqtSlot(int, str) # Receives percentage and message
    def update_batch_progress(self, percentage: int, message: str):
        """Update status bar progress bar percentage."""
        # This message provides context for the percentage update but might be brief
        # The main status text is handled by update_status_message
        # We pass None for message to update_progress to avoid overwriting
        self.status_bar.update_progress(percentage, None) 

    @pyqtSlot(str) # Receives message
    def update_progress(self, message: str):
         """Update status bar for indeterminate progress (single text)."""
         # For indeterminate, we just update the message
         self.status_bar.showMessage(message, status_type='busy')

    @pyqtSlot(dict) # Receives result dictionary for one run
    def handle_single_result(self, result_data: Dict[str, Any]):
        """Handle the result from the ExtractionWorker (single text)."""
        self.results = result_data # Store the complete result
        # Pass the current state of the metadata checkbox
        metadata_is_on = self.include_metadata.isChecked()
        self.results_display_widget.display_results(result_data, metadata_enabled=metadata_is_on)
        self.export_button.setEnabled(True) # Enable export
        # Keep INFO for completion message
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
            # Pass the current state of the metadata checkbox
            metadata_is_on = self.include_metadata.isChecked()
            self.results_display_widget.display_results(result_data, metadata_enabled=metadata_is_on)
            self.file_selector_label.show()
            self.file_selector_combo.show()
             
        # Keep INFO for individual file completion
        self.log(f"Processed file: {base_name}")
        # Status bar updated via update_batch_progress signal

    @pyqtSlot(str) # Receives error message string
    def handle_error(self, error_message: str):
        """Handle errors reported by workers."""
        logging.error(f"Processing Error: {error_message}") 
        status_type = 'error'
        # Format message with icon before passing to stop_progress
        formatted_message = f"Error: {error_message}" # Removed icon
        self.status_bar.stop_progress(formatted_message, status_type=status_type)
        QMessageBox.warning(self, "Processing Error", f"An error occurred: {error_message}")
        self.processing_error_occurred = True # Set error flag
        # Re-enable process button even on error? (Keep current behavior)
        # self.process_button.setEnabled(True) 

    @pyqtSlot(str)
    def processing_finished(self, worker_type: str):
        """Handle UI updates when a worker finishes, identified by worker_type."""
        # Keep INFO for this important event
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
            
            # Determine final message and status type
            status_type = 'success' # Default to success
            final_message_text = "Processing complete"
            if self.processing_error_occurred: 
                 status_type = 'error'
                 final_message_text = "Processing finished with errors"
            elif self.results is None or (isinstance(self.results, dict) and not self.results):
                 status_type = 'info' # Or 'warning'? 'info' seems okay.
                 final_message_text = "Processing finished (No results)"
                
            # Format message with icon
            final_message = final_message_text # Removed icon
            
            self.status_bar.stop_progress(final_message, status_type=status_type)
            
            # Enable export only if results exist AND no error occurred
            can_export = self.results is not None and not self.processing_error_occurred
            self.export_button.setEnabled(can_export)
            
            # Switch to output tab only on successful completion with results
            if can_export: # Use can_export flag which implies success and results exist
                 self.switch_to_output_tab()
            
            # Re-enable the process button if the extractor is ready
            if hasattr(self, 'process_button'):
                self.process_button.setEnabled(self.extractor_ready)
            
            # --- Auto-clear input fields after processing ---
            self.file_list_widget.clear_files()
            self.direct_text_input.clear()
            logging.debug("Input fields cleared automatically.")
            
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
            # Pass the current state of the metadata checkbox
            metadata_is_on = self.include_metadata.isChecked()
            self.results_display_widget.display_results(self.results[selected_file_path], metadata_enabled=metadata_is_on)
            logging.debug(f"Displayed results for selected file: {selected_file_path}")
        else:
            # This might happen if combo index changes before results are fully populated
            logging.warning(f"Could not find results for selected file: {selected_file_path}")

    def export_results(self):
        """Export the currently stored results to a file using unified naming."""
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
        # Get the optional prefix
        filename_prefix = self.export_filename_prefix_edit.text().strip() 
        
        # Generate timestamp string (YYYYMMDD_HHMMSS)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Keep INFO for start of export
        logging.info(f"Attempting export. Format: {export_format}, Dir: {export_dir}, Prefix: '{filename_prefix}'")

        try:
            from anpe.utils.export import ANPEExporter # Import here to avoid top-level dependency if export not used
            from pathlib import Path # Ensure Path is imported

            exporter = ANPEExporter()
            
            export_successful = False
            message = "" # Initialize message

            # Check if results are from batch processing (dict of file_path: result_data)
            # or single processing (single result_data dict)
            if isinstance(self.results, dict) and all(isinstance(k, str) and isinstance(v, dict) for k, v in self.results.items()):
                # Batch results: keys are file paths
                num_files = len(self.results)
                logging.info(f"Exporting batch results for {num_files} files.")
                
                exported_filenames = [] # Keep track of generated names for the message

                # Use unified naming based on input files + prefix + timestamp
                for file_path, result_data in self.results.items():
                    base_name = Path(file_path).stem 
                    # Construct filename: [prefix_]<stem>_anpe_results_<timestamp>.<format>
                    if filename_prefix:
                        output_filename = f"{filename_prefix}_{base_name}_anpe_results_{timestamp_str}.{export_format}"
                    else:
                        output_filename = f"{base_name}_anpe_results_{timestamp_str}.{export_format}"
                        
                    full_export_path = os.path.join(export_dir, output_filename)
                    logging.debug(f"Exporting '{file_path}' results to '{full_export_path}'")
                    # TODO: Add check for file overwrite here? (See suggestions)
                    exporter.export(result_data, format=export_format, output_filepath=full_export_path)
                    exported_filenames.append(output_filename)

                export_successful = True
                # Consider showing first few filenames if list is long?
                message = f"Results for {num_files} files exported successfully to {export_dir}" 
                
            elif isinstance(self.results, dict) and 'results' in self.results:
                # Single text result (has 'results' key)
                logging.info("Exporting single text results.")
                
                # Construct filename: [prefix_]anpe_text_results_<timestamp>.<format>
                if filename_prefix:
                     output_filename = f"{filename_prefix}_anpe_text_results_{timestamp_str}.{export_format}"
                else:
                     output_filename = f"anpe_text_results_{timestamp_str}.{export_format}"

                full_export_path = os.path.join(export_dir, output_filename)
                
                logging.debug(f"Exporting single text result to '{full_export_path}'")
                # TODO: Add check for file overwrite here? (See suggestions)
                exporter.export(self.results, format=export_format, output_filepath=full_export_path)
                
                export_successful = True
                message = f"Results exported successfully to {full_export_path}"
            else:
                # Unknown results format
                 logging.error(f"Cannot export. Unknown results format: {type(self.results)}")
                 QMessageBox.warning(self, "Export Error", "Cannot export results. Unknown data format.")
                 return

            if export_successful:
                logging.info(message) # Log the final message
                # Create a custom QMessageBox instead of static method
                msg_box = QMessageBox(self)
                msg_box.setIcon(QMessageBox.Icon.Information)
                msg_box.setWindowTitle("Export Successful")
                msg_box.setText(message)
                
                # Add standard OK button
                ok_button = msg_box.addButton(QMessageBox.StandardButton.Ok)
                
                # Add custom 'Open Directory' button
                open_dir_button = msg_box.addButton("Open Export Directory", QMessageBox.ButtonRole.ActionRole)
                
                # Execute the dialog
                msg_box.exec()
                
                # Check if the custom button was clicked
                if msg_box.clickedButton() == open_dir_button:
                    self.open_directory(export_dir)
            
        except ImportError:
             logging.error("Export failed: ANPEExporter not found. Is 'anpe' installed correctly?")
             QMessageBox.critical(self, "Export Error", "Export functionality requires the full 'anpe' library to be installed.")
        except Exception as e:
            logging.error(f"Error exporting results: {e}", exc_info=True)
            QMessageBox.warning(self, "Export Error", f"Error exporting results: {e}")

    def reset_workflow(self):
        """Reset the UI to start a new processing task. Clears inputs and results."""
        # Check if there are any results to warn about
        if self.results is not None:
            reply = QMessageBox.question(
                self,
                "Reset Workflow",
                "Proceeding will clear all current results. Are you sure you want to continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return

        logging.info("Resetting workflow...")
        # Clear input areas
        self.file_list_widget.clear_files()
        self.direct_text_input.clear()
        
        # Clear results area and stored results using the new widget
        self.results_display_widget.clear_display() 
        self.results = None
        
        # Hide file selector, disable export
        self.file_selector_label.hide()
        self.file_selector_combo.hide()
        self.file_selector_combo.clear()
        self.export_button.setEnabled(False)
        
        # Reset tabs and input mode to default (File Input on Input tab)
        self.main_tabs.setCurrentIndex(0) 
        self.file_input_button.setChecked(True) # Also triggers switch_input_mode via button group
        self.input_stack.setCurrentIndex(0)
        
        # Reset status bar
        status_type = 'ready'
        message = "ANPE Ready"
        self.status_bar.showMessage(message, status_type=status_type)
        self.status_bar.clear_progress()
        
        # Re-enable process button if extractor is ready
        if hasattr(self, 'process_button'):
            self.process_button.setEnabled(self.extractor_ready)
        
        logging.info("Workflow reset complete.")

    def restore_default_filters(self):
        """Restore all options to their default state."""
        logging.info("Restoring all options to default state...")
        
        # Reset General Settings
        self.include_nested.setChecked(False)
        self.include_metadata.setChecked(False)
        self.newline_breaks.setChecked(False)
        
        # Reset General Filtering Options
        self.min_length_check.setChecked(False)
        self.min_length_spin.setValue(2)
        self.max_length_check.setChecked(False)
        self.max_length_spin.setValue(10)
        self.accept_pronouns.setChecked(False)
        
        # Reset structure filtering - just reset the widget if it exists
        if hasattr(self, 'structure_filter_widget'):
            # Reset the widget by recreating it
            old_widget = self.structure_filter_widget
            self.structure_filter_widget = StructureFilterWidget()
            # Replace the old widget with the new one
            old_widget.parent().layout().replaceWidget(old_widget, self.structure_filter_widget)
            old_widget.deleteLater()
        
        logging.info("All options restored to default state.")

    # --- Utility and Helper Methods ---

    def create_log_panel(self) -> EnhancedLogPanel:
        """Create the log panel and configure the QtLogHandler."""
        log_widget = EnhancedLogPanel()
        
        # Store handler instance for removal on close
        self.qt_log_handler_instance = QtLogHandler()
        self.qt_log_handler_instance.log_signal.connect(log_widget.add_log_entry)

        # Define a simpler log format without duplicate level names
        log_format = '%(message)s'
        formatter = logging.Formatter(log_format)
        self.qt_log_handler_instance.setFormatter(formatter)

        # Configure Python's root logger
        logger = logging.getLogger()
        logger.addHandler(self.qt_log_handler_instance)
        logger.setLevel(logging.DEBUG) # Set desired logging level
        
        # Remove any existing handlers to prevent duplicate messages
        for handler in logger.handlers[:]:
            if not isinstance(handler, QtLogHandler):
                logger.removeHandler(handler)
        
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
         self.main_tabs.setCurrentIndex(1)

    def open_settings(self):
        """Opens the settings dialog, passing the initial model status."""
        logging.debug(f"Opening Settings Dialog. Passing initial status: {self.initial_model_status}") # LOGGING
        # Pass the stored initial status to the dialog
        dialog = SettingsDialog(self, initial_model_status=self.initial_model_status) 
        dialog.models_changed.connect(self.on_models_changed)
        dialog.model_usage_changed.connect(self.on_model_usage_preference_changed)
        dialog.exec()

    @pyqtSlot()
    def on_models_changed(self):
        """Slot called when models might have changed via the management dialog."""
        logging.info("MAIN: Models may have changed via dialog. Re-evaluating state...")
        self.status_bar.showMessage("Settings updated. Re-evaluating readiness...", 3000, status_type='info')

        # If the extractor wasn't ready before (e.g., models were missing),
        # try re-initializing now. The initialization process itself will
        # check model availability and update extractor_ready and button state.
        if not self.extractor_ready:
            logging.info("MAIN: Extractor was not ready. Triggering background initialization check...")
            # Use a timer to avoid potential race conditions with dialog closing
            QTimer.singleShot(200, self.start_background_initialization)
        else:
            # Extractor was already ready. Assume it still is unless initialization fails.
            # The process button enablement is tied to self.extractor_ready,
            # which is managed by the initialization process.
            logging.info("MAIN: Extractor was already ready. Status remains ready unless initialization fails upon next processing attempt.")
            # Ensure process button reflects current state (should be enabled if extractor_ready is True)
            if hasattr(self, 'process_button'):
                self.process_button.setEnabled(self.extractor_ready)
            # Optionally update status bar message if needed, but init handles most states.
            # self.status_bar.showMessage("Model status re-checked. ANPE Ready.", 3000, status_type='ready')

    @pyqtSlot()
    def on_model_usage_preference_changed(self):
        """Slot called when the model usage preference is changed in the SettingsDialog."""
        # For now, just log this event. The actual preference is read during apply_configuration.
        logging.info("MAIN: Model usage preference changed signal received.")

    def closeEvent(self, event):
        """Handle the main window closing."""
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

    # --- Help Function --- 
    def show_help(self, anchor: Optional[str] = None): # Modified to accept optional anchor
        """Creates and shows the custom HelpDialog, optionally scrolling to an anchor."""
        base_dir = get_base_path()
        help_file_path = base_dir / "docs" / "gui_help.md" # Use base_dir and correct filename

        # Check if file exists before creating dialog
        if not help_file_path.is_file():
            error_msg = f"Help file not found at: {help_file_path}"
            logging.error(error_msg)
            QMessageBox.warning(self, "Help Not Found", f"Could not find the help file.\\nExpected location: {help_file_path}")
            return

        try:
            # Create and execute the custom dialog, passing both versions
            dialog = HelpDialog(help_file_path, GUI_VERSION, self.anpe_version, self) # Pass the correct path object

            # If an anchor is provided, try to navigate after the dialog is shown
            if anchor:
                 # Use a single shot timer to allow the dialog to fully render first
                 QTimer.singleShot(100, lambda: self.navigate_help_to_anchor(dialog, anchor))

            dialog.exec()
        except Exception as e:
            # Catch potential errors during dialog creation/execution
            logging.error(f"Error showing help dialog: {e}", exc_info=True)
            QMessageBox.critical(self, "Help Error", f"An unexpected error occurred while trying to show help: {e}")

    def navigate_help_to_anchor(self, help_dialog, anchor_id: str):
        """Tries to scroll the help dialog's text browser to a specific anchor."""
        if not help_dialog: return
        
        # Attempt 1: Scroll using the tree navigation if possible (matches heading text)
        # Assuming anchor_id corresponds to a heading text like "Exporting Results"
        items = help_dialog.nav_tree.findItems(anchor_id, Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchRecursive)
        if items:
            item = items[0]
            help_dialog.nav_tree.setCurrentItem(item) # Select the item
            # Trigger the navigation click handler programmatically
            help_dialog.on_nav_item_clicked(item, 0) 
            logging.debug(f"Navigated help dialog to section: {anchor_id} using tree.")
            return

        # Attempt 2: Fallback to scrolling the QTextBrowser directly
        # Convert anchor_id to a suitable HTML anchor if needed (e.g., lowercase, replace spaces)
        # This assumes the HTML generation creates predictable anchors
        html_anchor = anchor_id.lower().replace(' ', '-') 
        logging.debug(f"Attempting to scroll help dialog QTextBrowser directly to anchor: #{html_anchor}")
        if hasattr(help_dialog, 'text_browser'):
            # Use scrollToAnchor method
            help_dialog.text_browser.scrollToAnchor(html_anchor)
            # For redundancy, try moving cursor too if positions are stored (optional)
            if hasattr(help_dialog, 'heading_positions') and anchor_id in help_dialog.heading_positions:
                position = help_dialog.heading_positions[anchor_id]
                cursor = help_dialog.text_browser.textCursor()
                cursor.setPosition(position)
                help_dialog.text_browser.setTextCursor(cursor)
            QCoreApplication.processEvents() # Force update
            logging.debug(f"Scrolled help dialog QTextBrowser to anchor: #{html_anchor}")
        else:
            logging.warning("Could not find text_browser in HelpDialog to scroll.")

    @pyqtSlot()
    def show_help_at_export_section(self):
        """Shows the main help dialog scrolled to the 'Exporting Results' section."""
        # The anchor should match the heading text in Help.md
        self.show_help(anchor="Exporting Results") 

    def _confirm_and_clear_results(self) -> bool:
        """Checks for existing results, prompts user to confirm clearing them,
        and clears them if confirmed. Returns True if processing should proceed, False otherwise."""
        if self.results is not None:
            # Create QMessageBox instance manually for styling
            msg_box = QMessageBox(self) # Parent to MainWindow
            msg_box.setWindowTitle("Clear Previous Results?")
            msg_box.setTextFormat(Qt.TextFormat.RichText) # Allow rich text if needed, though not strictly necessary here
            msg_box.setText(
                "Processing new input will clear your previous results. "
                "Make sure you have exported them if you wish to keep them.<br><br>"
                "<b>Do you want to proceed?</b>"
            )
            msg_box.setIcon(QMessageBox.Icon.Question)
            msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msg_box.setDefaultButton(QMessageBox.StandardButton.No) # Default focus on No

            # Find the 'Yes' button
            yes_button = msg_box.button(QMessageBox.StandardButton.Yes)
            if yes_button:
                # Apply custom style using the theme's ERROR_COLOR
                yes_button.setStyleSheet(f"""
                    QPushButton {{ 
                        background-color: {ERROR_COLOR}; 
                        color: white; 
                        border: none; 
                        padding: 5px 15px; 
                        border-radius: 4px; 
                        min-width: 60px; /* Ensure minimum width */
                    }}
                    QPushButton:hover {{ background-color: #d9534f; }}
                    QPushButton:pressed {{ background-color: #c9302c; }}
                """)
            
            # Execute the dialog
            reply = msg_box.exec()

            if reply == QMessageBox.StandardButton.No:
                logging.info("User cancelled processing to keep previous results.")
                return False # User cancelled

        # Proceed with clearing results (or if no results existed)
        logging.info("Clearing previous results before new processing run.")
        self.results_display_widget.clear_display()
        self.results = None
        self.file_selector_label.hide()
        self.file_selector_combo.hide()
        self.file_selector_combo.clear()
        self.export_button.setEnabled(False)
        # Optional: Reset status bar here if desired
        # self.status_bar.showMessage("Ready", status_type='ready')

        return True # OK to proceed

    @pyqtSlot()
    def handle_process_new_input_click(self):
        """Handles the 'Process New Input' button click."""
        logging.debug("Process New Input button clicked.")
        if self._confirm_and_clear_results():
            # If results were cleared (or none existed), switch to input tab
            try:
                # Find index of Input tab (more robust than assuming 0)
                input_tab_index = -1
                for i in range(self.main_tabs.count()):
                    if self.main_tabs.widget(i) == self.input_tab: # Check if it's the correct widget instance
                        input_tab_index = i
                        break
                if input_tab_index != -1:
                    self.main_tabs.setCurrentIndex(input_tab_index)
                    logging.debug(f"Switched to Input tab (index {input_tab_index}).")
                else:
                    logging.warning("Could not find Input tab to switch to.")
            except Exception as e:
                logging.error(f"Error switching to Input tab: {e}")

    def open_directory(self, path):
        """Opens the specified directory in the default file explorer."""
        try:
            if platform.system() == "Windows":
                # Use os.startfile on Windows
                os.startfile(path)
            elif platform.system() == "Darwin": # macOS
                # Use subprocess.Popen with 'open' on macOS
                subprocess.Popen(["open", path])
            else: # Linux and other Unix-like systems
                # Use subprocess.Popen with 'xdg-open' on Linux
                subprocess.Popen(["xdg-open", path])
            logging.info(f"Opened directory: {path}")
        except FileNotFoundError:
            logging.error(f"Cannot open directory: Path not found '{path}'")
            QMessageBox.warning(self, "Open Directory Error", f"The directory could not be found:\n{path}")
        except Exception as e:
            logging.error(f"Error opening directory '{path}': {e}", exc_info=True)
            QMessageBox.warning(self, "Open Directory Error", f"Could not open the directory:\n{e}")
