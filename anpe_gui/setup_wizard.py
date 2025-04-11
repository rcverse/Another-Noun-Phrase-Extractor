import sys
from PyQt6.QtWidgets import (QApplication, QWizard, QWizardPage, QVBoxLayout, QLabel,
                             QProgressBar, QPushButton, QMessageBox, QGroupBox, QDialog,
                             QSizePolicy, QGridLayout, QFrame, QWidget, QHBoxLayout, 
                             QTextEdit, QSpacerItem)
from PyQt6.QtCore import pyqtSignal, QThread, QObject, Qt, QTimer, QCoreApplication
from PyQt6.QtGui import QColor, QPalette, QFont, QIcon
from anpe.utils.setup_models import (setup_models, check_spacy_model,
                                   check_benepar_model, check_nltk_models,
                                   install_spacy_model, install_benepar_model, install_nltk_models)
import logging
# Import our custom activity indicator
from .widgets.activity_indicator import PulsingActivityIndicator

# --- Constants ---
PRIMARY_COLOR = "#2c3e50"     # Deep blue for headings and primary elements
ACCENT_COLOR = "#3498db"      # Light blue for highlights and accents  
SUCCESS_COLOR = "#27ae60"     # Green for success indicators
ERROR_COLOR = "#e74c3c"       # Red for error indicators
WARNING_COLOR = "#f39c12"     # Orange/Yellow for warnings/in-progress
NEUTRAL_COLOR = "#7f8c8d"     # Gray for neutral/pending status
BG_COLOR = "#ffffff"          # White background
BG_ALT_COLOR = "#f8f9fa"      # Light gray for alternate backgrounds
BORDER_COLOR = "#e0e0e0"      # Light gray for borders

# --- Worker Thread for Downloads ---
class SetupWorker(QObject):
    finished = pyqtSignal(bool)
    progress_update = pyqtSignal(str)
    model_status_update = pyqtSignal(str, str)
    setup_log_completed = pyqtSignal(str)  # Signal to pass the setup log

    def __init__(self):
        super().__init__()
        self.setup_log = []  # Track all messages for logging

    def _append_log(self, message):
        """Add a message to the log."""
        self.setup_log.append(message)
        
    def run(self):
        """Runs the model setup process with detailed status updates."""
        all_successful = True
        try:
            message = "Starting model check..."
            self.progress_update.emit(message)
            self._append_log(message)

            # --- Check Models ---
            models_to_install = {}

            message = "Checking spaCy model..."
            self._append_log(message)
            self.model_status_update.emit("spaCy", "Checking...")
            spacy_ok = check_spacy_model()
            status = "✓ Present" if spacy_ok else "✗ Missing"
            self.model_status_update.emit("spaCy", status)
            self._append_log(f"spaCy model status: {status}")
            if not spacy_ok: models_to_install["spaCy"] = install_spacy_model

            message = "Checking Benepar model..."
            self._append_log(message)
            self.model_status_update.emit("Benepar", "Checking...")
            benepar_ok = check_benepar_model()
            status = "✓ Present" if benepar_ok else "✗ Missing"
            self.model_status_update.emit("Benepar", status)
            self._append_log(f"Benepar model status: {status}")
            if not benepar_ok: models_to_install["Benepar"] = install_benepar_model

            message = "Checking NLTK models..."
            self._append_log(message)
            self.model_status_update.emit("NLTK", "Checking...")
            nltk_ok = check_nltk_models()
            status = "✓ Present" if nltk_ok else "✗ Missing"
            self.model_status_update.emit("NLTK", status)
            self._append_log(f"NLTK models status: {status}")
            if not nltk_ok: models_to_install["NLTK"] = install_nltk_models

            # --- Install Missing Models ---
            if not models_to_install:
                message = "All required models are already present!"
                self.progress_update.emit(message)
                self._append_log(message)
                self.setup_log_completed.emit("\n".join(self.setup_log))
                self.finished.emit(True)
                return

            message = f"Found {len(models_to_install)} missing model(s). Starting download/installation..."
            self.progress_update.emit(message)
            self._append_log(message)

            for name, install_func in models_to_install.items():
                try:
                    message = f"Setting up {name} model..."
                    self._append_log(message)
                    self.model_status_update.emit(name, "⏳ Downloading/Installing...")
                    self.progress_update.emit(message)
                    # We assume the download function handles both download and setup
                    # In a real scenario, these might need separate steps/signals
                    success = install_func()
                    if success:
                        # Verify after download
                        if name == "spaCy": spacy_ok = check_spacy_model()
                        elif name == "Benepar": benepar_ok = check_benepar_model()
                        elif name == "NLTK": nltk_ok = check_nltk_models()

                        if (name == "spaCy" and spacy_ok) or \
                           (name == "Benepar" and benepar_ok) or \
                           (name == "NLTK" and nltk_ok):
                            status = "✓ Installed"
                            self.model_status_update.emit(name, status)
                            self._append_log(f"{name} model {status}")
                        else:
                            # This case indicates download reported success, but check failed.
                            status = "✗ Verification Failed"
                            self.model_status_update.emit(name, status)
                            self._append_log(f"{name} model {status}")
                            all_successful = False
                    else:
                        status = "✗ Installation Failed"
                        self.model_status_update.emit(name, status)
                        self._append_log(f"{name} model {status}")
                        all_successful = False
                        # Option: stop on first failure? Or attempt all? Currently attempts all.
                except Exception as e:
                    error_message = f"✗ Error: {str(e)}"
                    self.model_status_update.emit(name, error_message)
                    self._append_log(f"Error installing {name}: {str(e)}")
                    all_successful = False
                    self.progress_update.emit(f"Error installing {name}: {str(e)}")

            # --- Final Status ---
            if all_successful:
                message = "All required models installed successfully!"
                self.progress_update.emit(message)
                self._append_log(message)
            else:
                message = "Some models failed to install. Please check status and logs."
                self.progress_update.emit(message)
                self._append_log(message)

            self.setup_log_completed.emit("\n".join(self.setup_log))
            self.finished.emit(all_successful)

        except Exception as e:
            message = f"An unexpected error occurred during setup: {str(e)}"
            self.progress_update.emit(message)
            self._append_log(message)
            self.setup_log_completed.emit("\n".join(self.setup_log))
            self.finished.emit(False)

# --- Wizard Pages ---

# Helper function to create standard page layout
def create_page_layout(page: QWizardPage):
    layout = QVBoxLayout(page)
    layout.setContentsMargins(30, 30, 30, 30)  # Increased margins for more whitespace
    layout.setSpacing(12)  # Consistent spacing

    # Create a QGridLayout for perfect alignment
    header_container = QWidget()
    header_container.setContentsMargins(0, 0, 0, 0)
    header_grid = QGridLayout(header_container)
    header_grid.setContentsMargins(0, 0, 0, 0)
    header_grid.setSpacing(2)
    header_grid.setColumnStretch(1, 1)  # Make second column stretch
    
    # Title with controlled styling
    page.title_label = QLabel("Default Title")
    page.title_label.setObjectName("PageTitle")
    page.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    page.title_label.setContentsMargins(0, 0, 0, 0)
    page.title_label.setIndent(0)
    page.title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    
    # Status with identical alignment settings
    page.status_indicator_label = QLabel("Default Status")
    page.status_indicator_label.setObjectName("StatusIndicator")
    page.status_indicator_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    page.status_indicator_label.setContentsMargins(0, 0, 0, 0)
    page.status_indicator_label.setIndent(0)
    page.status_indicator_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    page.status_indicator_label.setWordWrap(True)
    
    # Position labels in the grid - both in column 0 for perfect alignment
    header_grid.addWidget(page.title_label, 0, 0)
    header_grid.addWidget(page.status_indicator_label, 1, 0)
    
    # Separator line for visual clarity
    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setFrameShadow(QFrame.Shadow.Sunken)
    separator.setStyleSheet(f"background-color: {BORDER_COLOR}; max-height: 1px;")

    # Add header and separator
    layout.addWidget(header_container)
    layout.addWidget(separator)
    layout.addSpacing(10)  # Space after separator
    
    # Add explainer below separator
    page.explainer_label = QLabel("Default explanation.")
    page.explainer_label.setObjectName("ExplanatoryText")
    page.explainer_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
    page.explainer_label.setWordWrap(True)
    page.explainer_label.setContentsMargins(0, 0, 0, 0)
    page.explainer_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    layout.addWidget(page.explainer_label)
    layout.addSpacing(15)  # Space before content

    # Content area uses a widget to allow better styling
    content_widget = QWidget()
    content_widget.setObjectName("ContentArea")
    content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    page.content_layout = QVBoxLayout(content_widget)
    page.content_layout.setContentsMargins(0, 0, 0, 0)
    page.content_layout.setSpacing(15)
    layout.addWidget(content_widget, 1)  # Give content area ability to expand

    return layout

class WelcomePage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = create_page_layout(self)

        # Configure standard elements
        self.title_label.setText("Welcome to ANPE Setup")
        self.status_indicator_label.setText("Setup models to use ANPE")
        self.status_indicator_label.setStyleSheet(f"color: {NEUTRAL_COLOR};")

        self.explainer_label.setText(
            "ANPE requires essential language models to function correctly as a part of its dependency parsing pipeline. "
            "This wizard will guide you through the download and installation process.\n\n"
            "The following models will be checked and installed if missing:"
        )
        
        # Enhanced layout for model requirements
        models_group = QGroupBox("Required Models")
        models_layout = QGridLayout()
        models_layout.setVerticalSpacing(10)
        models_layout.setHorizontalSpacing(15)
        models_layout.setContentsMargins(10, 15, 10, 15)  # Reduce left/right margins
        
        for row, (model_name, size, desc) in enumerate([
            ("spaCy English model", "31MB", "Language processing and word embeddings"),
            ("Benepar parsing model", "67MB", "Berkeley Neural Parser for constituency parsing"),
            ("NLTK tokenizer data", "20MB", "Natural Language Toolkit for text segmentation")
        ]):
            # Name with size
            name_label = QLabel(f"<b>{model_name}</b> <span style='color: {NEUTRAL_COLOR}; font-size: 9pt;'>({size})</span>")
            
            # Description
            desc_label = QLabel(desc)
            desc_label.setStyleSheet(f"color: {NEUTRAL_COLOR}; font-size: 9pt;")
            
            # Use column 0 for name since we're removing the bullet points
            models_layout.addWidget(name_label, row, 0)
            models_layout.addWidget(desc_label, row, 1)
        
        models_group.setLayout(models_layout)
        self.content_layout.addWidget(models_group)

        # Enhanced requirements box
        req_box = QGroupBox("System Requirements")
        req_layout = QGridLayout()
        req_layout.setColumnStretch(1, 1)  # Make second column expand
        
        # Disk space
        disk_icon = QLabel("💾")
        disk_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        disk_label = QLabel("<b>Disk Space:</b>")
        disk_value = QLabel("At least 200MB free")
        disk_value.setStyleSheet(f"color: {NEUTRAL_COLOR};")
        
        # Internet
        net_icon = QLabel("🌐")
        net_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        net_label = QLabel("<b>Internet Connection:</b>")
        net_value = QLabel("Required for download")
        net_value.setStyleSheet(f"color: {NEUTRAL_COLOR};")
        
        # Time
        time_icon = QLabel("⏱️")
        time_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        time_label = QLabel("<b>Estimated Time:</b>")
        time_value = QLabel("1-5 minutes (depending on connection speed)")
        time_value.setStyleSheet(f"color: {NEUTRAL_COLOR};")
        
        req_layout.addWidget(disk_icon, 0, 0)
        req_layout.addWidget(disk_label, 0, 1)
        req_layout.addWidget(disk_value, 0, 2)
        
        req_layout.addWidget(net_icon, 1, 0)
        req_layout.addWidget(net_label, 1, 1)
        req_layout.addWidget(net_value, 1, 2)
        
        req_layout.addWidget(time_icon, 2, 0)
        req_layout.addWidget(time_label, 2, 1)
        req_layout.addWidget(time_value, 2, 2)
        
        req_box.setLayout(req_layout)
        self.content_layout.addWidget(req_box)
        
        # Add a note at the bottom
        note_label = QLabel("Please ensure you have a stable internet connection before proceeding.")
        note_label.setStyleSheet(f"color: {WARNING_COLOR}; font-style: italic; margin-top: 10px;")
        note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(note_label)

class DownloadPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCommitPage(True)

        layout = create_page_layout(self)

        # Override the button text immediately after setting commit page
        if self.wizard():
            self.wizard().setButtonText(QWizard.WizardButton.CommitButton, "Next")

        # Configure standard elements
        self.title_label.setText("Checking & Installing Models")
        self.status_indicator_label.setText("Preparing...")
        self.status_indicator_label.setStyleSheet(f"color: {NEUTRAL_COLOR};")
        self.explainer_label.setText("Please wait while the required models are checked and installed if necessary.")

        # Enhanced status group with better layout
        status_group = QGroupBox("Model Status")
        status_layout = QGridLayout()
        status_layout.setVerticalSpacing(10)
        status_layout.setHorizontalSpacing(15)
        status_layout.setColumnStretch(1, 1)  # Make description column expandable

        # Add status labels for each model component
        for row, (model_name, description) in enumerate([
            ("spaCy", "Python NLP library"),
            ("Benepar", "Berkeley Neural Parser"),
            ("NLTK", "Natural Language Toolkit")
        ]):
            # Component name (static) with improved styling
            name_widget = QLabel(f"{model_name}:")
            name_widget.setStyleSheet(f"font-weight: bold; color: {PRIMARY_COLOR};")
            status_layout.addWidget(name_widget, row, 0)
            
            # Description (static) with improved styling
            desc_widget = QLabel(description)
            desc_widget.setStyleSheet(f"color: {NEUTRAL_COLOR};")
            status_layout.addWidget(desc_widget, row, 1)
            
            # Status (dynamic) with improved styling
            status_widget = QLabel("⏳ Pending...")
            status_widget.setStyleSheet(f"color: {WARNING_COLOR}; padding-left: 10px;")
            if model_name == "spaCy":
                self.spacy_status = status_widget
            elif model_name == "Benepar":
                self.benepar_status = status_widget
            else:
                self.nltk_status = status_widget
            
            # Align status widget to left but allow expanding
            status_layout.addWidget(status_widget, row, 2)
            status_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        status_group.setLayout(status_layout)
        self.content_layout.addWidget(status_group)

        # Improved activity indicator layout
        self.activity_container = QWidget()
        self.activity_container.setObjectName("ActivityIndicatorContainer")
        self.activity_layout = QHBoxLayout(self.activity_container)
        self.activity_layout.setContentsMargins(0, 15, 0, 15)
        
        self.activity_indicator = PulsingActivityIndicator()
        self.activity_indicator.set_color(QColor(ACCENT_COLOR))  # Match accent color
        
        self.activity_status = QLabel("Initializing...")
        self.activity_status.setStyleSheet(f"color: {ACCENT_COLOR}; margin-left: 15px; font-weight: bold;")
        
        self.activity_layout.addWidget(self.activity_indicator)
        self.activity_layout.addWidget(self.activity_status)
        self.activity_layout.addStretch()
        
        self.content_layout.addWidget(self.activity_container)

        # --- Worker Thread setup ---
        self.thread = None
        self.worker = None
        self._setup_success = None
        self.setup_log = ""

    def initializePage(self):
        # Set the commit button text to "Next" explicitly
        if self.wizard():
            self.wizard().setButtonText(QWizard.WizardButton.CommitButton, "Next")
            
        # Reset status on re-entry
        self.status_indicator_label.setText("Preparing...")
        self.status_indicator_label.setStyleSheet(f"color: {NEUTRAL_COLOR};")
        self.explainer_label.setText("Please wait while the required models are checked and installed if necessary.")
        self.update_model_status("spaCy", "⏳ Pending...")
        self.update_model_status("Benepar", "⏳ Pending...")
        self.update_model_status("NLTK", "⏳ Pending...")
        
        # Start the activity indicator
        self.activity_indicator.start()
        self.activity_status.setText("Checking required models...")
        
        self._setup_success = None
        self.setup_log = ""
        self.wizard().button(QWizard.WizardButton.NextButton).setEnabled(False)

        if self.thread is None:
            self.thread = QThread()
            self.worker = SetupWorker()
            self.worker.moveToThread(self.thread)
            self.worker.progress_update.connect(self.update_progress_label)
            self.worker.model_status_update.connect(self.update_model_status)
            self.worker.finished.connect(self.on_setup_finished)
            self.worker.setup_log_completed.connect(self.on_log_completed)
            self.thread.started.connect(self.worker.run)
            self.thread.start()

    def update_progress_label(self, message):
        """Update the main explainer text and status indicator."""
        self.explainer_label.setText(message)
        self.activity_status.setText(message)

        # Update status indicator based on keywords
        status_text = "Status: In Progress"
        status_color = WARNING_COLOR # Orange/Yellow for progress

        if "Starting download" in message or "Setting up" in message:
            status_text = "Status: Installing Models..."
        elif "checking" in message.lower():
             status_text = "Status: Checking Models..."
        elif "All required models are already present" in message:
            status_text = "Status: Models Already Present"
            status_color = SUCCESS_COLOR
        elif "installed successfully" in message:
             status_text = "Status: Installation Complete"
             status_color = SUCCESS_COLOR
        elif "failed" in message.lower() or "error occurred" in message.lower():
             status_text = "Status: Error Occurred"
             status_color = ERROR_COLOR

        self.status_indicator_label.setText(status_text)
        self.status_indicator_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")

    def update_model_status(self, model_name, status_text):
        """Update individual model status label + styles."""
        target_label = None
        if model_name == "spaCy": target_label = self.spacy_status
        elif model_name == "Benepar": target_label = self.benepar_status
        elif model_name == "NLTK": target_label = self.nltk_status

        if target_label:
            target_label.setText(status_text)
            style = "padding-left: 10px;" # Base padding
            font_weight = "normal"
            if status_text.startswith("✓"): # Success
                style += f" color: {SUCCESS_COLOR};"
                font_weight = "bold"
            elif status_text.startswith("✗"): # Failure/Error
                style += f" color: {ERROR_COLOR};"
                font_weight = "bold"
            elif status_text.startswith("⏳") or "Checking" in status_text: # In progress
                style += f" color: {WARNING_COLOR};"
            else: # Default/Pending
                style += f" color: {NEUTRAL_COLOR};"

            target_label.setStyleSheet(f"QLabel {{ {style}; font-weight: {font_weight}; }}")

    def on_log_completed(self, log_text):
        """Store the log for the completion page."""
        self.setup_log = log_text
        self.wizard()._setup_log = log_text  # Store in wizard for other pages to access

    def on_setup_finished(self, success):
        # Stop activity indicator
        self.activity_indicator.stop()
        
        self._setup_success = success
        self.wizard()._setup_success = success

        if success:
            final_message = "Setup completed successfully! Click Next to continue."
            final_status = "Status: Installation Complete"
            status_color = SUCCESS_COLOR
            self.activity_status.setText("Installation Complete")
            self.activity_status.setStyleSheet(f"color: {SUCCESS_COLOR}; margin-left: 15px; font-weight: bold;")
        else:
            final_message = "Setup failed or was incomplete. Please check the status above. Click Next to see details."
            final_status = "Status: Installation Failed"
            status_color = ERROR_COLOR
            self.activity_status.setText("Installation Failed")
            self.activity_status.setStyleSheet(f"color: {ERROR_COLOR}; margin-left: 15px; font-weight: bold;")

        self.explainer_label.setText(final_message)
        self.status_indicator_label.setText(final_status)
        self.status_indicator_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")

        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None

        # Enable Next button after the animation completes
        QTimer.singleShot(1000, lambda: self.wizard().button(QWizard.WizardButton.NextButton).setEnabled(True))
        QTimer.singleShot(1000, lambda: self.completeChanged.emit())

    def isComplete(self) -> bool:
        """Enable next button only when setup is done."""
        # Also ensures that Next is only available *after* the worker finishes
        return self._setup_success is not None

    def nextId(self) -> int:
        """Determine the next page based on setup result."""
        # Normal behavior: go to success or failure page based on setup result
        if self._setup_success is True:
            return SetupWizard.Page_Success
        else:
            return SetupWizard.Page_Failure


class CompletionPage(QWizardPage):
    def __init__(self, parent=None, success=True):
        super().__init__(parent)
        self.success = success

        layout = create_page_layout(self) # Use helper

        if success:
            self.title_label.setText("Setup Completed Successfully")
            self.status_indicator_label.setText("Status: Success")
            self.status_indicator_label.setStyleSheet(f"color: {SUCCESS_COLOR}; font-weight: bold;")
            self.explainer_label.setText(
                "All required models have been installed successfully.\n\n"
                "You can now close this wizard and start using ANPE normally."
            )
            
            # Add log display for success case too
            log_group = QGroupBox("Setup Log")
            log_layout = QVBoxLayout()
            
            self.log_display = QTextEdit()
            self.log_display.setReadOnly(True)
            self.log_display.setStyleSheet(f"""
                QTextEdit {{
                    background-color: {BG_ALT_COLOR};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 4px;
                    font-family: Consolas, Monaco, monospace;
                    font-size: 9pt;
                    line-height: 1.3;
                    padding: 8px;
                }}
            """)
            self.log_display.setMinimumHeight(150)
            
            log_layout.addWidget(self.log_display)
            log_group.setLayout(log_layout)
            self.content_layout.addWidget(log_group)
            
        else:
            self.title_label.setText("Setup Failed")
            self.status_indicator_label.setText("Status: Failure")
            self.status_indicator_label.setStyleSheet(f"color: {ERROR_COLOR}; font-weight: bold;")
            self.explainer_label.setText(
                "Could not complete the model setup successfully.\n\n"
                "This might be due to:\n"
                "• Network connectivity issues\n"
                "• Insufficient disk space\n"
                "• Permission problems\n\n"
                "Please check the setup log below for more details:"
            )
            
            # Remove troubleshooting tips panel and directly show logs
            
            # Add log display with enhanced styling
            log_group = QGroupBox("Setup Log")
            log_layout = QVBoxLayout()
            
            self.log_display = QTextEdit()
            self.log_display.setReadOnly(True)
            self.log_display.setStyleSheet(f"""
                QTextEdit {{
                    background-color: {BG_ALT_COLOR};
                    border: 1px solid {BORDER_COLOR};
                    border-radius: 4px;
                    font-family: Consolas, Monaco, monospace;
                    font-size: 9pt;
                    line-height: 1.3;
                    padding: 8px;
                }}
            """)
            self.log_display.setMinimumHeight(250)  # Increased height since we removed the tips
            
            log_layout.addWidget(self.log_display)
            log_group.setLayout(log_layout)
            self.content_layout.addWidget(log_group)

    def initializePage(self):
        """Load the setup log when the page is shown."""
        if hasattr(self.wizard(), '_setup_log'):
            self.log_display.setText(self.wizard()._setup_log)

    def nextId(self) -> int:
        return -1 # Last page

# --- Main Wizard Class ---

class SetupWizard(QWizard):
    Page_Welcome = 1
    Page_Download = 2
    Page_Success = 3
    Page_Failure = 4

    setup_complete = pyqtSignal()
    setup_cancelled = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setWindowTitle("ANPE Setup")
        self.setMinimumSize(700, 550)  # Increase size for better layout

        # --- Global Stylesheet ---
        self.setStyleSheet(f"""
            QWizard {{
                background-color: {BG_COLOR};
            }}
            
            QWizardPage {{
                background-color: {BG_COLOR};
            }}

            /* Standardized Page Elements */
            QLabel#PageTitle {{
                font-size: 18pt;
                font-weight: bold;
                color: {PRIMARY_COLOR};
                margin: 0;
                padding: 0;
                text-indent: 0;
            }}
            
            QLabel#StatusIndicator {{
                font-size: 12pt;
                color: {NEUTRAL_COLOR};
                margin: 0;
                padding: 0;
                text-indent: 0;
            }}
            
            QLabel#ExplanatoryText {{
                font-size: 10pt;
                color: {PRIMARY_COLOR};
                line-height: 1.5;
                margin: 0;
                padding: 0;
            }}

            /* General QLabel */
            QLabel {{
                font-size: 10pt;
                color: {PRIMARY_COLOR};
            }}
            
            /* GroupBox Styling */
            QGroupBox {{
                font-weight: bold;
                font-size: 11pt;
                margin-top: 15px;
                padding: 15px;
                padding-top: 20px;
                border: 1px solid {BORDER_COLOR};
                border-radius: 6px;
                background-color: {BG_COLOR};
            }}
            
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 8px;
                margin-left: 8px;
                color: {PRIMARY_COLOR};
            }}

            /* Buttons */
            QPushButton {{
                background-color: {ACCENT_COLOR};
                color: white;
                padding: 10px 20px;
                font-size: 10pt;
                font-weight: bold;
                border: none;
                border-radius: 4px;
                min-width: 100px;
            }}
            
            QPushButton:disabled {{
                background-color: {NEUTRAL_COLOR};
                opacity: 0.7;
            }}
            
            QPushButton:hover:!disabled {{
                background-color: #2980b9;
            }}
            
            QPushButton:pressed:!disabled {{
                background-color: #1c6ea4;
            }}
            
            /* QTextEdit for Logs */
            QTextEdit {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                font-size: 9pt;
                selection-background-color: {ACCENT_COLOR};
                selection-color: white;
            }}
        """)

        # Add pages
        self.setPage(self.Page_Welcome, WelcomePage(self))
        self.setPage(self.Page_Download, DownloadPage(self))
        self.setPage(self.Page_Success, CompletionPage(self, success=True))
        self.setPage(self.Page_Failure, CompletionPage(self, success=False))

        # Connect signals
        self.finished.connect(self._on_wizard_finished)
        
        # Customize button text for clarity
        self.setButtonText(QWizard.WizardButton.NextButton, "Continue")
        self.setButtonText(QWizard.WizardButton.BackButton, "Back")
        self.setButtonText(QWizard.WizardButton.FinishButton, "Finish")
        self.setButtonText(QWizard.WizardButton.CancelButton, "Cancel")
        
        # Workaround for Next button initial state on DownloadPage
        self.currentIdChanged.connect(self._handle_page_change)


    def _handle_page_change(self, page_id):
        """Ensure Next button is disabled when entering Download page."""
        if page_id == self.Page_Download:
            # Ensure the button is disabled before initializePage runs fully
            QCoreApplication.processEvents() # Allow event queue to process potential earlier enables
            self.button(QWizard.WizardButton.NextButton).setEnabled(False)
            # Optionally hide Back button explicitly, though commit page should handle it
            self.button(QWizard.WizardButton.BackButton).setVisible(False)
        else:
             # Ensure back button is visible on other pages if needed
             self.button(QWizard.WizardButton.BackButton).setVisible(True)


    def _on_wizard_finished(self, result):
        """Handle wizard completion, ensuring signals are properly disconnected after emission."""
        try:
            # If accepted and setup was successful
            if result == QDialog.DialogCode.Accepted and hasattr(self, '_setup_success') and self._setup_success:
                logging.info("Setup wizard completed successfully, emitting setup_complete signal")
                self.setup_complete.emit()
            else:
                # Check if cancellation was explicit or due to failure navigation
                current_page_id = self.currentId()
                
                # If user clicked Finish/Commit on the Failure page or clicked Cancel/closed the dialog
                if (current_page_id == self.Page_Failure and result == QDialog.DialogCode.Accepted) or \
                   (result != QDialog.DialogCode.Accepted):
                    
                    # Check if models are present despite cancellation
                    try:
                        from anpe.utils.setup_models import check_all_models_present
                        models_present = check_all_models_present()
                        
                        if models_present:
                            # Models are actually present, treat as success
                            logging.info("Setup wizard cancelled, but models are present. Emitting setup_complete signal")
                            self.setup_complete.emit()
                        else:
                            # Models are missing, emit cancellation signal
                            logging.info("Setup wizard cancelled, models not present. Emitting setup_cancelled signal")
                            self.setup_cancelled.emit()
                    except Exception as model_check_error:
                        # If model check fails, assume models are missing
                        logging.error(f"Error checking model presence after cancellation: {model_check_error}", exc_info=True)
                        self.setup_cancelled.emit()
            
            # Ensure wizard closes properly
            QCoreApplication.processEvents()
            
        except Exception as e:
            logging.error(f"Error in setup wizard finished handler: {e}", exc_info=True)

        # Override button text
        if self.currentId() == self.Page_Download and result == QDialog.DialogCode.Accepted:
            self.button(QWizard.WizardButton.CommitButton).setText("&Next")
        elif (self.currentId() == self.Page_Success or self.currentId() == self.Page_Failure) and result == QDialog.DialogCode.Accepted:
            self.button(QWizard.WizardButton.FinishButton).setText("&Finish")


# Example usage
if __name__ == '__main__':
    app = QApplication(sys.argv)
    wizard = SetupWizard()
    wizard.show()
    sys.exit(app.exec()) 