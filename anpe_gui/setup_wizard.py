import sys
from PyQt6.QtWidgets import (QApplication, QWizard, QWizardPage, QVBoxLayout, QLabel,
                             QProgressBar, QPushButton, QMessageBox, QGroupBox, QDialog)
from PyQt6.QtCore import pyqtSignal, QThread, QObject
from anpe.utils.setup_models import (setup_models, check_spacy_model,  # Import individual functions
                                   check_benepar_model, check_nltk_models)

# --- Worker Thread for Downloads ---
# We need a separate thread to avoid freezing the GUI during downloads
class SetupWorker(QObject):
    finished = pyqtSignal(bool) # Signal indicating success (True) or failure (False)
    progress_update = pyqtSignal(str) # Signal for text updates
    model_check_update = pyqtSignal(str, bool) # Signal for individual model check results

    def run(self):
        """Runs the model setup process."""
        try:
            self.progress_update.emit("Starting model setup process...")
            
            # Check each model individually and report status
            self.progress_update.emit("Checking spaCy model...")
            spacy_ok = check_spacy_model()  # Use imported function directly
            self.model_check_update.emit("spaCy", spacy_ok)
            
            self.progress_update.emit("Checking Benepar model...")
            benepar_ok = check_benepar_model()  # Use imported function directly
            self.model_check_update.emit("Benepar", benepar_ok)
            
            self.progress_update.emit("Checking NLTK models...")
            nltk_ok = check_nltk_models()  # Use imported function directly
            self.model_check_update.emit("NLTK", nltk_ok)
            
            # If all models are present, we're done
            if all([spacy_ok, benepar_ok, nltk_ok]):
                self.progress_update.emit("All models are already present!")
                self.finished.emit(True)
                return
            
            # If any models are missing, proceed with installation
            self.progress_update.emit("Starting model installation...")
            success = setup_models()
            
            if success:
                self.progress_update.emit("All models downloaded and set up successfully!")
            else:
                self.progress_update.emit("Model setup failed. Please check logs or internet connection.")
            
            self.finished.emit(success)
        except Exception as e:
            self.progress_update.emit(f"An error occurred during setup: {str(e)}")
            self.finished.emit(False)

# --- Wizard Pages ---

class WelcomePage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Welcome to ANPE Setup")
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Welcome heading
        heading = QLabel("Model Setup Required")
        heading.setProperty("heading", True)
        layout.addWidget(heading)
        
        # Main description
        desc = QLabel(
            "ANPE requires essential language models to function correctly. "
            "This wizard will help you download and install them.\n\n"
            "The following models will be installed:\n"
            "• spaCy English model (≈ 50MB)\n"
            "• Benepar parsing model (≈ 350MB)\n"
            "• NLTK tokenizer data (≈ 2MB)\n\n"
            "Please ensure you have a stable internet connection before proceeding."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Requirements section
        req_box = QGroupBox("System Requirements")
        req_layout = QVBoxLayout()
        req_layout.addWidget(QLabel("• Disk Space: At least 500MB free"))
        req_layout.addWidget(QLabel("• Internet Connection: Required for download"))
        req_layout.addWidget(QLabel("• Estimated Time: 2-5 minutes"))
        req_box.setLayout(req_layout)
        layout.addWidget(req_box)
        
        layout.addStretch()

class DownloadPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setTitle("Installing Models")
        self.setCommitPage(True)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Status heading
        self.status_heading = QLabel("Downloading Required Models")
        self.status_heading.setProperty("heading", True)
        layout.addWidget(self.status_heading)
        
        # Progress description
        self.progress_label = QLabel("Please wait while the required models are downloaded and installed...")
        self.progress_label.setWordWrap(True)
        layout.addWidget(self.progress_label)
        
        # Model status group
        status_group = QGroupBox("Installation Progress")
        status_layout = QVBoxLayout()
        
        # Status labels with icons
        self.spacy_status = QLabel("spaCy: Checking...")
        self.benepar_status = QLabel("Benepar: Checking...")
        self.nltk_status = QLabel("NLTK: Checking...")
        
        for label in [self.spacy_status, self.benepar_status, self.nltk_status]:
            label.setStyleSheet("QLabel { padding: 5px; }")
            status_layout.addWidget(label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        layout.addWidget(self.progress_bar)
        
        layout.addStretch()
        
        # Setup worker thread
        self.thread = None
        self.worker = None

    def initializePage(self):
        """Start the setup process when page is shown."""
        if self.thread is None:
            self.thread = QThread()
            self.worker = SetupWorker()
            self.worker.moveToThread(self.thread)
            
            # Connect signals
            self.worker.progress_update.connect(self.update_progress_label)
            self.worker.model_check_update.connect(self.update_model_status)
            self.worker.finished.connect(self.on_setup_finished)
            self.thread.started.connect(self.worker.run)
            
            self.thread.start()

    def update_progress_label(self, message):
        """Update the progress description."""
        self.progress_label.setText(message)

    def update_model_status(self, model_name, status):
        """Update individual model status with icons."""
        icon = "✓" if status else "✗"
        status_text = "Present" if status else "Missing"
        style = "color: #27ae60;" if status else "color: #c0392b;"
        
        if model_name == "spaCy":
            self.spacy_status.setText(f"spaCy: {icon} {status_text}")
            self.spacy_status.setStyleSheet(f"QLabel {{ {style} }}")
        elif model_name == "Benepar":
            self.benepar_status.setText(f"Benepar: {icon} {status_text}")
            self.benepar_status.setStyleSheet(f"QLabel {{ {style} }}")
        elif model_name == "NLTK":
            self.nltk_status.setText(f"NLTK: {icon} {status_text}")
            self.nltk_status.setStyleSheet(f"QLabel {{ {style} }}")

    def on_setup_finished(self, success):
        """Handle setup completion."""
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        
        # Store result and determine next page
        self._setup_success = success
        if success:
            self.wizard()._setup_success = True
            self.progress_label.setText("Setup completed successfully!")
            self.status_heading.setText("Installation Complete")
            self.wizard().next()
        else:
            self.wizard()._setup_success = False
            self.progress_label.setText("Setup failed. Please check the error messages above.")
            self.status_heading.setText("Installation Failed")
            self.wizard().next()
        
        # Clean up thread
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None

    def isComplete(self) -> bool:
        """Enable next button only when setup is done."""
        return hasattr(self, '_setup_success')

    def nextId(self) -> int:
        """Determine the next page based on setup result."""
        if hasattr(self, '_setup_success') and self._setup_success:
            return SetupWizard.Page_Success
        return SetupWizard.Page_Failure

class CompletionPage(QWizardPage):
    def __init__(self, parent=None, success=True):
        super().__init__(parent)
        self.success = success
        
        if success:
            self.setTitle("Setup Complete")
            heading = "Setup Completed Successfully"
            message = (
                "All required models have been installed successfully.\n\n"
                "You can now start using ANPE to extract noun phrases from your text."
            )
            color = "#27ae60"  # Green
        else:
            self.setTitle("Setup Failed")
            heading = "Setup Failed"
            message = (
                "Could not complete the model setup.\n\n"
                "This might be due to:\n"
                "• Network connectivity issues\n"
                "• Insufficient disk space\n"
                "• Permission problems\n\n"
                "Please check your internet connection and try again.\n"
                "If the problem persists, check the application logs for more details."
            )
            color = "#c0392b"  # Red
        
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        
        # Status heading
        status_heading = QLabel(heading)
        status_heading.setProperty("heading", True)
        status_heading.setStyleSheet(f"QLabel[heading='true'] {{ color: {color}; }}")
        layout.addWidget(status_heading)
        
        # Status message
        status_message = QLabel(message)
        status_message.setWordWrap(True)
        layout.addWidget(status_message)
        
        layout.addStretch()
    
    def nextId(self) -> int:
        """This is the last page."""
        return -1

# --- Main Wizard Class ---

class SetupWizard(QWizard):
    """Wizard for setting up ANPE models."""
    
    # Page IDs
    Page_Welcome = 1
    Page_Download = 2
    Page_Success = 3
    Page_Failure = 4
    
    # Signals
    setup_complete = pyqtSignal()
    setup_cancelled = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Set wizard style
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)
        self.setWindowTitle("ANPE Setup")
        self.setMinimumSize(600, 400)
        
        # Set custom styling
        self.setStyleSheet("""
            QWizard {
                background-color: white;
            }
            QLabel {
                font-size: 11pt;
                color: #2c3e50;
            }
            QLabel[heading="true"] {
                font-size: 16pt;
                color: #2980b9;
                margin-bottom: 10px;
            }
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                text-align: center;
                background-color: #ecf0f1;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }
        """)
        
        # Add pages
        self.setPage(self.Page_Welcome, WelcomePage(self))
        self.setPage(self.Page_Download, DownloadPage(self))
        self.setPage(self.Page_Success, CompletionPage(self, success=True))
        self.setPage(self.Page_Failure, CompletionPage(self, success=False))
        
        # Connect signals
        self.finished.connect(self._on_wizard_finished)

    def _on_wizard_finished(self, result):
        """Handle wizard completion."""
        if result == QDialog.DialogCode.Accepted and hasattr(self, '_setup_success') and self._setup_success:
            self.setup_complete.emit()
        else:
            self.setup_cancelled.emit()

# Example usage (for testing the wizard independently)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    wizard = SetupWizard()
    wizard.show()
    sys.exit(app.exec()) 