import sys
from PyQt6.QtWidgets import (QApplication, QWizard, QWizardPage, QVBoxLayout, QLabel,
                             QProgressBar, QPushButton, QMessageBox, QGroupBox, QDialog,
                             QSizePolicy, QGridLayout, QFrame)
from PyQt6.QtCore import pyqtSignal, QThread, QObject, Qt
from anpe.utils.setup_models import (setup_models, check_spacy_model,
                                   check_benepar_model, check_nltk_models,
                                   install_spacy_model, install_benepar_model, install_nltk_models)

# --- Worker Thread for Downloads ---
class SetupWorker(QObject):
    finished = pyqtSignal(bool)
    progress_update = pyqtSignal(str)
    model_status_update = pyqtSignal(str, str)

    def run(self):
        """Runs the model setup process with detailed status updates."""
        all_successful = True
        try:
            self.progress_update.emit("Starting model check...")

            # --- Check Models ---
            models_to_install = {}

            self.model_status_update.emit("spaCy", "Checking...")
            spacy_ok = check_spacy_model()
            self.model_status_update.emit("spaCy", "✓ Present" if spacy_ok else "✗ Missing")
            if not spacy_ok: models_to_install["spaCy"] = install_spacy_model

            self.model_status_update.emit("Benepar", "Checking...")
            benepar_ok = check_benepar_model()
            self.model_status_update.emit("Benepar", "✓ Present" if benepar_ok else "✗ Missing")
            if not benepar_ok: models_to_install["Benepar"] = install_benepar_model

            self.model_status_update.emit("NLTK", "Checking...")
            nltk_ok = check_nltk_models()
            self.model_status_update.emit("NLTK", "✓ Present" if nltk_ok else "✗ Missing")
            if not nltk_ok: models_to_install["NLTK"] = install_nltk_models

            # --- Install Missing Models ---
            if not models_to_install:
                self.progress_update.emit("All required models are already present!")
                self.finished.emit(True)
                return

            self.progress_update.emit(f"Found {len(models_to_install)} missing model(s). Starting download/installation...")

            for name, install_func in models_to_install.items():
                try:
                    self.model_status_update.emit(name, "⏳ Downloading/Installing...")
                    self.progress_update.emit(f"Setting up {name} model...")
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
                            self.model_status_update.emit(name, "✓ Installed")
                        else:
                            # This case indicates download reported success, but check failed.
                            self.model_status_update.emit(name, "✗ Verification Failed")
                            all_successful = False
                    else:
                        self.model_status_update.emit(name, "✗ Installation Failed")
                        all_successful = False
                        # Option: stop on first failure? Or attempt all? Currently attempts all.
                except Exception as e:
                    self.model_status_update.emit(name, f"✗ Error: {str(e)}")
                    all_successful = False
                    self.progress_update.emit(f"Error installing {name}: {str(e)}")

            # --- Final Status ---
            if all_successful:
                self.progress_update.emit("All required models installed successfully!")
            else:
                self.progress_update.emit("Some models failed to install. Please check status and logs.")

            self.finished.emit(all_successful)

        except Exception as e:
            self.progress_update.emit(f"An unexpected error occurred during setup: {str(e)}")
            self.finished.emit(False)

# --- Wizard Pages ---

# Helper function to create standard page layout
def create_page_layout(page: QWizardPage):
    layout = QVBoxLayout(page)
    layout.setContentsMargins(25, 20, 25, 20)
    layout.setSpacing(10)

    page.title_label = QLabel("Default Title")
    page.title_label.setObjectName("PageTitle")
    page.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
    page.title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred) # Force horizontal expansion

    page.status_indicator_label = QLabel("Default Status")
    page.status_indicator_label.setObjectName("StatusIndicator")
    page.status_indicator_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
    page.status_indicator_label.setWordWrap(True)
    page.status_indicator_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred) # Force horizontal expansion

    page.explainer_label = QLabel("Default explanation.")
    page.explainer_label.setObjectName("ExplanatoryText")
    page.explainer_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
    page.explainer_label.setWordWrap(True)
    page.explainer_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred) # Force horizontal expansion

    layout.addWidget(page.title_label)
    layout.addWidget(page.status_indicator_label)
    layout.addWidget(page.explainer_label)

    layout.addSpacing(15)

    page.content_layout = QVBoxLayout()
    page.content_layout.setSpacing(15)
    layout.addLayout(page.content_layout)

    layout.addStretch(1)

    return layout

class WelcomePage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = create_page_layout(self)

        # Configure standard elements
        self.title_label.setText("Welcome to ANPE Setup")
        self.status_indicator_label.setText("Model Setup Required")
        self.status_indicator_label.setStyleSheet("color: #34495e;") # Neutral color

        self.explainer_label.setText(
            "ANPE requires essential language models to function correctly as a part of its dependency parsing pipeline. "
            "This wizard will guide you through the download and installation process.\n\n"
            "The following models will be checked and installed if missing:\n"
            "• spaCy English model (en_core_web_md, ≈ 31MB)\n" 
            "• Benepar parsing model (benepar_en3, ≈ 67MB)\n"   
            "• NLTK tokenizer data (punkt & punkt_tab, ≈ 20MB)\n\n" #
            "Please ensure you have a stable internet connection."
        )

        # Page-specific content (Requirements Box)
        req_box = QGroupBox("Requirements")
        req_layout = QVBoxLayout()
        req_layout.addWidget(QLabel("• <b>Disk Space:</b> At least 200MB free")) 
        req_layout.addWidget(QLabel("• <b>Internet Connection:</b> Required for download"))
        req_layout.addWidget(QLabel("• <b>Estimated Time:</b> 1-5 minutes (depending on connection speed)")) # Adjusted time
        req_box.setLayout(req_layout)
        self.content_layout.addWidget(req_box) # Add to content area

class DownloadPage(QWizardPage):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCommitPage(True)

        layout = create_page_layout(self)

        # Configure standard elements
        self.title_label.setText("Checking & Installing Models")
        self.status_indicator_label.setText("Preparing...")
        self.status_indicator_label.setStyleSheet("color: #7f8c8d;")
        self.explainer_label.setText("Please wait while the required models are checked and installed if necessary.")

        # Page-specific content (Status Group and Progress Bar)

        status_group = QGroupBox("Individual Model Status")
        status_layout = QGridLayout()
        status_layout.setColumnStretch(1, 1) # Ensure status column expands
        status_layout.setHorizontalSpacing(20) # Add horizontal spacing between columns
        # status_layout.setColumnMinimumWidth(0, 180) # Give label column a fixed minimum width - commented out, let's try sizing policy first

        # Status labels
        self.spacy_label = QLabel("spaCy (en_core_web_md):")
        self.benepar_label = QLabel("Benepar (benepar_en3):")
        self.nltk_label = QLabel("NLTK (punkt, punkt_tab):")

        # Set size policy for labels to prevent them from shrinking too much but allow expansion if needed
        # self.spacy_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        # self.benepar_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        # self.nltk_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self.spacy_status = QLabel("⏳ Pending...")
        self.benepar_status = QLabel("⏳ Pending...")
        self.nltk_status = QLabel("⏳ Pending...")

        # Add widgets to grid layout
        status_layout.addWidget(self.spacy_label, 0, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        status_layout.addWidget(self.spacy_status, 0, 1)
        status_layout.addWidget(self.benepar_label, 1, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        status_layout.addWidget(self.benepar_status, 1, 1)
        status_layout.addWidget(self.nltk_label, 2, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        status_layout.addWidget(self.nltk_status, 2, 1)

        # Apply alignment and padding to status labels only
        for i in range(status_layout.rowCount()):
             status_layout.setRowMinimumHeight(i, 30) # Keep vertical spacing
             # label_widget = status_layout.itemAtPosition(i, 0).widget()
             status_widget = status_layout.itemAtPosition(i, 1).widget()
             # if label_widget: label_widget.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter) # Alignment set during addWidget
             if status_widget:
                 status_widget.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                 # status_widget.setStyleSheet("padding-left: 10px;") # Padding handled by grid spacing now

        status_group.setLayout(status_layout)
        self.content_layout.addWidget(status_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        self.content_layout.addWidget(self.progress_bar)

        # --- Worker Thread setup (remains mostly the same) ---
        self.thread = None
        self.worker = None
        self._setup_success = None

    def initializePage(self):
        # Reset status on re-entry
        self.status_indicator_label.setText("Preparing...")
        self.status_indicator_label.setStyleSheet("color: #7f8c8d;")
        self.explainer_label.setText("Please wait while the required models are checked and installed if necessary.")
        self.update_model_status("spaCy", "⏳ Pending...")
        self.update_model_status("Benepar", "⏳ Pending...")
        self.update_model_status("NLTK", "⏳ Pending...")
        self.progress_bar.setRange(0, 0)
        self._setup_success = None
        self.wizard().button(QWizard.WizardButton.NextButton).setEnabled(False)

        if self.thread is None:
            self.thread = QThread()
            self.worker = SetupWorker()
            self.worker.moveToThread(self.thread)
            self.worker.progress_update.connect(self.update_progress_label)
            self.worker.model_status_update.connect(self.update_model_status)
            self.worker.finished.connect(self.on_setup_finished)
            self.thread.started.connect(self.worker.run)
            self.thread.start()

    def update_progress_label(self, message):
        """Update the main explainer text and status indicator."""
        self.explainer_label.setText(message)

        # Update status indicator based on keywords
        status_text = "Status: In Progress"
        status_color = "#f39c12" # Orange/Yellow for progress

        if "Starting download" in message or "Setting up" in message:
            status_text = "Status: Installing Models..."
        elif "checking" in message.lower():
             status_text = "Status: Checking Models..."
        elif "All required models are already present" in message:
            status_text = "Status: Models Already Present"
            status_color = "#27ae60" # Green
        elif "installed successfully" in message:
             status_text = "Status: Installation Complete"
             status_color = "#27ae60" # Green
        elif "failed" in message.lower() or "error occurred" in message.lower():
             status_text = "Status: Error Occurred"
             status_color = "#c0392b" # Red

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
                style += " color: #27ae60;" # Green
                font_weight = "bold"
            elif status_text.startswith("✗"): # Failure/Error
                style += " color: #c0392b;" # Red
                font_weight = "bold"
            elif status_text.startswith("⏳") or "Checking" in status_text: # In progress
                style += " color: #f39c12;" # Orange/Yellow
            else: # Default/Pending
                style += " color: #7f8c8d;" # Grey

            target_label.setStyleSheet(f"QLabel {{ {style}; font-weight: {font_weight}; }}")

    def on_setup_finished(self, success):
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self._setup_success = success
        self.wizard()._setup_success = success

        if success:
            final_message = "Setup completed successfully! Click Next to continue."
            final_status = "Status: Installation Complete"
            status_color = "#27ae60" # Green
        else:
            final_message = "Setup failed or was incomplete. Please check the status above. Click Next to see details."
            final_status = "Status: Installation Failed"
            status_color = "#c0392b" # Red

        self.explainer_label.setText(final_message)
        self.status_indicator_label.setText(final_status)
        self.status_indicator_label.setStyleSheet(f"color: {status_color}; font-weight: bold;")

        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None

        self.wizard().button(QWizard.WizardButton.NextButton).setEnabled(True)
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        """Enable next button only when setup is done."""
        # Also ensures that Next is only available *after* the worker finishes
        return self._setup_success is not None

    def nextId(self) -> int:
        """Determine the next page based on setup result."""
        # Default to failure page if something unexpected happens
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
            self.status_indicator_label.setStyleSheet("color: #27ae60; font-weight: bold;") # Green
            self.explainer_label.setText(
                "All required models have been installed successfully.\n\n"
                "You can now close this wizard and start using ANPE normally."
            )
        else:
            self.title_label.setText("Setup Failed")
            self.status_indicator_label.setText("Status: Failure")
            self.status_indicator_label.setStyleSheet("color: #c0392b; font-weight: bold;") # Red
            self.explainer_label.setText(
                "Could not complete the model setup successfully.\n\n"
                "This might be due to:\n"
                "• Network connectivity issues\n"
                "• Insufficient disk space\n"
                "• Permission problems\n\n"
                "Please check your internet connection, ensure sufficient disk space (~200MB), and try again.\n"
            )

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
        self.setMinimumSize(600, 480)

        # --- Global Stylesheet ---
        self.setStyleSheet("""
            QWizard {
                background-color: #f8f9fa;
            }
            QWizardPage {
                background-color: white;
            }

            /* Standardized Page Elements */
            QLabel#PageTitle {
                font-size: 16pt;
                font-weight: bold;
                color: #2c3e50; /* Deep Blue-Grey */
                margin-bottom: 5px;
            }
            QLabel#StatusIndicator {
                font-size: 12pt;
                color: #34495e; /* Default color */
                margin-bottom: 5px;
            }
            QLabel#ExplanatoryText {
                font-size: 10pt; /* Smaller for explanation */
                color: #34495e;
                line-height: 1.4; /* Improve readability */
                margin-bottom: 15px;
            }

            /* General QLabel */
            QLabel {
                font-size: 10pt; /* Base size */
                color: #34495e;
            }
            QLabel b { /* Style bold tags if used */
                font-weight: bold;
            }

            /* GroupBox Styling */
            QGroupBox {
                font-weight: bold;
                font-size: 10pt;
                margin-top: 10px;
                padding: 10px; /* Padding inside */
                padding-top: 15px; /* More padding at top */
                border: 1px solid #e0e0e0;
                border-radius: 4px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px 0px 5px; /* Adjust title padding */
                margin-left: 5px; /* Shift title slightly right */
                color: #34495e;
                font-size: 9pt; /* Smaller title */
            }

            /* ProgressBar */
            QProgressBar {
                border: 1px solid #bdc3c7;
                border-radius: 3px;
                text-align: center;
                background-color: #ecf0f1;
                height: 12px; /* Slightly taller */
                margin-top: 5px;
            }
            QProgressBar::chunk {
                background-color: #3498db;
            }

            /* Buttons */
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 8px 18px; /* Slightly wider padding */
                border: none;
                border-radius: 4px;
                font-size: 10pt;
                min-width: 80px; /* Minimum button width */
            }
            QPushButton:disabled {
                background-color: #bdc3c7;
            }
            QPushButton:hover:!disabled { /* Hover only when enabled */
                background-color: #2980b9;
            }
        """)

        # Add pages
        self.setPage(self.Page_Welcome, WelcomePage(self))
        self.setPage(self.Page_Download, DownloadPage(self))
        self.setPage(self.Page_Success, CompletionPage(self, success=True))
        self.setPage(self.Page_Failure, CompletionPage(self, success=False))

        # Connect signals
        self.finished.connect(self._on_wizard_finished)
        # Set start ID if needed, default is usually the first page added with lowest ID > 0
        # self.setStartId(self.Page_Welcome)

        # Workaround for Next button initial state on DownloadPage
        self.currentIdChanged.connect(self._handle_page_change)


    def _handle_page_change(self, page_id):
        """Ensure Next button is disabled when entering Download page."""
        if page_id == self.Page_Download:
            # Ensure the button is disabled before initializePage runs fully
            QApplication.processEvents() # Allow event queue to process potential earlier enables
            self.button(QWizard.WizardButton.NextButton).setEnabled(False)
            # Optionally hide Back button explicitly, though commit page should handle it
            self.button(QWizard.WizardButton.BackButton).setVisible(False)
        else:
             # Ensure back button is visible on other pages if needed
             self.button(QWizard.WizardButton.BackButton).setVisible(True)


    def _on_wizard_finished(self, result):
        if result == QDialog.DialogCode.Accepted and hasattr(self, '_setup_success') and self._setup_success:
            self.setup_complete.emit()
        else:
            # Check if cancellation was explicit or due to failure navigation
            current_page_id = self.currentId()
            if current_page_id == self.Page_Failure and result == QDialog.DialogCode.Accepted:
                 # User clicked Finish/Commit on the Failure page
                 self.setup_cancelled.emit() # Treat as cancelled/incomplete
            elif result != QDialog.DialogCode.Accepted:
                 # User clicked Cancel or closed the dialog
                 self.setup_cancelled.emit()
            # If setup failed but user finished on Success page (shouldn't happen with nextId logic)
            # or other edge cases, default to cancelled.

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