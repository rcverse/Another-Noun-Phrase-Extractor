"""
Dialog for managing ANPE models (spaCy, Benepar, NLTK).
Allows viewing status, running setup, and cleaning models.
"""

import sys
import logging
import nltk # Need nltk for the status check part
from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QGridLayout, QProgressBar, QMessageBox, QWidget, QSpacerItem, QSizePolicy,
    QApplication # Added QApplication import
)
from PyQt6.QtGui import QIcon, QPixmap

from anpe_gui.setup_wizard import SetupWizard # Import SetupWizard

# Assuming these utilities exist and work as expected
try:
    from anpe.utils.setup_models import (
        check_spacy_model, check_benepar_model, check_nltk_models, setup_models
    )
    from anpe.utils.clean_models import clean_all
except ImportError as e:
    logging.error(f"Failed to import ANPE utilities for Model Management: {e}")
    # Provide dummy functions if import fails, so GUI can load with errors
    def check_spacy_model(*args, **kwargs): return False
    def check_benepar_model(*args, **kwargs): return False
    def check_nltk_models(*args, **kwargs): return False
    def setup_models(*args, **kwargs): return False
    def clean_all(*args, **kwargs): return {}
    # Need dummy NLTK for status check part if main import fails
    class DummyNLTKData:
        def find(self, path): raise LookupError()
    class DummyNLTK: data = DummyNLTKData()
    nltk = DummyNLTK()


# --- Worker Objects for Background Tasks ---

class CleanWorker(QObject):
    """Worker to run clean_all in a background thread."""
    finished = pyqtSignal(dict) # Return status dictionary
    progress = pyqtSignal(str) # To send status updates

    @pyqtSlot()
    def run(self):
        self.progress.emit("Starting model cleanup process...")
        try:
            # clean_all also logs. Capture result dict.
            # Pass logger to clean_all if needed
            logger = logging.getLogger('clean_models') # Get the logger clean_models uses
            results = clean_all(logger=logger)
            self.progress.emit("Model cleanup process finished.")
            self.finished.emit(results)
        except Exception as e:
            logging.error(f"Exception during background model cleanup: {e}", exc_info=True)
            self.progress.emit(f"Error during cleanup: {e}")
            self.finished.emit({"spacy": False, "benepar": False, "nltk": False}) # Indicate failure


# --- Main Dialog Class ---

class ModelManagementDialog(QDialog):
    """Dialog window for managing ANPE models."""

    # Signal emitted when an action might require the main window to re-check things
    models_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ANPE Model Management")
        self.setMinimumWidth(500)
        # self.setWindowIcon(QIcon(":/icons/model_management.png")) # Optional: Add icon

        self.clean_thread = None
        self.clean_worker = None

        self.setup_ui()
        self.refresh_status() # Initial status check

    def setup_ui(self):
        """Create the UI elements for the dialog."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # --- Status Section ---\n        status_group = QGroupBox("Model Status")
        status_group = QGroupBox("Model Status")
        status_layout = QGridLayout(status_group)
        status_layout.setColumnStretch(1, 1) # Allow status label to expand

        self.status_labels = {} # Store labels for easy update
        models = {
            "spacy": "spaCy (en_core_web_md)",
            "benepar": "Benepar (benepar_en3)",
            "nltk_punkt": "NLTK (punkt)",
            "nltk_punkt_tab": "NLTK (punkt_tab)"
        }
        row = 0
        for key, name in models.items():
            label = QLabel(f"{name}:")
            status_label = QLabel("Checking...")
            #status_label.setFixedWidth(100) # Ensure consistent width initially
            status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.status_labels[key] = status_label
            status_layout.addWidget(label, row, 0)
            status_layout.addWidget(status_label, row, 1)
            row += 1

        layout.addWidget(status_group)

        # --- Progress/Action Status ---\n        self.progress_label = QLabel("")
        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.progress_label)

        # --- Actions Section ---\n        actions_group = QGroupBox("Actions")
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout(actions_group)

        # Use standard theme icons where available
        self.setup_button = QPushButton("Run Setup/Install")
        self.clean_button = QPushButton("Clean Models")
        self.refresh_button = QPushButton("Refresh Status")

        self.setup_button.clicked.connect(self.run_setup)
        self.clean_button.clicked.connect(self.run_clean)
        self.refresh_button.clicked.connect(self.refresh_status)

        actions_layout.addWidget(self.setup_button)
        actions_layout.addWidget(self.clean_button)
        actions_layout.addWidget(self.refresh_button)
        layout.addWidget(actions_group)

        # --- Close Button ---\n        button_layout = QHBoxLayout()
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept) # QDialog's accept closes the dialog
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

    @pyqtSlot()
    def refresh_status(self):
        """Check the status of all models and update labels."""
        self.progress_label.setText("Checking model status...")
        QApplication.processEvents() # Ensure label updates immediately

        try:
            # SpaCy
            spacy_present = check_spacy_model()
            self._update_status_label("spacy", spacy_present)

            # Benepar
            benepar_present = check_benepar_model()
            self._update_status_label("benepar", benepar_present)

            # NLTK
            # check_nltk_models(['punkt', 'punkt_tab']) checks both
            # We need individual checks for display
            try:
                # nltk.data.find raises LookupError if not found
                nltk.data.find('tokenizers/punkt')
                nltk_punkt_found = True
            except LookupError:
                nltk_punkt_found = False
            self._update_status_label("nltk_punkt", nltk_punkt_found)

            try:
                nltk.data.find('tokenizers/punkt_tab')
                nltk_punkt_tab_found = True
            except LookupError:
                nltk_punkt_tab_found = False
            self._update_status_label("nltk_punkt_tab", nltk_punkt_tab_found)

            self.progress_label.setText("Status updated.")
        except Exception as e:
             logging.error(f"Error refreshing model status: {e}", exc_info=True)
             self.progress_label.setText(f"Error checking status: {e}")
             # Set all to unknown/error state
             for key in self.status_labels:
                 self._update_status_label(key, None) # Use None for error state

        # Enable/disable buttons based on whether an action is running
        is_running = self.clean_thread and self.clean_thread.isRunning()
        self._set_buttons_enabled(not is_running)


    def _update_status_label(self, key: str, present: bool | None):
        """Helper to update a status label with icon and text."""
        if key not in self.status_labels:
            return
        label = self.status_labels[key]
        if present is True:
            # Using rich text for color styling
            text = "<b style='color:green;'>Present</b>"
        elif present is False:
            text = "<b style='color:red;'>Missing</b>"
        else: # None indicates error during check
            text = "<i style='color:orange;'>Error</i>"

        label.setText(text)

    def _set_buttons_enabled(self, enabled: bool):
        """Enable or disable action buttons."""
        self.setup_button.setEnabled(True) # Setup button always enabled
        self.clean_button.setEnabled(enabled)
        self.refresh_button.setEnabled(enabled)
        # Keep close button always enabled for simplicity
        # self.close_button.setEnabled(enabled)

    @pyqtSlot()
    def run_setup(self):
        """Launch the Setup Wizard."""
        parent_window = self.parent()
        if not parent_window or not hasattr(parent_window, 'on_setup_wizard_complete'):
            logging.error("Cannot launch Setup Wizard: Parent window or required slots not found.")
            QMessageBox.critical(self, "Error", "Could not launch Setup Wizard (internal error).")
            return
            
        logging.info("Launching Setup Wizard from Model Management Dialog...")
        
        # DON'T store the wizard as an instance variable of this dialog
        # since the dialog will be closed before the wizard finishes
        wizard = SetupWizard(parent_window) # Parent to main window
        
        # Connect directly to main window's slots
        # Don't connect through this dialog since it will be closed
        wizard.setup_complete.connect(parent_window.on_setup_wizard_complete)
        wizard.setup_cancelled.connect(parent_window.on_setup_wizard_cancelled)
        
        # Close this dialog before showing the wizard
        self.close()  # Use close() instead of accept() for a cleaner shutdown
        
        # Now that this dialog is closed, show the wizard
        wizard.show()

    @pyqtSlot()
    def run_clean(self):
        """Confirm and start the background cleaning process."""
        if self.clean_thread and self.clean_thread.isRunning():
            QMessageBox.warning(self, "In Progress", "Model cleanup is already running.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Cleanup",
            "This will attempt to remove all detected ANPE-related models \\n"
            "(spaCy, Benepar, NLTK punkt/punkt_tab) from your system.\\n\\n"
            "Models will need to be re-downloaded the next time they are needed.\\n\\n"
            "Are you sure you want to proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.No:
            return

        self._set_buttons_enabled(False)
        self.progress_label.setText("Starting cleanup...")

        self.clean_thread = QThread(self) # Parent thread to dialog
        self.clean_worker = CleanWorker()
        self.clean_worker.moveToThread(self.clean_thread)

        # Connect signals and slots
        self.clean_worker.finished.connect(self.on_clean_finished)
        self.clean_worker.progress.connect(self.progress_label.setText)
        self.clean_thread.started.connect(self.clean_worker.run)
        # Clean up thread and worker when finished
        self.clean_worker.finished.connect(self.clean_thread.quit)
        self.clean_worker.finished.connect(self.clean_worker.deleteLater)
        self.clean_thread.finished.connect(self.clean_thread.deleteLater)


        self.clean_thread.start()

    @pyqtSlot(dict)
    def on_clean_finished(self, results: dict):
        """Handle completion of the clean process."""
        self._set_buttons_enabled(True)
        self.refresh_status() # Update status labels

        all_succeeded = all(results.values())

        if all_succeeded:
            QMessageBox.information(self, "Cleanup Complete", "Model cleanup process finished.")
            self.models_changed.emit() # Signal that models might have changed
        else:
            failed_models = [model for model, success in results.items() if not success]
            QMessageBox.warning(self, "Cleanup Incomplete",
                                f"Model cleanup process finished, but some errors occurred "
                                f"(e.g., for {', '.join(failed_models)}). Please check logs.")
            self.models_changed.emit() # Still signal change as some might have been removed

        # References are cleared automatically by deleteLater connections
        self.clean_thread = None
        self.clean_worker = None

    def closeEvent(self, event):
        """Ensure threads are stopped if dialog is closed forcefully."""
        # Remove setup thread check
        needs_wait = False
        if self.clean_thread and self.clean_thread.isRunning():
            logging.warning("Clean thread still running during close, attempting to quit...")
            self.clean_thread.quit()
            needs_wait = True

        if needs_wait:
             # Give threads a moment to exit
             if self.clean_thread: self.clean_thread.wait(500)

        super().closeEvent(event)


# Example usage (for testing standalone)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Dummy functions for standalone testing if main imports fail
    try:
        # Try importing one to see if the real environment is available
        from anpe.utils.setup_models import check_spacy_model
    except ImportError:
        logging.info("Using dummy model utility functions for testing.")
        def check_spacy_model(*args, **kwargs): logging.debug("Dummy check_spacy_model"); import time; time.sleep(0.1); return True
        def check_benepar_model(*args, **kwargs): logging.debug("Dummy check_benepar_model"); import time; time.sleep(0.1); return False
        def setup_models(*args, **kwargs): logging.info("Running dummy setup_models..."); import time; time.sleep(3); return True
        def clean_all(*args, **kwargs): logging.info("Running dummy clean_all..."); import time; time.sleep(3); return {"spacy": True, "benepar": True, "nltk": True}
        # Need dummy NLTK for status check part
        class DummyNLTKData:
            def find(self, path):
                logging.debug(f"Dummy NLTK finding: {path}")
                if path == 'tokenizers/punkt': return True
                if path == 'tokenizers/punkt_tab': raise LookupError("Dummy lookup fail") # Simulate one missing
                raise LookupError()
        class DummyNLTK:
            data = DummyNLTKData()
        nltk = DummyNLTK() # Overwrite potentially failed import


    app = QApplication(sys.argv)
    dialog = ModelManagementDialog()
    dialog.show()
    sys.exit(app.exec()) 