from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QCheckBox, QPushButton, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt, pyqtSignal

class CompletionViewWidget(QWidget):
    """Widget for the completion screen of the setup wizard (Windows version)."""
    # Signals emitted when the final button is clicked
    shortcut_requested = pyqtSignal(bool)
    launch_requested = pyqtSignal(bool)
    close_requested = pyqtSignal()

    def __init__(self, parent=None):
        """Initialize the completion view."""
        super().__init__(parent)
        self._setup_ui()
        # Set initial state (e.g., assuming success until told otherwise)
        # self.set_success_state(True)

    def _setup_ui(self):
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        # --- Status Title ---
        self.status_title = QLabel("Setup Status") # Placeholder text
        self.status_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_title.setStyleSheet("font-size: 18pt; font-weight: bold;")
        layout.addWidget(self.status_title)

        # --- Info Text ---
        self.info_text = QLabel("Details about the setup process outcome...") # Placeholder
        self.info_text.setWordWrap(True)
        self.info_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_text)

        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))

        # --- Options (Windows Specific, shown on success) ---
        self.shortcut_checkbox = QCheckBox("Create Desktop/Start Menu Shortcut")
        self.shortcut_checkbox.setChecked(True) # Default to checked
        layout.addWidget(self.shortcut_checkbox)

        self.launch_checkbox = QCheckBox("Launch ANPE now")
        self.launch_checkbox.setChecked(True) # Default to checked
        layout.addWidget(self.launch_checkbox)

        # --- Complete/Close Button ---
        self.complete_button = QPushButton("Complete") # Text changes based on state
        self.complete_button.setObjectName("CompleteButton")
        self.complete_button.setStyleSheet("QPushButton#CompleteButton { font-size: 14pt; padding: 10px; }")
        self.complete_button.clicked.connect(self._handle_complete)
        layout.addWidget(self.complete_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Hide options initially, shown by set_success_state
        self.shortcut_checkbox.setVisible(False)
        self.launch_checkbox.setVisible(False)

    def set_success_state(self, success: bool):
        """Configure the view based on the setup outcome."""
        if success:
            self.status_title.setText("Setup Complete!")
            self.status_title.setStyleSheet("font-size: 18pt; font-weight: bold; color: green;")
            self.info_text.setText("ANPE has been successfully set up and is ready to use.")
            self.shortcut_checkbox.setVisible(True)
            self.launch_checkbox.setVisible(True)
            self.complete_button.setText("Complete")
        else:
            self.status_title.setText("Setup Failed")
            self.status_title.setStyleSheet("font-size: 18pt; font-weight: bold; color: red;")
            self.info_text.setText(
                "An error occurred during setup. Please check the logs (if available) "
                "or try running the setup again. If the problem persists, please report the issue."
            )
            self.shortcut_checkbox.setVisible(False)
            self.launch_checkbox.setVisible(False)
            self.complete_button.setText("Close")

    def _handle_complete(self):
        """Emit signals based on checkbox states and request closing."""
        # Only emit requests if setup was successful (when checkboxes are visible)
        if self.shortcut_checkbox.isVisible():
            self.shortcut_requested.emit(self.shortcut_checkbox.isChecked())
        if self.launch_checkbox.isVisible():
            self.launch_requested.emit(self.launch_checkbox.isChecked())

        # Always emit close request
        self.close_requested.emit()

# Example usage (for testing the view directly)
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication, QPushButton, QHBoxLayout

    app = QApplication(sys.argv)
    container = QWidget()
    main_layout = QVBoxLayout(container)

    completion_view = CompletionViewWidget()
    main_layout.addWidget(completion_view)

    # Add buttons to test state changes
    button_layout = QHBoxLayout()
    success_button = QPushButton("Set Success")
    fail_button = QPushButton("Set Failure")
    button_layout.addWidget(success_button)
    button_layout.addWidget(fail_button)
    main_layout.addLayout(button_layout)

    success_button.clicked.connect(lambda: completion_view.set_success_state(True))
    fail_button.clicked.connect(lambda: completion_view.set_success_state(False))

    # Connect signals to print output
    completion_view.shortcut_requested.connect(lambda checked: print(f"Shortcut requested: {checked}"))
    completion_view.launch_requested.connect(lambda checked: print(f"Launch requested: {checked}"))
    completion_view.close_requested.connect(lambda: print("Close requested"))

    container.resize(500, 300)
    container.show()
    completion_view.set_success_state(True) # Start in success state for demo

    sys.exit(app.exec())
