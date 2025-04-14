from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton, QTextEdit, QSizePolicy, QSpacerItem
)
from PyQt6.QtCore import Qt, pyqtSlot

class ProgressViewWidget(QWidget):
    """Widget for displaying progress during setup stages."""

    def __init__(self, title: str, parent=None):
        """Initialize the progress view.

        Args:
            title: The title to display for this progress stage (e.g., "Setting up Environment").
            parent: The parent widget.
        """
        super().__init__(parent)
        self._title = title
        self._setup_ui()

    def _setup_ui(self):
        """Set up the user interface elements."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 20, 30, 20)
        layout.setSpacing(15)

        # --- Title ---
        self.title_label = QLabel(self._title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(self.title_label)

        layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # --- Status Label ---
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        # --- Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0) # Indeterminate initially
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        layout.addSpacerItem(QSpacerItem(20, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))

        # --- Log Area (Initially Hidden) ---
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFontFamily("Courier New") # Monospaced font for logs
        self.log_area.setVisible(False) # Hidden by default
        self.log_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.log_area)

        # --- Details Button ---
        self.details_button = QPushButton("Show Details")
        self.details_button.setCheckable(True)
        self.details_button.toggled.connect(self._toggle_log_area)
        layout.addWidget(self.details_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Add stretch at the bottom if log area is hidden
        layout.addStretch(1)

    def _toggle_log_area(self, checked: bool):
        """Show or hide the log area."""
        self.log_area.setVisible(checked)
        self.details_button.setText("Hide Details" if checked else "Show Details")
        # Adjust layout - might need further refinement for smooth transitions
        self.layout().activate()

    # --- Public Slots for Updates ---
    @pyqtSlot(str)
    def update_status(self, status: str):
        """Update the status label text."""
        self.status_label.setText(status)

    @pyqtSlot(str)
    def append_log(self, message: str):
        """Append a message to the log area and scroll to the bottom."""
        self.log_area.append(message.strip())
        self.log_area.verticalScrollBar().setValue(self.log_area.verticalScrollBar().maximum())

    @pyqtSlot()
    def clear_log(self):
        """Clear the log area."""
        self.log_area.clear()

    @pyqtSlot(int, int)
    def set_progress_range(self, min_val: int, max_val: int):
        """Set the range for the progress bar (use 0, 0 for indeterminate)."""
        self.progress_bar.setRange(min_val, max_val)
        self.progress_bar.setTextVisible(min_val != 0 or max_val != 0)

    @pyqtSlot(int)
    def set_progress_value(self, value: int):
        """Set the current value of the progress bar."""
        self.progress_bar.setValue(value)

# Example usage (for testing the view directly)
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)
    progress_view = ProgressViewWidget("Testing Progress")
    progress_view.resize(500, 300) # Size for standalone testing
    progress_view.show()

    # Simulate updates
    timer = QTimer()
    # Use a list to hold the counter, allowing modification within the inner function
    counter_holder = [0]
    def simulate_updates():
        # nonlocal log_counter # No longer needed
        counter_holder[0] += 1
        log_counter = counter_holder[0] # Get current value for use in f-strings
        progress_view.update_status(f"Processing step {log_counter}...")
        progress_view.append_log(f"Log message {log_counter}\nAnother line for message {log_counter}")
        if log_counter == 1:
            progress_view.set_progress_range(0, 10)
        if log_counter <= 10:
            progress_view.set_progress_value(log_counter)
        if log_counter > 10:
            timer.stop()
            progress_view.update_status("Processing Complete!")
            progress_view.set_progress_value(10)

    timer.timeout.connect(simulate_updates)
    timer.start(500) # Update every 500ms

    sys.exit(app.exec())
