# anpe_gui/widgets/status_bar.py
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QProgressBar, 
                             QLabel, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer
# Import necessary theme colors
from ..theme import PRIMARY_COLOR, SUCCESS_COLOR, BORDER_COLOR

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
        # Add vertical margins (left/right 5, top/bottom 5)
        self.layout.setContentsMargins(0, 0, 5, 12)
        
        # Status label for text updates
        self.status_label = QLabel("Ready")
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False) # Text inside bar often looks cluttered
        self.progress_bar.setFixedHeight(16)  # Make it compact
        
        # Use a fixed width for the progress bar
        self.progress_bar.setFixedWidth(200)
        
        # Add separator line
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.Shape.VLine)
        self.separator.setFrameShadow(QFrame.Shadow.Sunken)
        
        # Add widgets to layout
        self.layout.addWidget(self.status_label)
        self.layout.addStretch()
        self.layout.addWidget(self.separator)
        self.layout.addWidget(self.progress_bar)
        
        # Set fixed height for the entire status bar
        # self.setFixedHeight(24) # Let the layout manager handle height
        
        # Hide progress bar initially
        self.progress_bar.hide()
        self.separator.hide()
        
    @pyqtSlot(str)
    def showMessage(self, message, timeout=0, status_type='info'):
        """Update the status message and set its style property."""
        self.status_label.setText(message)
        # Set the 'status' property for MainWindow's stylesheet to apply
        self.status_label.setProperty('status', status_type)
        # Force style re-evaluation
        self.style().unpolish(self.status_label)
        self.style().polish(self.status_label)

        # Handle timeout if provided (optional)
        # if timeout > 0:
        #     QTimer.singleShot(timeout, lambda: self.clearMessage()) # Define clearMessage if needed

    @pyqtSlot(int, str)
    def update_progress(self, value, message=None):
        """Update the progress bar, its styling, and optionally the status message."""
        if not self.progress_bar.isVisible():
            self.progress_bar.show()
            self.separator.show()

        self.progress_bar.setRange(0, 100) # Ensure determinate mode
        self.progress_bar.setValue(value)

        # Update progress bar chunk color based on value
        chunk_color = SUCCESS_COLOR if value == 100 else PRIMARY_COLOR
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                text-align: center;
                background-color: #f5f5f5;
                height: 16px;
            }}
            QProgressBar::chunk {{
                background-color: {chunk_color};
                border-radius: 3px;
                margin: 1px;
            }}
        """)

        if message:
            # Set status message with 'busy' type while progress is active
            self.showMessage(message, status_type='busy')

    def start_progress(self, message="Processing..."):
        """Start an indeterminate progress operation."""
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        # Apply a base style for indeterminate (can add gradient later if needed)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                text-align: center;
                background-color: #f5f5f5;
                height: 16px;
            }}
            QProgressBar::chunk {{
                background-color: {PRIMARY_COLOR};
                border-radius: 3px;
                margin: 1px;
            }}
        """)
        self.progress_bar.show()
        self.separator.show()
        self.showMessage(message, status_type='busy') # Show as busy

    def stop_progress(self, message="Complete", status_type='success'):
        """Stop the progress bar and update status with a specific type."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100) # Ensure it shows 100%
        # Explicitly apply success style to progress bar chunk
        self.progress_bar.setStyleSheet(f"""
             QProgressBar {{
                 border: 1px solid {BORDER_COLOR};
                 border-radius: 4px;
                 text-align: center;
                 background-color: #f5f5f5;
                 height: 16px;
             }}
             QProgressBar::chunk {{
                 background-color: {SUCCESS_COLOR};
                 border-radius: 3px;
                 margin: 1px;
             }}
         """)
        self.showMessage(message, status_type=status_type) # Show final message with its type

        # Hide progress after a delay
        QTimer.singleShot(3000, self.clear_progress)

    def clear_progress(self):
        """Clear the progress bar and reset status to Ready."""
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        self.separator.hide()
        # Reset status label to default 'ready' state
        self.showMessage("ANPE Ready", status_type='ready') 