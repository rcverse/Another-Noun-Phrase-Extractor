# anpe_gui/widgets/status_bar.py
from PyQt6.QtWidgets import (QWidget, QHBoxLayout, QProgressBar, 
                             QLabel, QFrame, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer, QPropertyAnimation, QEasingCurve, QByteArray
from PyQt6.QtGui import QPalette, QColor
# Import necessary theme colors
from ..theme import PRIMARY_COLOR, SUCCESS_COLOR, BORDER_COLOR
# Import our new activity indicator
from .activity_indicator import PulsingActivityIndicator

class StatusBar(QWidget):
    """
    Global status bar with progress indicator and text status.
    Displays beneath the log panel and provides feedback for all operations.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self._progress_animation = None # Initialize animation attribute
        
    def setup_ui(self):
        # Main layout
        self.layout = QHBoxLayout(self)
        # Add vertical margins (left/right 5, top/bottom 5)
        self.layout.setContentsMargins(5, 3, 5, 3) # Adjusted margins (left, top, right, bottom)
        
        # Status label for text updates
        self.status_label = QLabel("Ready")
        
        # Progress indicator container (holds activity indicator and progress bar)
        self.progress_container = QWidget()
        self.progress_layout = QHBoxLayout(self.progress_container)
        self.progress_layout.setContentsMargins(0, 0, 0, 0)
        self.progress_layout.setSpacing(8)  # Space between activity indicator and progress bar
        
        # Create the pulsing activity indicator
        self.activity_indicator = PulsingActivityIndicator(self)
        
        # Progress bar (still used for determinate progress)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True) # Show percentage for better feedback
        self.progress_bar.setFixedHeight(22) # Consistent height
        self.progress_bar.setObjectName("AnimatedProgressBar")
        self.progress_bar.setFormat("Waiting for tasks")
        
        # Use a responsive width for the progress bar (20% of available width)
        self.progress_bar.setMinimumWidth(250)
        self.progress_bar.setSizePolicy(
            QSizePolicy.Policy.Minimum, 
            QSizePolicy.Policy.Fixed
        )
        
        # Add widgets to progress container
        self.progress_layout.addWidget(self.activity_indicator)
        self.progress_layout.addWidget(self.progress_bar)
        
        # Add separator line
        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.Shape.VLine)
        self.separator.setFrameShadow(QFrame.Shadow.Sunken)
        
        # Add widgets to main layout
        self.layout.addWidget(self.status_label)
        self.layout.addStretch()
        self.layout.addWidget(self.separator)
        self.layout.addWidget(self.progress_container)
        
        # Set default idle style for progress bar
        self.set_idle_state()
        
        # Activity indicator is visible but not animating during idle
        # (removed auto-start of animation)
    
    def set_idle_state(self):
        """Set progress bar to idle state with 'Waiting for tasks' text."""
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Waiting for tasks")
        
        # Stop activity indicator animation but keep it visible
        self.activity_indicator.stop()
        
        # Apply idle styling
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                text-align: center;
                color: #666666;
                font-weight: normal;
                background-color: #f5f5f5;
                height: 22px;
            }}
            QProgressBar::chunk {{
                background-color: #dddddd; /* Light gray for idle state */
                border-radius: 2px;
                margin: 0px;
            }}
        """)

    @pyqtSlot(str)
    def showMessage(self, message, timeout=0, status_type='info'):
        """Update the status message and set its style property."""
        self.status_label.setText(message)
        # Set the 'status' property for MainWindow's stylesheet to apply
        self.status_label.setProperty('status', status_type)
        # Force style re-evaluation
        self.style().unpolish(self.status_label)
        self.style().polish(self.status_label)

    @pyqtSlot(int, str)
    def update_progress(self, value, message=None):
        """Update the progress bar, its styling, and optionally the status message."""
        # Start activity indicator animation if not running
        if not self.activity_indicator.active:
            self.activity_indicator.start()
            
        # Set progress bar to show percentage
        self.progress_bar.setFormat("%p%")
        
        # Update progress bar - cap at 95% for visual consistency
        self.progress_bar.setRange(0, 100) # Ensure determinate mode
        
        # Stop any existing animation
        if self._progress_animation and self._progress_animation.state() == QPropertyAnimation.State.Running:
            self._progress_animation.stop()
            
        # Create and configure the animation
        self._progress_animation = QPropertyAnimation(self.progress_bar, b"value", self)
        self._progress_animation.setDuration(400)  # Animation duration in milliseconds
        self._progress_animation.setStartValue(self.progress_bar.value()) # Start from current value
        self._progress_animation.setEndValue(value) # Animate to the new value
        self._progress_animation.setEasingCurve(QEasingCurve.Type.InOutQuad) # Smooth easing
        self._progress_animation.finished.connect(self._clear_animation_reference) # Clear reference on finish
        self._progress_animation.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped) # Start animation

        # Change text visibility based on progress
        self.progress_bar.setTextVisible(True)
        
        # Apply style based on progress state - always use PRIMARY_COLOR
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                text-align: center;
                color: #333333;
                font-weight: bold;
                background-color: #f5f5f5;
                height: 22px;
            }}
            QProgressBar::chunk {{
                background-color: #8EACC0; /* Light Morandi blue */
                border-radius: 2px;
                margin: 0px;
            }}
        """)

        if message:
            # Set status message with 'busy' type while progress is active
            self.showMessage(message, status_type='busy')

    def start_progress(self, message="Processing..."):
        """Start an indeterminate progress operation using the activity indicator."""
        # Start activity indicator animation
        self.activity_indicator.start()
        
        # Hide the progress bar's value and set text
        self.progress_bar.setRange(0, 0)  # Indeterminate mode
        self.progress_bar.setFormat("Processing...")
        
        # Apply active processing style
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid {BORDER_COLOR};
                border-radius: 4px;
                text-align: center;
                color: #333333;
                font-weight: bold;
                background-color: #f5f5f5;
                height: 22px;
            }}
            QProgressBar::chunk {{
                background-color: #8EACC0; /* Light Morandi blue */
                border-radius: 2px;
                margin: 0px;
            }}
        """)
        
        # Update status message
        self.showMessage(message, status_type='busy')

    def stop_progress(self, message="Complete", status_type='success'):
        """Stop the progress/activity indicator and update status."""
        # If progress bar was in indeterminate mode, show 100%
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100) # Ensure it shows 100%
        self.progress_bar.setFormat("Complete")
        
        # Apply completion style - using PRIMARY_COLOR instead of SUCCESS_COLOR
        self.progress_bar.setStyleSheet(f"""
             QProgressBar {{
                 border: 1px solid {BORDER_COLOR};
                 border-radius: 4px;
                 text-align: center;
                 color: #333333;
                 font-weight: bold;
                 background-color: #f5f5f5;
                 height: 22px;
             }}
             QProgressBar::chunk {{
                 background-color: #8EACC0; /* Light Morandi blue */
                 border-radius: 2px;
                 margin: 0px;
             }}
         """)
        
        # Update status message
        self.showMessage(message, status_type=status_type)

        # Reset to idle state after a delay
        QTimer.singleShot(3000, self.set_idle_state)

    def clear_progress(self):
        """Reset indicators to idle state."""
        self.set_idle_state()
        
        # Reset status label to default 'ready' state
        self.showMessage("ANPE Ready", status_type='ready')

    @pyqtSlot()
    def _clear_animation_reference(self):
        """Slot to clear the animation reference when it finishes."""
        self._progress_animation = None 