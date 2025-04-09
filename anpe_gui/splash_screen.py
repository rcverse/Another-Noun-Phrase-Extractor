"""
Splash screen for the ANPE GUI application.
"""

import os
import time
from PyQt6.QtWidgets import (QSplashScreen, QApplication, QProgressBar, 
                             QLabel, QVBoxLayout, QWidget)
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QCoreApplication


class SplashScreen(QSplashScreen):
    """
    Custom splash screen with banner and progress bar.
    """
    
    loading_finished = pyqtSignal()
    
    def __init__(self, banner_path=None):
        """
        Initialize the splash screen.
        """
        # Find the banner image
        actual_banner_path = self._find_banner(banner_path)
        
        # Load or create fallback banner
        if actual_banner_path:
            pixmap = QPixmap(actual_banner_path)
            # Resize if too large, maintaining aspect ratio
            if pixmap.width() > 400 or pixmap.height() > 200:
                pixmap = pixmap.scaled(400, 200, Qt.AspectRatioMode.KeepAspectRatio, 
                                       Qt.TransformationMode.SmoothTransformation)
        else:
            pixmap = self._create_fallback_banner()
        
        super().__init__(pixmap)
        self.pixmap_height = pixmap.height()
        self.pixmap_width = pixmap.width()
        
        # Window flags
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | 
                           Qt.WindowType.FramelessWindowHint)
        
        # Setup content widget below banner
        self.setup_content_area()
        
        # Set fixed size for the entire splash window
        content_height = self.content_widget.sizeHint().height()
        total_height = self.pixmap_height + content_height
        self.setFixedSize(self.pixmap_width, total_height)
        
        # Loading state
        self.loading_progress = 0
        self.status_label.setText("Initializing...")

        # Center the splash screen
        self._center_on_screen()

    def _find_banner(self, provided_path):
        """Locate the banner image file."""
        if provided_path and os.path.exists(provided_path):
            return provided_path
            
        possible_paths = [
            "pics/banner.png",
            "../pics/banner.png",
            os.path.join(os.path.dirname(__file__), "../pics/banner.png"),
            "c:/Users/b162274/Desktop/ANPE_public/pics/banner.png"
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None

    def _create_fallback_banner(self):
        """Create a fallback banner if the image is not found."""
        pixmap = QPixmap(400, 100) # Smaller fallback size
        pixmap.fill(QColor("#1a5276")) # Blue background
        painter = QPainter(pixmap)
        painter.setPen(QColor("white"))
        font = QFont("Arial", 16, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, 
                         "ANPE\nAnother Noun Phrase Extractor")
        painter.end()
        return pixmap

    def setup_content_area(self):
        """Set up the widget holding progress bar and status text."""
        self.content_widget = QWidget(self)
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(20, 10, 20, 15) # Margins for content area
        self.content_layout.setSpacing(8)

        # Status label
        self.status_label = QLabel("Loading...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #333;")

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False) # No text inside bar
        self.progress_bar.setFixedHeight(12) # Slimmer progress bar
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #bbb;
                border-radius: 3px;
                background: #f0f0f0;
            }
            QProgressBar::chunk {
                background-color: #3498db; /* Blue accent */
                border-radius: 2px;
            }
        """)
        
        self.content_layout.addWidget(self.status_label)
        self.content_layout.addWidget(self.progress_bar)

        # Position the content widget below the pixmap
        self.content_widget.setGeometry(0, self.pixmap_height, 
                                      self.pixmap_width, self.content_widget.sizeHint().height())

    def _center_on_screen(self):
        """Center the splash screen on the primary display."""
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.center() - self.rect().center())

    # Override drawContents to prevent default drawing over our content widget
    def drawContents(self, painter):
        pass # We handle drawing via the content widget

    def showMessage(self, message, alignment=Qt.AlignmentFlag.AlignLeft, color=QColor("black")):
        """Display a message (overrides QSplashScreen method)."""
        # Use our custom label instead
        self.status_label.setText(message)
        # self.repaint() # Not needed as label updates automatically

    def set_progress(self, value, status_message=None):
        """Update the progress bar and status message."""
        self.loading_progress = value
        
        # Add a smooth transition effect for better visual experience
        current = self.progress_bar.value()
        
        # Create smooth step animation if needed
        if abs(current - value) > 10:
            # Animate in 5 steps
            steps = 5
            step_size = (value - current) / steps
            
            def update_step(step=1):
                if step <= steps:
                    intermediate_value = int(current + (step_size * step))
                    self.progress_bar.setValue(intermediate_value)
                    # Continue animation with next step
                    QTimer.singleShot(30, lambda: update_step(step + 1))
                else:
                    # Final step - set exact value
                    self.progress_bar.setValue(value)
            
            # Start animation
            update_step()
        else:
            # Small change, update directly
            self.progress_bar.setValue(value)
        
        if status_message:
            self.status_label.setText(status_message)
            
        QCoreApplication.processEvents() # Ensure UI updates immediately

    def start_loading_animation(self, app):
        """
        Simulate loading process and emit signal when done.
        (Callback is removed, uses signal instead)
        """
        self.show()
        app.processEvents()
        
        loading_steps = [
            (20, "Loading UI..."),
            (40, "Initializing core components..."),
            (60, "Setting up ANPE extractor..."),
            (80, "Finalizing setup..."),
            (100, "Starting application...")
        ]

        def update_step(step_index=0):
            if step_index < len(loading_steps):
                progress, message = loading_steps[step_index]
                self.set_progress(progress, message)
                # Simulate work for this step
                QTimer.singleShot(400, lambda: update_step(step_index + 1))
            else:
                # Loading finished
                self.loading_finished.emit()
                self.close()
        
        # Start the first step
        QTimer.singleShot(200, lambda: update_step(0))
        
        # We don't return self anymore as it's not needed for chaining the old way
        # return self

# Example of how to use it in app.py (if needed)
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     splash = SplashScreen()
#     
#     def create_main():
#         # Replace with actual MainWindow creation
#         main_win = QLabel("Main Window Placeholder") 
#         main_win.resize(800, 600)
#         main_win.show()
#         return main_win
#         
#     splash.start_loading_animation(app, create_main)
#     sys.exit(app.exec()) 