"""
Splash screen for the ANPE GUI application.
"""

import os
import time
from PyQt6.QtWidgets import (QSplashScreen, QApplication, QProgressBar, 
                             QLabel, QVBoxLayout, QWidget)
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QLinearGradient, QBrush, QRegion, QPainterPath
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QCoreApplication, QPropertyAnimation, QEasingCurve, QPointF, pyqtProperty, QRect, QRectF


class SplashScreen(QSplashScreen):
    """
    Custom splash screen with banner and progress bar.
    """
    
    loading_finished = pyqtSignal()
    # Signal emitted when fade-out is complete and the splash can truly close
    fade_out_complete = pyqtSignal() 
    
    def __init__(self, banner_path=None):
        """
        Initialize the splash screen using the provided banner or fallback.
        """
        # --- Restore banner loading --- 
        actual_banner_path = self._find_banner(banner_path)
        
        # Load banner or create fallback
        if actual_banner_path:
            pixmap = QPixmap(actual_banner_path)
            # Resize if too large, maintaining aspect ratio
            max_width, max_height = 400, 200 # Define max dimensions
            if pixmap.width() > max_width or pixmap.height() > max_height:
                pixmap = pixmap.scaled(max_width, max_height, Qt.AspectRatioMode.KeepAspectRatio, 
                                       Qt.TransformationMode.SmoothTransformation)
        else:
            print("DEBUG: banner.png not found, using fallback banner.")
            pixmap = self._create_fallback_banner() # Use fallback if needed
        # --- End Restore banner loading --- 
        
        super().__init__(pixmap)
        self.pixmap_height = pixmap.height()
        self.pixmap_width = pixmap.width()
        
        # Window flags
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | 
                           Qt.WindowType.FramelessWindowHint)
        
        # --- Define border radius before using it ---
        self.border_radius = 10 # Define radius for consistency
        
        # Setup content widget below banner
        self.setup_content_area()
        
        # Set fixed size for the entire splash window
        content_height = self.content_widget.sizeHint().height()
        total_height = self.pixmap_height + content_height
        self.setFixedSize(self.pixmap_width, total_height)
        
        # --- Apply rounded corners mask ---
        mask_region = QRegion(self.rect(), QRegion.RegionType.Rectangle)
        # Create rectangle for mask, slightly smaller to avoid border artifacts if any
        mask_rect = self.rect().adjusted(1, 1, -1, -1) 
        mask_path = QPainterPath()
        mask_path.addRoundedRect(QRectF(mask_rect), self.border_radius, self.border_radius)
        self.setMask(QRegion(mask_path.toFillPolygon().toPolygon()))
        # --- End Apply rounded corners mask ---
        
        # Loading state
        self.loading_progress = 0
        self.status_label.setText("Initializing...")

        # Animation setup
        self._fade_animation = None
        self.setWindowOpacity(0.0) # Start fully transparent for fade-in

        # Center the splash screen
        self._center_on_screen()

    def _find_banner(self, provided_path):
        """Locate the banner image file, primarily in the resources directory."""
        # Option 1: Use provided path if valid
        if provided_path and os.path.exists(provided_path):
            print(f"DEBUG: Using provided banner path: {provided_path}")
            return provided_path
            
        # Option 2: Check the standard resources directory relative to the script
        script_dir = os.path.dirname(__file__)
        resource_path = os.path.join(script_dir, "resources", "banner.png")
        abs_resource_path = os.path.abspath(resource_path)

        if os.path.exists(abs_resource_path):
            print(f"DEBUG: Found banner at standard resource path: {abs_resource_path}")
            return abs_resource_path
            
        # Fallback if not found
        print(f"DEBUG: Banner image not found at {abs_resource_path}. Check packaging.")
        return None

    def _create_fallback_banner(self):
        """Create a flat, Morandi-colored fallback banner."""
        width, height = 400, 100 # Adjusted height for a potentially simpler look
        pixmap = QPixmap(width, height)
        
        # Flat background color (Light Grey/Off-white)
        background_color = QColor("#f5f5f5") 
        pixmap.fill(background_color)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # ANPE Title - Use Primary Color
        title_color = QColor("#005A9C") # PRIMARY_COLOR from theme.py
        painter.setPen(title_color)
        font = QFont("Segoe UI Variable", 20, QFont.Weight.Bold) # Slightly larger title
        if not font.exactMatch():
            font = QFont("Arial", 20, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(0, 0, width, int(height * 0.65), # Adjust vertical position
                         Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignBottom, 
                         "ANPE")
        
        # Subtitle - Use Standard Text Color
        subtitle_color = QColor("#212529") # TEXT_COLOR from theme.py
        painter.setPen(subtitle_color)
        subtitle_font = QFont("Segoe UI Variable", 9) # Slightly smaller subtitle
        if not subtitle_font.exactMatch():
            subtitle_font = QFont("Arial", 9)
        painter.setFont(subtitle_font)
        painter.drawText(0, int(height * 0.65), width, int(height * 0.3), # Adjust vertical position
                         Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignTop,
                         "Another Noun Phrase Extractor")
        
        painter.end()
        return pixmap

    def setup_content_area(self):
        """Set up the widget holding progress bar and status text with improved styles."""
        self.content_widget = QWidget(self)
        # Set background to white and add rounded *bottom* corners matching the mask
        self.content_widget.setStyleSheet(f"""
            background-color: white; 
            border-bottom-left-radius: {self.border_radius}px; 
            border-bottom-right-radius: {self.border_radius}px;
            border-top-left-radius: 0px; /* Ensure top corners are square */
            border-top-right-radius: 0px;
        """) 
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(25, 15, 25, 20) 
        self.content_layout.setSpacing(10)

        # Status label
        self.status_label = QLabel("Loading...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Use a slightly larger, darker font
        self.status_label.setStyleSheet("color: #444; font-size: 10pt;")

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(10) # Slightly thinner
        # Updated stylesheet: white background, keep gradient chunk, adjust radius
        progress_bar_radius = max(0, self.border_radius - 5) # Slightly smaller radius for inset look
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #cccccc;
                border-radius: {progress_bar_radius}px; 
                background-color: white; /* Changed background to white */
                height: 10px; 
            }}
            QProgressBar::chunk {{
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3498db, stop:1 #5dade2); 
                border-radius: {max(0, progress_bar_radius - 1)}px; /* Inner radius */
                margin: 0.5px; 
            }}
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
        """Fade the splash screen in."""
        self.setWindowOpacity(0.0) # Ensure starting point
        self.show() # Make it visible before animating opacity
        self._fade(0.0, 1.0, duration)

    def fade_out(self, duration=300):
        """Fade the splash screen out."""
        # Ensure we start from full opacity for the fade-out effect
        current_opacity = self.windowOpacity()
        self._fade(current_opacity, 0.0, duration, self._on_fade_out_complete)

    def _on_fade_out_complete(self):
        """Called when the fade-out animation finishes."""
        self.fade_out_complete.emit() # Emit signal *before* closing
        self.close() # Actually close the window now
        self._fade_animation = None # Clean up animation object

    def start_loading_animation(self, app):
        """
        Simulate loading process, show with fade-in, and fade out when done.
        """
        self.fade_in() # Show with fade-in instead of self.show()
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
                # Loading finished - emit signal first
                self.loading_finished.emit()
                # Don't close immediately, start fade-out instead
                self.fade_out() 
        
        # Start the first step
        QTimer.singleShot(200 + 300, lambda: update_step(0)) # Add fade-in duration delay

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