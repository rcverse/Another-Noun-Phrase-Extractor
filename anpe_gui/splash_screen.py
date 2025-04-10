"""
Splash screen for the ANPE GUI application.
"""

import os
import time
from PyQt6.QtWidgets import (QSplashScreen, QApplication, QProgressBar, 
                             QLabel, QVBoxLayout, QWidget)
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QLinearGradient, QBrush, QRegion, QPainterPath, QPen
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QCoreApplication, QPropertyAnimation, QEasingCurve, QPointF, pyqtProperty, QRect, QRectF
from anpe_gui.theme import PRIMARY_COLOR  # Import the primary color
from anpe_gui.version import __version__ as gui_version  # Import GUI version directly from version.py
try:
    from anpe import __version__ as core_version  # Import core version
except ImportError:
    core_version = "N/A"  # Fallback if core version can't be imported


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
        # Create a custom banner instead of loading from file
        pixmap = self._create_custom_banner()
        
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

    def _create_custom_banner(self):
        """Create a custom banner with logo on left and text on right."""
        width, height = 560, 200  # Wider banner for better spacing
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)  # Start with transparent background
        
        # Create high-quality painter with anti-aliasing
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        
        # Elegant background with subtle gradient
        gradient = QLinearGradient(0, 0, width, height)
        gradient.setColorAt(0, QColor("#f9f9f9"))  # Light color start
        gradient.setColorAt(1, QColor("#f0f0f0"))  # Slightly darker end
        painter.fillRect(0, 0, width, height, QBrush(gradient))
        
        # Clear separation - left side for logo, right side for text
        logo_section_width = 220  # Dedicated width for logo
        text_section_width = width - logo_section_width
        
        # Left side: Logo only
        icon_path = os.path.join(os.path.dirname(__file__), "resources", "app_icon.png")
        
        if os.path.exists(icon_path):
            # Load the PNG with higher resolution for better quality
            logo = QPixmap(icon_path)
            
            # Get device pixel ratio for high DPI displays
            device_pixel_ratio = QApplication.primaryScreen().devicePixelRatio()
            
            # Scale logo to appropriate size while maintaining aspect ratio
            # Multiply by device pixel ratio to ensure crisp display on high DPI screens
            logo_size = 160
            target_size = int(logo_size * device_pixel_ratio)
            
            # Only scale down, not up to avoid pixelation
            if logo.width() > target_size or logo.height() > target_size:
                logo = logo.scaled(target_size, target_size, 
                                Qt.AspectRatioMode.KeepAspectRatio, 
                                Qt.TransformationMode.SmoothTransformation)
            
            # Set the device pixel ratio to ensure proper rendering
            # This is a crucial step for high DPI displays
            if device_pixel_ratio > 1.0:
                logo.setDevicePixelRatio(device_pixel_ratio)
                
            # Calculate display size (adjusted for device pixel ratio)
            display_size = logo_size
            
            # Center the logo in the left section
            logo_x = (logo_section_width - display_size) // 2
            logo_y = (height - display_size) // 2
            
            # Draw the pixmap at the calculated position
            painter.drawPixmap(logo_x, logo_y, display_size, display_size, logo)
        else:
            # If logo not found, draw a placeholder
            painter.setPen(Qt.PenStyle.NoPen)
            # Use PRIMARY_COLOR from theme
            primary_color = QColor(PRIMARY_COLOR)
            painter.setBrush(primary_color)
            logo_size = 160
            logo_x = (logo_section_width - logo_size) // 2
            logo_y = (height - logo_size) // 2
            painter.drawRoundedRect(logo_x, logo_y, logo_size, logo_size, 10, 10)
        
        # Add a subtle separator line between logo and text
        separator_x = logo_section_width
        painter.setPen(QPen(QColor(220, 220, 220), 1))  # Lighter color for separator
        painter.drawLine(separator_x, 40, separator_x, height - 40)
        
        # Right side: Text content - nothing else
        text_start_x = logo_section_width + 30  # Start text with padding after separator
        
        # Draw main title "ANPE" using PRIMARY_COLOR
        primary_color = QColor(PRIMARY_COLOR)
        title_gradient = QLinearGradient(text_start_x, 0, text_start_x + 200, 0)
        title_gradient.setColorAt(0, primary_color.lighter(110))  # Slightly lighter version
        title_gradient.setColorAt(1, primary_color)  # Primary color
        
        # Use system fonts with better fallbacks for sharper text
        title_font = QFont()
        title_font.setFamily("Segoe UI")
        title_font.setPointSize(48)
        title_font.setWeight(QFont.Weight.Bold)
        if not QFont(title_font).exactMatch():
            title_font.setFamily("Arial")
            if not QFont(title_font).exactMatch():
                title_font.setFamily(title_font.defaultFamily())
            
        painter.setFont(title_font)
        # Use QPen with width 1.5 for slightly bolder, crisper text
        painter.setPen(QPen(QBrush(title_gradient), 1.5))
        
        # Draw title in its own region with proper vertical centering
        title_rect = QRect(text_start_x, 40, text_section_width - 40, 80)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom, "ANPE")
        
        # Draw subtitle with proper spacing below the title - LIGHTER COLOR
        subtitle_font = QFont()
        subtitle_font.setFamily("Segoe UI")
        subtitle_font.setPointSize(14)
        subtitle_font.setWeight(QFont.Weight.Normal)
        if not QFont(subtitle_font).exactMatch():
            subtitle_font.setFamily("Arial")
            if not QFont(subtitle_font).exactMatch():
                subtitle_font.setFamily(subtitle_font.defaultFamily())
                
        painter.setFont(subtitle_font)
        painter.setPen(QColor("#555555"))  # Darker gray for better contrast and readability
        
        # Subtitle in its own region below the title
        subtitle_rect = QRect(text_start_x, 120, text_section_width - 40, 30)
        painter.drawText(subtitle_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, 
                        "Another Noun Phrase Extractor")
        
        # Add version info with improved rendering
        version_font = QFont()
        version_font.setFamily("Segoe UI")
        version_font.setPointSize(11)  # Slightly larger for better readability
        version_font.setWeight(QFont.Weight.Medium)  # Medium weight instead of regular
        
        if not QFont(version_font).exactMatch():
            version_font.setFamily("Arial")
            if not QFont(version_font).exactMatch():
                version_font.setFamily(version_font.defaultFamily())
                
        painter.setFont(version_font)
        painter.setPen(QColor("#444444"))  # Darker for better contrast
        
        version_rect = QRect(text_start_x, 155, text_section_width - 40, 20)
        version_text = f"GUI v{gui_version} | Core v{core_version}"
        painter.drawText(version_rect, Qt.AlignmentFlag.AlignLeft, version_text)
        
        # Add author credit with improved rendering
        credit_font = QFont()
        credit_font.setFamily("Segoe UI")
        credit_font.setPointSize(10)
        credit_font.setWeight(QFont.Weight.Normal)
        
        if not QFont(credit_font).exactMatch():
            credit_font.setFamily("Arial")
            if not QFont(credit_font).exactMatch():
                credit_font.setFamily(credit_font.defaultFamily())
                
        painter.setFont(credit_font)
        painter.setPen(QColor("#666666"))
        
        credit_rect = QRect(text_start_x, 175, text_section_width - 40, 20)
        painter.drawText(credit_rect, Qt.AlignmentFlag.AlignLeft, "@rcverse")
        
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