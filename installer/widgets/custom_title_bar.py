from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPalette, QColor

class CustomTitleBar(QWidget):
    """A custom title bar widget for frameless windows."""

    # Signals for window control
    minimize_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._mouse_pressed = False
        self._mouse_press_pos = QPoint()
        self._window_pos = QPoint()

        self.setAutoFillBackground(True)
        self._update_background_color() # Set initial background

        self.setFixedHeight(35) # Adjust height as needed
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 5, 0) # Left, Top, Right, Bottom
        layout.setSpacing(10)

        # Title Label
        self.title_label = QLabel(self._title)
        self.title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.title_label.setStyleSheet("color: #333; font-weight: bold;") # Simple styling
        layout.addWidget(self.title_label)

        # Minimize Button
        self.minimize_button = QPushButton("_") # Simple text representation
        self.minimize_button.setFixedSize(25, 25)
        self.minimize_button.setStyleSheet("""
            QPushButton { 
                border: none; 
                background-color: transparent; 
                font-size: 14pt; 
                font-weight: bold; 
                color: #555; 
            }
            QPushButton:hover { background-color: #e0e0e0; }
            QPushButton:pressed { background-color: #cccccc; }
        """)
        self.minimize_button.clicked.connect(self.minimize_requested.emit)
        layout.addWidget(self.minimize_button)

        # Close Button
        self.close_button = QPushButton("✕") # Unicode multiplication sign
        self.close_button.setFixedSize(25, 25)
        self.close_button.setStyleSheet("""
            QPushButton { 
                border: none; 
                background-color: transparent; 
                font-size: 12pt; 
                font-weight: bold; 
                color: #555; 
            }
            QPushButton:hover { background-color: #f44336; color: white; }
            QPushButton:pressed { background-color: #d32f2f; color: white; }
        """)
        self.close_button.clicked.connect(self.close_requested.emit)
        layout.addWidget(self.close_button)

    def set_title(self, title: str):
        """Update the window title text."""
        self._title = title
        self.title_label.setText(title)

    def _update_background_color(self):
        """Sets the background color slightly off-white for visibility."""
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor("#f0f0f0")) # Light gray background
        self.setPalette(palette)

    # --- Window Dragging Logic --- 
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if press is on the bar itself, not on buttons
            if self.childAt(event.pos()) is None or self.childAt(event.pos()) == self.title_label:
                self._mouse_pressed = True
                self._mouse_press_pos = event.globalPosition().toPoint()
                # Get the parent window's position
                self._window_pos = self.window().pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._mouse_pressed:
            delta = event.globalPosition().toPoint() - self._mouse_press_pos
            self.window().move(self._window_pos + delta)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_pressed = False
        super().mouseReleaseEvent(event)

    def changeEvent(self, event):
        # Handle changes like window activation to potentially change title bar style
        if event.type() == event.Type.ActivationChange:
            # Example: Dim background when inactive (optional)
            # palette = self.palette()
            # color = QColor("#f0f0f0") if self.window().isActiveWindow() else QColor("#fafafa")
            # palette.setColor(QPalette.ColorRole.Window, color)
            # self.setPalette(palette)
            pass
        super().changeEvent(event)

# Example usage (for testing the title bar directly)
if __name__ == '__main__':
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
            self.resize(400, 200)

            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            main_layout = QVBoxLayout(central_widget)
            main_layout.setContentsMargins(0,0,0,0)
            main_layout.setSpacing(0)

            self.title_bar = CustomTitleBar("Test Frameless Window")
            main_layout.addWidget(self.title_bar)

            # Add some content below the title bar
            content_label = QLabel("Window Content Area")
            content_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            content_label.setStyleSheet("background-color: white; padding: 20px;")
            main_layout.addWidget(content_label, 1) # Make it stretch

            # Connect title bar signals
            self.title_bar.minimize_requested.connect(self.showMinimized)
            self.title_bar.close_requested.connect(self.close)

    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())
