from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton, QSizePolicy
from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QMouseEvent, QPalette, QColor

# Import theme colors
from anpe_gui.theme import PRIMARY_COLOR, TEXT_COLOR, HOVER_COLOR, PRESSED_COLOR

class CustomTitleBar(QWidget):
    """A custom title bar widget for frameless windows, styled for ANPE GUI."""

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

        self.setFixedHeight(35) # Standard height
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 5, 0) # Left, Top, Right, Bottom
        layout.setSpacing(10)

        # Title Label
        self.title_label = QLabel(self._title)
        self.title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        # Use theme text color
        self.title_label.setStyleSheet(f"color: {TEXT_COLOR}; font-weight: bold;")
        layout.addWidget(self.title_label)

        # Minimize Button
        self.minimize_button = QPushButton("_")
        self.minimize_button.setFixedSize(25, 25)
        # Use theme colors for hover/pressed states
        self.minimize_button.setStyleSheet(f"""
            QPushButton {{
                border: none;
                background-color: transparent;
                font-size: 14pt;
                font-weight: bold;
                color: #555;
            }}
            QPushButton:hover {{ background-color: #e0e0e0; }} /* Keep light grey hover */
            QPushButton:pressed {{ background-color: #cccccc; }}
        """)
        self.minimize_button.clicked.connect(self.minimize_requested.emit)
        layout.addWidget(self.minimize_button)

        # Close Button
        self.close_button = QPushButton("✕")
        self.close_button.setFixedSize(25, 25)
        # Use theme colors for hover/pressed states - distinct red for close
        self.close_button.setStyleSheet(f"""
            QPushButton {{
                border: none;
                background-color: transparent;
                font-size: 12pt;
                font-weight: bold;
                color: #555;
            }}
            QPushButton:hover {{ background-color: #f44336; color: white; }} /* Standard red hover */
            QPushButton:pressed {{ background-color: #d32f2f; color: white; }} /* Darker red pressed */
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
        # Keep light gray background for visual separation
        palette.setColor(QPalette.ColorRole.Window, QColor("#f0f0f0"))
        self.setPalette(palette)

    # --- Window Dragging Logic ---
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if press is on the bar itself, not on buttons
            if self.childAt(event.pos()) is None or self.childAt(event.pos()) == self.title_label:
                self._mouse_pressed = True
                self._mouse_press_pos = event.globalPosition().toPoint()
                self._window_pos = self.window().pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._mouse_pressed:
            delta = event.globalPosition().toPoint() - self._mouse_press_pos
            # Ensure the parent window exists before moving
            window = self.window()
            if window:
                window.move(self._window_pos + delta)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._mouse_pressed = False
        super().mouseReleaseEvent(event)

    # Optional: Handle window activation changes if needed in the future
    # def changeEvent(self, event):
    #     if event.type() == event.Type.ActivationChange:
    #         pass # Style changes could go here
    #     super().changeEvent(event)

# Example usage removed as this is part of the main app now 