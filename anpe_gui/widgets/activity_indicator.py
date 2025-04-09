from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtGui import QPainter, QColor, QBrush, QPen, QPainterPath
import math

class PulsingActivityIndicator(QWidget):
    """
    A pulsing and rotating activity indicator showing processing activity
    without implying specific progress percentage.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Configure widget appearance
        self.setMinimumSize(24, 24)
        self.setMaximumHeight(24)
        
        # Animation properties
        self.pulse_size = 0
        self.rotation_angle = 0
        self.growing = True
        self.active = False
        self.color = QColor(52, 152, 219)  # Default blue color
        
        # Set up timer for animation
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.setInterval(50)  # Update every 50ms (smoother animation)
    
    def start(self):
        """Start the animation."""
        self.active = True
        self.timer.start()
    
    def stop(self):
        """Stop the animation but keep the indicator visible with a static appearance."""
        self.active = False
        self.timer.stop()
        # Reset angle but keep pulse size at a visible value
        self.rotation_angle = 0
        self.pulse_size = 5  # Keep a medium size for visibility when static
        self.update()  # Make sure the static indicator is drawn
    
    def set_color(self, color):
        """Set the color of the indicator."""
        self.color = QColor(color)
        self.update()
    
    def update_animation(self):
        """Update animation parameters and trigger a repaint."""
        if self.active:
            # Update pulse size
            if self.growing:
                self.pulse_size += 1
                if self.pulse_size >= 10:
                    self.growing = False
            else:
                self.pulse_size -= 1
                if self.pulse_size <= 0:
                    self.growing = True
            
            # Update rotation angle
            self.rotation_angle = (self.rotation_angle + 10) % 360
            
            # Trigger repaint
            self.update()
    
    def paintEvent(self, event):
        """Paint the animated indicator."""
        if not self.active and self.pulse_size == 0:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        center = QPoint(width // 2, height // 2)
        
        # Save painter state before rotation
        painter.save()
        painter.translate(center)
        painter.rotate(self.rotation_angle)
        painter.translate(-center.x(), -center.y())
        
        # Calculate size based on pulse state
        min_radius = min(width, height) * 0.2
        max_radius = min(width, height) * 0.4
        
        # Calculate current size
        size_ratio = self.pulse_size / 10
        radius = min_radius + (max_radius - min_radius) * size_ratio
        
        # Draw the main circle
        color = QColor(self.color)
        color.setAlpha(max(80, 255 - (self.pulse_size * 10)))
        
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        
        # For QPoint center, we need to use the rectangle form instead of center/radius
        int_radius = int(radius)
        painter.drawEllipse(
            center.x() - int_radius,
            center.y() - int_radius,
            int_radius * 2,
            int_radius * 2
        )
        
        # Draw 3 smaller circles around the edge for a more distinctive spinner effect
        small_radius = int(radius * 0.25)
        distance = radius * 0.7
        
        # Draw smaller circles with different opacities
        for i in range(3):
            angle = i * 120  # 120 degrees apart (3 dots)
            rad_angle = angle * (3.14159 / 180)  # Convert to radians
            
            # Position the smaller circle
            x = center.x() + int(distance * math.cos(rad_angle))
            y = center.y() + int(distance * math.sin(rad_angle))
            
            # Vary opacity based on position
            opacity = 150 + i * 35
            dot_color = QColor(self.color)
            dot_color.setAlpha(opacity)
            
            painter.setBrush(QBrush(dot_color))
            painter.drawEllipse(
                x - small_radius,
                y - small_radius,
                small_radius * 2,
                small_radius * 2
            )
        
        # Restore painter state
        painter.restore() 