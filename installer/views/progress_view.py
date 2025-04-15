from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QProgressBar, QPushButton, QTextEdit, QSizePolicy, QSpacerItem, QHBoxLayout, QFrame, QListWidget, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSlot, QSize
from PyQt6.QtGui import QPixmap, QFont, QIcon, QColor
import os
import re
from typing import Dict, Any, Optional

from ..widgets.task_list_widget import TaskListWidget, TaskStatus
from ..utils import get_resource_path
from ..styles import SECONDARY_BUTTON_STYLE, PROGRESS_BAR_STYLE, LOG_TEXT_AREA_STYLE, TITLE_LABEL_STYLE, STATUS_LABEL_STYLE, PRIMARY_BUTTON_STYLE

class ProgressViewWidget(QWidget):
    """Widget for displaying progress during setup stages."""

    def __init__(self, title: str, parent=None):
        """Initialize the progress view.
        """
        super().__init__(parent)
        self._title = title
        self._tasks = {}  # Initialize _tasks as an empty dictionary
        self._setup_ui()

    def _setup_ui(self):
        """Set up the UI elements."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 20, 30, 20)
        
        # --- Header with Logo and Title ---
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        logo_path = get_resource_path("assets/app_icon_logo.png")
        
        # Check if the logo file exists
        if os.path.exists(logo_path):
            logo_pixmap = QPixmap(logo_path)
            logo_label.setPixmap(logo_pixmap.scaled(70, 70, Qt.AspectRatioMode.KeepAspectRatio))
        else:
            # Create a text label as fallback
            logo_label.setText("ANPE")
            logo_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #0066b2;")
            
        header_layout.addWidget(logo_label)
        
        # Add spacing between logo and title
        header_layout.addSpacing(20)
        
        title_layout = QVBoxLayout()
        stage_label = QLabel(self._title)
        stage_label.setStyleSheet(TITLE_LABEL_STYLE)
        title_layout.addWidget(stage_label)
        
        # Add explanation text based on the title
        explanation_text = ""
        if "Environment" in self._title:
            explanation_text = "Setting up a dedicated Python environment with required dependencies for ANPE."
        elif "Language Models" in self._title:
            explanation_text = "Downloading and installing the language processing models needed for text analysis."
        
        if explanation_text:
            explanation_label = QLabel(explanation_text)
            explanation_label.setWordWrap(True)
            explanation_label.setStyleSheet("color: #666666;")
            title_layout.addWidget(explanation_label)
            
        header_layout.addLayout(title_layout)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # Increase spacing after header (was 10)
        main_layout.addSpacing(5)
        
        # --- Task List ---
        self._task_list = TaskListWidget()
        if self._tasks:
            for task_id, task_name in self._tasks.items():
                self._task_list.add_task(task_id, task_name)
        main_layout.addWidget(self._task_list)
        
        main_layout.addSpacing(10)
        
        # --- Progress Layout (Bar and Status) ---
        progress_container = QWidget()
        progress_layout = QVBoxLayout(progress_container)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        # Set up progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 0)  # Indeterminate initially
        self._progress_bar.setFixedHeight(4)
        self._progress_bar.setTextVisible(False)
        
        # Use centralized progress bar style
        self._progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        progress_layout.addWidget(self._progress_bar)
        
        # --- Status Label (moved below progress bar) ---
        self._status_label = QLabel("Preparing...")
        self._status_label.setStyleSheet(STATUS_LABEL_STYLE)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        progress_layout.addWidget(self._status_label)
        
        main_layout.addWidget(progress_container)
        
        # Create a fixed container for details and logs
        details_container = QWidget()
        details_layout = QVBoxLayout(details_container)
        details_layout.setContentsMargins(0, 0, 0, 0)
        
        # --- Details Toggle Button ---
        self._details_button = QPushButton("Show Details")
        self._details_button.setStyleSheet(PRIMARY_BUTTON_STYLE)
        self._details_button.setMaximumWidth(120)
        self._details_button.clicked.connect(self._toggle_details)
        
        details_button_layout = QHBoxLayout()
        details_button_layout.addWidget(self._details_button)
        details_button_layout.addStretch()
        details_layout.addLayout(details_button_layout)
        
        # --- Log Text Area ---
        self._log_area = QTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setFont(QFont("Consolas", 9))
        self._log_area.setStyleSheet(LOG_TEXT_AREA_STYLE)
        self._log_area.setMinimumHeight(120)
        self._log_area.setVisible(False)
        details_layout.addWidget(self._log_area)
        
        main_layout.addWidget(details_container)

    def _toggle_details(self, checked=None):
        """Show or hide the log area."""
        # Explicitly toggle visibility state
        current_state = self._log_area.isVisible()
        new_state = not current_state
        
        # Debug print to console
        print(f"Toggle details: current={current_state}, new={new_state}")
        
        # Force update the visibility
        self._log_area.setVisible(new_state)
        self._log_area.repaint()  # Force immediate repaint
        
        # Update button text
        self._details_button.setText("Hide Details" if new_state else "Show Details")

    def setup_tasks(self, tasks: dict):
        """Set up task list with provided tasks."""
        self._task_list.clear_tasks()
        for task_id, task_name in tasks.items():
            self._task_list.add_task(task_id, task_name)

    def setup_tasks_from_worker(self, worker):
        """Set up task list from a worker object that has a _tasks dictionary."""
        if hasattr(worker, '_tasks'):
            self.setup_tasks(worker._tasks)

    # --- Public Slots for Updates ---
    @pyqtSlot(str)
    def update_status(self, status: str):
        """Update the status label text."""
        self._status_label.setText(status)
        self._log_area.append(status)

    @pyqtSlot(str)
    def append_log(self, message: str):
        """Append a message to the log area and scroll to the bottom."""
        self._log_area.append(message.strip())
        self._log_area.verticalScrollBar().setValue(self._log_area.verticalScrollBar().maximum())

    @pyqtSlot()
    def clear_log(self):
        """Clear the log area."""
        self._log_area.clear()

    @pyqtSlot(int, int)
    def set_progress_range(self, min_val: int, max_val: int):
        """Set the range for the progress bar (use 0, 0 for indeterminate)."""
        self._progress_bar.setRange(min_val, max_val)
        self._progress_bar.setTextVisible(min_val != 0 or max_val != 0)

    @pyqtSlot(int)
    def set_progress_value(self, value: int):
        """Set the current value of the progress bar."""
        self._progress_bar.setValue(value)
        
    @pyqtSlot(str, int)
    def update_task_status(self, task_id: str, status: int):
        """Update the status of a task."""
        self._task_list.update_task_status(task_id, status)

    def handle_status_update(self, status: str):
        """Update the status label with the current status."""
        if not status:
            return
        
        # For model setup page, clean up the status message
        if "Setting up Language Models" in self._title:
            # Remove timestamp and log level if present
            status = self._clean_model_status(status)
        
        self._status_label.setText(status)
        self._log_area.append(status)
        
    def _clean_model_status(self, status: str) -> str:
        """Clean up model setup status messages to be more user-friendly."""
        import re
        
        # Remove timestamp, log level, and module name
        timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}'
        log_pattern = r'- (anpe\.setup_models)? - (INFO|DEBUG|WARNING|ERROR) - '
        
        status = re.sub(f'{timestamp_pattern} {log_pattern}', '', status)
        status = re.sub(f'{timestamp_pattern}', '', status)
        status = re.sub(f'{log_pattern}', '', status)
        
        # Clean up specific messages for better readability
        status = re.sub(r'^Downloading spaCy model: en_core_web_md$', 'Downloading English language model', status)
        status = re.sub(r'^en_core_web_md$', 'English model', status)
        
        return status.strip()
        
    def handle_log_update(self, log_line: str):
        """Update the log area with a new log line."""
        if self._log_area:
            self._log_area.append(log_line)
            # Auto-scroll to bottom
            scrollbar = self._log_area.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
