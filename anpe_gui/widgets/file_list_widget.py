"""
File list widget for selecting and managing files for processing.
"""

import os
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QFileDialog, QLabel, QMessageBox, QApplication
)
from PyQt6.QtCore import pyqtSignal, Qt

# Import theme colors
from anpe_gui.theme import PRIMARY_COLOR, LIGHT_HOVER_BLUE, TEXT_COLOR, BORDER_COLOR


class FileListWidget(QWidget):
    """
    Widget for selecting and managing multiple files for processing.
    Supports adding individual files, directories of files, and removing files.
    NOTE: Text input functionality has been moved to MainWindow.
    """
    
    # Signal emitted when the file list changes
    filesChanged = pyqtSignal(list)  # List of file paths
    
    def __init__(self, parent=None):
        """
        Initialize the file list widget.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        # File paths
        self.file_paths = []
        # self.is_file_mode_flag = True # No longer needed
        
        # Set up the UI
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the UI components. Focus only on file management."""
        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0) 
        self.layout.setSpacing(5) # Consistent spacing
        
        # Remove Input Mode Selection Buttons
        # self.input_mode_layout = QHBoxLayout() ... 
        
        # --- File List ---
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.layout.addWidget(self.file_list, 1) # Give list stretch factor

        # Apply custom styling for list items
        self.file_list.setStyleSheet(f"""
            QListWidget {{
                border: 1px solid {BORDER_COLOR}; /* Match theme border */
                background-color: white;
                outline: 0; /* Remove focus outline */
            }}
            QListWidget::item {{
                padding: 4px 6px; /* Adjust padding as needed */
                color: {TEXT_COLOR};
                background-color: white;
                outline: 0;
            }}
            QListWidget::item:selected {{
                background-color: {PRIMARY_COLOR};
                color: white;
                border: none; /* Remove selection border if any */
                outline: 0;
            }}
            QListWidget::item:hover:!selected {{
                background-color: {LIGHT_HOVER_BLUE};
                color: {TEXT_COLOR};
                outline: 0;
            }}
            /* Optional: Style for selected item when widget doesn't have focus */
            /* QListWidget::item:selected:!active {{
                background-color: #DDDDDD; 
                color: {TEXT_COLOR};
            }} */
        """)

        # --- File Action Buttons ---
        self.file_button_layout = QHBoxLayout() 
        self.add_files_button = QPushButton("Add Files")
        self.add_dir_button = QPushButton("Add Dir")
        self.remove_button = QPushButton("Remove Selected") # Be more specific
        self.clear_files_button = QPushButton("Clear All Files")
        self.file_button_layout.addWidget(self.add_files_button)
        self.file_button_layout.addWidget(self.add_dir_button)
        self.file_button_layout.addStretch()
        self.file_button_layout.addWidget(self.remove_button)
        self.file_button_layout.addWidget(self.clear_files_button)
        self.layout.addLayout(self.file_button_layout)
        
        # --- Status Label ---
        self.status_label = QLabel("No files selected")
        self.layout.addWidget(self.status_label)

        # Remove Text Input Widgets Container
        # self.text_widgets_container = QWidget() ...
        
        # --- Connect Signals --- 
        # Remove input_group connection
        self.add_files_button.clicked.connect(self.add_files)
        self.add_dir_button.clicked.connect(self.add_directory)
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_files_button.clicked.connect(self.clear_files)
        self.file_list.itemSelectionChanged.connect(self.update_status)
        # Remove text button connections

        # --- Initial State ---
        self.update_status() # Set initial status label

    def add_files(self):
        """Add files to the list using a file dialog."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Files to Process", "", "Text Files (*.txt);;All Files (*)"
        )
        
        if file_paths:
            added_count = 0
            for path in file_paths:
                if path not in self.file_paths:
                    self.file_paths.append(path)
                    item = QListWidgetItem(os.path.basename(path))
                    item.setToolTip(path)  # Show full path on hover
                    self.file_list.addItem(item)
                    added_count += 1
            
            if added_count > 0:
                self.update_status()
                self.filesChanged.emit(self.file_paths)
    
    def add_directory(self):
        """Add all text files from a directory."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory with Text Files"
        )
        
        if dir_path:
            added_count = 0
            text_files = []
            try: # Add error handling for os.walk
                for root, _, files in os.walk(dir_path):
                    for file in files:
                        if file.lower().endswith('.txt'): # Case-insensitive
                            full_path = os.path.join(root, file)
                            if full_path not in self.file_paths:
                                text_files.append(full_path)
            except OSError as e:
                 QMessageBox.warning(self, "Error Reading Directory", f"Could not read directory contents:\n{e}")
                 return

            if not text_files:
                 QMessageBox.information(
                     self, "No New Text Files", 
                     f"No new text files (.txt) found in the selected directory or its subdirectories."
                 )
                 return

            for path in text_files:
                 self.file_paths.append(path)
                 item = QListWidgetItem(os.path.basename(path))
                 item.setToolTip(path)
                 self.file_list.addItem(item)
                 added_count += 1

            if added_count > 0:
                self.update_status()
                self.filesChanged.emit(self.file_paths)
    
    def remove_selected(self):
        """Remove selected files from the list."""
        selected_items = self.file_list.selectedItems()
        if not selected_items: return

        removed_count = 0
        # Iterate backwards when removing items by index/row
        rows_to_remove = sorted([self.file_list.row(item) for item in selected_items], reverse=True)

        for row in rows_to_remove:
             item = self.file_list.takeItem(row)
             path_to_remove = item.toolTip() # Get full path from tooltip
             if path_to_remove in self.file_paths:
                 self.file_paths.remove(path_to_remove)
                 removed_count += 1
             del item # Explicitly delete item

        if removed_count > 0:
            self.update_status()
            self.filesChanged.emit(self.file_paths)

    def clear_files(self):
        """Clear all files from the list."""
        if not self.file_paths: return # Don't do anything if already empty
        self.file_list.clear()
        self.file_paths = []
        self.update_status()
        self.filesChanged.emit(self.file_paths)
    
    def update_status(self):
        """Update the status label with file count."""
        count = len(self.file_paths)
        
        if count == 0:
            self.status_label.setText("No files imported")
        elif count == 1:
            self.status_label.setText("1 file imported")
        else:
            self.status_label.setText(f"{count} files imported")
    
    def get_files(self):
        """
        Get the list of selected file paths.
        
        Returns:
            List of file paths
        """
        return self.file_paths.copy() 