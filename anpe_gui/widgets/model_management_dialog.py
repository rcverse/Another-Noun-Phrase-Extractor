from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel

class ModelManagementDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Management")
        self.setMinimumWidth(400)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Add title label
        title_label = QLabel("Model Management")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Add buttons
        self.load_button = QPushButton("Load Model")
        self.save_button = QPushButton("Save Model")
        self.delete_button = QPushButton("Delete Model")
        
        layout.addWidget(self.load_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.delete_button)
        
        self.setLayout(layout)
        
        # Set stylesheet
        self.setStyleSheet("""
            /* Buttons */
            QPushButton {
                background-color: #005A9C;
                color: white;
                padding: 8px 15px;
                font-size: 10pt;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #E0E0E0;
            }
            QPushButton:hover:!disabled {
                background-color: #004C8C;
            }
            QPushButton:pressed:!disabled {
                background-color: #003366;
            }
        """) 