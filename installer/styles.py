"""
Centralized styles for the installer application.

This module provides consistent styling across all views in the application.
"""

# Button Styles
PRIMARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #005A9C;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 16px;
        font-size: 14pt;
        min-width: 120px;
    }
    QPushButton:hover {
        background-color: #0066b2;
    }
    QPushButton:pressed {
        background-color: #005299;
    }
    QPushButton:disabled {
        background-color: #80bbeb;
    }
"""

SECONDARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #0078d7;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 13px;
    }
    QPushButton:hover {
        background-color: #0066b2;
    }
    QPushButton:pressed {
        background-color: #005299;
    }
"""

BROWSE_BUTTON_STYLE = """
    QPushButton {
        background-color: #005A9C;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 5px 10px;
        font-size: 12px;
    }
    QPushButton:hover {
        background-color: #0066b2;
    }
    QPushButton:pressed {
        background-color: #005299;
    }
"""

# Progress Bar Style
PROGRESS_BAR_STYLE = """
    QProgressBar {
        background-color: #e0e0e0;
        border: none;
        border-radius: 2px;
    }
    QProgressBar::chunk {
        background-color: #0078d7;
        border-radius: 2px;
    }
"""

# Text Area Style
LOG_TEXT_AREA_STYLE = """
    QTextEdit {
        background-color: #f8f8f8;
        border: 1px solid #e0e0e0;
        border-radius: 4px;
        padding: 5px;
        font-family: Consolas, monospace;
    }
"""

# Label Styles
TITLE_LABEL_STYLE = "font-size: 24px; font-weight: bold; color: #0066b2;"
STATUS_LABEL_STYLE = "font-size: 14px; margin-top: 5px; margin-bottom: 5px;"
INFO_LABEL_STYLE = "color: #666666; font-style: italic;"
LINK_LABEL_STYLE = "color: #0066b2; text-decoration: none;"

# Compact Button Styles (for Uninstaller, etc.)
COMPACT_PRIMARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #005A9C;
        color: white;
        border: none;
        border-radius: 3px;
        padding: 5px 12px;
        font-size: 12px;
        min-width: 80px;
    }
    QPushButton:hover {
        background-color: #0066b2;
    }
    QPushButton:pressed {
        background-color: #005299;
    }
    QPushButton:disabled {
        background-color: #80bbeb;
    }
"""

COMPACT_SECONDARY_BUTTON_STYLE = """
    QPushButton {
        background-color: #e0e0e0; /* Lighter background for secondary */
        color: #333333;
        border: 1px solid #cccccc;
        border-radius: 3px;
        padding: 4px 10px;
        font-size: 12px;
        min-width: 80px;
    }
    QPushButton:hover {
        background-color: #d0d0d0;
        border-color: #bbbbbb;
    }
    QPushButton:pressed {
        background-color: #c0c0c0;
    }
    QPushButton:disabled {
        background-color: #f0f0f0;
        color: #aaaaaa;
    }
"""

# Destructive Action Button Style (Red)
DANGER_BUTTON_STYLE = """
    QPushButton {
        background-color: #d32f2f; /* Red color */
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 16px;
        font-size: 14pt;
        min-width: 120px;
    }
    QPushButton:hover {
        background-color: #c62828; /* Darker red on hover */
    }
    QPushButton:pressed {
        background-color: #b71c1c; /* Even darker red on press */
    }
    QPushButton:disabled {
        background-color: #ef9a9a; /* Lighter red when disabled */
        color: #e0e0e0;
    }
"""

# Compact Destructive Action Button Style (Red)
COMPACT_DANGER_BUTTON_STYLE = """
    QPushButton {
        background-color: #d32f2f; /* Red color */
        color: white;
        border: none;
        border-radius: 3px;
        padding: 5px 12px;
        font-size: 12px;
        min-width: 80px;
    }
    QPushButton:hover {
        background-color: #c62828; /* Darker red on hover */
    }
    QPushButton:pressed {
        background-color: #b71c1c; /* Even darker red on press */
    }
    QPushButton:disabled {
        background-color: #ef9a9a; /* Lighter red when disabled */
        color: #e0e0e0;
    }
""" 