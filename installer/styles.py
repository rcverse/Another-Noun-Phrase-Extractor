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