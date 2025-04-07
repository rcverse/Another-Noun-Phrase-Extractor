"""
Theme constants and stylesheet definitions for the ANPE GUI application.
Uses a cohesive blue theme.
"""
from PyQt6.QtGui import QPalette, QColor

# --- Color Palette (Simplified Grey/White Style) --- 
PRIMARY_COLOR = "#005A9C"     # Dark blue for key elements
SECONDARY_COLOR = "#003F7A"   # Darker blue accent
ACCENT_COLOR = "#00AEEF"     # Bright accent (e.g., progress bar chunk)
BACKGROUND_COLOR = "#FFFFFF" # White background
TEXT_COLOR = "#212529"       # Standard dark text
HOVER_COLOR = "#004C8C"    
PRESSED_COLOR = "#003366"   
DISABLED_COLOR = "#E0E0E0"   # Light grey for disabled/inactive elements
BORDER_COLOR = "#CCCCCC"    # Standard grey border
ERROR_COLOR = "#DC3545"       # Red for errors
SUCCESS_COLOR = "#28A745"     # Green for success
WARNING_COLOR = "#FFC107"     # Yellow for warnings
INFO_COLOR = "#17A2B8"        # Teal for info


# --- Base Stylesheet --- 
def get_stylesheet():
    return f"""
    /* General Styling */
    QWidget {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
        font-family: Segoe UI, Arial, sans-serif; 
        font-size: 9pt; /* Slightly smaller base font */
    }}

    QMainWindow {{
        background-color: {BACKGROUND_COLOR};
    }}

    QFrame[frameShape="4"], /* HLine */
    QFrame[frameShape="5"] /* VLine */
    {{
        background-color: {BORDER_COLOR};
        border: none;
    }}

    /* Headings and Labels */
    QLabel {{
        background-color: transparent; /* Ensure labels don't obscure background */
        padding: 2px;
    }}
    QLabel[heading="true"] {{
        font-size: 18pt;
        font-weight: bold;
        color: {PRIMARY_COLOR};
        padding-bottom: 5px;
    }}
     QLabel[subheading="true"] {{
        font-size: 12pt;
        font-weight: bold;
        color: {SECONDARY_COLOR};
        padding-top: 5px;
        padding-bottom: 3px;
    }}

    /* Buttons */
    QPushButton {{
        background-color: {PRIMARY_COLOR};
        color: white;
        border: 1px solid {SECONDARY_COLOR};
        padding: 5px 10px; /* Reduced padding */
        border-radius: 3px; /* Slightly smaller radius */
        min-width: 70px; /* Adjust min width */
    }}
    QPushButton:hover {{
        background-color: {HOVER_COLOR};
        border: 1px solid {PRIMARY_COLOR};
    }}
    QPushButton:pressed {{
        background-color: {PRESSED_COLOR};
    }}
    QPushButton:disabled {{
        background-color: {DISABLED_COLOR};
        color: #888888;
        border: 1px solid #AAAAAA;
    }}
    QPushButton[checkable="true"]:checked {{
        background-color: {SECONDARY_COLOR}; /* Darker blue when checked */
        color: white; /* Ensure text remains white */
        border: 1px solid {PRIMARY_COLOR};
        font-weight: bold; /* Make text bold when checked */
    }}
    QPushButton[checkable="true"]:!checked {{
        background-color: {BACKGROUND_COLOR}; /* Match background when unchecked */
        color: {PRIMARY_COLOR}; /* Use primary text color */
        border: 1px solid {BORDER_COLOR}; /* Standard border */
    }}
    QPushButton[checkable="true"]:!checked:hover {{
        background-color: #E8E8E8; /* Light grey hover for unchecked */
        border: 1px solid {SECONDARY_COLOR};
    }}

    /* Specific Buttons (e.g., for Add/Remove) */
    FileListWidget QPushButton,
    StructureFilterWidget QPushButton {{
        padding: 5px 10px; /* Slightly smaller padding */
        min-width: 60px;
    }}

    /* Input Fields */
    QLineEdit, QTextEdit, QSpinBox {{
        background-color: white;
        border: 1px solid {BORDER_COLOR};
        padding: 4px;
        border-radius: 2px;
        min-height: 18px; /* Ensure minimum height */
    }}
    QLineEdit:focus, QTextEdit:focus, QSpinBox:focus {{
        border: 1px solid {PRIMARY_COLOR};
    }}
    QLineEdit:disabled, QTextEdit:disabled, QSpinBox:disabled {{
        background-color: #EEEEEE;
        color: #777777;
    }}
    /* SpinBox Specific Styling */
    QSpinBox {{
        padding: 2px;
        border: 1px solid {BORDER_COLOR};
        border-radius: 3px;
        min-width: 60px; /* Ensure some minimum width */
    }}
    QSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right; 
        width: 16px; 
        border-left: 1px solid {BORDER_COLOR};
        border-bottom: 1px solid {BORDER_COLOR};
        border-top-right-radius: 3px; 
        background-color: #E0E0E0; /* Light gray background */
    }}
    QSpinBox::up-button:hover {{
        background-color: #D0D0D0;
    }}
    QSpinBox::up-arrow {{
        image: url(:/qt-project.org/styles/commonstyle/images/standardbutton-up-16.png); /* Use standard Qt arrow */
        width: 9px;
        height: 9px;
    }}
    QSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 16px;
        border-left: 1px solid {BORDER_COLOR};
        border-top: 1px solid {BORDER_COLOR};
        border-bottom-right-radius: 3px;
        background-color: #E0E0E0;
    }}
    QSpinBox::down-button:hover {{
        background-color: #D0D0D0;
    }}
    QSpinBox::down-arrow {{
        image: url(:/qt-project.org/styles/commonstyle/images/standardbutton-down-16.png); /* Use standard Qt arrow */
        width: 9px;
        height: 9px;
    }}

    /* Checkboxes and Radio Buttons */
    QCheckBox, QRadioButton {{ spacing: 4px; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 14px; height: 14px; }}
    QCheckBox::indicator:unchecked {{
        border: 1px solid {BORDER_COLOR};
        background-color: white;
        border-radius: 2px;
    }}
    /* Remove custom checked style to allow native rendering with checkmark */
    /* QCheckBox::indicator:checked {{
        border: 2px solid {PRIMARY_COLOR}; 
        background-color: transparent; 
        border-radius: 2px; 
    }} */
     QCheckBox::indicator:disabled {{
        border: 1px solid #AAAAAA;
        background-color: #DDDDDD;
    }}
     QRadioButton::indicator:unchecked {{
        border: 1px solid {SECONDARY_COLOR};
        background-color: white;
        border-radius: 8px; /* Circle */
    }}
    QRadioButton::indicator:checked {{
        background-color: {PRIMARY_COLOR};
        border: 1px solid {SECONDARY_COLOR};
        image: url(icons/radio_dot_white.svg); /* Needs an icon */
        border-radius: 8px;
    }}
    QRadioButton::indicator:disabled {{
        border: 1px solid #AAAAAA;
        background-color: #DDDDDD;
        border-radius: 8px;
    }}

    /* Group Boxes */
    QGroupBox {{
        border: 1px solid {BORDER_COLOR};
        border-radius: 4px;
        margin-top: 8px; /* Further reduced margin */
        padding: 8px; /* Reduced padding */
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 4px 0 4px; /* Reduced padding */
        left: 8px; 
        background-color: {BACKGROUND_COLOR}; 
        color: {PRIMARY_COLOR};
        font-weight: bold;
    }}
    /* Remove custom checked style for groupbox indicator */
    /* QGroupBox[checkable="true"]::indicator:checked {{
        border: 2px solid {PRIMARY_COLOR}; 
        background-color: transparent; 
        border-radius: 2px; 
    }} */
    # Keep unchecked style consistent if needed
    QGroupBox[checkable="true"]::indicator:unchecked {{
         border: 1px solid {BORDER_COLOR};
         background-color: white;
         border-radius: 2px;
         margin-right: 3px; /* Align with checkbox */
    }}

    /* Combo Boxes */
    QComboBox {{
        border: 1px solid {BORDER_COLOR};
        padding: 5px;
        background-color: white;
        min-width: 100px;
        border-radius: 3px;
    }}
    QComboBox:disabled {{
        background-color: #EEEEEE;
        color: #777777;
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 20px;
        border-left-width: 1px;
        border-left-color: {BORDER_COLOR};
        border-left-style: solid;
        border-top-right-radius: 3px;
        border-bottom-right-radius: 3px;
        background-color: {PRIMARY_COLOR};
    }}
    QComboBox::down-arrow {{
        image: url(icons/arrow_down_white.svg); /* Needs an icon */
        width: 12px;
        height: 12px;
    }}
    QComboBox QAbstractItemView {{ /* Style for the dropdown list */
        border: 1px solid {BORDER_COLOR};
        background-color: white;
        selection-background-color: {PRIMARY_COLOR};
        selection-color: white;
        padding: 2px;
    }}

    /* List Widgets */
    QListWidget {{
        border: 1px solid {BORDER_COLOR};
        padding: 3px;
        background-color: white;
        border-radius: 3px;
    }}
    QListWidget::item {{
        padding: 4px;
    }}
    QListWidget::item:selected {{
        background-color: {PRIMARY_COLOR};
        color: white;
    }}
    QListWidget::item:hover {{
        background-color: {HOVER_COLOR}30; /* Light hover background */
    }}

    /* Tab Bar Styling */
    QTabWidget::pane {{ /* The container for the tab pages */
        border-top: 1px solid {BORDER_COLOR};
        margin-top: -1px; /* Overlap with tab bar border */
    }}
    QTabBar {{
        qproperty-drawBase: 0; /* Turn off the default tab bar base */
        border-bottom: 1px solid {BORDER_COLOR}; /* Add border below tabs */
        margin-left: 5px; /* Space before first tab */
    }}
    QTabBar::tab {{
        background: {DISABLED_COLOR}; /* Background for unselected tabs */
        color: #666666; /* Text color for unselected tabs */
        border: 1px solid {BORDER_COLOR};
        border-bottom: none; /* No border at the bottom */
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
        padding: 8px 15px; /* Increased padding */
        min-height: 25px; /* Ensure minimum height */
        margin-right: 2px; /* Space between tabs */
    }}
    QTabBar::tab:hover {{
        background: #E0E0E0; /* Lighter background on hover */
        color: #333333;
    }}
    QTabBar::tab:selected {{
        background: {BACKGROUND_COLOR}; /* Match window background for selected */
        color: {PRIMARY_COLOR}; /* Primary color text for selected */
        border: 1px solid {BORDER_COLOR};
        border-bottom: 1px solid {BACKGROUND_COLOR}; /* Overlap pane border */
        margin-bottom: -1px; /* Pull tab down slightly */
    }}
    /* Style for tabs not selected but in a selected window */
    QTabBar::tab:!selected {{
        margin-top: 2px; /* Push non-selected tabs down slightly */
    }}

    /* Scroll Bars */
    QScrollBar:vertical {{
        border: 1px solid {BORDER_COLOR};
        background: {BACKGROUND_COLOR}; 
        width: 10px; 
        margin: 10px 0 10px 0; 
    }}
    QScrollBar::handle:vertical {{
        background: #A0A0A0; 
        min-height: 20px;
        border-radius: 5px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: #808080; 
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        border: none;
        background: none; 
        height: 10px;
    }}
    /* QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }} */

    QScrollBar:horizontal {{
        border: 1px solid {BORDER_COLOR};
        background: {BACKGROUND_COLOR};
        height: 10px;
        margin: 0 10px 0 10px;
    }}
    QScrollBar::handle:horizontal {{
        background: #A0A0A0;
        min-width: 20px;
        border-radius: 5px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: #808080;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        border: none;
        background: none;
        width: 10px;
    }}
    /* QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{ background: none; }} */

    /* Splitter */
    QSplitter::handle {{
        background-color: {BORDER_COLOR};
        height: 4px; 
    }}
    QSplitter::handle:horizontal {{
        width: 4px;
        height: 1px; 
    }}
     QSplitter::handle:vertical {{
        height: 4px;
        width: 1px; 
    }}
    QSplitter::handle:hover {{
        background-color: {SECONDARY_COLOR};
    }}

    /* Progress Bar */
    QProgressBar {{
        border: 1px solid {BORDER_COLOR};
        border-radius: 5px;
        text-align: center;
        background-color: white;
        height: 18px;
    }}
    QProgressBar::chunk {{
        background-color: {PRIMARY_COLOR};
        border-radius: 5px;
        width: 10px; /* Width of the moving chunk for indeterminate */
        margin: 1px;
    }}

    /* Status Bar */
    StatusBar QWidget {{
        background-color: {SECONDARY_COLOR}; /* Darker background for status bar */
        color: white;
    }}
    StatusBar QLabel {{
        color: white;
        padding: 2px 5px;
    }}
    StatusBar QProgressBar {{
        max-height: 14px;
        margin: 2px;
    }}
    StatusBar QProgressBar::chunk {{
        background-color: {ACCENT_COLOR};
    }}

    /* Enhanced Log Panel */
    EnhancedLogPanel QTextEdit {{
        background-color: #FEFEFE; /* Slightly off-white */
        font-family: Consolas, monospace;
        font-size: 9pt;
    }}
    EnhancedLogPanel QPushButton {{
        min-width: 40px; /* Smaller buttons for log panel */
        padding: 3px 8px;
    }}
    EnhancedLogPanel QComboBox {{
         min-width: 60px;
         padding: 3px 5px;
    }}

    /* Tooltip Styling (Optional but nice) */
    # ... (Tooltip styles if any) ...

    """

def apply_theme(app):
    """
    Apply the ANPE GUI theme to the application.
    
    Args:
        app: QApplication instance
    """
    # Set the application style sheet
    app.setStyleSheet(get_stylesheet())
    
    # Set default palette colors
    palette = app.palette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND_COLOR))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT_COLOR))
    palette.setColor(QPalette.ColorRole.Base, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#f5f5f5"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT_COLOR))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT_COLOR))
    palette.setColor(QPalette.ColorRole.Button, QColor(SECONDARY_COLOR))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_COLOR))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    app.setPalette(palette) 