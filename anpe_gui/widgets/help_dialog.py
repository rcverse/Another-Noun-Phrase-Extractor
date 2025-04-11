"""
Custom QDialog to display help content from a Markdown file
using a QTextBrowser for better formatting and scrolling.
"""

import logging
import sys
import re
from pathlib import Path
from typing import Optional, Dict, List
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QTextBrowser, QPushButton, QDialogButtonBox, QLabel, QWidget, QHBoxLayout, QFrame, QGridLayout,
    QSplitter, QTreeWidget, QTreeWidgetItem, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, QUrl, QSize, QCoreApplication
from PyQt6.QtGui import QDesktopServices, QIcon, QPixmap, QFont, QColor, QTextDocument, QGuiApplication # For opening URLs and painting
from PyQt6.QtWidgets import QMessageBox # For About box
from anpe_gui.theme import PRIMARY_COLOR, get_scroll_bar_style, LIGHT_HOVER_BLUE  # Import theme colors and scroll bar style
from anpe_gui.resource_manager import ResourceManager

class HelpDialog(QDialog):
    def __init__(self, help_file_path: Path, gui_version: str, core_version: str, parent=None):
        super().__init__(parent)
        self.help_file_path = help_file_path
        self.gui_version = gui_version
        self.core_version = core_version
        self.setWindowTitle("ANPE GUI Help")
        self.setMinimumSize(800, 700)  # Reduced minimum width
        self.resize(800, 700)  # Reduced default width
        
        # Store sections for navigation
        self.sections = {}  # Will hold section name -> anchor mapping
        self.section_levels = {}  # Will track heading levels
        self.html_content = ""  # Will store the final HTML content
        
        # Store heading positions for direct navigation
        self.heading_positions = {}  # Will store heading text -> cursor position mapping

        self.setup_ui()
        self.load_help_content()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)  # Reduce spacing between elements
        layout.setContentsMargins(0, 0, 0, 0)  # No margins for main layout
        
        # Create a splitter as the main container
        content_splitter = QSplitter(Qt.Orientation.Horizontal)
        content_splitter.setChildrenCollapsible(False)  # Prevent collapsing sections
        content_splitter.setHandleWidth(0)  # Remove the visible resize handle
        
        # Left side: Navigation tree
        nav_container = QWidget()
        nav_layout = QVBoxLayout(nav_container)
        nav_layout.setContentsMargins(10, 10, 0, 10)
        nav_layout.setSpacing(5)
        
        # Add a simple header for the navigation
        nav_header = QLabel("Contents")
        nav_header.setStyleSheet(f"""
            QLabel {{
                color: {PRIMARY_COLOR};
                font-size: 14px;
                font-weight: bold;
                padding-bottom: 5px;
                border-bottom: 1px solid #ddd;
            }}
        """)
        nav_layout.addWidget(nav_header)
        
        # Tree widget for navigation
        self.nav_tree = QTreeWidget()
        self.nav_tree.setHeaderHidden(True)
        self.nav_tree.setFixedWidth(250)  # Increased width from 230 to 250
        
        # Use specific SVG files for branch indicators
        expand_close_url = ResourceManager.get_style_url("expand_close.svg")
        expand_open_url = ResourceManager.get_style_url("expand_open.svg")
        
        self.nav_tree.setStyleSheet(f"""
            QTreeWidget {{
                background-color: #F7F7F7;
                border: none;
                font-size: 10pt;
                padding: 5px;
            }}
            QTreeWidget::item {{
                padding: 4px 2px; /* vertical padding = 4px, horizontal padding = 2px */
                min-height: 22px; /* Set minimum height to prevent crowding */
                border-bottom: 1px solid #EEEEEE;
                border: none; /* Remove border */
                outline: none; /* Remove outline */
            }}
            QTreeWidget::item:hover:!selected {{
                background-color: {LIGHT_HOVER_BLUE}; /* Light blue background on hover */
                color: {PRIMARY_COLOR}; /* Use primary color text on hover */
            }}
            QTreeWidget::item:selected {{
                background-color: #E1EDFF; /* Light blue background when selected */
                color: #333333; /* Keep text dark for better contrast */
                border: none; /* No border */
                outline: none; /* Remove focus rectangle */
                border-radius: 0px;
            }}
            QTreeWidget:focus {{
                outline: none; /* Remove focus outline */
            }}
            /* Use specific SVG files for branch indicators */
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {{
                image: url("{expand_close_url}");
                margin-right: 3px; /* Add some space */
            }}
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {{
                image: url("{expand_open_url}"); 
                margin-right: 3px; /* Add some space */
            }}
            {get_scroll_bar_style(vertical_width=8, horizontal_height=8, handle_min_size=30, border_radius=4)}
        """)
        self.nav_tree.itemClicked.connect(self.on_nav_item_clicked)
        nav_layout.addWidget(self.nav_tree)
        content_splitter.addWidget(nav_container)
        
        # Right side: Content
        content_container = QWidget()
        content_layout = QVBoxLayout(content_container)
        content_layout.setContentsMargins(20, 10, 20, 10)  # Increased horizontal margins
        content_layout.setSpacing(10)
        
        # Text browser for content
        self.text_browser = QTextBrowser()
        self.text_browser.setReadOnly(True)
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.anchorClicked.connect(self.handle_anchor_click)
        
        # Style the text browser
        self.text_browser.setStyleSheet(f"""
            QTextBrowser {{
                border: none;
                padding: 15px 25px;  /* Increased padding to create narrower text column */
                font-size: 13px;
                line-height: 170%;
                background-color: white;
            }}
            {get_scroll_bar_style(vertical_width=10, horizontal_height=10, handle_min_size=30, border_radius=5)}
        """)
        
        # Set document margins for better readability
        document = self.text_browser.document()
        document.setDocumentMargin(20)  # Increased document margin
        
        content_layout.addWidget(self.text_browser)
        
        # Add buttons at the bottom
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 5, 0, 0)
        button_layout.setSpacing(10)
        
        self.project_page_button = QPushButton("Project Page")
        self.about_button = QPushButton("About")
        self.ok_button = QPushButton("OK")
        
        button_style = """
            QPushButton {
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 12px;
            }
        """
        self.project_page_button.setStyleSheet(button_style)
        self.about_button.setStyleSheet(button_style)
        self.ok_button.setStyleSheet(button_style)
        
        button_layout.addStretch()
        button_layout.addWidget(self.project_page_button)
        button_layout.addWidget(self.about_button)
        button_layout.addWidget(self.ok_button)
        
        self.ok_button.clicked.connect(self.accept)
        self.project_page_button.clicked.connect(self.go_to_project_page)
        self.about_button.clicked.connect(self.show_about_info)
        
        content_layout.addWidget(button_container)
        content_splitter.addWidget(content_container)
        
        # Set fixed sizes for the splitter (navigation panel fixed, content panel takes the rest)
        content_splitter.setSizes([250, 550])  # Adjusted to match the new navigation width
        content_splitter.setStretchFactor(0, 0)  # Don't allow navigation panel to stretch
        content_splitter.setStretchFactor(1, 1)  # Allow content panel to stretch
        layout.addWidget(content_splitter)

    def load_help_content(self):
        """Load and display the help content from the markdown file."""
        try:
            with open(self.help_file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # First, extract headings for navigation
                self.process_content(content)
                
                # Pre-process the markdown content for better readability
                content = self.preprocess_markdown(content)
                
                # Convert markdown to HTML using our custom converter
                html_content = self.markdown_to_html(content)
                
                # Apply styling to HTML content
                styled_html = self.apply_styling(html_content)
                
                # Set HTML content in the text browser
                self.text_browser.setHtml(styled_html)
                
                # Find and store positions of all headings for navigation
                self.find_heading_positions()
                
                # Populate the navigation tree
                self.populate_navigation_tree()
                
        except Exception as e:
            self.text_browser.setPlainText(f"Error loading help content: {str(e)}")
            logging.error(f"Error loading help content: {str(e)}")

    def process_content(self, content):
        """Process markdown content to extract sections for navigation."""
        self.sections = {}
        self.section_levels = {}
        
        # Regular expression to find headings
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
        for match in heading_pattern.finditer(content):
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            # Create anchor ID - simple and consistent
            anchor_id = heading_text.lower().replace(' ', '-')
            anchor_id = re.sub(r'[^\w\-]', '', anchor_id)
            
            # Store section information
            self.sections[heading_text] = anchor_id
            self.section_levels[heading_text] = level

    def find_heading_positions(self):
        """Find and store the positions of all headings in the document."""
        self.heading_positions = {}
        
        # Get the document and search through all blocks
        doc = self.text_browser.document()
        
        # Loop through all blocks in the document
        for i in range(doc.blockCount()):
            block = doc.findBlockByNumber(i)
            text = block.text().strip()
            
            # For each heading in our sections, try to find it in the text
            for heading in self.sections.keys():
                # Strip any HTML tags from the text for comparison
                plain_text = re.sub(r'<[^>]+>', '', text)
                
                # Check if this block contains our heading
                if heading in plain_text:
                    self.heading_positions[heading] = block.position()
                    break
        
        # Debug logging
        if len(self.heading_positions) != len(self.sections):
            logging.warning(f"Only found positions for {len(self.heading_positions)} out of {len(self.sections)} headings")

    def populate_navigation_tree(self):
        """Populate the navigation tree with sections."""
        if not self.sections:
            return
            
        # Create parent-child relationships based on heading levels
        current_parents = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None}
        
        # Process all sections
        for heading, anchor in self.sections.items():
            level = self.section_levels[heading]
            
            # Create tree item
            item = QTreeWidgetItem([heading])
            item.setData(0, Qt.ItemDataRole.UserRole, anchor)
            
            # Style based on level
            font = item.font(0)
            if level == 1:
                font.setBold(True)
                font.setPointSize(11)
            elif level == 2:
                font.setBold(True)
            item.setFont(0, font)
            
            # Update current parent for this level
            current_parents[level] = item
            
            # Find parent based on heading level
            if level == 1:
                # Top-level heading
                self.nav_tree.addTopLevelItem(item)
            else:
                # Find the closest parent level
                parent_level = level - 1
                while parent_level > 0:
                    if current_parents[parent_level] is not None:
                        current_parents[parent_level].addChild(item)
                        break
                    parent_level -= 1
                if parent_level == 0:
                    # No parent found, add as top-level
                    self.nav_tree.addTopLevelItem(item)
        
        # Expand all tree items by default
        self.nav_tree.expandAll()

    def on_nav_item_clicked(self, item, column):
        """Handle navigation tree item click."""
        heading = item.text(0)  # Get the heading text
        
        # Method 1: Use stored position for direct navigation
        if heading in self.heading_positions:
            position = self.heading_positions[heading]
            
            # First ensure cursor is at the heading position
            cursor = self.text_browser.textCursor()
            cursor.setPosition(position)
            self.text_browser.setTextCursor(cursor)
            
            # Force the heading to appear at the top by doing the following:
            # 1. Get the current block position (y-coordinate)
            current_block = cursor.block()
            document = self.text_browser.document()
            layout = document.documentLayout()
            block_position = layout.blockBoundingRect(current_block).y()
            
            # 2. Set the vertical scroll bar position to match this y-coordinate
            scroll_bar = self.text_browser.verticalScrollBar()
            scroll_bar.setValue(int(block_position))
            
            # Force UI update
            QCoreApplication.processEvents()
            return
            
        # Method 2: Fallback to anchor-based navigation
        anchor = item.data(0, Qt.ItemDataRole.UserRole)
        if anchor:
            # Try to scroll to the anchor as fallback
            self.text_browser.scrollToAnchor(anchor)
            QCoreApplication.processEvents()

    def handle_anchor_click(self, url):
        """Handle clicked links."""
        if url.scheme() in ('http', 'https'):
            # External URL - open in browser
            QDesktopServices.openUrl(url)
            return True
        else:
            # Internal link - try to scroll to anchor
            fragment = url.fragment()
            if fragment:
                self.text_browser.scrollToAnchor(fragment)
                return True
        return False

    def go_to_project_page(self):
        """Open the project page in a browser."""
        QDesktopServices.openUrl(QUrl("https://github.com/rcverse/Another-Noun-Phrase-Extractor"))

    def show_about_info(self):
        """Show the about dialog."""
        from anpe_gui.version import __version__ as gui_version
        try:
            from anpe import __version__ as core_version
        except ImportError:
            core_version = "N/A"
        
        about_dialog = AboutDialog(gui_version, core_version, self)
        about_dialog.exec()

    def preprocess_markdown(self, content):
        """Pre-process markdown content to improve readability."""
        # Add extra newlines before headings for better spacing
        content = re.sub(r'(\n#{1,6}\s+)', r'\n\n\1', content)
        
        # Add extra newlines after headings
        content = re.sub(r'(#{1,6}\s+.+)(\n)', r'\1\n\2', content)
        
        # Add extra newlines before lists for better spacing
        content = re.sub(r'(\n\s*[-*]\s+)', r'\n\1', content)
        
        # Reduce indentation in lists - replace multiple spaces with just 2
        content = re.sub(r'(\n)( {4,})([*-])', r'\1  \3', content)
        
        # Add extra spacing between list items
        content = re.sub(r'(\n\s*[-*]\s+[^\n]+)(\n\s*[-*]\s+)', r'\1\n\2', content)
        
        # Add horizontal line before major section headings (##)
        content = re.sub(r'(\n## )', r'\n\n---\n\n\1', content)
        
        # Add extra spacing after paragraphs
        content = re.sub(r'(\n[^\s#>*-][^\n]+)(\n[^\s#>*-])', r'\1\n\2', content)
        
        return content

    def markdown_to_html(self, markdown_text):
        """Basic markdown to HTML conversion using explicit tags."""
        html = markdown_text

        # Convert custom tags first
        html = re.sub(r'<button>(.*?)</button>', r'<span class="custom-button">\1</span>', html)
        html = re.sub(r'<option>(.*?)</option>', r'<span class="custom-option">\1</span>', html)
        html = re.sub(r'<format>(.*?)</format>', r'<span class="custom-format">\1</span>', html)

        # Convert standard markdown headings (e.g., # Heading -> <h1>Heading</h1>)
        for i in range(6, 0, -1):
            pattern = r'^{0}\s+(.+?)$'.format('#' * i)
            replacement = r'<h{0}>\1</h{0}>'.format(i)
            html = re.sub(pattern, replacement, html, flags=re.MULTILINE)

        # Convert bold (e.g., **text** -> <strong>text</strong>)
        html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html)

        # Convert code (e.g., `text` -> <code>text</code>)
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

        # Convert horizontal rule (e.g., --- -> <hr>)
        html = re.sub(r'^---+$', r'<hr>', html, flags=re.MULTILINE)

        # Convert unordered lists
        # First, detect list items and wrap with <li> tags
        html = re.sub(r'^\s*[-*]\s+(.+?)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        # Then, group consecutive <li> items into <ul> lists
        html = re.sub(r'(?:<li>.*?</li>\s*)+', lambda m: f'<ul>{m.group(0)}</ul>', html, flags=re.DOTALL)

        # Convert links (e.g., [text](url) -> <a href="url">text</a>)
        html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)

        # Convert paragraphs (blocks of text not otherwise formatted)
        # First, split into blocks separated by blank lines
        blocks = re.split(r'\n\s*\n', html)
        for i, block in enumerate(blocks):
            # Skip blocks that are already wrapped with HTML tags (headings, lists, hr, etc.)
            stripped_block = block.strip()
            if not stripped_block or re.match(r'^\s*<(h[1-6]|ul|li|hr|p|code|strong|a|span)', stripped_block.lower()):
                continue
            # Wrap other blocks with <p> tags
            blocks[i] = f'<p>{stripped_block}</p>'
        html = '\n\n'.join(blocks)

        return html

    def apply_styling(self, html_content):
        """Apply custom styling to HTML content."""
        # Add CSS styles
        css = """
        <style>
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 0;
                color: #333; /* Default text color */
            }
            h1, h2, h3, h4, h5, h6 {
                color: #005A9C; /* Heading color */
                margin-top: 1.2em;
                margin-bottom: 0.6em;
                line-height: 1.3;
                font-weight: bold;
            }
            h1 { font-size: 1.8em; }
            h2 { font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
            h3 { font-size: 1.3em; }
            h4 { font-size: 1.1em; margin-top: 1em; margin-bottom: 0.4em; }
            p { margin: 0.6em 0 0.8em 0; }
            ul, ol {
                margin: 0.6em 0;
                padding-left: 1.5em;  /* Standard list indentation */
            }
            li {
                margin: 0.3em 0;
                list-style-position: outside;
            }
            hr {
                height: 0.15em;
                padding: 0;
                margin: 1.5em 0; /* Increased margin for visual separation */
                background-color: #e1e4e8;
                border: 0;
            }
            code {
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
                background-color: #f5f5f5; /* Light grey background for code */
                color: #333;
                padding: 2px 5px;
                border-radius: 4px;
                font-size: 0.95em;
            }
            strong {
                font-weight: bold;
            }
            a {
                color: #005A9C; /* Link color */
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            /* Custom tag styles */
            .custom-button {
                display: inline-block;
                background-color: #e7f3ff; /* Light blue background */
                color: #005A9C; /* Dark blue text */
                font-weight: bold;
                padding: 4px 10px; /* Further Increased padding */
                border-radius: 12px; /* Further Rounded corners */
                white-space: nowrap;
                font-style: normal;
                font-size: 0.95em;
                line-height: 1.3; /* Ensure line-height doesn't interfere */
            }
            .custom-option {
                display: inline-block;
                background-color: #f0f0f0; /* Light grey background */
                color: #333; /* Dark text */
                font-weight: normal;
                padding: 4px 10px; /* Further Increased padding */
                border-radius: 12px; /* Further Rounded corners */
                white-space: nowrap;
                font-style: normal;
                font-size: 0.95em;
                line-height: 1.3; /* Ensure line-height doesn't interfere */
            }
            .custom-format {
                display: inline-block;
                background-color: #f0f0f0; /* Light grey background */
                color: #333; /* Dark text */
                font-weight: normal;
                padding: 4px 10px; /* Further Increased padding */
                border-radius: 12px; /* Further Rounded corners */
                white-space: nowrap;
                font-style: italic;
                font-size: 0.95em;
                line-height: 1.3; /* Ensure line-height doesn't interfere */
            }
        </style>
        """

        # Wrap in HTML structure
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            {css}
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

class AboutDialog(QDialog):
    """Custom About dialog with proper icon display."""
    
    def __init__(self, gui_version, core_version, parent=None):
        super().__init__(parent)
        self.gui_version = gui_version
        self.core_version = core_version
        self.setWindowTitle("About ANPE GUI")
        self.setMinimumWidth(600)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        
        self.setup_ui()
        
    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # --- Header Section ---
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setSpacing(20) # Increased spacing
        header_layout.setContentsMargins(0, 0, 0, 10) # Add bottom margin

        # Icon (left)
        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = ResourceManager.get_pixmap("app_icon.png")
        pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        icon_label.setPixmap(pixmap)
        icon_label.setFixedSize(100, 100) # Fix size for consistent layout
        header_layout.addWidget(icon_label, 0, Qt.AlignmentFlag.AlignTop) # Align icon to top

        # Title/Subtitle (right)
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(5)
        title_layout.addStretch(1) # Push content down vertically

        title_label = QLabel("ANPE")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #005A9C;")
        title_layout.addWidget(title_label)

        subtitle_label = QLabel("Another Noun Phrase Extractor")
        subtitle_label.setStyleSheet("font-size: 16px; color: #666666;")
        title_layout.addWidget(subtitle_label)
        title_layout.addStretch(1) # Push content up vertically

        header_layout.addWidget(title_container, 1)
        main_layout.addWidget(header_widget)

        # --- Info Grid ---
        info_widget = QWidget()
        info_layout = QGridLayout(info_widget)
        info_layout.setContentsMargins(10, 0, 10, 10) # Reduced top margin
        info_layout.setSpacing(8)
        info_layout.setColumnStretch(1, 1) # Allow value column to stretch

        # Labels column width
        label_width = 110

        # GUI Version
        gui_label = QLabel("GUI Version:")
        gui_label.setStyleSheet("font-weight: bold;")
        gui_label.setFixedWidth(label_width)
        gui_value = QLabel(self.gui_version)
        info_layout.addWidget(gui_label, 0, 0, Qt.AlignmentFlag.AlignLeft)
        info_layout.addWidget(gui_value, 0, 1, Qt.AlignmentFlag.AlignLeft)

        # Core Version
        core_label = QLabel("Core Version:")
        core_label.setStyleSheet("font-weight: bold;")
        core_label.setFixedWidth(label_width)
        core_value = QLabel(self.core_version)
        info_layout.addWidget(core_label, 1, 0, Qt.AlignmentFlag.AlignLeft)
        info_layout.addWidget(core_value, 1, 1, Qt.AlignmentFlag.AlignLeft)

        # Author
        author_label = QLabel("Author:")
        author_label.setStyleSheet("font-weight: bold;")
        author_label.setFixedWidth(label_width)
        author_value = QLabel("Richard Chen (@rcverse)")
        info_layout.addWidget(author_label, 2, 0, Qt.AlignmentFlag.AlignLeft)
        info_layout.addWidget(author_value, 2, 1, Qt.AlignmentFlag.AlignLeft)

        # License
        license_label = QLabel("License:")
        license_label.setStyleSheet("font-weight: bold;")
        license_label.setFixedWidth(label_width)
        license_value = QLabel("GNU General Public License v3") # View button moved
        info_layout.addWidget(license_label, 3, 0, Qt.AlignmentFlag.AlignLeft)
        info_layout.addWidget(license_value, 3, 1, Qt.AlignmentFlag.AlignLeft)

        # Contact
        email_label = QLabel("Contact:")
        email_label.setStyleSheet("font-weight: bold;")
        email_label.setFixedWidth(label_width)
        email_value = QLabel('<a href="mailto:rcverse6@gmail.com">rcverse6@gmail.com</a>')
        email_value.setTextFormat(Qt.TextFormat.RichText)
        email_value.setOpenExternalLinks(True)
        info_layout.addWidget(email_label, 4, 0, Qt.AlignmentFlag.AlignLeft)
        info_layout.addWidget(email_value, 4, 1, Qt.AlignmentFlag.AlignLeft)

        main_layout.addWidget(info_widget)

        # --- Separator ---
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #e0e0e0;")
        main_layout.addWidget(separator)


        # --- Acknowledgements Section ---
        ack_widget = QWidget()
        ack_layout = QVBoxLayout(ack_widget)
        ack_layout.setContentsMargins(10, 5, 10, 5) # Reduced vertical margins
        ack_layout.setSpacing(5)

        ack_title = QLabel("Acknowledgements")
        ack_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #333; margin-bottom: 5px;")
        ack_layout.addWidget(ack_title)

        ack_text = (
            "This application uses the following open-source libraries:<br>"
            "• <b>PyQt6</b> (GPLv3 / Commercial)<br>"
            "• <b>spaCy</b> (MIT License)<br>"
            "• <b>Benepar</b> (MIT License)<br>"
            "• <b>NLTK</b> (Apache License 2.0)<br><br>"
            "We are grateful for the developers of these packages that make ANPE and ANPE GUI possible. "
            "You can find more detailed license information in the application's About dialog."
        )
        ack_label = QLabel(ack_text)
        ack_label.setWordWrap(True)
        ack_label.setTextFormat(Qt.TextFormat.RichText) # Use RichText for <br>
        ack_label.setStyleSheet("font-size: 11px; color: #555;")
        ack_layout.addWidget(ack_label)

        main_layout.addWidget(ack_widget)
        main_layout.addStretch(1) # Push buttons to the bottom

        # --- Button Bar ---
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 10, 0, 0) # Add top margin
        button_layout.setSpacing(10)

        # Project Page Button
        project_page_button = QPushButton("Visit Project Page")
        project_page_button.clicked.connect(self._visit_project_page)
        button_layout.addWidget(project_page_button)
        
        # View License Button
        self.license_view_button = QPushButton("View License") # Store as member
        self.license_view_button.clicked.connect(self._show_license)
        button_layout.addWidget(self.license_view_button)


        button_layout.addStretch(1) # Push OK button to the right

        # OK Button
        ok_button = QPushButton("OK")
        ok_button.setDefault(True) # Make OK the default button
        ok_button.setFixedWidth(100) # Slightly wider
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)

        main_layout.addWidget(button_widget)

    def _show_license(self):
        """Show the license dialog."""
        # Center the license dialog on the main window or screen
        try:
            from anpe_gui.widgets.license_dialog import LicenseDialog
            license_dialog = LicenseDialog(self)
            parent_geometry = self.geometry()
            screen_geometry = QGuiApplication.primaryScreen().availableGeometry()
            dialog_size = license_dialog.sizeHint() # Get the recommended size

            # Calculate center position relative to parent
            x = parent_geometry.x() + (parent_geometry.width() - dialog_size.width()) // 2
            y = parent_geometry.y() + (parent_geometry.height() - dialog_size.height()) // 2

            # Ensure it's within screen bounds
            x = max(screen_geometry.x(), min(x, screen_geometry.x() + screen_geometry.width() - dialog_size.width()))
            y = max(screen_geometry.y(), min(y, screen_geometry.y() + screen_geometry.height() - dialog_size.height()))

            license_dialog.move(x, y)
            license_dialog.exec()
        except Exception as e:
            logging.error(f"Failed to show license dialog: {e}")
            QMessageBox.warning(self, "Error", "Could not display the license information.")
        
    def _visit_project_page(self):
        """Open project page in browser."""
        QDesktopServices.openUrl(QUrl("https://github.com/rcverse/Another-Noun-Phrase-Extractor")) 