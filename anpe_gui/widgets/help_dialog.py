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
from PyQt6.QtGui import QDesktopServices, QIcon, QPixmap, QFont, QColor, QTextDocument # For opening URLs and painting
from PyQt6.QtWidgets import QMessageBox # For About box
from anpe_gui.theme import PRIMARY_COLOR, get_scroll_bar_style  # Import theme colors and scroll bar style

class HelpDialog(QDialog):
    def __init__(self, help_file_path: Path, gui_version: str, core_version: str, parent=None):
        super().__init__(parent)
        self.help_file_path = help_file_path
        self.gui_version = gui_version
        self.core_version = core_version
        self.setWindowTitle("ANPE GUI Help")
        self.setMinimumSize(700, 600)  # Reduced minimum width
        self.resize(700, 600)  # Reduced default width
        
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
        self.nav_tree.setMinimumWidth(200)
        self.nav_tree.setMaximumWidth(250)
        self.nav_tree.setStyleSheet(f"""
            QTreeWidget {{
                border: none;
                background-color: #f8f9fa;
                font-size: 12px;
            }}
            QTreeWidget::item {{
                padding: 4px 4px;
                border-bottom: 1px solid #f0f0f0;
            }}
            QTreeWidget::item:selected {{
                background-color: #e7f3ff;
                color: #005A9C;
                border: none;
                border-radius: 0px;
            }}
            /* Use simple triangle indicators with high contrast */
            QTreeWidget::branch:has-children:!has-siblings:closed,
            QTreeWidget::branch:closed:has-children:has-siblings {{
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iOCIgaGVpZ2h0PSI4IiB2aWV3Qm94PSIwIDAgOCA4IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0yIDFsNCAzLjUtNCAzLjV2LTd6IiBmaWxsPSIjMDAwIi8+PC9zdmc+);
            }}
            QTreeWidget::branch:open:has-children:!has-siblings,
            QTreeWidget::branch:open:has-children:has-siblings {{
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iOCIgaGVpZ2h0PSI4IiB2aWV3Qm94PSIwIDAgOCA4IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPjxwYXRoIGQ9Ik0xIDJsMyA0IDMtNEgxeiIgZmlsbD0iIzAwMCIvPjwvc3ZnPg==);
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
        
        # Set splitter sizes (25% navigation, 75% content)
        content_splitter.setSizes([225, 675])  # Adjusted ratio to give navigation more space
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
        """Basic markdown to HTML conversion without external dependencies."""
        html = markdown_text
        
        # Convert headings (e.g., # Heading -> <h1>Heading</h1>)
        for i in range(6, 0, -1):
            pattern = r'^{0}\s+(.+?)$'.format('#' * i)
            replacement = r'<h{0}>\1</h{0}>'.format(i)
            html = re.sub(pattern, replacement, html, flags=re.MULTILINE)
        
        # Check for special patterns that shouldn't be italicized
        # Mark button-like terms before italic conversion
        buttons = ['Add Files', 'Add Directory', 'Remove/Clear All', 'Clear All', 'Process', 'Export', 'Reset', 'Paste', 'Clear']
        for btn in buttons:
            html = re.sub(r'\b' + re.escape(btn) + r'\b', r'__NOFORMAT__' + btn + r'__NOFORMAT__', html)
        
        # Convert bold (e.g., **text** -> <strong>text</strong>)
        html = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', html)
        
        # Convert italic (e.g., *text* -> <em>text</em>) - but not in URLs or special cases
        # Look for single asterisks that aren't part of double asterisks or URLs
        def replace_italic(match):
            content = match.group(1)
            if '://' in content or content.strip() == '':  # Skip URLs and empty content
                return '*' + content + '*'
            return '<em>' + content + '</em>'
        
        html = re.sub(r'(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)', replace_italic, html)
        
        # Restore special button-like terms
        html = html.replace('__NOFORMAT__', '')
        
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
            # Skip blocks that are already wrapped with HTML tags
            if not block.strip() or re.match(r'^\s*<[a-z]+', block):
                continue
            # Wrap other blocks with <p> tags
            blocks[i] = f'<p>{block.strip()}</p>'
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
                color: #333;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #005A9C;
                margin-top: 1.2em;
                margin-bottom: 0.6em;
                line-height: 1.3;
            }
            h1 { font-size: 1.8em; }
            h2 { font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
            h3 { font-size: 1.3em; }
            h4 { font-size: 1.1em; margin-top: 1em; margin-bottom: 0.4em; }
            p { margin: 0.6em 0 0.8em 0; }
            ul, ol {
                margin: 0.6em 0;
                padding-left: 1.2em;  /* Reduce indentation */
            }
            li {
                margin: 0.2em 0;
                list-style-position: outside;
            }
            hr {
                height: 0.15em;
                padding: 0;
                margin: 1.2em 0;
                background-color: #e1e4e8;
                border: 0;
            }
            code {
                font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
                background-color: #f5f5f5;
                color: #333;
                padding: 2px 4px;
                border-radius: 3px;
            }
            a {
                color: #005A9C;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
            .button {
                display: inline-block;
                background-color: #e7f3ff;
                color: #005A9C;
                font-weight: bold;
                padding: 2px 6px;
                border-radius: 3px;
                white-space: nowrap;
                font-style: normal;
            }
            .ui-element {
                color: #005A9C;
                font-weight: bold;
                white-space: nowrap;
                font-style: normal;
            }
            .option-badge {
                display: inline-block;
                background-color: #e7f3ff;
                color: #005A9C;
                font-weight: bold;
                padding: 2px 6px;
                border-radius: 3px;
                white-space: nowrap;
                font-style: normal;
            }
            .file-format {
                font-family: monospace;
                background-color: #f5f5f5;
                padding: 1px 3px;
                border-radius: 2px;
                color: #333;
                font-style: normal;
            }
            .button-label {
                white-space: nowrap;
                font-style: normal;
            }
            .option-label {
                white-space: nowrap;
                font-style: normal;
            }
            /* Fix for inconsistent italics */
            em {
                font-style: italic;
            }
            p, li, span {
                font-style: normal;
            }
        </style>
        """
        
        # Fix line breaks by wrapping "The X button" phrases in a nowrap span
        buttons = ['Process', 'Export', 'Reset', 'Add Files', 'Add Directory', 'Remove/Clear All', 'Remove', 'Clear All', 'Paste', 'Clear', 'Default']
        for button in buttons:
            # Match the entire phrase "The X button" and wrap it to prevent breaks
            pattern = rf'The\s+("{button}"|{button})\s+(button|buttons|option|options)'
            replacement = rf'<span class="button-label">The <span class="button">\1</span> \2</span>'
            html_content = re.sub(pattern, replacement, html_content)
        
        # Define all option names we want to style
        options = [
            'Include nested phrases', 'Add metadata to output', 'Treat newlines as sentence boundaries',
            'Min Length', 'Max Length', 'Accept Pronouns', 'General Settings', 'Filtering Options',
            'Structure Filtering', 'Control Buttons'
        ]
        
        # First style option names when they appear in "The X option" pattern
        for option in options:
            pattern = rf'The\s+("{option}"|{option})\s+(option|setting)'
            replacement = rf'<span class="option-label">The <span class="ui-element">\1</span> \2</span>'
            html_content = re.sub(pattern, replacement, html_content)
        
        # Then style option names wherever they appear as standalone options
        for option in options:
            # Match the option name when it appears as standalone text, checking for quotes and word boundaries
            pattern = rf'\b("{option}"|{option})\b(?!\s+(option|setting|button|buttons))'
            replacement = rf'<span class="option-badge">\1</span>'
            html_content = re.sub(pattern, replacement, html_content)
        
        # Special handling for specific option-like terms that should be styled
        special_options = ['Include nested phrases', 'Add metadata to output', 'Treat newlines as sentence boundaries', 
                          'Min Length', 'Max Length', 'Accept Pronouns']
        for option in special_options:
            # Make sure these specific options always get the option badge styling
            html_content = re.sub(r'\b' + re.escape(option) + r'\b', 
                               r'<span class="option-badge">' + option + r'</span>', 
                               html_content)
        
        # Style file formats with light grey badges
        html_content = re.sub(r'\b(\.txt|\.csv|\.json|TXT|CSV|JSON)\b', 
                              r'<span class="file-format">\1</span>',
                              html_content)
        
        # Clean up any text that might still have "The X button" without styling
        html_content = re.sub(r'\bThe\s+"([^"]+)"\s+button', r'<span class="button-label">The <span class="button">\1</span> button</span>', html_content)
        
        # Prevent breaks in "Add X" actions
        html_content = re.sub(r'\b(Add\s+[A-Za-z\s]+)\b', r'<span class="ui-element">\1</span>', html_content)
        
        # Fix for missing styles on some elements
        html_content = re.sub(r'\b(option|button|file|format):',
                              r'<span style="font-style: normal;">\1</span>:',
                              html_content)
        
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
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header section with icon on left and title on right
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setSpacing(15)
        
        # Icon container (left side)
        icon_container = QWidget()
        icon_layout = QVBoxLayout(icon_container)
        icon_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_container.setMinimumHeight(110)  # Ensure enough vertical space for the icon
        
        # Load and display the PNG icon directly
        icon_path = Path(__file__).parent.parent / "resources" / "app_icon.png"
        icon_label = QLabel()
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        if icon_path.exists():
            # Simply load and display the PNG
            pixmap = QPixmap(str(icon_path))
            pixmap = pixmap.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            icon_label.setPixmap(pixmap)
        else:
            # Fallback if icon loading fails
            icon_label.setText("ANPE")
            icon_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #005A9C;")
        
        icon_layout.addWidget(icon_label)
        header_layout.addWidget(icon_container)
        
        # Title container (right side)
        title_container = QWidget()
        title_layout = QVBoxLayout(title_container)
        title_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        
        # Title
        title_label = QLabel("ANPE")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #005A9C;")
        title_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Another Noun Phrase Extractor")
        subtitle_label.setStyleSheet("font-size: 16px; color: #666666;")
        title_layout.addWidget(subtitle_label)
        
        header_layout.addWidget(title_container, 1)  # Give the title section more stretch
        layout.addWidget(header_widget)
        
        # Add a separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setStyleSheet("background-color: #ddd; max-height: 1px;")
        layout.addWidget(separator)
        
        # Version information - Now in a table-like layout with better alignment
        version_widget = QWidget()
        version_layout = QGridLayout(version_widget)
        version_layout.setSpacing(8)  # Increased spacing
        version_layout.setContentsMargins(10, 10, 10, 10)
        
        gui_label = QLabel("GUI Version:")
        gui_label.setStyleSheet("font-weight: bold;")
        gui_label.setFixedWidth(120)  # Fixed width for alignment
        gui_value = QLabel(self.gui_version)
        
        core_label = QLabel("Core Version:")
        core_label.setStyleSheet("font-weight: bold;")
        core_label.setFixedWidth(120)  # Fixed width for alignment
        core_value = QLabel(self.core_version)
        
        version_layout.addWidget(gui_label, 0, 0, Qt.AlignmentFlag.AlignLeft)
        version_layout.addWidget(gui_value, 0, 1, Qt.AlignmentFlag.AlignLeft)
        version_layout.addWidget(core_label, 1, 0, Qt.AlignmentFlag.AlignLeft)
        version_layout.addWidget(core_value, 1, 1, Qt.AlignmentFlag.AlignLeft)
        version_layout.setColumnStretch(1, 1)  # Give more stretch to value column
        
        layout.addWidget(version_widget)
        
        # Another separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        separator2.setStyleSheet("background-color: #ddd; max-height: 1px;")
        layout.addWidget(separator2)
        
        # Description section
        desc_widget = QWidget()
        desc_layout = QVBoxLayout(desc_widget)
        desc_layout.setContentsMargins(0, 0, 0, 0)
        desc_layout.setSpacing(10)
        
        desc_title = QLabel("Description")
        desc_title.setStyleSheet(
            f"font-size: 16px; color: #005A9C; font-weight: bold; padding-bottom: 4px;"
        )
        desc_layout.addWidget(desc_title)
        
        desc_content = QLabel(
            "ANPE GUI is a user-friendly graphical interface designed to simplify the extraction "
            "of noun phrases from text. It provides an accessible way to perform NP extraction "
            "without requiring programming knowledge."
        )
        desc_content.setWordWrap(True)
        desc_layout.addWidget(desc_content)
        
        layout.addWidget(desc_widget)
        
        # Built with section
        built_widget = QWidget()
        built_layout = QVBoxLayout(built_widget)
        built_layout.setContentsMargins(0, 0, 0, 0)
        built_layout.setSpacing(10)
        
        built_title = QLabel("Built with")
        built_title.setStyleSheet(
            f"font-size: 16px; color: #005A9C; font-weight: bold; padding-bottom: 4px;"
        )
        built_layout.addWidget(built_title)
        
        bullet_list = QLabel(
            "• <b>Benepar</b> - Berkeley Neural Parser, the core constituency parsing engine<br>"
            "• <b>spaCy</b> - Industrial-strength Natural Language Processing<br>"
            "• <b>NLTK</b> - Natural Language Toolkit for text processing"
        )
        bullet_list.setWordWrap(True)
        bullet_list.setTextFormat(Qt.TextFormat.RichText)
        bullet_list.setContentsMargins(20, 0, 0, 0)
        built_layout.addWidget(bullet_list)
        
        layout.addWidget(built_widget)
        
        # Acknowledgments section
        ack_widget = QWidget()
        ack_layout = QVBoxLayout(ack_widget)
        ack_layout.setContentsMargins(0, 0, 0, 0)
        ack_layout.setSpacing(10)
        
        ack_title = QLabel("Acknowledgments")
        ack_title.setStyleSheet(
            f"font-size: 16px; color: #005A9C; font-weight: bold; padding-bottom: 4px;"
        )
        ack_layout.addWidget(ack_title)
        
        ack_content = QLabel(
            "This project builds upon the excellent work of the Berkeley Neural Parser (Benepar), "
            "spaCy, and NLTK communities. Special thanks to the developers of these "
            "open source projects that made ANPE possible.\n\n"
            "ANPE is distributed as open source software under the MIT License, allowing "
            "for free use, modification, and distribution."
        )
        ack_content.setWordWrap(True)
        ack_layout.addWidget(ack_content)
        
        layout.addWidget(ack_widget)
        
        # Author info
        author_widget = QWidget()
        author_layout = QGridLayout(author_widget)
        author_layout.setContentsMargins(0, 10, 0, 0)
        author_layout.setSpacing(8)
        
        author_label = QLabel("Author:")
        author_label.setStyleSheet("font-weight: bold;")
        author_label.setFixedWidth(120)  # Fixed width for alignment
        author_value = QLabel("Richard Chen (@rcverse)")
        
        license_label = QLabel("License:")
        license_label.setStyleSheet("font-weight: bold;")
        license_label.setFixedWidth(120)  # Fixed width for alignment
        license_value = QLabel("MIT License")

        email_label = QLabel("Contact:")
        email_label.setStyleSheet("font-weight: bold;")
        email_label.setFixedWidth(120)  # Fixed width for alignment
        email_value = QLabel('<a href="mailto:rcverse6@gmail.com">rcverse6@gmail.com</a>')
        email_value.setTextFormat(Qt.TextFormat.RichText)
        email_value.setOpenExternalLinks(True)
        
        author_layout.addWidget(author_label, 0, 0, Qt.AlignmentFlag.AlignLeft)
        author_layout.addWidget(author_value, 0, 1, Qt.AlignmentFlag.AlignLeft)
        author_layout.addWidget(license_label, 1, 0, Qt.AlignmentFlag.AlignLeft)
        author_layout.addWidget(license_value, 1, 1, Qt.AlignmentFlag.AlignLeft)
        author_layout.addWidget(email_label, 2, 0, Qt.AlignmentFlag.AlignLeft)
        author_layout.addWidget(email_value, 2, 1, Qt.AlignmentFlag.AlignLeft)
        author_layout.setColumnStretch(1, 1)  # Give more stretch to value column
        
        layout.addWidget(author_widget)
        
        # Project page link
        link_widget = QWidget()
        link_layout = QHBoxLayout(link_widget)
        link_layout.setContentsMargins(0, 10, 0, 0)
        
        project_page_button = QPushButton("Visit Project Page")
        project_page_button.setStyleSheet(
            "QPushButton { color: #005A9C; background-color: transparent; border: 1px solid #005A9C; "
            "border-radius: 3px; padding: 5px 10px; }"
            "QPushButton:hover { background-color: #e0e0e0; }"
        )
        project_page_button.clicked.connect(self._visit_project_page)
        link_layout.addWidget(project_page_button)
        link_layout.addStretch()
        
        layout.addWidget(link_widget)
        
        # OK button
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        ok_button = QPushButton("OK")
        ok_button.setFixedWidth(80)
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
    
    def _visit_project_page(self):
        """Open project page in browser."""
        QDesktopServices.openUrl(QUrl("https://github.com/rcverse/Another-Noun-Phrase-Extractor")) 