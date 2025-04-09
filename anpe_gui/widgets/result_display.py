"""
Widget for displaying formatted ANPE extraction results.
"""

import logging
from typing import Dict, List, Any
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QTreeView,
    QAbstractItemView,
    QHeaderView,
    QLineEdit
)
from PyQt6.QtGui import QColor, QFont, QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, QAbstractItemModel, QModelIndex, QVariant, QSortFilterProxyModel, QRegularExpression

# Attempt relative import first, then absolute
try:
    from ..theme import get_scroll_bar_style
except ImportError:
    try:
        from anpe_gui.theme import get_scroll_bar_style
    except ImportError:
        logging.warning("Could not import get_scroll_bar_style. Scrollbar styling may be missing.")
        def get_scroll_bar_style(): # Dummy function
            return ""

# --- Placeholder/Structure for Tree Item --- 
class NpTreeItem:
    """A node in the Noun Phrase Tree Model."""
    def __init__(self, data, parent=None):
        self.parent_item = parent
        self.item_data = data # List of strings for columns
        self.child_items = []

    def appendChild(self, item):
        self.child_items.append(item)

    def child(self, row):
        if row < 0 or row >= len(self.child_items):
            return None
        return self.child_items[row]

    def childCount(self):
        return len(self.child_items)

    def columnCount(self):
        # Should match model's columnCount
        return len(self.item_data) 

    def data(self, column):
        try:
            return self.item_data[column]
        except IndexError:
            return None

    def parent(self):
        return self.parent_item

    def row(self):
        if self.parent_item:
            return self.parent_item.child_items.index(self)
        return 0

# --- Placeholder/Structure for Tree Model --- 
class AnpeResultModel(QAbstractItemModel):
    """Provides data from ANPE results to the QTreeView."""
    COL_ID = 0
    COL_NP = 1
    COL_LEN = 2
    COL_STRUCT = 3
    NUM_COLUMNS = 4
    
    def __init__(self, result_data, parent=None):
        super().__init__(parent)
        # Updated header data for four columns
        self.root_item = NpTreeItem(["ID", "Noun Phrase", "Length", "Structures"])
        self.setupModelData(result_data, self.root_item)
        
    def setupModelData(self, result_data, parent_node):
        """Parse the ANPE result dictionary and build the tree structure."""
        # TODO: Implement parsing logic
        if not result_data or 'results' not in result_data:
            return # No data to parse
        
        np_results = result_data.get('results', [])
        self._recursive_setup(np_results, parent_node)

    def _recursive_setup(self, np_items, parent_item):
        """Helper to recursively build the tree from NP items for four columns."""
        for np_item in np_items:
            # Extract data
            np_text = np_item.get('noun_phrase', 'N/A')
            # Get ID as string without brackets and strip whitespace
            np_id = str(np_item.get('id', 'N/A')).strip()
            length_str = ""
            structures_str = ""
            
            metadata = np_item.get("metadata", {})
            if metadata:
                if "length" in metadata:
                    length_str = str(metadata['length'])
                if "structures" in metadata:
                    structs = metadata.get('structures', [])
                    # Just the comma-separated list for the column
                    structures_str = ", ".join(map(str, structs)) if structs else ""
            
            # Create item data list for the four columns
            item_data = [np_id, np_text, length_str, structures_str]
            new_item = NpTreeItem(item_data, parent_item)
            parent_item.appendChild(new_item)
            
            children = np_item.get("children")
            if children and isinstance(children, list):
                self._recursive_setup(children, new_item)

    def rowCount(self, parent=QModelIndex()):
        if parent.column() > 0:
            return 0
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        return parent_item.childCount()

    def columnCount(self, parent=QModelIndex()):
        # Return the defined number of columns
        return self.NUM_COLUMNS

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()

        item = index.internalPointer()
        column = index.column()
        
        # --- Font Settings ---
        default_font = QFont("Segoe UI", 10) # Use Segoe UI, size 10

        # --- Display Role --- 
        if role == Qt.ItemDataRole.DisplayRole:
            return QVariant(item.data(column))
        
        # --- Font Role --- 
        elif role == Qt.ItemDataRole.FontRole:
             # Apply the default font to all columns
             # Remove previous bolding logic
             return QVariant(default_font)
                 
        # --- Foreground (Text Color) Role --- 
        elif role == Qt.ItemDataRole.ForegroundRole:
            if column == self.COL_ID:
                 return QVariant(QColor("#005fb8")) 
            elif column == self.COL_NP:
                 return QVariant(QColor("#000000"))
            elif column == self.COL_LEN or column == self.COL_STRUCT:
                 return QVariant(QColor("#666666"))
        
        # --- Text Alignment Role --- 
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            # Center align the Length column
            if column == self.COL_LEN:
                return QVariant(int(Qt.AlignmentFlag.AlignCenter))
            # ID column will default to left alignment
        
        # --- Tooltip Role (Example) --- 
        elif role == Qt.ItemDataRole.ToolTipRole:
            if column == self.COL_STRUCT:
                # Show full structure list as tooltip if it might be cut off
                struct_data = item.data(column)
                if struct_data:
                    return QVariant(f"Structures: {struct_data}")
            elif column == self.COL_NP:
                # Show full NP as tooltip
                np_data = item.data(column)
                if np_data:
                    return QVariant(np_data)

        # TODO: Add roles for background color (for structure labels - requires delegate)

        return QVariant()

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal:
            if role == Qt.ItemDataRole.DisplayRole:
                # Return header data based on column index (section)
                if section < self.root_item.columnCount():
                     return QVariant(self.root_item.data(section))
            elif role == Qt.ItemDataRole.TextAlignmentRole:
                 # Center align the Length column header
                 if section == self.COL_LEN:
                     return QVariant(int(Qt.AlignmentFlag.AlignCenter))
                 # ID header will default to left alignment
        return QVariant()

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()

        child_item = parent_item.child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        else:
            return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        child_item = index.internalPointer()
        parent_item = child_item.parent()

        if parent_item == self.root_item:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable

# --- Result Display Widget --- 
class ResultDisplayWidget(QWidget):
    """A widget dedicated to displaying ANPE results using a QTreeView."""

    PLACEHOLDER_TEXT = "Process files or texts to view the extraction results."

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("ResultDisplayWidget") 
        self._setup_ui()
        self.source_model = None # To hold the original AnpeResultModel
        self.proxy_model = None # To hold the QSortFilterProxyModel

    def _setup_ui(self):
        """Initialize the UI components of the widget."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5) # Add some spacing between filter and tree
        
        # --- Filter Input --- 
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter results (e.g., by Noun Phrase or Structure)...")
        self.filter_input.textChanged.connect(self.update_filter)
        layout.addWidget(self.filter_input)

        # --- Tree View --- 
        self.tree_view = QTreeView()
        self.tree_view.setObjectName("ResultsTreeView")
        self.tree_view.setMinimumHeight(300)
        self.tree_view.setHeaderHidden(False) 
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree_view.setSortingEnabled(False) # Disable sorting
        self.tree_view.setIndentation(10) # Increase indentation further
        
        # Apply Item Padding, Branch Indicator Styling, and Header Padding
        # Note: Using forward slashes for paths in stylesheets, even on Windows
        self.tree_view.setStyleSheet(f"""
            /* TreeView Item Styles */
            QTreeView::item {{ 
                padding-top: 3px; 
                padding-bottom: 3px; 
            }}
            QTreeView::item:selected {{ 
                /* Ensure selection style doesn't remove padding */ 
                padding-top: 3px; 
                padding-bottom: 3px; 
            }}
            /* TreeView Branch Indicator Styles */
            QTreeView::branch:open:has-children {{
                image: url(anpe_gui/resources/expand_open.svg);
            }}
            QTreeView::branch:closed:has-children {{
                image: url(anpe_gui/resources/expand_close.svg);
            }}
            /* Header Section Padding */
            QHeaderView::section:horizontal:first {{
                padding-left: 10px; /* Add left padding to the first header section (ID) */
            }}
            /* Include scrollbar style */
            {get_scroll_bar_style()}
            """)
        
        layout.addWidget(self.tree_view)

    def clear_display(self):
        """Clear the results display area by removing the model."""
        self.tree_view.setModel(None) # Remove the model
        self.source_model = None # Clear reference
        self.proxy_model = None # Clear reference
        logging.debug("Results display cleared.")

    def set_placeholder_text(self, text: str):
         """Set the placeholder text (Not directly applicable to QTreeView)."""
         logging.warning(f"set_placeholder_text('{text}') ignored; QTreeView used.")

    def update_filter(self, text):
        """Slot to update the filter on the proxy model."""
        if self.proxy_model:
            # Use QRegularExpression for case-insensitive filtering
            regex = QRegularExpression(text, QRegularExpression.PatternOption.CaseInsensitiveOption)
            self.proxy_model.setFilterRegularExpression(regex)
            logging.debug(f"Filter updated to: '{text}'")
        else:
            logging.warning("Attempted to filter but proxy model is not set.")

    def display_results(self, data: Dict[str, Any]):
        """Create models and display results, with sorting disabled."""
        if not data or not isinstance(data, dict) or 'metadata' not in data or 'results' not in data:
            self.clear_display() 
            logging.warning(f"Invalid or empty data passed to display_results: {data}")
            return

        try:
            logging.debug("Creating source model...")
            self.source_model = AnpeResultModel(data)
            
            logging.debug("Creating standard proxy model...")
            # Use the standard QSortFilterProxyModel
            self.proxy_model = QSortFilterProxyModel() 
            self.proxy_model.setSourceModel(self.source_model)
            self.proxy_model.setFilterKeyColumn(AnpeResultModel.COL_NP) # Filter by NP
            self.proxy_model.setRecursiveFilteringEnabled(True)
            self.proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            
            logging.debug("Setting model on tree view...")
            self.tree_view.setModel(self.proxy_model)
            
            # --- Configure Header ---
            header = self.tree_view.header()
            # Make Structures stretch to fill available space
            header.setSectionResizeMode(AnpeResultModel.COL_STRUCT, QHeaderView.ResizeMode.Stretch)
            # Make other columns interactive (resizable by user)
            header.setSectionResizeMode(AnpeResultModel.COL_ID, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(AnpeResultModel.COL_NP, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(AnpeResultModel.COL_LEN, QHeaderView.ResizeMode.Interactive)
            # StretchLastSection is implicitly handled by the Stretch setting on Structures column
            # header.setStretchLastSection(False) 
            # Remove sort indicator display
            # header.setSortIndicatorShown(True)
            
            # Set fixed initial widths for ID, NP, and Length. 
            # Structures stretches to fill the rest.
            header.resizeSection(AnpeResultModel.COL_ID, 50)  # Set fixed initial width for ID
            header.resizeSection(AnpeResultModel.COL_NP, 400) # Set larger fixed initial width for NP
            header.resizeSection(AnpeResultModel.COL_LEN, 50) # Set fixed initial width for Length
            # self.tree_view.resizeColumnToContents(AnpeResultModel.COL_STRUCT) # Don't set initial width for stretching column
            # ---------------------------------------------

            # Collapse all items by default
            self.tree_view.collapseAll()

            # Update filter based on current text (if any)
            # self.tree_view.expandToDepth(0) # Remove default expansion
            self.update_filter(self.filter_input.text())

            logging.debug("Results display updated: ID stretches, others interactive, Length centered, no indent, collapsed default.")

        except Exception as e:
             logging.error(f"Error during minimal results display setup: {e}", exc_info=True)
             self.clear_display() 

    # def format_np_for_display(...): # Removed - logic moved to model

    # --- Potential Future Enhancements ---
    # ...

# Remove the extraneous tag below this line 