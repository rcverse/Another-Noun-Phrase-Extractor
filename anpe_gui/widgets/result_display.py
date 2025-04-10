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
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QSizePolicy
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
        self.length_value = 0 # Initialize length_value

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
            length_val = 0 # Default length value
            
            metadata = np_item.get("metadata", {})
            if metadata:
                if "length" in metadata:
                    length_str = str(metadata['length'])
                    length_val = metadata['length']
                if "structures" in metadata:
                    structs = metadata.get('structures', [])
                    # Just the comma-separated list for the column
                    structures_str = ", ".join(map(str, structs)) if structs else ""
            
            # Create item data list for the four columns
            # Store raw length value alongside string representation if needed,
            # but the current model data() method handles conversion.
            item_data = [np_id, np_text, length_str, structures_str] 
            new_item = NpTreeItem(item_data, parent_item)
            # Store the raw integer length directly on the item for easier access in data()
            new_item.length_value = length_val # Assign potentially updated value
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
        
        # --- Top Layout (Filter + Sort Buttons) ---
        top_layout = QHBoxLayout()
        top_layout.setSpacing(5)
        
        # --- Filter Input ---
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search results...")
        self.filter_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed) # Allow horizontal expansion
        self.filter_input.textChanged.connect(self.update_filter)
        top_layout.addWidget(self.filter_input) # Add to top layout

        # --- Sorting Buttons ---
        self.sort_order_button = QPushButton("Sort by Order")
        self.sort_order_button.setToolTip("Reset sorting to the original order of appearance.")
        self.sort_order_button.clicked.connect(self._sort_by_order)
        self.sort_order_button.setVisible(False) # Initially hidden
        top_layout.addWidget(self.sort_order_button) # Add to top layout

        self.sort_length_button = QPushButton("Sort by Length")
        self.sort_length_button.setToolTip("Sort results by noun phrase length (click again to reverse order).")
        self.sort_length_button.clicked.connect(self._sort_by_length)
        self.sort_length_button.setVisible(False) # Initially hidden
        top_layout.addWidget(self.sort_length_button) # Add to top layout

        self.sort_structure_button = QPushButton("Sort by Structure")
        self.sort_structure_button.setToolTip("Sort results alphabetically by structure type (click again to reverse order).")
        self.sort_structure_button.clicked.connect(self._sort_by_structure)
        self.sort_structure_button.setVisible(False) # Initially hidden
        top_layout.addWidget(self.sort_structure_button) # Add to top layout
        
        layout.addLayout(top_layout) # Add combined layout to main layout

        # --- Tree View --- 
        self.tree_view = QTreeView()
        self.tree_view.setObjectName("ResultsTreeView")
        self.tree_view.setMinimumHeight(300)
        self.tree_view.setHeaderHidden(False) 
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree_view.setSortingEnabled(True) # Enable sorting
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
        # Hide buttons when display is cleared
        self.sort_order_button.setVisible(False)
        self.sort_length_button.setVisible(False)
        self.sort_structure_button.setVisible(False)
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

    def display_results(self, data: Dict[str, Any], metadata_enabled: bool = True):
        """Create models and display results, optionally hiding metadata columns/buttons."""
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
            # Reset sorting when new data is loaded to default (order)
            self.proxy_model.sort(-1, Qt.SortOrder.AscendingOrder) 
            
            # --- Show/Hide Columns based on metadata flag ---
            self.tree_view.setColumnHidden(AnpeResultModel.COL_LEN, not metadata_enabled)
            self.tree_view.setColumnHidden(AnpeResultModel.COL_STRUCT, not metadata_enabled)
            
            # --- Configure Header ---
            header = self.tree_view.header()
            header.setSectionsClickable(False) # Disable header clicking for sorting
            # Make Structures stretch to fill available space
            if metadata_enabled:
                header.setSectionResizeMode(AnpeResultModel.COL_STRUCT, QHeaderView.ResizeMode.Stretch)
            # Make other columns interactive (resizable by user)
            header.setSectionResizeMode(AnpeResultModel.COL_ID, QHeaderView.ResizeMode.Interactive)
            header.setSectionResizeMode(AnpeResultModel.COL_NP, QHeaderView.ResizeMode.Interactive)
            if metadata_enabled:
                header.setSectionResizeMode(AnpeResultModel.COL_LEN, QHeaderView.ResizeMode.Interactive)
            
            # Allow sorting indicators on headers (always enabled, but only relevant columns are shown)
            header.setSortIndicatorShown(False) # Disable visual indicator

            # Set fixed initial widths for ID, NP, and Length. 
            # Structures stretches to fill the rest.
            header.resizeSection(AnpeResultModel.COL_ID, 50)  # Set fixed initial width for ID
            if metadata_enabled: # Only set NP width if Structures is also present
                header.resizeSection(AnpeResultModel.COL_NP, 400) # Set larger fixed initial width for NP
            if metadata_enabled:
                header.resizeSection(AnpeResultModel.COL_LEN, 60) # Slightly wider for Length + indicator
            # ---------------------------------------------

            # Collapse all items by default
            self.tree_view.collapseAll()

            # Update filter based on current text (if any)
            # self.tree_view.expandToDepth(0) # Remove default expansion
            self.update_filter(self.filter_input.text())

            # Enable sorting buttons now that data is present
            # Show relevant buttons based on metadata flag
            self.sort_order_button.setVisible(metadata_enabled)
            self.sort_length_button.setVisible(metadata_enabled)
            self.sort_structure_button.setVisible(metadata_enabled)
            self._update_button_styles() # Set initial button style

            logging.debug("Results display updated with sorting enabled.")

        except Exception as e:
             logging.error(f"Error during minimal results display setup: {e}", exc_info=True)
             self.clear_display() 

    def _update_button_styles(self):
        """Updates button appearance based on the current sort column and order."""
        if not self.proxy_model or not self.sort_order_button.isVisible():
            return
             
        sort_col = self.proxy_model.sortColumn()
        arrow = " ↑" if self.proxy_model.sortOrder() == Qt.SortOrder.AscendingOrder else " ↓"
        
        if sort_col == -1: # Sorted by Order (default)
            self.sort_order_button.setProperty("activeSort", True)
            self.sort_order_button.setText("Sort by Order" + arrow) # Indicate default is usually asc
            if self.sort_length_button.isVisible():
               self.sort_length_button.setText("Sort by Length")
            if self.sort_structure_button.isVisible():
               self.sort_structure_button.setText("Sort by Structure")
        elif sort_col == AnpeResultModel.COL_LEN:
            self.sort_order_button.setText("Sort by Order")
            if self.sort_length_button.isVisible():
               self.sort_length_button.setProperty("activeSort", True)
               self.sort_length_button.setText("Sort by Length" + arrow)
            if self.sort_structure_button.isVisible():
               self.sort_structure_button.setText("Sort by Structure")
        elif sort_col == AnpeResultModel.COL_STRUCT:
            self.sort_order_button.setText("Sort by Order")
            if self.sort_length_button.isVisible():
               self.sort_length_button.setText("Sort by Length")
            if self.sort_structure_button.isVisible():
               self.sort_structure_button.setProperty("activeSort", True)
               self.sort_structure_button.setText("Sort by Structure" + arrow)
        else: # Should not happen with current setup, but reset just in case
            self.sort_order_button.setText("Sort by Order")
            if self.sort_length_button.isVisible():
               self.sort_length_button.setText("Sort by Length")
            if self.sort_structure_button.isVisible():
               self.sort_structure_button.setText("Sort by Structure")

        # Re-apply stylesheet to update appearance based on property
        # This might require a theme.py or centralized styling approach

    # --- Sorting Slots ---
    def _sort_by_order(self):
        """Sorts the results by the original order (resets sorting)."""
        if self.proxy_model:
            self.proxy_model.sort(-1) # -1 resets sorting to source model order
            logging.debug("Sorted by original order.")
        else:
            logging.warning("Attempted to sort by order but proxy model is not set.")

    def _sort_by_length(self):
        """Sorts the results by the Length column."""
        if self.proxy_model:
            # Check current sort column and toggle order
            current_col = self.proxy_model.sortColumn()
            current_order = self.proxy_model.sortOrder()
            new_order = Qt.SortOrder.AscendingOrder
            if current_col == AnpeResultModel.COL_LEN and current_order == Qt.SortOrder.AscendingOrder:
                new_order = Qt.SortOrder.DescendingOrder
                
            self.proxy_model.sort(AnpeResultModel.COL_LEN, new_order)
            logging.debug(f"Sorted by Length ({new_order}).")
        else:
            logging.warning("Attempted to sort by length but proxy model is not set.")

    def _sort_by_structure(self):
        """Sorts the results by the Structures column."""
        if self.proxy_model:
            # Check current sort column and toggle order
            current_col = self.proxy_model.sortColumn()
            current_order = self.proxy_model.sortOrder()
            new_order = Qt.SortOrder.AscendingOrder
            if current_col == AnpeResultModel.COL_STRUCT and current_order == Qt.SortOrder.AscendingOrder:
                new_order = Qt.SortOrder.DescendingOrder

            self.proxy_model.sort(AnpeResultModel.COL_STRUCT, new_order)
            logging.debug(f"Sorted by Structure ({new_order}).")
        else:
            logging.warning("Attempted to sort by structure but proxy model is not set.")

    # --- Potential Future Enhancements ---
    # ...

# Remove the extraneous tag below this line 