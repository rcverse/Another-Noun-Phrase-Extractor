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
    QSizePolicy,
    QMainWindow,
    QDialog,
    QToolButton
)
from PyQt6.QtGui import QColor, QFont, QStandardItemModel, QStandardItem, QIcon, QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QAbstractItemModel, QModelIndex, QVariant, QSortFilterProxyModel, QRegularExpression, QSize
from anpe_gui.resource_manager import ResourceManager

# Attempt relative import first, then absolute
try:
    from ..theme import get_scroll_bar_style, LIGHT_HOVER_BLUE, PRIMARY_COLOR
except ImportError:
    try:
        from anpe_gui.theme import get_scroll_bar_style, LIGHT_HOVER_BLUE, PRIMARY_COLOR
    except ImportError:
        logging.warning("Could not import theme constants. Styling may be affected.")
        # Provide fallback values if necessary
        def get_scroll_bar_style(): return ""
        LIGHT_HOVER_BLUE = "#EFF5FB" # Example fallback
        PRIMARY_COLOR = "#005A9C"   # Example fallback

# --- Custom Sort Filter Proxy Model ---
class AnpeResultProxyModel(QSortFilterProxyModel):
    """Custom proxy model that provides proper numeric sorting for the Length column."""
    
    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:
        """Override to properly sort the Length column numerically instead of lexicographically."""
        source_model = self.sourceModel()
        column = left.column()
        
        # For Length column, compare numeric values
        if column == AnpeResultModel.COL_LEN:
            # Get the actual items from the source model
            left_item = source_model.data(left, Qt.ItemDataRole.UserRole)
            right_item = source_model.data(right, Qt.ItemDataRole.UserRole)
            
            # In PyQt6, get values properly from QVariant
            if left_item is not None and right_item is not None:
                left_value = 0
                right_value = 0
                
                # QVariant to int conversion for PyQt6
                if isinstance(left_item, QVariant) and left_item.isValid():
                    left_value = left_item.value() or 0
                elif isinstance(left_item, (int, float)):
                    left_value = left_item
                elif isinstance(left_item, str) and left_item.isdigit():
                    left_value = int(left_item)
                    
                if isinstance(right_item, QVariant) and right_item.isValid():
                    right_value = right_item.value() or 0
                elif isinstance(right_item, (int, float)):
                    right_value = right_item
                elif isinstance(right_item, str) and right_item.isdigit():
                    right_value = int(right_item)
                
                logging.debug(f"Comparing length values: {left_value} vs {right_value}")
                return left_value < right_value
                
        # For other columns, use default string comparison
        return super().lessThan(left, right)

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
                    length_val = int(metadata['length']) if isinstance(metadata['length'], (int, float, str)) else 0
                    logging.debug(f"Length value loaded: {length_val}, type: {type(length_val)}")
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
        
        # --- User Role (for Sorting) ---
        elif role == Qt.ItemDataRole.UserRole:
            # Provide numeric length value for sorting
            if column == self.COL_LEN:
                # Return raw integer value, not wrapped in QVariant
                return item.length_value
        
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

# --- Detached Result Window ---
class DetachedResultWindow(QMainWindow):
    """A standalone window for displaying extraction results."""
    
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("ANPE Results Viewer")
        self.resize(800, 600)  # Set default size
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Create filter input
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search results...")
        layout.addWidget(self.filter_input)
        
        # Create buttons layout
        buttons_layout = QHBoxLayout()
        
        # Sorting buttons
        self.sort_order_button = QPushButton("Sort by Order")
        self.sort_length_button = QPushButton("Sort by Length ↑")  # Default arrow
        self.sort_structure_button = QPushButton("Sort by Structure")
        
        buttons_layout.addWidget(self.sort_order_button)
        buttons_layout.addWidget(self.sort_length_button)
        buttons_layout.addWidget(self.sort_structure_button)
        buttons_layout.addStretch()
        
        layout.addLayout(buttons_layout)
        
        # Create tree view
        self.tree_view = QTreeView()
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree_view.setSortingEnabled(True)
        self.tree_view.setIndentation(12)
        self.tree_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.tree_view)
        
        # Apply the same styling as the embedded view
        expand_open_url = ResourceManager.get_style_url("expand_open.svg")
        expand_close_url = ResourceManager.get_style_url("expand_close.svg")
        
        self.tree_view.setStyleSheet(f"""
            QTreeView {{
                outline: none;
                background-color: white;
            }}
            QTreeView::item {{
                padding-top: 3px;
                padding-bottom: 3px;
            }}
            QTreeView::item:hover:!selected {{
                background-color: {LIGHT_HOVER_BLUE};
                color: {PRIMARY_COLOR};
            }}
            QTreeView::item:selected {{
                background-color: {PRIMARY_COLOR};
                color: white;
                padding-top: 3px;
                padding-bottom: 3px;
                border: none;
                outline: none;
            }}
            QTreeView::branch:open:has-children,
            QTreeView::branch:closed:has-children {{
                margin-right: 1px;
                background-color: transparent !important;
                border: none;
            }}
            QTreeView::branch:open:has-children {{
                image: url({expand_open_url});
                background-color: transparent;
            }}
            QTreeView::branch:closed:has-children {{
                image: url({expand_close_url});
                background-color: transparent;
            }}
            QTreeView::branch:!has-children {{
                background-color: transparent;
                border: none;
            }}
            QHeaderView::section:horizontal:first {{
                padding-left: 10px;
            }}
            {get_scroll_bar_style()}
        """)
        
        # Add keyboard shortcuts
        self.setup_shortcuts()
    
    def setup_shortcuts(self):
        """Setup keyboard shortcuts for the detached window."""
        # Add keyboard shortcut for expanding all items
        expand_shortcut = QShortcut(QKeySequence("Ctrl+E"), self)
        expand_shortcut.activated.connect(self.tree_view.expandAll)
        
        # Add keyboard shortcut for collapsing all items
        collapse_shortcut = QShortcut(QKeySequence("Ctrl+C"), self)
        collapse_shortcut.activated.connect(self.tree_view.collapseAll)
        
        # Add keyboard shortcut for search focus
        search_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        search_shortcut.activated.connect(self.filter_input.setFocus)
    
    def update_button_styles(self):
        """Updates arrow indicator on Sort by Length button based on sort order."""
        if not self.tree_view.model():
            return
        
        model = self.tree_view.model()
        sort_col = model.sortColumn()
        
        # Only change the arrow on the length button, based on sort state
        if sort_col == AnpeResultModel.COL_LEN:
            # Show up or down arrow based on sort order
            arrow = "↑" if model.sortOrder() == Qt.SortOrder.AscendingOrder else "↓"
            self.sort_length_button.setText(f"Sort by Length {arrow}")
        else:
            # Default to up arrow if not sorting by length
            self.sort_length_button.setText("Sort by Length ↑")

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
        self.detached_window = None # Reference to detached window when active

    def _setup_ui(self):
        """Initialize the UI components of the widget."""
        # Get resource URLs
        expand_open_url = ResourceManager.get_style_url("expand_open.svg")
        expand_close_url = ResourceManager.get_style_url("expand_close.svg")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5) # Add some spacing between filter and tree
        
        # --- Top Layout (Filter + Sort Buttons + Eject Button) ---
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

        self.sort_length_button = QPushButton("Sort by Length ↑")  # Default arrow
        self.sort_length_button.setToolTip("Sort results by noun phrase length (click again to reverse order).")
        self.sort_length_button.clicked.connect(self._sort_by_length)
        self.sort_length_button.setVisible(False) # Initially hidden
        top_layout.addWidget(self.sort_length_button) # Add to top layout

        self.sort_structure_button = QPushButton("Sort by Structure")
        self.sort_structure_button.setToolTip("Sort results alphabetically by structure type (click again to reverse order).")
        self.sort_structure_button.clicked.connect(self._sort_by_structure)
        self.sort_structure_button.setVisible(False) # Initially hidden
        top_layout.addWidget(self.sort_structure_button) # Add to top layout
        
        # --- Eject Button ---
        self.eject_button = QToolButton()
        self.eject_button.setText("⇱")  # Unicode external link symbol
        self.eject_button.setToolTip("Open results in a separate window")
        self.eject_button.clicked.connect(self._eject_results)
        self.eject_button.setVisible(False)  # Initially hidden like other buttons
        self.eject_button.setFixedSize(QSize(24, 24))  # Fixed size for the button
        top_layout.addWidget(self.eject_button)
        
        layout.addLayout(top_layout) # Add combined layout to main layout

        # --- Tree View --- 
        self.tree_view = QTreeView()
        self.tree_view.setObjectName("ResultsTreeView")
        self.tree_view.setMinimumHeight(300)
        self.tree_view.setHeaderHidden(False) 
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree_view.setSortingEnabled(True) # Enable sorting
        self.tree_view.setIndentation(12) 
        
        # Selection behavior for entire rows
        self.tree_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        
        # Connect the clicked signal to handle row expansion
        self.tree_view.clicked.connect(self._handle_item_click)
        
        # Apply Item Padding, Branch Indicator Styles, and Header Padding
        # Note: Using forward slashes for paths in stylesheets, even on Windows
        self.tree_view.setStyleSheet(f"""
            QTreeView {{
                outline: none; /* Remove focus rectangle from the view itself */
                background-color: white; /* Ensure white background */
            }}
            /* TreeView Item Styles */
            QTreeView::item {{
                padding-top: 3px;
                padding-bottom: 3px;
            }}
            QTreeView::item:hover:!selected {{
                background-color: {LIGHT_HOVER_BLUE}; /* Light blue background on hover */
                color: {PRIMARY_COLOR}; /* Use primary color text on hover */
            }}
            QTreeView::item:selected {{
                background-color: {PRIMARY_COLOR}; /* Primary color background when selected */
                color: white; /* White text when selected */
                /* Keep padding consistent */
                padding-top: 3px;
                padding-bottom: 3px;
                border: none; /* Remove border on selection */
                outline: none; /* Remove focus outline */
            }}
            /* TreeView Branch Indicator Styles */
            QTreeView::branch:open:has-children,
            QTreeView::branch:closed:has-children {{
                margin-right: 1px; /* Minimal space between arrow and text */
                background-color: transparent !important; /* Force transparency */
                border: none; /* Ensure no border */
            }}
            QTreeView::branch:open:has-children {{
                image: url({expand_open_url});
                background-color: transparent; /* Force transparency for open branch */
            }}
            QTreeView::branch:closed:has-children {{
                image: url({expand_close_url});
                background-color: transparent; /* Force transparency for closed branch */
            }}
            /* Explicitly style branches without children to ensure consistent appearance */
            QTreeView::branch:!has-children {{
                background-color: transparent; /* Force transparency for leaf item branches */
                border: none;
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
        self.eject_button.setVisible(False)
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
            
            logging.debug("Creating custom proxy model for numeric sorting...")
            # Use our custom QSortFilterProxyModel that handles numeric sorting
            self.proxy_model = AnpeResultProxyModel() 
            self.proxy_model.setSourceModel(self.source_model)
            self.proxy_model.setFilterKeyColumn(AnpeResultModel.COL_NP) # Filter by NP
            self.proxy_model.setRecursiveFilteringEnabled(True)
            self.proxy_model.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            
            logging.debug("Setting model on tree view...")
            self.tree_view.setModel(self.proxy_model)
            # Reset sorting when new data is loaded to default (order)
            # Important: -1 means unsorted (original order)
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
                header.resizeSection(AnpeResultModel.COL_NP, 350) # Set larger fixed initial width for NP
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
            # Show eject button whenever we have results
            self.eject_button.setVisible(True)
            self._update_button_styles() # Set initial button style

            logging.debug("Results display updated with sorting enabled.")

        except Exception as e:
             logging.error(f"Error during minimal results display setup: {e}", exc_info=True)
             self.clear_display() 

    def _update_button_styles(self):
        """Updates arrow indicator on Sort by Length button based on sort order."""
        if not self.proxy_model or not self.sort_order_button.isVisible():
            return
             
        sort_col = self.proxy_model.sortColumn()
        
        # Only change the arrow on the length button, based on sort state
        if sort_col == AnpeResultModel.COL_LEN:
            # Show up or down arrow based on sort order
            arrow = "↑" if self.proxy_model.sortOrder() == Qt.SortOrder.AscendingOrder else "↓"
            self.sort_length_button.setText(f"Sort by Length {arrow}")
        else:
            # Default to up arrow if not sorting by length
            self.sort_length_button.setText("Sort by Length ↑")

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
                
            logging.debug(f"Sorting by Length column ({AnpeResultModel.COL_LEN}) with order: {new_order}")
            self.proxy_model.sort(AnpeResultModel.COL_LEN, new_order)
            
            # Update button style after sort
            self._update_button_styles()
            logging.debug(f"Sort completed. Current sort column: {self.proxy_model.sortColumn()}")
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

    # --- Click Handling ---
    def _handle_item_click(self, index: QModelIndex):
        """Handle clicks on items, specifically to toggle expansion when a row is clicked."""
        if not index.isValid():
            return
        
        # Get the root index (first column) for this row regardless of which column was clicked
        root_index = self.proxy_model.index(index.row(), 0, index.parent())
        
        # For any valid click, if the item has children, toggle expansion
        # Important: We must use the proxy_model to properly handle children
        if self.proxy_model and self.proxy_model.hasChildren(root_index):
            # Toggle expansion state
            self.tree_view.setExpanded(root_index, not self.tree_view.isExpanded(root_index))
            logging.debug(f"Toggled expansion for item at row {root_index.row()} via row click.")

    def _eject_results(self):
        """Open the results in a separate, resizable window."""
        if not self.proxy_model:
            logging.warning("No results to display in detached window.")
            return
            
        # Create or show detached window
        if not self.detached_window:
            self.detached_window = DetachedResultWindow(self)
            
            # Connect signals for closing
            self.detached_window.destroyed.connect(self._on_detached_window_closed)
            
            # Copy filter, sorting, and data from current view
            if self.filter_input.text():
                self.detached_window.filter_input.setText(self.filter_input.text())
                
            # Connect signals
            self.detached_window.filter_input.textChanged.connect(self._update_detached_filter)
            self.detached_window.sort_order_button.clicked.connect(self._detached_sort_by_order)
            self.detached_window.sort_length_button.clicked.connect(self._detached_sort_by_length)
            self.detached_window.sort_structure_button.clicked.connect(self._detached_sort_by_structure)
            
            # Create new proxy model for detached window
            detached_proxy = AnpeResultProxyModel()
            detached_proxy.setSourceModel(self.source_model)
            detached_proxy.setFilterKeyColumn(AnpeResultModel.COL_NP)
            detached_proxy.setRecursiveFilteringEnabled(True)
            detached_proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            
            # Set model on detached tree view
            self.detached_window.tree_view.setModel(detached_proxy)
            
            # Default to sort by order - match the main view's sorting
            detached_proxy.sort(-1, Qt.SortOrder.AscendingOrder)
            
            # Update sort button styles
            self.detached_window.update_button_styles()
            
            # Configure header just like in the main window
            header = self.detached_window.tree_view.header()
            header.setSectionsClickable(False)
            
            # Check if metadata columns should be visible
            # Look at the main tree view to determine if metadata is enabled
            metadata_enabled = not self.tree_view.isColumnHidden(AnpeResultModel.COL_LEN)
            
            # Show/hide columns based on metadata_enabled flag
            self.detached_window.tree_view.setColumnHidden(AnpeResultModel.COL_LEN, not metadata_enabled)
            self.detached_window.tree_view.setColumnHidden(AnpeResultModel.COL_STRUCT, not metadata_enabled)
            
            # Show/hide buttons based on metadata_enabled flag
            self.detached_window.sort_order_button.setVisible(metadata_enabled)
            self.detached_window.sort_length_button.setVisible(metadata_enabled)
            self.detached_window.sort_structure_button.setVisible(metadata_enabled)
            
            # Only stretch the Structures column if it's visible
            if metadata_enabled:
                header.setSectionResizeMode(AnpeResultModel.COL_STRUCT, QHeaderView.ResizeMode.Stretch)
            else:
                # If no metadata, make the NP column stretch
                header.setSectionResizeMode(AnpeResultModel.COL_NP, QHeaderView.ResizeMode.Stretch)
            
            # Make other columns interactive (resizable by user)
            header.setSectionResizeMode(AnpeResultModel.COL_ID, QHeaderView.ResizeMode.Interactive)
            if not metadata_enabled:
                # If no metadata, ensure NP can stretch
                header.setSectionResizeMode(AnpeResultModel.COL_NP, QHeaderView.ResizeMode.Stretch)
            else:
                # With metadata, NP is resizable but not stretching
                header.setSectionResizeMode(AnpeResultModel.COL_NP, QHeaderView.ResizeMode.Interactive)
                header.setSectionResizeMode(AnpeResultModel.COL_LEN, QHeaderView.ResizeMode.Interactive)
                
            header.setSortIndicatorShown(False)
            
            # Set initial column widths
            header.resizeSection(AnpeResultModel.COL_ID, 50)
            header.resizeSection(AnpeResultModel.COL_NP, 350)
            if metadata_enabled:
                header.resizeSection(AnpeResultModel.COL_LEN, 60)
            
            # Apply current filter
            if self.filter_input.text():
                regex = QRegularExpression(self.filter_input.text(), 
                                          QRegularExpression.PatternOption.CaseInsensitiveOption)
                detached_proxy.setFilterRegularExpression(regex)
            
            # Connect click signal for expansion
            self.detached_window.tree_view.clicked.connect(self._handle_detached_item_click)
            
            # Collapse all items by default
            self.detached_window.tree_view.collapseAll()
            
            # Show the window
            self.detached_window.show()
            
        else:
            # Window exists, just show it and bring to front
            self.detached_window.show()
            self.detached_window.raise_()
            self.detached_window.activateWindow()
            
        logging.debug("Results displayed in detached window.")
    
    def _on_detached_window_closed(self):
        """Handle detached window being closed."""
        self.detached_window = None
        logging.debug("Detached results window closed.")
    
    # --- Detached Window Event Handlers ---
    def _update_detached_filter(self, text):
        """Update filter in the detached window."""
        if self.detached_window and self.detached_window.tree_view.model():
            regex = QRegularExpression(text, QRegularExpression.PatternOption.CaseInsensitiveOption)
            self.detached_window.tree_view.model().setFilterRegularExpression(regex)
    
    def _detached_sort_by_order(self):
        """Sort by order in detached window."""
        if self.detached_window and self.detached_window.tree_view.model():
            self.detached_window.tree_view.model().sort(-1)
            self.detached_window.update_button_styles()
    
    def _detached_sort_by_length(self):
        """Sort by length in detached window."""
        if self.detached_window and self.detached_window.tree_view.model():
            model = self.detached_window.tree_view.model()
            current_col = model.sortColumn()
            current_order = model.sortOrder()
            new_order = Qt.SortOrder.AscendingOrder
            if current_col == AnpeResultModel.COL_LEN and current_order == Qt.SortOrder.AscendingOrder:
                new_order = Qt.SortOrder.DescendingOrder
                
            logging.debug(f"Detached window: Sorting by Length column ({AnpeResultModel.COL_LEN}) with order: {new_order}")
            model.sort(AnpeResultModel.COL_LEN, new_order)
            self.detached_window.update_button_styles()
            logging.debug(f"Detached window: Sort completed. Current sort column: {model.sortColumn()}")
    
    def _detached_sort_by_structure(self):
        """Sort by structure in detached window."""
        if self.detached_window and self.detached_window.tree_view.model():
            model = self.detached_window.tree_view.model()
            current_col = model.sortColumn()
            current_order = model.sortOrder()
            new_order = Qt.SortOrder.AscendingOrder
            if current_col == AnpeResultModel.COL_STRUCT and current_order == Qt.SortOrder.AscendingOrder:
                new_order = Qt.SortOrder.DescendingOrder
            model.sort(AnpeResultModel.COL_STRUCT, new_order)
            self.detached_window.update_button_styles()
    
    def _handle_detached_item_click(self, index):
        """Handle clicks in the detached window tree view."""
        if not index.isValid():
            return
        
        model = self.detached_window.tree_view.model()
        root_index = model.index(index.row(), 0, index.parent())
        
        if model.hasChildren(root_index):
            self.detached_window.tree_view.setExpanded(
                root_index, 
                not self.detached_window.tree_view.isExpanded(root_index)
            )

# Remove the extraneous tag below this line 