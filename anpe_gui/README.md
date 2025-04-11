# ANPE GUI - Developer Guide

A PyQt6-based graphical user interface for [ANPE (Another Noun Phrase Extractor)](https://github.com/rcverse/anpe), implemented using a modular architecture with separate workers for background processing.

## Architecture Overview

ANPE GUI follows a modular structure with careful separation of concerns:

- **UI Layer**: PyQt6-based widgets and windows
- **Processing Layer**: Background workers that run ANPE operations without blocking the UI
- **Data Models**: Classes for managing and transforming extraction results
- **Configuration**: Settings management and persistence

## Code Organization

```
anpe_gui/
├── __init__.py             # Package initialization
├── __main__.py             # Entry point for module execution
├── app.py                  # Application initialization and QApplication setup
├── main_window.py          # Main application window implementation
├── run.py                  # Convenience script for launching the application
├── setup_wizard.py         # First-run setup and model downloader
├── theme.py                # UI styling, colors, and theme constants
├── version.py              # Version information
├── splash_screen.py        # Application splash screen
├── docs/                   # Documentation files
│   ├── Help.md             # User help documentation
├── resources/              # UI assets (icons, images, stylesheets)
├── widgets/                # Reusable UI components
│   ├── input_widgets.py    # Text and file input widgets
│   ├── result_display.py   # Tree-based result visualization
│   ├── config_panel.py     # Configuration UI components
│   └── ...                 # Other widget modules
└── workers/                # Background processing classes
    ├── extractor.py        # ANPE processing worker
    ├── exporter.py         # Results export worker
    └── ...                 # Other worker modules
```

## Key Components

### Main Window (`main_window.py`)

The central controller that orchestrates all UI components and manages application flow. Implements the tab-based interface and coordinates between input, configuration, and output components.

### Widgets (`widgets/`)

Reusable UI components that encapsulate specific functionalities:

- **input_widgets.py**: Implements text input and file selection interfaces
- **config_panel.py**: UI for configuring ANPE extraction parameters
- **result_display.py**: Tree-based visualization of extraction results with filtering and sorting

### Workers (`workers/`)

Background processing components that run in separate threads to keep the UI responsive:

- **extractor.py**: Runs ANPE extraction operations without blocking the UI
- **exporter.py**: Handles saving results to different output formats

### Theme (`theme.py`)

Central styling definitions including colors, spacing, and stylesheet generators for consistent UI appearance.

## Development Workflow

### Setting Up Development Environment

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/rcverse/anpe.git
   cd anpe
   ```

2. **Install Dependencies:**
   ```bash
   pip install -e .  # Install ANPE in development mode
   cd anpe_gui
   pip install -r requirements.txt  # Install GUI dependencies
   ```

3. **Run in Development Mode:**
   ```bash
   python run.py
   ```

### Adding New Features

#### Extending the UI

1. Create or modify widget classes in the `widgets/` directory
2. Update the main window to integrate the new widget
3. Connect signals and slots for interaction

Example:
```python
# In a widget file (widgets/my_feature.py)
class MyFeatureWidget(QWidget):
    dataReady = pyqtSignal(object)  # Define custom signals
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        # Create and arrange UI components
        layout = QVBoxLayout(self)
        # ... add widgets to layout
```

## Building and Packaging

### Using PyInstaller

The application is packaged using PyInstaller with custom hooks to ensure all dependencies are properly included.

1. **Prepare for packaging:**
   ```bash
   cd anpe_gui
   ```

2. **Run the build script:**
   ```bash
   python setup_cx.py
   ```

### Platform-Specific Considerations

- **Windows**: Ensure that the `pywin32` package is installed
- **macOS**: Set up code signing if you plan to distribute the application
- **Linux**: Consider packaging as AppImage or distribution-specific package

### Customizing the Build

Edit `setup_cx.py` to modify:
- Application metadata
- Included/excluded modules
- Icon and resource handling
- Platform-specific optimizations

## Testing

The codebase doesn't currently include automated tests, but the recommended approach for implementing them would be:

1. Unit tests for individual components using `pytest-qt`
2. Integration tests for worker interactions
3. UI automation tests for end-to-end scenarios

## Contributing

1. **Coding Style**: Follow PEP 8 with consistent indentation (4 spaces)
2. **Documentation**: Document classes and complex functions with docstrings
3. **UI Design**: Maintain consistency with existing UI elements and theme
4. **Submit PRs**: Create pull requests with clear descriptions of changes

## Key Technical Implementations

### Resource Management

The application uses Qt's resource system to handle icons, images, and stylesheet resources:

- **ResourceManager**: A centralized class that provides access to resources both in development and packaged environments
- **Qt Resource Collection (QRC)**: Resources are defined in `resources.qrc` and compiled to binary `.rcc` files
- **Consistent Access**: All UI components access resources through the ResourceManager class rather than direct file paths
- **Fallback Mechanism**: If Qt resources aren't available, falls back to file-based resources

To compile resources:
- Windows: Run `compile_resources.bat`
- Unix/Mac: Run `compile_resources.sh`

### QTreeView-based Result Display

The result display (`widgets/result_display.py`) uses a custom tree model to visualize hierarchical noun phrase structures with filtering and sorting capabilities. It includes:

- Custom `AnpeResultModel` extending `QAbstractItemModel`
- Custom proxy model for numeric sorting
- Detachable window for enhanced viewing
- Tree traversal algorithms for data manipulation

### Background Processing

The application uses `QThread`-based workers to perform processing without blocking the UI:

- Signal-based communication between threads
- Progress reporting to keep the UI updated
- Error handling with user-friendly messages
- Resource management to prevent memory leaks

## License

This project is licensed under GPLv3 - see the [LICENSE](../LICENSE) file for details. 