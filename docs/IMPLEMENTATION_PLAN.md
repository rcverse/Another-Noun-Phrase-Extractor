# ANPE GUI Implementation Plan

## Phase 1: Technical Foundation

### 1.1. Module Structure
- Fix Python module imports and package structure
- Update package imports to ensure proper loading
- Create consistent directory structure and necessary `__init__.py` files

### 1.2. Basic Code Cleanup
- Standardize naming conventions across codebase
- Remove redundant code and consolidate duplicate functions
- Separate UI components into reusable widget classes

## Phase 2: GUI Overhaul

### 2.1. Two-Tab Main Structure
- Implement "Input" and "Output" tabs with QTabWidget
- Add horizontal splitter for main content and log panel
- Create header with "ANPE" and "Another Noun Phrase Extractor" subtitle

### 2.2. Input Tab Implementation
- Create toggle buttons for File/Text input modes
- Implement file selection list widget with action buttons
- Design configuration sections with proper headers and spacing
- Create master toggle for structure filtering
- Implement QSpinBox for min/max length inputs

### 2.3. Output Tab Implementation
- Create results display area with proper formatting
- Implement file selector dropdown for multiple files
- Add action buttons for navigation and export

### 2.4. Status Bar Implementation
- Create global status bar at the bottom of the application window
- Implement progress bar component with text status display
- Connect status updates to all processing operations
- Ensure status bar appears beneath the log panel
- Add clear status method for operation completion

### 2.5. Splash Screen Fix
- Resize splash screen to appropriate dimensions
- Fix progress bar positioning
- Update text to match application title

## Phase 3: Functional Improvements

### 3.1. Processing Logic
- Implement background worker thread for processing
- Add progress reporting to status bar during processing
- Create automatic tab switching after processing
- Preserve all core ANPE functionality

### 3.2. Batch Processing Enhancement
- Implement file selector for navigating results
- Create proper progress reporting during batch processing
- Add batch status indicators
- Update status bar with file counts and progress

### 3.3. Log Panel Improvement
- Add log filtering by level
- Implement Clear and Copy buttons
- Create proper log capturing from ANPE core
- Add color coding based on log severity

## Phase 4: Testing and Refinement

### 4.1. Cross-Platform Testing
- Test on Windows, macOS, and Linux
- Fix platform-specific display issues
- Ensure consistent behavior across environments

### 4.2. User Experience Polish
- Add tooltips for interactive elements
- Improve error handling and user feedback
- Ensure keyboard navigation works properly
- Fix any remaining alignment or spacing issues

## Implementation Files

### New Files
- `anpe_gui/widgets/file_list_widget.py`: File selection list
- `anpe_gui/widgets/structure_filter_widget.py`: Structure filter grid
- `anpe_gui/widgets/enhanced_log_panel.py`: Improved log panel
- `anpe_gui/widgets/status_bar.py`: Status bar with progress
- `anpe_gui/utils/style_constants.py`: UI styling constants

### Modified Files
- `anpe_gui/main_window.py`: Two-tab UI implementation
- `anpe_gui/splash_screen.py`: Fixed splash screen
- `anpe_gui/app.py`: Application startup and theme
- `anpe_gui/widgets/qt_log_handler.py`: Enhanced log handling

IMPORTANT: All modifications must preserve the existing core functionality that communicates with the ANPE library. 