# ANPE GUI Improvements Analysis

## Current Issues Analysis

### 1. Visual Layout and Alignment Issues
- **Input Controls**: Poor alignment and unknown squares next to inputs
- **Component Spacing**: Inconsistent spacing between elements
- **Structure Filter Layout**: Uneven grid layout for structure types

### 2. Splash Screen Issues
- **Size and Positioning**: Splash screen too large with overlapping elements
- **Progress Bar**: Incorrectly positioned progress bar

### 3. UI Structure Needs
- **Two-Tab System**: Need clear "Input" and "Output" tabs
- **Fixed Configuration Section**: Configuration should be directly visible (not collapsible)
- **Input Mode Toggles**: Need clear "File Input" and "Text Input" toggle buttons

### 4. Batch Processing Limitations
- **Progress Indication**: No progress bar during multi-file processing
- **File Results Selection**: No way to select between processed file results
- **Processing Feedback**: No consistent status updates during processing

### 5. Log Panel Improvements
- **Log Filtering**: Need file-level log filtering
- **Copy and Clear**: Need buttons for log management

### 6. Status Bar Requirement
- **Global Progress Bar**: Need a persistent progress bar beneath the log panel
- **Status Text**: Need text status tracking for all operations
- **Consistent Feedback**: Status updates should be shown for all processes

## Implementation Requirements

### 1. Visual Layout
- Create consistent tab structure with "Input" and "Output" tabs
- Implement fixed-width log panel on the right side
- Use proper input controls (QSpinBox for numeric inputs)
- Correct header with "ANPE" and "Another Noun Phrase Extractor" subtitle

### 2. Input Tab Features
- File/Text input mode toggle buttons
- File selection list with Add Files/Add Dir/Remove/Clear All buttons
- Properly laid out configuration sections with labeled headers
- Master toggle for structure filtering

### 3. Output Tab Features
- File selector dropdown when multiple files are processed
- Results display area with proper formatting
- Action buttons for returning to input or exporting results

### 4. Processing Improvements
- Progress bar during file processing
- Background worker thread to prevent UI freezing
- Direct tab switch from Input to Output when processing completes

### 5. Log Panel
- Add filter dropdown and Clear/Copy buttons
- Ensure proper log capture from core ANPE components
- Implement color coding based on log level

### 6. Status Bar Implementation
- Create global status bar at the bottom of the application
- Include both progress bar and text status indicator
- Ensure visibility during all processing operations
- Connect status updates to all worker processes

IMPORTANT: All improvements must preserve the existing core functionality that communicates with the ANPE library. 