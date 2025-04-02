# ANPE GUI Implementation Plan

## Phase 1: Technical Foundation & Module Structure

### 1.1. Fix Python Module Structure
- Create proper `__init__.py` files in all directories
- Update relative imports to absolute imports
- Fix module path issues in run scripts
- Expected outcome: Application launches without import errors

### 1.2. Basic Code Structure Cleanup
- Reorganize widget classes for better separation of concerns
- Standardize naming conventions across codebase
- Remove redundant code and consolidate duplicate functions
- Expected outcome: Cleaner, more maintainable codebase

## Phase 2: UI Component Refinement

### 2.1. Standardize Input Controls
- Replace custom input areas with Qt standard components:
  - Use QSpinBox for numeric inputs (min/max length)
  - Use QCheckBox with proper styling for toggles
  - Fix layout issues with unknown squares
- Implementation files:
  - `anpe_gui/widgets/standardized_inputs.py` (new)
  - Update references in `main_window.py`

### 2.2. Implement Grid Layout System
- Convert all form layouts to QGridLayout
- Establish consistent margin and spacing rules
- Align all labels and inputs properly
- Implementation files:
  - Update all widget layout methods in `main_window.py`
  - Create layout helper functions in `anpe_gui/utils/layout_helpers.py` (new)

### 2.3. Create Collapsible Panel Widget
- Implement a collapsible section widget for configuration options
- Include animation for smooth expand/collapse
- Add toggle button with arrow indicator
- Implementation files:
  - `anpe_gui/widgets/collapsible_section.py` (new)

## Phase 3: Workflow Restructuring

### 3.1. Two-Tab Main Interface
- Consolidate UI into two main tabs:
  - "Input & Settings" tab (with collapsible config section)
  - "Results" tab
- Update step indicator to reflect new structure
- Implementation files:
  - Major update to `main_window.py`
  - Update `step_indicator.py`

### 3.2. Configuration Section
- Move configuration options into collapsible panel within Input tab
- Group related settings with clear headings
- Add save/load configuration functionality
- Implementation files:
  - `anpe_gui/widgets/config_panel.py` (new)
  - Update `main_window.py`

### 3.3. Button Consolidation
- Remove duplicate "Process New Input" buttons
- Standardize button placement and styling
- Create consistent action flow
- Implementation files:
  - Update button creation and placement in `main_window.py`

## Phase 4: Splash Screen Improvements

### 4.1. Resize and Reposition
- Limit splash screen to 400x300px maximum
- Fix banner and progress bar positioning
- Add proper margins between elements
- Implementation files:
  - Update `splash_screen.py`

### 4.2. Visual Enhancements
- Add subtle shadow effect to splash window
- Implement smooth fade-in/fade-out animations
- Ensure progress bar updates properly
- Implementation files:
  - Update `splash_screen.py` with QGraphicsEffect
  - Add animation code using QPropertyAnimation

## Phase 5: Enhanced Logging System

### 5.1. Improved Log Handling
- Update QtLogHandler to capture all console output
- Ensure log panel reflects configured log level
- Add timestamps and proper formatting
- Implementation files:
  - Update `qt_log_handler.py`
  - Enhance log display widget

### 5.2. Log Panel Features
- Add log level filtering (dropdown or buttons)
- Implement simple search functionality
- Add copy-to-clipboard feature
- Auto-scroll with option to pause
- Implementation files:
  - Create `anpe_gui/widgets/enhanced_log_panel.py` (new)
  - Update log panel references in `main_window.py`

## Phase 6: Final Polish and Testing

### 6.1. Visual Consistency
- Apply consistent color scheme throughout
- Standardize font sizes and weights
- Ensure proper spacing and alignment everywhere
- Implementation files:
  - Update `theme.py`
  - Create `anpe_gui/utils/style_constants.py` (new)

### 6.2. User Experience Enhancements
- Add tooltips to all interactive elements
- Implement keyboard shortcuts for common actions
- Add context-sensitive help where appropriate
- Implementation files:
  - Update all widget classes with tooltip information

### 6.3. Cross-Platform Testing
- Test on Windows, macOS, and Linux
- Fix platform-specific issues
- Ensure consistent appearance across environments
- Implementation files:
  - Create platform-specific style overrides if needed

## Timeline Estimate

- **Phase 1**: 1-2 days
- **Phase 2**: 2-3 days
- **Phase 3**: 2-3 days
- **Phase 4**: 1 day
- **Phase 5**: 1-2 days
- **Phase 6**: 2-3 days

**Total Estimated Time**: 9-14 days

## Implementation Priority

1. Fix module structure issues (critical)
2. Implement two-tab system with collapsible configuration
3. Standardize input controls and fix layout issues
4. Fix splash screen size and positioning
5. Enhance logging system
6. Polish and finalize

This phased approach ensures that critical issues are addressed first while providing a systematic path to a fully refined UI. 