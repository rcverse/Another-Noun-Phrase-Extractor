# ANPE GUI

A graphical user interface for [ANPE (Another Noun Phrase Extractor)](https://github.com/rcverse/anpe).

<!-- Optional: Insert Screenshot Here -->
<!-- ![ANPE GUI Screenshot](resources/screenshot.png) -->

## Features

- **User-friendly interface** with distinct Input and Output tabs.
- **Input Modes**: Process text via Direct Text Input (with Paste/Clear) or File Input.
- **File Handling**: Add single files or entire directories; view and manage the list.
- **Batch Processing**: Automatically handles multiple files from selected directories.
- **Visual Configuration**: Easily configure all ANPE settings:
    - General: Include Nested Phrases, Include Metadata, Treat Newlines as Boundaries.
    - Filtering: Min/Max NP length, Accept Pronouns.
    - Structure Filtering: Master toggle switch and individual selection for specific NP structures (Determiner, Compound, Relative Clause, etc.).
    - Tooltips: Hover over options for detailed explanations.
- **Real-time Log Viewer**: Track operations and potential issues with log level filtering.
- **Results Visualization**: View formatted extraction results in the Output tab.
- **Batch Result Navigation**: Use a dropdown to view results for specific files when processing batches.
- **Export Options**: Export results to TXT, CSV, or JSON formats to a selected directory.
- **Status Bar**: Provides feedback on application readiness, processing progress, and completion status.
- **Workflow Control**: Process button initiates extraction, Reset button clears inputs/outputs for a new task.

## Installation & Usage

The ANPE GUI application can be used in two main ways:

### 1. Run from source (Recommended for Development)

Requires Python 3.9+ and the necessary dependencies. Navigate to the `anpe_gui` directory and run:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python run.py
```

### 2. Use Standalone Executable (Recommended for End Users)

Download the standalone executable for your platform (e.g., `ANPE.exe` for Windows, `ANPE` for macOS) from the project's GitHub Releases page (if available).

No installation is required - just download and run!

## Building Standalone Executable from Source

To create a standalone executable for distribution:

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Navigate to Project Directory:**
   ```bash
   cd anpe_gui
   ```

3. **Run Build Script:**
   ```bash
   python build.py
   ```

The build script (`build.py`) will:
- Create a single executable file
- Include all necessary resources and documentation
- Optimize the build for performance
- Handle platform-specific configurations

The executable will be created in the `dist` directory.

### Build Options

The build script includes several optimizations:
- Bytecode optimization (`--optimize=2`)
- Exclusion of unnecessary modules (tkinter, matplotlib, etc.)
- Automatic inclusion of required dependencies
- Resource file bundling
- Icon support (if available)

### Platform-Specific Notes

- **Windows**: The executable will be named `ANPE.exe`
- **macOS**: The executable will be named `ANPE`
- You must build the executable on each target platform separately

## License

This project is licensed under the same terms as ANPE - see the [LICENSE](../LICENSE) file for details. 