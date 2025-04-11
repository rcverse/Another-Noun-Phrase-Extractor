# ANPE GUI Resource Management Strategy (PyQt6)

This document outlines the strategy used for managing application resources (icons, SVGs, etc.) in the ANPE GUI, which uses PyQt6.

## Background

PyQt6 removed the `pyrcc` tool, which was commonly used in PyQt5 to compile Qt Resource Collection (`.qrc`) files into Python modules or binary `.rcc` files. This required adapting the resource management approach.

## Strategy Overview

The current strategy leverages Qt's standard resource system but adapts the compilation process for PyQt6:

1.  **`.qrc` File:** Resources are defined in `anpe_gui/resources.qrc` using the standard Qt XML format.
2.  **Compilation to Python:** Instead of `pyrcc`, we use Qt's `rcc` tool (via the `pyside6-rcc` command-line wrapper, typically installed with the `PySide6` package) to compile `resources.qrc` into a Python module (`anpe_gui/resources_rc.py`).
    *   Command: `pyside6-rcc -o anpe_gui/resources_rc.py anpe_gui/resources.qrc`
3.  **PyQt6 Compatibility Fix:** The generated `resources_rc.py` file includes `from PySide6 import QtCore`. This line **must** be changed to `from PyQt6 import QtCore` to make it compatible with our project.
4.  **Automation Script:** The compilation and modification steps are automated by the script `gui_scripts/compile_resources.py`. This script should be run whenever `resources.qrc` or the files it references are updated.
5.  **Resource Registration:** The modified `anpe_gui/resources_rc.py` module is imported early in the application's startup sequence (in `anpe_gui/app.py`). This automatically executes the code within the module, registering the embedded resources with Qt's internal resource system.
6.  **Abstraction Layer (`ResourceManager`):**
    *   The `anpe_gui/resource_manager.py` class provides a centralized point for accessing resources.
    *   Its `initialize()` method checks if the embedded resources were loaded successfully (by attempting to access a known resource like `:/icons/app_icon.png` using `QFile.exists`).
    *   Methods like `get_icon()`, `get_pixmap()`, and `get_style_url()` check an internal flag set by `initialize()`.
    *   If resources are embedded, they return paths using the `:/` prefix (e.g., `:/icons/my_icon.png`). This prefix works reliably both for direct use in code (`QIcon`, `QPixmap`) and within stylesheet `url()` properties.
    *   If resources are *not* embedded (e.g., running from source without compiling, fallback mode), it constructs direct filesystem paths relative to the `anpe_gui/resources/` directory, ensuring correct formatting (string path for constructors, relative path or `file:///` URI for stylesheets).
7.  **Packaging (`gui_setup_cx.py`):**
    *   The `cx_Freeze` setup script (`gui_setup_cx.py`) now executes `gui_scripts/compile_resources.py` at the beginning of the build process.
    *   It *no longer* includes the old `.rcc` file or the raw `anpe_gui/resources/` directory.
    *   It relies on `cx_Freeze`'s automatic module discovery to include the generated `anpe_gui/resources_rc.py` because it is imported by `anpe_gui/app.py`.

## Build-Time Dependency

Note that the *build environment* (where you run `gui_setup_cx.py`) requires `PySide6` to be installed to provide the `pyside6-rcc` tool used by the `compile_resources.py` script. However, `PySide6` is *not* a runtime dependency for the packaged application, which only requires `PyQt6`.

## Benefits

*   Uses the standard Qt resource system.
*   Embeds resources directly into the application code via the Python module, simplifying distribution.
*   Provides a fallback mechanism for easier development (running directly from source without compiling resources).
*   Centralizes resource access logic in `ResourceManager`.
*   Automates the necessary compilation and modification steps. 