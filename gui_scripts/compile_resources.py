import subprocess
import sys
import os
from pathlib import Path

def main():
    """Compiles the .qrc file to a .py module and fixes the import for PyQt6."""
    print("Starting resource compilation...")

    # Determine paths relative to this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent # Assumes gui_scripts is one level below project root
    anpe_gui_dir = project_root / "anpe_gui"
    qrc_file = anpe_gui_dir / "resources.qrc"
    output_py_file = anpe_gui_dir / "resources_rc.py"

    # Ensure paths exist before proceeding
    if not qrc_file.is_file():
        print(f"ERROR: Input QRC file not found: {qrc_file}", file=sys.stderr)
        sys.exit(1)
    if not anpe_gui_dir.is_dir():
        print(f"ERROR: anpe_gui directory not found: {anpe_gui_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"  Project Root: {project_root}")
    print(f"  Input QRC:    {qrc_file}")
    print(f"  Output PY:    {output_py_file}")

    # --- Step 1: Compile .qrc to .py using pyside6-rcc --- 
    print("  Running pyside6-rcc...")
    try:
        # Use the current Python executable to find the pyside6-rcc script module
        cmd = [
            sys.executable,
            "-m",
            "PySide6.scripts.rcc", # Module path for pyside6-rcc
            "-o",
            str(output_py_file),
            str(qrc_file)
        ]
        print(f"    Command: {' '.join(cmd)}") # Log the command being run
        
        # Run the command
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print("    pyside6-rcc completed successfully.")
        if result.stdout:
            print("    Output:\n", result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"ERROR: pyside6-rcc failed with exit code {e.returncode}.", file=sys.stderr)
        if e.stderr:
            print("  Error Output:\n", e.stderr, file=sys.stderr)
        else:
            print("  No standard error output captured.", file=sys.stderr)
        if e.stdout:
             print("  Standard Output:\n", e.stdout, file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        # This might happen if sys.executable is weird, or the module isn't found
        print(f"ERROR: Could not execute Python or find PySide6.scripts.rcc module.", file=sys.stderr)
        print(f"       Make sure PySide6 is installed in the environment: {sys.executable}", file=sys.stderr)
        print(f"       Attempted command: {' '.join(cmd)}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during compilation: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Step 2: Modify the generated .py file --- 
    print(f"  Modifying {output_py_file} for PyQt6...")
    try:
        if not output_py_file.is_file():
             print(f"ERROR: Generated file {output_py_file} not found after compilation step.", file=sys.stderr)
             sys.exit(1)
             
        with open(output_py_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified = False
        new_lines = []
        for line_num, line in enumerate(lines):
            # Target the specific import line more carefully
            if line.strip().startswith("from PySide6 import QtCore"):
                new_lines.append("from PyQt6 import QtCore\n")
                modified = True
                print(f"    Replaced PySide6 import with PyQt6 import on line {line_num + 1}.")
            else:
                new_lines.append(line)
                
        if not modified:
            print(f"WARNING: Did not find 'from PySide6 import QtCore' in {output_py_file}. File may already be modified or generated differently.")

        with open(output_py_file, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("    File modification successful.")

    except Exception as e:
        print(f"An error occurred during file modification: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nResource compilation finished successfully!")

if __name__ == "__main__":
    main() 