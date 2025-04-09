import os
import sys
import platform
import subprocess
from pathlib import Path

def build_app():
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Define the main script path
    main_script = current_dir / "run.py"
    
    # Define the output directory
    output_dir = current_dir / "dist"
    
    # Define the icon path
    icon_path = current_dir / "resources" / "icon.ico"
    
    # Define the data files to include
    data_files = [
        (str(current_dir / "resources"), "resources"),
        (str(current_dir / "docs"), "docs"),
    ]
    
    # Create PyInstaller command
    cmd = [
        "pyinstaller",
        "--name=ANPE",
        "--windowed",  # For GUI applications
        "--onefile",   # Create a single executable
        "--clean",     # Clean PyInstaller cache
        "--noconfirm", # Replace output directory without asking
        "--optimize=2", # Optimize bytecode
        "--exclude-module=tkinter",
        "--exclude-module=matplotlib",
        "--exclude-module=notebook",
        "--exclude-module=scipy",
        "--exclude-module=pandas",
        "--hidden-import=anpe",
        "--hidden-import=PyQt6",
        "--hidden-import=os",
        "--hidden-import=sys",
        "--hidden-import=traceback",
        "--hidden-import=logging",
    ]
    
    # Add icon if it exists
    if icon_path.exists():
        cmd.extend(["--icon", str(icon_path)])
    
    # Add data files
    for src, dst in data_files:
        cmd.extend(["--add-data", f"{src}{os.pathsep}{dst}"])
    
    # Add the main script
    cmd.append(str(main_script))
    
    # Run PyInstaller
    print("Building application...")
    subprocess.run(cmd, check=True)
    
    print(f"\nBuild completed! The executable can be found in: {output_dir}")

if __name__ == "__main__":
    build_app() 