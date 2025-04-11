import sys
import os
import subprocess
from cx_Freeze import setup, Executable
from pathlib import Path

# Get application version
with open('anpe_gui/version.py') as f:
    version_info = {}
    exec(f.read(), version_info)
    VERSION = version_info.get('__version__', '0.1.0')

# Define application name, description and entry point
APP_NAME = "ANPE"
DESCRIPTION = "Another Noun Phrase Extractor"
MAIN_SCRIPT = "anpe_gui/run.py"

# Compile Qt resources
print("Compiling Qt resources...")
try:
    # Check if pyrcc6 is available
    subprocess.run(["pyrcc6", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    
    # Compile the resources
    res_file = os.path.abspath("anpe_gui/resources.qrc")
    out_file = os.path.abspath("anpe_gui/resources.rcc")
    
    subprocess.run(
        ["pyrcc6", "-o", out_file, res_file], 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        check=True
    )
    print(f"Resources compiled to {out_file}")
except Exception as e:
    print(f"Warning: Could not compile Qt resources: {e}")
    print("Resources will be included as files instead.")

# Determine base for executable
# base="Win32GUI" for Windows GUI applications,
# None for console applications, or "MacOSXApp" for macOS
base = None
icon = None

if sys.platform == "win32":
    base = "Win32GUI"
    icon = "anpe_gui/resources/app_icon_logo.png"
elif sys.platform == "darwin":
    base = "MacOSXApp"
    icon = "anpe_gui/resources/app_icon_logo.png"

# Define entry point executables
executables = [
    Executable(
        MAIN_SCRIPT,
        base=base,
        target_name=APP_NAME,
        icon=icon,
        copyright="Copyright © 2025",
    )
]

# Essential Python packages to include
packages = [
    "PyQt6",
    "anpe",  # Your core package
    "nltk",
    "spacy",
    "benepar",
    "os",
    "sys", 
    "traceback",
    "logging",
]

# Exclude packages that aren't needed
excludes = [
    "tkinter",
    "matplotlib",
    "notebook",
    "scipy",
    "pandas",  # Unless you actually use these
    "test",
    "unittest",
    "pyinstaller", # Not needed in the final package
]

# Additional files/folders to include
include_files = [
    ("anpe_gui/resources.rcc", "resources.rcc"),  # Include compiled resources
    ("anpe_gui/resources", "resources"),  # Include original resources as fallback
    ("anpe_gui/docs", "docs"),
]

# Define build options
build_options = {
    "packages": packages,
    "excludes": excludes,
    "include_files": include_files,
    "optimize": 2,  # Optimize bytecode
    "include_msvcr": True,  # Include Microsoft Visual C Runtime (Windows)
    "zip_include_packages": "*",  # Compress as many packages as possible into a zip file
    "zip_exclude_packages": [],  # Packages that shouldn't be compressed (empty for max compression)
}

# Create MSI installer options for Windows
bdist_msi_options = {
    "upgrade_code": "{12345678-1234-5678-abcd-1234567890ab}",  # Use a fixed, unique GUID
    "add_to_path": False,
    "initial_target_dir": r"[ProgramFilesFolder]\%s" % APP_NAME,
}

# macOS DMG options
bdist_dmg_options = {
    "applications_shortcut": True,  # Add shortcut to Applications folder
    "volume_label": f"{APP_NAME} Installer",
}

# Setup parameters
setup(
    name=APP_NAME,
    version=VERSION,
    description=DESCRIPTION,
    options={
        "build_exe": build_options,
        "bdist_msi": bdist_msi_options,
        "bdist_mac": {"iconfile": icon} if sys.platform == "darwin" else {},
        "bdist_dmg": bdist_dmg_options if sys.platform == "darwin" else {},
    },
    executables=executables,
) 