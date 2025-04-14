# -*- mode: python ; coding: utf-8 -*-

import sys
import re # Import regex module
from PyInstaller.utils.hooks import collect_data_files

# --- Configuration based on gui_setup_cx.py ---
APP_NAME = "ANPE"
MAIN_SCRIPT = "anpe_gui/run.py"
ICON_PATH = "anpe_gui/resources/app_icon_logo.ico"
HELP_FILE_SOURCE = "docs/gui_help.md"
HELP_FILE_DEST = "docs"
UPX_PATH = r"C:\Users\b162274\Downloads\upx-5.0.0-win64"  # Path to UPX directory

# Set application version directly
VERSION = '1.0.0b'  # Hardcoded version
VERSION_TUPLE = (1, 0, 0, 0)  # (major, minor, patch, build)

# --- Basic Analysis --- 
block_cipher = None

a = Analysis(
    [MAIN_SCRIPT],
    pathex=[],  # Add paths if PyInstaller struggles to find modules (e.g., your 'anpe' package source)
    binaries=[],
    datas=[
        (HELP_FILE_SOURCE, HELP_FILE_DEST), # Include the help file
        # Data files for nltk, spacy, benepar are now handled by the app's setup wizard
        # No longer need:
        # *collect_data_files('nltk', include_py_files=True),
        # *collect_data_files('benepar', include_py_files=True),
        # *collect_data_files('spacy'),
        # Add other data files if needed
    ],
    hiddenimports=[
        # Core dependencies
        'PyQt6',
        'anpe',
        'nltk',
        'spacy',
        'benepar',
        # Common PyQt6 modules sometimes missed
        'PyQt6.sip',
        'PyQt6.QtWidgets',
        'PyQt6.QtGui',
        'PyQt6.QtCore',
        'PyQt6.QtNetwork',
        'PyQt6.QtPrintSupport',
        # Torch related imports that were missing
        'torch',
        'torch.ao',
        'torch.fx',
        'torch._dispatch',
        'unittest',
        'unittest.mock',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "notebook",
        "scipy",
        "pandas",
        "test",
        # Removed unittest from excludes since torch needs it
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# --- Executable --- 
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe_options = {
    "name": APP_NAME,
    "debug": False,
    "bootloader_ignore_signals": False,
    "strip": False,
    "upx": True, # UPX compression enabled
    "upx_dir": UPX_PATH,  # Set the UPX directory path
    "console": False, # False for GUI apps (like base="Win32GUI")
    "disable_windowed_traceback": False,
    "argv_emulation": False,
    "target_arch": None,
    "codesign_identity": None,
    "entitlements_file": None,
}

# Add icon based on platform
if sys.platform == "win32":
    exe_options["icon"] = ICON_PATH
elif sys.platform == "darwin": # macOS
    # Icon handled in BUNDLE/app for Mac
    pass 

exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, **exe_options)

# Create a directory-based executable (onedir mode)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)

# --- macOS Specific Bundle --- 
app = None
if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name=f"{APP_NAME}.app",
        icon=ICON_PATH,
        bundle_identifier=None, # e.g., com.yourcompany.anpe
    )

# --- Windows Specific: Version Info (Optional) ---
# Embeds version metadata into the .exe properties
# Requires: pip install pywin32-ctypes
try:
    import PyInstaller.utils.win32.versioninfo as vinfo

    vi = vinfo.VSVersionInfo(
        ffi=vinfo.FixedFileInfo(
            # filevers and prodvers should be tuples of four integers: (major, minor, patch, build)
            filevers=VERSION_TUPLE, # Use version from version.py
            prodvers=VERSION_TUPLE, # Use version from version.py
            mask=0x3f, # Contains valid entries
            flags=0x0, # Not debug, prerelease, patched, private
            OS=0x40004, # Windows NT 64-bit
            fileType=0x1, # VFT_APP
            subtype=0x0, # Not used
            date=(0, 0) # Optional
        ),
        kids=[
            vinfo.StringFileInfo([
                vinfo.StringTable(
                    u'040904b0', # Lang/Codepage -> US English, Unicode
                    [vinfo.StringStruct(u'CompanyName', u'Richard Chen'), # <-- !!! SET YOUR NAME/HANDLE !!!
                     vinfo.StringStruct(u'FileDescription', u'Another Noun Phrase Extractor'),
                     vinfo.StringStruct(u'FileVersion', VERSION),
                     vinfo.StringStruct(u'InternalName', APP_NAME),
                     vinfo.StringStruct(u'LegalCopyright', u'Copyright © 2025 Richard Chen'), # <-- !!! SET YEAR & YOUR NAME/HANDLE !!!
                     vinfo.StringStruct(u'OriginalFilename', f'{APP_NAME}.exe'),
                     vinfo.StringStruct(u'ProductName', APP_NAME),
                     vinfo.StringStruct(u'ProductVersion', VERSION)])
            ]),
            vinfo.VarFileInfo([vinfo.VarStruct(u'Translation', [1033, 1200])]) # Lang/Codepage: 1033->US English, 1200->Unicode
        ]
    )
    # Assign vi to exe for Windows build
    if sys.platform == "win32":
       # Recreate EXE object with version info
       exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, version=vi, **exe_options)
except ImportError:
    print("Warning: pywin32-ctypes not found. Run 'pip install pywin32-ctypes' to embed version info.")
except Exception as e:
    print(f"Warning: Error creating version info: {e}")
