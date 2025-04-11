# -*- mode: python ; coding: utf-8 -*-

import sys
from PyInstaller.utils.hooks import collect_data_files

# --- Configuration based on gui_setup_cx.py ---
APP_NAME = "ANPE"
MAIN_SCRIPT = "anpe_gui/run.py"
ICON_PATH = "anpe_gui/resources/app_icon_logo.ico"
HELP_FILE_SOURCE = "docs/gui_help.md"
HELP_FILE_DEST = "docs"

# --- Basic Analysis --- 
block_cipher = None

a = Analysis(
    [MAIN_SCRIPT],
    pathex=[],  # Add paths if PyInstaller struggles to find modules (e.g., your 'anpe' package source)
    binaries=[],
    datas=[
        (HELP_FILE_SOURCE, HELP_FILE_DEST), # Include the help file
        # Add data files for nltk, spacy, benepar if needed
        # Example: *collect_data_files('spacy'),
        # Example: *collect_data_files('nltk'),
        # Example: *collect_data_files('benepar')
        # You might need specific sub-modules/data, e.g.:
        # ('/path/to/venv/Lib/site-packages/spacy/data/en_core_web_sm', 'spacy/data/en_core_web_sm')
    ],
    hiddenimports=[
        'PyQt6', # Sometimes helps ensure all Qt components are found
        'anpe', 
        'nltk',
        'spacy',
        'benepar',
        # Add specific submodules if PyInstaller misses them
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
        "unittest",
        # PyInstaller itself is not needed in the bundle
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
    "upx": True, # UPX compression can reduce size but might trigger antivirus
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
# You can add version information for the .exe on Windows
# Requires pywin32-ctypes: pip install pywin32-ctypes
# try:
#     import PyInstaller.utils.win32.versioninfo as vinfo
#     # Read version from your version.py
#     with open('anpe_gui/version.py') as f:
#         version_info = {}
#         exec(f.read(), version_info)
#         VERSION = version_info.get('__version__', '0.1.0')
# 
#     vi = vinfo.VSVersionInfo(
#         ffi=vinfo.FixedFileInfo(
#             # ... Set version numbers, etc. ... 
#             filevers=(0, 0, 0, 0), # Example
#             prodvers=(0, 0, 0, 0), # Example
#             # ... other fields ...
#         ),
#         kids=[
#             vinfo.StringFileInfo([
#                 vinfo.StringTable(
#                     u'040904b0', # Lang/Codepage -> US English, Unicode
#                     [vinfo.StringStruct(u'CompanyName', u'Your Company Name'),
#                      vinfo.StringStruct(u'FileDescription', u'Another Noun Phrase Extractor'),
#                      vinfo.StringStruct(u'FileVersion', VERSION),
#                      vinfo.StringStruct(u'InternalName', APP_NAME),
#                      vinfo.StringStruct(u'LegalCopyright', u'Copyright © 2025 Your Name/Company'),
#                      vinfo.StringStruct(u'OriginalFilename', f'{APP_NAME}.exe'),
#                      vinfo.StringStruct(u'ProductName', APP_NAME),
#                      vinfo.StringStruct(u'ProductVersion', VERSION)])
#             ]),
#             vinfo.VarFileInfo([vinfo.VarStruct(u'Translation', [1033, 1200])]) # Lang/Codepage: 1033->US English, 1200->Unicode
#         ]
#     )
#     # Assign vi to exe for Windows build
#     if sys.platform == "win32":
#        exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, version=vi, **exe_options)
# except ImportError:
#     print("pywin32-ctypes not found, skipping version info embedding for Windows exe.")
# except Exception as e:
#     print(f"Error creating version info: {e}")
