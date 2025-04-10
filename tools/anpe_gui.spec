"""
PyInstaller spec file for the ANPE GUI application.
"""

import os
import sys
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

block_cipher = None

a = Analysis(
    [os.path.join(parent_dir, 'anpe_gui', 'run.py')],
    pathex=[parent_dir],
    binaries=[],
    datas=[
        (os.path.join(parent_dir, 'pics'), 'pics'),  # Include pics directory
        (os.path.join(parent_dir, 'anpe_gui'), 'anpe_gui')  # Include anpe_gui package
    ],
    hiddenimports=['anpe', 'PyQt6', 'nltk', 'spacy', 'benepar'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'notebook', 'scipy', 'pandas'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(
    a.pure, 
    a.zipped_data,
    cipher=block_cipher
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='anpe_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=(
        os.path.join(parent_dir, 'anpe_gui', 'resources', 'app_icon.icns') 
        if sys.platform == 'darwin' and os.path.exists(os.path.join(parent_dir, 'anpe_gui', 'resources', 'app_icon.icns'))
        else os.path.join(parent_dir, 'anpe_gui', 'resources', 'app_icon.ico') 
        if sys.platform == 'win32' and os.path.exists(os.path.join(parent_dir, 'anpe_gui', 'resources', 'app_icon.ico'))
        else os.path.join(parent_dir, 'anpe_gui', 'resources', 'app_icon.png')
    ),
) 