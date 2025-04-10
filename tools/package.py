#!/usr/bin/env python3
"""
Package the ANPE GUI application as a standalone executable.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
from PIL import Image

def main():
    """Main packaging function."""
    print("Packaging ANPE GUI application...")
    
    # Add parent directory to path for imports
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    # Determine the operating system
    system = platform.system()
    print(f"Detected operating system: {system}")
    
    # Create the dist directory if it doesn't exist
    dist_dir = Path(parent_dir) / "dist"
    dist_dir.mkdir(exist_ok=True)
    
    # Package using PyInstaller
    try:
        print("Running PyInstaller...")
        subprocess.run(
            ["pyinstaller", "--onefile", "--windowed", 
             os.path.join(os.path.dirname(__file__), "anpe_gui.spec")],
            check=True,
            cwd=parent_dir  # Run from parent directory
        )
        print("PyInstaller completed successfully.")
        
        # Additional steps for specific platforms
        if system == "Darwin":  # macOS
            print("Performing additional steps for macOS...")
            
            # Check if we need to convert the PNG to ICNS for better macOS integration
            try:
                png_icon_path = Path(parent_dir) / "anpe_gui" / "resources" / "app_icon.png"
                if png_icon_path.exists():
                    # Create a temporary iconset directory
                    iconset_dir = Path(parent_dir) / "tmp.iconset"
                    iconset_dir.mkdir(exist_ok=True)
                    
                    # Create multiple size icons for macOS
                    icon_sizes = [16, 32, 64, 128, 256, 512, 1024]
                    for size in icon_sizes:
                        subprocess.run([
                            "sips", "-z", str(size), str(size), 
                            str(png_icon_path), "--out", 
                            str(iconset_dir / f"icon_{size}x{size}.png")
                        ], check=True, capture_output=True)
                        
                        # Create 2x versions for Retina displays
                        if size <= 512:
                            subprocess.run([
                                "sips", "-z", str(size*2), str(size*2), 
                                str(png_icon_path), "--out", 
                                str(iconset_dir / f"icon_{size}x{size}@2x.png")
                            ], check=True, capture_output=True)
                    
                    # Convert iconset to icns
                    icns_path = Path(parent_dir) / "anpe_gui" / "resources" / "app_icon.icns"
                    subprocess.run([
                        "iconutil", "-c", "icns", str(iconset_dir), 
                        "-o", str(icns_path)
                    ], check=True, capture_output=True)
                    
                    # Update the PyInstaller spec file to use the ICNS file
                    if icns_path.exists():
                        # Clean up
                        shutil.rmtree(iconset_dir)
                        print(f"Created macOS icon file: {icns_path}")
                
            except Exception as e:
                print(f"Warning: Could not create macOS icon: {e}")
            
            # Create a DMG file if possible
            try:
                app_path = dist_dir / "anpe_gui.app"
                if app_path.exists():
                    subprocess.run(
                        ["hdiutil", "create", "-volname", "ANPE GUI", 
                         "-srcfolder", str(app_path), "-ov", "-format", "UDZO", 
                         str(dist_dir / "anpe_gui.dmg")],
                        check=True
                    )
                    print("Created DMG file: dist/anpe_gui.dmg")
            except Exception as e:
                print(f"Warning: Could not create DMG file: {e}")
        
        elif system == "Windows":
            print("Performing additional steps for Windows...")
            
            # Convert PNG to ICO for Windows
            try:
                png_icon_path = Path(parent_dir) / "anpe_gui" / "resources" / "app_icon.png"
                if png_icon_path.exists():
                    ico_path = Path(parent_dir) / "anpe_gui" / "resources" / "app_icon.ico"
                    
                    # Open the PNG and convert to ICO
                    img = Image.open(str(png_icon_path))
                    
                    # Create multiple sizes for the ICO
                    icon_sizes = [(16,16), (32,32), (48,48), (64,64), (128,128), (256,256)]
                    img_list = []
                    
                    for size in icon_sizes:
                        resized_img = img.resize(size, Image.LANCZOS)
                        img_list.append(resized_img)
                    
                    # Save as ICO
                    img_list[0].save(
                        str(ico_path), 
                        format='ICO', 
                        sizes=[(img.width, img.height) for img in img_list],
                        append_images=img_list[1:]
                    )
                    
                    print(f"Created Windows icon file: {ico_path}")
            except Exception as e:
                print(f"Warning: Could not create Windows icon: {e}")
            
            # Create a simple ZIP file if possible
            try:
                exe_path = dist_dir / "anpe_gui.exe"
                if exe_path.exists():
                    import zipfile
                    zip_path = dist_dir / "anpe_gui_windows.zip"
                    with zipfile.ZipFile(str(zip_path), 'w') as zipf:
                        zipf.write(str(exe_path), arcname="anpe_gui.exe")
                    print(f"Created ZIP file: {zip_path}")
            except Exception as e:
                print(f"Warning: Could not create ZIP file: {e}")
        
        print("Packaging completed successfully!")
        print(f"Executable available in the {dist_dir} directory.")
        
    except subprocess.CalledProcessError as e:
        print(f"Error: PyInstaller failed with error code {e.returncode}")
        print(e.output)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 