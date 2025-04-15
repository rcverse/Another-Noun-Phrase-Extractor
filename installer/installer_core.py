import sys
import os
import platform
import shutil
import subprocess
import zipfile
import tarfile
import urllib.request
import tempfile

# Import utils differently depending on how script is run
try:
    # If run as part of the package (e.g., via setup_windows.pyw)
    from .utils import get_resource_path
except ImportError:
    # If run directly as a script
    try:
        from utils import get_resource_path
    except ImportError:
        # Fallback if utils.py is not in python path when run directly
        print("CRITICAL WARNING: Could not import get_resource_path. Using basic relative paths.", file=sys.stderr)
        def get_resource_path(relative_path):
            base_path = os.path.dirname(os.path.abspath(__file__))
            return os.path.join(base_path, relative_path)

# --- Constants ---
# Use helper function to get absolute path to assets dir
ASSETS_DIR = get_resource_path('assets')
PYTHON_DIR_NAME = "python"
APP_CODE_DIR_NAME = "anpe_gui"
# Add explicit name for app code source folder
APP_SOURCE_FOLDER_NAME = "anpe_gui"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
# Add constant for docs directory
DOCS_DIR_NAME = "docs"

# --- Helper Functions ---

def print_step(message: str):
    """Print a progress step message to stdout."""
    print(f"STEP: {message}", flush=True)

def print_success(message: str):
    """Print a success message and exit."""
    print(f"SUCCESS: {message}", flush=True)
    sys.exit(0)

def print_failure(message: str):
    """Print a failure message and exit."""
    # Log the error to stderr as well for visibility in logs
    print(f"ERROR: {message}", file=sys.stderr, flush=True)
    print(f"FAILURE: {message}", flush=True)
    sys.exit(1)

def print_python_path(path: str):
    """Print the detected Python executable path."""
    print(f"Python executable found: {path}", flush=True)

def find_asset(filename_prefix: str, expected_suffix: str = "") -> str:
    """Find an asset file in the assets directory based on prefix and optional suffix."""
    print_step(f"Searching for asset starting with '{filename_prefix}'{f' ending with \'{expected_suffix}\'' if expected_suffix else ''}...")
    # Use try-except for path resolution
    try:
        assets_dir_abs = get_resource_path('assets')
    except Exception as e:
        print_failure(f"Could not resolve assets directory path: {e}")
        
    if not os.path.isdir(assets_dir_abs):
        print_failure(f"Assets directory not found or is not a directory at the expected location: {assets_dir_abs}")

    found_files = []
    try:
        for item in os.listdir(assets_dir_abs):
            if item.startswith(filename_prefix) and (not expected_suffix or item.endswith(expected_suffix)):
                found_files.append(os.path.join(assets_dir_abs, item))
    except Exception as e:
        print_failure(f"Error listing assets directory {assets_dir_abs}: {e}")

    if not found_files:
        print_failure(f"Could not find any asset starting with '{filename_prefix}'{f' ending with \'{expected_suffix}\'' if expected_suffix else ''} in {assets_dir_abs}")
    if len(found_files) > 1:
        print_failure(f"Found multiple matching assets for '{filename_prefix}': {found_files}. Please ensure only one exists.")

    found_path = found_files[0]
    print_step(f"Found asset: {os.path.basename(found_path)}")
    return found_path

def unpack_python(target_install_path: str):
    """Unpack the bundled Python distribution."""
    python_archive_path = ""
    system = platform.system()
    archive_type = ""

    if system == "Windows":
        # Assuming embeddable zip like 'python-3.x.x-embed-amd64.zip'
        python_archive_path = find_asset("python-", expected_suffix=".zip")
        archive_type = "zip"
    # TODO: Add elif for macOS (e.g., .tar.gz)
    # elif system == "Darwin":
    #     python_archive_path = find_asset("python-", expected_suffix=".tar.gz")
    #     archive_type = "tar.gz"
    else:
        print_failure(f"Unsupported operating system: {system}")

    python_extract_path = os.path.join(target_install_path, PYTHON_DIR_NAME)

    if os.path.exists(python_extract_path):
        print_step(f"Removing existing directory: {python_extract_path}")
        try:
            # Add retry logic? shutil.rmtree can fail on Windows sometimes
            shutil.rmtree(python_extract_path)
        except OSError as e:
            print_failure(f"Failed to remove existing Python directory: {python_extract_path}. Error: {e}")

    print_step(f"Unpacking Python to {python_extract_path}...")
    try:
        if archive_type == "zip":
            with zipfile.ZipFile(python_archive_path, 'r') as zip_ref:
                zip_ref.extractall(python_extract_path)
        # TODO: Add tarfile extraction for macOS
        # elif archive_type == "tar.gz":
        #     with tarfile.open(python_archive_path, "r:gz") as tar_ref:
        #         tar_ref.extractall(python_extract_path)
    except zipfile.BadZipFile:
        print_failure(f"Failed to unpack Python: The file '{python_archive_path}' is not a valid zip file or is corrupted.")
    except Exception as e:
        print_failure(f"Failed to unpack Python archive '{os.path.basename(python_archive_path)}': {e}")

    print_step("Python unpacked successfully.")
    return python_extract_path

def find_python_executable(python_extract_path: str) -> str:
    """Find the python executable within the unpacked directory."""
    print_step("Locating Python executable...")
    system = platform.system()
    expected_path = ""
    if system == "Windows":
        # Standard embeddable package location
        expected_path = os.path.join(python_extract_path, "python.exe")
    # TODO: Add elif for macOS standard standalone build location
    # elif system == "Darwin":
    #     expected_path = os.path.join(python_extract_path, "bin", "python3") # Adjust as needed
    else:
         print_failure(f"Cannot determine Python executable path for OS: {system}")

    if not os.path.isfile(expected_path):
        # Provide more context on failure
        print(f"Searched for Python executable at: {expected_path}", file=sys.stderr)
        print(f"Contents of {python_extract_path}: {os.listdir(python_extract_path) if os.path.isdir(python_extract_path) else 'Not a directory'}", file=sys.stderr)
        print_failure(f"Python executable not found after unpacking.")

    print_python_path(expected_path) # CRITICAL: Print path for the GUI worker
    return expected_path

def enable_site_packages(python_extract_path: str):
    """Find the ._pth file and uncomment 'import site' to enable site-packages."""
    print_step("Enabling site-packages in Python environment...")
    pth_file_path = None
    try:
        for item in os.listdir(python_extract_path):
            # Look for python<version>._pth, e.g., python312._pth
            if item.startswith("python") and item.endswith("._pth"):
                pth_file_path = os.path.join(python_extract_path, item)
                break
    except Exception as e:
        print_failure(f"Error searching for ._pth file in {python_extract_path}: {e}")

    if not pth_file_path:
        print_failure(f"Could not find the python*._pth file in {python_extract_path}")

    print_step(f"Found ._pth file: {os.path.basename(pth_file_path)}")
    updated_lines = []
    modified = False
    try:
        with open(pth_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            stripped_line = line.strip()
            if stripped_line == "#import site":
                updated_lines.append("import site\n") # Uncomment the line
                modified = True
                print_step("Uncommented 'import site' in ._pth file.")
            elif stripped_line == "import site":
                 updated_lines.append(line) # Already uncommented
                 print_step("'import site' already uncommented in ._pth file.")
                 modified = False # No need to write if it was already correct
                 break # Assume only one import site line
            else:
                updated_lines.append(line)
        
        if modified:
            print_step(f"Writing updated ._pth file: {pth_file_path}")
            with open(pth_file_path, 'w', encoding='utf-8') as f:
                f.writelines(updated_lines)
            print_step("._pth file updated successfully.")
        
    except Exception as e:
        print_failure(f"Error reading or writing ._pth file {pth_file_path}: {e}")

def bootstrap_pip(python_exe: str):
    """Download and run get-pip.py to install pip."""
    print_step("Bootstrapping pip installation...")
    get_pip_path = ""
    try:
        # Download get-pip.py to a temporary file
        print_step(f"Downloading get-pip.py from {GET_PIP_URL}...")
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix=".py") as tmp_file:
            with urllib.request.urlopen(GET_PIP_URL) as response:
                if response.status != 200:
                    print_failure(f"Failed to download get-pip.py. Status code: {response.status}")
                tmp_file.write(response.read())
                get_pip_path = tmp_file.name
        print_step(f"Downloaded get-pip.py to: {get_pip_path}")

        # Run get-pip.py using the target python executable
        print_step(f"Running get-pip.py using {python_exe}...")
        command = [python_exe, get_pip_path]
        creationflags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace', creationflags=creationflags)
        print(f"get-pip.py STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"get-pip.py STDERR:\n{result.stderr}", file=sys.stderr)
        print_step("pip bootstrapped successfully.")

    except urllib.error.URLError as e:
        print_failure(f"Network error downloading get-pip.py: {e}")
    except subprocess.CalledProcessError as e:
        error_message = f"Failed command: {' '.join(e.cmd)}\nExit Code: {e.returncode}\n"
        error_message += f"STDOUT:\n{e.stdout}\n"
        error_message += f"STDERR:\n{e.stderr}"
        print(error_message, file=sys.stderr)
        print_failure(f"Failed to execute get-pip.py. Check logs for details.")
    except Exception as e:
        print_failure(f"An unexpected error occurred during pip bootstrapping: {e}")
    finally:
        # Clean up the temporary file
        if get_pip_path and os.path.exists(get_pip_path):
            try:
                os.remove(get_pip_path)
                print_step(f"Removed temporary get-pip.py file: {get_pip_path}")
            except OSError as e:
                print(f"Warning: Failed to remove temporary file {get_pip_path}: {e}", file=sys.stderr)

def run_pip_install(python_exe: str, package: str):
    """Run pip install for a given package using the specified Python executable."""
    if package == "--upgrade pip": # Check specifically for the upgrade command
        print_step(f"Upgrading pip...")
        command = [python_exe, "-m", "pip", "install", "--upgrade", "pip"]
    elif package.startswith("--"): # Handle other potential future args starting with --
        print_failure(f"Unsupported pip argument style encountered: {package}")
    else:
        print_step(f"Installing {package}...")
        command = [python_exe, "-m", "pip", "install", package]
    
    print(f"Executing: {' '.join(command)}") # Log the command being run
    try:
        # Use CREATE_NO_WINDOW on Windows to prevent console pop-ups
        creationflags = subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace', creationflags=creationflags)
        # Print stdout only if install was successful
        print(f"PIP STDOUT for {package}:\n{result.stdout}")
        if result.stderr:
            # Treat stderr as warnings unless exit code is non-zero
            print(f"PIP STDERR for {package}:\n{result.stderr}", file=sys.stderr)
        print_step(f"Successfully processed {package}.")
    except subprocess.CalledProcessError as e:
        # Log detailed error output before failing
        error_message = f"Failed command: {' '.join(e.cmd)}\nExit Code: {e.returncode}\n"
        error_message += f"PIP STDOUT:\n{e.stdout}\n"
        error_message += f"PIP STDERR:\n{e.stderr}"
        print(error_message, file=sys.stderr)
        print_failure(f"Failed to process {package}. Check logs for details.")
    except FileNotFoundError:
        print_failure(f"Failed to run pip. Command '{command[0]}' not found. Is Python correctly unpacked at '{python_exe}'?")
    except Exception as e:
        # Catch other potential errors during subprocess execution
         print_failure(f"An unexpected error occurred while running pip for {package}: {e}")

def copy_app_code(target_install_path: str):
    """Copy the anpe_gui source code, finding it dynamically based on execution context."""
    print_step("Determining application source code location...")

    source_app_dir = None
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Running as a bundled app (e.g., PyInstaller)
        print_step("Running in packaged mode. Locating bundled assets...")
        try:
            # Assume get_resource_path finds the base path (e.g., _MEIPASS)
            # And anpe_gui was packaged inside an 'assets' directory within the bundle
            assets_dir_abs = get_resource_path('assets')
            source_app_dir = os.path.join(assets_dir_abs, APP_SOURCE_FOLDER_NAME)
            print_step(f"Expecting bundled source at: {source_app_dir}")
        except Exception as e:
            print_failure(f"Could not resolve bundled assets directory path: {e}")
    else:
        # Running from source (development mode)
        print_step("Running in development mode. Locating source relative to script...")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            source_app_dir = os.path.join(project_root, APP_SOURCE_FOLDER_NAME)
            print_step(f"Expecting development source at: {source_app_dir}")
        except Exception as e:
            print_failure(f"Could not determine development source path: {e}")

    if not source_app_dir:
        # This means path determination failed above, error already printed.
        sys.exit(1) # Exit cleanly after failure message

    target_app_dir = os.path.join(target_install_path, APP_CODE_DIR_NAME)

    print_step(f"Verifying application source code at {source_app_dir}...")
    if not os.path.isdir(source_app_dir):
        # Provide more context in the error message
        mode = "bundled in assets" if getattr(sys, 'frozen', False) else "at project root"
        print_failure(f"Application source code directory '{APP_SOURCE_FOLDER_NAME}' not found {mode}: {source_app_dir}")

    if os.path.exists(target_app_dir):
        print_step(f"Removing existing app directory: {target_app_dir}")
        try:
            shutil.rmtree(target_app_dir)
        except OSError as e:
            print_failure(f"Failed to remove existing app code directory: {target_app_dir}. Error: {e}")

    print_step(f"Copying application code from {source_app_dir} to {target_app_dir}...")
    try:
        # Create a function to ignore __pycache__ directories
        def ignore_pycache(dir, files):
            return [f for f in files if f == '__pycache__']
            
        # Use the ignore function with copytree
        shutil.copytree(source_app_dir, target_app_dir, dirs_exist_ok=False, ignore=ignore_pycache)
    except FileExistsError:
        # This shouldn't happen due to the rmtree above, but handle defensively
        print_failure(f"Target application directory already exists after attempting removal: {target_app_dir}")
    except Exception as e:
        print_failure(f"Failed to copy application code: {e}")
    print_step("Application code copied successfully.")

# --- NEW FUNCTION: Copy Documentation --- 
def copy_docs(target_install_path: str):
    """Copy the documentation files (e.g., gui_help.md)."""
    print_step("Determining documentation source location...")
    source_docs_dir = None
    help_file_name = "gui_help.md"

    # Assume docs directory is at the project root (sibling to installer, anpe_gui)
    # This logic might need adjustment for bundled mode if docs are packaged differently
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        # Basic handling for packaged mode - assumes docs are bundled inside assets
        # This might need refinement in the PyInstaller spec later
        print_step("Running in packaged mode. Expecting docs in assets...")
        try:
            assets_dir_abs = get_resource_path('assets')
            source_docs_dir = os.path.join(assets_dir_abs, DOCS_DIR_NAME)
            print_step(f"Expecting bundled docs source at: {source_docs_dir}")
        except Exception as e:
            # Fail gracefully if docs aren't critical, maybe just warn?
            print(f"WARNING: Could not resolve bundled assets/docs directory path: {e}. Skipping docs copy.", file=sys.stderr)
            return # Don't proceed if path fails
    else:
        # Development mode: Find ../docs relative to this script
        print_step("Running in development mode. Locating source relative to script...")
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            source_docs_dir = os.path.join(project_root, DOCS_DIR_NAME)
            print_step(f"Expecting development docs source at: {source_docs_dir}")
        except Exception as e:
            print(f"WARNING: Could not determine development docs path: {e}. Skipping docs copy.", file=sys.stderr)
            return # Don't proceed if path fails

    if not source_docs_dir or not os.path.isdir(source_docs_dir):
        mode = "bundled in assets" if getattr(sys, 'frozen', False) else "at project root"
        print(f"WARNING: Documentation source directory '{DOCS_DIR_NAME}' not found {mode}: {source_docs_dir}. Skipping docs copy.", file=sys.stderr)
        return

    # Define source and target paths for the help file
    source_help_file = os.path.join(source_docs_dir, help_file_name)
    target_docs_dir = os.path.join(target_install_path, DOCS_DIR_NAME)
    target_help_file = os.path.join(target_docs_dir, help_file_name)

    print_step(f"Verifying help file at {source_help_file}...")
    if not os.path.isfile(source_help_file):
        print(f"WARNING: Help file '{help_file_name}' not found in source directory: {source_docs_dir}. Skipping docs copy.", file=sys.stderr)
        return

    # Create target directory
    print_step(f"Ensuring target documentation directory exists: {target_docs_dir}")
    try:
        os.makedirs(target_docs_dir, exist_ok=True)
    except Exception as e:
        print_failure(f"Failed to create target documentation directory: {target_docs_dir}. Error: {e}")

    # Copy the help file
    print_step(f"Copying {help_file_name} to {target_docs_dir}...")
    try:
        shutil.copy2(source_help_file, target_help_file) # copy2 preserves metadata
    except Exception as e:
        print_failure(f"Failed to copy help file '{help_file_name}': {e}")
    
    print_step(f"Documentation file '{help_file_name}' copied successfully.")
# --- End Documentation Copy --- 


# --- Main Execution ---

def main(install_path: str):
    """Main logic for Stage 1 setup."""
    print_step(f"Starting ANPE Environment Setup in {install_path}")

    # 1. Validate install path
    print_step("Validating installation path...")
    install_path_abs = os.path.abspath(install_path)
    install_parent_dir = os.path.dirname(install_path_abs)

    if not os.path.isdir(install_parent_dir):
         print_step(f"Parent directory does not exist, attempting to create: {install_parent_dir}")
         try:
             os.makedirs(install_parent_dir, exist_ok=True)
             print_step(f"Successfully created parent directory: {install_parent_dir}")
         except Exception as e:
             print_failure(f"Installation path parent directory cannot be created: {install_parent_dir}. Error: {e}")
    
    # Create target directory if it doesn't exist
    if not os.path.exists(install_path_abs):
        print_step(f"Target directory does not exist, attempting to create: {install_path_abs}")
        try:
            os.makedirs(install_path_abs, exist_ok=True)
            print_step(f"Successfully created target directory: {install_path_abs}")
        except Exception as e:
             print_failure(f"Target installation directory cannot be created: {install_path_abs}. Error: {e}")

    # Writability check
    try:
        test_file = os.path.join(install_path_abs, ".anpe_installer_write_test")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
    except Exception as e:
        print_failure(f"Installation path is not writable: {install_path_abs}. Error: {e}")
    print_step("Installation path is valid and writable.")

    # 2. Unpack Python
    python_extract_path = unpack_python(install_path_abs)

    # 3. Find Python executable AND enable site-packages
    python_exe = find_python_executable(python_extract_path)
    enable_site_packages(python_extract_path)

    # 4. Bootstrap Pip
    bootstrap_pip(python_exe)

    # 5. Upgrade pip
    run_pip_install(python_exe, "--upgrade pip")

    # 6. Install packages
    required_packages = [
        # "PyQt6", # Keep only if anpe_gui directly needs it AND anpe doesn't depend on it
        "nltk",
        "spacy",
        "benepar",
        "anpe",
        "pyshortcuts", # Keep: Used by installer GUI post-install
        "PyQt6"
    ]
    for package in required_packages:
        run_pip_install(python_exe, package)

    # 7. Copy anpe_gui application code
    copy_app_code(install_path_abs)

    # 8. Copy documentation
    copy_docs(install_path_abs)

    print_success("Environment setup complete.")

if __name__ == "__main__":
    # Basic check, GUI should provide validated path
    if len(sys.argv) != 2 or not sys.argv[1]:
        print_failure("Usage: python installer_core.py <target_install_path>")

    target_install_path = sys.argv[1]
    # Run main logic within a try-except block for catching unexpected errors
    try:
        main(target_install_path)
    except Exception as e:
        # Catch-all for unexpected issues in main()
        import traceback
        print_failure(f"An unexpected error occurred during setup: {e}\n{traceback.format_exc()}")
