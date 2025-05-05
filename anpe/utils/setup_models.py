#!/usr/bin/env python3
"""Utilities for finding installed ANPE models and selecting the best one to use."""

# --- Standard Logging Setup ---
import logging
logger = logging.getLogger(__name__) # Use standard pattern
# --- End Standard Logging ---

import subprocess
import sys
import nltk
import os
import spacy
import benepar
import shutil
import zipfile
import importlib.metadata
from typing import Callable, Optional  # Add import for callback types
import site

# --- Model Name Mappings ---
# Map user-friendly aliases to actual spaCy model names
SPACY_MODEL_MAP = {
    "sm": "en_core_web_sm",
    "md": "en_core_web_md",
    "lg": "en_core_web_lg",
    "trf": "en_core_web_trf",
    # Also allow direct full names for flexibility
    "en_core_web_sm": "en_core_web_sm",
    "en_core_web_md": "en_core_web_md",
    "en_core_web_lg": "en_core_web_lg",
    "en_core_web_trf": "en_core_web_trf",
}
DEFAULT_SPACY_ALIAS = "md"

# Map user-friendly aliases to actual Benepar model names
BENEPAR_MODEL_MAP = {
    "default": "benepar_en3",
    "large": "benepar_en3_large",
    # Also allow direct full names
    "benepar_en3": "benepar_en3",
    "benepar_en3_large": "benepar_en3_large",
}
DEFAULT_BENEPAR_ALIAS = "default"

# Set up NLTK data path focusing on user's directory
def setup_nltk_data_dir() -> str:
    """Ensures user's NLTK data directory exists and is preferred.

    Returns:
        str: The path to the user's NLTK data directory.
    """
    try:
        # User-specific directory (in home directory)
        home = os.path.expanduser("~")
        nltk_user_dir = os.path.join(home, "nltk_data")

        # Create directory if it doesn't exist
        os.makedirs(nltk_user_dir, exist_ok=True)
        msg = f"Ensured NLTK user data directory exists: {nltk_user_dir}"
        logger.info(msg)

        # Ensure this directory is the first path NLTK checks
        if nltk_user_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_user_dir)
            logger.debug(f"Prepended {nltk_user_dir} to nltk.data.path")

        # Set environment variable for potential subprocess use (like benepar download)
        os.environ['NLTK_DATA'] = nltk_user_dir
        msg = f"Set NLTK_DATA environment variable to: {nltk_user_dir}"
        logger.debug(msg)

        # Verify it's the primary path
        if nltk.data.path[0] != nltk_user_dir:
             msg = f"Expected {nltk_user_dir} to be the first NLTK path, but found {nltk.data.path[0]}. This might cause issues."
             logger.warning(msg)

        return nltk_user_dir

    except PermissionError:
        msg = f"Permission denied creating or accessing NLTK data directory at {nltk_user_dir}. Please check permissions."
        logger.error(msg)
        return nltk_user_dir
    except Exception as e:
        msg = f"Unexpected error during NLTK data directory setup: {e}"
        logger.error(msg)
        # Try returning a default path
        return os.path.join(os.path.expanduser("~"), "nltk_data")

# Get the primary NLTK data directory path
NLTK_DATA_DIR = setup_nltk_data_dir()

# --- Model Checking Functions ---

def check_spacy_model(model_name: str = "en_core_web_md") -> bool:
    """Check if the specified spaCy model is installed and loadable."""
    try:
        spacy.load(model_name)
        logger.debug(f"spaCy model '{model_name}' is present.")
        return True
    except OSError:
        logger.debug(f"spaCy model '{model_name}' not found.")
        return False
    except Exception as e: # Catch other potential errors during loading
        logger.error(f"Error checking spaCy model '{model_name}': {e}")
        return False

def check_benepar_model(model_name: str = "benepar_en3") -> bool:
    """Check if the specified Benepar model exists."""
    try:
        # Use NLTK's find function since Benepar stores models in NLTK's data directory
        model_path = nltk.data.find(f'models/{model_name}')
        logger.debug(f"Benepar model found at: {model_path}")
        return True
    except LookupError:
        logger.debug(f"Benepar model '{model_name}' not found")
        return False
    except Exception as e:
        logger.error(f"Error checking Benepar model: {e}")
        return False

def check_all_models_present(
    spacy_model_alias: str = DEFAULT_SPACY_ALIAS,
    benepar_model_alias: str = DEFAULT_BENEPAR_ALIAS,
    log_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """Check if all required models (specified spaCy/Benepar) are present."""
    # Map aliases to actual names for checking
    # Handle None case for aliases before calling .lower()
    spacy_model_name = SPACY_MODEL_MAP.get(spacy_model_alias.lower() if spacy_model_alias else None, SPACY_MODEL_MAP[DEFAULT_SPACY_ALIAS])
    benepar_model_name = BENEPAR_MODEL_MAP.get(benepar_model_alias.lower() if benepar_model_alias else None, BENEPAR_MODEL_MAP[DEFAULT_BENEPAR_ALIAS])

    msg = f"Checking for presence of specified models (spaCy: {spacy_model_name}, Benepar: {benepar_model_name})..."
    logger.info(msg)
    if log_callback:
        log_callback(msg)
        
    results = {
        "spacy": check_spacy_model(model_name=spacy_model_name),
        "benepar": check_benepar_model(model_name=benepar_model_name),
    }
    all_present = all(results.values())
    if all_present:
        msg = f"All specified models ({spacy_model_name}, {benepar_model_name}) are present."
        logger.info(msg)
        if log_callback:
            log_callback(msg)
    else:
        # Log which specific model is missing
        missing = [name for name, present in results.items() if not present]
        # Create the message string first
        status_msg = f"Status: spaCy({spacy_model_name}): {'Present' if results['spacy'] else 'Missing'}, Benepar({benepar_model_name}): {'Present' if results['benepar'] else 'Missing'}"
        msg = f"One or more specified models are missing: {', '.join(missing)}. {status_msg}"
        logger.warning(msg)
        if log_callback:
            log_callback(msg)
    return all_present

# --- Model Installation Functions ---

def _check_spacy_physical_path(model_name: str) -> bool:
    """Check if a directory named model_name exists in site-packages."""
    site_packages_dirs = site.getsitepackages()
    # Also include user site directory
    user_site = site.getusersitepackages()
    if user_site and os.path.isdir(user_site) and user_site not in site_packages_dirs:
        site_packages_dirs.append(user_site)

    logger.debug(f"[_check_spacy_physical_path] Checking for {model_name} in: {site_packages_dirs}")
    for sp_dir in site_packages_dirs:
        expected_path = os.path.join(sp_dir, model_name)
        if os.path.isdir(expected_path):
            logger.debug(f"[_check_spacy_physical_path] Found physical path: {expected_path}")
            return True
    logger.debug(f"[_check_spacy_physical_path] Physical path for {model_name} not found in site-packages.")
    return False

def _extract_zip_archive(zip_path: str, destination_dir: str, archive_name: str, log_callback: Optional[Callable[[str], None]] = None) -> bool:
    """Extracts a zip archive, removing existing target dir first.

    Args:
        zip_path: Path to the .zip file.
        destination_dir: Directory where the archive should be extracted.
        archive_name: The base name of the archive (used for logging and expected dir).
        log_callback (Optional[Callable[[str], None]]): Optional callback function to receive real-time log output.

    Returns:
        True if extraction was successful or target directory already existed, False otherwise.
    """
    extract_path = os.path.join(destination_dir, archive_name)
    if not os.path.exists(zip_path):
        msg = f"Zip file {zip_path} not found, but target directory {extract_path} exists. Assuming pre-extracted."
        logger.debug(msg)
        if log_callback:
            log_callback(msg)
            
        # If the extracted path already exists, maybe it's okay?
        exists = os.path.exists(extract_path)
        if exists:
            msg = f"Zip file {zip_path} not found, but target directory {extract_path} exists. Assuming pre-extracted."
            logger.debug(msg)
            if log_callback:
                log_callback(msg)
        return exists

    msg = f"Extracting {zip_path}..."
    logger.info(msg)
    if log_callback:
        log_callback(msg)
        
    try:
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
            msg = f"Removed existing directory: {extract_path}"
            logger.debug(msg)
            if log_callback:
                log_callback(msg)
                
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files for progress updates
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            # Log the number of files to extract
            msg = f"Extracting {total_files} files from {archive_name}"
            logger.debug(msg)
            if log_callback:
                log_callback(msg)
                
            # Extract files one by one for progress reporting
            for i, file in enumerate(file_list):
                zip_ref.extract(file, destination_dir)
                # Log progress every 50 files or at the end
                if (i + 1) % 50 == 0 or i == total_files - 1:
                    progress = f"Extraction progress: {i+1}/{total_files} files ({((i+1)/total_files)*100:.1f}%)"
                    logger.debug(progress)
                    if log_callback:
                        log_callback(progress)
                        
        msg = f"Successfully extracted {archive_name} to {extract_path}"
        logger.info(msg)
        if log_callback:
            log_callback(msg)
            
        # Optionally remove the zip file after successful extraction
        os.remove(zip_path)
        msg = f"Removed {zip_path}"
        logger.debug(msg)
        if log_callback:
            log_callback(msg)
            
        return True
    except zipfile.BadZipFile:
        msg = f"Error: {zip_path} is not a valid zip file."
        logger.error(msg)
        if log_callback:
            log_callback(msg)
        return False
    except Exception as e:
        msg = f"Error extracting {zip_path}: {e}"
        logger.error(msg)
        if log_callback:
            log_callback(msg)
        return False

def install_spacy_model(model_name: str = "en_core_web_md", log_callback: Optional[Callable[[str], None]] = None) -> bool:
    """Install the specified spaCy model.
    
    Args:
        model_name (str): Name of the spaCy model to install.
        log_callback (Optional[Callable[[str], None]]): Optional callback function to receive real-time log output.
            
    Returns:
        bool: True if installation was successful, False otherwise.
    """
    if not model_name: # Basic validation
        logger.error("No spaCy model name provided to install.")
        if log_callback:
            log_callback("Error: No spaCy model name provided to install.")
        return False

    is_transformer_model = model_name.endswith("_trf")

    # --- Check and install spacy-transformers if needed ---
    if is_transformer_model:
        msg = f"Model '{model_name}' is a transformer model. Checking for 'spacy-transformers' package..."
        logger.info(msg)
        if log_callback:
            log_callback(msg)
            
        try:
            importlib.metadata.version("spacy-transformers")
            msg = "'spacy-transformers' is already installed."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        except importlib.metadata.PackageNotFoundError:
            msg = "'spacy-transformers' package not found. Attempting installation..."
            logger.warning(msg)
            if log_callback:
                log_callback(msg)
                
            try:
                cmd = [sys.executable, '-m', 'pip', 'install', "spacy[transformers]"]
                msg = f"Running command: {' '.join(cmd)}"
                logger.info(msg)
                if log_callback:
                    log_callback(msg)
                    
                # Use Popen instead of run for real-time output
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    bufsize=1
                )
                
                # Process output in real time
                for line in process.stdout:
                    line = line.strip()
                    logger.debug(f"pip install stdout: {line}")
                    if log_callback:
                        log_callback(line)
                
                # Wait for process to complete and check return code
                return_code = process.wait()
                if return_code == 0:
                    msg = "'spacy-transformers' installed successfully."
                    logger.info(msg)
                    if log_callback:
                        log_callback(msg)
                else:
                    msg = f"Failed to install 'spacy-transformers'. Return code: {return_code}"
                    logger.error(msg)
                    if log_callback:
                        log_callback(msg)
                    # Continue anyway, but warn the user
                    msg = "Will attempt to continue with spaCy model download, but it may fail later."
                    logger.warning(msg)
                    if log_callback:
                        log_callback(msg)
            except Exception as e:
                msg = f"An unexpected error occurred during 'spacy-transformers' installation: {e}"
                logger.error(msg)
                if log_callback:
                    log_callback(msg)
                msg = "Will attempt to continue with spaCy model download, but it may fail later."
                logger.warning(msg)
                if log_callback:
                    log_callback(msg)

    # --- Download the spaCy model ---
    try:
        msg = f"Attempting to download and install spaCy model: {model_name}"
        logger.info(msg)
        if log_callback:
            log_callback(msg)
            
        cmd = [sys.executable, '-m', 'spacy', 'download', model_name]
        msg = f"Running command: {' '.join(cmd)}"
        logger.info(msg)
        if log_callback:
            log_callback(msg)
            
        # Use Popen instead of run for real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        
        # Process output in real time
        for line in process.stdout:
            line = line.strip()
            logger.debug(f"spaCy download stdout: {line}")
            if log_callback:
                log_callback(line)
        
        # Wait for process to complete and check return code
        return_code = process.wait()
        if return_code == 0:
            msg = f"spaCy model '{model_name}' download command finished successfully."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
                
            # --- Verification Step 1: Use importlib.util.find_spec --- 
            msg = f"Verifying installation for '{model_name}' using importlib..."
            logger.info(msg)
            if log_callback:
                log_callback(msg)

            importlib.invalidate_caches()
            importlib_verified = False
            importlib_error = False
            physical_path_verified = False # Added for secondary check
            
            try:
                spec = importlib.util.find_spec(model_name)
                if spec and spec.origin:
                    msg = f"Verification Step 1 OK: '{model_name}' package found by importlib at {spec.origin}."
                    logger.info(msg)
                    if log_callback:
                        log_callback(msg)
                    importlib_verified = True
                else:
                    # This case means find_spec found something, but not a valid package origin
                    msg = f"Verification Step 1 Failed: importlib found '{model_name}' but no valid origin/spec."
                    logger.warning(msg) # Warning, as we'll do a secondary check
                    if log_callback:
                        log_callback(msg)
                    # Proceed to physical path check
            except ModuleNotFoundError:
                # find_spec couldn't find the module via import system path
                msg = f"Verification Step 1 Failed: ModuleNotFoundError for '{model_name}'."
                logger.warning(msg) # Warning, as we'll do a secondary check
                if log_callback:
                    log_callback(msg)
                # Proceed to physical path check
            except Exception as find_err:
                # Other errors during find_spec
                msg = f"Verification Step 1 Error: An unexpected error occurred while trying to find spec for '{model_name}': {find_err}"
                logger.error(msg, exc_info=True)
                if log_callback:
                    log_callback(msg)
                importlib_error = True # Mark as error, skip physical check

            # --- Verification Step 2: Physical Path Check (if importlib failed/was inconclusive) --- 
            if not importlib_verified and not importlib_error:
                msg = f"Verification Step 2: Checking physical path for '{model_name}' in site-packages..."
                logger.info(msg)
                if log_callback:
                    log_callback(msg)
                physical_path_verified = _check_spacy_physical_path(model_name)
                if physical_path_verified:
                    msg = f"Verification Step 2 OK: Physical directory found for '{model_name}'. Assuming OK for next run."
                    logger.warning(msg) # Warn that importlib didn't verify, but files exist
                    if log_callback:
                        log_callback(msg)
                else:
                     msg = f"Verification Step 2 Failed: Physical directory for '{model_name}' not found in site-packages."
                     logger.error(msg)
                     if log_callback:
                         log_callback(msg)
            
            # --- Final Decision --- 
            if importlib_verified or physical_path_verified:
                 return True # Consider it successful if either check passed
            else:
                 # Only return False if both importlib failed/errored AND physical path check failed
                 msg = f"Installation failed for '{model_name}': Could not verify via importlib or physical path."
                 logger.error(msg)
                 if log_callback:
                     log_callback(msg)
                 return False
                 
        else: # download return_code != 0
            msg = f"Failed to download spaCy model '{model_name}'. Return code: {return_code}"
            logger.error(msg)
            if log_callback:
                log_callback(msg)
            return False
    except Exception as e:
        msg = f"An unexpected error occurred during spaCy model installation for '{model_name}': {e}"
        logger.error(msg, exc_info=True)
        if log_callback:
            log_callback(msg)
        return False

def install_benepar_model(model_name: str = "benepar_en3", log_callback: Optional[Callable[[str], None]] = None) -> bool:
    """Install the specified Benepar model, with manual extraction fallback.
    
    Args:
        model_name (str): Name of the Benepar model to install.
        log_callback (Optional[Callable[[str], None]]): Optional callback function to receive real-time log output.
            
    Returns:
        bool: True if installation was successful, False otherwise.
    """
    if not model_name: # Basic validation
        msg = "No Benepar model name provided to install."
        logger.error(msg)
        if log_callback:
            log_callback(msg)
        return False
        
    models_dir = os.path.join(NLTK_DATA_DIR, "models")
    model_dir_path = os.path.join(models_dir, model_name)
    model_zip_path = os.path.join(models_dir, f"{model_name}.zip")
    subprocess_ok = False # Track subprocess success
    files_ok_after_attempt = False # Track if files seem okay after attempt

    # --- Start: Added logic to remove orphan zip --- 
    if not os.path.exists(model_dir_path) and os.path.exists(model_zip_path):
        msg = f"Found existing zip file {model_zip_path} but missing directory {model_dir_path}. " \
              f"Removing zip before attempting download to ensure proper extraction."
        logger.info(msg)
        if log_callback:
            log_callback(msg)
            
        try:
            os.remove(model_zip_path)
            msg = f"Removed existing zip: {model_zip_path}"
            logger.debug(msg)
            if log_callback:
                log_callback(msg)
        except OSError as remove_err:
            # Log as error because failure here might prevent successful download
            msg = f"Failed to remove existing zip {model_zip_path}: {remove_err}. " \
                  f"Download might fail or use the corrupted zip."
            logger.error(msg)
            if log_callback:
                log_callback(msg)
            # Optionally, could return False here if removing the zip is critical 
            # return False 
    # --- End: Added logic ---

    try:
        msg = f"Attempting to download Benepar model '{model_name}' using subprocess..."
        logger.info(msg)
        if log_callback:
            log_callback(msg)
            
        # Ensure NLTK_DATA is in the environment for the subprocess
        current_env = dict(os.environ)
        current_env['NLTK_DATA'] = NLTK_DATA_DIR
        msg = f"Subprocess environment will use NLTK_DATA: {current_env.get('NLTK_DATA')}"
        logger.debug(msg)
        if log_callback:
            log_callback(msg)

        # Escape backslashes in the path for the command string
        escaped_nltk_data_dir = NLTK_DATA_DIR.replace('\\', '\\\\') # Double escape needed for f-string then command

        # Create the Python command as a string
        py_command = f"import nltk; import benepar; nltk.data.path.insert(0, r'{escaped_nltk_data_dir}'); benepar.download('{model_name}')"
        
        # Use Popen instead of run for real-time output
        cmd = [sys.executable, '-c', py_command]
        msg = f"Running command: {' '.join(cmd)}"
        logger.info(msg)
        if log_callback:
            log_callback(msg)
            
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=current_env
        )
        
        # Process output in real time
        for line in process.stdout:
            line = line.strip()
            logger.debug(f"Benepar download stdout: {line}")
            if log_callback:
                log_callback(line)
        
        # Wait for process to complete with timeout
        try:
            return_code = process.wait(timeout=1200)  # 20 minutes timeout
            if return_code == 0:
                msg = f"Benepar download subprocess finished successfully for '{model_name}'."
                logger.info(msg)
                if log_callback:
                    log_callback(msg)
                subprocess_ok = True
            else:
                msg = f"Benepar download subprocess failed for '{model_name}' with return code {return_code}."
                logger.warning(msg)
                if log_callback:
                    log_callback(msg)
                subprocess_ok = False
                # If subprocess failed, return False immediately
                return False 
        except subprocess.TimeoutExpired:
            msg = f"Benepar download subprocess timed out for '{model_name}'."
            logger.error(msg)
            if log_callback:
                log_callback(msg)
            process.kill()
            return False

        # This part is now only reached if subprocess_ok was potentially True
        # Check if the model directory or extracted zip exists now
        if os.path.isdir(model_dir_path):
            msg = f"Benepar model directory found at: {model_dir_path}"
            logger.info(msg)
            if log_callback:
                log_callback(msg)
            files_ok_after_attempt = True
        else:
            msg = f"Benepar model directory not found at {model_dir_path}. Checking for zip file..."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
                
            if os.path.isfile(model_zip_path):
                msg = f"Found zip file: {model_zip_path}. Attempting manual extraction."
                logger.info(msg)
                if log_callback:
                    log_callback(msg)
                    
                if _extract_zip_archive(model_zip_path, models_dir, model_name, log_callback):
                    msg = f"Manual extraction of {model_name} successful."
                    logger.info(msg)
                    if log_callback:
                        log_callback(msg)
                    files_ok_after_attempt = True # Extracted successfully
                else:
                    msg = f"Manual extraction of {model_zip_path} failed."
                    logger.error(msg)
                    if log_callback:
                        log_callback(msg)
                    files_ok_after_attempt = False # Extraction failed
            else:
                msg = f"Neither Benepar model directory nor zip file found after download attempt."
                logger.warning(msg)
                if log_callback:
                    log_callback(msg)
                files_ok_after_attempt = False

        # Final verification only makes sense if files seemed okay after the attempt
        if files_ok_after_attempt:
            msg = f"Verifying Benepar model '{model_name}' using check function..."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
                
            if check_benepar_model(model_name):
                msg = f"Benepar model '{model_name}' is present and verified."
                logger.info(msg)
                if log_callback:
                    log_callback(msg)
                # Return True only if the final check passes
                return True
            else:
                msg = f"Benepar model '{model_name}' verification failed even though files seemed present."
                logger.error(msg)
                if log_callback:
                    log_callback(msg)
                return False
        else:
             # If files weren't okay after attempt, the installation failed.
             msg = f"Installation failed for Benepar model '{model_name}' - files not found or extraction failed."
             logger.error(msg)
             if log_callback:
                 log_callback(msg)
             return False

    except Exception as e:
        msg = f"An unexpected error occurred during Benepar model '{model_name}' installation: {e}"
        logger.error(msg, exc_info=True)
        if log_callback:
            log_callback(msg)
        # Log stderr if it was a CalledProcessError originally wrapped
        if hasattr(e, 'stderr'):
             msg = f"Subprocess stderr: {e.stderr}"
             logger.error(msg)
             if log_callback:
                 log_callback(msg)
        return False

def setup_models(
    spacy_model_alias: str = DEFAULT_SPACY_ALIAS,
    benepar_model_alias: str = DEFAULT_BENEPAR_ALIAS,
    log_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Checks for required models (specified spaCy/Benepar) and attempts
    to install any that are missing. Uses user-friendly aliases for models.

    Args:
        spacy_model_alias (str): Alias for spaCy model ('sm', 'md', 'lg', 'trf').
                                 Defaults to 'md'.
        benepar_model_alias (str): Alias for Benepar model ('default', 'large').
                                   Defaults to 'default'.
        log_callback (Optional[Callable[[str], None]]): Optional callback function to receive 
                                                        real-time log output.

    Returns:
        bool: True if all required models (based on selected aliases) are
              present after the check/install process, False otherwise.
    """
    msg = f"Starting model setup process for spaCy='{spacy_model_alias}', benepar='{benepar_model_alias}'..."
    logger.info(msg)
    if log_callback:
        log_callback(msg)

    final_status = True  # Tracks if all models are OK by the end
    spacy_needed = bool(spacy_model_alias)
    benepar_needed = bool(benepar_model_alias)

    actual_spacy_model = None
    if spacy_needed:
        spacy_alias_lower = spacy_model_alias.lower()
        actual_spacy_model = SPACY_MODEL_MAP.get(spacy_alias_lower)
        if not actual_spacy_model:
            msg = f"Error: Invalid spaCy model alias '{spacy_model_alias}'. Available: {list(SPACY_MODEL_MAP.keys())}."
            logger.error(msg)
            if log_callback:
                log_callback(msg)
            return False # Fail immediately on invalid alias
    else:
        msg = "spaCy model setup skipped (alias not provided)."
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    actual_benepar_model = None
    if benepar_needed:
        benepar_alias_lower = benepar_model_alias.lower()
        actual_benepar_model = BENEPAR_MODEL_MAP.get(benepar_alias_lower)
        if not actual_benepar_model:
            msg = f"Error: Invalid Benepar model alias '{benepar_model_alias}'. Available: {list(BENEPAR_MODEL_MAP.keys())}."
            logger.error(msg)
            if log_callback:
                log_callback(msg)
            return False # Fail immediately on invalid alias
    else:
        msg = "Benepar model setup skipped (alias not provided)."
        logger.info(msg)
        if log_callback:
            log_callback(msg)

    # --- Check and Install spaCy ---
    if spacy_needed:
        msg_check = f"Checking for required spaCy model '{actual_spacy_model}' (alias '{spacy_model_alias}')..."
        logger.info(msg_check)
        if log_callback:
            log_callback(msg_check)

        if check_spacy_model(model_name=actual_spacy_model):
            msg = f"Required spaCy model '{actual_spacy_model}' already installed."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        else:
            msg = f"Required spaCy model '{actual_spacy_model}' not found. Attempting installation..."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
            if not install_spacy_model(actual_spacy_model, log_callback=log_callback):
                msg = f"Failed to install required spaCy model '{actual_spacy_model}'."
                logger.error(msg)
                if log_callback:
                    log_callback(msg)
                final_status = False
            else:
                 msg = f"Successfully installed spaCy model '{actual_spacy_model}'."
                 logger.info(msg)
                 if log_callback:
                     log_callback(msg)

    # --- Check and Install Benepar ---
    if benepar_needed:
        msg_check = f"Checking for required Benepar model '{actual_benepar_model}' (alias '{benepar_model_alias}')..."
        logger.info(msg_check)
        if log_callback:
            log_callback(msg_check)

        if check_benepar_model(model_name=actual_benepar_model):
            msg = f"Required Benepar model '{actual_benepar_model}' already installed."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        else:
            msg = f"Required Benepar model '{actual_benepar_model}' not found. Attempting installation..."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
            if not install_benepar_model(actual_benepar_model, log_callback=log_callback):
                msg = f"Failed to install required Benepar model '{actual_benepar_model}'."
                logger.error(msg)
                if log_callback:
                    log_callback(msg)
                final_status = False
            else:
                msg = f"Successfully installed Benepar model '{actual_benepar_model}'."
                logger.info(msg)
                if log_callback:
                    log_callback(msg)

    # --- Final Status ---
    if final_status:
        msg = "Model setup process completed successfully."
        logger.info(msg)
        if log_callback:
            log_callback(msg)
    else:
        msg = "Model setup process failed for one or more models. Please review logs."
        logger.error(msg)
        if log_callback:
            log_callback(msg)

    return final_status

def main(log_callback: Optional[Callable[[str], None]] = None) -> int:
    """
    Entry point for CLI invocation.
    
    Args:
        log_callback (Optional[Callable[[str], None]]): Optional callback for real-time logging.
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # This main is primarily for direct CLI invocation: `python -m anpe.utils.setup_models`
    # Configure logging if run directly (Application should configure otherwise)
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Running ANPE Model Setup Utility (using default models) --- ")
    
    # First check if the default models are already present
    if check_all_models_present(log_callback=log_callback): # Uses default arguments
        logger.info("--- All required default models are already present. No installation needed. --- ")
        return 0
    
    # If not all default models are present, run the setup process with defaults
    if setup_models(log_callback=log_callback): # Uses default arguments + callback
        logger.info("--- Setup Complete: All required default models are now present. --- ")
        return 0
    else:
        logger.error("--- Setup Failed: One or more default models could not be installed or verified. Please check logs above. --- ")
        return 1

if __name__ == "__main__":
    # Removed sys.exit call, let main return the code
    main() # Calls main which configures basic logging
