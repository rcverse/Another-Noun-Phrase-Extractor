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
import importlib # Added import

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
    """Install the specified spaCy model using pip and spacy download.

    This function first checks if the model is a transformer model. If so, it attempts
    to install `spacy[transformers]` via pip. Then, it proceeds to download the
    specified spaCy model using `python -m spacy download`.

    Args:
        model_name (str): The name of the spaCy model to install (e.g., "en_core_web_trf").
        log_callback (Optional[Callable[[str], None]]): Optional callback for logging.

    Returns:
        bool: True if installation (including spacy-transformers if needed) and download
              were successful, False otherwise.
    """
    full_model_name = SPACY_MODEL_MAP.get(model_name.lower(), model_name) # Ensure we use the full name

    # --- Step 1: Handle spacy-transformers dependency for _trf models ---
    is_transformer_model = full_model_name.endswith("_trf")
    if is_transformer_model:
        msg = f"Model {full_model_name} is a transformer model. Ensuring spacy[transformers] is installed."
        logger.info(msg)
        if log_callback:
            log_callback(msg)

        try:
            # Check if spacy-transformers is already effectively installed and usable
            import spacy as spacy_check_initial
            nlp_blank_initial = spacy_check_initial.blank("en")
            if "transformer" in nlp_blank_initial.factories:
                msg = "'transformer' factory already available in current spaCy runtime. Skipping pip install spacy[transformers]."
                logger.info(msg)
                if log_callback:
                    log_callback(msg)
            else:
                # Attempt to install spacy[transformers]
                pip_command = [sys.executable, "-m", "pip", "install", "spacy[transformers]"]
                msg = f"Running pip install for spacy-transformers: {' '.join(pip_command)}"
                logger.info(msg)
                if log_callback:
                    log_callback(msg)

                process = subprocess.Popen(pip_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                stdout, stderr = process.communicate()
                return_code = process.returncode

                if return_code == 0:
                    msg = "spacy[transformers] installed successfully via pip."
                    logger.info(msg)
                    if log_callback:
                        log_callback(msg)

                    # --- Verification for spacy-transformers ---
                    logger.info("[Verification] Invalidating importlib caches before library/factory check...")
                    importlib.invalidate_caches()
                    
                    spacy_transformers_import_ok = False
                    try:
                        # Try to get a fresh import of spacy_transformers
                        if 'spacy_transformers' in sys.modules:
                            logger.debug("[Verification] Temporarily removing 'spacy_transformers' from sys.modules.")
                            original_spacy_transformers = sys.modules['spacy_transformers']
                            del sys.modules['spacy_transformers']
                            try:
                                import spacy_transformers
                                logger.debug("[Verification] Re-imported 'spacy_transformers' successfully.")
                            finally:
                                # Restore original, whether import succeeded or failed, to avoid breaking other code
                                if 'spacy_transformers' not in sys.modules:
                                     sys.modules['spacy_transformers'] = original_spacy_transformers
                                     logger.debug("[Verification] Restored original 'spacy_transformers' to sys.modules.")
                        else:
                            import spacy_transformers # Standard import if not already in sys.modules
                            logger.debug("[Verification] Imported 'spacy_transformers' freshly.")
                        
                        spacy_transformers_import_ok = True
                        logger.info("[Verification OK] 'spacy-transformers' library imported successfully after pip install.")
                        
                        # Now, attempt the factory check (best effort, for logging/warning)
                        fresh_spacy_module_for_check = None
                        original_spacy_module_in_sys = sys.modules.get('spacy')
                        if original_spacy_module_in_sys:
                            logger.debug("[Verification Factory Check] Temporarily removing 'spacy' from sys.modules.")
                            del sys.modules['spacy']
                        
                        try:
                            import spacy as spacy_for_factory_check
                            fresh_spacy_module_for_check = spacy_for_factory_check
                            if original_spacy_module_in_sys:
                                logger.debug("[Verification Factory Check] Re-imported 'spacy' for factory check.")
                            else:
                                logger.debug("[Verification Factory Check] Freshly imported 'spacy' for factory check.")
                        except ImportError:
                            logger.warning("[Verification Factory Check] Could not import spaCy to check factories.")
                        finally:
                            if original_spacy_module_in_sys and 'spacy' not in sys.modules:
                                sys.modules['spacy'] = original_spacy_module_in_sys
                                logger.debug("[Verification Factory Check] Restored original 'spacy' to sys.modules.")
                            elif original_spacy_module_in_sys and fresh_spacy_module_for_check and id(fresh_spacy_module_for_check) != id(original_spacy_module_in_sys):
                                sys.modules['spacy'] = original_spacy_module_in_sys
                                logger.debug("[Verification Factory Check] Restored original 'spacy' to sys.modules after factory check.")

                        if fresh_spacy_module_for_check:
                            nlp_blank_check = fresh_spacy_module_for_check.blank("en")
                            if "transformer" in nlp_blank_check.factories:
                                logger.info("[Verification Note] 'transformer' factory IS present in current spaCy runtime.")
                            else:
                                logger.warning("[Verification Warning] 'transformer' factory is NOT (yet) present in current spaCy runtime. "
                                               "A restart of the application might be needed to use transformer models.")
                            if "curated_transformer" not in nlp_blank_check.factories:
                                logger.warning("[Verification Note] 'curated_transformer' factory is NOT (yet) present in current spaCy runtime.")
                        else:
                            logger.warning("[Verification Warning] Could not get a spaCy instance to perform post-install factory check.")

                    except ImportError:
                        logger.error("[Verification Failed] CRITICAL: Could not import 'spacy-transformers' library even after pip reported successful installation.")
                        if log_callback:
                            log_callback("Failed to import 'spacy-transformers' post-install.")
                        return False # This is a hard failure for the library itself.
                    except Exception as e_fact: # Catch other errors during the factory check attempt
                        logger.warning(f"[Verification Warning] Non-critical error during post-install factory check attempt: {e_fact}. Proceeding with model download.")

                    # If spacy_transformers_import_ok is True (which it would be to reach here unless an exception above returned False),
                    # we consider the spacy-transformers *dependency* itself "installed".
                    # The final model loadability will be stringently checked after `spacy download`.

                else: # pip install spacy[transformers] failed
                    error_msg = f"Failed to install spacy[transformers]. Pip exit code: {return_code}.\nStdout: {stdout}\nStderr: {stderr}"
                    logger.error(error_msg)
                    if log_callback:
                        log_callback(error_msg)
                    return False # Failed to install spacy-transformers

        except ImportError:
            # This case handles if spacy itself is not installed, which shouldn't happen if ANPE is running.
            # Or if the initial spacy_check_initial fails.
            error_msg = "Failed to import spaCy to check for 'transformer' factory. Cannot proceed with transformer model setup."
            logger.error(error_msg)
            if log_callback:
                log_callback(error_msg)
            return False
        except Exception as e:
            error_msg = f"An unexpected error occurred during spacy-transformers pre-check or installation: {e}"
            logger.error(error_msg)
            if log_callback:
                log_callback(error_msg)
            return False

    # --- Step 2: Download the spaCy model ---
    # Proceed to download the model if spacy-transformers (if needed) is okay or not needed.
    msg = f"Attempting to download spaCy model: {full_model_name}..."
    logger.info(msg)
    if log_callback:
        log_callback(msg)

    command = [sys.executable, "-m", "spacy", "download", full_model_name]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        return_code = process.returncode

        if return_code == 0:
            msg = f"spaCy model '{full_model_name}' downloaded successfully.\nStdout: {stdout}"
            logger.info(msg)
            if log_callback:
                log_callback(msg)
                log_callback(f"stdout: {stdout}") # Pass stdout for GUI display
            # Final check: ensure model is physically present (spaCy download can be weird)
            # And also try to load it as the ultimate verification.
            if _check_spacy_physical_path(full_model_name) and check_spacy_model(full_model_name):
                msg = f"Verification successful: '{full_model_name}' is installed and loadable."
                logger.info(msg)
                if log_callback:
                    log_callback(msg)
                return True
            else:
                error_msg = f"Post-download verification failed for '{full_model_name}'. Model files might be missing or model is not loadable despite successful download command."
                logger.error(error_msg)
                if log_callback:
                    log_callback(error_msg)
                return False
        else:
            error_msg = f"Failed to download spaCy model '{full_model_name}'. Exit code: {return_code}.\nStdout: {stdout}\nStderr: {stderr}"
            logger.error(error_msg)
            if log_callback:
                log_callback(error_msg)
                log_callback(f"stderr: {stderr}") # Pass stderr for GUI display
            return False
    except FileNotFoundError:
        error_msg = f"Error: The command 'python' or 'spacy' was not found. Make sure Python and spaCy are installed and in your PATH."
        logger.error(error_msg)
        if log_callback:
            log_callback(error_msg)
        return False
    except Exception as e:
        error_msg = f"An unexpected error occurred during spaCy model download: {e}"
        logger.error(error_msg)
        if log_callback:
            log_callback(error_msg)
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
