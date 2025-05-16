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

# Define canonical lists of aliases intended for "install all" functionality
INSTALLABLE_SPACY_ALIASES = ["sm", "md", "lg", "trf"]
INSTALLABLE_BENEPAR_ALIASES = ["default", "large"]

# Set up NLTK data path focusing on user's directory
def setup_nltk_data_dir() -> str:
    """Ensures user's NLTK data directory exists and is preferred.

    Returns:
        str: The path to the user's NLTK data directory.
    """
    logger.info("--- Starting: NLTK Data Directory Setup ---")
    nltk_user_dir_base = os.path.join(os.path.expanduser("~"), "nltk_data") # Define potential path for error messages

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
        
        logger.info(f"--- Finished: NLTK Data Directory Setup (Path: {nltk_user_dir}) ---")
        return nltk_user_dir

    except PermissionError:
        msg = f"Permission denied creating or accessing NLTK data directory at {nltk_user_dir_base}. Please check permissions."
        logger.error(msg)
        logger.error("--- Finished: NLTK Data Directory Setup (Error: Permission Denied) ---")
        return nltk_user_dir_base # Return the base path even on error
    except Exception as e:
        msg = f"Unexpected error during NLTK data directory setup: {e}"
        logger.error(msg)
        logger.error(f"--- Finished: NLTK Data Directory Setup (Error: {e}) ---")
        # Try returning a default path
        return nltk_user_dir_base # Return the base path even on error

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
    log_entry = "--- Starting: Check All Models Present ---"
    logger.info(log_entry)
    if log_callback: log_callback(log_entry)

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
        log_entry = f"--- Finished: Check All Models Present (Result: All Found: spaCy='{spacy_model_name}', Benepar='{benepar_model_name}') ---"
        logger.info(log_entry)
        if log_callback: log_callback(log_entry)
    else:
        # Log which specific model is missing
        missing = [name for name, present in results.items() if not present]
        # Create the message string first
        status_msg = f"Status: spaCy({spacy_model_name}): {'Present' if results['spacy'] else 'Missing'}, Benepar({benepar_model_name}): {'Present' if results['benepar'] else 'Missing'}"
        msg = f"One or more specified models are missing: {', '.join(missing)}. {status_msg}"
        logger.warning(msg)
        if log_callback:
            log_callback(msg)
        log_entry = f"--- Finished: Check All Models Present (Result: Missing Models - {', '.join(missing)}. Status: {status_msg}) ---"
        logger.warning(log_entry)
        if log_callback: log_callback(log_entry)
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
    
    log_entry = f"--- Starting: ZIP Extraction ({archive_name}) ---"
    logger.info(log_entry)
    if log_callback: log_callback(log_entry)

    if not os.path.exists(zip_path):
        msg = f"Zip file {zip_path} not found, but target directory {extract_path} exists. Assuming pre-extracted."
        logger.debug(msg)
        if log_callback:
            log_callback(msg)
            
        # If the extracted path already exists, maybe it's okay?
        if os.path.exists(extract_path):
            msg = f"Zip file {zip_path} not found, but target directory {extract_path} exists. Assuming pre-extracted."
            logger.debug(msg)
            if log_callback:
                log_callback(msg)
            log_entry_finish = f"--- Finished: ZIP Extraction ({archive_name}) (Result: Skipped, target exists) ---"
            logger.info(log_entry_finish)
            if log_callback: log_callback(log_entry_finish)
            return True # Assuming it's okay if target dir already exists
        else:
            msg = f"Zip file {zip_path} not found, and target directory {extract_path} does not exist. Cannot proceed."
            logger.warning(msg)
            if log_callback:
                log_callback(msg)
            log_entry_finish = f"--- Finished: ZIP Extraction ({archive_name}) (Result: Failed, zip and target missing) ---"
            logger.warning(log_entry_finish)
            if log_callback: log_callback(log_entry_finish)
            return False

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
            
        log_entry_finish = f"--- Finished: ZIP Extraction ({archive_name}) (Result: Success) ---"
        logger.info(log_entry_finish)
        if log_callback: log_callback(log_entry_finish)
        return True
    except zipfile.BadZipFile:
        msg = f"Error: {zip_path} is not a valid zip file."
        logger.error(msg)
        if log_callback:
            log_callback(msg)
        log_entry_finish = f"--- Finished: ZIP Extraction ({archive_name}) (Result: Failed, BadZipFile) ---"
        logger.error(log_entry_finish)
        if log_callback: log_callback(log_entry_finish)
        return False
    except Exception as e:
        msg = f"Error extracting {zip_path}: {e}"
        logger.error(msg)
        if log_callback:
            log_callback(msg)
        log_entry_finish = f"--- Finished: ZIP Extraction ({archive_name}) (Result: Failed, Exception: {e}) ---"
        logger.error(log_entry_finish)
        if log_callback: log_callback(log_entry_finish)
        return False

def install_spacy_model(model_name: str = "en_core_web_md", log_callback: Optional[Callable[[str], None]] = None) -> bool:
    """Install the specified spaCy model using pip and spacy download.

    This function first checks if the model is a transformer model. If so, it attempts
    to install `spacy[transformers]` via pip. Then, it proceeds to download the
    specified spaCy model using `python -m spacy download`.

    Args:
        model_name (str): The alias (e.g., 'md', 'trf') or full name (e.g., 'en_core_web_md') of the spaCy model to install.
        log_callback (Optional[Callable[[str], None]]): Optional callback for logging.

    Returns:
        bool: True if installation (including spacy-transformers if needed) and download
              were successful, False otherwise.
    """
    full_model_name = SPACY_MODEL_MAP.get(model_name.lower(), model_name) # Ensure we use the full name
    log_entry_start = f"--- Starting: Install spaCy Model ({full_model_name}) ---"
    logger.info(log_entry_start)
    if log_callback: log_callback(log_entry_start)

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
                logger.info("  [Action]: Installing spacy[transformers] via pip...")

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
                        logger.error(f"--- Aborting: Install spaCy Model ({full_model_name}) (Error: spacy-transformers import failed post-pip) ---")
                        if log_callback: log_callback(f"--- Aborting: Install spaCy Model ({full_model_name}) (Error: spacy-transformers import failed post-pip) ---")
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
                    logger.error(f"--- Aborting: Install spaCy Model ({full_model_name}) (Error: pip install spacy[transformers] failed) ---")
                    if log_callback: log_callback(f"--- Aborting: Install spaCy Model ({full_model_name}) (Error: pip install spacy[transformers] failed) ---")
                    return False # Failed to install spacy-transformers

        except ImportError:
            # This case handles if spacy itself is not installed, which shouldn't happen if ANPE is running.
            # Or if the initial spacy_check_initial fails.
            error_msg = "Failed to import spaCy to check for 'transformer' factory. Cannot proceed with transformer model setup."
            logger.error(error_msg)
            if log_callback:
                log_callback(error_msg)
            logger.error(f"--- Aborting: Install spaCy Model ({full_model_name}) (Error: spaCy import failed for transformer check) ---")
            if log_callback: log_callback(f"--- Aborting: Install spaCy Model ({full_model_name}) (Error: spaCy import failed for transformer check) ---")
            return False
        except Exception as e:
            error_msg = f"An unexpected error occurred during spacy-transformers pre-check or installation: {e}"
            logger.error(error_msg)
            if log_callback:
                log_callback(error_msg)
            logger.error(f"--- Aborting: Install spaCy Model ({full_model_name}) (Error: Unexpected error in spacy-transformers setup: {e}) ---")
            if log_callback: log_callback(f"--- Aborting: Install spaCy Model ({full_model_name}) (Error: Unexpected error in spacy-transformers setup: {e}) ---")
            return False

    # --- Step 2: Download the spaCy model ---
    # Proceed to download the model if spacy-transformers (if needed) is okay or not needed.
    msg = f"Attempting to download spaCy model: {full_model_name}..."
    logger.info(msg)
    if log_callback:
        log_callback(msg)
    logger.info(f"  [Action]: Downloading spaCy model '{full_model_name}' using 'spacy download'...")

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
            # --- Start of modified logic for post-download verification ---
            is_physically_present = _check_spacy_physical_path(full_model_name)

            if not is_physically_present:
                error_msg = f"Post-download check failed: Model files for '{full_model_name}' are missing despite a successful download command. Please try again."
                logger.error(error_msg)
                if log_callback:
                    log_callback(error_msg)
                log_entry_finish = f"--- Finished: Install spaCy Model ({full_model_name}) (Result: Failed, physical files missing post-download) ---"
                logger.error(log_entry_finish)
                if log_callback: log_callback(log_entry_finish)
                return False
            else: # Files are physically present
                path_msg = f"Physical files for '{full_model_name}' confirmed present."
                logger.info(path_msg)
                if log_callback:
                    log_callback(path_msg)
                
                is_loadable = check_spacy_model(full_model_name)

                if is_loadable:
                    success_msg = f"Verification successful: '{full_model_name}' is installed and loadable."
                    logger.info(success_msg)
                    if log_callback:
                        log_callback(success_msg)
                    log_entry_finish = f"--- Finished: Install spaCy Model ({full_model_name}) (Result: Success, loadable) ---"
                    logger.info(log_entry_finish)
                    if log_callback: log_callback(log_entry_finish)
                    return True
                else: # Not loadable
                    # is_transformer_model is defined at the start of the function
                    if is_transformer_model:
                        # Special case: Transformer model installed, files present, but not loading (likely needs restart)
                        restart_needed_msg = f"SUCCESS_RESTART_NEEDED: spaCy transformer model '{full_model_name}' installed successfully. An application restart or Python environment refresh is likely required to activate and use this model."
                        logger.warning(restart_needed_msg) # Log as warning due to the caveat
                        if log_callback:
                            log_callback(restart_needed_msg)
                        log_entry_finish = f"--- Finished: Install spaCy Model ({full_model_name}) (Result: Success, Restart Needed for Transformer) ---"
                        logger.info(log_entry_finish) # Info because install itself was "successful"
                        if log_callback: log_callback(log_entry_finish)
                        return True # Installation considered successful, but needs restart
                    else:
                        # Non-transformer model, files present, but not loadable - this is an unexpected error
                        error_msg = f"Post-download verification failed for non-transformer model '{full_model_name}'. Model files are present, but the model is not loadable. This may indicate a deeper issue with the model or spaCy environment."
                        logger.error(error_msg)
                        if log_callback:
                            log_callback(error_msg)
                        log_entry_finish = f"--- Finished: Install spaCy Model ({full_model_name}) (Result: Failed, present but not loadable) ---"
                        logger.error(log_entry_finish)
                        if log_callback: log_callback(log_entry_finish)
                        return False
            # --- End of modified logic ---
        else:
            error_msg = f"Failed to download spaCy model '{full_model_name}'. Exit code: {return_code}.\nStdout: {stdout}\nStderr: {stderr}"
            logger.error(error_msg)
            if log_callback:
                log_callback(error_msg)
                log_callback(f"stderr: {stderr}") # Pass stderr for GUI display
            log_entry_finish = f"--- Finished: Install spaCy Model ({full_model_name}) (Result: Failed, spacy download command error) ---"
            logger.error(log_entry_finish)
            if log_callback: log_callback(log_entry_finish)
            return False
    except FileNotFoundError:
        error_msg = f"Error: The command 'python' or 'spacy' was not found. Make sure Python and spaCy are installed and in your PATH."
        logger.error(error_msg)
        if log_callback:
            log_callback(error_msg)
        log_entry_finish = f"--- Finished: Install spaCy Model ({full_model_name}) (Result: Failed, FileNotFoundError for python/spacy) ---"
        logger.error(log_entry_finish)
        if log_callback: log_callback(log_entry_finish)
        return False
    except Exception as e:
        error_msg = f"An unexpected error occurred during spaCy model download: {e}"
        logger.error(error_msg)
        if log_callback:
            log_callback(error_msg)
        log_entry_finish = f"--- Finished: Install spaCy Model ({full_model_name}) (Result: Failed, Unexpected Exception: {e}) ---"
        logger.error(log_entry_finish)
        if log_callback: log_callback(log_entry_finish)
        return False

def install_benepar_model(model_name: str = "benepar_en3", log_callback: Optional[Callable[[str], None]] = None) -> bool:
    """Install the specified Benepar model, with manual extraction fallback.
    
    Args:
        model_name (str): Alias (e.g., 'default') or full name (e.g., 'benepar_en3') of the Benepar model to install.
        log_callback (Optional[Callable[[str], None]]): Optional callback function to receive real-time log output.
            
    Returns:
        bool: True if installation was successful, False otherwise.
    """
    # Resolve alias to full model name, or use model_name if it's already a full name
    # .lower() is used for the lookup key if model_name is an alias.
    # If model_name is not in the map (e.g., it's already a full name), it will be used as is.
    resolved_model_name = BENEPAR_MODEL_MAP.get(model_name.lower() if isinstance(model_name, str) else model_name, model_name)

    log_entry_start = f"--- Starting: Install Benepar Model (Input: '{model_name}', Resolved: '{resolved_model_name}') ---"
    logger.info(log_entry_start)
    if log_callback: log_callback(log_entry_start)

    if not resolved_model_name: # Should ideally not happen if model_name is valid
        msg = f"Cannot resolve Benepar model identifier: '{model_name}'."
        logger.error(msg)
        if log_callback:
            log_callback(msg)
        log_entry_finish = f"--- Finished: Install Benepar Model ('{model_name}') (Result: Failed, Could not resolve identifier) ---"
        logger.error(log_entry_finish)
        if log_callback: log_callback(log_entry_finish)
        return False
        
    models_dir = os.path.join(NLTK_DATA_DIR, "models")
    model_dir_path = os.path.join(models_dir, resolved_model_name)
    model_zip_path = os.path.join(models_dir, f"{resolved_model_name}.zip")
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
    logger.info(f"  Target NLTK_DATA_DIR for Benepar: {NLTK_DATA_DIR}")
    logger.info(f"  Expected model directory: {model_dir_path}")
    logger.info(f"  Expected model zip: {model_zip_path}")

    try:
        msg = f"Attempting to download Benepar model '{resolved_model_name}' using subprocess..."
        logger.info(msg)
        if log_callback:
            log_callback(msg)
            
        # Ensure NLTK_DATA is in the environment for the subprocess
        current_env = dict(os.environ)
        current_env['NLTK_DATA'] = NLTK_DATA_DIR
        msg = f"Subprocess environment will use NLTK_DATA: {current_env.get('NLTK_DATA')}"
        logger.debug(msg)
        # No need to callback debug usually
        # if log_callback:
        #     log_callback(msg)

        # Escape backslashes in the path for the command string
        escaped_nltk_data_dir = NLTK_DATA_DIR.replace('\\', '\\\\') # Double escape needed for f-string then command

        # Create the Python command as a string
        py_command = f"import nltk; import benepar; nltk.data.path.insert(0, r'{escaped_nltk_data_dir}'); benepar.download('{resolved_model_name}')"
        
        # Use Popen instead of run for real-time output
        cmd = [sys.executable, '-c', py_command]
        msg = f"  [Action]: Running Benepar download command: {' '.join(cmd)}"
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
                msg = f"Benepar download subprocess finished successfully for '{resolved_model_name}'."
                logger.info(msg)
                if log_callback:
                    log_callback(msg)
                subprocess_ok = True
            else:
                msg = f"Benepar download subprocess failed for '{resolved_model_name}' with return code {return_code}."
                logger.warning(msg)
                if log_callback:
                    log_callback(msg)
                subprocess_ok = False
                # If subprocess failed, return False immediately
                # Make sure to log the finish status before returning
                log_entry_finish = f"--- Finished: Install Benepar Model ('{resolved_model_name}') (Result: Failed, subprocess error code {return_code}) ---"
                logger.error(log_entry_finish)
                if log_callback: log_callback(log_entry_finish)
                return False 
        except subprocess.TimeoutExpired:
            msg = f"Benepar download subprocess timed out for '{resolved_model_name}'."
            logger.error(msg)
            if log_callback:
                log_callback(msg)
            process.kill()
            # Log finish status
            log_entry_finish = f"--- Finished: Install Benepar Model ('{resolved_model_name}') (Result: Failed, subprocess timeout) ---"
            logger.error(log_entry_finish)
            if log_callback: log_callback(log_entry_finish)
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
                    
                if _extract_zip_archive(model_zip_path, models_dir, resolved_model_name, log_callback):
                    msg = f"Manual extraction of {resolved_model_name} successful."
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
            msg = f"Verifying Benepar model '{resolved_model_name}' using check function..."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
                
            if check_benepar_model(resolved_model_name): # Pass resolved_model_name here
                msg = f"Benepar model '{resolved_model_name}' is present and verified."
                logger.info(msg)
                if log_callback:
                    log_callback(msg)
                # Return True only if the final check passes
                log_entry_finish = f"--- Finished: Install Benepar Model ('{resolved_model_name}') (Result: Success, verified) ---"
                logger.info(log_entry_finish)
                if log_callback: log_callback(log_entry_finish)
                return True
            else:
                msg = f"Benepar model '{resolved_model_name}' verification failed even though files seemed present."
                logger.error(msg)
                if log_callback:
                    log_callback(msg)
                log_entry_finish = f"--- Finished: Install Benepar Model ('{resolved_model_name}') (Result: Failed, verification check failed) ---"
                logger.error(log_entry_finish)
                if log_callback: log_callback(log_entry_finish)
                return False
        else:
             # If files weren't okay after attempt, the installation failed.
             msg = f"Installation failed for Benepar model '{resolved_model_name}' - files not found or extraction failed."
             logger.error(msg)
             if log_callback:
                 log_callback(msg)
             log_entry_finish = f"--- Finished: Install Benepar Model ('{resolved_model_name}') (Result: Failed, files not found or extraction failed) ---"
             logger.error(log_entry_finish)
             if log_callback: log_callback(log_entry_finish)
             return False

    except Exception as e:
        msg = f"An unexpected error occurred during Benepar model '{resolved_model_name}' installation: {e}"
        logger.error(msg, exc_info=True) # exc_info=True is good for unexpected errors
        if log_callback:
            log_callback(msg)
        # Log stderr if it was a CalledProcessError originally wrapped
        if hasattr(e, 'stderr'):
             err_stderr_msg = f"Subprocess stderr: {e.stderr}"
             logger.error(err_stderr_msg)
             if log_callback:
                 log_callback(err_stderr_msg)
        log_entry_finish = f"--- Finished: Install Benepar Model ('{resolved_model_name}') (Result: Failed, Unexpected Exception: {e}) ---"
        logger.error(log_entry_finish)
        if log_callback: log_callback(log_entry_finish)
        return False

def setup_models(
    spacy_model_alias: Optional[str] = None, 
    benepar_model_alias: Optional[str] = None,
    log_callback: Optional[Callable[[str], None]] = None
) -> bool:
    """
    Set up specified spaCy and Benepar models.
    If an alias is None, attempts to install/verify the default model for that type.
    Returns True if all necessary operations succeeded, False otherwise.
    """
    
    # Helper for logging via callback or logger
    def _log(msg: str, level: str = "INFO"):
        if log_callback:
            log_callback(msg) # Callback usually implies INFO level for CLI
        
        # Map level string to logger method more robustly
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(msg)


    _log(f"--- Starting: Overall Model Setup Process (spaCy='{spacy_model_alias if spacy_model_alias else 'Default'}', Benepar='{benepar_model_alias if benepar_model_alias else 'Default'}') ---", level="INFO")

    spacy_overall_success = True
    benepar_overall_success = True

    # --- SpaCy Model Setup ---
    # Determine the actual spaCy alias to process: use provided, or default if None.
    effective_spacy_alias = spacy_model_alias if spacy_model_alias is not None else DEFAULT_SPACY_ALIAS
    _log(f"  [Section]: Effective spaCy alias for setup: {effective_spacy_alias}", level="INFO")

    spacy_full_name = SPACY_MODEL_MAP.get(effective_spacy_alias.lower() if effective_spacy_alias else None) # Ensure lower for map lookup

    if not spacy_full_name:
        _log(f"Invalid spaCy model alias provided: '{effective_spacy_alias}'. Cannot proceed with spaCy setup.", level="ERROR")
        spacy_overall_success = False
    else:
        _log(f"  Processing spaCy model: {spacy_full_name} (alias: {effective_spacy_alias})", level="INFO")
        if not check_spacy_model(model_name=spacy_full_name):
            _log(f"  SpaCy model '{spacy_full_name}' not found. Attempting installation for alias '{effective_spacy_alias}'...", level="INFO")
            # install_spacy_model expects the alias, not the full name, to map to download URL etc.
            # Pass effective_spacy_alias to install_spacy_model
            if not install_spacy_model(model_name=effective_spacy_alias, log_callback=log_callback):
                _log(f"  Failed to install spaCy model '{spacy_full_name}' (alias: {effective_spacy_alias}).", level="ERROR")
                spacy_overall_success = False
            else:
                # install_spacy_model now has its own detailed success/restart_needed logging.
                # We just confirm if it returned True here.
                 _log(f"  SpaCy model installation process for '{spacy_full_name}' (alias: {effective_spacy_alias}) completed (check logs from install_spacy_model for final status).", level="INFO" if spacy_overall_success else "WARNING")
                 # Re-check after install attempt to set spacy_overall_success accurately based on loadability.
                 # The install_spacy_model returning True might mean "installed, restart needed".
                 # For overall_success, we need to know if it's USABLE NOW or will be after restart.
                 # For simplicity here, if install_spacy_model returns true, we trust its outcome.
                 # The check_spacy_model *after* might be misleading if a restart is needed.
                 # Let's rely on the return of install_spacy_model for success here.
                 # If it returned False, spacy_overall_success is already False.
                 # If it returned True (even with restart needed), we consider this part of setup "successful".
                 pass # No need to change spacy_overall_success if install_spacy_model returned True
        else:
            _log(f"  SpaCy model '{spacy_full_name}' (alias: {effective_spacy_alias}) is already present.", level="INFO")

    # --- Benepar Model Setup ---
    effective_benepar_alias = benepar_model_alias if benepar_model_alias is not None else DEFAULT_BENEPAR_ALIAS
    _log(f"  [Section]: Effective Benepar alias for setup: {effective_benepar_alias}", level="INFO")
    
    # benepar_full_name is still useful for the initial check_benepar_model call
    benepar_full_name = BENEPAR_MODEL_MAP.get(effective_benepar_alias.lower() if effective_benepar_alias else None) 

    if not benepar_full_name: # Check if alias is valid
        _log(f"Invalid Benepar model alias provided: '{effective_benepar_alias}'. Cannot proceed with Benepar setup.", level="ERROR")
        benepar_overall_success = False
    else:
        _log(f"  Processing Benepar model: {benepar_full_name} (alias: {effective_benepar_alias})", level="INFO")
        if not check_benepar_model(model_name=benepar_full_name): # Check using the full name
            _log(f"  Benepar model '{benepar_full_name}' not found. Attempting installation for alias '{effective_benepar_alias}'...", level="INFO")
            # Call install_benepar_model with the alias; it will resolve it internally.
            if not install_benepar_model(model_name=effective_benepar_alias, log_callback=log_callback):
                _log(f"  Failed to install Benepar model using alias '{effective_benepar_alias}' (resolved to '{benepar_full_name}').", level="ERROR")
                benepar_overall_success = False
            else:
                _log(f"  Benepar model (alias: '{effective_benepar_alias}', resolved to '{benepar_full_name}') installation process completed.", level="INFO")
        else:
            _log(f"  Benepar model '{benepar_full_name}' (alias: {effective_benepar_alias}) is already present.", level="INFO")

    final_success = spacy_overall_success and benepar_overall_success
    if final_success:
        _log("--- Finished: Overall Model Setup Process (Result: Success) ---", level="INFO")
    else:
        _log("--- Finished: Overall Model Setup Process (Result: Encountered Errors) ---", level="ERROR")
        
    return final_success

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
    # BasicConfig should ideally be called only once. If this module is imported,
    # the calling application should set up logging.
    if not logging.getLogger().hasHandlers(): # Check if root logger has handlers
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main_logger = logging.getLogger(__name__ + ".main") # Create a child logger for main
    main_logger.info("======== STARTING ANPE MODEL SETUP UTILITY (CLI MODE) ========")
    
    # First check if the default models are already present
    main_logger.info("--- Checking for default models (spaCy: %s, Benepar: %s) ---", DEFAULT_SPACY_ALIAS, DEFAULT_BENEPAR_ALIAS)
    if check_all_models_present(log_callback=log_callback): # Uses default arguments
        main_logger.info("--- All required default models are already present. No installation needed. ---")
        main_logger.info("======== ANPE MODEL SETUP UTILITY FINISHED (SUCCESS) ========")
        return 0
    
    main_logger.info("--- Not all default models present. Proceeding with setup for defaults. ---")
    # If not all default models are present, run the setup process with defaults
    if setup_models(log_callback=log_callback): # Uses default arguments + callback
        main_logger.info("--- Setup Complete: All required default models should now be present. --- ")
        main_logger.info("======== ANPE MODEL SETUP UTILITY FINISHED (SUCCESS) ========")
        return 0
    else:
        main_logger.error("--- Setup Failed: One or more default models could not be installed or verified. Please check logs above. --- ")
        main_logger.error("======== ANPE MODEL SETUP UTILITY FINISHED (ERRORS) ========")
        return 1

if __name__ == "__main__":
    # Removed sys.exit call, let main return the code
    # The main function now configures basic logging if no handlers are set.
    exit_code = main() # Calls main which configures basic logging
    sys.exit(exit_code) # Explicitly exit with the code from main
