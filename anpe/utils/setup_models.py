from anpe.utils.anpe_logger import get_logger  # Updated import path
import subprocess
import sys
import nltk
import os
import spacy
import benepar
import logging
import shutil
import zipfile
import importlib.metadata

logger = get_logger('setup_models')
logger.setLevel(logging.DEBUG)

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
        logger.info(f"Ensured NLTK user data directory exists: {nltk_user_dir}")

        # Ensure this directory is the first path NLTK checks
        if nltk_user_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_user_dir)
            logger.debug(f"Prepended {nltk_user_dir} to nltk.data.path")

        # Set environment variable for potential subprocess use (like benepar download)
        os.environ['NLTK_DATA'] = nltk_user_dir
        logger.debug(f"Set NLTK_DATA environment variable to: {nltk_user_dir}")

        # Verify it's the primary path
        if nltk.data.path[0] != nltk_user_dir:
             logger.warning(f"Expected {nltk_user_dir} to be the first NLTK path, but found {nltk.data.path[0]}. This might cause issues.")

        return nltk_user_dir

    except PermissionError:
        logger.error(f"Permission denied creating or accessing NLTK data directory at {nltk_user_dir}. Please check permissions.")
        return nltk_user_dir
    except Exception as e:
        logger.error(f"Unexpected error during NLTK data directory setup: {e}")
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
    benepar_model_alias: str = DEFAULT_BENEPAR_ALIAS
) -> bool:
    """Check if all required models (specified spaCy/Benepar) are present."""
    # Map aliases to actual names for checking
    spacy_model_name = SPACY_MODEL_MAP.get(spacy_model_alias.lower(), SPACY_MODEL_MAP[DEFAULT_SPACY_ALIAS])
    benepar_model_name = BENEPAR_MODEL_MAP.get(benepar_model_alias.lower(), BENEPAR_MODEL_MAP[DEFAULT_BENEPAR_ALIAS])

    logger.info(f"Checking for presence of specified models (spaCy: {spacy_model_name}, Benepar: {benepar_model_name})...")
    results = {
        "spacy": check_spacy_model(model_name=spacy_model_name),
        "benepar": check_benepar_model(model_name=benepar_model_name),
    }
    all_present = all(results.values())
    if all_present:
        logger.info(f"All specified models ({spacy_model_name}, {benepar_model_name}) are present.")
    else:
        # Log which specific model is missing
        missing = [name for name, present in results.items() if not present]
        logger.warning(f"One or more specified models are missing: {', '.join(missing)}. Status: {results}")
    return all_present

# --- Model Installation Functions ---

def _extract_zip_archive(zip_path: str, destination_dir: str, archive_name: str) -> bool:
    """Extracts a zip archive, removing existing target dir first.

    Args:
        zip_path: Path to the .zip file.
        destination_dir: Directory where the archive should be extracted.
        archive_name: The base name of the archive (used for logging and expected dir).

    Returns:
        True if extraction was successful or target directory already existed, False otherwise.
    """
    extract_path = os.path.join(destination_dir, archive_name)
    if not os.path.exists(zip_path):
        logger.warning(f"{archive_name} zip file not found at: {zip_path}")
        # If the extracted path already exists, maybe it's okay?
        exists = os.path.exists(extract_path)
        if exists:
            logger.debug(f"Zip file {zip_path} not found, but target directory {extract_path} exists. Assuming pre-extracted.")
        return exists

    logger.info(f"Extracting {zip_path}...")
    try:
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
            logger.debug(f"Removed existing directory: {extract_path}")
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)
        logger.info(f"Successfully extracted {archive_name} to {extract_path}")
        # Optionally remove the zip file after successful extraction
        os.remove(zip_path)
        logger.debug(f"Removed {zip_path}")
        return True
    except zipfile.BadZipFile:
        logger.error(f"Error: {zip_path} is not a valid zip file.")
        return False
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return False

def install_spacy_model(model_name: str = "en_core_web_md") -> bool:
    """Install the specified spaCy model."""
    if not model_name: # Basic validation
        logger.error("No spaCy model name provided to install.")
        return False

    is_transformer_model = model_name.endswith("_trf")

    # --- Check and install spacy-transformers if needed ---
    if is_transformer_model:
        logger.info(f"Model '{model_name}' is a transformer model. Checking for 'spacy-transformers' package...")
        try:
            importlib.metadata.version("spacy-transformers")
            logger.info("'spacy-transformers' is already installed.")
        except importlib.metadata.PackageNotFoundError:
            logger.warning("'spacy-transformers' package not found. Attempting installation...")
            try:
                install_result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', "spacy[transformers]"],
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info("'spacy-transformers' installed successfully.")
                logger.debug(f"pip install stdout: {install_result.stdout}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install 'spacy-transformers'. Subprocess error.")
                logger.error(f"Stderr: {e.stderr}")
                logger.error(f"You may need to install it manually: pip install 'spacy[transformers]'")
                # Continue attempting model download, but it will likely fail to load later
                # return False # Optionally return False here to indicate setup failure
            except Exception as e:
                 logger.error(f"An unexpected error occurred during 'spacy-transformers' installation: {e}")
                 # return False

    # --- Download the spaCy model ---
    try:
        logger.info(f"Attempting to download and install spaCy model: {model_name}")
        result = subprocess.run(
            [sys.executable, '-m', 'spacy', 'download', model_name],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"spaCy model '{model_name}' download command finished successfully.")
        logger.debug(f"spaCy download stdout:\n{result.stdout}")
        # Verify after download command
        if check_spacy_model(model_name):
            logger.info(f"Successfully installed and verified spaCy model: {model_name}")
            return True
        else:
            logger.error(f"spaCy download command finished for '{model_name}', but model is still not loadable. Check logs and environment.")
            return False
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download spaCy model '{model_name}'. Subprocess error.")
        logger.error(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during spaCy model installation for '{model_name}': {e}")
        return False

def install_benepar_model(model_name: str = "benepar_en3") -> bool:
    """Install the specified Benepar model, with manual extraction fallback."""
    if not model_name: # Basic validation
        logger.error("No Benepar model name provided to install.")
        return False
    models_dir = os.path.join(NLTK_DATA_DIR, "models")
    model_dir_path = os.path.join(models_dir, model_name)
    model_zip_path = os.path.join(models_dir, f"{model_name}.zip")
    subprocess_ok = False # Track subprocess success
    files_ok_after_attempt = False # Track if files seem okay after attempt

    # --- Start: Added logic to remove orphan zip --- 
    if not os.path.exists(model_dir_path) and os.path.exists(model_zip_path):
        logger.info(f"Found existing zip file {model_zip_path} but missing directory {model_dir_path}. "
                      f"Removing zip before attempting download to ensure proper extraction.")
        try:
            os.remove(model_zip_path)
            logger.debug(f"Removed existing zip: {model_zip_path}")
        except OSError as remove_err:
            # Log as error because failure here might prevent successful download
            logger.error(f"Failed to remove existing zip {model_zip_path}: {remove_err}. " 
                         f"Download might fail or use the corrupted zip.")
            # Optionally, could return False here if removing the zip is critical 
            # return False 
    # --- End: Added logic ---

    try:
        logger.info(f"Attempting to download Benepar model '{model_name}' using subprocess...")
        # Ensure NLTK_DATA is in the environment for the subprocess
        current_env = dict(os.environ)
        current_env['NLTK_DATA'] = NLTK_DATA_DIR
        logger.debug(f"Subprocess environment will use NLTK_DATA: {current_env.get('NLTK_DATA')}")

        # Escape backslashes in the path for the command string
        escaped_nltk_data_dir = NLTK_DATA_DIR.replace('\\', '\\\\') # Double escape needed for f-string then command

        result = subprocess.run(
            [sys.executable, '-c',
             f"import nltk; import benepar; nltk.data.path.insert(0, r'{escaped_nltk_data_dir}'); benepar.download('{model_name}')"],
            check=False, # Don't raise error immediately, check output/result
            capture_output=True,
            text=True,
            env=current_env, # Pass the modified environment
            timeout=1200 # Add a timeout (e.g., 20 minutes)
        )

        logger.debug(f"Benepar download subprocess stdout:\n{result.stdout}")
        if result.returncode == 0:
            logger.info(f"Benepar download subprocess finished successfully for '{model_name}'.")
            subprocess_ok = True
        else:
            logger.warning(f"Benepar download subprocess failed for '{model_name}' with return code {result.returncode}.")
            logger.warning(f"Subprocess stderr:\n{result.stderr}")
            subprocess_ok = False
            # Even if subprocess failed, continue to check files; maybe it worked partially or existed before.

        # Check if the model directory or extracted zip exists now
        if os.path.isdir(model_dir_path):
            logger.info(f"Benepar model directory found at: {model_dir_path}")
            files_ok_after_attempt = True
        else:
            logger.info(f"Benepar model directory not found at {model_dir_path}. Checking for zip file...")
            if os.path.isfile(model_zip_path):
                logger.info(f"Found zip file: {model_zip_path}. Attempting manual extraction.")
                if _extract_zip_archive(model_zip_path, models_dir, model_name):
                    logger.info(f"Manual extraction of {model_name} successful.")
                    files_ok_after_attempt = True # Extracted successfully
                else:
                    logger.error(f"Manual extraction of {model_zip_path} failed.")
                    files_ok_after_attempt = False # Extraction failed
            else:
                logger.warning(f"Neither Benepar model directory nor zip file found after download attempt.")
                files_ok_after_attempt = False

        # Final verification only makes sense if files seemed okay after the attempt
        if files_ok_after_attempt:
            logger.info(f"Verifying Benepar model '{model_name}' using check function...")
            if check_benepar_model(model_name):
                logger.info(f"Benepar model '{model_name}' is present and verified.")
                # Return True only if the final check passes
                return True
            else:
                logger.error(f"Benepar model '{model_name}' verification failed even though files seemed present.")
                return False
        else:
             # If files weren't okay after attempt, the installation failed.
             logger.error(f"Installation failed for Benepar model '{model_name}' - files not found or extraction failed.")
             return False

    except subprocess.TimeoutExpired:
        logger.error(f"Benepar download subprocess timed out for '{model_name}'.")
        return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during Benepar model '{model_name}' installation: {e}")
        # Log stderr if it was a CalledProcessError originally wrapped
        if hasattr(e, 'stderr'):
             logger.error(f"Subprocess stderr: {e.stderr}")
        return False

def setup_models(
    spacy_model_alias: str = DEFAULT_SPACY_ALIAS,
    benepar_model_alias: str = DEFAULT_BENEPAR_ALIAS
) -> bool:
    """
    Checks for required models (specified spaCy/Benepar) and attempts
    to install any that are missing. Uses user-friendly aliases for models.

    Args:
        spacy_model_alias (str): Alias for spaCy model ('sm', 'md', 'lg', 'trf').
                                 Defaults to 'md'.
        benepar_model_alias (str): Alias for Benepar model ('default', 'large').
                                   Defaults to 'default'.

    Returns:
        bool: True if all required models (based on selected aliases) are
              present after the check/install process, False otherwise.
    """
    logger.info(f"Starting model setup process for spaCy='{spacy_model_alias}', benepar='{benepar_model_alias}'...")
    final_status = True  # Tracks if all models are OK by the end

    # --- Map Aliases ---
    actual_spacy_model = SPACY_MODEL_MAP.get(spacy_model_alias.lower())
    if not actual_spacy_model:
        logger.warning(f"Invalid spaCy model alias '{spacy_model_alias}'. Falling back to default '{DEFAULT_SPACY_ALIAS}'.")
        actual_spacy_model = SPACY_MODEL_MAP[DEFAULT_SPACY_ALIAS]

    actual_benepar_model = BENEPAR_MODEL_MAP.get(benepar_model_alias.lower())
    if not actual_benepar_model:
        logger.warning(f"Invalid Benepar model alias '{benepar_model_alias}'. Falling back to default '{DEFAULT_BENEPAR_ALIAS}'.")
        actual_benepar_model = BENEPAR_MODEL_MAP[DEFAULT_BENEPAR_ALIAS]

    # --- 1. spaCy Model ---
    logger.info(f"Checking spaCy model ({actual_spacy_model} from alias '{spacy_model_alias}')...")
    if not check_spacy_model(model_name=actual_spacy_model):
        logger.info(f"spaCy model '{actual_spacy_model}' not found. Attempting download...")
        if not install_spacy_model(model_name=actual_spacy_model):
            logger.error(f"Failed to download/install spaCy model '{actual_spacy_model}'.")
            final_status = False
        # Verification is now part of install_spacy_model
        # elif not check_spacy_model(model_name=actual_spacy_model):
        #     logger.error(f"spaCy model '{actual_spacy_model}' installed but still not loadable.")
        #     final_status = False
        else:
            logger.info(f"Successfully downloaded and verified spaCy model '{actual_spacy_model}'.")
    else:
        logger.info(f"spaCy model '{actual_spacy_model}' is already present.")

    # --- 2. Benepar Model ---
    logger.info(f"Checking Benepar model ({actual_benepar_model} from alias '{benepar_model_alias}')...")
    if not check_benepar_model(model_name=actual_benepar_model):
        logger.info(f"Benepar model '{actual_benepar_model}' not found. Attempting download...")
        if not install_benepar_model(model_name=actual_benepar_model):
            logger.error(f"Failed to download/install Benepar model '{actual_benepar_model}'.")
            final_status = False
        # Verification is now part of install_benepar_model
        # elif not check_benepar_model(model_name=actual_benepar_model):
        #     logger.error(f"Benepar model '{actual_benepar_model}' downloaded but not found.")
        #     final_status = False
        else:
            logger.info(f"Successfully downloaded and verified Benepar model '{actual_benepar_model}'.")
    else:
        logger.info(f"Benepar model '{actual_benepar_model}' is already present.")

    # --- Final Summary ---
    if final_status:
        logger.info("Model setup process completed successfully.")
    else:
        logger.error("Model setup process failed for one or more models. Please review logs.")

    return final_status

def main():
    # This main is primarily for direct CLI invocation: `python -m anpe.utils.setup_models`
    # It ensures required models are present, installing if needed.
    # It will use the *default* models ('md', 'default')
    print("--- Running ANPE Model Setup Utility (using default models) ---")
    
    # First check if the default models are already present
    if check_all_models_present(): # Uses default arguments
        print("--- All required default models are already present. No installation needed. ---")
        sys.exit(0)
    
    # If not all default models are present, run the setup process with defaults
    if setup_models(): # Uses default arguments
        print("--- Setup Complete: All required default models are now present. ---")
        sys.exit(0)
    else:
        print("--- Setup Failed: One or more default models could not be installed or verified. Please check logs above. ---")
        sys.exit(1)

if __name__ == "__main__":
    main()
