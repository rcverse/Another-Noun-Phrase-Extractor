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

logger = get_logger('setup_models')
logger.setLevel(logging.DEBUG)

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
        # Fallback or raise? For now, log error and return potentially non-functional path
        # Returning allows checks to potentially still run if models exist elsewhere.
        return nltk_user_dir # Or raise an exception?
    except Exception as e:
        logger.error(f"Unexpected error during NLTK data directory setup: {e}")
        # Try returning a default path, though it might not work
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

def check_nltk_models(models: list[str] = ['punkt', 'punkt_tab']) -> bool:
    """Check if the specified NLTK models/data are available."""
    all_present = True
    
    for model in models:
        try:
            # Use NLTK's built-in find function which handles all path resolution
            nltk.data.find(f'tokenizers/{model}')
            logger.debug(f"NLTK resource 'tokenizers/{model}' found")
        except LookupError:
            logger.debug(f"NLTK resource 'tokenizers/{model}' not found")
            all_present = False
        except Exception as e:
            logger.error(f"Error checking NLTK resource '{model}': {e}")
            all_present = False
    
    return all_present

def check_all_models_present() -> bool:
    """Check if all required models (spaCy, Benepar, NLTK) are present."""
    logger.info("Checking for presence of all required models...")
    results = {
        "spacy": check_spacy_model(),
        "benepar": check_benepar_model(),
        "nltk": check_nltk_models()
    }
    all_present = all(results.values())
    if all_present:
        logger.info("All required models are present.")
    else:
        logger.warning(f"One or more models are missing: {results}")
    return all_present

# --- Model Installation Functions ---

def _extract_zip_archive(zip_path: str, destination_dir: str, archive_name: str) -> bool:
    """Extracts a zip archive, removing existing target dir first.

    Args:
        zip_path: Path to the .zip file.
        destination_dir: Directory where the archive should be extracted.
        archive_name: The base name of the archive (used for logging and expected dir).

    Returns:
        True if extraction was successful, False otherwise.
    """
    extract_path = os.path.join(destination_dir, archive_name)
    if not os.path.exists(zip_path):
        logger.warning(f"{archive_name} zip file not found at: {zip_path}")
        # If the extracted path already exists, maybe it's okay?
        # Return True only if the final directory exists, otherwise False.
        return os.path.exists(extract_path)

    logger.info(f"Extracting {zip_path}...")
    try:
        if os.path.exists(extract_path):
            shutil.rmtree(extract_path)
            logger.debug(f"Removed existing directory: {extract_path}")
            
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_dir)
        logger.info(f"Successfully extracted {archive_name} to {extract_path}")
        # Optionally remove the zip file after successful extraction
        # os.remove(zip_path)
        # logger.debug(f"Removed {zip_path}")
        return True
    except zipfile.BadZipFile:
        logger.error(f"Error: {zip_path} is not a valid zip file.")
        return False
    except Exception as e:
        logger.error(f"Error extracting {zip_path}: {e}")
        return False

def install_spacy_model(model_name: str = "en_core_web_md") -> bool:
    """Install the specified spaCy model."""
    try:
        logger.info(f"Downloading spaCy model: {model_name}")
        result = subprocess.run(
            [sys.executable, '-m', 'spacy', 'download', model_name],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"spaCy model '{model_name}' installed successfully")
        logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install spaCy model: {e.stderr}")
        return False

def install_benepar_model(model_name: str = "benepar_en3") -> bool:
    """Install the specified Benepar model."""
    try:
        logger.info(f"Downloading Benepar model: {model_name}")
        
        # Rely on NLTK_DATA environment variable set by setup_nltk_data_dir
        result = subprocess.run(
            [sys.executable, '-c', 
             f"import benepar; benepar.download('{model_name}')"], # Simplified command
            check=True,
            capture_output=True,
            text=True,
        )
        
        # Extract the downloaded zip archive
        model_zip_path = os.path.join(NLTK_DATA_DIR, "models", f"{model_name}.zip")
        models_dir = os.path.join(NLTK_DATA_DIR, "models")
        if not _extract_zip_archive(model_zip_path, models_dir, model_name):
             # If extraction fails or zip wasn't found but dir doesn't exist, fail.
             if not os.path.exists(os.path.join(models_dir, model_name)):
                  logger.error(f"Benepar model extraction failed and directory not found.")
                  return False
             else:
                  logger.warning(f"Benepar extraction reported an issue, but target directory exists. Continuing verification.")

        # Verify installation using the standard check function
        if check_benepar_model(model_name):
            logger.info(f"Benepar model '{model_name}' installed and verified successfully.")
            return True
        else:
            logger.error(f"Benepar model '{model_name}' download seemed successful, but check_benepar_model failed. Check NLTK paths and permissions.")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Benepar model: {e.stderr}")
        # Also log stdout for completeness in case of error
        if e.stdout:
            logger.error(f"Subprocess stdout during Benepar install error: {e.stdout}")
        return False

def install_nltk_models() -> bool:
    """Install required NLTK models."""
    try:
        logger.info("Downloading NLTK Punkt tokenizer")
        tokenizers_dir = os.path.join(NLTK_DATA_DIR, "tokenizers")
        
        # Download punkt (rely on NLTK finding the correct path)
        nltk.download('punkt', quiet=True)
        
        # Extract the downloaded punkt archive
        punkt_zip_path = os.path.join(NLTK_DATA_DIR, "tokenizers", "punkt.zip")
        if not _extract_zip_archive(punkt_zip_path, tokenizers_dir, "punkt"):
             # If extraction fails or zip wasn't found but dir doesn't exist, fail.
             if not os.path.exists(os.path.join(tokenizers_dir, "punkt")):
                  logger.error(f"Punkt extraction failed and directory not found. Cannot create punkt_tab.")
                  return False
             else:
                  logger.warning(f"Punkt extraction reported an issue, but target directory exists. Attempting punkt_tab setup.")

        # Setup punkt_tab (for Benepar compatibility)
        logger.info("Setting up punkt_tab for Benepar compatibility")
        punkt_dir = os.path.join(NLTK_DATA_DIR, "tokenizers", "punkt")
        punkt_tab_dir = os.path.join(NLTK_DATA_DIR, "tokenizers", "punkt_tab")
        os.makedirs(punkt_tab_dir, exist_ok=True)
        logger.debug(f"Ensured punkt_tab directory exists at: {punkt_tab_dir}")
        
        # Copy files from punkt to punkt_tab if punkt exists
        if os.path.exists(punkt_dir):
            for lang in ['english']:
                lang_src_dir = os.path.join(punkt_dir, lang)
                lang_dst_dir = os.path.join(punkt_tab_dir, lang)
                # Ensure the specific language source directory exists before copying
                if os.path.exists(lang_src_dir):
                    if os.path.exists(lang_dst_dir):
                        shutil.rmtree(lang_dst_dir) # Remove existing dest to avoid merge issues
                    shutil.copytree(lang_src_dir, lang_dst_dir)
                    logger.debug(f"Copied {lang} files from {lang_src_dir} to: {lang_dst_dir}")
                else:
                    logger.warning(f"Source directory for language '{lang}' not found in punkt: {lang_src_dir}")
        else:
            logger.error(f"Punkt source directory not found after download: {punkt_dir}. Cannot create punkt_tab.")
            return False # Cannot proceed without punkt source
        
        # Verify that nltk can find the downloaded/created resources
        punkt_found = False
        punkt_tab_found = False
        
        try:
            nltk.data.find('tokenizers/punkt')
            logger.debug("NLTK verification successful for: tokenizers/punkt")
            punkt_found = True
        except LookupError:
            logger.error("NLTK verification failed for: tokenizers/punkt")
        except Exception as e:
            logger.error(f"Error during NLTK verification for tokenizers/punkt: {e}")
            
        try:
            nltk.data.find('tokenizers/punkt_tab')
            logger.debug("NLTK verification successful for: tokenizers/punkt_tab")
            punkt_tab_found = True
        except LookupError:
            logger.error("NLTK verification failed for: tokenizers/punkt_tab")
        except Exception as e:
            logger.error(f"Error during NLTK verification for tokenizers/punkt_tab: {e}")

        if punkt_found and punkt_tab_found:
            logger.info("NLTK models installed and verified successfully")
            return True
        else:
            logger.error("NLTK models installation failed verification.")
            return False
        
    except Exception as e:
        logger.error(f"Failed to install NLTK models: {str(e)}")
        return False

def setup_models() -> bool:
    """
    Checks for required models (spaCy, Benepar, NLTK) and attempts to
    install any that are missing.

    Returns:
        bool: True if all required models are present after the check/install process,
              False otherwise.
    """
    logger.info("Starting model setup process...")
    final_status = True  # Tracks if all models are OK by the end

    # --- 1. spaCy Model ---
    logger.info("Checking spaCy model (en_core_web_md)...")
    if not check_spacy_model():
        logger.info("spaCy model not found. Attempting download...")
        if not install_spacy_model():
            logger.error("Failed to download spaCy model.")
            final_status = False
        elif not check_spacy_model():  # Verify installation
            logger.error("spaCy model installed but still not loadable. Check environment/permissions.")
            final_status = False
        else:
            logger.info("Successfully downloaded and verified spaCy model.")
    else:
        logger.info("spaCy model is already present.")

    # --- 2. Benepar Model ---
    logger.info("Checking Benepar model (benepar_en3)...")
    if not check_benepar_model():
        logger.info("Benepar model not found. Attempting download...")
        if not install_benepar_model():
            logger.error("Failed to download Benepar model.")
            final_status = False
        elif not check_benepar_model():  # Verify installation
            logger.error("Benepar model downloaded but files not found where expected. Check environment/permissions.")
            final_status = False
        else:
            logger.info("Successfully downloaded and verified Benepar model.")
    else:
        logger.info("Benepar model is already present.")

    # --- 3. NLTK Models ---
    logger.info("Checking NLTK models (punkt, punkt_tab)...")
    if not check_nltk_models():
        logger.info("One or more NLTK models not found. Attempting download/setup...")
        if not install_nltk_models():
            logger.error("Failed to download/setup NLTK models.")
            final_status = False
        elif not check_nltk_models():  # Verify installation
            logger.error("NLTK setup completed but resources still not found. Check NLTK data path/permissions.")
            final_status = False
        else:
            logger.info("Successfully downloaded and verified NLTK models.")
    else:
        logger.info("Required NLTK models are already present.")

    # --- Final Summary ---
    if final_status:
        logger.info("Model setup process completed successfully.")
    else:
        logger.error("Model setup process failed for one or more models. Please review logs.")

    return final_status

def main():
    # This main is primarily for direct CLI invocation: `python -m anpe.utils.setup_models`
    # It ensures required models are present, installing if needed.
    print("--- Running ANPE Model Setup Utility ---")
    
    # First check if all models are already present
    if check_all_models_present():
        print("--- All required models are already present. No installation needed. ---")
        sys.exit(0)
    
    # If not all models are present, run the setup process
    if setup_models():
        print("--- Setup Complete: All required models are now present. ---")
        sys.exit(0)
    else:
        print("--- Setup Failed: One or more models could not be installed or verified. Please check logs above. ---")
        sys.exit(1)

if __name__ == "__main__":
    main()
