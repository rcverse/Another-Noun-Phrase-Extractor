from anpe.utils.anpe_logger import get_logger  # Updated import path
import subprocess
import sys
import nltk
import os
import spacy
import benepar
import site
import logging
import shutil

logger = get_logger('setup_models')
logger.setLevel(logging.DEBUG)

# Set up NLTK data path
def setup_nltk_data_dir():
    """Set up and verify NLTK data directory."""
    # Use user's home directory as default
    home = os.path.expanduser("~")
    nltk_data = os.path.join(home, "nltk_data")
    os.makedirs(nltk_data, exist_ok=True)
    
    # Ensure this is the first path NLTK checks
    if nltk_data not in nltk.data.path:
        nltk.data.path.insert(0, nltk_data)
    
    logger.debug(f"Primary NLTK data directory: {nltk_data}")
    return nltk_data

NLTK_DATA = setup_nltk_data_dir()

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

def find_model_locations() -> dict[str, list[str]]:
    """Find all possible model locations."""
    locations = {
        "nltk": [],
        "spacy": [],
        "benepar": []
    }
    
    # Get NLTK's data path
    try:
        import nltk.data
        locations["nltk"] = nltk.data.path
        logger.debug("NLTK search paths:")
        for path in locations["nltk"]:
            logger.debug(f"  - {path}")
    except Exception as e:
        logger.error(f"Error getting NLTK data paths: {e}")
    
    # spaCy paths
    try:
        spacy_path = spacy.util.get_data_path()
        locations["spacy"].append(spacy_path)
        logger.debug(f"spaCy data path: {spacy_path}")
    except Exception as e:
        logger.error(f"Error getting spaCy data path: {e}")
    
    return locations

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

def install_package(package_name: str) -> bool:
    """Install a Python package."""
    try:
        logger.info(f"Installing {package_name}")
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package_name],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully installed {package_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package_name}: {e.stderr}")
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
        logger.debug(f"Using NLTK data directory: {NLTK_DATA}")
        
        # Set NLTK_DATA environment variable to ensure Benepar uses our directory
        os.environ['NLTK_DATA'] = NLTK_DATA
        
        result = subprocess.run(
            [sys.executable, '-c', f"import benepar; benepar.download('{model_name}')"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, 'NLTK_DATA': NLTK_DATA}  # Ensure subprocess sees NLTK_DATA
        )
        
        # Verify installation
        model_path = os.path.join(NLTK_DATA, "models", model_name)
        if os.path.exists(model_path):
            logger.debug(f"Verified Benepar model installation at: {model_path}")
            logger.info(f"Benepar model '{model_name}' installed successfully")
            return True
        else:
            logger.error(f"Benepar model not found at expected location: {model_path}")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Benepar model: {e.stderr}")
        return False

def install_nltk_models() -> bool:
    """Install required NLTK models."""
    try:
        logger.info("Downloading NLTK Punkt tokenizer")
        logger.debug(f"Using NLTK data directory: {NLTK_DATA}")
        
        # Download punkt
        nltk.download('punkt', download_dir=NLTK_DATA, quiet=True)
        punkt_dir = os.path.join(NLTK_DATA, "tokenizers", "punkt")
        if os.path.exists(punkt_dir):
            logger.debug(f"Verified punkt installation at: {punkt_dir}")
        
        # Set up punkt_tab
        logger.info("Setting up punkt_tab for Benepar compatibility")
        punkt_tab_dir = os.path.join(NLTK_DATA, "tokenizers", "punkt_tab")
        os.makedirs(punkt_tab_dir, exist_ok=True)
        logger.debug(f"Created punkt_tab directory at: {punkt_tab_dir}")
        
        # Copy files
        for lang in ['english']:
            lang_src = os.path.join(punkt_dir, lang)
            lang_dst = os.path.join(punkt_tab_dir, lang)
            if os.path.exists(lang_src) and not os.path.exists(lang_dst):
                os.makedirs(os.path.dirname(lang_dst), exist_ok=True)
                shutil.copytree(lang_src, lang_dst)
                logger.debug(f"Copied {lang} files to: {lang_dst}")
        
        logger.info("NLTK models installed successfully")
        return True
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