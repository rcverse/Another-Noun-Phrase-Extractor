import subprocess
import sys
import nltk
from anpe.utils.logging import get_logger

logger = get_logger('setup_models')

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

def install_spacy_model(model_name: str = "en_core_web_sm") -> bool:
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
        result = subprocess.run(
            [sys.executable, '-c', f"import benepar; benepar.download('{model_name}')"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Benepar model '{model_name}' installed successfully")
        logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install Benepar model: {e.stderr}")
        return False

def install_nltk_punkt() -> bool:
    """Install the NLTK Punkt tokenizer."""
    try:
        logger.info("Downloading NLTK Punkt tokenizer")
        nltk.download('punkt', quiet=True)
        logger.info("NLTK Punkt tokenizer installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install NLTK Punkt tokenizer: {str(e)}")
        return False

def setup_models() -> bool:
    """Download all required models."""
    # First ensure required packages are installed
    if not install_package('spacy'):
        return False
    if not install_package('benepar'):
        return False
    if not install_package('nltk'):
        return False
    
    # Then install the models
    results = [
        install_spacy_model(),
        install_benepar_model(),
        install_nltk_punkt()
    ]
    
    return all(results)

def main():
    if setup_models():
        print("All models installed successfully")
        sys.exit(0)
    else:
        print("Failed to install some models")
        sys.exit(1)

if __name__ == "__main__":
    main() 