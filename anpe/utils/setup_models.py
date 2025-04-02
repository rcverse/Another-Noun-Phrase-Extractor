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

def install_nltk_models() -> bool:
    """Install required NLTK models."""
    try:
        logger.info("Downloading NLTK Punkt tokenizer")
        nltk.download('punkt', quiet=True)
        
        # This is the missing part - we need to download punkt_tab or simulate it
        logger.info("Setting up punkt_tab for Benepar compatibility")
        try:
            # First try to see if punkt_tab exists as a downloadable resource
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            # If not available as direct download, we can create a symlink or copy
            # This is a workaround since punkt_tab isn't in standard NLTK
            import shutil
            import os
            from nltk.data import path as nltk_path
            
            # Find punkt data directory
            punkt_dir = None
            for path in nltk_path:
                potential_path = os.path.join(path, 'tokenizers', 'punkt')
                if os.path.exists(potential_path):
                    punkt_dir = potential_path
                    break
            
            if punkt_dir:
                # Create punkt_tab directory if needed
                punkt_tab_dir = os.path.join(os.path.dirname(punkt_dir), 'punkt_tab')
                os.makedirs(punkt_tab_dir, exist_ok=True)
                
                # Copy or link language files
                for lang in ['english']:  # Add more languages if needed
                    lang_src = os.path.join(punkt_dir, lang)
                    lang_dst = os.path.join(punkt_tab_dir, lang)
                    if os.path.exists(lang_src) and not os.path.exists(lang_dst):
                        os.makedirs(os.path.dirname(lang_dst), exist_ok=True)
                        shutil.copytree(lang_src, lang_dst)
            
        logger.info("NLTK models installed successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to install NLTK models: {str(e)}")
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
        install_nltk_models()
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