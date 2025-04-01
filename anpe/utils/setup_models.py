import subprocess
import sys
import nltk
from pathlib import Path
from typing import Optional
from anpe.utils.logging import get_logger

logger = get_logger('setup_models')

def install_spacy_model(model_name: str = "en_core_web_sm") -> bool:
    """Install the specified spaCy model."""
    try:
        logger.info(f"Downloading spaCy model: {model_name}")
        subprocess.check_call([sys.executable, '-m', 'spacy', 'download', model_name])
        logger.info(f"spaCy model '{model_name}' installed successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to install spaCy model: {e}")
        return False

def install_benepar_model(model_name: str = "benepar_en3") -> bool:
    """Install the specified Benepar model."""
    try:
        logger.info(f"Downloading Benepar model: {model_name}")
        subprocess.check_call([sys.executable, '-m', 'benepar', 'download', model_name])
        logger.info(f"Benepar model '{model_name}' installed successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to install Benepar model: {e}")
        return False

def install_nltk_punkt() -> bool:
    """Install the NLTK Punkt tokenizer."""
    try:
        logger.info("Downloading NLTK Punkt tokenizer")
        nltk.download('punkt')
        logger.info("NLTK Punkt tokenizer installed successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to install NLTK Punkt tokenizer: {e}")
        return False

def setup_models() -> bool:
    """Download all required models."""
    success = True
    success &= install_spacy_model()
    success &= install_benepar_model()
    success &= install_nltk_punkt()
    return success 