#!/usr/bin/env python3
"""Utilities for finding installed ANPE models and selecting the best one to use."""

import os
import spacy
import nltk
from typing import List, Optional
import logging

# Assuming setup_models is adjacent or correctly in path for map import
from .setup_models import SPACY_MODEL_MAP, BENEPAR_MODEL_MAP, NLTK_DATA_DIR

logger = logging.getLogger(__name__) # Use module-specific logger

# Define preference order (most preferred first)
SPACY_PREFERENCE = ["en_core_web_trf", "en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
BENEPAR_PREFERENCE = ["benepar_en3_large", "benepar_en3"]

def find_installed_spacy_models() -> List[str]:
    """Find installed and loadable spaCy models relevant to ANPE.

    This function iterates through known ANPE-relevant spaCy models and
    attempts to load each one using `spacy.load()`. This is a more robust
    check than relying on `spacy.util.get_installed_models` or path checks.

    Returns:
        List[str]: A list of loadable spaCy model names relevant to ANPE.
    """
    verified_models = []
    known_anpe_spacy_models = set(SPACY_MODEL_MAP.values())
    logger.debug(f"Checking for known ANPE spaCy models: {known_anpe_spacy_models}")

    for model_name in known_anpe_spacy_models:
        try:
            # Attempt to load the model to verify it's actually installed and usable
            spacy.load(model_name)
            verified_models.append(model_name)
            logger.debug(f"Successfully loaded and verified spaCy model: '{model_name}'")
        except (OSError, ImportError):
            # OSError is common if the model data isn't found
            # ImportError can happen for various reasons related to model loading
            logger.debug(f"SpaCy model '{model_name}' not found or could not be loaded. Skipping.")
        except Exception as e: # Catch other unexpected errors during loading
            logger.warning(f"An unexpected error occurred while trying to load spaCy model '{model_name}': {e}. Skipping.")

    if not verified_models:
        logger.warning("No loadable ANPE-relevant spaCy models found.")
    else:
        logger.info(f"Found loadable ANPE-relevant spaCy models: {verified_models}")

    return verified_models

def find_installed_benepar_models() -> List[str]:
    """Find installed Benepar models in NLTK data paths.

    Returns:
        List[str]: A list of installed Benepar model names.
    """
    installed_models = []
    known_benepar_models = set(BENEPAR_MODEL_MAP.values())
    checked_paths = set() # Avoid redundant checks if paths overlap

    # Use NLTK data path directly, as Benepar models live there
    nltk_paths_to_check = nltk.data.path
    # Ensure the primary user path is checked first if not already there
    if NLTK_DATA_DIR not in nltk_paths_to_check:
        nltk_paths_to_check.insert(0, NLTK_DATA_DIR)

    logger.debug(f"Checking NLTK paths for Benepar models: {nltk_paths_to_check}")
    
    for nltk_path in nltk_paths_to_check:
        if nltk_path in checked_paths:
             continue
        checked_paths.add(nltk_path)
        
        models_dir = os.path.join(nltk_path, "models")
        if os.path.isdir(models_dir):
            for model_name in known_benepar_models:
                model_path = os.path.join(models_dir, model_name)
                # Check if the directory exists (primary indicator)
                if os.path.isdir(model_path):
                    if model_name not in installed_models:
                        logger.debug(f"Found installed Benepar model directory: {model_path}")
                        installed_models.append(model_name)
                else:
                     # Check for zip file as fallback (might exist if download incomplete)
                     zip_path = model_path + ".zip"
                     if os.path.isfile(zip_path):
                          if model_name not in installed_models:
                              # We found the zip, but maybe not the dir. Should we count it?
                              # Let's count it, but prefer models where the dir exists. 
                              # The selection logic will handle preference.
                              logger.debug(f"Found Benepar model zip file (but not dir): {zip_path}")
                              installed_models.append(model_name)
                              
    if not installed_models:
        logger.warning("No installed Benepar models found in checked NLTK paths.")
    else:
        logger.info(f"Found installed Benepar models: {installed_models}")
    return installed_models

def select_best_spacy_model(models: List[str]) -> Optional[str]:
    """Selects the best spaCy model from a list based on preference order.

    Args:
        models (List[str]): List of installed spaCy model names.

    Returns:
        Optional[str]: The preferred model name, or None if list is empty.
    """
    if not models:
        return None

    # New logic: Prioritize default model if installed
    default_spacy_model = "en_core_web_md" # Assuming 'md' is the default alias
    if default_spacy_model in models:
        logger.debug(f"Default spaCy model '{default_spacy_model}' is installed. Selecting it.")
        return default_spacy_model

    # Fallback: Use the preference list if default is not installed
    logger.debug(f"Default spaCy model '{default_spacy_model}' not found. Checking preference list: {SPACY_PREFERENCE}")
    for preferred_model in SPACY_PREFERENCE:
        if preferred_model in models:
            logger.debug(f"Selected preferred spaCy model from list: {preferred_model}")
            return preferred_model
            
    # Fallback: return the first one found if no preferred model is present
    # Ensure models list is not empty before accessing index 0
    if models:
        fallback_model = models[0]
        logger.debug(f"No specifically preferred spaCy model found, falling back to: {fallback_model}")
        return fallback_model
    else:
        logger.warning("No spaCy models available to select from.")
        return None

def select_best_benepar_model(models: List[str]) -> Optional[str]:
    """Selects the best Benepar model from a list based on preference order.

    Args:
        models (List[str]): List of installed Benepar model names.

    Returns:
        Optional[str]: The preferred model name, or None if list is empty.
    """
    if not models:
        return None

    # New logic: Prioritize default model if installed
    default_benepar_model = "benepar_en3" # Assuming 'default' alias maps to this
    if default_benepar_model in models:
        logger.debug(f"Default Benepar model '{default_benepar_model}' is installed. Selecting it.")
        return default_benepar_model

    # Fallback: Use the preference list if default is not installed
    logger.debug(f"Default Benepar model '{default_benepar_model}' not found. Checking preference list: {BENEPAR_PREFERENCE}")
    for preferred_model in BENEPAR_PREFERENCE:
        if preferred_model in models:
            logger.debug(f"Selected preferred Benepar model from list: {preferred_model}")
            return preferred_model
            
    # Fallback
    fallback_model = models[0]
    logger.debug(f"No specifically preferred Benepar model found, falling back to: {fallback_model}")
    return fallback_model

# Example usage (for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Testing Model Finder --- ")
    
    installed_spacy = find_installed_spacy_models()
    print(f"Installed spaCy models found: {installed_spacy}")
    best_spacy = select_best_spacy_model(installed_spacy)
    print(f"Best spaCy model selected: {best_spacy}")

    print("-" * 20)
    
    installed_benepar = find_installed_benepar_models()
    print(f"Installed Benepar models found: {installed_benepar}")
    best_benepar = select_best_benepar_model(installed_benepar)
    print(f"Best Benepar model selected: {best_benepar}")
    
    logger.info("--- Test Complete --- ") 