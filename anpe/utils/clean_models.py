#!/usr/bin/env python3
"""Utility script to remove all ANPE-related models and caches."""

import os
import shutil
import subprocess
import sys
import nltk
from pathlib import Path
import argparse
from typing import Dict, List, Optional
import logging
import site
import spacy

def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    logger = logging.getLogger('clean_models')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter('%(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

def find_model_locations() -> Dict[str, List[str]]:
    """Find all possible model locations."""
    locations = {
        "nltk": [],
        "spacy": [],
        "benepar": []
    }
    
    # NLTK data paths
    locations["nltk"].extend(nltk.data.path)
    
    # Add user's home directory NLTK path
    home = os.path.expanduser("~")
    locations["nltk"].append(os.path.join(home, "nltk_data"))
    
    # Add site-packages NLTK paths
    for site_pkg in site.getsitepackages():
        locations["nltk"].append(os.path.join(site_pkg, "nltk_data"))
    
    # spaCy paths
    try:
        locations["spacy"].append(spacy.util.get_data_path())
    except:
        pass
    
    # Benepar paths (usually inside NLTK data)
    for nltk_path in locations["nltk"]:
        locations["benepar"].append(os.path.join(nltk_path, "models"))
    
    return locations

def remove_spacy_model(model_name: str = "en_core_web_md", logger: logging.Logger = None) -> bool:
    """Remove spaCy model."""
    try:
        logger.info("Removing spaCy model...")
        # First try pip uninstall
        result = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", model_name],
            check=False,
            capture_output=True,
            text=True
        )
        
        # Also try to remove from spacy data directory
        try:
            data_path = spacy.util.get_data_path()
            model_path = os.path.join(data_path, model_name)
            if os.path.exists(model_path):
                if os.path.isdir(model_path):
                    shutil.rmtree(model_path)
                else:
                    os.remove(model_path)
                logger.info(f"Removed spaCy model from data directory: {model_path}")
        except Exception as e:
            logger.debug(f"Could not remove from spaCy data directory: {e}")
        
        if result.returncode == 0:
            logger.info("✓ Successfully removed spaCy model")
            return True
        else:
            logger.warning(f"! spaCy model removal returned: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"! Error removing spaCy model: {e}")
        return False

def remove_benepar_model(model_name: str = "benepar_en3", logger: logging.Logger = None) -> bool:
    """Remove Benepar model from all possible locations."""
    try:
        logger.info("Removing Benepar model...")
        locations = find_model_locations()
        found = False
        
        for base_path in locations["benepar"]:
            model_path = os.path.join(base_path, model_name)
            if os.path.exists(model_path):
                found = True
                try:
                    if os.path.isdir(model_path):
                        shutil.rmtree(model_path)
                    else:
                        os.remove(model_path)
                    logger.info(f"✓ Removed Benepar model from: {model_path}")
                except Exception as e:
                    logger.warning(f"! Could not remove from {model_path}: {e}")
                    found = False
        
        if found:
            logger.info("✓ Successfully removed Benepar model")
            return True
        else:
            logger.info("! Benepar model not found in any known location")
            return True  # Return True as the model is effectively "removed"
            
    except Exception as e:
        logger.error(f"! Error removing Benepar model: {e}")
        return False

def remove_nltk_data(resources: List[str] = None, logger: logging.Logger = None) -> bool:
    """Remove NLTK resources from all possible locations."""
    if resources is None:
        resources = ['punkt', 'punkt_tab']
    
    success = True
    try:
        logger.info("Removing NLTK resources...")
        locations = find_model_locations()
        
        for nltk_path in locations["nltk"]:
            for resource in resources:
                resource_path = os.path.join(nltk_path, "tokenizers", resource)
                if os.path.exists(resource_path):
                    try:
                        if os.path.isdir(resource_path):
                            shutil.rmtree(resource_path)
                        else:
                            os.remove(resource_path)
                        logger.info(f"✓ Removed NLTK resource from {resource_path}")
                    except Exception as e:
                        logger.warning(f"! Could not remove {resource} from {resource_path}: {e}")
                        success = False
        
        if success:
            logger.info("✓ Successfully removed all NLTK resources")
        else:
            logger.warning("! Some NLTK resources could not be removed")
            
    except Exception as e:
        logger.error(f"! Error removing NLTK resources: {e}")
        success = False
    
    return success

def find_resources() -> Dict[str, List[str]]:
    """Find all existing resources in known locations."""
    locations = find_model_locations()
    resources = {
        "spacy": [],
        "benepar": [],
        "nltk": []
    }
    
    # Find spaCy models
    try:
        data_path = spacy.util.get_data_path()
        if os.path.exists(data_path):
            for item in os.listdir(data_path):
                if item.startswith("en_core_web"):
                    resources["spacy"].append(os.path.join(data_path, item))
    except Exception as e:
        pass
    
    # Find Benepar models
    for base_path in locations["benepar"]:
        if os.path.exists(base_path):
            for item in os.listdir(base_path):
                if item.startswith("benepar"):
                    resources["benepar"].append(os.path.join(base_path, item))
    
    # Find NLTK resources
    for nltk_path in locations["nltk"]:
        tokenizers_path = os.path.join(nltk_path, "tokenizers")
        if os.path.exists(tokenizers_path):
            for item in os.listdir(tokenizers_path):
                if item in ['punkt', 'punkt_tab']:
                    resources["nltk"].append(os.path.join(tokenizers_path, item))
    
    return resources

def clean_all(verbose: bool = False, logger: logging.Logger = None) -> Dict[str, bool]:
    """Remove all ANPE-related models and return status of each operation."""
    if logger is None:
        logger = setup_logging(verbose)
    
    logger.info("Starting ANPE model cleanup...")
    logger.info("-" * 50)
    
    # First show all found resources
    logger.info("Scanning for existing resources...")
    resources = find_resources()
    
    if any(resources.values()):
        logger.info("Found the following resources:")
        for model_type, paths in resources.items():
            if paths:
                logger.info(f"{model_type}:")
                for path in paths:
                    logger.info(f"  - {path}")
    else:
        logger.info("No ANPE-related resources found.")
    
    logger.info("-" * 50)
    
    if not any(resources.values()):
        return {
            "spacy": True,
            "benepar": True,
            "nltk": True
        }
    
    results = {
        "spacy": remove_spacy_model(logger=logger),
        "benepar": remove_benepar_model(logger=logger),
        "nltk": remove_nltk_data(logger=logger)
    }
    
    logger.info("-" * 50)
    logger.info("Cleanup Summary:")
    for model, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"{model}: {status}")
    
    return results

def main():
    """Main entry point for the cleanup script."""
    parser = argparse.ArgumentParser(
        description="Remove ANPE-related models and caches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("-y", "--yes", action="store_true",
                       help="Skip confirmation prompt")
    args = parser.parse_args()
    
    logger = setup_logging(args.verbose)
    
    if not args.yes:
        logger.warning("This will remove all ANPE-related models from your system.")
        logger.warning("Models will need to be re-downloaded when you next use ANPE.")
        response = input("Do you want to continue? [y/N] ").lower()
        if response != 'y':
            logger.info("Operation cancelled.")
            return 1
    
    results = clean_all(args.verbose, logger)
    
    # Return 0 if all operations succeeded, 1 if any failed
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main()) 