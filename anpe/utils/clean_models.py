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

# Import model maps to know all variants
from anpe.utils.setup_models import SPACY_MODEL_MAP, BENEPAR_MODEL_MAP

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
    
    # Remove duplicates
    for key in locations:
        locations[key] = sorted(list(set(locations[key])))
        
    return locations

def uninstall_spacy_model(model_name: str, logger: logging.Logger) -> bool:
    """Attempt to uninstall a specific spaCy model.
    
    Args:
        model_name (str): The full name of the spaCy model (e.g., 'en_core_web_md').
        logger (logging.Logger): Logger instance.

    Returns:
        bool: True if removal was successful or model wasn't found, False if an error occurred during removal.
    """
    if not model_name:
        logger.error("[Uninstall spaCy] No model name provided.")
        return False
        
    logger.info(f"[Uninstall spaCy] Attempting uninstall for: {model_name}")
    overall_success = True
    found_via_pip = False
    found_in_data = False

    # 1. Try pip uninstall
    try:
        logger.debug(f"[Uninstall spaCy {model_name}] Running pip uninstall...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "-y", model_name],
            check=False,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info(f"[Uninstall spaCy {model_name}] ✓ Successfully uninstalled via pip.")
            found_via_pip = True # It was installed via pip
        elif "not installed" in result.stderr.lower():
             logger.info(f"[Uninstall spaCy {model_name}] - Model not installed via pip.")
        else:
             # Log actual pip error
             logger.warning(f"[Uninstall spaCy {model_name}] ! pip uninstall command failed. Stderr: {result.stderr.strip()}")
             # This is a failure in the process, even if the model might not exist
             overall_success = False 
    except Exception as e:
        logger.error(f"[Uninstall spaCy {model_name}] ! Error running pip uninstall: {e}")
        overall_success = False

    # 2. Try removing from spaCy data directory
    try:
        data_path = spacy.util.get_data_path()
        if data_path and os.path.exists(data_path):
             model_path = os.path.join(data_path, model_name)
             logger.debug(f"[Uninstall spaCy {model_name}] Checking data path: {model_path}")
             if os.path.exists(model_path):
                 found_in_data = True
                 logger.info(f"[Uninstall spaCy {model_name}] Found in data path. Attempting removal...")
                 try:
                     if os.path.isdir(model_path):
                         shutil.rmtree(model_path)
                     else:
                         os.remove(model_path)
                     logger.info(f"[Uninstall spaCy {model_name}] ✓ Removed from data directory: {model_path}")
                 except Exception as e:
                     logger.error(f"[Uninstall spaCy {model_name}] ! Failed to remove from data directory {model_path}: {e}")
                     overall_success = False
             else:
                 logger.debug(f"[Uninstall spaCy {model_name}] - Not found in data path.")
        else:
             logger.debug("[Uninstall spaCy] - SpaCy data path not found or inaccessible for removal check.")
             
    except Exception as e:
        logger.warning(f"[Uninstall spaCy {model_name}] ! Could not access or process spaCy data directory: {e}")
        # Don't necessarily mark as overall failure if we just couldn't check

    # Final status log for this specific model
    if overall_success and (found_via_pip or found_in_data):
        logger.info(f"[Uninstall spaCy {model_name}] Completed removal attempts successfully.")
    elif overall_success and not (found_via_pip or found_in_data):
        logger.info(f"[Uninstall spaCy {model_name}] Model was not found via pip or in data path.")
    elif not overall_success:
         logger.error(f"[Uninstall spaCy {model_name}] Finished removal attempts with errors.")
         
    return overall_success

def uninstall_benepar_model(model_name: str, logger: logging.Logger) -> bool:
    """Attempt to uninstall a specific Benepar model.

    Args:
        model_name (str): The full name of the Benepar model (e.g., 'benepar_en3').
        logger (logging.Logger): Logger instance.
        
    Returns:
        bool: True if removal was successful or model wasn't found, False if an error occurred during removal.
    """
    if not model_name:
        logger.error("[Uninstall Benepar] No model name provided.")
        return False
        
    logger.info(f"[Uninstall Benepar] Attempting uninstall for: {model_name}")
    overall_success = True
    found_something = False

    try:
        locations = find_model_locations()
        checked_paths = set()
        
        for base_nltk_path in locations["nltk"]:
            if base_nltk_path in checked_paths:
                 continue
            checked_paths.add(base_nltk_path)
            
            base_path = os.path.join(base_nltk_path, "models") # Benepar models are in 'models' subdir
            if not os.path.isdir(base_path):
                logger.debug(f"[Uninstall Benepar {model_name}] NLTK models path not found or not a directory: {base_path}")
                continue
                 
            logger.debug(f"[Uninstall Benepar {model_name}] Checking in: {base_path}")
            # Remove directory
            model_path = os.path.join(base_path, model_name)
            if os.path.exists(model_path):
                found_something = True
                logger.info(f"[Uninstall Benepar {model_name}] Found directory. Removing: {model_path}")
                try:
                    if os.path.isdir(model_path):
                        shutil.rmtree(model_path)
                    else:
                        os.remove(model_path)
                    logger.info(f"[Uninstall Benepar {model_name}] ✓ Removed: {model_path}")
                except Exception as e:
                    logger.error(f"[Uninstall Benepar {model_name}] ! Failed to remove {model_path}: {e}")
                    overall_success = False
            else:
                 logger.debug(f"[Uninstall Benepar {model_name}] - Directory not found: {model_path}")
            
            # Remove corresponding zip file
            zip_path = model_path + ".zip"
            if os.path.exists(zip_path):
                found_something = True
                logger.info(f"[Uninstall Benepar {model_name}] Found zip file. Removing: {zip_path}")
                try:
                    os.remove(zip_path)
                    logger.info(f"[Uninstall Benepar {model_name}] ✓ Removed zip: {zip_path}")
                except Exception as e:
                    logger.error(f"[Uninstall Benepar {model_name}] ! Failed to remove zip file {zip_path}: {e}")
                    overall_success = False
            else:
                 logger.debug(f"[Uninstall Benepar {model_name}] - Zip file not found: {zip_path}")

        # Final status log
        if overall_success and found_something:
            logger.info(f"[Uninstall Benepar {model_name}] Completed removal attempts successfully.")
        elif overall_success and not found_something:
            logger.info(f"[Uninstall Benepar {model_name}] Model directory/zip not found in checked locations.")
        elif not overall_success:
            logger.error(f"[Uninstall Benepar {model_name}] Finished removal attempts with errors.")

    except Exception as e:
        logger.error(f"[Uninstall Benepar {model_name}] ! An unexpected error occurred: {e}")
        overall_success = False
        
    return overall_success

def remove_nltk_data(resources: List[str] = None, logger: logging.Logger = None) -> bool:
    """Remove NLTK resources (directories and zips) from all possible locations."""
    if resources is None:
        resources = ['punkt', 'punkt_tab']
    
    overall_success = True
    removed_something = False
    try:
        logger.info(f"[Remove NLTK] Removing resources: {resources}...") # Added prefix
        locations = find_model_locations()
        
        for nltk_path in locations["nltk"]:
            tokenizers_base = os.path.join(nltk_path, "tokenizers")
            if not os.path.isdir(tokenizers_base):
                logger.debug(f"[Remove NLTK] Path not found or not dir: {tokenizers_base}")
                continue # Skip if tokenizers subdir doesn't exist in this nltk path
                
            for resource in resources:
                logger.debug(f"[Remove NLTK] Checking for '{resource}' in {tokenizers_base}")
                # Remove directory
                resource_path = os.path.join(tokenizers_base, resource)
                if os.path.exists(resource_path):
                    removed_something = True
                    logger.info(f"[Remove NLTK] Found resource. Removing: {resource_path}")
                    try:
                        if os.path.isdir(resource_path):
                            shutil.rmtree(resource_path)
                        else:
                            os.remove(resource_path)
                        logger.info(f"[Remove NLTK] ✓ Removed resource dir/file: {resource_path}")
                    except Exception as e:
                        logger.warning(f"[Remove NLTK] ! Could not remove {resource} from {resource_path}: {e}")
                        overall_success = False
                else:
                     logger.debug(f"[Remove NLTK] - Resource dir/file not found: {resource_path}")
                
                # Remove corresponding zip file
                zip_path = resource_path + ".zip"
                if os.path.exists(zip_path):
                    removed_something = True
                    logger.info(f"[Remove NLTK] Found resource zip. Removing: {zip_path}")
                    try:
                        os.remove(zip_path)
                        logger.info(f"[Remove NLTK] ✓ Removed resource zip: {zip_path}")
                    except Exception as e:
                        logger.warning(f"[Remove NLTK] ! Could not remove zip file {zip_path}: {e}")
                        overall_success = False
                else:
                     logger.debug(f"[Remove NLTK] - Resource zip not found: {zip_path}")
        
        # Final log
        if removed_something and overall_success:
            logger.info(f"[Remove NLTK] ✓ Successfully removed specified resources ({', '.join(resources)}). ")
        elif removed_something and not overall_success:
            logger.warning(f"[Remove NLTK] ! Partially removed specified NLTK resources ({', '.join(resources)}). Some errors occurred.")
        elif not removed_something:
             logger.info(f"[Remove NLTK] ! Specified resources ({', '.join(resources)}) not found in any known location.")
            
    except Exception as e:
        logger.error(f"[Remove NLTK] ! Error removing NLTK resources: {e}")
        overall_success = False
    
    return overall_success

def find_resources() -> Dict[str, List[str]]:
    """Find all existing ANPE-related resources (dirs and zips) in known locations."""
    locations = find_model_locations()
    found_resources = {
        "spacy": [],
        "benepar": [],
        "nltk": []
    }
    
    # Find spaCy models
    spacy_model_names = set(SPACY_MODEL_MAP.values())
    try:
        data_path = spacy.util.get_data_path()
        if data_path and os.path.exists(data_path):
            for item in os.listdir(data_path):
                item_path = os.path.join(data_path, item)
                # Check if it matches known model names and is a directory
                if item in spacy_model_names and os.path.isdir(item_path):
                    found_resources["spacy"].append(item_path)
    except Exception as e: # Ignore errors if spacy or data path is missing
         logging.getLogger('clean_models').debug(f"Could not scan spacy data path: {e}")
        
    
    # Find Benepar models (dir and zip)
    benepar_model_names = set(BENEPAR_MODEL_MAP.values())
    for base_path in locations["benepar"]:
        if os.path.exists(base_path):
            for model_name in benepar_model_names:
                 # Check for directory
                 dir_path = os.path.join(base_path, model_name)
                 if os.path.exists(dir_path):
                      if dir_path not in found_resources["benepar"]:
                          found_resources["benepar"].append(dir_path)
                 # Check for zip file
                 zip_path = dir_path + ".zip"
                 if os.path.exists(zip_path):
                     if zip_path not in found_resources["benepar"]:
                          found_resources["benepar"].append(zip_path)
    
    # Find NLTK resources (dirs and zips)
    nltk_resources_to_find = ['punkt', 'punkt_tab']
    for nltk_path in locations["nltk"]:
        tokenizers_path = os.path.join(nltk_path, "tokenizers")
        if os.path.exists(tokenizers_path):
            for resource_name in nltk_resources_to_find:
                # Check for directory
                dir_path = os.path.join(tokenizers_path, resource_name)
                if os.path.exists(dir_path):
                     found_resources["nltk"].append(dir_path)
                # Check for zip file
                zip_path = dir_path + ".zip"
                if os.path.exists(zip_path) and zip_path not in found_resources["nltk"]:
                     found_resources["nltk"].append(zip_path)
    
    return found_resources

def clean_all(verbose: bool = False, logger: logging.Logger = None) -> Dict[str, bool]:
    """Remove ALL known ANPE-related models using granular uninstall functions.
       Returns status summary for each category (spaCy, Benepar, NLTK).
    """
    if logger is None:
        logger = setup_logging(verbose)
    
    logger.info("[Clean All] Starting ANPE model cleanup...")
    logger.info("[Clean All] " + "-" * 50)
    
    # Optional: Keep the resource scan for user info, though clean will try all anyway
    logger.info("[Clean All] Scanning for existing ANPE-related resources...")
    resources = find_resources()
    if any(resources.values()):
        logger.info("[Clean All] Found the following potential resources (will attempt removal regardless):")
        for model_type, paths in resources.items():
            if paths:
                logger.info(f"  {model_type}:")
                for path in paths:
                    logger.info(f"    - {path}")
    else:
        logger.info("[Clean All] No ANPE-related resources initially found (still attempting removal)." )
    
    logger.info("[Clean All] " + "-" * 50)
    logger.info("[Clean All] Attempting removal of all known models...")
    
    # Track overall success for each category
    spacy_success = True
    benepar_success = True
    
    # Uninstall all known spaCy models
    logger.info("[Clean All] --- spaCy Cleanup --- ")
    for model_name in set(SPACY_MODEL_MAP.values()): # Use set to avoid duplicates if map contains them
        if not uninstall_spacy_model(model_name, logger):
             spacy_success = False # Mark category as failed if any sub-task fails
             
    # Uninstall all known Benepar models
    logger.info("[Clean All] --- Benepar Cleanup --- ")
    for model_name in set(BENEPAR_MODEL_MAP.values()):
         if not uninstall_benepar_model(model_name, logger):
              benepar_success = False
              
    # Remove NLTK data (uses existing function)
    logger.info("[Clean All] --- NLTK Cleanup --- ")
    nltk_success = remove_nltk_data(logger=logger)
    
    # Compile results summary
    results = {
        "spacy": spacy_success,
        "benepar": benepar_success,
        "nltk": nltk_success
    }
    
    logger.info("[Clean All] " + "-" * 50)
    logger.info("[Clean All] Cleanup Summary:")
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
        logger.warning("This will attempt to remove all known ANPE-related models")
        logger.warning(f" (spaCy: {', '.join(set(SPACY_MODEL_MAP.values()))},")
        logger.warning(f"  Benepar: {', '.join(set(BENEPAR_MODEL_MAP.values()))},")
        logger.warning(f"  NLTK: punkt, punkt_tab)")
        logger.warning("from all known locations on your system.")
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