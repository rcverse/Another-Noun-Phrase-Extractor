#!/usr/bin/env python3
"""Utility script to remove all ANPE-related models and caches."""

import os
import shutil
import subprocess
import sys
import nltk
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Callable, TypedDict
import logging
import site
import spacy
import importlib.util
import importlib.metadata

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

def _find_potential_nltk_benepar_locations() -> Dict[str, List[str]]:
    """Find base directories where NLTK/Benepar models might reside."""
    locations = {
        "nltk": [],
        "benepar": []
        # No longer searching for spaCy paths here, handled by importlib elsewhere
    }
    
    # NLTK data paths
    locations["nltk"].extend(nltk.data.path)
    
    # Add user's home directory NLTK path
    home = os.path.expanduser("~")
    nltk_home_path = os.path.join(home, "nltk_data")
    if os.path.isdir(nltk_home_path) and nltk_home_path not in locations["nltk"]:
        locations["nltk"].append(nltk_home_path)
    
    # Add site-packages NLTK paths
    for site_pkg in site.getsitepackages():
        nltk_site_path = os.path.join(site_pkg, "nltk_data")
        if os.path.isdir(nltk_site_path) and nltk_site_path not in locations["nltk"]:
             locations["nltk"].append(nltk_site_path)
    
    # Benepar paths (usually inside NLTK data/models)
    # Check existence before adding
    for nltk_path in locations["nltk"]:
        benepar_base = os.path.join(nltk_path, "models")
        # We only care about the base 'models' dir for Benepar lookups
        if os.path.isdir(benepar_base) and benepar_base not in locations["benepar"]:
             locations["benepar"].append(benepar_base)

    # Remove duplicates and ensure paths exist
    for key in locations:
        # Filter out non-existent paths just in case, though checks were added above
        locations[key] = sorted(list(set(p for p in locations[key] if os.path.isdir(p))))
        
    return locations

def _find_spacy_package_path(model_name: str) -> Optional[str]:
    """Find the directory path of an installed spaCy model package."""
    logger = logging.getLogger('clean_models') # Use named logger
    try:
        spec = importlib.util.find_spec(model_name)
        if spec and spec.origin:
            # The path is usually the directory containing the __init__.py file
            model_path = str(Path(spec.origin).parent)
            # Basic check: does the path look plausible? (e.g., contains 'en_core_web')
            # Ensure it's actually a directory, not just a file spec
            if os.path.isdir(model_path) and model_name in model_path:
                 logger.debug(f"[_find_spacy_package_path] Found path for {model_name}: {model_path}") # Updated log prefix
                 return model_path
            else:
                logger.debug(f"[_find_spacy_package_path] Found spec origin for {model_name} ({spec.origin}), but parent dir ({model_path}) seems incorrect or doesn't contain model name.") # Updated log prefix
                return None # Path doesn't look right
    except ModuleNotFoundError:
        logger.debug(f"[_find_spacy_package_path] Model {model_name} package not found.") # Updated log prefix
        pass # Model not found as an importable package
    except Exception as e:
        # Log unexpected errors during spec finding
        logger.debug(f"[_find_spacy_package_path] Error finding spec for {model_name}: {e}", exc_info=True) # Updated log prefix
        pass
    return None

# --- Uninstallation Helpers ---

def _remove_path_with_logging(resource_path: str, resource_type: str, logger: logging.Logger, log_callback: Optional[Callable[[str], None]] = None) -> bool:
    """Helper to remove a specific file or directory path with logging to callback."""
    base_name = os.path.basename(resource_path).replace('.zip', '')
    log_prefix = f"[_remove_path_with_logging {resource_type} {base_name}]"

    if not os.path.exists(resource_path):
        msg = f"{log_prefix} Path not found, nothing to remove: {resource_path}"
        logger.debug(msg)
        if log_callback:
            log_callback(msg)
        return True

    msg = f"{log_prefix} Attempting removal of: {resource_path}"
    logger.info(msg)
    if log_callback:
        log_callback(msg)

    try:
        if os.path.isdir(resource_path):
            shutil.rmtree(resource_path)
            msg = f"{log_prefix} ✓ Removed directory: {resource_path}"
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        elif os.path.isfile(resource_path):
            os.remove(resource_path)
            msg = f"{log_prefix} ✓ Removed file: {resource_path}"
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        else:
            msg = f"{log_prefix} ? Path exists but is neither file nor directory: {resource_path}"
            logger.warning(msg)
            if log_callback:
                log_callback(msg)
            return False # Treat as failure
        return True
    except Exception as e:
        msg = f"{log_prefix} ! Failed to remove {resource_path}: {e}"
        logger.error(msg, exc_info=True)
        if log_callback:
            log_callback(msg)
        return False

def uninstall_spacy_model(model_name: str, logger: logging.Logger = None, log_callback: Optional[Callable[[str], None]] = None) -> bool:
    """Attempt to uninstall a specific spaCy model package.

    Performs `pip uninstall` and then verifies removal by checking the expected
    installation path, attempting manual removal if needed.

    Args:
        model_name (str): The package name (e.g., 'en_core_web_md').
        logger (logging.Logger, optional): Logger instance. If None, gets 'clean_models'. Defaults to None.
        log_callback (Optional[Callable[[str], None]]): Optional callback for real-time log output.

    Returns:
        bool: True if removal was successful (or resource wasn't found), False if an error occurred.
    """
    if logger is None:
        logger = logging.getLogger('clean_models')

    # --- Handle Package Name Input (Validation Added) --- 
    if '/' in model_name or '\\' in model_name or not model_name:
         msg = f"[Uninstall spaCy] Invalid input '{model_name}'. Expected a valid spaCy model package name."
         logger.error(msg)
         if log_callback:
             log_callback(msg)
         return False
    
    # Proceed with package name logic
    package_name = model_name
    msg = f"[Uninstall spaCy {package_name}] Attempting uninstall via pip first..."
    logger.info(msg)
    if log_callback:
        log_callback(msg)
        
    pip_uninstalled_ok = False
    pip_said_not_installed = False
    manual_remove_needed = False
    manual_remove_ok = True # Default to True unless manual removal is tried and fails
    final_path_check = None

    # 1. Attempt pip uninstall
    try:
        pip_command = [sys.executable, "-m", "pip", "uninstall", "-y", package_name]
        msg = f"[Uninstall spaCy {package_name}] Running command: {' '.join(pip_command)}"
        logger.info(msg)
        if log_callback:
            log_callback(msg)
            
        process = subprocess.Popen(
            pip_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1
        )
        
        stdout_lines = []
        for line in process.stdout:
            line = line.strip()
            stdout_lines.append(line)
            logger.debug(f"pip uninstall stdout: {line}")
            if log_callback:
                log_callback(line)
                
        return_code = process.wait()
        stdout_text = "\n".join(stdout_lines)
        
        if return_code == 0:
            msg = f"[Uninstall spaCy {package_name}] ✓ pip uninstall command finished successfully."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
            pip_uninstalled_ok = True
            manual_remove_needed = True # Still check path as a cleanup step
        elif "not installed" in stdout_text.lower() or "WARNING: Skipping" in stdout_text:
            msg = f"[Uninstall spaCy {package_name}] - pip reported package not installed."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
            pip_said_not_installed = True
            manual_remove_needed = True # Definitely check path if pip didn't find it
        else:
            msg = f"[Uninstall spaCy {package_name}] ! pip uninstall command failed. Return code: {return_code}"
            logger.error(msg)
            if log_callback:
                log_callback(msg)
            # Overall failure will be determined later, but pip part failed.
            manual_remove_needed = True # Try manual removal as fallback
            
    except Exception as e:
        msg = f"[Uninstall spaCy {package_name}] ! Error running pip uninstall: {e}"
        logger.error(msg, exc_info=True)
        if log_callback:
            log_callback(msg)
        manual_remove_needed = True # Try manual removal after exception

    # 2. Manual Path Check & Removal (if needed)
    if manual_remove_needed:
        msg = f"[Uninstall spaCy {package_name}] Checking installation path for potential manual cleanup..."
        logger.info(msg)
        if log_callback:
            log_callback(msg)
            
        final_path_check = _find_spacy_package_path(package_name)
        
        if final_path_check and os.path.exists(final_path_check):
            msg = f"[Uninstall spaCy {package_name}] Found existing directory: {final_path_check}. Attempting removal..."
            # No need for extra logging here, helper does it.
            # logger.info(msg)
            # if log_callback:
            #     log_callback(msg)
            # Call the new helper function for removal
            manual_remove_ok = _remove_path_with_logging(
                resource_path=final_path_check,
                resource_type="spaCy", # Pass descriptive type
                logger=logger,
                log_callback=log_callback
            )
        elif final_path_check:
             msg = f"[Uninstall spaCy {package_name}] - Path {final_path_check} reported but does not exist. No manual removal needed."
             logger.debug(msg)
             if log_callback:
                 log_callback(msg)
        else:
            msg = f"[Uninstall spaCy {package_name}] - Package path could not be found via importlib. No manual removal possible."
            logger.info(msg)
            if log_callback:
                log_callback(msg)

    # 3. Determine Final Status
    # Success if: 
    # - pip uninstall worked AND manual removal wasn't needed or succeeded.
    # - pip said not installed AND manual removal found nothing or succeeded.
    # Failure if: 
    # - pip failed AND manual removal failed or wasn't possible.
    # - pip worked/not installed BUT manual removal failed.
    
    overall_success = (pip_uninstalled_ok and manual_remove_ok) or \
                      (pip_said_not_installed and manual_remove_ok and not (final_path_check and os.path.exists(final_path_check)))
                      
    if overall_success:
         if pip_uninstalled_ok:
             status_message = f"[Uninstall spaCy {package_name}] Completed: Successfully uninstalled via pip" + (f" and verified path clean ({final_path_check})." if final_path_check else ".")
         elif pip_said_not_installed and not (final_path_check and os.path.exists(final_path_check)):
             status_message = f"[Uninstall spaCy {package_name}] Completed: Model not found via pip or path."
         elif pip_said_not_installed and manual_remove_ok:
             status_message = f"[Uninstall spaCy {package_name}] Completed: Model not found via pip, but found and removed path {final_path_check}."
         else: # Should not happen with current logic, but catch-all
             status_message = f"[Uninstall spaCy {package_name}] Completed successfully (unknown state)."
         logger.info(status_message)
         if log_callback:
             log_callback(status_message)
    else:
        status_message = f"[Uninstall spaCy {package_name}] Failed: Could not completely remove model. Check previous errors."
        logger.error(status_message)
        if log_callback:
            log_callback(status_message)
            
    return overall_success

def uninstall_benepar_model(model_name: str, logger: logging.Logger = None, log_callback: Optional[Callable[[str], None]] = None) -> bool:
    """Attempt to uninstall a specific Benepar model by searching known locations.

    Searches NLTK data paths for the model directory and corresponding .zip file
    and removes them if found.

    Args:
        model_name (str): The base name of the Benepar model (e.g., 'benepar_en3').
        logger (logging.Logger, optional): Logger instance. If None, gets 'clean_models'. Defaults to None.
        log_callback (Optional[Callable[[str], None]]): Optional callback function to receive real-time log output.

    Returns:
        bool: True if removal was successful or model wasn't found, False if an error occurred during removal.
    """
    # Use 'clean_models' logger if none is provided
    if logger is None:
        logger = logging.getLogger('clean_models')

    # Validate model_name (Added)
    if '/' in model_name or '\\' in model_name or not model_name:
         msg = f"[Uninstall Benepar] Invalid input '{model_name}'. Expected a valid Benepar model name (e.g., 'benepar_en3')."
         logger.error(msg)
         if log_callback:
             log_callback(msg)
         return False

    msg = f"[Uninstall Benepar] Attempting uninstall for name: {model_name}"
    logger.info(msg)
    if log_callback:
        log_callback(msg)
        
    overall_success = True
    found_something = False

    try:
        msg = "[Uninstall Benepar] Finding potential NLTK/Benepar model locations..."
        logger.debug(msg)
        if log_callback:
            log_callback(msg)
            
        # Use find_model_locations which consolidates logic for finding NLTK/Benepar paths
        locations = _find_potential_nltk_benepar_locations() # Use renamed helper
        # We check all potential base dirs where 'models/' might live
        benepar_base_dirs = locations.get("benepar", []) 
        # Also check other NLTK dirs just in case structure is odd
        nltk_dirs = locations.get("nltk", [])
        all_potential_bases = set(benepar_base_dirs + [os.path.join(p, "models") for p in nltk_dirs]) 
        
        checked_paths = set()

        if not all_potential_bases:
            msg = f"[Uninstall Benepar {model_name}] No potential NLTK/Benepar data paths found to check."
            logger.warning(msg)
            if log_callback:
                log_callback(msg)
            return True # Nothing to do, count as success

        for base_path in all_potential_bases:
            # Skip non-directories or paths checked via another route
            norm_base_path = os.path.normpath(base_path)
            if not os.path.isdir(norm_base_path) or norm_base_path in checked_paths:
                continue
            checked_paths.add(norm_base_path)

            msg = f"[Uninstall Benepar {model_name}] Checking in: {norm_base_path}"
            logger.debug(msg)
            if log_callback:
                log_callback(msg)
                
            # --- Attempt to remove directory --- 
            model_dir_path = os.path.join(norm_base_path, model_name)
            if os.path.exists(model_dir_path):
                found_something = True
                # No need for extra logging here, helper does it.
                # msg = f"[Uninstall Benepar {model_name}] Found resource directory. Attempting removal: {model_dir_path}"
                # logger.info(msg)
                # if log_callback:
                #     log_callback(msg)
                # Use helper for removal
                removal_success = _remove_path_with_logging(
                    resource_path=model_dir_path,
                    resource_type="Benepar", # Pass descriptive type
                    logger=logger,
                    log_callback=log_callback
                )
                if not removal_success:
                    overall_success = False
            
            # --- Attempt to remove corresponding zip file --- 
            zip_path = os.path.join(norm_base_path, f"{model_name}.zip")
            if os.path.exists(zip_path):
                # Don't mark found_something=True here unless dir wasn't found, 
                # as zip is often an intermediate artifact.
                if not os.path.exists(model_dir_path): found_something = True 
                
                # No need for extra logging here, helper does it.
                # msg = f"[Uninstall Benepar {model_name}] Found zip file. Attempting removal: {zip_path}"
                # logger.info(msg)
                # if log_callback:
                #     log_callback(msg)
                # Use helper for removal
                removal_success = _remove_path_with_logging(
                    resource_path=zip_path,
                    resource_type="Benepar Zip", # Pass descriptive type
                    logger=logger,
                    log_callback=log_callback
                )
                if not removal_success:
                    overall_success = False # Failed removal is an overall failure

        # Final status message based on findings and success
        if not found_something:
            msg = f"[Uninstall Benepar {model_name}] No matching resources found to remove in checked locations."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        elif overall_success:
            msg = f"[Uninstall Benepar {model_name}] Successfully completed removal attempts for found resources."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        else:
            msg = f"[Uninstall Benepar {model_name}] Completed removal attempts with errors (failed to remove some resources)."
            logger.error(msg)
            if log_callback:
                log_callback(msg)

        return overall_success
        
    except Exception as e:
        msg = f"[Uninstall Benepar {model_name}] ! Unexpected error during uninstall process: {e}"
        logger.error(msg, exc_info=True)
        if log_callback:
            log_callback(msg)
        return False

# Define a type hint for the spaCy package info
class SpacyPackageInfo(TypedDict):
    name: str
    path: Optional[str] # Updated: Path might not be found

# Define a type hint for Benepar resource info (NEW)
class BeneparResourceInfo(TypedDict):
    name: str
    path: str # Path is the primary identifier

# Define a type hint for the overall structure
class FoundResources(TypedDict):
    spacy: List[SpacyPackageInfo]
    benepar: List[BeneparResourceInfo] # Updated: Use new Benepar type hint

def find_resources() -> FoundResources:
    """Find all existing ANPE-related resources and their locations.

    Identifies installed spaCy models as packages (name and optional path) and
    Benepar models/zips by scanning known NLTK locations.

    Returns:
        FoundResources: Dictionary containing lists of found resources.
                        'spacy' contains dicts {'name': str, 'path': Optional[str]}.
                        'benepar' contains dicts {'name': str, 'path': str}.
    """
    # Get the correct named logger
    logger = logging.getLogger('clean_models')
    logger.debug("[Find Resources] Starting resource discovery...")

    locations = _find_potential_nltk_benepar_locations() # Use renamed helper
    found: FoundResources = {
        "spacy": [],
        "benepar": [],
    }

    # --- Find spaCy Models (as Packages) ---
    logger.debug("[Find Resources] Searching for installed spaCy model packages via importlib...")
    installed_packages = list(importlib.metadata.distributions())
    spacy_model_package_names = set(SPACY_MODEL_MAP.values()) # Actual package names

    found_spacy_names = set()
    for dist in installed_packages:
        pkg_name = dist.metadata['Name']
        if pkg_name in spacy_model_package_names:
            logger.debug(f"[Find Resources] Found potential spaCy package: {pkg_name}")
            # Use renamed helper
            model_path = _find_spacy_package_path(pkg_name)
            # Path might be None, still record the package name
            if pkg_name not in found_spacy_names:
                 found["spacy"].append({"name": pkg_name, "path": model_path})
                 found_spacy_names.add(pkg_name)
                 if model_path and os.path.exists(model_path):
                     logger.debug(f"[Find Resources] Verified path for {pkg_name}: {model_path}")
                 elif model_path:
                     logger.warning(f"[Find Resources] Found spaCy package {pkg_name}, path reported as {model_path} but does not exist.")
                 else:
                     logger.warning(f"[Find Resources] Found spaCy package {pkg_name} but could not determine its path.")
            else:
                 # This case should be less likely now with the check above, but keep for safety
                 logger.debug(f"[Find Resources] Duplicate detection for {pkg_name}, skipping.")
            # if model_path and os.path.exists(model_path): # Ensure path is valid
            #     logger.debug(f"[Find Resources] Verified path for {pkg_name}: {model_path}")
            #     if pkg_name not in found_spacy_names: # Avoid duplicates if found multiple ways
            #          found["spacy"].append({"name": pkg_name, "path": model_path})
            #          found_spacy_names.add(pkg_name)
            #     else:
            #         logger.debug(f"[Find Resources] Duplicate detection for {pkg_name} at {model_path}, skipping.")
            # else:
            #      logger.warning(f"[Find Resources] Found spaCy package {pkg_name} but couldn't verify its path ({model_path}). Skipping.")

    # --- Find Benepar Models (Dirs and Zips in NLTK paths) ---
    benepar_model_names = set(BENEPAR_MODEL_MAP.values())
    logger.debug(f"[Find Resources] Searching for {len(benepar_model_names)} Benepar models in NLTK paths...")

    checked_paths = set() # Track checked NLTK base paths
    found_benepar_resources = {} # Use dict to avoid duplicates based on path: {path: name}

    # Use specific benepar base dirs found by _find_potential_nltk_benepar_locations
    potential_benepar_bases = locations.get("benepar", [])
    # Also check standard nltk data dirs directly for models/ folder just in case
    potential_benepar_bases.extend([os.path.join(p, "models") for p in locations.get("nltk", [])])

    for base_path in set(potential_benepar_bases): # Use set to avoid checking same base path twice
        unique_base_path = os.path.normpath(base_path)
        if unique_base_path in checked_paths or not os.path.isdir(unique_base_path):
            continue
        checked_paths.add(unique_base_path)

        logger.debug(f"[Find Resources] Checking Benepar base path: {unique_base_path}")

        for model_name in benepar_model_names:
            # Check for directory
            dir_path = os.path.join(unique_base_path, model_name)
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                if dir_path not in found_benepar_resources:
                    # Store path -> name mapping
                    found_benepar_resources[dir_path] = model_name
                    logger.debug(f"[Find Resources] Found Benepar model directory: {dir_path} (Name: {model_name})")

            # Check for zip file
            zip_path = os.path.join(unique_base_path, f"{model_name}.zip")
            if os.path.exists(zip_path) and os.path.isfile(zip_path):
                 if zip_path not in found_benepar_resources:
                    # Store path -> name mapping
                    found_benepar_resources[zip_path] = model_name
                    logger.debug(f"[Find Resources] Found Benepar model zip: {zip_path} (Name: {model_name})")

    # Convert found benepar resources {path: name} to list of BeneparResourceInfo dicts
    found["benepar"] = [{"name": name, "path": path} for path, name in found_benepar_resources.items()]

    # --- Log Summary ---
    logger.debug("[Find Resources] Resource discovery finished.")

    spacy_count = len(found["spacy"])
    benepar_count = len(found["benepar"])

    if spacy_count > 0 or benepar_count > 0:
        logger.info(f"[Find Resources] Found {spacy_count} spaCy package(s) and {benepar_count} Benepar resource path(s).") # Changed to info level
        if found["spacy"]:
             logger.debug("  spaCy packages:")
             for pkg_info in found["spacy"]:
                 path_str = pkg_info['path'] if pkg_info['path'] else "Path not found"
                 logger.debug(f"    - Name: {pkg_info['name']}, Path: {path_str}")
        if found["benepar"]:
            logger.debug("  Benepar resources:")
            for res_info in found["benepar"]:
                logger.debug(f"    - Name: {res_info['name']}, Path: {res_info['path']}")
    else:
        logger.info("[Find Resources] No known ANPE resources found in checked locations.") # Changed to info level

    return found

def clean_all(logger: logging.Logger, log_callback: Optional[Callable[[str], None]] = None, force: bool = False) -> Dict[str, bool]:
    """Find and remove all known ANPE-related models and caches.

    Args:
        logger (logging.Logger): Logger instance.
        log_callback (Optional[Callable[[str], None]]): Optional callback for real-time logging.
        force (bool, optional): If True, skip user confirmation before removing resources. Defaults to False.

    Returns:
        Dict[str, bool]: Status dict indicating success (True) or failure (False) for each component
                         ('spacy', 'benepar') and an 'overall' status.
    """
    if logger is None: # Ensure logger exists even if called directly
        logger = logging.getLogger('clean_models')
        # If logger wasn't passed, we likely don't have a callback configured, 
        # but check anyway for safety
        if not hasattr(logger, 'handlers') or not logger.handlers:
             # Basic setup if completely unconfigured
             ch = logging.StreamHandler()
             ch.setLevel(logging.INFO)
             formatter = logging.Formatter('%(message)s')
             ch.setFormatter(formatter)
             logger.addHandler(ch)
             logger.setLevel(logging.INFO)

    msg = "[Clean All] Starting cleanup process..."
    logger.info(msg)
    if log_callback:
        log_callback(msg)

    status = {
        "spacy": True,
        "benepar": True,
        "overall": True,
    }

    # Find all resources first
    logger.info("[Clean All] Finding installed ANPE resources...")
    if log_callback:
        log_callback("[Clean All] Finding installed ANPE resources...") # Keep user informed
    found = find_resources()

    spacy_packages = found.get("spacy", [])
    benepar_resources = found.get("benepar", [])
    spacy_count = len(spacy_packages)
    benepar_count = len(benepar_resources)

    if spacy_count == 0 and benepar_count == 0:
        msg = "[Clean All] No installed spaCy packages or Benepar resources found."
        logger.info(msg)
        if log_callback:
            log_callback(msg)
        return status # Return default success status

    # Log details of what was found
    msg = f"[Clean All] Found {spacy_count} spaCy package(s) and {benepar_count} Benepar resource(s)."
    logger.info(msg)
    if log_callback:
        log_callback(msg)
    if spacy_packages:
        logger.info("  SpaCy Packages:")
        if log_callback: log_callback("  SpaCy Packages:")
        for pkg in spacy_packages:
            path_info = f"(Path: {pkg['path']})" if pkg['path'] else "(Path not found)"
            detail_msg = f"    - {pkg['name']} {path_info}"
            logger.info(detail_msg)
            if log_callback: log_callback(detail_msg)
    if benepar_resources:
        logger.info("  Benepar Resources:")
        if log_callback: log_callback("  Benepar Resources:")
        for res in benepar_resources:
            detail_msg = f"    - {res['name']} (Path: {res['path']})"
            logger.info(detail_msg)
            if log_callback: log_callback(detail_msg)

    # --- Add User Confirmation ---
    proceed_with_removal = True # Default to True if force=True
    if not force:
        prompt_msg = "\n[Clean All] Proceed with removing all found resources? (y/N): "
        logger.info(prompt_msg.strip()) # Log prompt without newline
        if log_callback:
            log_callback(prompt_msg.strip()) # Send prompt to callback

        try:
            user_confirmation = input()
        except EOFError: # Handle non-interactive environments
            user_confirmation = 'n'
            err_msg = "[Clean All] Non-interactive environment detected or input stream closed. Assuming 'No'."
            logger.warning(err_msg)
            if log_callback:
                log_callback(err_msg)

        if user_confirmation.lower() != 'y':
            cancel_msg = "[Clean All] Removal cancelled by user."
            logger.info(cancel_msg)
            if log_callback:
                log_callback(cancel_msg)
            # Return success=True as no errors occurred, just cancellation
            proceed_with_removal = False # Don't proceed
            # Return status early if cancelled
            return status 
    
    if not proceed_with_removal: # Should only be reached if cancelled above
         # This return is technically redundant due to the return in the if block,
         # but kept for clarity. The logic inside the `if not force` block handles cancellation.
        return status

    # Log confirmation only if we went through the prompt or if forced
    if force:
         confirm_msg = "[Clean All] Force flag set. Proceeding with removal without confirmation..."
         logger.info(confirm_msg)
         if log_callback:
             log_callback(confirm_msg)
    elif proceed_with_removal: # Only log this if user confirmed 'y'
        confirm_msg = "[Clean All] User confirmed. Proceeding with removal..."
        logger.info(confirm_msg)
        if log_callback:
            log_callback(confirm_msg)

    # 1. Clean SpaCy Models (using standalone uninstaller)
    if not spacy_packages:
        msg = "[Clean All] No installed spaCy model packages found to remove."
        logger.info(msg)
        if log_callback:
            log_callback(msg)
    else:
        msg = f"[Clean All] Removing {spacy_count} spaCy model package(s)..."
        logger.info(msg)
        if log_callback:
            log_callback(msg)
        
        spacy_success_count = 0
        for package_info in spacy_packages:
            # Call standalone uninstall using only the PACKAGE NAME
            if uninstall_spacy_model(model_name=package_info['name'], logger=logger, log_callback=log_callback):
                spacy_success_count += 1
            else:
                # Failure for even one model marks this section as failed
                status["spacy"] = False
                # Error is logged within uninstall_spacy_model
                # msg = f"[Clean All] Failed to completely remove spaCy package: {package_info['name']}"
                # logger.error(msg)
                # if log_callback:
                #     log_callback(msg)
        
        if status["spacy"]:
            msg = f"[Clean All] Successfully completed removal attempts for all {spacy_count} spaCy package(s)."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        else:
             msg = f"[Clean All] Completed spaCy removal attempts with {spacy_count - spacy_success_count} failure(s). Check logs."
             logger.warning(msg)
             if log_callback:
                 log_callback(msg)

    # 2. Clean Benepar Models/Zips (using standalone uninstaller)
    if not benepar_resources:
        msg = "[Clean All] No Benepar model resources found to remove."
        logger.info(msg)
        if log_callback:
            log_callback(msg)
    else:
        msg = f"[Clean All] Removing {benepar_count} Benepar resource(s)..."
        logger.info(msg)
        if log_callback:
            log_callback(msg)
        
        benepar_success_count = 0
        for res_info in benepar_resources:
            # Call standalone uninstall using only the MODEL NAME
            if uninstall_benepar_model(model_name=res_info['name'], logger=logger, log_callback=log_callback):
                benepar_success_count += 1
            else:
                status["benepar"] = False
                # Error is logged within uninstall_benepar_model
                # msg = f"[Clean All] Failed to remove Benepar resource: {res_info['name']} at {res_info['path']}"
                # logger.error(msg)
                # if log_callback:
                #     log_callback(msg)
        
        if status["benepar"]:
            msg = f"[Clean All] Successfully completed removal attempts for all {benepar_count} Benepar resource(s)."
            logger.info(msg)
            if log_callback:
                log_callback(msg)
        else:
             msg = f"[Clean All] Completed Benepar removal attempts with {benepar_count - benepar_success_count} failure(s). Check logs."
             logger.warning(msg)
             if log_callback:
                 log_callback(msg)

    # 3. Clean Caches (Placeholder - implement if needed)
    # Example: 
    # msg = "[Clean All] Cleaning caches..."
    # logger.info(msg)
    # if log_callback: log_callback(msg)

    # Determine overall status
    status["overall"] = status["spacy"] and status["benepar"]

    if status["overall"]:
        final_msg = "[Clean All] Cleanup process completed successfully."
        logger.info(final_msg)
        if log_callback:
            log_callback(final_msg)
    else:
        failed_parts = [k for k, v in status.items() if k != "overall" and not v]
        final_msg = f"[Clean All] Cleanup process completed with errors in: {', '.join(failed_parts)}."
        logger.error(final_msg)
        if log_callback:
            log_callback(final_msg)

    return status

def main(log_callback: Optional[Callable[[str], None]] = None) -> int:
    """
    Entry point for CLI invocation.
    
    Args:
        log_callback (Optional[Callable[[str], None]]): Optional callback for real-time logging.
        
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Clean ANPE-related models and caches.')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--force', '-f', action='store_true', help='Force removal without user confirmation')
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(args.verbose)
    
    # Print banner
    banner_msg = "=== ANPE Model Cleanup Utility ==="
    print(banner_msg)
    if log_callback:
        log_callback(banner_msg)
    
    # Run the cleanup (Pass logger and callback)
    status = clean_all(logger=logger, log_callback=log_callback, force=args.force)
    
    # Print final status
    if status["overall"]:
        final_msg = "=== Cleanup completed successfully ==="
        print(final_msg)
        if log_callback:
            log_callback(final_msg)
        return 0
    else:
        failed_parts = [k for k, v in status.items() if k != "overall" and not v]
        final_msg = f"=== Cleanup completed with errors in: {', '.join(failed_parts)} ==="
        print(final_msg)
        if log_callback:
            log_callback(final_msg)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 