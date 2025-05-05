"""Export functionality for noun phrases."""

# --- Standard Logging Setup ---
import logging
logger = logging.getLogger(__name__)
# --- End Standard Logging ---

import json
import csv
import os
from pathlib import Path
from typing import  Dict, Optional
import datetime

# REMOVED: from anpe.utils.anpe_logger import get_logger


class ANPEExporter:
    """Handles export of noun phrases in various formats."""
    
    def __init__(self):
        """Initialize the exporter."""
        # REMOVED: self.logger = get_logger('exporter')
        logger.debug("Initializing ANPEExporter") # Use module logger
    
    def export(self, 
               data: Dict,
               format: str,
               output_filepath: str) -> str:
        """
        Export noun phrases to file.

        Args:
            data: Dictionary containing extraction results.
            format: Export format ("txt", "csv", or "json").
            output_filepath: Full path to the output file.
            
        Returns:
            The resolved path to the exported file.
            
        Raises:
            ValueError: If an invalid format is specified or data structure is incorrect.
            IOError: If there are issues writing the file.
        """
        # Resolve the path immediately
        resolved_path = Path(output_filepath).resolve()
        logger.info(f"Exporting noun phrases to {resolved_path} in {format} format") # Use module logger
        
        # Validate format
        valid_formats = ["txt", "csv", "json"]
        if format not in valid_formats:
            error_msg = f"Invalid format: {format}. Must be one of {valid_formats}"
            logger.error(error_msg) # Use module logger
            raise ValueError(error_msg)
        
        # Check data structure
        if not isinstance(data, dict) or "results" not in data:
            error_msg = "Invalid data structure. Expected dictionary with 'results' key"
            logger.error(error_msg) # Use module logger
            raise ValueError(error_msg)
        
        logger.debug(f"Exporting {len(data['results'])} top-level noun phrases to {resolved_path}") # Use module logger
        
        # Use resolved path for all operations
        try:
            if format == "txt":
                return str(self._export_txt(data, resolved_path))
            elif format == "csv":
                return str(self._export_csv(data, resolved_path))
            elif format == "json":
                return str(self._export_json(data, resolved_path))
        except Exception as e:
            logger.error(f"Error exporting to {format} at {resolved_path}: {str(e)}") # Use module logger
            raise
    
    def _export_txt(self, data: Dict, output_filepath: Path) -> Path:
        """Exports data to a simple text file, one NP per line."""
        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                # Write Header
                f.write("--- ANPE Noun Phrase Extraction Results ---\n")
                f.write(f"Timestamp: {data.get('timestamp', 'N/A')}\n")
                f.write("\n--- Configuration Used ---\n")
                
                config = data.get('configuration', {})
                
                # Read flags from the config dict for reporting
                nested_requested = config.get('nested_requested', False) # Default to False if missing
                metadata_requested = config.get('metadata_requested', False) # Default to False if missing
                
                f.write(f"Output includes Nested NPs: {nested_requested}\n")
                f.write(f"Output includes Metadata: {metadata_requested}\n")
                f.write("-----\n")
                
                # Write other config details from the data dictionary
                f.write(f"Accept Pronouns: {config.get('accept_pronouns', True)}\n")
                structure_filters = config.get('structure_filters', None)
                structure_filters_str = ", ".join(structure_filters) if structure_filters else "None"
                f.write(f"Structure Filters: {structure_filters_str}\n")
                f.write(f"Newline Breaks: {config.get('newline_breaks', True)}\n")
                f.write(f"Spacy Model Used: {config.get('spacy_model_used', 'unknown')}\n")
                f.write(f"Benepar Model Used: {config.get('benepar_model_used', 'unknown')}\n")
                f.write("--------------------------\n")

                f.write("\n--- Extraction Results ---\n")
                if not data.get('results'):
                    f.write("(No noun phrases extracted with the given configuration.)\n")
                else:
                    for item in data['results']:
                        self._write_txt_item(f, item, metadata=metadata_requested, include_nested=nested_requested)
            
            return output_filepath
        except IOError as e:
            logger.error(f"Error writing to text file {output_filepath}: {e}") # Use module logger
            raise
    
    def _write_txt_item(self, file, np_item, metadata=False, include_nested=False):
        """Write a noun phrase and its children to a text file with proper indentation."""
        # --- Add check: Ensure np_item is a dictionary ---
        if not isinstance(np_item, dict):
            logger.error(f"_write_txt_item expected a dict but received {type(np_item)}: {np_item}") # Use module logger
            return # Skip this item
        # --- End check ---

        # Determine bullet type based on level (Safely get level, default to 1 if missing)
        level = np_item.get("level", 1)
        bullet = "•" if level == 1 else "◦"
        indent = "  " * (level - 1) # Indentation starts from 0 for level 1
        
        # Write NP with ID (always included for reference)
        file.write(f"{indent}{bullet} [{np_item.get('id', 'N/A')}] {np_item.get('noun_phrase', '(Error: Missing Text)')}\n")
        
        # Write metadata if present and requested
        # Check if metadata key exists and its value is a dictionary
        metadata_dict = np_item.get("metadata")
        if metadata and isinstance(metadata_dict, dict):
            if "length" in metadata_dict:
                file.write(f"{indent}  Length: {metadata_dict['length']}\n")
            if "structures" in metadata_dict:
                structures = metadata_dict.get('structures', []) # Safely get structures list
                structures_str = ", ".join(structures) if isinstance(structures, list) else str(structures)
                file.write(f"{indent}  Structures: {structures_str}\n")
        
        # Write children recursively
        children_list = np_item.get("children", [])
        if isinstance(children_list, list): # Ensure children is a list
            for child in children_list:
                # Recursive call already has the type check at the beginning
                self._write_txt_item(file, child, metadata=metadata, include_nested=include_nested)
        elif children_list: # Log if children exists but is not a list
             logger.warning(f"Expected 'children' to be a list but found {type(children_list)} for item ID {np_item.get('id')}") # Use module logger
        
        # Add a blank line after top-level NPs
        if level == 1:
            file.write("\n")
    
    def _export_csv(self, data, output_filepath):
        """Export to CSV file with flattened hierarchy."""
        logger.debug(f"Exporting to CSV file: {output_filepath}") # Use module logger
        
        try:
            # Flatten the hierarchical structure
            flattened_nps = []
            for np_item in data['results']:
                self._flatten_np_hierarchy(np_item, flattened_list=flattened_nps)
            
            # Determine which columns to include
            # includes_metadata = data['metadata'].get('includes_metadata', False)
            # Read from the new location in configuration
            includes_metadata = data.get('configuration', {}).get('metadata_requested', False)
            
            with open(output_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # Write header row
                if includes_metadata:
                    writer.writerow(["ID", "Level", "Parent_ID", "Noun_Phrase", "Length", "Structures"])
                else:
                    writer.writerow(["ID", "Level", "Parent_ID", "Noun_Phrase"])
                
                # Write data rows
                for np in flattened_nps:
                    if includes_metadata:
                        structures_str = "|".join(np.get("structures", [])) if isinstance(np.get("structures", []), list) else np.get("structures", "")
                        writer.writerow([
                            np["id"],
                            np["level"],
                            np.get("parent_id", ""),
                            np["noun_phrase"],
                            np.get("length", ""),
                            structures_str
                        ])
                    else:
                        writer.writerow([
                            np["id"],
                            np["level"],
                            np.get("parent_id", ""),
                            np["noun_phrase"]
                        ])
            
            return output_filepath
            
        except Exception as e:
            logger.error(f"Error exporting to CSV file: {str(e)}") # Use module logger
            raise
    
    def _flatten_np_hierarchy(self, np_structure, parent_id=None, flattened_list=None):
        """
        Convert hierarchical NP structure to flat list for CSV export.
        """
        if flattened_list is None:
            flattened_list = []
            
        # --- Add check: Ensure np_structure is a dictionary ---
        if not isinstance(np_structure, dict):
            logger.error(f"_flatten_np_hierarchy expected a dict but received {type(np_structure)}: {np_structure}") # Use module logger
            return # Skip this item
        # --- End check ---

        # Create flat representation of current NP using .get for safety
        flat_np = {
            "id": np_structure.get("id", "N/A"),
            "level": np_structure.get("level", -1),
            "noun_phrase": np_structure.get("noun_phrase", "(Error: Missing Text)"),
            "parent_id": parent_id
        }
        
        # Add metadata if present and is a dictionary
        metadata_dict = np_structure.get("metadata")
        if isinstance(metadata_dict, dict):
            flat_np["length"] = metadata_dict.get("length", "")
            flat_np["structures"] = metadata_dict.get("structures", [])
        else:
            # Ensure keys exist even if metadata is missing/malformed
            flat_np["length"] = ""
            flat_np["structures"] = []

        flattened_list.append(flat_np)
        
        # Recursively process children if it's a list
        children_list = np_structure.get("children", [])
        if isinstance(children_list, list):
            current_np_id = np_structure.get("id", "N/A") # Get ID for parent_id reference
            for child in children_list:
                # Recursive call already has the type check
                self._flatten_np_hierarchy(child, current_np_id, flattened_list)
        elif children_list:
             logger.warning(f"Expected 'children' to be a list but found {type(children_list)} for item ID {np_structure.get('id')}") # Use module logger
            
        return flattened_list
    
    def _export_json(self, data, output_filepath):
        """Export to JSON file with hierarchical structure."""
        logger.debug(f"Exporting to JSON file: {output_filepath}")
        
        try:
            # Clean up non-JSON-serializable objects if any
            cleaned_data = self._clean_for_json(data)
            
            # Write properly formatted JSON
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2)
            
            return output_filepath
            
        except Exception as e:
            logger.error(f"Error exporting to JSON file: {str(e)}")
            raise
            
    def _clean_for_json(self, obj):
        """
        Ensure all objects are JSON-serializable.
        """
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Convert any other type to string
            return str(obj) 