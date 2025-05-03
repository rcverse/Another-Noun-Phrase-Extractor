"""Export functionality for noun phrases."""

import json
import csv
import os
from pathlib import Path
from typing import  Dict, Optional
import datetime

from anpe.utils.anpe_logger import get_logger


class ANPEExporter:
    """Handles export of noun phrases in various formats."""
    
    def __init__(self):
        """Initialize the exporter."""
        self.logger = get_logger('exporter')
        self.logger.debug("Initializing ANPEExporter")
    
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
        self.logger.info(f"Exporting noun phrases to {resolved_path} in {format} format")
        
        # Validate format
        valid_formats = ["txt", "csv", "json"]
        if format not in valid_formats:
            error_msg = f"Invalid format: {format}. Must be one of {valid_formats}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check data structure
        if not isinstance(data, dict) or "results" not in data:
            error_msg = "Invalid data structure. Expected dictionary with 'results' key"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.debug(f"Exporting {len(data['results'])} top-level noun phrases to {resolved_path}")
        
        # Use resolved path for all operations
        try:
            if format == "txt":
                return str(self._export_txt(data, resolved_path))
            elif format == "csv":
                return str(self._export_csv(data, resolved_path))
            elif format == "json":
                return str(self._export_json(data, resolved_path))
        except Exception as e:
            self.logger.error(f"Error exporting to {format} at {resolved_path}: {str(e)}")
            raise
    
    def _export_txt(self, data, output_filepath):
        """Export to text file with hierarchical structure."""
        self.logger.debug(f"Exporting to text file: {output_filepath}")
        
        try:
            with open(output_filepath, 'w', encoding='utf-8') as f:
                # Write Header
                f.write("--- ANPE Noun Phrase Extraction Results ---\n")
                # Write Timestamp (now top-level)
                f.write(f"Timestamp: {data.get('timestamp', 'N/A')}\n")

                # Write Configuration Used (if available)
                config = data.get("configuration", {})
                if config:
                    f.write("\n--- Configuration Used ---\n")
                    # Display output flags from configuration first
                    includes_nested = config.get('nested_requested', False)
                    includes_metadata = config.get('metadata_requested', False)
                    f.write(f"Output includes Nested NPs: {includes_nested}\n")
                    f.write(f"Output includes Metadata: {includes_metadata}\n")
                    f.write("-----\n") # Separator within config section

                    for key, value in config.items():
                        # Skip flags already displayed
                        if key in ['nested_requested', 'metadata_requested']:
                            continue
                        
                        # Format list values nicely
                        if isinstance(value, list):
                            value_str = ", ".join(map(str, value)) if value else "None"
                        else:
                            value_str = str(value)
                        # Format key nicely (replace underscores, title case)
                        key_str = key.replace("_", " ").title()
                        f.write(f"{key_str}: {value_str}\n")
                    f.write("--------------------------\n") # End configuration section
                else:
                    # Fallback if configuration section is missing
                    f.write("Output includes Nested NPs: N/A\n") 
                    f.write("Output includes Metadata: N/A\n")
                    f.write("Configuration details not available.\n")
                
                f.write("\n--- Extraction Results ---\n") # Separator before results
                
                # Export each top-level NP and its nested NPs
                if "results" in data and data["results"]:
                    for np_item in data['results']:
                        self._write_np_to_txt(f, np_item, level=0)
                else:
                    f.write("No noun phrases extracted.\n")
            
            self.logger.info(f"Successfully exported to text file: {output_filepath}")
            return output_filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting to text file: {str(e)}")
            raise
    
    def _write_np_to_txt(self, file, np_item, level=0):
        """Write a noun phrase and its children to a text file with proper indentation."""
        # Determine bullet type based on level
        bullet = "•" if level == 0 else "◦"
        indent = "  " * level
        
        # Write NP with ID (always included for reference)
        file.write(f"{indent}{bullet} [{np_item['id']}] {np_item['noun_phrase']}\n")
        
        # Write metadata if present
        if "metadata" in np_item:
            metadata = np_item["metadata"]
            if "length" in metadata:
                file.write(f"{indent}  Length: {metadata['length']}\n")
            if "structures" in metadata:
                structures_str = ", ".join(metadata['structures']) if isinstance(metadata['structures'], list) else metadata['structures']
                file.write(f"{indent}  Structures: [{structures_str}]\n")
        
        # Write children recursively
        for child in np_item.get("children", []):
            self._write_np_to_txt(file, child, level + 1)
            
        # Add a blank line after top-level NPs
        if level == 0:
            file.write("\n")
    
    def _export_csv(self, data, output_filepath):
        """Export to CSV file with flattened hierarchy."""
        self.logger.debug(f"Exporting to CSV file: {output_filepath}")
        
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
            
            self.logger.info(f"Successfully exported to CSV file: {output_filepath}")
            return output_filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV file: {str(e)}")
            raise
    
    def _flatten_np_hierarchy(self, np_structure, parent_id=None, flattened_list=None):
        """
        Convert hierarchical NP structure to flat list for CSV export.
        """
        if flattened_list is None:
            flattened_list = []
            
        # Create flat representation of current NP
        flat_np = {
            "id": np_structure["id"],
            "level": np_structure["level"],
            "parent_id": parent_id,
            "noun_phrase": np_structure["noun_phrase"]
        }
        
        # Add metadata if present
        if "metadata" in np_structure:
            if "length" in np_structure["metadata"]:
                flat_np["length"] = np_structure["metadata"]["length"]
            if "structures" in np_structure["metadata"]:
                flat_np["structures"] = np_structure["metadata"]["structures"]
            
        flattened_list.append(flat_np)
        
        # Process children recursively
        for child in np_structure.get("children", []):
            self._flatten_np_hierarchy(
                child, 
                parent_id=np_structure["id"], 
                flattened_list=flattened_list
            )
            
        return flattened_list
    
    def _export_json(self, data, output_filepath):
        """Export to JSON file with hierarchical structure."""
        self.logger.debug(f"Exporting to JSON file: {output_filepath}")
        
        try:
            # Clean up non-JSON-serializable objects if any
            cleaned_data = self._clean_for_json(data)
            
            # Write properly formatted JSON
            with open(output_filepath, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2)
            
            self.logger.info(f"Successfully exported to JSON file: {output_filepath}")
            return output_filepath
            
        except Exception as e:
            self.logger.error(f"Error exporting to JSON file: {str(e)}")
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