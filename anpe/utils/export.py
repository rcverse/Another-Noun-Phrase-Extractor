"""Export functionality for noun phrases."""

import json
import csv
import os
from pathlib import Path
from typing import  Dict, Optional
import datetime

from anpe.utils.logging import get_logger


class ANPEExporter:
    """Handles export of noun phrases in various formats."""
    
    def __init__(self):
        """Initialize the exporter."""
        self.logger = get_logger('exporter')
        self.logger.debug("Initializing ANPEExporter")
    
    def export(self, 
               data: Dict,
               format: str = "txt", 
               export_dir: Optional[str] = None) -> str:
        """
        Export noun phrases to file.
        """
        self.logger.info(f"Exporting noun phrases in {format} format")
        
        # Validate format
        valid_formats = ["txt", "csv", "json"]
        if format not in valid_formats:
            error_msg = f"Invalid format: {format}. Must be one of {valid_formats}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Use current directory if no export directory is provided
        if not export_dir:
            export_dir = os.getcwd()
            self.logger.debug(f"No export directory provided, using current directory")
        
        # Create directory path object
        dir_path = Path(export_dir)
        
        # Ensure directory exists
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Export directory verified/created: {export_dir}")
        except Exception as e:
            self.logger.error(f"Error with export directory: {str(e)}")
            raise ValueError(f"Invalid export directory: {export_dir}")
            
        # Generate timestamped filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"anpe_export_{timestamp}.{format}"
        
        # Create full path
        export_path = str(dir_path / filename)
        self.logger.debug(f"Generated export path: {export_path}")
        
        # Check data structure
        if not isinstance(data, dict) or "results" not in data:
            error_msg = "Invalid data structure. Expected dictionary with 'results' key"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.debug(f"Exporting {len(data['results'])} top-level noun phrases")
        
        # Export based on format
        try:
            if format == "txt":
                return self._export_txt(data, export_path)
            elif format == "csv":
                return self._export_csv(data, export_path)
            elif format == "json":
                return self._export_json(data, export_path)
        except Exception as e:
            self.logger.error(f"Error exporting to {format}: {str(e)}")
            raise
    
    def _export_txt(self, data, export_path):
        """Export to text file with hierarchical structure."""
        self.logger.debug(f"Exporting to text file: {export_path}")
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(f"ANPE Noun Phrase Extraction Results\n")
                f.write(f"Timestamp: {data['metadata'].get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}\n")
                f.write(f"Includes Nested NPs: {data['metadata'].get('includes_nested', False)}\n")
                f.write(f"Includes Metadata: {data['metadata'].get('includes_metadata', False)}\n\n")
                
                # Export each top-level NP and its nested NPs
                for np_item in data['results']:
                    self._write_np_to_txt(f, np_item, level=0)
            
            self.logger.info(f"Successfully exported to text file: {export_path}")
            return export_path
            
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
    
    def _export_csv(self, data, export_path):
        """Export to CSV file with flattened hierarchy."""
        self.logger.debug(f"Exporting to CSV file: {export_path}")
        
        try:
            # Flatten the hierarchical structure
            flattened_nps = []
            for np_item in data['results']:
                self._flatten_np_hierarchy(np_item, flattened_list=flattened_nps)
            
            # Determine which columns to include
            includes_metadata = data['metadata'].get('includes_metadata', False)
            
            with open(export_path, 'w', newline='', encoding='utf-8') as f:
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
            
            self.logger.info(f"Successfully exported to CSV file: {export_path}")
            return export_path
            
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
    
    def _export_json(self, data, export_path):
        """Export to JSON file with hierarchical structure."""
        self.logger.debug(f"Exporting to JSON file: {export_path}")
        
        try:
            # Clean up non-JSON-serializable objects if any
            cleaned_data = self._clean_for_json(data)
            
            # Write properly formatted JSON
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_data, f, indent=2)
            
            self.logger.info(f"Successfully exported to JSON file: {export_path}")
            return export_path
            
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