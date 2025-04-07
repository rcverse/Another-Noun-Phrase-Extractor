import unittest
import tempfile
import os
import json
from pathlib import Path
from anpe.cli import main, parse_args
import csv

class TestCLI(unittest.TestCase):
    """Test cases for CLI functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_text = "The team of scientists published their exciting research on climate change."
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = Path(self.temp_dir.name) / "input.txt"
        self.output_dir = Path(self.temp_dir.name) / "output_cli"
        self.output_dir.mkdir(exist_ok=True)
        
        # Create a test input file
        with open(self.input_file, 'w', encoding='utf-8') as f:
            f.write(self.test_text)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_parse_args(self):
        """Test argument parsing."""
        args = parse_args(["extract", "-f", str(self.input_file), "-o", str(self.output_dir), "-t", "json"])
        self.assertEqual(args.command, "extract")
        self.assertEqual(args.file, str(self.input_file))
        self.assertEqual(args.output, str(self.output_dir))
        self.assertEqual(args.type, "json")
    
    def test_version_command(self):
        """Test version command."""
        args = parse_args(["version"])
        self.assertEqual(args.command, "version")
    
    def test_extract_from_file_to_dir(self):
        """Test extraction from file, outputting to a directory."""
        args = ["extract", "-f", str(self.input_file), "-o", str(self.output_dir), "-t", "json", "--metadata"]
        exit_code = main(args)
        self.assertEqual(exit_code, 0)
        
        # Check if output file was created in the directory
        output_files = list(self.output_dir.glob("anpe_export_*.json"))
        self.assertGreater(len(output_files), 0, "No JSON file created in output directory")
        
        # Verify content of the first output file
        with open(output_files[0], 'r', encoding='utf-8') as f:
            result = json.load(f)
        self.assertIn("results", result)
        self.assertGreater(len(result["results"]), 0)
        self.assertIn("metadata", result["results"][0])
        
    def test_extract_to_specific_file(self):
        """Test extraction outputting to a specific file path."""
        specific_output_file = self.output_dir / "specific_cli_output.csv"
        args = ["extract", "-f", str(self.input_file), "-o", str(specific_output_file), "-t", "csv"]
        exit_code = main(args)
        self.assertEqual(exit_code, 0)

        # Check that the specific file exists
        self.assertTrue(specific_output_file.exists(), f"Specific output file {specific_output_file} was not created.")
        self.assertTrue(specific_output_file.is_file())

        # Basic content check (e.g., is it a valid CSV?)
        with open(specific_output_file, 'r', encoding='utf-8', newline='') as f:
            try:
                reader = csv.reader(f)
                header = next(reader) # Read header
                self.assertIsNotNone(header)
                # Could add more specific content checks if needed
            except Exception as e:
                self.fail(f"Failed to read CSV content from {specific_output_file}: {e}")

    def test_extract_with_nested_to_dir(self):
        """Test extraction with nested argument, outputting to a directory."""
        args = ["extract", "-f", str(self.input_file), "-o", str(self.output_dir), 
                "-t", "json", "--nested"]
        exit_code = main(args)
        self.assertEqual(exit_code, 0)
        
        # Check if output file was created
        output_files = list(self.output_dir.glob("anpe_export_*.json"))
        self.assertGreater(len(output_files), 0, "No JSON file created in output directory for nested test")
        
        # Verify nested structure exists in the first file found
        with open(output_files[0], 'r', encoding='utf-8') as f:
            result = json.load(f)
        self.assertIn("results", result)
        
        found_parent_child = False
        for np in result["results"]:
            self.assertIn("children", np)
            if "their exciting research on climate change" in np["noun_phrase"].lower():
                for child in np["children"]:
                    if child["noun_phrase"].lower() == "climate change":
                        found_parent_child = True
                        break
            if found_parent_child:
                break
        
        self.assertTrue(found_parent_child, 
                        "Expected parent-child relationship was not found in CLI output")

    def test_extract_with_unsupported_extension(self):
        """Test CLI handling of unsupported file extensions."""
        # Create a path with an unsupported extension (.xlsx)
        unsupported_path = self.output_dir / "result_data.xlsx"
        
        args = ["extract", "-f", str(self.input_file), "-o", str(unsupported_path), "-t", "json"]
        exit_code = main(args)
        self.assertEqual(exit_code, 0)
        
        # The CLI should still succeed but create a file in the parent directory with a timestamped name
        output_files = list(self.output_dir.glob("anpe_export_*.json"))
        self.assertGreater(len(output_files), 0, "No JSON file created for unsupported extension test")
        
        # Verify one of the output files contains valid JSON with our expected data
        found_valid_content = False
        for output_file in output_files:
            with open(output_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if "results" in data and len(data["results"]) > 0:
                        # Check if data contains our expected content
                        for np in data["results"]:
                            if "team of scientists" in np["noun_phrase"].lower():
                                found_valid_content = True
                                break
                except json.JSONDecodeError:
                    continue
            if found_valid_content:
                break
                
        self.assertTrue(found_valid_content, "No output file contained the expected extraction results")

if __name__ == "__main__":
    unittest.main() 