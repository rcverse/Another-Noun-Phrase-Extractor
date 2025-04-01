import unittest
import tempfile
import os
import json
from pathlib import Path
from anpe.cli import main, parse_args

class TestCLI(unittest.TestCase):
    """Test cases for CLI functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_text = "The team of scientists published their exciting research on climate change."
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = Path(self.temp_dir.name) / "input.txt"
        self.output_dir = Path(self.temp_dir.name) / "output"
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
        self.assertEqual(args.output_dir, str(self.output_dir))
        self.assertEqual(args.type, "json")
    
    def test_version_command(self):
        """Test version command."""
        args = parse_args(["version"])
        self.assertEqual(args.command, "version")
    
    def test_extract_from_file(self):
        """Test extraction from file."""
        # This is a simplified test as we can't easily capture stdout
        args = ["extract", "-f", str(self.input_file), "-o", str(self.output_dir), "-t", "json", "--metadata"]
        exit_code = main(args)
        self.assertEqual(exit_code, 0)
        
        # Check if output file was created
        output_files = list(self.output_dir.glob("*.json"))
        self.assertGreater(len(output_files), 0)
        
        # Verify content of the first output file
        with open(output_files[0], 'r', encoding='utf-8') as f:
            result = json.load(f)
        self.assertIn("results", result)
        self.assertGreater(len(result["results"]), 0)
        
        # Expected NPs in our test text: 
        # - "The team of scientists"
        # - "their exciting research on climate change"
        # - "climate change"
        expected_nps = [
            "The team of scientists",
            "their exciting research on climate change",
            "climate change"
        ]
        
        # Check that expected NPs are found (using case-insensitive comparison)
        found_nps = [np["noun_phrase"].lower() for np in result["results"]]
        for np in expected_nps:
            self.assertTrue(any(np.lower() in found_np for found_np in found_nps),
                            f"Expected NP '{np}' not found in extracted NPs")
        
        # Check metadata exists
        for np in result["results"]:
            self.assertIn("metadata", np)
    
    def test_extract_with_nested(self):
        """Test extraction with nested argument."""
        args = ["extract", "-f", str(self.input_file), "-o", str(self.output_dir), 
                "-t", "json", "--nested"]
        exit_code = main(args)
        self.assertEqual(exit_code, 0)
        
        # Check if output file was created
        output_files = list(self.output_dir.glob("*.json"))
        self.assertGreater(len(output_files), 0)
        
        # Verify nested structure exists
        with open(output_files[0], 'r', encoding='utf-8') as f:
            result = json.load(f)
        self.assertIn("results", result)
        
        # For our test sentence, "climate change" should be a child of 
        # "their exciting research on climate change"
        found_parent_child = False
        for np in result["results"]:
            self.assertIn("children", np)
            if np["noun_phrase"].lower() == "their exciting research on climate change":
                for child in np["children"]:
                    if child["noun_phrase"].lower() == "climate change":
                        found_parent_child = True
                        break
        
        # The parent-child relationship should exist
        self.assertTrue(found_parent_child, 
                        "Expected parent-child relationship was not found")

if __name__ == "__main__":
    unittest.main() 