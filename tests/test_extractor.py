import unittest
import tempfile
from pathlib import Path
import os
import json
import csv
import time
from anpe import ANPEExtractor

class TestANPEExtractor(unittest.TestCase):
    """Test cases for ANPEExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ANPEExtractor()
        self.test_text = "The team of scientists published their exciting research on climate change."
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that the extractor initializes correctly."""
        self.assertIsInstance(self.extractor, ANPEExtractor)
    
    def test_basic_extraction(self):
        """Test basic noun phrase extraction."""
        result = self.extractor.extract(self.test_text)
        self.assertIn("results", result)
        self.assertIsInstance(result["results"], list)
        
        # Expected top-level NPs in our test text:
        # - "The team of scientists"
        # - "their exciting research on climate change"
        # Note: "climate change" is a nested NP, not a top-level one
        self.assertEqual(len(result["results"]), 2)
        
        # Verify these expected noun phrases are found
        expected_nps = [
            "The team of scientists",
            "their exciting research on climate change"
        ]
        found_nps = [np["noun_phrase"].lower() for np in result["results"]]
        for expected_np in expected_nps:
            self.assertTrue(any(expected_np.lower() == found_np for found_np in found_nps),
                           f"Expected NP '{expected_np}' not found in results")
    
    def test_extraction_with_metadata(self):
        """Test extraction with metadata."""
        result = self.extractor.extract(self.test_text, metadata=True)
        self.assertIn("results", result)
        for np in result["results"]:
            self.assertIn("metadata", np)
            self.assertIn("length", np["metadata"])
            self.assertIn("structures", np["metadata"])
            
            # Verify specific structures for each noun phrase
            if np["noun_phrase"].lower() == "the team of scientists":
                self.assertIn("determiner", np["metadata"]["structures"])
                self.assertIn("prepositional_modifier", np["metadata"]["structures"])
            elif np["noun_phrase"].lower() == "their exciting research on climate change":
                self.assertIn("adjectival_modifier", np["metadata"]["structures"])
                self.assertIn("prepositional_modifier", np["metadata"]["structures"])
                self.assertIn("possessive", np["metadata"]["structures"])
                self.assertIn("compound", np["metadata"]["structures"])
    
    def test_extraction_with_nested(self):
        """Test extraction with nested noun phrases."""
        result = self.extractor.extract(self.test_text, include_nested=True)
        self.assertIn("results", result)
        
        # In our sentence, "climate change" should be a child of 
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
                       "Expected parent-child relationship not found")
    
    def test_extraction_with_config(self):
        """Test extraction with custom configuration."""
        custom_extractor = ANPEExtractor({
            "min_length": 2,
            "max_length": 4
        })
        result = custom_extractor.extract(self.test_text)
        
        # With min_length=2 and max_length=4, we should only get phrases with 2-4 words
        for np in result["results"]:
            tokens = np["noun_phrase"].split()
            self.assertGreaterEqual(len(tokens), 2)
            self.assertLessEqual(len(tokens), 4)
            
        # "The team of scientists" should be included (4 words)
        # "their exciting research on climate change" should be excluded (6 words)
        found_nps = [np["noun_phrase"].lower() for np in result["results"]]
        self.assertTrue(any("the team of scientists" in found_np for found_np in found_nps),
                        "Expected NP 'The team of scientists' not found")
        
        # Make sure the longer phrase is not included
        self.assertFalse(any("their exciting research on climate change" in found_np 
                              for found_np in found_nps),
                         "Phrase 'their exciting research on climate change' should be excluded by max_length")

    # --- Tests for Export Functionality --- 

    def test_export_to_directory(self):
        """Test exporting to a directory (generates timestamped filename)."""
        # Test for each format
        for fmt in ["txt", "csv", "json"]:
            # Export providing the directory path
            exported_file_path_str = self.extractor.export(
                self.test_text, 
                output=str(self.output_dir),
                format=fmt,
                metadata=True,
                include_nested=True
            )
            exported_file_path = Path(exported_file_path_str)

            # Verify the returned path is within the directory and has the correct extension
            self.assertEqual(exported_file_path.parent, self.output_dir.resolve())
            self.assertTrue(exported_file_path.name.startswith("anpe_export_"))
            self.assertTrue(exported_file_path.name.endswith(f".{fmt}"))
            self.assertTrue(exported_file_path.exists())
            self.assertTrue(exported_file_path.is_file())

            # Basic content check (e.g., non-empty for txt)
            if fmt == "txt":
                self.assertGreater(exported_file_path.stat().st_size, 0)

    def test_export_to_specific_file(self):
        """Test exporting to a specific file path."""
        output_filepath = self.output_dir / "my_specific_results.json"
        exported_file_path_str = self.extractor.export(
            self.test_text,
            output=str(output_filepath),
            format="json", # Format matches extension
            metadata=True,
            include_nested=True
        )
        exported_file_path = Path(exported_file_path_str)

        # Verify the exact file was created and returned
        self.assertEqual(exported_file_path, output_filepath.resolve())
        self.assertTrue(exported_file_path.exists())
        
        # Verify content
        with open(exported_file_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        self.assertIn("results", result)
        self.assertTrue(len(result["results"]) > 0)

    def test_export_creates_parent_directory(self):
        """Test that export creates non-existent parent directories."""
        nested_output_filepath = self.output_dir / "new_subdir" / "nested_file.txt"
        
        # Ensure parent does not exist initially
        self.assertFalse(nested_output_filepath.parent.exists())

        exported_file_path_str = self.extractor.export(
            self.test_text,
            output=str(nested_output_filepath),
            format="txt"
        )
        exported_file_path = Path(exported_file_path_str)

        # Verify the file was created in the new subdirectory
        self.assertEqual(exported_file_path, nested_output_filepath.resolve())
        self.assertTrue(exported_file_path.parent.exists())
        self.assertTrue(exported_file_path.parent.is_dir())
        self.assertTrue(exported_file_path.exists())
        self.assertGreater(exported_file_path.stat().st_size, 0)
        
    def test_export_extension_mismatch(self):
        """Test export behavior when file extension mismatches format."""
        output_filepath = self.output_dir / "mismatch_test.txt" # .txt extension
        exported_file_path_str = self.extractor.export(
            self.test_text,
            output=str(output_filepath),
            format="json", # but format is json
            metadata=True,
            include_nested=True
        )
        exported_file_path = Path(exported_file_path_str)
        
        # Verify the exact file (with .txt extension) was created
        self.assertEqual(exported_file_path, output_filepath.resolve())
        self.assertTrue(exported_file_path.exists())

        # Verify the content is JSON despite the .txt extension
        try:
            with open(exported_file_path, 'r', encoding='utf-8') as f:
                result = json.load(f)
            self.assertIn("metadata", result)
            self.assertIn("results", result)
            self.assertTrue(len(result["results"]) > 0)
        except json.JSONDecodeError:
            self.fail("File mismatch_test.txt did not contain valid JSON content.")

    def test_export_unsupported_extension(self):
        """Test export behavior with unsupported file extension."""
        # Create a path with an unsupported extension (.xlsx)
        unsupported_path = self.output_dir / "data.xlsx"
        
        exported_file_path_str = self.extractor.export(
            self.test_text,
            output=str(unsupported_path),
            format="json",
            metadata=True
        )
        exported_file_path = Path(exported_file_path_str)
        
        # Verify that the file was created in the same directory but with a timestamped name
        self.assertEqual(exported_file_path.parent, unsupported_path.parent)
        self.assertTrue(exported_file_path.name.startswith("anpe_export_"))
        self.assertTrue(exported_file_path.name.endswith(".json"))  # Should have the format extension
        self.assertNotEqual(exported_file_path.name, "data.xlsx")   # Should not be the original name
        
        # Verify the file exists and contains valid JSON
        self.assertTrue(exported_file_path.exists())
        with open(exported_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertIn("results", data)
            self.assertTrue(len(data["results"]) > 0)

if __name__ == "__main__":
    unittest.main() 