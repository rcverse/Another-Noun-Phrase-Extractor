import unittest
import os
import tempfile
import json
import csv
from pathlib import Path

from anpe.utils.export import ANPEExporter
from anpe.utils.anpe_logger import ANPELogger, get_logger
from anpe.utils.analyzer import ANPEAnalyzer

class TestExporter(unittest.TestCase):
    """Test cases for ANPEExporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exporter = ANPEExporter()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.export_dir = Path(self.temp_dir.name)
        
        # Sample extraction result using our test text
        self.sample_result = {
            "metadata": {
                "timestamp": "2023-04-01T12:00:00",
                "includes_nested": True,
                "includes_metadata": True
            },
            "results": [
                {
                    "id": "1",
                    "noun_phrase": "The team of scientists",
                    "level": 1,
                    "metadata": {
                        "length": 4,
                        "structures": ["determiner", "prepositional_modifier"]
                    },
                    "children": []
                },
                {
                    "id": "2",
                    "noun_phrase": "their exciting research on climate change",
                    "level": 1,
                    "metadata": {
                        "length": 6,
                        "structures": ["adjectival_modifier", "prepositional_modifier", "possessive", "compound"]
                    },
                    "children": [
                        {
                            "id": "3",
                            "noun_phrase": "climate change",
                            "level": 2,
                            "metadata": {
                                "length": 2,
                                "structures": ["compound"]
                            },
                            "children": []
                        }
                    ]
                }
            ]
        }
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_export_json(self):
        """Test JSON export."""
        output_filepath = self.export_dir / "test_output.json"
        self.exporter.export(self.sample_result, format="json", output_filepath=str(output_filepath))
        
        # Check if the specific file was created
        self.assertTrue(output_filepath.exists())
        self.assertTrue(output_filepath.is_file())
        
        # Verify content
        with open(output_filepath, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["noun_phrase"], "The team of scientists")
    
    def test_export_csv(self):
        """Test CSV export."""
        output_filepath = self.export_dir / "test_output.csv"
        self.exporter.export(self.sample_result, format="csv", output_filepath=str(output_filepath))
        
        # Check if the specific file was created
        self.assertTrue(output_filepath.exists())
        self.assertTrue(output_filepath.is_file())
        
        # Verify content
        with open(output_filepath, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Header + 3 data rows (for top-level NPs and nested child)
        self.assertEqual(len(rows), 4)
        # Check headers
        self.assertEqual(rows[0], ["ID", "Level", "Parent_ID", "Noun_Phrase", "Length", "Structures"])
        # Check data
        self.assertEqual(rows[1], ["1", "1", "", "The team of scientists", "4", "determiner|prepositional_modifier"])
        self.assertEqual(rows[2], ["2", "1", "", "their exciting research on climate change", "6", "adjectival_modifier|prepositional_modifier|possessive|compound"])
        self.assertEqual(rows[3], ["3", "2", "2", "climate change", "2", "compound"])

    def test_export_txt(self):
        """Test TXT export."""
        output_filepath = self.export_dir / "test_output.txt"
        self.exporter.export(self.sample_result, format="txt", output_filepath=str(output_filepath))

        # Check if the specific file was created
        self.assertTrue(output_filepath.exists())
        self.assertTrue(output_filepath.is_file())

        # Verify content (check for specific lines)
        with open(output_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("ANPE Noun Phrase Extraction Results", content)
        self.assertIn("• [1] The team of scientists", content)
        self.assertIn("  Length: 4", content)
        self.assertIn("  Structures: [determiner, prepositional_modifier]", content)
        self.assertIn("• [2] their exciting research on climate change", content)
        self.assertIn("  ◦ [3] climate change", content)

class TestLogger(unittest.TestCase):
    """Test cases for ANPELogger."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_file = Path(self.temp_dir.name) / "test.log"
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.temp_dir.cleanup()
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        # Initialize logger without a log file for testing
        logger = ANPELogger(log_level="DEBUG")
        self.assertIsNotNone(logger)
    
    def test_get_logger(self):
        """Test get_logger function."""
        ANPELogger(log_level="DEBUG")
        logger = get_logger("test")
        self.assertIsNotNone(logger)
        
        # Test logging
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")

class TestAnalyzer(unittest.TestCase):
    """Test cases for ANPEAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = ANPEAnalyzer()
    
    def test_analyze_single_np(self):
        """Test analyzing a single noun phrase."""
        # Test with determiner structure
        structures = self.analyzer.analyze_single_np("the team")
        self.assertIn("determiner", structures)
        
        # Test with adjectival structure
        structures = self.analyzer.analyze_single_np("the exciting research")
        self.assertIn("adjectival_modifier", structures)
        
        # Test with compound structure
        structures = self.analyzer.analyze_single_np("climate change")
        self.assertIn("compound", structures)
        
        # Test with prepositional structure
        structures = self.analyzer.analyze_single_np("team of scientists")
        self.assertIn("prepositional_modifier", structures)
        
        # Test with combined structures
        structures = self.analyzer.analyze_single_np("the team of scientists")
        self.assertIn("determiner", structures)
        self.assertIn("prepositional_modifier", structures)

if __name__ == "__main__":
    unittest.main() 