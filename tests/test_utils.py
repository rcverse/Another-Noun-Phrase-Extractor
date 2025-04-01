import unittest
import os
import tempfile
import json
import csv
from pathlib import Path

from anpe.utils.export import ANPEExporter
from anpe.utils.logging import ANPELogger, get_logger
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
        self.exporter.export(self.sample_result, format="json", export_dir=self.export_dir)
        
        # Check if file was created
        output_files = list(self.export_dir.glob("*.json"))
        self.assertEqual(len(output_files), 1)
        
        # Verify content
        with open(output_files[0], 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["noun_phrase"], "The team of scientists")
    
    def test_export_csv(self):
        """Test CSV export."""
        self.exporter.export(self.sample_result, format="csv", export_dir=self.export_dir)
        
        # Check if file was created
        output_files = list(self.export_dir.glob("*.csv"))
        self.assertEqual(len(output_files), 1)
        
        # Verify content
        with open(output_files[0], 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Header + 3 data rows (for top-level NPs and nested child)
        self.assertEqual(len(rows), 4)
        # Check headers (column names may be capitalized)
        self.assertIn("Noun_Phrase", rows[0])
        self.assertIn("ID", rows[0])
        # Check data
        found_team = False
        for row in rows[1:]:  # Skip header row
            if "The team of scientists" in str(row):
                found_team = True
                break
        self.assertTrue(found_team, "Expected data 'The team of scientists' not found in CSV")

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