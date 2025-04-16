import unittest
import os
import tempfile
import json
import csv
from pathlib import Path

from anpe.utils.export import ANPEExporter
from anpe.utils.anpe_logger import ANPELogger, get_logger
from anpe.utils.analyzer import ANPEAnalyzer

# Import functions to test
from anpe.utils.model_finder import select_best_spacy_model, select_best_benepar_model

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

class TestModelFinder(unittest.TestCase):
    """Test cases for model finding and selection utilities."""

    # --- Test select_best_spacy_model --- 

    def test_select_spacy_prioritizes_default_md(self):
        """Test that 'md' is selected if present, regardless of preference list."""
        installed = ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]
        selected = select_best_spacy_model(installed)
        self.assertEqual(selected, "en_core_web_md")

    def test_select_spacy_falls_back_to_preference_if_md_missing(self):
        """Test that the preference list is used when 'md' is not installed."""
        installed = ["en_core_web_sm", "en_core_web_lg", "en_core_web_trf"]
        # Expect 'trf' based on preference list, as 'md' is missing
        selected = select_best_spacy_model(installed)
        self.assertEqual(selected, "en_core_web_trf")

    def test_select_spacy_falls_back_to_lower_preference(self):
        """Test selection works correctly when only lower preference models are installed."""
        installed = ["en_core_web_sm"] # 'md', 'lg', 'trf' are missing
        selected = select_best_spacy_model(installed)
        self.assertEqual(selected, "en_core_web_sm")

    def test_select_spacy_falls_back_to_first_if_non_preferred(self):
        """Test that the first installed model is returned if none are in preference list."""
        installed = ["some_other_model", "en_core_web_sm"] 
        # Assume "some_other_model" is not in SPACY_PREFERENCE
        # Default 'md' is also missing. 
        # It should select 'sm' as it's the first preferred one found
        selected = select_best_spacy_model(installed)
        self.assertEqual(selected, "en_core_web_sm")
        
        installed_only_unknown = ["some_other_model", "another_unknown_model"]
        # Default 'md' missing, none from preference list are present.
        # Should fall back to the *first* one in the list.
        selected_unknown = select_best_spacy_model(installed_only_unknown)
        self.assertEqual(selected_unknown, "some_other_model")

    def test_select_spacy_returns_none_for_empty_list(self):
        """Test that None is returned if the list of installed models is empty."""
        installed = []
        selected = select_best_spacy_model(installed)
        self.assertIsNone(selected)

    # --- Test select_best_benepar_model --- 

    def test_select_benepar_prioritizes_default_en3(self):
        """Test that 'benepar_en3' is selected if present."""
        installed = ["benepar_en3", "benepar_en3_large"]
        selected = select_best_benepar_model(installed)
        self.assertEqual(selected, "benepar_en3")

    def test_select_benepar_falls_back_to_preference_if_en3_missing(self):
        """Test that 'large' is selected when 'en3' is not installed."""
        installed = ["benepar_en3_large"] # Default 'en3' is missing
        selected = select_best_benepar_model(installed)
        self.assertEqual(selected, "benepar_en3_large")

    def test_select_benepar_falls_back_to_first_if_non_preferred(self):
        """Test that the first installed model is returned if none are in preference list."""
        installed = ["some_other_benepar", "benepar_en3_large"] 
        # Assume "some_other_benepar" is not in BENEPAR_PREFERENCE
        # Default 'en3' is also missing.
        # It should select 'large' as it's the first preferred one found
        selected = select_best_benepar_model(installed)
        self.assertEqual(selected, "benepar_en3_large")

        installed_only_unknown = ["some_other_benepar", "another_unknown_benepar"]
        # Default 'en3' missing, none from preference list are present.
        # Should fall back to the *first* one in the list.
        selected_unknown = select_best_benepar_model(installed_only_unknown)
        self.assertEqual(selected_unknown, "some_other_benepar")

    def test_select_benepar_returns_none_for_empty_list(self):
        """Test that None is returned if the list of installed models is empty."""
        installed = []
        selected = select_best_benepar_model(installed)
        self.assertIsNone(selected)

if __name__ == "__main__":
    unittest.main() 