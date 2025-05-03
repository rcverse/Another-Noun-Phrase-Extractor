import unittest
import os
import tempfile
import json
import csv
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import nltk
import spacy
import subprocess # Import subprocess for mocking
import logging
import io # Import io for StringIO

from anpe.utils.export import ANPEExporter
from anpe.utils.anpe_logger import ANPELogger, get_logger
from anpe.utils.analyzer import ANPEAnalyzer

# Import functions to test
from anpe.utils.model_finder import (
    select_best_spacy_model, 
    select_best_benepar_model, 
    find_installed_spacy_models,
    find_installed_benepar_models
)
from anpe.utils.setup_models import (
    check_spacy_model, 
    check_benepar_model,
    check_all_models_present,
    SPACY_MODEL_MAP,
    BENEPAR_MODEL_MAP,
    install_spacy_model,
    install_benepar_model,
    setup_models
)

class TestExporter(unittest.TestCase):
    """Test cases for ANPEExporter."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exporter = ANPEExporter()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.export_dir = Path(self.temp_dir.name)
        
        # Sample extraction result using the NEW structure
        self.sample_result = {
            "timestamp": "2023-04-01T12:00:00", # Moved to top level
            "configuration": {
                "min_length": None, # Example config fields
                "max_length": None,
                "accept_pronouns": True,
                "structure_filters": [],
                "newline_breaks": True,
                "spacy_model_used": "en_core_web_md", # Example
                "benepar_model_used": "benepar_en3", # Example
                "metadata_requested": True, # Flag for output content
                "nested_requested": True    # Flag for output content
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

        # Verify content (check for specific lines based on NEW header format)
        with open(output_filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check header lines
        self.assertIn("--- ANPE Noun Phrase Extraction Results ---", content)
        self.assertIn("Timestamp: 2023-04-01T12:00:00", content)
        
        # Check configuration section
        self.assertIn("--- Configuration Used ---", content)
        self.assertIn("Output includes Nested NPs: True", content)
        self.assertIn("Output includes Metadata: True", content)
        self.assertIn("Spacy Model Used: en_core_web_md", content)
        self.assertIn("Accept Pronouns: True", content)
        self.assertIn("Structure Filters: None", content) # Assuming empty list formats as None
        self.assertIn("--------------------------", content)
        
        # Check results separator and content
        self.assertIn("--- Extraction Results ---", content)
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
        self.log_file_path = Path(self.temp_dir.name) / "test.log"
        # Ensure clean slate for handlers before each test
        self._cleanup_handlers()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self._cleanup_handlers()
        self.temp_dir.cleanup()

    def _cleanup_handlers(self):
        """Close and remove handlers to prevent file locking."""
        loggers_to_check = [
            logging.getLogger("init_test"), 
            logging.getLogger("get_test"), 
            logging.getLogger("level_test"), 
            logging.getLogger("file_test"),
            logging.getLogger("anpe"), # Check logger used by ANPELogger itself
            logging.getLogger() # Root logger
        ]
        for logger in loggers_to_check:
            if hasattr(logger, 'handlers'):
                for handler in logger.handlers[:]: # Iterate over a copy
                    if isinstance(handler, logging.FileHandler):
                        handler.flush()
                        handler.close()
                    # Remove all handlers associated with this logger
                    logger.removeHandler(handler)
            # Reset level just in case
            logger.setLevel(logging.NOTSET)
            
    def test_logger_initialization(self):
        """Test logger initialization."""
        # Initialize logger without a log file for testing
        logger_instance = ANPELogger(log_level="DEBUG")
        self.assertIsNotNone(logger_instance)
        # Check that get_logger retrieves a logger
        logger = get_logger("init_test")
        self.assertIsNotNone(logger)
        # Check the *effective* level, which considers root logger settings
        self.assertEqual(logger.getEffectiveLevel(), logging.DEBUG)
    
    def test_get_logger(self):
        """Test get_logger function retrieves configured logger."""
        ANPELogger(log_level="WARNING") # Set level to WARNING
        logger = get_logger("get_test")
        self.assertIsNotNone(logger)
        # Check the *effective* level
        self.assertEqual(logger.getEffectiveLevel(), logging.WARNING)

    def test_log_level_filtering(self):
        """Test that messages below the set log level are filtered."""
        # Initialize ANPELogger - its configuration might affect the root logger
        ANPELogger(log_level="INFO") 
        
        logger_name = "level_test_manual"
        logger = get_logger(logger_name)
        
        # Ensure the logger itself is set to process INFO level
        logger.setLevel(logging.INFO) 
        # Prevent propagation to avoid interference from root logger handlers if any
        logger.propagate = False 

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        # Set a specific format for easy checking (optional)
        formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
        handler.setFormatter(formatter)
        # Handler level also needs to be appropriate
        handler.setLevel(logging.INFO)
        
        logger.addHandler(handler)
        
        try:
            logger.debug("This should be filtered.")
            logger.info("This should be logged.")
            logger.warning("This should also be logged.")
            
            log_output = log_stream.getvalue()
            
            # Check the captured output
            output_lines = log_output.strip().split('\n')
            self.assertEqual(len(output_lines), 2, f"Expected 2 lines, got: {output_lines}")
            self.assertIn("INFO:anpe.level_test_manual:This should be logged.", output_lines[0])
            self.assertIn("WARNING:anpe.level_test_manual:This should also be logged.", output_lines[1])
            
        finally:
            # Clean up the handler
            logger.removeHandler(handler)
            handler.close()
            # Reset propagation if needed for other tests
            logger.propagate = True 
            # Reset level
            logger.setLevel(logging.NOTSET)

    def test_log_file_creation_and_content(self):
        """Test that specifying a log file path creates the file and logs to it."""
        # Initialize logger with file path and DEBUG level for visibility
        ANPELogger(log_level="DEBUG", log_file=str(self.log_file_path))
        logger = get_logger("file_test")
        
        # Log some messages
        message1 = "Testing file logging."
        message2 = "Another debug message."
        logger.info(message1)
        logger.debug(message2)

        # Flush logs (optional, but good practice before file check)
        # Find *any* handler associated with this logger or parents and flush
        curr_logger = logger
        while curr_logger:
            for handler in curr_logger.handlers:
                 handler.flush()
            if not curr_logger.propagate:
                 break
            curr_logger = curr_logger.parent
        # Also flush root handlers just in case
        for handler in logging.getLogger().handlers:
            handler.flush()
            
        # No need to close/remove handlers here, _cleanup_handlers in tearDown will handle it.

        # Check if the log file was created
        self.assertTrue(self.log_file_path.exists(), "Log file was not created.")
        self.assertTrue(self.log_file_path.is_file())

        # Check if the log file contains the logged messages
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
            
        self.assertIn(message1, log_content)
        self.assertIn(message2, log_content)
        # Remove brittle regex checks
        # self.assertRegex(log_content, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - file_test - Testing file logging.")
        # self.assertRegex(log_content, r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - DEBUG - file_test - Another debug message.")

class TestAnalyzer(unittest.TestCase):
    """Test cases for ANPEAnalyzer."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock nlp object needed for ANPEAnalyzer init
        self.mock_nlp = MagicMock()
        # Configure the mock doc to have basic functionality if needed by tests
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1 # Example: pretend doc has length 1
        mock_doc.__getitem__.return_value = MagicMock() # Allow indexing
        self.mock_nlp.return_value = mock_doc # Make nlp(text) return the mock doc
        
        self.analyzer = ANPEAnalyzer(nlp=self.mock_nlp)
    
    def test_analyze_single_np(self):
        """Test analyzing a single noun phrase."""
        # Since nlp is mocked, this test might need adjustment 
        # or more sophisticated mocking of the doc/token attributes 
        # depending on how _analyze_structure uses them.
        # For now, just assert it runs without error and returns a list.
        structures = self.analyzer.analyze_single_np("the team")
        self.assertIsInstance(structures, list)
        # To make this test meaningful, we'd need to mock the token properties 
        # (pos_, dep_, tag_, head) used by the _detect_* methods.
        # Example (if _detect_determiner_np needs pos_ == 'DET'):
        # token_mock = MagicMock(pos_='DET', dep_='det', head=MagicMock(pos_='NOUN'))
        # self.mock_nlp.return_value = [token_mock] # Mock doc is now iterable
        # structures = self.analyzer.analyze_single_np("the team")
        # self.assertIn("determiner", structures)

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

    # --- Test find_installed_spacy_models ---

    @patch('anpe.utils.model_finder.spacy.load')
    @patch('anpe.utils.model_finder.SPACY_MODEL_MAP', {
        'sm': 'en_core_web_sm', 
        'md': 'en_core_web_md', 
        'lg': 'en_core_web_lg', 
        'trf': 'en_core_web_trf'
    })
    def test_find_spacy_models_some_loadable(self, mock_spacy_load):
        """Test find_installed_spacy_models correctly identifies loadable models."""
        # Simulate spacy.load behavior:
        # - Success for 'sm' and 'lg'
        # - OSError for 'md'
        # - ImportError for 'trf'
        def load_side_effect(model_name):
            if model_name == 'en_core_web_sm':
                return MagicMock() # Simulate success
            elif model_name == 'en_core_web_lg':
                return MagicMock() # Simulate success
            elif model_name == 'en_core_web_md':
                raise OSError(f"Mock Error: Cannot load {model_name}")
            elif model_name == 'en_core_web_trf':
                 raise ImportError(f"Mock Error: Cannot import {model_name}")
            else:
                raise ValueError(f"Unexpected model name: {model_name}")

        mock_spacy_load.side_effect = load_side_effect

        expected_models = ['en_core_web_sm', 'en_core_web_lg']
        with self.assertLogs('anpe.utils.model_finder', level='DEBUG') as log: # Capture DEBUG logs
             result = find_installed_spacy_models()

        self.assertCountEqual(result, expected_models) # Check if the correct models are returned

        # Verify spacy.load was called for all known models
        expected_calls = [call('en_core_web_sm'), call('en_core_web_md'), call('en_core_web_lg'), call('en_core_web_trf')]
        mock_spacy_load.assert_has_calls(expected_calls, any_order=True)
        self.assertEqual(mock_spacy_load.call_count, 4)

        # Verify logging messages for skipped models
        self.assertTrue(any(f"model 'en_core_web_md' not found or could not be loaded" in msg for msg in log.output))
        self.assertTrue(any(f"model 'en_core_web_trf' not found or could not be loaded" in msg for msg in log.output))
        self.assertTrue(any(f"Successfully loaded and verified spaCy model: 'en_core_web_sm'" in msg for msg in log.output))

    @patch('anpe.utils.model_finder.spacy.load')
    @patch('anpe.utils.model_finder.SPACY_MODEL_MAP', {
        'sm': 'en_core_web_sm', 
        'md': 'en_core_web_md'
    })
    def test_find_spacy_models_none_loadable(self, mock_spacy_load):
        """Test find_installed_spacy_models returns [] and logs warning when none are loadable."""
        # Simulate spacy.load failing for all known models
        mock_spacy_load.side_effect = OSError("Mock Error: Cannot load any model")

        # Specify the logger name and level to capture logs correctly
        with self.assertLogs('anpe.utils.model_finder', level='WARNING') as log:
             result = find_installed_spacy_models()

        self.assertEqual(result, []) # Expect an empty list

        # Verify spacy.load was attempted for all models
        expected_calls = [call('en_core_web_sm'), call('en_core_web_md')]
        mock_spacy_load.assert_has_calls(expected_calls, any_order=True)

        # Check that the specific warning message for no loadable models was logged
        self.assertTrue(any("No loadable ANPE-relevant spaCy models found" in message for message in log.output))

    # --- Test find_installed_benepar_models ---

    @patch('anpe.utils.model_finder.nltk.data')
    @patch('anpe.utils.model_finder.os.path.isdir')
    @patch('anpe.utils.model_finder.os.path.isfile')
    @patch('anpe.utils.model_finder.BENEPAR_MODEL_MAP', {'default': 'benepar_en3', 'large': 'benepar_en3_large'})
    @patch('anpe.utils.model_finder.NLTK_DATA_DIR', '/fake/nltk_data') # Mock primary dir
    def test_find_benepar_models_none_installed(self, mock_isfile, mock_isdir, mock_nltk_data):
        """Test find_installed_benepar_models when no models are found."""
        fake_path_1 = '/fake/nltk_data'
        fake_path_2 = '/other/path'
        mock_nltk_data.path = [fake_path_1, fake_path_2] # Mock NLTK paths
        
        # Simulate parent 'models' dirs exist, but model dirs/zips don't
        models_dir_1 = os.path.join(fake_path_1, 'models')
        models_dir_2 = os.path.join(fake_path_2, 'models')
        def isdir_side_effect(path):
            return path in [models_dir_1, models_dir_2]
        mock_isdir.side_effect = isdir_side_effect
        mock_isfile.return_value = False # No zip files found
        
        self.assertEqual(find_installed_benepar_models(), [])
        
        # Check os.path.isdir was called for parent and model dirs
        self.assertTrue(mock_isdir.called)
        mock_isdir.assert_any_call(models_dir_1) # Check parent dir was checked
        mock_isdir.assert_any_call(os.path.join(models_dir_1, 'benepar_en3')) # Check model dir was checked
        mock_isdir.assert_any_call(models_dir_2) # Check other parent dir
        mock_isdir.assert_any_call(os.path.join(models_dir_2, 'benepar_en3')) # Check model in other parent
        # isfile should be called if model dir is not found
        mock_isfile.assert_any_call(os.path.join(models_dir_1, 'benepar_en3.zip'))

    @patch('anpe.utils.model_finder.nltk.data')
    @patch('anpe.utils.model_finder.os.path.isdir')
    @patch('anpe.utils.model_finder.os.path.isfile') # Mock isfile
    @patch('anpe.utils.model_finder.BENEPAR_MODEL_MAP', {'default': 'benepar_en3', 'large': 'benepar_en3_large'})
    @patch('anpe.utils.model_finder.NLTK_DATA_DIR', '/fake/nltk_data') 
    def test_find_benepar_models_dir_exists(self, mock_isfile, mock_isdir, mock_nltk_data):
        """Test find_installed_benepar_models when a model directory exists."""
        fake_path = '/fake/nltk_data'
        mock_nltk_data.path = [fake_path]
        models_dir = os.path.join(fake_path, 'models')
        target_model_dir = os.path.join(models_dir, 'benepar_en3_large')
        other_model_dir = os.path.join(models_dir, 'benepar_en3') # Path for the other model
        other_model_zip = other_model_dir + '.zip'
        
        # Simulate parent 'models' dir AND target model dir exist
        def isdir_side_effect(path):
            return path in [models_dir, target_model_dir]
        mock_isdir.side_effect = isdir_side_effect
        # Mock isfile return value (doesn't matter much here, but good practice)
        mock_isfile.return_value = False 

        self.assertEqual(find_installed_benepar_models(), ['benepar_en3_large'])
        
        # Use os.path.join for assertions
        mock_isdir.assert_any_call(models_dir) # Check parent dir
        mock_isdir.assert_any_call(other_model_dir) # Check other model dir (returned False)
        mock_isdir.assert_any_call(target_model_dir) # Check target model dir (returned True)
        # Verify isfile *was* called for the model whose dir was not found
        mock_isfile.assert_called_once_with(other_model_zip)

    @patch('anpe.utils.model_finder.nltk.data')
    @patch('anpe.utils.model_finder.os.path.isdir')
    @patch('anpe.utils.model_finder.os.path.isfile')
    @patch('anpe.utils.model_finder.BENEPAR_MODEL_MAP', {'default': 'benepar_en3', 'large': 'benepar_en3_large'})
    @patch('anpe.utils.model_finder.NLTK_DATA_DIR', '/fake/nltk_data') 
    def test_find_benepar_models_zip_exists(self, mock_isfile, mock_isdir, mock_nltk_data):
        """Test find_installed_benepar_models when only a model zip file exists."""
        fake_path = '/fake/nltk_data'
        mock_nltk_data.path = [fake_path]
        models_dir = os.path.join(fake_path, 'models')
        target_model_dir = os.path.join(models_dir, 'benepar_en3')
        target_zip = os.path.join(models_dir, 'benepar_en3.zip')
        
        # Simulate parent 'models' dir exists, but model dir does NOT
        def isdir_side_effect(path):
            return path == models_dir # Only parent dir exists
        mock_isdir.side_effect = isdir_side_effect
        
        # Simulate zip file exists
        def isfile_side_effect(path):
            return path == target_zip
        mock_isfile.side_effect = isfile_side_effect
        
        self.assertEqual(find_installed_benepar_models(), ['benepar_en3'])
        
        # Use os.path.join for assertions
        mock_isdir.assert_any_call(models_dir) # Check parent dir
        mock_isdir.assert_any_call(target_model_dir) # Check model dir (returned False)
        mock_isfile.assert_any_call(target_zip) # Check zip file (returned True)

    @patch('anpe.utils.model_finder.nltk.data')
    @patch('anpe.utils.model_finder.os.path.isdir')
    @patch('anpe.utils.model_finder.os.path.isfile')
    @patch('anpe.utils.model_finder.BENEPAR_MODEL_MAP', {'default': 'benepar_en3', 'large': 'benepar_en3_large'})
    @patch('anpe.utils.model_finder.NLTK_DATA_DIR', '/user/nltk_data') # Set a specific primary dir
    def test_find_benepar_models_multiple_paths_and_duplicates(self, mock_isfile, mock_isdir, mock_nltk_data):
        """Test finding models across multiple paths and handling duplicates."""
        user_path = '/user/nltk_data'
        system_path = '/system/nltk_data'
        mock_nltk_data.path = [user_path, system_path, user_path] # Include duplicate user path
        
        user_models_dir = os.path.join(user_path, 'models')
        system_models_dir = os.path.join(system_path, 'models')
        system_en3_path = os.path.join(system_models_dir, 'benepar_en3')
        user_large_path = os.path.join(user_models_dir, 'benepar_en3_large')
        
        # Simulate relevant parent and model dirs exist
        def isdir_side_effect(path):
            return path in [user_models_dir, system_models_dir, system_en3_path, user_large_path]
        mock_isdir.side_effect = isdir_side_effect
        mock_isfile.return_value = False

        expected = ['benepar_en3_large', 'benepar_en3'] # Order depends on path checking order
        result = find_installed_benepar_models()
        self.assertCountEqual(result, expected) # Use assertCountEqual for order independence

        # Check that the duplicate path was effectively ignored by checked_paths logic
        # Count how many times isdir was called for the *parent* user models dir
        isdir_calls_for_user_parent = [
            call for call in mock_isdir.call_args_list if call[0][0] == user_models_dir
        ]
        # It should be checked once when iterating through user_path the first time.
        # The second time user_path appears, checked_paths should prevent re-checking its contents.
        self.assertEqual(len(isdir_calls_for_user_parent), 1, f"Expected isdir({user_models_dir}) to be called once")

# --- New Test Class for Model Checking --- 
class TestModelChecking(unittest.TestCase):
    """Test cases for model checking utility functions."""

    @patch('anpe.utils.setup_models.spacy.load')
    def test_check_spacy_model_present(self, mock_spacy_load):
        """Test check_spacy_model when the model is present."""
        # Configure mock to simulate successful loading (no error)
        mock_spacy_load.return_value = MagicMock()
        self.assertTrue(check_spacy_model("en_core_web_md"))
        mock_spacy_load.assert_called_once_with("en_core_web_md")

    @patch('anpe.utils.setup_models.spacy.load')
    def test_check_spacy_model_missing(self, mock_spacy_load):
        """Test check_spacy_model when the model is missing."""
        # Configure mock to raise OSError on load attempt
        mock_spacy_load.side_effect = OSError("Model not found")
        self.assertFalse(check_spacy_model("en_core_web_md"))
        mock_spacy_load.assert_called_once_with("en_core_web_md")

    @patch('anpe.utils.setup_models.nltk.data.find')
    def test_check_benepar_model_present(self, mock_nltk_find):
        """Test check_benepar_model when the model is present."""
        # Configure mock to return a dummy path (simulating found)
        mock_nltk_find.return_value = "/fake/path/models/benepar_en3"
        self.assertTrue(check_benepar_model("benepar_en3"))
        mock_nltk_find.assert_called_once_with('models/benepar_en3')

    @patch('anpe.utils.setup_models.nltk.data.find')
    def test_check_benepar_model_missing(self, mock_nltk_find):
        """Test check_benepar_model when the model is missing."""
        # Configure mock to raise LookupError
        mock_nltk_find.side_effect = LookupError("Resource not found")
        self.assertFalse(check_benepar_model("benepar_en3"))
        mock_nltk_find.assert_called_once_with('models/benepar_en3')

    # Test check_all_models_present relies on the individual checks,
    # so we mock those.
    @patch('anpe.utils.setup_models.check_spacy_model')
    @patch('anpe.utils.setup_models.check_benepar_model')
    def test_check_all_models_present_all_true(self, mock_check_benepar, mock_check_spacy):
        """Test check_all_models_present when all sub-checks return True."""
        mock_check_spacy.return_value = True
        mock_check_benepar.return_value = True
        
        # Test with default aliases
        self.assertTrue(check_all_models_present())
        mock_check_spacy.assert_called_once_with(model_name=SPACY_MODEL_MAP['md'])
        mock_check_benepar.assert_called_once_with(model_name=BENEPAR_MODEL_MAP['default'])
        
        # Reset mocks and test with specific aliases
        mock_check_spacy.reset_mock()
        mock_check_benepar.reset_mock()
        self.assertTrue(check_all_models_present(spacy_model_alias='lg', benepar_model_alias='large'))
        mock_check_spacy.assert_called_once_with(model_name=SPACY_MODEL_MAP['lg'])
        mock_check_benepar.assert_called_once_with(model_name=BENEPAR_MODEL_MAP['large'])

    @patch('anpe.utils.setup_models.check_spacy_model')
    @patch('anpe.utils.setup_models.check_benepar_model')
    def test_check_all_models_present_one_false(self, mock_check_benepar, mock_check_spacy):
        """Test check_all_models_present when one sub-check returns False."""
        mock_check_spacy.return_value = True
        mock_check_benepar.return_value = False # Simulate Benepar missing
        
        self.assertFalse(check_all_models_present())
        mock_check_spacy.assert_called_once()
        mock_check_benepar.assert_called_once()

# --- New Test Class for Model Installation --- 
class TestModelInstallation(unittest.TestCase):
    """Test cases for model installation utility functions."""

    # Mocks needed for install_spacy_model
    @patch('anpe.utils.setup_models.subprocess.Popen')
    @patch('anpe.utils.setup_models.importlib.util.find_spec')
    @patch('anpe.utils.setup_models._check_spacy_physical_path') 
    def test_install_spacy_model_success(self, mock_check_physical_path, mock_find_spec, mock_popen):
        """Test install_spacy_model successful installation via importlib verification."""
        from anpe.utils.setup_models import install_spacy_model
        # Simulate subprocess success
        mock_process = MagicMock()
        mock_process.stdout = ["Download successful"].__iter__() # Simulate some output
        mock_process.wait.return_value = 0 # Success code
        mock_popen.return_value = mock_process
        
        # Simulate verification success via find_spec
        mock_spec = MagicMock()
        mock_spec.origin = "/path/to/model"
        mock_find_spec.return_value = mock_spec
        mock_check_physical_path.return_value = False # Should not be needed if find_spec works
        
        self.assertTrue(install_spacy_model("en_core_web_sm"))
        mock_popen.assert_called_once() # Verify download command was run
        mock_find_spec.assert_called_once_with("en_core_web_sm") # Verify find_spec check
        mock_check_physical_path.assert_not_called() # Physical path check skipped

    @patch('anpe.utils.setup_models.subprocess.Popen')
    @patch('anpe.utils.setup_models.importlib.util.find_spec')
    @patch('anpe.utils.setup_models._check_spacy_physical_path')
    def test_install_spacy_model_success_physical_path(self, mock_check_physical_path, mock_find_spec, mock_popen):
        """Test install_spacy_model successful installation via physical path verification."""
        from anpe.utils.setup_models import install_spacy_model
        # Simulate subprocess success
        mock_process = MagicMock()
        mock_process.stdout = ["Download successful"].__iter__()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        # Simulate verification failure via find_spec, success via physical path
        mock_find_spec.return_value = None # Fails
        mock_check_physical_path.return_value = True # Succeeds

        self.assertTrue(install_spacy_model("en_core_web_sm"))
        mock_popen.assert_called_once()
        mock_find_spec.assert_called_once_with("en_core_web_sm")
        mock_check_physical_path.assert_called_once_with("en_core_web_sm") # Physical path checked

    @patch('anpe.utils.setup_models.subprocess.Popen')
    @patch('anpe.utils.setup_models.importlib.util.find_spec') 
    @patch('anpe.utils.setup_models._check_spacy_physical_path') 
    def test_install_spacy_model_fail_subprocess(self, mock_check_physical_path, mock_find_spec, mock_popen):
        """Test install_spacy_model when subprocess fails."""
        from anpe.utils.setup_models import install_spacy_model
        # Simulate subprocess failure
        mock_process = MagicMock()
        mock_process.stdout = ["Download failed"].__iter__()
        mock_process.wait.return_value = 1 # Error code
        mock_popen.return_value = mock_process
        
        self.assertFalse(install_spacy_model("en_core_web_sm")) 
        mock_popen.assert_called_once()
        # Verification checks should NOT be called if subprocess fails
        mock_find_spec.assert_not_called()
        mock_check_physical_path.assert_not_called()

    @patch('anpe.utils.setup_models.subprocess.Popen')
    @patch('anpe.utils.setup_models.importlib.util.find_spec') 
    @patch('anpe.utils.setup_models._check_spacy_physical_path') 
    def test_install_spacy_model_fail_check(self, mock_check_physical_path, mock_find_spec, mock_popen):
        """Test install_spacy_model when subprocess succeeds but all final checks fail."""
        from anpe.utils.setup_models import install_spacy_model
        # Simulate subprocess success
        mock_process = MagicMock()
        mock_process.stdout = ["Download successful"].__iter__()
        mock_process.wait.return_value = 0 
        mock_popen.return_value = mock_process
        
        # Simulate FINAL checks failing
        mock_find_spec.return_value = None
        mock_check_physical_path.return_value = False
        
        # ** This test might still fail if the function's logic incorrectly returns True **
        # ** based solely on subprocess success. Let's see. **
        self.assertFalse(install_spacy_model("en_core_web_sm")) 
        mock_popen.assert_called_once()
        mock_find_spec.assert_called_once_with("en_core_web_sm") # find_spec is called
        mock_check_physical_path.assert_called_once_with("en_core_web_sm") # physical check is called

    # Mocks needed for install_benepar_model
    @patch('anpe.utils.setup_models.subprocess.Popen')
    @patch('anpe.utils.setup_models.check_benepar_model') # Mock for the FINAL check after install attempt
    @patch('anpe.utils.setup_models.os.path.isdir') 
    @patch('anpe.utils.setup_models.os.path.isfile') 
    @patch('anpe.utils.setup_models._extract_zip_archive') # Mock zip extraction helper
    def test_install_benepar_model_success(self, mock_extract_zip, mock_isfile, mock_isdir, mock_final_check_benepar, mock_popen):
        """Test install_benepar_model successful installation."""
        from anpe.utils.setup_models import install_benepar_model
        # Simulate subprocess success
        mock_process = MagicMock()
        mock_process.stdout = ["Download successful"].__iter__()
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process
        
        # Simulate directory IS found after subprocess run
        mock_isdir.return_value = True 
        mock_isfile.return_value = False # Zip file doesn't exist or isn't checked
        mock_extract_zip.assert_not_called() # Shouldn't be called if dir exists
        
        # Simulate FINAL check confirms success
        mock_final_check_benepar.return_value = True
        
        self.assertTrue(install_benepar_model("benepar_en3"))
        mock_popen.assert_called_once() 
        mock_isdir.assert_called() # isdir is checked
        mock_final_check_benepar.assert_called_once_with("benepar_en3") # Final check is called

    @patch('anpe.utils.setup_models.subprocess.Popen')
    @patch('anpe.utils.setup_models.check_benepar_model') # Mock for the FINAL check
    @patch('anpe.utils.setup_models.os.path.isdir') 
    @patch('anpe.utils.setup_models.os.path.isfile') 
    @patch('anpe.utils.setup_models._extract_zip_archive')
    def test_install_benepar_model_fail_subprocess(self, mock_extract_zip, mock_isfile, mock_isdir, mock_final_check_benepar, mock_popen):
        """Test install_benepar_model when subprocess fails and model files don't exist."""
        from anpe.utils.setup_models import install_benepar_model
        # Simulate subprocess failure
        mock_process = MagicMock()
        mock_process.stdout = ["Download failed"].__iter__()
        mock_process.wait.return_value = 1 # Error code
        mock_popen.return_value = mock_process
        
        # Simulate model dir/zip NOT found after failed attempt
        mock_isdir.return_value = False
        mock_isfile.return_value = False
        mock_extract_zip.assert_not_called()
        
        self.assertFalse(install_benepar_model("benepar_en3"), "Function should return False when subprocess fails and model files are not found.")
        mock_popen.assert_called_once() 
        mock_isdir.assert_called() # isdir is checked
        mock_isfile.assert_called() # isfile is checked if isdir fails
        # Final check_benepar_model should NOT be called if install fails so early
        mock_final_check_benepar.assert_not_called() 

    # --- Tests for setup_models --- 

    @patch('anpe.utils.setup_models.install_spacy_model')
    @patch('anpe.utils.setup_models.install_benepar_model')
    @patch('anpe.utils.setup_models.check_spacy_model')
    @patch('anpe.utils.setup_models.check_benepar_model')
    def test_setup_models_all_missing_success(self, mock_check_benepar, mock_check_spacy, 
                                             mock_install_benepar, mock_install_spacy):
        """Test setup_models when all models are missing and installation succeeds."""
        from anpe.utils.setup_models import setup_models
        # Simulate all models missing initially
        mock_check_spacy.return_value = False      
        mock_check_benepar.return_value = False     
        # Simulate all installs succeed
        mock_install_spacy.return_value = True
        mock_install_benepar.return_value = True
        
        self.assertTrue(setup_models()) # Use default aliases
        # Check installs were called with log_callback=None
        mock_install_spacy.assert_called_once_with(model_name=SPACY_MODEL_MAP['md'], log_callback=None)
        mock_install_benepar.assert_called_once_with(model_name=BENEPAR_MODEL_MAP['default'], log_callback=None)
        # Check that checks were called the correct number of times
        mock_check_spacy.assert_called_once()  
        mock_check_benepar.assert_called_once() 
        
        # --- Additional check: Test with specific aliases ---
        mock_check_spacy.reset_mock()
        mock_check_benepar.reset_mock()
        mock_install_spacy.reset_mock()
        mock_install_benepar.reset_mock()

        # Simulate models missing again
        mock_check_spacy.return_value = False     
        mock_check_benepar.return_value = False    
        # Simulate installs succeed again
        mock_install_spacy.return_value = True
        mock_install_benepar.return_value = True
        
        # Test with specific aliases and a dummy log callback
        dummy_callback = lambda x: None
        self.assertTrue(setup_models(spacy_model_alias='trf', benepar_model_alias='large', log_callback=dummy_callback))
        # Check installs were called with the actual callback
        mock_install_spacy.assert_called_once_with(model_name=SPACY_MODEL_MAP['trf'], log_callback=dummy_callback)
        mock_install_benepar.assert_called_once_with(model_name=BENEPAR_MODEL_MAP['large'], log_callback=dummy_callback)
        # Check checks were called with the correct model names (they don't receive callback)
        mock_check_spacy.assert_called_once_with(model_name=SPACY_MODEL_MAP['trf'])
        mock_check_benepar.assert_called_once_with(model_name=BENEPAR_MODEL_MAP['large'])

if __name__ == "__main__":
    unittest.main() 