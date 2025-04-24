import unittest
import tempfile
import os
import json
from pathlib import Path
from anpe.cli import main, parse_args
import csv
from unittest.mock import patch, MagicMock, call

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

    def test_extract_direct_text_input(self):
        """Test extraction using direct text input."""
        # Outputting to a specific file for easier verification
        specific_output_file = self.output_dir / "direct_text_output.txt"
        args = ["extract", self.test_text, "-o", str(specific_output_file), "-t", "txt"]
        exit_code = main(args)
        self.assertEqual(exit_code, 0)
        self.assertTrue(specific_output_file.exists())
        # Check if output contains expected NP (basic check)
        with open(specific_output_file, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertIn("The team of scientists", content)

    def test_extract_from_directory(self):
        """Test extraction using directory input (-d)."""
        # Create a subdirectory with another test file
        input_subdir = Path(self.temp_dir.name) / "input_dir"
        input_subdir.mkdir()
        subdir_file = input_subdir / "subdir_input.txt"
        with open(subdir_file, 'w', encoding='utf-8') as f:
            f.write("Another test sentence with noun phrases.")
        
        # Output to a directory
        output_subdir = self.output_dir / "dir_output"
        args = ["extract", "-d", str(input_subdir), "-o", str(output_subdir), "-t", "json"]
        exit_code = main(args)
        self.assertEqual(exit_code, 0)
        
        # Check if the output directory was created
        self.assertTrue(output_subdir.exists())
        self.assertTrue(output_subdir.is_dir())
        
        # Check if an output file was created inside the output directory
        output_files = list(output_subdir.glob("anpe_export_*.json"))
        self.assertGreater(len(output_files), 0, "No JSON file created in output directory for directory input test")
        
        # Verify content of the created file
        with open(output_files[0], 'r', encoding='utf-8') as f:
            result = json.load(f)
            self.assertIn("results", result)
            found_np = any("noun phrases" in np["noun_phrase"].lower() for np in result["results"])
            self.assertTrue(found_np, "Expected NP not found in directory input test output")

    @patch('anpe.cli.create_extractor')
    def test_extract_cli_flags_pass_to_config(self, mock_create_extractor):
        """Test that CLI processing flags correctly modify extractor config."""
        # Mock the actual extractor instance to prevent real processing
        mock_extractor_instance = unittest.mock.MagicMock()
        mock_extractor_instance.export.return_value = None # Mock export behaviour
        mock_create_extractor.return_value = mock_extractor_instance
        
        # Test various flags
        args = [
            "extract", 
            "-f", str(self.input_file),
            "-o", str(self.output_dir), # Output dir needed for code path
            "--min-length", "2", 
            "--max-length", "5", 
            "--no-pronouns", 
            "--no-newline-breaks", 
            "--structures", "determiner,compound",
            "--spacy-model", "lg",
            "--benepar-model", "large"
        ]
        exit_code = main(args)
        self.assertEqual(exit_code, 0)
        
        # Assert create_extractor was called once
        mock_create_extractor.assert_called_once()
        
        # Get the actual Namespace object passed to create_extractor
        call_args, call_kwargs = mock_create_extractor.call_args
        parsed_args_passed = call_args[0] # The Namespace is the first positional argument
        
        # Verify flags exist as attributes on the passed Namespace object
        self.assertEqual(parsed_args_passed.min_length, 2)
        self.assertEqual(parsed_args_passed.max_length, 5)
        self.assertTrue(parsed_args_passed.no_pronouns) # Flag presence means True
        self.assertTrue(parsed_args_passed.no_newline_breaks) # Flag presence means True
        self.assertEqual(parsed_args_passed.structures, "determiner,compound")
        self.assertEqual(parsed_args_passed.spacy_model, "lg") 
        self.assertEqual(parsed_args_passed.benepar_model, "large")
        
        # Ensure extractor.export was called (or whatever method process_file calls)
        # Since we test file input, process_file gets called, which calls extractor.export
        mock_extractor_instance.export.assert_called_once()

    # --- Tests for Setup Command ---

    @patch('anpe.cli.check_all_models_present')
    @patch('anpe.cli.setup_models')
    @patch('anpe.cli.ANPELogger') # Mock logger setup too
    def test_setup_command_models_present(self, mock_logger_cls, mock_setup_models, mock_check_all):
        """Test setup command when all models are already present (installation path)."""
        # Mock logger instance behavior
        mock_logger_instance = MagicMock()
        mock_logger_cls.setup_logging.return_value = mock_logger_instance
        
        mock_check_all.return_value = True # Simulate models are present
        
        # Use defaults by not providing model flags
        args = ["setup"]
        exit_code = main(args)
        
        self.assertEqual(exit_code, 0)
        # Installation logic shouldn't run check_all_models_present anymore
        # mock_check_all.assert_called_once_with(spacy_model_alias='md', benepar_model_alias='default')
        mock_check_all.assert_not_called() # No check before install
        # setup_models should NOT be called if check_all returns True (Old logic, removed)
        # New logic: setup_models IS called regardless of pre-check
        mock_setup_models.assert_called_once()

    @patch('anpe.cli.setup_models')
    @patch('anpe.cli.get_logger')
    def test_setup_command_models_missing_success(self, mock_get_logger, mock_setup_models):
        """Test setup command when models are missing and installation succeeds."""
        mock_setup_logger = MagicMock()
        mock_get_logger.return_value = mock_setup_logger
        
        # Simulate successful installation by not raising Exception
        mock_setup_models.return_value = None 

        args = ["setup"]
        exit_code = main(args)
        
        self.assertEqual(exit_code, 0)
        # setup_models should be called with default aliases and the logger
        mock_setup_models.assert_called_once()
        call_args, call_kwargs = mock_setup_models.call_args
        self.assertEqual(call_kwargs.get('spacy_model_alias'), 'md')
        self.assertEqual(call_kwargs.get('benepar_model_alias'), 'default')
        self.assertIsNotNone(call_kwargs.get('logger')) # Check logger was passed

    @patch('anpe.cli.setup_models')
    @patch('anpe.cli.get_logger')
    def test_setup_command_models_missing_fail(self, mock_get_logger, mock_setup_models):
        """Test setup command when models are missing and installation fails."""
        mock_setup_logger = MagicMock()
        mock_get_logger.return_value = mock_setup_logger
    
        # Simulate failed installation
        exception_message = "Download failed"
        mock_setup_models.side_effect = Exception(exception_message)
    
        args = ["setup"]
        exit_code = main(args)
    
        self.assertEqual(exit_code, 1) # Should exit with error code 1
        mock_setup_models.assert_called_once() # Still called
        # Check logger was called with the correct error message
        mock_setup_logger.error.assert_any_call(f"Model setup failed: {exception_message}")

    @patch('anpe.cli.setup_models')
    @patch('anpe.cli.ANPELogger')
    def test_setup_command_specific_models(self, mock_logger_cls, mock_setup_models):
        """Test setup command with specific model flags."""
        mock_logger_instance = MagicMock()
        mock_logger_cls.setup_logging.return_value = mock_logger_instance
        mock_setup_models.return_value = None # Simulate success

        args = ["setup", "--spacy-model", "lg", "--benepar-model", "large"]
        exit_code = main(args)
        
        self.assertEqual(exit_code, 0)
        # Check if setup_models was called with the correct specific aliases
        mock_setup_models.assert_called_once()
        call_args, call_kwargs = mock_setup_models.call_args
        self.assertEqual(call_kwargs.get('spacy_model_alias'), 'lg')
        self.assertEqual(call_kwargs.get('benepar_model_alias'), 'large')
        self.assertIsNotNone(call_kwargs.get('logger'))

    @patch('anpe.cli.setup_models')
    @patch('anpe.cli.ANPELogger')
    def test_setup_command_only_spacy(self, mock_logger_cls, mock_setup_models):
        """Test setup command specifying only spaCy model."""
        mock_logger_instance = MagicMock()
        mock_logger_cls.setup_logging.return_value = mock_logger_instance
        mock_setup_models.return_value = None

        args = ["setup", "--spacy-model", "sm"]
        exit_code = main(args)
        
        self.assertEqual(exit_code, 0)
        # Should setup with specified spacy and default benepar
        mock_setup_models.assert_called_once()
        call_args, call_kwargs = mock_setup_models.call_args
        self.assertEqual(call_kwargs.get('spacy_model_alias'), 'sm')
        self.assertEqual(call_kwargs.get('benepar_model_alias'), 'default') # Default used
        self.assertIsNotNone(call_kwargs.get('logger'))
        
    @patch('anpe.cli.setup_models')
    @patch('anpe.cli.ANPELogger')
    def test_setup_command_only_benepar(self, mock_logger_cls, mock_setup_models):
        """Test setup command specifying only Benepar model."""
        mock_logger_instance = MagicMock()
        mock_logger_cls.setup_logging.return_value = mock_logger_instance
        mock_setup_models.return_value = None

        args = ["setup", "--benepar-model", "large"]
        exit_code = main(args)
        
        self.assertEqual(exit_code, 0)
        # Should setup with default spacy and specified benepar
        mock_setup_models.assert_called_once()
        call_args, call_kwargs = mock_setup_models.call_args
        self.assertEqual(call_kwargs.get('spacy_model_alias'), 'md') # Default used
        self.assertEqual(call_kwargs.get('benepar_model_alias'), 'large')
        self.assertIsNotNone(call_kwargs.get('logger'))
        
    # --- Tests for Clean Models --- 
    
    @patch('anpe.cli.clean_all')
    @patch('anpe.cli.get_logger')
    def test_setup_clean_confirm_yes_success(self, mock_get_logger, mock_clean_all):
        """Test setup --clean-models with confirmation 'y' (no -f flag) simulated by clean_all succeeding."""
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        # Simulate clean_all succeeding (as if user confirmed 'y')
        mock_clean_all.return_value = {"spacy": True, "benepar": True, "overall": True}

        args = ["setup", "--clean-models"] # No -f flag
        exit_code = main(args)

        self.assertEqual(exit_code, 0)
        # Assert clean_all was called once with force=False
        mock_clean_all.assert_called_once()
        call_args, call_kwargs = mock_clean_all.call_args
        self.assertIn('force', call_kwargs)
        self.assertFalse(call_kwargs['force'])

    @patch('anpe.cli.clean_all')
    @patch('anpe.cli.get_logger')
    def test_setup_clean_confirm_no(self, mock_get_logger, mock_clean_all):
        """Test setup --clean-models with confirmation 'n' (no -f flag) simulated by clean_all succeeding overall (handles cancellation gracefully)."""
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        # Simulate clean_all reporting overall success even if user cancels ('n')
        # because cancellation is handled internally and is not an error state.
        # The function might log the cancellation but exits cleanly.
        mock_clean_all.return_value = {"spacy": True, "benepar": True, "overall": True}

        args = ["setup", "--clean-models"] # No -f flag
        exit_code = main(args)

        # Exit code 0 because clean_all handles cancellation gracefully
        self.assertEqual(exit_code, 0)
        # Assert clean_all was called once with force=False
        mock_clean_all.assert_called_once()
        call_args, call_kwargs = mock_clean_all.call_args
        self.assertIn('force', call_kwargs)
        self.assertFalse(call_kwargs['force'])

    @patch('anpe.cli.clean_all')
    @patch('anpe.cli.get_logger')
    def test_setup_clean_force_flag_success(self, mock_get_logger, mock_clean_all):
        """Test setup --clean-models -f (force flag) with success."""
        mock_logger_instance = MagicMock()
        mock_get_logger.return_value = mock_logger_instance
        mock_clean_all.return_value = {"spacy": True, "benepar": True, "overall": True}

        args = ["setup", "--clean-models", "-f"] # Use -f flag
        exit_code = main(args)

        self.assertEqual(exit_code, 0)
        # No input prompt when force=True
        # mock_input.assert_not_called() # Removed as mock_input is not used
        mock_clean_all.assert_called_once()
        call_args, call_kwargs = mock_clean_all.call_args
        self.assertTrue(call_kwargs.get('force'))

    @patch('anpe.cli.clean_all')
    @patch('anpe.cli.get_logger')
    def test_setup_clean_force_flag_failure(self, mock_get_logger, mock_clean_all):
        """Test setup --clean-models -f when clean_all fails."""
        mock_setup_logger = MagicMock()
        mock_get_logger.return_value = mock_setup_logger
        # Simulate overall failure
        mock_clean_all.return_value = {"spacy": False, "benepar": True, "overall": False}

        args = ["setup", "--clean-models", "-f"] # Use -f flag
        exit_code = main(args)

        self.assertEqual(exit_code, 1) # Exit code 1 on overall failure
        mock_clean_all.assert_called_once() # Should call clean_all
        call_args, call_kwargs = mock_clean_all.call_args
        self.assertTrue(call_kwargs.get('force')) # force should be True
        # Assert on the logger returned by get_logger for the correct message
        mock_setup_logger.error.assert_any_call("Model cleanup failed.")

    @patch('anpe.cli.clean_all')
    @patch('anpe.cli.get_logger')
    def test_setup_clean_conflicting_flags(self, mock_get_logger, mock_clean_all):
        """Test setup --clean-models with conflicting installation flag."""
        mock_setup_logger = MagicMock()
        mock_get_logger.return_value = mock_setup_logger

        args = ["setup", "--clean-models", "--spacy-model", "md"]
        exit_code = main(args)

        self.assertEqual(exit_code, 1) # Error due to conflict
        mock_clean_all.assert_not_called() # clean_all should not be called
        # Check specific error log on the correct logger
        mock_setup_logger.error.assert_called_once_with(
            "Cannot use --clean-models with specific model installation flags (--spacy-model, --benepar-model)."
        )


if __name__ == "__main__":
    unittest.main() 