import pytest
import argparse
from unittest.mock import patch, MagicMock, ANY
import sys
import logging
from _pytest.logging import LogCaptureFixture

# Import the CLI module itself
from anpe import cli
# Import constants needed for assertions
from anpe.utils.setup_models import DEFAULT_SPACY_ALIAS, DEFAULT_BENEPAR_ALIAS, SPACY_MODEL_MAP, BENEPAR_MODEL_MAP

# --- Test Fixtures ---

# No CliRunner fixture needed anymore

# --- Tests for 'extract' command ---

@patch('anpe.cli.sys.exit') # Mock sys.exit to prevent test termination
@patch('anpe.cli.print_result_to_stdout') # Mock printing for easy checks
@patch('anpe.cli.create_extractor') # Mock the helper that creates the instance
@patch('anpe.cli.logging.getLogger') # Patch the logger
def test_cli_extract_basic(mock_get_logger, mock_create_extractor, mock_print_results, mock_sys_exit):
    """Test basic CLI extract command without file output using argparse style."""
    # Configure mock logger (even if not asserted, needed for setup)
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    mock_extractor_instance = MagicMock()
    # Mock the extract method on the *instance* that create_extractor will return
    mock_extractor_instance.extract.return_value = {
        'configuration': {'metadata_requested': False, 'nested_requested': False}, # Simplified for example
        'results': [{'id': '1', 'level': 1, 'noun_phrase': 'test np', 'children': []}]
    }
    mock_create_extractor.return_value = mock_extractor_instance
    
    input_text = "This is test text."
    args_list = ['extract', input_text] # Pass text directly as positional arg
    
    # Call the main function directly
    exit_code = cli.main(args_list)
    
    # Assertions
    assert exit_code == 0 # Expect successful exit
    mock_sys_exit.assert_not_called() # Ensure no premature exit

    # Check create_extractor was called (config checks later)
    mock_create_extractor.assert_called_once()
    
    # Check extract was called correctly via process_text
    # Note: The actual call is inside process_text, which is called by main.
    # We check the mock instance returned by create_extractor.
    mock_extractor_instance.extract.assert_called_once_with(
        input_text, 
        metadata=False, 
        include_nested=False
    )
    
    # Check that print_result_to_stdout was called with the result data
    mock_print_results.assert_called_once_with(mock_extractor_instance.extract.return_value)

@patch('anpe.cli.sys.exit') 
@patch('anpe.cli.create_extractor')
@patch('anpe.cli.logging.getLogger') # Patch the logger
def test_cli_extract_with_options_and_export(mock_get_logger, mock_create_extractor, mock_sys_exit):
    """Test CLI extract with options and export using argparse style."""
    # Configure mock logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    mock_extractor_instance = MagicMock()
    mock_extractor_instance.export.return_value = "/fake/output.csv"
    mock_create_extractor.return_value = mock_extractor_instance
    
    input_text = "Another test."
    output_file = "out.csv"
    
    args_list = [
        'extract',
        input_text,
        '--output', output_file,
        '--type', 'csv', 
        '--metadata',
        '--nested',
        '--min-length', '2', 
        '--no-pronouns' 
    ]

    exit_code = cli.main(args_list)

    assert exit_code == 0
    mock_sys_exit.assert_not_called()

    mock_create_extractor.assert_called_once() 
    call_args, call_kwargs = mock_create_extractor.call_args
    passed_args_obj = call_args[0] 
    assert isinstance(passed_args_obj, argparse.Namespace)
    assert passed_args_obj.min_length == 2
    assert passed_args_obj.no_pronouns is True 
    assert passed_args_obj.metadata is True
    assert passed_args_obj.nested is True
    assert passed_args_obj.type == 'csv'
    assert passed_args_obj.output == output_file
    
    mock_extractor_instance.export.assert_called_once_with(
        text=input_text,
        format='csv',
        output=output_file,
        metadata=True,
        include_nested=True
    )

@patch('anpe.cli.sys.exit')
@patch('anpe.cli.create_extractor')
@patch('anpe.cli.logging.getLogger') # Patch the logger
@patch('anpe.cli.sys.stdin') # Mock stdin
def test_cli_extract_missing_input_handled(mock_stdin, mock_get_logger, mock_create_extractor, mock_sys_exit):
    """Test CLI extract fails gracefully if text arg missing and stdin is empty."""
    # Configure mock logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    # Simulate stdin being empty
    mock_stdin.read.return_value = ""
    
    # Run main without the positional text argument
    exit_code = cli.main(['extract'])
    
    # Expect non-zero exit code because input is truly missing
    assert exit_code == 1
    mock_sys_exit.assert_not_called()
    
    # Check extractor was NOT created because input failed first
    mock_create_extractor.assert_not_called()
    
    # Check error message was logged
    mock_logger.error.assert_called_once()
    assert "No input text provided" in mock_logger.error.call_args[0][0]

def test_cli_extract_invalid_format_args():
    """Test parse_args fails with an invalid format/type choice."""
    input_text = "Some text."
    with pytest.raises(SystemExit):
        cli.parse_args([
            'extract',
            input_text,
            '--type', 'invalid_format' # Bad choice for --type
        ])

@patch('anpe.cli.sys.exit')
@patch('anpe.cli.create_extractor')
@patch('anpe.cli.logging.getLogger') # Patch the logger
def test_cli_extract_init_error(mock_get_logger, mock_create_extractor, mock_sys_exit):
    """Test CLI extract handles errors during ANPEExtractor initialization."""
    # Configure mock logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    mock_create_extractor.side_effect = RuntimeError("Mock Extractor Init Error")
    
    input_text = "This will fail initialization."
    exit_code = cli.main(['extract', input_text])

    assert exit_code == 1 
    mock_sys_exit.assert_not_called() 

    mock_create_extractor.assert_called_once()
    # Check error message was logged
    mock_logger.error.assert_called_once()
    assert "Error creating extractor: Mock Extractor Init Error" in mock_logger.error.call_args[0][0]

@patch('anpe.cli.sys.exit')
@patch('anpe.cli.create_extractor') 
@patch('anpe.cli.logging.getLogger') # Patch the logger
def test_cli_extract_export_error(mock_get_logger, mock_create_extractor, mock_sys_exit, capsys):
    """Test CLI extract handles errors during the export process."""
    # Configure mock logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    mock_extractor_instance = MagicMock()
    mock_extractor_instance.export.side_effect = ValueError("Mock Export Error")
    mock_create_extractor.return_value = mock_extractor_instance
    
    input_text = "Test export failure."
    output_file = "out.txt"
    
    exit_code = cli.main([
        'extract',
        input_text,
        '--output', output_file,
        '--type', 'txt'
    ])
    
    assert exit_code == 1 
    mock_sys_exit.assert_not_called()
    
    mock_create_extractor.assert_called_once()
    mock_extractor_instance.export.assert_called_once()
    
    # Check the error message printed to stderr
    captured = capsys.readouterr()
    # Match the actual logged error message format (printed by except block)
    assert "Error: Mock Export Error" in captured.err

# --- Tests for 'setup' command ---

@patch('anpe.cli.sys.exit')
@patch('anpe.utils.setup_models.install_benepar_model')
@patch('anpe.utils.setup_models.install_spacy_model')
@patch('anpe.utils.setup_models.check_benepar_model')
@patch('anpe.utils.setup_models.check_spacy_model')
@patch('anpe.cli.logging.getLogger') # Keep logger patch if used
def test_cli_setup_default(
    mock_get_logger, 
    mock_check_spacy, 
    mock_check_benepar, 
    mock_install_spacy, 
    mock_install_benepar, 
    mock_sys_exit
):
    """Test CLI setup command with default models triggering installations."""
    # Configure mock logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    # Simulate models are not installed
    mock_check_spacy.return_value = False
    mock_check_benepar.return_value = False
    # Simulate successful installation
    mock_install_spacy.return_value = True
    mock_install_benepar.return_value = True

    # Expected default aliases from anpe.utils.setup_models constants
    from anpe.utils.setup_models import DEFAULT_SPACY_ALIAS, DEFAULT_BENEPAR_ALIAS, SPACY_MODEL_MAP, BENEPAR_MODEL_MAP

    exit_code = cli.main(['setup'])

    assert exit_code == 0
    mock_sys_exit.assert_not_called()

    # Verify checks were called with full model names derived from default aliases
    mock_check_spacy.assert_called_once_with(model_name=SPACY_MODEL_MAP[DEFAULT_SPACY_ALIAS], adhoc_trf_module_install=False)
    mock_check_benepar.assert_called_once_with(model_name=BENEPAR_MODEL_MAP[DEFAULT_BENEPAR_ALIAS])

    # Verify install functions were called with default aliases since checks returned False
    mock_install_spacy.assert_called_once_with(model_name=DEFAULT_SPACY_ALIAS, log_callback=ANY)
    mock_install_benepar.assert_called_once_with(model_name=DEFAULT_BENEPAR_ALIAS, log_callback=ANY)

@patch('anpe.cli.sys.exit')
@patch('anpe.cli.setup_models')
@patch('anpe.cli.logging.getLogger') # Patch the logger
def test_cli_setup_specific_models(mock_get_logger, mock_setup, mock_sys_exit):
    """Test CLI setup command with specific model aliases."""
    # Configure mock logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    exit_code = cli.main([
        'setup',
        '--spacy-model', 'lg', 
        '--benepar-model', 'benepar_en3'
    ])
    
    assert exit_code == 0
    mock_sys_exit.assert_not_called()
    mock_setup.assert_called_once()
    call_args, call_kwargs = mock_setup.call_args
    assert call_kwargs.get('spacy_model_alias') == 'lg'
    assert call_kwargs.get('benepar_model_alias') == 'benepar_en3'
    assert 'log_callback' in call_kwargs # Correct key is log_callback
    assert callable(call_kwargs['log_callback'])

@patch('anpe.cli.sys.exit')
@patch('anpe.utils.setup_models.install_benepar_model') # Mock the actual function in setup_models
@patch('anpe.utils.setup_models.install_spacy_model')   # Mock the actual function in setup_models
@patch('anpe.utils.setup_models.check_benepar_model')
@patch('anpe.utils.setup_models.check_spacy_model')
@patch('anpe.cli.logging.getLogger')
def test_cli_setup_error(
    mock_get_logger, 
    mock_check_spacy, 
    mock_check_benepar, 
    mock_install_spacy, 
    mock_install_benepar, 
    mock_sys_exit, 
    capsys
):
    """Test CLI setup handles errors during model setup (e.g., spaCy model install fails)."""
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger

    # Simulate spaCy model is not installed, but Benepar is (or its check passes)
    mock_check_spacy.return_value = False
    mock_check_benepar.return_value = True # Assume Benepar is fine or its check passes

    # Simulate an error during spaCy model installation
    mock_install_spacy.return_value = False # Indicates failure
    mock_install_benepar.return_value = True # Assume Benepar would install if called

    # Expected default aliases from anpe.utils.setup_models constants
    from anpe.utils.setup_models import DEFAULT_SPACY_ALIAS, DEFAULT_BENEPAR_ALIAS, SPACY_MODEL_MAP, BENEPAR_MODEL_MAP

    exit_code = cli.main(['setup']) # Run default setup
    
    assert exit_code == 1 # Expect failure exit code
    mock_sys_exit.assert_not_called() # main should return 1, not call sys.exit directly for this

    mock_check_spacy.assert_called_once_with(model_name=SPACY_MODEL_MAP[DEFAULT_SPACY_ALIAS], adhoc_trf_module_install=False)
    # Benepar check might also be called depending on logic, or not if spaCy fails first. 
    # For this test, let's assume it's called.
    mock_check_benepar.assert_called_once_with(model_name=BENEPAR_MODEL_MAP[DEFAULT_BENEPAR_ALIAS])

    # Assert spaCy installation was attempted
    mock_install_spacy.assert_called_once_with(model_name=DEFAULT_SPACY_ALIAS, log_callback=ANY)
    # Benepar installation should NOT be attempted if its check returned True
    mock_install_benepar.assert_not_called() 

    # Check the error message logged by cli.main for failed setup
    # The specific model failure is logged by setup_models internal logger (or callback).
    # cli.main logs a more general message based on the False return from setup_models.
    mock_logger.error.assert_any_call('One or more models failed to install during the final setup phase.')

    # Verify stderr output if cli.py directly prints for this specific failure case
    captured = capsys.readouterr()
    # Example: assert "Failed to install spaCy model" in captured.err
    # However, current cli.py relies on logging for these messages rather than direct print to stderr for individual model failures.

# --- Tests for 'clean' command ---

@patch('anpe.cli.sys.exit')
@patch('anpe.cli.clean_all')
@patch('anpe.cli.logging.getLogger') # Patch the logger
def test_cli_clean(mock_get_logger, mock_clean, mock_sys_exit):
    """Test CLI setup --clean-models command (assuming no confirmation prompt or force=True)."""
    # Configure mock logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    # Add --force flag assuming it bypasses confirmation, or remove if clean always runs
    exit_code = cli.main(['setup', '--clean-models', '--force'])
    
    assert exit_code == 0
    mock_sys_exit.assert_not_called()
    mock_clean.assert_called_once()
    call_args, call_kwargs = mock_clean.call_args
    assert 'logger' in call_kwargs # Corrected assertion from log_callback to logger

@patch('anpe.cli.sys.exit')
@patch('anpe.cli.clean_all')
@patch('anpe.cli.logging.getLogger') # Patch the logger
@patch('builtins.input', return_value='n') # Keep input mock here to test abort
@patch('anpe.cli.sys.stdin.isatty', return_value=True) # Force interactive check to pass
def test_cli_clean_aborted(mock_get_logger, mock_isatty, mock_input, mock_clean, mock_sys_exit, capsys):
    """Test CLI clean command aborts if confirmation is requested and denied."""
    # Configure mock logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    # Run *without* --force to trigger confirmation
    exit_code = cli.main(['setup', '--clean-models'])
    
    assert exit_code == 1 # Expect abort code
    mock_sys_exit.assert_not_called()
    mock_isatty.assert_called_once() # Ensure interactive check happened
    mock_input.assert_called_once() # Check confirmation was asked
    mock_clean.assert_not_called()
    
    # Check the abort message printed to stderr
    captured = capsys.readouterr()
    # Check the specific abort message from cli.py
    assert "Cleanup aborted." in captured.err # Assuming message goes to stderr

@patch('anpe.cli.sys.exit')
@patch('anpe.cli.clean_all')
@patch('anpe.cli.logging.getLogger') # Patch the logger
@patch('builtins.input', return_value='y') # Keep input mock in case --force logic is different
def test_cli_clean_error(mock_get_logger, mock_input, mock_clean, mock_sys_exit, capsys):
    """Test CLI clean handles errors during model cleaning."""
    # Configure mock logger
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    
    mock_clean.side_effect = OSError("Mock Model Deletion Error")
    
    # Run with --force if that's needed to bypass confirmation before error
    exit_code = cli.main(['setup', '--clean-models', '--force'])
    
    assert exit_code == 1 
    mock_sys_exit.assert_not_called()
    mock_clean.assert_called_once()
    
    # Check the error message printed to stderr
    captured = capsys.readouterr()
    # Match actual logged error format (printed by except block)
    assert "Error: Mock Model Deletion Error" in captured.err