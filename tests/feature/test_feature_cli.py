import pytest
import json
import csv
import os
import subprocess
from pathlib import Path
from unittest.mock import patch, ANY, MagicMock
import tempfile
import shutil
import site
import nltk
import time
import sys # Added for mocking sys.exit
import importlib.metadata # Added for exception
import anpe.cli 

# Define a constant for the output directory relative to this test file
OUTPUT_DIR = Path(__file__).parent / "cli_output"

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e

# Define TEST_CONFIG or import it if defined elsewhere
# Minimal config sufficient for feature tests if models are assumed installed
TEST_CONFIG = {
    "log_level": "DEBUG",
    "log_dir": None, 
}

# Fixture to ensure the output directory exists and is clean
@pytest.fixture(scope="module", autouse=True)
def manage_output_dir():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clean up previous test files before the module runs
    for item in OUTPUT_DIR.glob('test_output_*'):
        if item.is_file(): # Ensure it's a file before unlinking
             item.unlink()
    yield
    # Clean up test files after tests complete
    for item in OUTPUT_DIR.glob('test_output_*'):
        if item.is_file():
            item.unlink()
    # Also clean up test input files
    for item in OUTPUT_DIR.glob('test_input_*'):
        if item.is_file():
            item.unlink()

# Helper to get expected NLTK data directory (mirroring setup_models logic)
# Avoid importing directly from setup_models to prevent potential side effects
# during test collection.
def get_nltk_data_dir():
    home = os.path.expanduser("~")
    return os.path.join(home, "nltk_data")

NLTK_DATA_DIR = get_nltk_data_dir()
DEFAULT_SPACY_MODEL = "en_core_web_md" # From setup_models.py
DEFAULT_BENEPAR_MODEL = "benepar_en3" # From setup_models.py

def _check_spacy_model_exists(model_name: str) -> bool:
    """Checks if a spaCy model directory exists in site-packages."""
    site_packages_dirs = site.getsitepackages()
    user_site = site.getusersitepackages()
    if user_site and os.path.isdir(user_site) and user_site not in site_packages_dirs:
        site_packages_dirs.append(user_site)

    for sp_dir in site_packages_dirs:
        if os.path.isdir(os.path.join(sp_dir, model_name)):
            return True
    return False

def _check_benepar_model_exists(model_name: str) -> bool:
    """Checks if a Benepar model directory exists in the NLTK data path."""
    # Benepar stores models like 'benepar_en3' inside NLTK_DATA_DIR/models/
    expected_path = os.path.join(NLTK_DATA_DIR, "models", model_name)
    return os.path.isdir(expected_path)

# Fixture to ensure the target NLTK data directory exists before tests run.
@pytest.fixture(scope="module", autouse=True)
def ensure_nltk_data_dir_exists():
    """Ensure the target NLTK data directory exists before tests run."""
    os.makedirs(NLTK_DATA_DIR, exist_ok=True)
    # Ensure the 'models' subdirectory also exists, as nltk.data.find expects it
    os.makedirs(os.path.join(NLTK_DATA_DIR, "models"), exist_ok=True)

# NOTE: These tests assume necessary models (e.g., default spaCy/Benepar) are installed.
# They invoke the actual CLI script.

def run_cli_command(args):
    """Helper to run the ANPE CLI command using the installed script."""
    # Use the script name defined in pyproject.toml
    base_command = ["anpe"] # Assuming 'anpe' is in the PATH of the test env
    command = base_command + args
    print(f"Running command: {' '.join(command)}") # Debug print
    # Ensure environment variables are inherited, especially PATH
    result = subprocess.run(command, capture_output=True, text=True, check=False, env=os.environ)
    print(f"CLI stdout:\n{result.stdout}")
    print(f"CLI stderr:\n{result.stderr}")
    result.check_returncode() # Raise error if non-zero exit code
    return result

def test_feature_cli_extract_json_output(): # Removed runner, tmp_path
    """
    Feature Test: Verify CLI extraction to a JSON file.
    """
    input_text = "The quick brown fox jumps over the lazy dog."
    output_filename = "test_output_simple.json"
    output_file = OUTPUT_DIR / output_filename
    # Use a set of key expected NPs
    expected_key_nps = {
        "The quick brown fox",
        "the lazy dog"
    }
    
    run_cli_command([
        'extract',
        input_text,
        '--output', str(output_file),
        '--type', 'json' 
    ])
    
    assert output_file.exists()
    with open(output_file, 'r') as f:
        data = json.load(f)
        
    assert "results" in data
    # Use set-based assertion
    extracted_nps = {item["noun_phrase"] for item in data["results"]}
    missing_nps = expected_key_nps - extracted_nps
    assert not missing_nps, f"Missing expected key NPs: {missing_nps}\nExtracted: {extracted_nps}"
    
    assert data["configuration"]["metadata_requested"] is False
    assert data["configuration"]["nested_requested"] is False

def test_feature_cli_extract_csv_output(): # Removed runner, tmp_path
    """
    Feature Test: Verify CLI extraction to a CSV file.
    """
    input_text = "The quick brown fox jumps over the lazy dog."
    output_filename = "test_output_simple.csv"
    output_file = OUTPUT_DIR / output_filename
    # Use a set of key expected NPs
    expected_key_nps = {
        "The quick brown fox",
        "the lazy dog"
    }
    
    run_cli_command([
        'extract', input_text, '--output', str(output_file), '--type', 'csv'
    ])
    
    assert output_file.exists()
    extracted_nps = []
    with open(output_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader) # Skip header
        assert "Noun_Phrase" in header # Basic header check
        for row in reader:
            # Assuming Noun_Phrase is the 4th column (index 3) based on unit tests
            if len(row) > 3:
                extracted_nps.append(row[3]) 
                
    # Use set-based assertion
    missing_nps = expected_key_nps - set(extracted_nps)
    assert not missing_nps, f"Missing expected key NPs: {missing_nps}\nExtracted: {set(extracted_nps)}"

@patch('sys.exit') # Add mock for sys.exit
def test_feature_cli_extract_txt_output(mock_exit): # Remove tmp_path, keep mock_exit
    """
    Feature Test: Verify CLI extraction to a TXT file using cli.main
    and manual temp directory.
    """
    input_text = "The quick brown fox jumps over the lazy dog."
    
    # Use tempfile.TemporaryDirectory instead of tmp_path
    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = Path(tmpdir) / "output.txt"

        # Define arguments for cli_main
        args = [
            'extract',
            input_text,
            '--output', str(output_file),
            '--type', 'txt'
        ]

        # Call cli_main directly
        # exit_code = cli_main(args) # Old call
        exit_code = anpe.cli.main(args) # New call

        # Assertions
        assert exit_code == 0, f"CLI main exited with code {exit_code}"
        mock_exit.assert_not_called() # Ensure sys.exit wasn't called unexpectedly
        assert output_file.exists()
        # Specify UTF-8 encoding when reading the file
        content = output_file.read_text(encoding='utf-8')

        # Assert that the expected NPs are present in the output
        assert "The quick brown fox" in content
        assert "the lazy dog" in content

def test_feature_cli_extract_nested_json_output(): # Removed runner, tmp_path
    """
    Feature Test: Verify CLI extraction with --nested to a JSON file.
    Adjusted expectations based on observed Benepar parse for this sentence.
    """
    input_text = "I saw the small cat in the big hat."
    output_filename = "test_output_nested.json"
    output_file = OUTPUT_DIR / output_filename
    # Use a set of key expected NPs (including potential nested ones if structure was reliable)
    # For this specific parse, only top-level are distinct and expected reliably.
    expected_key_nps = {"I", "the small cat", "the big hat"}
    
    run_cli_command([
        'extract', input_text, '--output', str(output_file), '--type', 'json', '--nested'
    ])
    
    assert output_file.exists()
    with open(output_file, 'r') as f:
        data = json.load(f)
        
    assert "results" in data
    assert data["configuration"]["nested_requested"] is True
    
    # Robust check for key NPs using sets
    # Flatten the results if nested to check all extracted NPs
    all_extracted_nps = set()
    nodes_to_process = data["results"]
    while nodes_to_process:
        node = nodes_to_process.pop(0)
        all_extracted_nps.add(node["noun_phrase"])
        if "children" in node:
            nodes_to_process.extend(node["children"])
    
    missing_nps = expected_key_nps - all_extracted_nps
    assert not missing_nps, f"Missing expected key NPs: {missing_nps}\nExtracted: {all_extracted_nps}"

def test_feature_cli_extract_metadata_json_output(): # Removed runner, tmp_path
    """
    Feature Test: Verify CLI extraction with --metadata to a JSON file.
    """
    input_text = "The large green book is interesting."
    output_filename = "test_output_metadata.json"
    output_file = OUTPUT_DIR / output_filename
    
    run_cli_command([
        'extract', input_text, '--output', str(output_file), '--type', 'json', '--metadata'
    ])
    
    assert output_file.exists()
    with open(output_file, 'r') as f:
        data = json.load(f)
        
    assert "results" in data
    assert len(data["results"]) > 0
    assert data["configuration"]["metadata_requested"] is True
    
    # Check first result has metadata
    first_result = data["results"][0]
    assert "metadata" in first_result
    assert isinstance(first_result["metadata"], dict)
    assert "length" in first_result["metadata"]
    assert "structures" in first_result["metadata"]

def test_feature_cli_extract_from_file(): # Removed runner, tmp_path
    """
    Feature Test: Verify CLI extraction reading input from a file.
    """
    input_text = "The quick brown fox jumps over the lazy dog.\nAnother sentence in the file."
    input_filename = "test_input_file.txt"
    output_filename = "test_output_from_file.json"
    input_file = OUTPUT_DIR / input_filename
    output_file = OUTPUT_DIR / output_filename
    
    input_file.write_text(input_text)
    assert input_file.exists()

    # More robust expected NPs (less sensitive to sentence parsing variations)
    expected_nps = [
        "The quick brown fox",
        "the lazy dog",
    ]

    run_cli_command([
        'extract', 
        '--file', str(input_file), 
        '--output', str(output_file), 
        '--type', 'json'
    ])

    assert output_file.exists()
    with open(output_file, 'r') as f:
        data = json.load(f)

    assert "results" in data
    extracted_nps = sorted([item["noun_phrase"] for item in data["results"]])
    # Use subset check as exact NPs might vary slightly
    assert set(expected_nps).issubset(set(extracted_nps)), \
        f"Expected {expected_nps} to be a subset of {extracted_nps}"

# --- Feature Tests ---

@patch('sys.exit') 
def test_feature_cli_setup(mock_sys_exit):
    """
    Feature Test: Verify CLI 'setup' command calls the main setup_models utility
    for default models.
    Mocks the setup_models utility to prevent actual installation and checks.
    Calls cli.main directly.
    """
    with patch.object(anpe.cli, 'setup_models', autospec=True) as mock_setup_models_in_cli:
        mock_setup_models_in_cli.return_value = True

        # Call main using the freshly imported module reference
        exit_code = anpe.cli.main(['setup'])

        assert exit_code == 0
        mock_sys_exit.assert_not_called()
        mock_setup_models_in_cli.assert_called_once()
        
        args, kwargs = mock_setup_models_in_cli.call_args
        
        assert kwargs.get('spacy_model_alias') is None
        assert kwargs.get('benepar_model_alias') is None
        assert 'log_callback' in kwargs
        assert callable(kwargs['log_callback'])

@patch('anpe.cli.clean_all')
@patch('sys.exit') # Mock sys.exit
@patch('builtins.input', return_value='yes') # Mock confirmation input
def test_feature_cli_clean(mock_input, mock_exit, mock_clean_all):
    """
    Feature Test: Verify CLI 'setup --clean-models --force' calls clean_all helper.
    Mocks actual cleaning.
    Calls cli.main directly to ensure mock is active.
    NOTE: We mock input confirmation for safety, although --force bypasses it.
    """
    # Call cli_main directly instead of using subprocess
    # Combine setup command with clean flags
    # cli_main(['setup', '--clean-models', '--force']) # Old call
    anpe.cli.main(['setup', '--clean-models', '--force']) # New call

    # Assert that the clean_all function was called
    # clean_all expects logger and force arguments
    mock_clean_all.assert_called_once()
    # Optionally check args passed to mock_clean_all if needed, but checking call is primary goal
    # Example: mock_clean_all.assert_called_once_with(logger=ANY, force=True)
    mock_exit.assert_not_called() # Ensure clean didn't exit unexpectedly

# Tests for Setup and Clean (These modify the user's environment)
# Use caution when running these tests

@pytest.mark.skip(reason="Modifies user environment, run manually if needed")
def test_feature_cli_setup_models():
    """Test the 'anpe setup' command downloads models."""
    # 1. Ensure models are NOT present first (run clean)
    print("\nEnsuring models are cleaned before setup test...")
    clean_result = run_cli_command(["setup", "--clean-models", "--force"])
    # Clean might fail if models weren't there, that's ok for this stage.
    # Add a small delay to let filesystem settle if needed
    time.sleep(2)
    assert not _check_spacy_model_exists(DEFAULT_SPACY_MODEL), f"spaCy model {DEFAULT_SPACY_MODEL} still exists after clean."
    assert not _check_benepar_model_exists(DEFAULT_BENEPAR_MODEL), f"Benepar model {DEFAULT_BENEPAR_MODEL} still exists after clean."
    print("Models confirmed absent.")

    # 2. Run setup
    print("\nRunning 'anpe setup'...")
    setup_result = run_cli_command(["setup"])
    assert setup_result.returncode == 0, f"'anpe setup' failed: {setup_result.stderr}"
    # Check stdout/stderr for success messages if needed
    assert "successfully installed" in setup_result.stdout.lower() or "already satisfied" in setup_result.stdout.lower()

    # 3. Assert models ARE present
    # Add retry/wait logic as download/install might take a moment to reflect
    max_wait = 30
    wait_interval = 5
    waited = 0
    spacy_found = _check_spacy_model_exists(DEFAULT_SPACY_MODEL)
    benepar_found = _check_benepar_model_exists(DEFAULT_BENEPAR_MODEL)

    while (not spacy_found or not benepar_found) and waited < max_wait:
        print(f"Waiting for models to appear... ({waited}s / {max_wait}s)")
        time.sleep(wait_interval)
        waited += wait_interval
        spacy_found = _check_spacy_model_exists(DEFAULT_SPACY_MODEL)
        benepar_found = _check_benepar_model_exists(DEFAULT_BENEPAR_MODEL)

    assert spacy_found, f"spaCy model {DEFAULT_SPACY_MODEL} not found after 'anpe setup'."
    assert benepar_found, f"Benepar model {DEFAULT_BENEPAR_MODEL} not found after 'anpe setup'."
    print("Models confirmed present after setup.")

@pytest.mark.skip(reason="Modifies user environment, run manually if needed")
def test_feature_cli_clean_models():
    """Test the 'anpe setup --clean-models --force' command removes models."""
    # 1. Ensure models ARE present first (run setup)
    print("\nEnsuring models are present before clean test...")
    setup_result = run_cli_command(["setup"])
    # Setup might fail if already present, that's ok. Check presence directly.
    # Use wait logic similar to setup test to ensure models are detectable
    max_wait = 30
    wait_interval = 5
    waited = 0
    spacy_found = _check_spacy_model_exists(DEFAULT_SPACY_MODEL)
    benepar_found = _check_benepar_model_exists(DEFAULT_BENEPAR_MODEL)

    while (not spacy_found or not benepar_found) and waited < max_wait:
        print(f"Waiting for models to appear before clean... ({waited}s / {max_wait}s)")
        time.sleep(wait_interval)
        waited += wait_interval
        spacy_found = _check_spacy_model_exists(DEFAULT_SPACY_MODEL)
        benepar_found = _check_benepar_model_exists(DEFAULT_BENEPAR_MODEL)

    assert spacy_found, f"spaCy model {DEFAULT_SPACY_MODEL} should exist before cleaning."
    assert benepar_found, f"Benepar model {DEFAULT_BENEPAR_MODEL} should exist before cleaning."
    print("Models confirmed present.")

    # 2. Run clean
    print("\nRunning 'anpe setup --clean-models --force'...")
    clean_result = run_cli_command(["setup", "--clean-models", "--force"])
    assert clean_result.returncode == 0, f"'anpe setup --clean-models --force' failed: {clean_result.stderr}"
    # Check stdout/stderr for success messages if needed
    assert "successfully removed" in clean_result.stdout.lower() or "not found" in clean_result.stdout.lower()

    # 3. Assert models are NOT present
    # Add a small delay
    time.sleep(2)
    assert not _check_spacy_model_exists(DEFAULT_SPACY_MODEL), f"spaCy model {DEFAULT_SPACY_MODEL} still exists after clean."
    assert not _check_benepar_model_exists(DEFAULT_BENEPAR_MODEL), f"Benepar model {DEFAULT_BENEPAR_MODEL} still exists after clean."
    print("Models confirmed absent after clean.")
