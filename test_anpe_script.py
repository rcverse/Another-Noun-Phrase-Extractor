import anpe
from pathlib import Path
import datetime
import shutil
import logging

# --- Configuration ---
INPUT_TEXT_FILE = Path("complex_test_text.txt")
OUTPUT_DIR = Path("./anpe_test_outputs_v2")
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Helper Function ---
def run_test_config(config_name: str, extractor_config: dict, extract_params: dict):
    """Runs extraction and exports for a given configuration."""
    print(f"\n--- Running Test: {config_name} ---")
    print(f"Extractor Config: {extractor_config}")
    print(f"Extract Params: {extract_params}")

    try:
        # Create extractor with specific config
        extractor = anpe.ANPEExtractor(extractor_config)

        # Read input text
        with open(INPUT_TEXT_FILE, 'r', encoding='utf-8') as f:
            text = f.read()

        # Define base output name
        base_output_name = f"{config_name}_{TIMESTAMP}"

        # Export to all three formats
        for fmt in ["txt", "csv", "json"]:
            output_file = OUTPUT_DIR / f"{base_output_name}.{fmt}"
            print(f"Exporting to {fmt.upper()} -> {output_file}")
            try:
                extractor.export(
                    text,
                    format=fmt,
                    output=str(output_file),
                    **extract_params
                )
                print(f" -> Success.")
            except Exception as e:
                print(f" -> FAILED exporting {fmt}: {e}")

    except Exception as e:
        print(f" -> FAILED creating extractor or reading input for {config_name}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # Configure basic logging for the test script
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Ensure input file exists
    if not INPUT_TEXT_FILE.exists():
        print(f"Error: Input file '{INPUT_TEXT_FILE}' not found.")
        print("Please create it with the complex test text.")
        exit(1)

    # Clean and Create output directory
    if OUTPUT_DIR.exists():
        print(f"Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    print(f"Creating output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True)

    # === Test Configurations ===

    # Config 1: Default behavior + Metadata
    run_test_config(
        config_name="default_meta",
        extractor_config={}, # Minimal config - removed log_level
        extract_params={"metadata": True, "include_nested": False}
    )

    # Config 2: Include Nested + Metadata
    run_test_config(
        config_name="nested_meta",
        extractor_config={}, # Removed log_level
        extract_params={"metadata": True, "include_nested": True}
    )

    # Config 3: Filtering (Length, No Pronouns, Specific Structures)
    run_test_config(
        config_name="filtered",
        extractor_config={
            # Removed log_level
            "min_length": 3,
            "max_length": 7,
            "accept_pronouns": False,
            "structure_filters": ["possessive", "relative_clause", "appositive"]
        },
        extract_params={"metadata": True, "include_nested": True} # Need metadata for filters
    )

    # Config 4: With Newline Breaks (Default)
    run_test_config(
        config_name="with_newline_breaks",
        extractor_config={
            # Removed log_level
            "newline_breaks": True # Explicitly True (or rely on default)
        },
        extract_params={"metadata": True, "include_nested": False}
    )

    # Config 5: No Newline Breaks
    run_test_config(
        config_name="no_newline_breaks",
        extractor_config={
            # Removed log_level
            "newline_breaks": False
        },
        extract_params={"metadata": False, "include_nested": False}
    )

    # === Model Comparison Tests ===
    # Use consistent extract params for easier comparison across models
    model_compare_extract_params = {"metadata": True, "include_nested": True}

    # Config 6: SpaCy Large, Benepar Default
    run_test_config(
        config_name="spacy_lg_benepar_default",
        extractor_config={
            # Removed log_level
            "spacy_model": "lg", 
            "benepar_model": "default"
        },
        extract_params=model_compare_extract_params
    )

    # Config 7: SpaCy Transformer, Benepar Default
    run_test_config(
        config_name="spacy_trf_benepar_default",
        extractor_config={
            # Removed log_level
            "spacy_model": "trf",
            "benepar_model": "default"
        },
        extract_params=model_compare_extract_params
    )
    
    # Config 8: SpaCy Default (MD), Benepar Large
    run_test_config(
        config_name="spacy_md_benepar_large",
        extractor_config={
            # Removed log_level
            "spacy_model": "md", # Explicitly specify default md
            "benepar_model": "large"
        },
        extract_params=model_compare_extract_params
    )

    # Config 9: SpaCy Large, Benepar Large
    run_test_config(
        config_name="spacy_lg_benepar_large",
        extractor_config={
            # Removed log_level
            "spacy_model": "lg",
            "benepar_model": "large"
        },
        extract_params=model_compare_extract_params
    )
    

    print(f"\n--- Testing Complete ---")
    print(f"Please manually inspect the generated files in: {OUTPUT_DIR.resolve()}") 