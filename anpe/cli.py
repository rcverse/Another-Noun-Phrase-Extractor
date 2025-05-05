import argparse
import sys
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
import datetime

# --- Standard Logging Setup ---
import logging
logger = logging.getLogger(__name__) # Logger for CLI messages
# --- End Standard Logging ---

import anpe 
from anpe import ANPEExtractor
from anpe.utils.setup_models import (  # Import specific functions
    setup_models,
    check_all_models_present,
    check_spacy_model,
    check_benepar_model,
    # Import maps and defaults for choices/validation
    SPACY_MODEL_MAP,
    BENEPAR_MODEL_MAP,
    DEFAULT_SPACY_ALIAS,
    DEFAULT_BENEPAR_ALIAS
)
# Import cleaning utility
from anpe.utils.clean_models import clean_all, SPACY_MODEL_MAP as CLEAN_SPACY_MAP, BENEPAR_MODEL_MAP as CLEAN_BENEPAR_MAP

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ANPE: Another Noun Phrase Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract noun phrases from text")
    
    # Input options
    input_group = extract_parser.add_argument_group("Input Options")
    input_group.add_argument("text", nargs="?", 
                        help="Text to process (if not provided, reads from STDIN)")
    input_group.add_argument("-f", "--file", 
                        help="Input file path (instead of direct text)")
    input_group.add_argument("-d", "--dir",
                        help="Input directory containing text files to process")
    
    # Processing options
    process_group = extract_parser.add_argument_group("Processing Options")
    process_group.add_argument("--metadata", action="store_true",
                        help="Include metadata (length and structure analysis)")
    process_group.add_argument("--nested", action="store_true",
                        help="Extract nested noun phrases")
    process_group.add_argument("--min-length", type=int, 
                        help="Minimum NP length in tokens")
    process_group.add_argument("--max-length", type=int, 
                        help="Maximum NP length in tokens")
    process_group.add_argument("--no-pronouns", action="store_true", 
                        help="Exclude pronouns from results")
    process_group.add_argument("--no-newline-breaks", action="store_true",
                        help="Don't treat newlines as sentence boundaries")
    process_group.add_argument("--structures", type=str,
                        help="Comma-separated list of structure patterns to include")
    process_group.add_argument(
        "--spacy-model", 
        choices=list(SPACY_MODEL_MAP.keys()), # Use keys from the map as choices
        default=None, # Let extractor handle default/auto-detection
        help="Specify spaCy model alias to USE for extraction (e.g., md, lg). Overrides auto-detection."
    )
    process_group.add_argument(
        "--benepar-model",
        choices=list(BENEPAR_MODEL_MAP.keys()), # Use keys from the map as choices
        default=None, # Let extractor handle default/auto-detection
        help="Specify Benepar model alias to USE for extraction (e.g., default, large). Overrides auto-detection."
    )
    
    # Output options
    output_group = extract_parser.add_argument_group("Output Options")
    output_group.add_argument("-o", "--output", 
                        help="Output file path or directory. If a directory, a timestamped file is created.")
    output_group.add_argument("-t", "--type", choices=["txt", "csv", "json"], 
                        default="txt", help="Output format (used for filename generation if output is a directory, or content formatting)")
    
    # Logging options
    log_group = extract_parser.add_argument_group("Logging Options")
    log_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Logging level")
    log_group.add_argument("--log-dir",
                        help="Directory path for log files")
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Display version information")
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup required models or clean existing models')
    setup_group = setup_parser.add_argument_group("Installation Options")
    setup_group.add_argument(
        "--spacy-model",
        choices=list(SPACY_MODEL_MAP.keys()), # Use keys from the map as choices
        default=None, # Make default None so we can check if user specified it
        help="Specify the spaCy model alias to install (e.g., sm, md, lg, trf). If not specified, installs default."
    )
    setup_group.add_argument(
        "--benepar-model",
        choices=list(BENEPAR_MODEL_MAP.keys()), # Use keys from the map as choices
        default=None, # Make default None
        help="Specify the Benepar model alias to install (e.g., default, large). If not specified, installs default."
    )
    
    # Mutually exclusive group for clean vs. install
    clean_group = setup_parser.add_argument_group("Cleanup Options")
    clean_group.add_argument(
        "--clean-models",
        action="store_true",
        help="Remove all known ANPE-related models (spaCy and Benepar). Mutually exclusive with specific model installation."
    )
    clean_group.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force removal without user confirmation when using --clean-models."
    )
    
    # Logging options (apply to both setup and clean)
    setup_log_group = setup_parser.add_argument_group("Logging Options")
    setup_log_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Logging level")
    setup_log_group.add_argument("--log-dir",
                        help="Directory path for log files")
    
    return parser.parse_args(args)

def read_input_text(args: argparse.Namespace) -> str:
    """Read input text from the specified source."""
    if args.file:
        logger.debug(f"Reading input from file: {args.file}")
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            raise FileNotFoundError(f"Could not read input file: {args.file}")
    
    elif args.text:
        logger.debug("Using text provided via command line")
        return args.text
    
    else:
        logger.debug("Reading input from standard input")
        text = sys.stdin.read()
        if not text:
            logger.error("No input text provided")
            raise ValueError("No input text provided")
        return text

def create_extractor(args: argparse.Namespace) -> ANPEExtractor:
    """Create an ANPEExtractor with the specified configuration."""
    config = {
        "accept_pronouns": not args.no_pronouns,
        "newline_breaks": not args.no_newline_breaks
    }
    
    # Add model overrides if specified via CLI for extract command
    if hasattr(args, 'spacy_model') and args.spacy_model:
        # Map alias to actual name if needed (though extractor also does this)
        actual_spacy_model = SPACY_MODEL_MAP.get(args.spacy_model, args.spacy_model)
        config["spacy_model"] = actual_spacy_model
        logger.debug(f"CLI overriding spaCy model for extraction: {actual_spacy_model}")
        
    if hasattr(args, 'benepar_model') and args.benepar_model:
        # Map alias to actual name if needed
        actual_benepar_model = BENEPAR_MODEL_MAP.get(args.benepar_model, args.benepar_model)
        config["benepar_model"] = actual_benepar_model
        logger.debug(f"CLI overriding Benepar model for extraction: {actual_benepar_model}")
    
    # Add other configurations
    if args.min_length is not None:
        config["min_length"] = args.min_length
    if args.max_length is not None:
        config["max_length"] = args.max_length
    if args.structures:
        config["structure_filters"] = [s.strip() for s in args.structures.split(',')]
    
    logger.debug(f"Creating extractor with config (excluding logging): {config}")
    # Pass filtered config (extractor's __init__ now ignores logger keys anyway)
    return ANPEExtractor(config=config)

def process_text(text: str, output: Optional[str], format: str, 
                metadata: bool, nested: bool, extractor: ANPEExtractor) -> None:
    """Process text: Extract and either print to stdout or export using the extractor's export method."""
    try:
        if output:
            # Let the extractor handle both extraction and export logic
            logger.info(f"Extracting and exporting results to: {output} in {format} format")
            # The extractor.export method handles path logic and calls extract internally
            exported_path = extractor.export(
                text=text, # Pass the raw text
                format=format,
                output=output,
                metadata=metadata,
                include_nested=nested
            )
            logger.info(f"Export complete: {exported_path}")
        else:
            # Extract and print to stdout
            logger.info("Extracting results and printing to stdout")
            result_data = extractor.extract(
                text, 
                metadata=metadata,
                include_nested=nested
            )
            print_result_to_stdout(result_data)
            
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}", exc_info=True) # Log traceback
        raise # Re-raise after logging

def process_file(input_file: str, output: Optional[str], format: str,
                metadata: bool, nested: bool, extractor: ANPEExtractor) -> None:
    """Process a single file by reading its content and calling process_text."""
    logger.info(f"Processing file: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        # Pass to process_text which handles extraction and output logic
        process_text(text, output, format, metadata, nested, extractor)
    except FileNotFoundError as e:
        logger.error(f"Input file not found: {input_file}") 
        # Don't re-raise if part of batch
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}", exc_info=True) # Log traceback
        # Don't re-raise if part of batch

def process_directory(input_dir: str, output: Optional[str], format: str,
                     metadata: bool, nested: bool, extractor: ANPEExtractor) -> None:
    """Process all text files in a directory."""
    logger.info(f"Processing directory: {input_dir}")
    processed_files = 0
    
    # Determine if output is a directory upfront for logging/logic
    output_is_dir = False
    if output:
        output_path = Path(output)
        if output_path.is_dir() or not output_path.suffix:
            output_is_dir = True
            output_path.mkdir(parents=True, exist_ok=True) # Ensure dir exists
            logger.debug(f"Output directory confirmed: {output_path}")
        elif output_path.parent:
             # Single output file specified for potentially multiple inputs
             logger.warning(f"Output '{output}' is a file path, but input is a directory. Output will be overwritten by the last processed file.")
             output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent dir exists
        else:
             # Output file in current dir? Ensure parent exists (which is '.')
             logger.warning(f"Output '{output}' is a file path relative to current dir, but input is a directory. Output will be overwritten by the last processed file.")
    
    try:
        for root, _, files in os.walk(input_dir):
            for file in files:
                # Consider other text file extensions if needed
                if file.lower().endswith(('.txt')): 
                    input_filepath = Path(root) / file
                    logger.info(f"Processing file: {input_filepath}")
                    try:
                        with open(input_filepath, 'r', encoding='utf-8') as f:
                            text = f.read()

                        # Call process_text to handle logic for this file's text
                        # process_text will call extractor.export or extractor.extract+print
                        process_text(
                            text,
                            output, # Pass the original output arg (extractor.export handles path logic)
                            format,
                            metadata,
                            nested,
                            extractor
                        )
                        processed_files += 1
                            
                    except FileNotFoundError:
                        logger.error(f"Could not find file during directory walk: {input_filepath}")
                        continue # Skip this file
                    except Exception as e:
                        logger.error(f"Error processing file {input_filepath}: {str(e)}", exc_info=True)
                        continue # Skip this file
        
        logger.info(f"Finished processing directory. Processed {processed_files} file(s).")

    except Exception as e:
        logger.error(f"Error walking directory {input_dir}: {str(e)}", exc_info=True)
        raise

def print_result_to_stdout(data: Dict) -> None:
    """Print extraction results to stdout."""
    print("--- ANPE Noun Phrase Extraction Results ---")
    # Read timestamp from top level
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    # Read output flags from configuration
    config = data.get('configuration', {})
    print(f"Output includes Nested NPs: {config.get('nested_requested', False)}")
    print(f"Output includes Metadata: {config.get('metadata_requested', False)}")
    print("-------------------------------------------") # Separator
    print()
    
    if "results" in data and data["results"]:
        for np_item in data["results"]:
            print_np_to_stdout(np_item, 0)
    else:
        print("No noun phrases extracted.")

def print_np_to_stdout(np_item: Dict, level: int = 0) -> None:
    """Print a noun phrase to stdout."""
    bullet = "•" if level == 0 else "◦"
    indent = "  " * level
    
    print(f"{indent}{bullet} [{np_item['id']}] {np_item['noun_phrase']}")
    
    if "metadata" in np_item:
        metadata = np_item["metadata"]
        if "length" in metadata:
            print(f"{indent}  Length: {metadata['length']}")
        if "structures" in metadata:
            structures_str = ", ".join(metadata['structures']) if isinstance(metadata['structures'], list) else metadata['structures']
            print(f"{indent}  Structures: [{structures_str}]")
    
    for child in np_item.get("children", []):
        print_np_to_stdout(child, level + 1)
    
    if level == 0:
        print()

def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parsed_args = parse_args(args)
    
    # --- Setup Logging using standard logging --- 
    log_level_str = parsed_args.log_level if hasattr(parsed_args, 'log_level') else 'INFO'
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    log_handlers = []
    # Console Handler (always add)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    # Set console handler level directly - basicConfig level sets root logger level
    console_handler.setLevel(numeric_level) 
    log_handlers.append(console_handler)
    
    # File Handler (optional)
    log_filepath = None
    if hasattr(parsed_args, 'log_dir') and parsed_args.log_dir:
        try:
            log_dir = Path(parsed_args.log_dir).resolve()
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filepath = log_dir / f"anpe_cli_{timestamp}.log"
            
            file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
            file_handler.setLevel(logging.DEBUG) # File handler always captures DEBUG and up
            log_handlers.append(file_handler)
            print(f"[CLI Setup] Logging detailed output to: {log_filepath}", file=sys.stderr)
        except Exception as e:
            print(f"[CLI Setup] Warning: Could not configure file logging: {e}", file=sys.stderr)
            # Continue with console logging
    
    # Configure the root logger
    # Set level to the lowest level needed by any handler (DEBUG if file, else numeric_level)
    root_level = logging.DEBUG if log_filepath else numeric_level
    logging.basicConfig(level=root_level, handlers=log_handlers, force=True)
    # force=True is needed if basicConfig might have been called implicitly before (e.g., by a dependency)
    
    # Now get the logger instance for this module *after* configuration
    logger = logging.getLogger(__name__) # Re-get logger after basicConfig
    # --- End Logging Setup ---
    
    logger.info(f"Executing command: {parsed_args.command}")
    logger.debug(f"Parsed arguments: {parsed_args}") # Log parsed args at debug level
    
    try:
        if parsed_args.command == "version":
            print(f"ANPE Version: {anpe.__version__}")
        
        elif parsed_args.command == "extract":
            # Read input FIRST
            try:
                input_text = read_input_text(parsed_args)
            except (FileNotFoundError, ValueError) as e:
                logger.error(str(e))
                return 1 # Exit if input reading fails
                
            # Create extractor only AFTER successful input reading
            try:
                extractor = create_extractor(parsed_args)
            except Exception as e:
                logger.error(f"Error creating extractor: {str(e)}")
                return 1 # Exit if extractor creation fails
            
            # Now process the text
            process_text(
                text=input_text, 
                output=parsed_args.output, 
                format=parsed_args.type, 
                metadata=parsed_args.metadata, 
                nested=parsed_args.nested, 
                extractor=extractor
            )

        elif parsed_args.command == "setup":
            # Define callback using standard logging
            def cli_log_callback(message: str):
                level = "DEBUG" if "Downloading" in message or "Extracting" in message else "INFO"
                setup_logger = logging.getLogger("anpe.setup") # Use specific name
                if level == "DEBUG":
                    setup_logger.debug(message)
                else:
                    setup_logger.info(message)
            
            if parsed_args.clean_models:
                logger.info("Executing model cleanup...")
                print("--- Executing Model Cleanup ---", file=sys.stderr)
                # Confirmation logic moved into clean_all, handle abort based on return?
                # Or keep simple confirmation here?
                confirmed = False
                if parsed_args.force:
                    confirmed = True
                    logger.debug("--force flag used, skipping confirmation.")
                else:
                    # Check if running interactively
                    if sys.stdin.isatty():
                        try:
                            confirm = input("Remove all detected models? This cannot be undone. [y/N]: ")
                            if confirm.lower() == 'y':
                                confirmed = True
                        except EOFError: 
                            logger.warning("Non-interactive input detected, confirmation skipped. Use --force to bypass.")
                    else:
                         logger.warning("Non-interactive session detected, confirmation skipped. Use --force to bypass.")
                         
                if confirmed:
                    try:
                        # Pass the logger instance to clean_all
                        clean_all(logger=logger, force=parsed_args.force)
                        print("--- Model Cleanup Finished Successfully. ---", file=sys.stderr)
                        logger.info("Model cleanup finished successfully.")
                    except Exception as e:
                        logger.error(f"Cleanup failed: {str(e)}", exc_info=True)
                        print(f"Error: {str(e)}", file=sys.stderr)
                        return 1 # Error during cleanup
                else:
                    logger.info("Cleanup aborted.") # Changed from ERROR to INFO
                    print("Cleanup aborted.", file=sys.stderr) # Also print to stderr
                    return 1 # <<< FIX: Ensure non-zero return on abort

            else: # Normal setup
                spacy_alias = parsed_args.spacy_model
                benepar_alias = parsed_args.benepar_model
                
                if not spacy_alias and not benepar_alias:
                    logger.info("No specific models provided, setting up default spaCy (md) and Benepar (default) models.")
                    spacy_alias = DEFAULT_SPACY_ALIAS
                    benepar_alias = DEFAULT_BENEPAR_ALIAS
                elif not spacy_alias:
                    logger.info(f"Setting up default spaCy ({DEFAULT_SPACY_ALIAS}) and specified Benepar ({benepar_alias}).")
                    spacy_alias = DEFAULT_SPACY_ALIAS
                elif not benepar_alias:
                    logger.info(f"Setting up specified spaCy ({spacy_alias}) and default Benepar ({DEFAULT_BENEPAR_ALIAS}).")
                    benepar_alias = DEFAULT_BENEPAR_ALIAS
                else:
                    logger.info(f"Setting up spaCy='{spacy_alias}', Benepar='{benepar_alias}'")
                    
                try:
                    setup_models(
                        spacy_model_alias=spacy_alias, 
                        benepar_model_alias=benepar_alias, 
                        log_callback=cli_log_callback
                    )
                    print(f"--- Setup for spaCy='{spacy_alias}', Benepar='{benepar_alias}' finished successfully. ---", file=sys.stderr)
                except Exception as e:
                    logger.error(f"Setup failed: {str(e)}")
                    print(f"Error: {str(e)}", file=sys.stderr)
                    return 1 # Error during setup
        
        else:
            # Should not happen if argparse is set up correctly
            logger.error(f"Unknown command: {parsed_args.command}")
            return 1
            
    except Exception as e:
        logger.error(f"Unhandled error during command execution: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1 # Failure
    
    logger.debug("CLI command finished successfully.")
    return 0 # Success


if __name__ == '__main__':
    sys.exit(main())