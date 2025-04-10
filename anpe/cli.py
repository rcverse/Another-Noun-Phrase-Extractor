import argparse
import sys
import os
from typing import List, Dict, Optional, Any
from pathlib import Path
import datetime


import anpe 
from anpe import ANPEExtractor
from anpe.utils.anpe_logger import ANPELogger, get_logger
from anpe.utils.setup_models import (  # Import specific functions
    setup_models,
    check_all_models_present,
    check_spacy_model,
    check_benepar_model,
    check_nltk_models
)

# Initialize logger at module level
logger = get_logger("cli")

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
    setup_parser = subparsers.add_parser('setup', help='Setup required models')
    setup_parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO", help="Logging level")
    setup_parser.add_argument("--log-dir",
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
        "log_level": args.log_level,
        "accept_pronouns": not args.no_pronouns,
        "newline_breaks": not args.no_newline_breaks
    }
    
    # Add log directory to config if specified
    if args.log_dir:
        config["log_dir"] = args.log_dir
    
    # Add other configurations
    if args.min_length is not None:
        config["min_length"] = args.min_length
    if args.max_length is not None:
        config["max_length"] = args.max_length
    if args.structures:
        config["structure_filters"] = [s.strip() for s in args.structures.split(',')]
    
    logger.debug(f"Creating extractor with config: {config}")
    return ANPEExtractor(config)

def process_text(text: str, output: Optional[str], format: str, 
                metadata: bool, nested: bool, extractor: ANPEExtractor) -> None:
    """Process text and either print to stdout or export using the extractor."""
    try:
        if output:
            # Let the extractor handle export logic (including path handling)
            logger.info(f"Exporting results to: {output} in {format} format")
            extractor.export(
                text,
                format=format,
                output=output,
                metadata=metadata,
                include_nested=nested
            )
        else:
            # Extract and print to stdout
            logger.info("Extracting results and printing to stdout")
            result = extractor.extract(
                text, 
                metadata=metadata,
                include_nested=nested
            )
            print_result_to_stdout(result)
            
    except Exception as e:
        logger.error(f"Error processing text: {str(e)}")
        raise

def process_file(input_file: str, output: Optional[str], format: str,
                metadata: bool, nested: bool, extractor: ANPEExtractor) -> None:
    """Process a single file."""
    logger.info(f"Processing file: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        # Pass directly to process_text which now handles export or printing
        process_text(text, output, format, metadata, nested, extractor)
    except Exception as e:
        logger.error(f"Error processing file {input_file}: {str(e)}")
        # Optionally re-raise or just log and continue if part of batch processing
        # raise # Uncomment if one file failure should stop the batch

def process_directory(input_dir: str, output: Optional[str], format: str,
                     metadata: bool, nested: bool, extractor: ANPEExtractor) -> None:
    """Process all text files in a directory."""
    logger.info(f"Processing directory: {input_dir}")
    processed_files = 0
    try:
        for root, _, files in os.walk(input_dir):
            for file in files:
                # Consider other text file extensions if needed
                if file.lower().endswith('.txt'): 
                    input_file = os.path.join(root, file)
                    # Note: If output is a directory, extractor.export will handle unique filenames.
                    # If output is a file path, it will be overwritten for each input file. 
                    # This might require further refinement depending on desired batch behavior for file output.
                    # For now, we proceed assuming the user understands this behavior or provides a directory.
                    if output and not Path(output).is_dir() and not output.endswith(os.sep):
                         logger.warning(f"Output '{output}' is a file path. It will be overwritten by each file processed in the directory '{input_dir}'. Provide a directory for unique outputs per input file.")
                    
                    process_file(input_file, output, format, metadata, nested, extractor)
                    processed_files += 1
        
        if processed_files == 0:
            logger.warning(f"No .txt files found in directory: {input_dir}")
        else:
            logger.info(f"Finished processing {processed_files} file(s) from directory: {input_dir}")

    except Exception as e:
        logger.error(f"Error processing directory {input_dir}: {str(e)}")
        raise

def print_result_to_stdout(data: Dict) -> None:
    """Print extraction results to stdout."""
    print("ANPE Noun Phrase Extraction Results")
    print(f"Timestamp: {data['metadata'].get('timestamp')}")
    print(f"Includes Nested NPs: {data['metadata'].get('includes_nested')}")
    print(f"Includes Metadata: {data['metadata'].get('includes_metadata')}")
    print()
    
    for np_item in data["results"]:
        print_np_to_stdout(np_item, 0)

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
    """Main entry point for the CLI."""
    parsed_args = parse_args(args)
    
    # Handle version command immediately
    if parsed_args.command == "version":
        from anpe import __version__
        print(f"ANPE version {__version__}")
        return 0
    
    # Initialize logging
    try:
        # Convert log directory to file path if specified
        log_file = None
        if hasattr(parsed_args, 'log_dir') and parsed_args.log_dir:
            log_dir = Path(parsed_args.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = str(log_dir / f"log_anpe_export_{timestamp}.log")
        
        # Initialize logger with proper configuration
        log_level = parsed_args.log_level if hasattr(parsed_args, 'log_level') else "INFO"
        ANPELogger(
            log_level=log_level, 
            log_file=log_file
        )
        logger = get_logger("cli")
    except Exception as e:
        print(f"Error initializing logger: {e}", file=sys.stderr)
        return 1
    
    logger.debug(f"ANPE CLI started with command: {parsed_args.command}")
    
    try:
        if parsed_args.command == "extract":
            # Create extractor
            extractor = create_extractor(parsed_args)
            
            # Process based on input type
            if parsed_args.dir:
                process_directory(
                    parsed_args.dir,
                    parsed_args.output,
                    parsed_args.type,
                    parsed_args.metadata,
                    parsed_args.nested,
                    extractor
                )
            elif parsed_args.file:
                process_file(
                    parsed_args.file,
                    parsed_args.output,
                    parsed_args.type,
                    parsed_args.metadata,
                    parsed_args.nested,
                    extractor
                )
            else:
                # Read from stdin or direct text
                input_text = read_input_text(parsed_args)
                process_text(
                    input_text, 
                    parsed_args.output,
                    parsed_args.type, 
                    parsed_args.metadata, 
                    parsed_args.nested, 
                    extractor
                )
            
            return 0
        
        elif parsed_args.command == "setup":
            # First check if models are already present
            if check_all_models_present():
                logger.info("All required models are already present. No installation needed.")
                return 0
                
            # If not all models are present, show current status
            logger.info("Checking current model status...")
            results = {
                "spacy": check_spacy_model(),
                "benepar": check_benepar_model(),
                "nltk": check_nltk_models()
            }
            for model, status in results.items():
                logger.info(f"{model}: {'Present' if status else 'Missing'}")
            
            # Run the setup process
            logger.info("Starting model installation process...")
            if setup_models():
                logger.info("All models installed successfully")
                return 0
            else:
                logger.error("Failed to install some models")
                return 1
        
        else:
            logger.error(f"Unknown command: {parsed_args.command}")
            print(f"Unknown command: {parsed_args.command}. Use --help for usage information.", file=sys.stderr)
            return 1
    
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())