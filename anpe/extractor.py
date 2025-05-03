from typing import Dict, List, Optional, Any
import spacy
import benepar
import nltk
from nltk.tree import Tree
import datetime
import os
import warnings
import spacy.cli
from pathlib import Path
import sys


from anpe.config import DEFAULT_CONFIG
from anpe.utils.anpe_logger import get_logger, ANPELogger
from anpe.utils.setup_models import setup_models
from anpe.utils.export import ANPEExporter
from anpe.utils.model_finder import (
    find_installed_spacy_models,
    find_installed_benepar_models,
    select_best_spacy_model,
    select_best_benepar_model
)
from anpe.utils.setup_models import SPACY_MODEL_MAP, BENEPAR_MODEL_MAP


class ANPEExtractor:
    """Main extractor class for noun phrase extraction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extractor.

        Args:
            config: Configuration dictionary with options:
                - log_level: Logging level
                - log_dir: Directory path for log files (optional)
                - accept_pronouns: Whether to include pronouns as valid NPs
                - min_length: Minimum token length for NPs
                - max_length: Maximum token length for NPs
                - newline_breaks: Whether to treat newlines as sentence boundaries
                - structure_filters: List of structure types to include (empty=all)
        """
        
        # Initialize default config
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)

        # Initialize configuration - logging
        try:
            # Convert log directory to file path if specified
            log_file = None
            if 'log_dir' in self.config and self.config['log_dir']:
                log_dir = Path(self.config['log_dir'])
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = str(log_dir / f"log_anpe_export_{timestamp}.log")

            # Initialize logger with configuration
            ANPELogger(
                log_level=self.config.get('log_level', 'INFO'),
                log_file=log_file
            )
            self.logger = get_logger('extractor')

        except Exception as e:
            print(f"Error initializing logger: {e}", file=sys.stderr)
            raise

        # Initialize other user configuration
        self.logger.info("Initializing Config...")
        self.min_length = self.config.get("min_length")
        self.max_length = self.config.get("max_length")
        self.accept_pronouns = self.config.get("accept_pronouns")
        self.structure_filters = self.config.get("structure_filters", [])
        self.newline_breaks = self.config.get("newline_breaks")

        # Add environment variables to suppress advisory warnings
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Suppress specific PyTorch warnings
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributions")
        warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values`")
        warnings.filterwarnings("ignore", message=".*default legacy behaviour.*")
        warnings.filterwarnings("ignore", message=".*EncoderDecoderCache.*")

        self.logger.info("Initializing ANPEExtractor")

        # --- Determine and Load Models ---
        try:
            # Determine spaCy model to use
            spacy_model_to_use = self.config.get("spacy_model")
            if spacy_model_to_use:
                # Validate if the user provided a known model name/alias
                if spacy_model_to_use not in SPACY_MODEL_MAP:
                     self.logger.warning(f"Specified spaCy model '{spacy_model_to_use}' is not a recognized alias/name. Attempting to use it anyway.")
                     # Map might not be exhaustive, let spacy.load handle unknown models later
                else:
                     # If it's an alias, map it to the actual name
                     spacy_model_to_use = SPACY_MODEL_MAP.get(spacy_model_to_use, spacy_model_to_use)
                self.logger.info(f"Using specified spaCy model: {spacy_model_to_use}")
            else:
                self.logger.debug("No spaCy model specified in config, attempting auto-detection...")
                installed_spacy = find_installed_spacy_models()
                spacy_model_to_use = select_best_spacy_model(installed_spacy)
                if spacy_model_to_use:
                    self.logger.info(f"Auto-detected best available spaCy model: {spacy_model_to_use}")
                else:
                    # Fallback to default if none detected
                    spacy_model_to_use = SPACY_MODEL_MAP['md'] # Default: en_core_web_md
                    self.logger.warning(f"Could not auto-detect any installed spaCy models. Falling back to default: {spacy_model_to_use}")
            # Store the decided model back in config for reference
            self.config["spacy_model"] = spacy_model_to_use

            # Determine Benepar model to use
            benepar_model_to_use = self.config.get("benepar_model")
            if benepar_model_to_use:
                 # Validate if the user provided a known model name/alias
                if benepar_model_to_use not in BENEPAR_MODEL_MAP:
                     self.logger.warning(f"Specified Benepar model '{benepar_model_to_use}' is not a recognized alias/name. Attempting to use it anyway.")
                else:
                     # Map alias if necessary
                     benepar_model_to_use = BENEPAR_MODEL_MAP.get(benepar_model_to_use, benepar_model_to_use)
                self.logger.info(f"Using specified Benepar model: {benepar_model_to_use}")
            else:
                self.logger.debug("No Benepar model specified in config, attempting auto-detection...")
                installed_benepar = find_installed_benepar_models()
                benepar_model_to_use = select_best_benepar_model(installed_benepar)
                if benepar_model_to_use:
                    self.logger.info(f"Auto-detected best available Benepar model: {benepar_model_to_use}")
                else:
                    # Fallback to default
                    benepar_model_to_use = BENEPAR_MODEL_MAP['default'] # Default: benepar_en3
                    self.logger.warning(f"Could not auto-detect any installed Benepar models. Falling back to default: {benepar_model_to_use}")
            # Store the decided model back in config
            self.config["benepar_model"] = benepar_model_to_use
            # Store the actual model name determined for loading
            self._loaded_benepar_model_name = benepar_model_to_use
            
            # --- Check spacy-transformers dependency if needed ---
            spacy_model_name_to_load = self.config['spacy_model']
            if spacy_model_name_to_load.endswith('_trf'):
                self.logger.debug(f"Transformer model '{spacy_model_name_to_load}' selected. Checking for 'spacy-transformers' library...")
                try:
                    import spacy_transformers
                    self.logger.debug("'spacy-transformers' library found.")
                except ImportError:
                    self.logger.critical(f"The spaCy model '{spacy_model_name_to_load}' requires the 'spacy-transformers' library, but it is not installed.")
                    self.logger.critical(f"Please install it using: pip install 'spacy[transformers]'")
                    self.logger.critical("Alternatively, run the setup command: python -m anpe.cli setup --spacy-model <your_trf_model_alias> --benepar-model <benepar_alias>")
                    raise RuntimeError(f"Missing required library 'spacy-transformers' for model '{spacy_model_name_to_load}'.")
            
            # --- Load Models --- 
            # Load the determined spaCy model
            self.logger.info(f"Loading spaCy model: {self.config['spacy_model']}")
            try:
                self.nlp = spacy.load(self.config['spacy_model'])
                self.logger.info("spaCy model loaded successfully.")
            except OSError as e:
                 self.logger.error(f"Failed to load spaCy model '{self.config['spacy_model']}'. Is it installed? Error: {e}")
                 # Add specific hint for transformer models
                 if self.config['spacy_model'].endswith('_trf'):
                     self.logger.error("Ensure the 'spacy-transformers' library is also installed: pip install 'spacy[transformers]'")
                 # Attempt setup *only* if auto-detection failed and we used the default fallback
                 if not self.config.get("spacy_model") and self.config["spacy_model"] == SPACY_MODEL_MAP['md']:
                     self.logger.info("Attempting to install default spaCy model...")
                     if setup_models(spacy_model_alias='md', benepar_model_alias='default'): # Attempt default setup
                          self.logger.info("Default models installed. Re-initializing extractor...")
                          self.__init__(config) # Re-run init
                          return # Exit current init call
                     else:
                          self.logger.critical("Failed to install default models after loading error. Cannot proceed.")
                          raise RuntimeError(f"Failed to load or install required spaCy model '{self.config['spacy_model']}'") from e
                 else:
                      # If user specified a model or auto-detect found something else that failed to load
                      self.logger.critical(f"Specified/detected spaCy model '{self.config['spacy_model']}' could not be loaded. Please ensure it is installed correctly.")
                      raise RuntimeError(f"Failed to load required spaCy model '{self.config['spacy_model']}'") from e


            # Initialize spaCy configuration - adding the sentencizer to the pipeline
            if "sentencizer" not in self.nlp.pipe_names:
                self.logger.debug("Adding sentencizer to spaCy pipeline")
                self.nlp.add_pipe("sentencizer", before="parser")
            sentencizer = self.nlp.get_pipe("sentencizer")

            # Initialize spaCy configuration - apply newline breaking control based on configuration
            if self.newline_breaks:
                self.logger.debug("Configuring newlines as sentence boundaries")
                sentencizer.punct_chars.add('\n')
            else:
                self.logger.info("Configuring newlines to NOT be treated as sentence boundaries")
                if '\n' in sentencizer.punct_chars:
                    sentencizer.punct_chars.remove('\n')

            # --- Add Benepar to spaCy pipeline ---
            # (Replaces the standalone Benepar parser loading)
            self.logger.info(f"Adding Benepar component '{self.config['benepar_model']}' to spaCy pipeline.")
            benepar_model_name = self.config['benepar_model']
            if "benepar" not in self.nlp.pipe_names:
                try:
                    # Ensure benepar is imported if not already
                    import benepar 
                    # Add the pipe
                    self.nlp.add_pipe("benepar", config={"model": benepar_model_name})
                    self.logger.info(f"Added Benepar component '{benepar_model_name}' to spaCy pipeline.")
                except ImportError:
                    self.logger.critical("The 'benepar' library is required but not installed.")
                    self.logger.critical("Please install it, e.g., using: pip install benepar")
                    raise RuntimeError("Missing required library 'benepar'.")
                except Exception as e:
                    # Catch potential errors during add_pipe (e.g., model not found by benepar)
                    self.logger.error(f"Failed to add Benepar component '{benepar_model_name}' to spaCy pipeline. Error: {e}")
                    # Attempt setup *only* if auto-detection failed and we used the default fallback
                    if not self.config.get("benepar_model") and self.config["benepar_model"] == BENEPAR_MODEL_MAP['default']:
                         self.logger.info("Attempting to install default Benepar model...")
                         if setup_models(spacy_model_alias='md', benepar_model_alias='default'): # Attempt default setup
                              self.logger.info("Default models installed. Re-initializing extractor...")
                              self.__init__(config) # Re-run init
                              return # Exit current init call
                         else:
                              self.logger.critical("Failed to install default models after Benepar add_pipe error. Cannot proceed.")
                              raise RuntimeError(f"Failed to add or install required Benepar model '{self.config['benepar_model']}'") from e
                    else:
                        # If user specified a model or auto-detect found something else that failed to load via add_pipe
                        self.logger.critical(f"Specified/detected Benepar model '{self.config['benepar_model']}' could not be added to the pipeline. Ensure it's installed and compatible.")
                        raise RuntimeError(f"Failed to add required Benepar model '{self.config['benepar_model']}' to pipeline") from e
            else:
                self.logger.debug("Benepar component already exists in the pipeline.")

        except (RuntimeError, OSError) as e:
             # Catch errors raised from our specific model loading logic or add_pipe
             self.logger.critical(f"Model loading or pipeline configuration failed: {e}")
             raise # Re-raise the critical error
             
        except Exception as e:
            # Catch other unexpected initialization errors
            self.logger.error(f"Unexpected error during model initialization: {str(e)}", exc_info=True)
            # Avoid the old recursive setup attempt here, as specific load errors are handled above
            raise RuntimeError(f"Unexpected error during extractor initialization: {e}") from e

        # Initialize analyzer
        try:
            self.logger.info("Initializing structure analyzer")
            from anpe.utils.analyzer import ANPEAnalyzer
            self.analyzer = ANPEAnalyzer(nlp=self.nlp)
            self.logger.debug("Structure analyzer initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing structure analyzer: {str(e)}")
            raise

        self.logger.info("ANPE initialized successfully")

    def extract(self, text: str, metadata: bool = False, include_nested: bool = False) -> Dict:
        """
        Extract noun phrases from text.

        Args:
            text: Input text string
            metadata: Whether to include metadata (length and structural analysis)
            include_nested: Whether to include nested noun phrases

        Returns:
            Dict: Dictionary containing extraction results
        """
        self.logger.info(f"Extracting noun phrases with metadata={metadata}, include_nested={include_nested}")

        # Parse the input text
        try:
            tree = self._parse(text)
        except Exception as e:
            self.logger.error(f"Error parsing text: {str(e)}")
            raise

        # Extract NPs based on whether to include nested NPs
        noun_phrases = []
        if include_nested:
            # Extract with hierarchy
            np_hierarchies = self._extract_nps_with_hierarchy(tree)

            # Process noun phrases with correct order of fields
            for i, np_dict in enumerate(np_hierarchies, 1):
                # Create new dictionaries with fields in the correct order
                np_dict = self._process_np_with_ordered_fields(np_dict, str(i), metadata)
                noun_phrases.append(np_dict)

            self.logger.info(f"Extracted {len(noun_phrases)} top-level noun phrases")
        else:
            # Extract only top-level NPs
            np_trees = self._extract_highest_level_nps(tree)
            np_texts = [self._tree_to_text(np).strip() for np in np_trees]
            validated_nps = self._validate_nps(np_texts)

            # Convert to the same dictionary structure
            for i, np_text in enumerate(validated_nps, 1):
                # Create dictionary with fields in the correct order
                np_dict = {
                    "noun_phrase": np_text,
                    "id": str(i),
                    "level": 1
                }

                # Add metadata if requested
                if metadata:
                    length = len(np_text.split())
                    structures = self.analyzer.analyze_single_np(np_text)
                    np_dict["metadata"] = {
                        "length": length,
                        "structures": structures
                    }
                
                # Add empty children list as the last field
                np_dict["children"] = []

                noun_phrases.append(np_dict)

            self.logger.info(f"Extracted {len(noun_phrases)} top-level noun phrases")

        # Prepare the result dictionary
        # Assemble configuration used for this extraction
        config_used = {
            "min_length": self.config.get("min_length"),
            "max_length": self.config.get("max_length"),
            "accept_pronouns": self.config.get("accept_pronouns"),
            "structure_filters": self.config.get("structure_filters", []),
            "newline_breaks": self.config.get("newline_breaks"),
            # Capture the actual models loaded by the extractor
            "spacy_model_used": self.nlp.meta.get("name", "unknown"),
            # Get the actual Benepar model name stored during init
            "benepar_model_used": getattr(self, '_loaded_benepar_model_name', 'unknown'), 
            # Include parameters passed to the extract call
            "metadata_requested": metadata,
            "nested_requested": include_nested
        }
        
        # Filter out None values from config for cleaner output
        config_used = {k: v for k, v in config_used.items() if v is not None}

        # Prepare the final result structure
        result = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Add the configuration section
            "configuration": config_used,
            # The flags about what the output contains are now in configuration
            # (metadata_requested, nested_requested)
            "results": noun_phrases
        }

        return result
    
    def export(self, text: str, format: str = "txt", output: Optional[str] = None, 
             metadata: bool = False, include_nested: bool = False) -> str:
        """
        Extract noun phrases and export to the specified format.
        
        Args:
            text: Input text string
            format: Export format ("txt", "csv", or "json")
            output: Path to the output file or directory. 
                    If path ends with .txt, .csv, or .json, treated as a file.
                    Otherwise, treated as a directory (creating a timestamped file).
                    Defaults to the current working directory.
            metadata: Whether to include metadata (length and structural analysis)
            include_nested: Whether to include nested noun phrases
            
        Returns:
            str: Full path to the exported file
            
        Raises:
            ValueError: If an invalid format is specified.
            IOError: If there are issues creating directories or writing the file.
        """
        self.logger.info(f"Extracting and exporting with metadata={metadata}, include_nested={include_nested}, format={format}")
        
        # Validate format
        valid_formats = ["txt", "csv", "json"]
        if format not in valid_formats:
            self.logger.error(f"Invalid format: {format}. Must be one of {valid_formats}")
            raise ValueError(f"Invalid format: {format}. Must be one of {valid_formats}")
        
        # Default to current directory
        if output is None:
            output = os.getcwd()
        
        path = Path(output).resolve()
        valid_extensions = ['.txt', '.csv', '.json']
        
        # Determine if path is a file or directory
        if path.suffix.lower() in valid_extensions:
            # FILE MODE: Path has a known extension - treat as file
            file_path = path
            
            # Check if file exists
            if file_path.exists():
                self.logger.warning(f"Output file '{file_path}' already exists and will be overwritten.")
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check for extension/format mismatch
            expected_ext = f".{format.lower()}"
            if path.suffix.lower() != expected_ext:
                self.logger.warning(
                    f"Output file extension '{path.suffix}' does not match the specified format '{format}'. "
                    f"File will be saved with {format} content but keep the original extension."
                )
            
            final_filepath = str(file_path)
            self.logger.debug(f"Output '{output}' treated as file path: {final_filepath}")
            
        else:
            # Check if path has ANY extension (even if not supported)
            if path.suffix and path.suffix != '.':
                # Path has an unsupported extension - warn and use parent directory
                self.logger.warning(
                    f"File extension '{path.suffix}' is not recognized as a supported format (.txt, .csv, .json). "
                    f"Treating '{output}' as a directory path and generating a timestamped file with .{format} extension."
                )
                # Use parent directory if the path is a file with unsupported extension
                dir_path = path.parent
            else:
                # DIRECTORY MODE: No extension - treat as directory
                dir_path = path
            
            # Ensure directory exists
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Cannot create or access directory '{dir_path}': {str(e)}")
                raise IOError(f"Cannot create or access directory '{dir_path}': {str(e)}")
            
            # Generate timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"anpe_export_{timestamp}.{format}"
            
            final_filepath = str(dir_path / filename)
            self.logger.debug(f"Output '{output}' treated as directory. Generated filepath: {final_filepath}")
        
        # Extract noun phrases with appropriate parameters
        result = self.extract(text, metadata=metadata, include_nested=include_nested)
        
        # Use ANPEExporter for exporting
        try:
            exporter = ANPEExporter()
            exported_file = exporter.export(result, format=format, output_filepath=final_filepath)
            self.logger.info(f"Successfully exported to: {exported_file}")
            return exported_file
        except Exception as e:
            self.logger.error(f"Error during file export to {final_filepath}: {str(e)}")
            raise

    
    def _extract_highest_level_nps(self, tree: Tree) -> List[Tree]:
        """
        Extract only the highest level NPs from the parse tree.
        
        Args:
            tree: NLTK parse tree
            
        Returns:
            List[Tree]: List of highest-level NP subtrees
        """
        self.logger.debug("Extracting highest-level NPs from parse tree")
        highest_nps = []
        
        def traverse(node, parent_is_np=False):
            if isinstance(node, str):
                return
                
            label = node.label()
            is_np = label == "NP"
            
            # If this is an NP and its parent is not an NP, it's a highest-level NP
            if is_np and not parent_is_np:
                highest_nps.append(node)
                # For the children of this NP, set parent_is_np to True
                for child in node:
                    traverse(child, parent_is_np=True)
            else:
                # Continue traversing but pass down whether the parent is an NP
                for child in node:
                    traverse(child, parent_is_np=parent_is_np or is_np)
        
        # Start traversal from the root
        traverse(tree)
        self.logger.debug(f"Found {len(highest_nps)} highest-level NPs")
        return highest_nps
    
    def _tree_to_text(self, tree: Tree) -> str:
        """
        Convert a parse tree to text by joining terminal nodes.
        
        Args:
            tree: NLTK parse tree
            
        Returns:
            str: Plain text of the tree
        """
        if isinstance(tree, str):
            return tree
        
        words = []
        for child in tree:
            words.append(self._tree_to_text(child))
        
        return " ".join(words)
    
    def _validate_nps(self, nps: List[str]) -> List[str]:
        """
        Validate noun phrases based on configuration criteria.
        
        Args:
            nps: List of noun phrase strings
            
        Returns:
            List of valid noun phrase strings
        """
        self.logger.debug(f"Validating {len(nps)} noun phrases")
        validated_nps = []
        
        for np in nps:
            if self._is_valid_np(np):
                validated_nps.append(np)
        
        self.logger.debug(f"{len(validated_nps)} noun phrases passed validation")
        return validated_nps
    
    def _is_valid_np(self, np: str) -> bool:
        """
        Check if a noun phrase is valid based on configuration criteria.
        
        Args:
            np: A noun phrase string
            
        Returns:
            bool: Whether the noun phrase is valid
        """
        # Check if it contains any words
        if not np.strip():
            return False
        
        # Split into tokens (simple whitespace splitting)
        tokens = np.split()
        token_count = len(tokens)
        
        # Apply minimum length filter if configured
        if self.min_length is not None and token_count < self.min_length:
            self.logger.debug(f"Rejected NP '{np}' (length {token_count} < min_length {self.min_length})")
            return False
        
        # Apply maximum length filter if configured
        if self.max_length is not None and token_count > self.max_length:
            self.logger.debug(f"Rejected NP '{np}' (length {token_count} > max_length {self.max_length})")
            return False
        
        # Apply pronoun filter if configured
        if not self.accept_pronouns:
            # Simple check for single-word pronouns
            if self.analyzer.is_standalone_pronoun(np):
                self.logger.debug(f"Rejected pronoun NP '{np}'")
                return False
        
        # Apply structure filter if configured
        if self.structure_filters:
            # Get structures for this NP
            structures = self.analyzer.analyze_single_np(np)
            
            # Check if any of the required structures are present
            if not any(structure in structures for structure in self.structure_filters):
                self.logger.debug(f"Rejected NP '{np}' (no matching structure in {self.structure_filters})")
                return False
            
            # If we're filtering by structure but not including metadata, log a warning
            if not hasattr(self, '_structure_filter_warning_logged'):
                self.logger.warning("Applying structure filters but metadata=False. " +
                                  "Structure analysis is happening but won't be included in results.")
                self._structure_filter_warning_logged = True
        
        return True
    
    def _extract_nps_with_hierarchy(self, tree: Tree) -> List[Dict]:
        """
        Extract noun phrases from the parse tree with hierarchical structure.
        
        Args:
            tree: NLTK parse tree
            
        Returns:
            List[Dict]: List of NP dictionaries with hierarchy information
        """
        self.logger.debug("Extracting NPs with hierarchy from parse tree")
        top_level_nps = []
        
        def traverse(node, parent_is_np=False):
            if isinstance(node, str):
                return
                
            label = node.label()
            is_np = label == "NP"
            
            # If this is an NP and its parent is not an NP, it's a top-level NP
            if is_np and not parent_is_np:
                np_dict = self._extract_nested_nps(node)
                if np_dict:
                    top_level_nps.append(np_dict)
            else:
                # Continue traversing but pass down whether the parent is an NP
                for child in node:
                    traverse(child, parent_is_np=parent_is_np or is_np)
        
        # Start traversal from the root
        traverse(tree)
        self.logger.debug(f"Found {len(top_level_nps)} top-level NPs with hierarchy")
        return top_level_nps
    
    def _parse(self, text: str) -> Tree:
        """
        Parse input text using spaCy pipeline with integrated Benepar.

        Args:
            text: Input text string

        Returns:
            Tree: A combined NLTK parse tree for the entire text.
                  Returns Tree('S', []) if no sentences are successfully parsed.
        """
        self.logger.debug("Starting parsing process with spaCy+Benepar pipeline.")

        # 1. Split text into sentences using the main spaCy object's sentencizer
        # This respects the newline_breaks configuration applied in __init__.
        # Temporarily disable benepar for this step to avoid retokenization errors on full text.
        try:
            with self.nlp.select_pipes(disable=["benepar"]):
                 doc_for_sentencization = self.nlp(text) # Process text to get sentence boundaries
            sentences = [sent.text for sent in doc_for_sentencization.sents]
            self.logger.debug(f"Split text into {len(sentences)} sentences using spaCy sentencizer (benepar disabled).")
        except Exception as e:
            self.logger.error(f"Error during spaCy sentence splitting: {e}", exc_info=True)
            raise RuntimeError(f"Failed during initial sentence splitting: {e}") from e

        sentence_trees: List[Tree] = []

        # 2. Process each sentence string individually through the pipeline
        for i, sentence_text in enumerate(sentences):
            if not sentence_text.strip():
                self.logger.debug(f"Skipping empty or whitespace-only sentence {i+1}.")
                continue

            self.logger.debug(f"Processing sentence {i+1}/{len(sentences)}: '{sentence_text[:50]}...'")
            try:
                # Process the individual sentence string
                # Strip whitespace to avoid potential issues with trailing newlines in Benepar
                cleaned_sentence_text = sentence_text.strip()
                if not cleaned_sentence_text:
                    self.logger.debug(f"Skipping sentence {i+1} after stripping resulted in empty string.")
                    continue
                    
                doc = self.nlp(cleaned_sentence_text)

                # Benepar expects one sentence per doc. Handle edge cases.
                sents_in_doc = list(doc.sents)
                if len(sents_in_doc) == 0:
                    self.logger.warning(f"spaCy+Benepar pipeline returned no sentences for input: '{sentence_text}'. Skipping.")
                    continue
                if len(sents_in_doc) > 1:
                    self.logger.warning(
                        f"Sentence '{sentence_text[:50]}...' was split into {len(sents_in_doc)} parts "
                        f"by spaCy pipeline unexpectedly during individual processing. Processing only the first part."
                    )
                sent = sents_in_doc[0] # Process the first (usually only) sentence

                # 3. Retrieve the parse string from the Benepar extension
                if not sent.has_extension("parse_string"):
                    self.logger.warning(f"Benepar attribute '_.parse_string' not found for sentence: '{sent.text}'. Skipping.")
                    continue
                parse_string = sent._.parse_string
                if not parse_string or not parse_string.strip():
                     self.logger.warning(f"Benepar returned an empty parse string for sentence: '{sent.text}'. Skipping.")
                     continue

                # 4. Convert parse string to NLTK Tree
                try:
                    sentence_tree = Tree.fromstring(parse_string)
                    sentence_trees.append(sentence_tree)
                    self.logger.debug(f"Successfully parsed sentence {i+1} into NLTK Tree.")
                except ValueError as e:
                    self.logger.warning(
                        f"Could not parse benepar output string into NLTK Tree. Error: {e}. "
                        f"Sentence: '{sent.text}'. Raw parse string: '{parse_string}'"
                    )
                    continue # Skip this sentence if conversion fails
            except Exception as e:
                # Catch potential errors during the self.nlp(sentence_text) call or subsequent steps
                self.logger.error(f"Error processing sentence {i+1} ('{sentence_text[:50]}...'): {e}", exc_info=True)
                continue # Skip this sentence on error

        # 5. Combine sentence trees into a single root Tree
        if not sentence_trees:
            self.logger.warning("No sentences were successfully parsed into NLTK Trees.")
            return Tree('S', []) # Return empty root tree

        self.logger.info(f"Successfully parsed {len(sentence_trees)} sentences into NLTK Trees.")
        # Combine into a single root tree, mimicking the old structure if multiple sentences exist
        combined_tree = Tree('S', sentence_trees)
        return combined_tree

    def _process_np_with_ordered_fields(self, np_dict: Dict, base_id: str, include_metadata: bool, level: int = 1) -> Dict:
        """
        Process a noun phrase dictionary to ensure fields are in the correct order.
        
        Args:
            np_dict: Original NP dictionary
            base_id: ID to assign for this NP
            include_metadata: Whether to include metadata
            level: Hierarchy level
            
        Returns:
            Dict: Reordered NP dictionary
        """
        # Get the noun phrase text
        np_text = np_dict["noun_phrase"]
        
        # Build a new dictionary with fields in the correct order
        ordered_dict = {
            "noun_phrase": np_text,
            "id": base_id,
            "level": level
        }
        
        # Add metadata if requested
        if include_metadata:
            length = len(np_text.split())
            structures = self.analyzer.analyze_single_np(np_text)
            ordered_dict["metadata"] = {
                "length": length,
                "structures": structures
            }
        
        # Process children
        children = []
        for i, child in enumerate(np_dict.get("children", []), 1):
            child_id = f"{base_id}.{i}"
            ordered_child = self._process_np_with_ordered_fields(child, child_id, include_metadata, level + 1)
            children.append(ordered_child)
        
        # Add children as the last field
        ordered_dict["children"] = children
        
        return ordered_dict
    
    def _extract_nested_nps(self, tree: Tree) -> Dict:
        """
        Extract a single NP with its nested NPs.
        
        Args:
            tree: A parsed constituency tree
            
        Returns:
            Dict: A dictionary representing the NP and its nested structure
        """
        if not isinstance(tree, Tree) or tree.label() != "NP":
            return None

        # Extract the current NP text
        np_text = self._tree_to_text(tree).strip()
        
        # Validate the NP - apply all filters here
        if not np_text or not self._is_valid_np(np_text):
            return None

        # Initialize NP dictionary with "children" as the last field
        np_dict = {
            "noun_phrase": np_text,
            "level": None
        }
        
        # Create empty children list that will be populated
        children = []
        
        # Look for nested NPs
        self._find_all_nested_nps(tree, children)
        
        # Add children as the last field
        np_dict["children"] = children

        return np_dict

    def _find_all_nested_nps(self, tree: Tree, children_list: List) -> None:
        """
        Recursively find all nested NPs in the given tree and add them to children_list.

        Args:
            tree: Current tree node to search in
            children_list: List to add child NPs to
        """
        # Process each child of the current node
        for subtree in tree:
            # Skip leaf nodes (words)
            if isinstance(subtree, str):
                continue

            # If this is an NP (and not the same as the parent)
            if subtree.label() == "NP":
                nested_np_text = self._tree_to_text(subtree).strip()

                # Validate the nested NP
                if nested_np_text and self._is_valid_np(nested_np_text):
                    # Create a new NP dictionary for this nested NP with children as the last field
                    nested_np = {
                        "noun_phrase": nested_np_text,
                        "level": None  # Temporary level, will be updated later
                    }
                    
                    # Create empty children list for this nested NP
                    nested_children = []
                    
                    # Recursively find NPs inside this one
                    self._find_all_nested_nps(subtree, nested_children)
                    
                    # Add children as the last field
                    nested_np["children"] = nested_children

                    # Add this NP to the parent's children
                    children_list.append(nested_np)
            else:
                # Not an NP, but continue searching deeper in the tree for NPs
                # This is what allows us to find NPs in relative clauses, PPs, etc.
                self._find_all_nested_nps(subtree, children_list)
    

def extract(text: str, metadata: bool = False, include_nested: bool = False, **kwargs) -> Dict:
    """
    Convenience function to extract noun phrases from text using default or custom settings.
    """
    # Pass kwargs as a configuration dictionary to ANPEExtractor
    extractor = ANPEExtractor(kwargs)
    return extractor.extract(text, metadata=metadata, include_nested=include_nested)

def export(text: str, format: str = "txt", output: Optional[str] = None, 
           metadata: bool = False, include_nested: bool = False, **kwargs) -> str:
    """
    Convenience function to extract and export noun phrases in one step.
    """
    # Create an extractor instance with the provided configuration
    extractor = ANPEExtractor(kwargs)
    
    # Extract and export the results
    return extractor.export(text, format=format, output=output, 
                           metadata=metadata, include_nested=include_nested)