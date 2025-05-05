from typing import Dict, List, Optional, Any, Tuple, Set
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
# Add Span and Doc imports
from spacy.tokens import Doc, Span, Token
import time


from anpe.config import DEFAULT_CONFIG
from anpe.utils.setup_models import setup_models
from anpe.utils.export import ANPEExporter
from anpe.utils.model_finder import (
    find_installed_spacy_models,
    find_installed_benepar_models,
    select_best_spacy_model,
    select_best_benepar_model
)
from anpe.utils.setup_models import SPACY_MODEL_MAP, BENEPAR_MODEL_MAP
# Import Analyzer
from anpe.utils.analyzer import ANPEAnalyzer

# Define a type alias for the node information dictionary
NPNodeInfo = Dict[str, Any] # Typically {'node': Tree, 'text': str, 'span': Optional[Span]}
# Define a type alias for the hierarchical node structure
NPNodeHierarchy = Dict[str, Any] # Typically {'info': NPNodeInfo, 'children': List['NPNodeHierarchy']}

# --- Standard Logging Setup ---
import logging
logger = logging.getLogger(__name__)
# --- End Standard Logging ---

class ANPEExtractor:
    """Main extractor class for noun phrase extraction."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extractor.

        Args:
            config: Configuration dictionary with options:
                - accept_pronouns: Whether to include pronouns as valid NPs
                - min_length: Minimum token length for NPs
                - max_length: Maximum token length for NPs
                - newline_breaks: Whether to treat newlines as sentence boundaries
                - structure_filters: List of structure types to include (empty=all)
        """
        # --- Use Standard Logger ---
        logger.debug("ANPEExtractor initializing...")
        # --- End Logger Setup ---

        # Initialize default config
        self.config = DEFAULT_CONFIG.copy()
        if config:
            # Filter out any potential logger keys passed inadvertently
            logger_keys = ['log_level', 'log_file', 'log_dir']
            provided_config = {k: v for k, v in config.items() if k not in logger_keys}
            self.config.update(provided_config)
            logger.debug(f"Updated config with provided keys: {list(provided_config.keys())}")
        else:
             logger.debug("Using default config, no overrides provided.")

        # Initialize other user configuration
        logger.info("Initializing Extractor Config...")
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

        logger.info("Loading NLP Models...")

        # --- Determine spaCy model to load ---
        try:
            # Determine spacy model to use
            spacy_model_to_use = self.config.get("spacy_model")
            if spacy_model_to_use:
                # Validate if the user provided a known model name/alias
                if spacy_model_to_use not in SPACY_MODEL_MAP:
                     logger.warning(f"Specified spaCy model '{spacy_model_to_use}' is not a recognized alias/name. Attempting to use it anyway.")
                     # Map might not be exhaustive, let spacy.load handle unknown models later
                else:
                     # If it's an alias, map it to the actual name
                     spacy_model_to_use = SPACY_MODEL_MAP.get(spacy_model_to_use, spacy_model_to_use)
                logger.info(f"Using specified spaCy model: {spacy_model_to_use}")
            else:
                logger.debug("No spaCy model specified in config, attempting auto-detection...")
                installed_spacy = find_installed_spacy_models()
                spacy_model_to_use = select_best_spacy_model(installed_spacy)
                if spacy_model_to_use:
                    logger.info(f"Auto-detected best available spaCy model: {spacy_model_to_use}")
                else:
                    # Fallback to default if none detected
                    spacy_model_to_use = SPACY_MODEL_MAP['md'] # Default: en_core_web_md
                    logger.warning(f"Could not auto-detect any installed spaCy models. Falling back to default: {spacy_model_to_use}")
            self.config["spacy_model"] = spacy_model_to_use

            # --- Determine Benepar model to use ---
            _user_specified_benepar_initially = self.config.get("benepar_model") is not None
            benepar_model_to_load = self.config.get("benepar_model") # Get user preference

            if not benepar_model_to_load:
                logger.debug("No Benepar model specified in config, attempting auto-detection...")
                installed_benepar = find_installed_benepar_models()
                benepar_model_to_load = select_best_benepar_model(installed_benepar)
                if benepar_model_to_load:
                    logger.info(f"Auto-detected best available Benepar model: {benepar_model_to_load}")
                else:
                    benepar_model_to_load = BENEPAR_MODEL_MAP['default']
                    logger.warning(f"Could not auto-detect any installed Benepar models. Falling back to default: {benepar_model_to_load}")
            else:
                # User specified a model, potentially an alias - resolve it
                if benepar_model_to_load not in BENEPAR_MODEL_MAP:
                     logger.warning(f"Specified Benepar model '{benepar_model_to_load}' is not a recognized alias/name. Attempting to use it anyway.")
                else:
                     benepar_model_to_load = BENEPAR_MODEL_MAP.get(benepar_model_to_load, benepar_model_to_load)
                logger.info(f"Using specified Benepar model (resolved): {benepar_model_to_load}")

            # Store the *intended* model back in config for reference
            self.config["benepar_model"] = benepar_model_to_load
            self._loaded_benepar_model_name = "unknown" # Initialize loaded name

            # --- Check spacy-transformers dependency (SIMPLIFIED) ---
            self.use_transformers = False # Initialize flag
            if spacy_model_to_use and spacy_model_to_use.endswith('_trf'):
                self.use_transformers = True
                logger.info(f"Using a spaCy transformer model ('{spacy_model_to_use}'). Underlying transformer handled by spaCy.")
                # Ensure library is installed
                try:
                    import spacy_transformers
                    logger.debug("'spacy-transformers' library found.")
                except ImportError:
                    logger.critical(f"The spaCy model '{spacy_model_to_use}' requires 'spacy-transformers', but it is not installed.")
                    logger.critical(f"Please install it: pip install 'spacy[transformers]'")
                    raise RuntimeError(f"Missing required library 'spacy-transformers' for model '{spacy_model_to_use}'.")
            else:
                logger.info(f"Not using a spaCy transformer model (resolved model: '{spacy_model_to_use}').")
            # --- End Simplified Transformer Check ---

            # --- Load Models ---
            logger.info(f"Loading spaCy model: '{spacy_model_to_use}'")
            try:
                self.nlp = spacy.load(spacy_model_to_use)
                logger.info("spaCy model loaded successfully.")
            except OSError as e:
                logger.error(f"Failed to load spaCy model '{spacy_model_to_use}'. Is it installed? Error: {e}")
                # Add specific hint for transformer models
                if spacy_model_to_use.endswith('_trf'):
                    logger.error("Ensure the 'spacy-transformers' library is also installed: pip install 'spacy[transformers]'")
                # Attempt setup *only* if auto-detection failed and we used the default fallback
                if not self.config.get("spacy_model") and spacy_model_to_use == SPACY_MODEL_MAP['md']:
                    logger.info("Attempting to install default spaCy model...")
                    if setup_models(spacy_model_alias='md', benepar_model_alias='default'): # Attempt default setup
                         logger.info("Default models installed. Re-initializing extractor...")
                         self.__init__(config) # Re-run init
                         return # Exit current init call
                    else:
                         logger.critical("Failed to install default models after loading error. Cannot proceed.")
                         raise RuntimeError(f"Failed to load or install required spaCy model '{spacy_model_to_use}'") from e
                else:
                     # If user specified a model or auto-detect found something else that failed to load
                     logger.critical(f"Specified/detected spaCy model '{spacy_model_to_use}' could not be loaded. Please ensure it is installed correctly.")
                     raise RuntimeError(f"Failed to load required spaCy model '{spacy_model_to_use}'") from e


            # --- Add and Configure the Rule-Based Sentencizer ---
            # Always add the sentencizer to ensure consistent, controllable sentence boundary rules.
            # Add it before the parser if the parser exists, otherwise add it first.
            if "parser" in self.nlp.pipe_names:
                logger.debug("Adding 'sentencizer' component before 'parser'.")
                # Check if it already exists to avoid errors on re-init
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer", before="parser")
                else:
                    logger.debug("'sentencizer' already in pipeline.")
            else:
                logger.debug("Adding 'sentencizer' component first (no parser found).")
                # Check if it already exists
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer", first=True)
                else:
                    logger.debug("'sentencizer' already in pipeline.")
            
            # Get the sentencizer pipe
            sbd_pipe = self.nlp.get_pipe("sentencizer")
            sbd_component_name = "sentencizer" # We know the name now
            logger.debug(f"Using '{sbd_component_name}' component for sentence boundary detection.")

            # Apply newline breaking control based on configuration
            if self.newline_breaks:
                logger.info(f"Configuring '{sbd_component_name}' to treat newlines as sentence boundaries")
                # Add check before accessing punct_chars
                if hasattr(sbd_pipe, 'punct_chars'):
                    if '\\n' not in sbd_pipe.punct_chars:
                        sbd_pipe.punct_chars.add('\\n')
                        logger.debug(f"Added '\\n' to punct_chars for '{sbd_component_name}'")
                    else: # '\\n' already in punct_chars
                        logger.debug(f"'\\n' already present in punct_chars for '{sbd_component_name}'")
                else:
                     logger.warning(f"Component '{sbd_component_name}' does not have 'punct_chars' attribute. Cannot add newline break rule.")
            else:
                logger.info(f"Configuring '{sbd_component_name}' to NOT treat newlines as sentence boundaries")
                # Add check before accessing punct_chars
                if hasattr(sbd_pipe, 'punct_chars'):
                    if '\\n' in sbd_pipe.punct_chars:
                        sbd_pipe.punct_chars.remove('\\n')
                        logger.debug(f"Removed '\\n' from punct_chars for '{sbd_component_name}'")
                    else: # '\\n' not in punct_chars
                        logger.debug(f"'\\n' not found in punct_chars for '{sbd_component_name}'.")
                else:
                     logger.warning(f"Component '{sbd_component_name}' does not have 'punct_chars' attribute. Cannot remove newline break rule.")


            # --- Add Benepar to spaCy pipeline ---
            logger.debug(f"Ensuring Benepar component is added...")
            # --- Check if user specified the model *before* potentially overwriting config ---
            # Store the initial user specification status
            # _user_specified_benepar_initially is now determined above
            # --- End Check ---
            default_benepar_model = BENEPAR_MODEL_MAP['default']
            
            if "benepar" not in self.nlp.pipe_names:
                benepar_loaded = False
                try:
                    logger.info(f"Attempting to add Benepar component with model: '{benepar_model_to_load}'")
                    self.nlp.add_pipe("benepar", config={"model": benepar_model_to_load})
                    logger.info(f"Benepar component ('{benepar_model_to_load}') added successfully.")
                    self._loaded_benepar_model_name = benepar_model_to_load # Store the name that loaded
                    benepar_loaded = True
                except ValueError as e:
                    logger.warning(f"Failed to load Benepar model '{benepar_model_to_load}': {e}")
                    # More robust check for the specific errors indicating a missing model package/plugin
                    e_str = str(e).lower()
                    is_load_error = isinstance(e, ValueError) and \
                                    ("can't find package" in e_str or "can't load plugin" in e_str)
                    # Simplified condition: Fallback if load error and model wasn't user-specified.
                    # Use the initially stored status for the check
                    if is_load_error and not _user_specified_benepar_initially:
                        logger.info(f"Attempting fallback to default Benepar model: '{default_benepar_model}'")
                        try:
                            self.nlp.add_pipe("benepar", config={"model": default_benepar_model})
                            logger.info(f"Benepar component ('{default_benepar_model}') added successfully via fallback.")
                            self._loaded_benepar_model_name = default_benepar_model # Store the name that loaded
                            self.config["benepar_model"] = default_benepar_model # Update config
                            benepar_loaded = True
                        except ValueError as fallback_e:
                            logger.critical(f"Fallback attempt with default Benepar model '{default_benepar_model}' also failed: {fallback_e}")
                            # Error message includes both attempts if fallback was tried
                            raise RuntimeError(f"Failed to add required Benepar component. Attempted '{benepar_model_to_load}' and fallback '{default_benepar_model}'.") from fallback_e
                        except Exception as fallback_ex:
                             logger.critical(f"Unexpected error during Benepar fallback: {fallback_ex}")
                             raise RuntimeError(f"Unexpected error during Benepar fallback attempt for model '{default_benepar_model}'") from fallback_ex
                    else:
                        # Error if primary model failed and no fallback was attempted/successful
                        logger.critical(f"Failed to load required Benepar model '{benepar_model_to_load}' (either specified or auto-detected). Fallback not applicable or failed. Cannot proceed.")
                        raise RuntimeError(f"Failed to add required Benepar component with model '{benepar_model_to_load}'.") from e
                except Exception as ex: # Catch other potential errors during initial add_pipe
                    logger.critical(f"Unexpected error adding Benepar component '{benepar_model_to_load}': {ex}")
                    raise RuntimeError(f"Unexpected error adding Benepar component for model '{benepar_model_to_load}'") from ex

                if not benepar_loaded:
                     # Should be unreachable if errors raise correctly, but acts as a final check
                     raise RuntimeError(f"Benepar component could not be loaded for model '{benepar_model_to_load}' or its fallback.")
            else:
                logger.info("Benepar component already found in the pipeline.")
                # Attempt to get the name from the existing pipe if possible
                try:
                    benepar_pipe = self.nlp.get_pipe("benepar")
                    # Accessing config might be fragile, depends on how benepar stores it
                    loaded_name = benepar_pipe.cfg.get("model", self.config.get("benepar_model", "existing/unknown"))
                    self._loaded_benepar_model_name = loaded_name
                except: # Broad except as getting config might fail
                     self._loaded_benepar_model_name = self.config.get("benepar_model", "existing/unknown")
                logger.debug(f"Existing Benepar component assumed model: {self._loaded_benepar_model_name}")

        except Exception as e:
             logger.critical(f"Fatal error during ANPEExtractor initialization: {str(e)}", exc_info=True)
             raise

        # Initialize Analyzer after nlp pipeline is fully set up
        self.analyzer = ANPEAnalyzer(self.nlp) # Pass the fully configured nlp object
        logger.info("ANPEExtractor initialized successfully")

    def extract(self, text: str, metadata: bool = False, include_nested: bool = False) -> Dict:
        """
        Extract noun phrases from text using Spans and in-context analysis.

        Args:
            text: Input text string
            metadata: Whether to include metadata (length and structural analysis)
            include_nested: Whether to include nested noun phrases

        Returns:
            Dict: Dictionary containing extraction results
        """
        logger.info(f"Extracting noun phrases with metadata={metadata}, include_nested={include_nested}")
        start_time = datetime.datetime.now()

        # --- Early exit for empty input ---
        if not text or text.isspace():
            logger.info("Input text is empty or whitespace. Returning empty result.")
            end_time = datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()
            # Return the standard empty result structure (similar to the end)
            config_used = {
                # Try to get basic config even if models aren't loaded
                "min_length": self.min_length,
                "max_length": self.max_length,
                "accept_pronouns": self.accept_pronouns,
                "structure_filters": self.structure_filters,
                "newline_breaks": self.newline_breaks,
                "spacy_model_used": self.config.get('spacy_model', 'N/A'),
                "benepar_model_used": self.config.get('benepar_model', 'N/A'), 
                "metadata_requested": metadata,
                "nested_requested": include_nested
            }
            config_used = {k: v for k, v in config_used.items() if v is not None}
            return {
                "timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_duration_seconds": round(duration, 3),
                "configuration": config_used,
                "results": []
            }

        # --- Pre-process text for internal newlines --- 
        processed_text = text
        if not self.newline_breaks:
            # If not treating newlines as breaks, replace single newlines with spaces
            # to help Benepar parse correctly. Keep double newlines as potential para breaks.
            logger.debug("Replacing single internal newlines with spaces before parsing (newline_breaks=False)")
            # Avoid replacing \n\n - use a lookahead/lookbehind or simpler replace sequence
            processed_text = processed_text.replace('\r\n', '\n') # Normalize line endings first
            processed_text = processed_text.replace('\n\n', '__PARABREAK__') # Temporarily mark real breaks
            processed_text = processed_text.replace('\n', ' ') # Replace remaining single newlines
            processed_text = processed_text.replace('__PARABREAK__', '\n\n') # Restore real breaks
        else:
            logger.debug("Keeping newlines as potential sentence breaks (newline_breaks=True)")
            # Optional: Normalize line endings anyway?
            processed_text = processed_text.replace('\r\n', '\n')

        # Parse the entire text with the full pipeline (spaCy + Benepar)
        try:
            logger.debug("Parsing text with spaCy+Benepar...")
            doc = self.nlp(processed_text)
            logger.debug("Text parsed successfully.")
        except Exception as e:
            logger.error(f"Error parsing text with spaCy pipeline: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed during text parsing: {e}") from e

        # Prepare list to hold results
        processed_results = []
        top_level_id_counter = 0
        all_sentence_hierarchies = []

        # Iterate through sentences in the parsed Doc
        for i, sent in enumerate(doc.sents):
            logger.debug(f"[extract loop] Processing sentence {i}: '{sent.text[:50]}...'")
            if not sent.has_extension("parse_string"):
                logger.warning(f"[extract loop] Sentence does not have '_.parse_string'. Skipping: '{sent.text}'")
                continue
            parse_string = sent._.parse_string
            if not parse_string or not parse_string.strip():
                 logger.warning(f"[extract loop] Benepar returned an empty parse string for sentence: '{sent.text}'. Skipping.")
                 continue
            
            # Parse the string into an NLTK Tree
            constituents_tree = None # Initialize before try block
            try:
                logger.debug(f"[extract loop] Attempting Tree.fromstring...")
                constituents_tree = Tree.fromstring(parse_string)
                logger.debug(f"[extract loop] Successfully created constituents_tree. Type: {type(constituents_tree)}")
            except ValueError as e:
                 logger.warning(
                     f"[extract loop] Could not parse benepar output string into NLTK Tree. Error: {e}. "
                     f"Sentence: '{sent.text}'. Raw parse string: '{parse_string}'"
                 )
                 continue # Skip this sentence if conversion fails
            except Exception as e: # Catch other unexpected errors during parsing
                 logger.error(f"[extract loop] Unexpected error parsing benepar string for sentence '{sent.text}': {e}", exc_info=True)
                 continue

            # Log before calling extraction helpers
            logger.debug(f"[extract loop] Tree created. Calling extraction helpers (include_nested={include_nested})...")
            # Extract NPs based on whether to include nested NPs
            if include_nested:
                 # 1. Collect info for all NP nodes in the sentence tree
                 #    (Node, Text, Span (or None))
                 np_nodes_info = self._collect_np_nodes_info(constituents_tree, sent)
                 logger.debug(f"[extract loop] Collected info for {len(np_nodes_info)} NP nodes in sentence.")

                 np_info_map = {id(info['node']): info for info in np_nodes_info} # MAP IS CREATED

                 # 2. Build hierarchy based on the tree structure, linking the collected info
                 sentence_hierarchy = self._build_hierarchy_from_tree(constituents_tree, np_info_map) # MAP IS PASSED
                 logger.debug(f"[extract loop] Built hierarchy with {len(sentence_hierarchy)} top-level nodes.")
                 all_sentence_hierarchies.extend(sentence_hierarchy)

            else:
                # Extract only top-level NP Spans using the tree-based method
                logger.debug(f"[extract loop] Extracting highest-level NPs (include_nested=False). Tree: {constituents_tree}")
                candidate_np_spans = self._extract_highest_level_np_spans(constituents_tree, sent)
                
                # --- FILTERING STEP for include_nested=False ---
                # Remove spans that are contained within other spans in the *same* list
                highest_level_only_spans = []
                for i, span1 in enumerate(candidate_np_spans):
                    is_contained = False
                    for j, span2 in enumerate(candidate_np_spans):
                        if i == j: continue # Don't compare to self
                        # Check if span1 is contained within span2
                        if span2.start <= span1.start and span2.end >= span1.end and span1 != span2:
                            is_contained = True
                            logger.debug(f"  Filtering contained span: '{span1.text}' (is inside '{span2.text}')")
                            break # Found a container, no need to check further
                    if not is_contained:
                        highest_level_only_spans.append(span1)
                
                logger.debug(f"[extract loop] Filtered {len(candidate_np_spans) - len(highest_level_only_spans)} contained spans. Processing {len(highest_level_only_spans)} highest-level spans.")
                # --- END FILTERING STEP ---
                
                # Validate and process each *filtered* highest-level NP span
                # Use 'highest_level_only_spans' instead of 'candidate_np_spans'
                for np_span in highest_level_only_spans: 
                     # Perform structural analysis first if needed for validation/output
                    structures = []
                    if metadata or self.structure_filters: # Analyze if needed for metadata OR structure filtering
                        structures = self.analyzer.analyze_single_np(np_span)
                    
                    # Validate the span based on config and (optionally) structures
                    if self._is_valid_np(np_span, structures):
                        top_level_id_counter += 1
                        np_text = np_span.text.strip() # Use strip here for consistency
                        
                        # Create dictionary with fields in the correct order
                        np_dict = {
                            "noun_phrase": np_text,
                            "id": str(top_level_id_counter),
                            "level": 1
                        }

                        # Add metadata if requested (structures already computed if needed)
                        if metadata:
                            # Ensure structures are computed if not already
                            if not structures:
                                structures = self.analyzer.analyze_single_np(np_span)
                            np_dict["metadata"] = {
                                "length": len(np_span), # Use Span length (consistent with validation)
                                "structures": structures
                            }
                        
                        # Add empty children list as the last field for consistency
                        np_dict["children"] = []
                        processed_results.append(np_dict)


        # --- Process Hierarchies if include_nested=True ---
        if include_nested:
            logger.info(f"Processing {len(all_sentence_hierarchies)} top-level NP node hierarchies...")
            for hierarchy_node in all_sentence_hierarchies:
                 top_level_id_counter += 1
                 # Process the hierarchy node, adding metadata internally if requested
                 processed_dict = self._process_np_node_info(
                     hierarchy_node, str(top_level_id_counter), metadata
                 )
                 if processed_dict:
                     processed_results.append(processed_dict)
                 else:
                     # Log that the top-level item itself (or its entire branch due to filtering) was skipped
                      original_text = (hierarchy_node.get('info') or {}).get('text', 'N/A')
                      logger.debug(f"Top-level NP hierarchy starting with text '{original_text}' (attempted id {top_level_id_counter}) was filtered out.")

        if include_nested:
             # Ensure correct logging based on the *processed* results
             num_processed_top_level = len(processed_results) 
             logger.info(f"Extracted and processed {num_processed_top_level} valid top-level noun phrase hierarchies (with nesting)")
        else:
             # Logging already counts processed NPs via top_level_id_counter if validation passes
             # Or use len(processed_results) for consistency
             logger.info(f"Extracted and processed {len(processed_results)} valid highest-level noun phrases (no nesting)")


        # Prepare the result dictionary
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Assemble configuration used for this extraction using instance attributes
        config_used = {
            "min_length": self.min_length,
            "max_length": self.max_length,
            "accept_pronouns": self.accept_pronouns,
            "structure_filters": self.structure_filters,
            "newline_breaks": self.newline_breaks,
            # Get model names from attributes if available, else config/unknown
            "spacy_model_used": getattr(self.nlp, 'meta', {}).get("name", self.config.get('spacy_model', "unknown")), 
            "benepar_model_used": getattr(self, '_loaded_benepar_model_name', self.config.get('benepar_model', 'unknown')), 
            # Include parameters passed to the extract call
            "metadata_requested": metadata,
            "nested_requested": include_nested
        }
        
        # Filter out None values from config for cleaner output
        config_used = {k: v for k, v in config_used.items() if v is not None}

        # Prepare the final result structure
        result = {
            "timestamp": end_time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_duration_seconds": round(duration, 3),
            # Add the configuration section
            "configuration": config_used,
            # The flags about what the output contains are now in configuration
            # (metadata_requested, nested_requested)
            "results": processed_results # Use the unified list name
        }

        return result

    def export(self, text: str, format: str = "txt", output: Optional[str] = None, 
             metadata: bool = False, include_nested: bool = False) -> str:
        """
        Extract noun phrases and export to the specified format.
        (No changes needed here as it calls the refactored `extract`)
        
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
        logger.info(f"Extracting and exporting with metadata={metadata}, include_nested={include_nested}, format={format}")
        
        # Validate format
        valid_formats = ["txt", "csv", "json"]
        if format not in valid_formats:
            logger.error(f"Invalid format: {format}. Must be one of {valid_formats}")
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
                logger.warning(f"Output file '{file_path}' already exists and will be overwritten.")
            
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check for extension/format mismatch
            expected_ext = f".{format.lower()}"
            if path.suffix.lower() != expected_ext:
                logger.warning(
                    f"Output file extension '{path.suffix}' does not match the specified format '{format}'. "
                    f"File will be saved with {format} content but keep the original extension."
                )
            
            final_filepath = str(file_path)
            logger.debug(f"Output '{output}' treated as file path: {final_filepath}")
            
        else:
            # Check if path has ANY extension (even if not supported)
            if path.suffix and path.suffix != '.':
                # Path has an unsupported extension - warn and use parent directory
                logger.warning(
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
                logger.error(f"Cannot create or access directory '{dir_path}': {str(e)}")
                raise IOError(f"Cannot create or access directory '{dir_path}': {str(e)}")
            
            # Generate timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"anpe_export_{timestamp}.{format}"
            
            final_filepath = str(dir_path / filename)
            logger.debug(f"Output '{output}' treated as directory. Generated filepath: {final_filepath}")
        
        # Extract noun phrases with appropriate parameters using the refactored method
        result = self.extract(text, metadata=metadata, include_nested=include_nested)
        
        # Use ANPEExporter for exporting
        try:
            exporter = ANPEExporter()
            exported_file = exporter.export(result, format=format, output_filepath=final_filepath)
            logger.info(f"Successfully exported to: {exported_file}")
            return exported_file
        except Exception as e:
            logger.error(f"Error during file export to {final_filepath}: {str(e)}")
            raise

    # --------------------------------------------------------------------------
    # Private Helper Methods for Extraction Logic
    # --------------------------------------------------------------------------

    def _tree_to_text(self, tree: Tree) -> str:
        """
        Convert an NLTK parse tree (or subtree) to text by joining terminal nodes.
        Handles potential non-string leaves gracefully.

        Args:
            tree: NLTK parse tree or subtree.

        Returns:
            str: Plain text representation of the tree's leaves.
        """
        words = []
        try:
            for leaf in tree.leaves():
                # Ensure leaf is converted to string, handling potential non-string types
                words.append(str(leaf))
        except AttributeError:
             # If input is not a Tree (e.g., already a string leaf), return it directly
             if isinstance(tree, str):
                 return tree
             else:
                 logger.warning(f"Input to _tree_to_text was not an NLTK Tree or string: {type(tree)}. Returning empty string.")
                 return ""
        except Exception as e:
            logger.error(f"Unexpected error in _tree_to_text for tree '{tree}': {e}", exc_info=True)
            return " ".join(words) # Return potentially partial result on error

        return " ".join(words)


    def _get_span_from_token_indices(self, sent: Span, start_token_idx: int, end_token_idx: int) -> Optional[Span]:
        """
        Safely creates a spaCy Span from token indices within a sentence Span.
        Uses char_span for robustness, falling back to slicing.

        Args:
            sent: The spaCy Span for the sentence.
            start_token_idx: Start token index (inclusive, relative to sent).
            end_token_idx: End token index (inclusive, relative to sent).

        Returns:
            Optional[Span]: The created spaCy Span, or None if indices are invalid.
        """
        if start_token_idx >= 0 and end_token_idx >= start_token_idx and end_token_idx < len(sent):
            try:
                # Use char_span for robustness against tokenization differences
                start_char = sent[start_token_idx].idx - sent.start_char
                # Ensure end_token_idx is within bounds before accessing
                end_token = sent[end_token_idx]
                end_char = end_token.idx + len(end_token.text) - sent.start_char
                
                span = sent.char_span(start_char, end_char, alignment_mode="contract")
                if span is None:
                    logger.warning(f"Could not create char span ({start_char}, {end_char}) for tokens {start_token_idx}-{end_token_idx} in '{sent.text}'. Trying direct token slicing.")
                    # Fallback to direct slicing (less robust)
                    span = sent[start_token_idx : end_token_idx + 1]
                return span
            except IndexError:
                logger.warning(f"Token index out of bounds ({start_token_idx} or {end_token_idx}) for sentence length {len(sent)}. Cannot create span.")
                return None
            except Exception as e:
                logger.error(f"Unexpected error creating span ({start_token_idx}-{end_token_idx}) in _get_span_from_token_indices: {e}", exc_info=True)
                return None
        else:
            # Log if indices were invalid from the start
            if not (start_token_idx >= 0 and end_token_idx >= start_token_idx):
                 logger.warning(f"Invalid start/end indices ({start_token_idx}, {end_token_idx}) passed to _get_span_from_token_indices.")
            elif not (end_token_idx < len(sent)):
                 logger.warning(f"End token index {end_token_idx} >= sentence length {len(sent)} in _get_span_from_token_indices.")
            return None


    def _find_token_indices_for_leaves(self, sent_tokens: List[Token], leaves: List[str]) -> Optional[Tuple[int, int]]:
        """
        Attempts to find the start and end token indices in a spaCy sentence
        that correspond to a given sequence of leaves from an NLTK tree.

        Args:
            sent_tokens: List of spaCy Token objects for the sentence.
            leaves: List of leaf strings from the NLTK tree node.

        Returns:
            Optional[Tuple[int, int]]: Tuple of (start_token_idx, end_token_idx) inclusive,
                                      or None if no exact match is found.
        """
        if not leaves or not sent_tokens:
            return None
            
        num_leaves = len(leaves)
        num_sent_tokens = len(sent_tokens)

        if num_leaves == 0: return None

        # Iterate through all possible start positions in the sentence tokens
        for i in range(num_sent_tokens - num_leaves + 1):
            match = True
            # Check if the sequence of token texts matches the leaves
            for j in range(num_leaves):
                if leaves[j] != sent_tokens[i + j].text:
                    match = False
                    break
            if match:
                # Found a match, return start and *inclusive* end index
                return i, i + num_leaves - 1

        # If no match found after checking all possible start positions
        logger.warning(f"Could not map leaves {leaves} to tokens in sentence. Span creation might fail.")
        return None # Indicate failure to map

    # --- Highest Level NP Extraction (include_nested=False) ---

    def _extract_highest_level_np_spans(self, tree: Tree, sent: Span) -> List[Span]:
        """
        Extracts spaCy Spans for the highest level NPs from a Benepar constituency tree
        relative to the original sentence Span. Relies on finding NP nodes whose
        immediate parent in the tree is not also an NP.

        Args:
            tree: NLTK parse tree for the sentence.
            sent: The spaCy Span object for the sentence.

        Returns:
            List[Span]: List of spaCy Spans for highest-level NPs based on tree structure.
        """
        logger.debug(f"Extracting highest-level NP spans from sentence: '{sent.text[:50]}...'")
        highest_np_spans = []
        
        try:
            sent_tokens = list(sent)
        except Exception as e:
             logger.error(f"Failed to get tokens from sentence span: {e}", exc_info=True)
             return [] # Cannot proceed without tokens


        def traverse(node: Tree, parent_is_np=False):
            nonlocal highest_np_spans
            if isinstance(node, str): # Reached a leaf node
                return

            is_np = node.label() == "NP"

            # If this is an NP and its immediate parent was not an NP
            if is_np and not parent_is_np:
                leaves = node.leaves()
                indices = self._find_token_indices_for_leaves(sent_tokens, leaves)

                if indices:
                    start_idx, end_idx = indices
                    np_span = self._get_span_from_token_indices(sent, start_idx, end_idx)
                    if np_span:
                        highest_np_spans.append(np_span)
                    else:
                        logger.warning(f"Found highest-level NP node for '{self._tree_to_text(node)}' but failed to create span from indices {indices}.")
                else:
                    # Log failure to find indices for this potential highest-level NP
                    logger.warning(f"Could not map leaves {leaves} to sentence tokens for potential highest-level NP '{self._tree_to_text(node)}'.")

                # Children of this highest-level NP node are considered nested,
                # so their 'parent_is_np' flag becomes True.
                for child in node:
                    traverse(child, parent_is_np=True)
            else:
                # Continue traversal, passing the current node's NP status
                # (or the parent's status if the current node isn't NP)
                # down to the children.
                current_node_is_np_for_children = parent_is_np or is_np
                for child in node:
                    traverse(child, parent_is_np=current_node_is_np_for_children)

        # Start traversal from the sentence root tree
        traverse(tree)
        logger.debug(f"Found {len(highest_np_spans)} candidate highest-level NP spans based on tree structure.")
        return highest_np_spans

    # --- Nested NP Extraction (include_nested=True) ---

    def _collect_np_nodes_info(self, tree: Tree, sent: Span) -> List[NPNodeInfo]:
        """
        Traverses the NLTK tree, identifies all NP nodes, extracts their text,
        and attempts to map them to spaCy Spans.

        Args:
            tree: The NLTK constituency tree for the sentence.
            sent: The spaCy Span object for the sentence.

        Returns:
            List[NPNodeInfo]: A list of dictionaries, each containing:
                              {'node': Tree, 'text': str, 'span': Optional[Span]}
        """
        np_info_list = []
        try:
            sent_tokens = list(sent)
        except Exception as e:
             logger.error(f"Failed to get tokens from sentence span in _collect_np_nodes_info: {e}", exc_info=True)
             return [] # Cannot proceed

        def traverse(node: Tree):
            nonlocal np_info_list
            if isinstance(node, str):
                return

            if node.label() == 'NP':
                # 1. Extract text reliably from the tree node
                np_text = self._tree_to_text(node).strip()
                if not np_text: # Skip if node yields empty text
                    logger.debug("Skipping NP node with empty text.")
                else:
                    # 2. Attempt to map leaves to spaCy Span
                    np_span = None
                    leaves = node.leaves()
                    indices = self._find_token_indices_for_leaves(sent_tokens, leaves)
                    if indices:
                        np_span = self._get_span_from_token_indices(sent, indices[0], indices[1])
                        if not np_span:
                            logger.warning(f"Mapping failure: Could not create span for NP node with text '{np_text}' from indices {indices}.")
                    else:
                        logger.warning(f"Mapping failure: Could not find token indices for NP node with text '{np_text}' (leaves: {leaves}).")

                    # 3. Store info (including potentially None span)
                    np_info_list.append({'node': node, 'text': np_text, 'span': np_span})

            # Continue traversal for children
            for child in node:
                traverse(child)

        traverse(tree)
        logger.debug(f"_collect_np_nodes_info found {len(np_info_list)} NP nodes.")
        return np_info_list


    def _build_hierarchy_from_tree(self, current_node: Tree, np_info_map: Dict[Tree, NPNodeInfo]) -> List[NPNodeHierarchy]:
        """
        Recursively builds the NP hierarchy based on the NLTK tree structure,
        attaching the pre-collected node information. Only includes nodes marked as NP.

        Args:
            current_node: The current NLTK Tree node being processed.
            np_info_map: A mapping from NLTK Tree NP nodes to their collected info dictionaries.

        Returns:
            List[NPNodeHierarchy]: A list of hierarchy dictionaries for the direct NP children
                                    of the current_node.
        """
        child_hierarchies = []
        if isinstance(current_node, str):
            return [] # Cannot build hierarchy from a leaf

        for child_node in current_node:
            if isinstance(child_node, str):
                continue # Skip leaves

            if child_node.label() == "NP":
                # This child is an NP node itself.
                # Look up its info (it should exist if collected properly)
                node_info = np_info_map.get(id(child_node))
                if node_info:
                    # Recursively build the hierarchy for its children
                    grandchildren = self._build_hierarchy_from_tree(child_node, np_info_map)
                    hierarchy_entry: NPNodeHierarchy = {'info': node_info, 'children': grandchildren}
                    child_hierarchies.append(hierarchy_entry)
                else:
                    # This should ideally not happen if _collect_np_nodes_info worked correctly
                    logger.warning(f"NP node found in tree traversal ({self._tree_to_text(child_node)}) but missing from np_info_map. Skipping.")
                    # Still traverse deeper in case its children were collected
                    child_hierarchies.extend(self._build_hierarchy_from_tree(child_node, np_info_map))
            else:
                # This child is not an NP, but its descendants might be.
                # Recursively call to find NPs deeper within this non-NP child.
                child_hierarchies.extend(self._build_hierarchy_from_tree(child_node, np_info_map))

        return child_hierarchies


    # --- Validation Logic ---
    
    def _is_valid_np(self, np_span: Span, structures: List[str]) -> bool:
        """
        Validate a noun phrase Span based on configuration.

        Args:
            np_span: The spaCy Span object for the NP.
            structures: List of structural analysis results (needed for filters).

        Returns:
            bool: True if the NP span is valid, False otherwise.
        """
        # Basic text check (should be redundant if span is valid, but safe)
        if not np_span or not np_span.text.strip():
            return False
        
        # Length check (using number of tokens in the span)
        np_length = len(np_span)
        if self.min_length is not None and np_length < self.min_length:
            logger.debug(f"NP span '{np_span.text}' rejected: length {np_length} < min_length {self.min_length}")
            return False
        if self.max_length is not None and np_length > self.max_length:
            logger.debug(f"NP span '{np_span.text}' rejected: length {np_length} > max_length {self.max_length}")
            return False

        # Pronoun check
        if not self.accept_pronouns:
            if np_length == 1:
                 try:
                     # Use the span's token's POS tag
                     if np_span[0].pos_ == "PRON":
                         logger.debug(f"NP span '{np_span.text}' rejected: pronoun (accept_pronouns=False)")
                         return False
                 except IndexError:
                     logger.warning(f"IndexError checking POS for single-token span '{np_span.text}'. Allowing.")
                     pass # Should not happen if len is 1, but allow if it does
                 except Exception as e:
                    logger.error(f"Error checking pronoun POS for span '{np_span.text}': {e}", exc_info=True)
                    # Be conservative: reject if we can't reliably check pronoun status
                    return False
            
        # Structure filter check
        if self.structure_filters:
            # Apply structure filter only if structures were computed (relevant for validation)
            if not structures:
                 # This case should ideally only happen if validation is called without prior analysis
                 # which might occur internally, but structure filter cannot be applied.
                 # If filters are set, implicitly this means it's not valid yet based on structure.
                 # However, the calling logic should ensure analysis happens first if filters are set.
                 # Let's assume the caller handles this. If structures are empty, it means no
                 # relevant structures were found, so filtering proceeds based on that.
                 logger.debug(f"NP span '{np_span.text}' considered for structure filter {self.structure_filters}, but no structures were provided/found for analysis.")
                 # Fall through to check if *any* structure matches (which will be false if structures is empty)

            if not any(s in self.structure_filters for s in structures):
                logger.debug(f"NP span '{np_span.text}' rejected: structures {structures} not in filter list {self.structure_filters}")
                return False
                
        # Passed all checks
        return True

    # --- Processing and Formatting ---

    def _process_np_node_info(self, hierarchy_node: NPNodeHierarchy, base_id: str, include_metadata: bool, level: int = 1) -> Dict:
        """
        Processes a node from the NP hierarchy (built from the NLTK tree).
        Validates using the spaCy Span, performs analysis on the Span,
        but uses the reliable text from the tree for the final output.
        Prunes the branch if the Span mapping failed or validation fails.

        Args:
            hierarchy_node: Dictionary {'info': NPNodeInfo, 'children': List[NPNodeHierarchy]}.
            base_id: ID string for this NP.
            include_metadata: Whether to compute and include metadata.
            level: Current hierarchy level.

        Returns:
            Dict: Formatted NP dictionary for final output, or {} if pruned.
        """
        np_info = hierarchy_node.get("info")
        if not np_info:
             logger.error(f"Missing 'info' key in _process_np_node_info for id {base_id}")
             return {} 
             
        np_text: str = np_info.get("text")
        np_span: Optional[Span] = np_info.get("span")

        # --- CRUCIAL CHECK: Require a valid Span for processing ---
        if np_span is None:
            logger.debug(f"Pruning NP node (text: '{np_text}') at id {base_id} because spaCy Span mapping failed.")
            return {} # Cannot validate or analyze without a Span

        # --- Validation & Analysis (using the Span) ---
        structures = []
        # Analyze if metadata is requested OR if structure filters are active
        needs_analysis = include_metadata or bool(self.structure_filters)
        if needs_analysis:
            structures = self.analyzer.analyze_single_np(np_span)
            logger.debug(f"Analyzed span '{np_span.text}' (id {base_id}): {structures}")

        # Validate using the Span and computed structures
        if not self._is_valid_np(np_span, structures):
            logger.debug(f"NP span '{np_span.text}' (id {base_id}) failed validation (length/pronoun/structure) and was pruned.")
            return {} # Prune this branch if validation fails

        # --- Build Output Dictionary (using reliable text, Span for metadata) ---
        ordered_dict = {
            "noun_phrase": np_text, # Use reliable text from tree
            "id": base_id,
            "level": level
        }
        
        if include_metadata:
            # Structures were already computed if needed
            ordered_dict["metadata"] = {
                "length": len(np_span), # Use Span length (consistent with validation)
                "structures": structures
            }
            
        # --- Process Children Recursively ---
        children_dicts = []
        child_nodes = hierarchy_node.get("children", [])
        valid_child_counter = 0 # Counter for children that pass validation/filtering

        # --- Debug: Check type of hierarchy_node before accessing children ---
        if not isinstance(hierarchy_node, dict):
             logger.error(f"[DEBUG PROCESS] Expected hierarchy_node to be dict, but got {type(hierarchy_node)} for base_id {base_id}")
             # Cannot proceed if hierarchy_node isn't a dict
             ordered_dict["children"] = [] # Assign empty list and return
             return ordered_dict 
             
        child_nodes = hierarchy_node.get("children", []) # Get children list (List[NPNodeHierarchy])
        valid_child_counter = 0 # Counter for children that pass validation/filtering

        # --- Debug: Check type of child_nodes ---
        if not isinstance(child_nodes, list):
             logger.error(f"[DEBUG PROCESS] Expected child_nodes to be list, but got {type(child_nodes)} for base_id {base_id}")
             child_nodes = [] # Attempt to recover by setting to empty list

        for i, child_node_hierarchy in enumerate(child_nodes): 
            # --- Debug: Check type of item being passed to recursive call ---
            if not isinstance(child_node_hierarchy, dict):
                logger.error(f"[DEBUG PROCESS] Expected child_node_hierarchy to be dict, but got {type(child_node_hierarchy)} inside children loop for base_id {base_id}")
                continue # Skip this malformed child

            # ID uses valid child counter to ensure sequential numbering after filtering
            child_id = f"{base_id}.{valid_child_counter + 1}" 
            processed_child = self._process_np_node_info(
                child_node_hierarchy, child_id, include_metadata, level + 1
            )
            # Only append if the recursive call didn't return an empty dict (i.e., wasn't filtered)
            if processed_child:
                 valid_child_counter += 1
                 children_dicts.append(processed_child)
            else:
                 child_text = (child_node_hierarchy.get('info') or {}).get('text', 'N/A')
                 logger.debug(f"Child NP (text: '{child_text}') of parent '{np_text}' (id {base_id}) was filtered out during recursive processing.")
        
        # Add children list as the last field
        ordered_dict["children"] = children_dicts
        
        return ordered_dict


# --- Standalone Functions ---

def extract(text: str, metadata: bool = False, include_nested: bool = False, **kwargs) -> Dict:
    """Standalone function to extract noun phrases."""
    extractor = ANPEExtractor(config=kwargs)
    return extractor.extract(text, metadata=metadata, include_nested=include_nested)

def export(text: str, format: str = "txt", output: Optional[str] = None, 
           metadata: bool = False, include_nested: bool = False, **kwargs) -> str:
    """Standalone function to extract and export noun phrases."""
    extractor = ANPEExtractor(config=kwargs)
    return extractor.export(text, format=format, output=output, metadata=metadata, include_nested=include_nested)