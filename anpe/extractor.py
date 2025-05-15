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
import re
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
            config: Configuration dictionary (excluding logging options).
        """
        logger.debug("ANPEExtractor initializing...")
        # --- Step 1: Initialize Config ---
        self._initialize_config(config)

        # --- Step 2: Setup Environment ---
        self._setup_environment()

        # Keep track of initial user preference for setup/fallback logic
        _user_specified_spacy_initially = self.config.get("spacy_model") is not None
        _user_specified_benepar_initially = self.config.get("benepar_model") is not None

        # --- Step 3: Resolve Model Names ---
        logger.info("Resolving model names...")
        spacy_model_to_load = self._resolve_spacy_model_name()
        benepar_model_to_load = self._resolve_benepar_model_name()

        # --- Steps 4-7: Load and Configure NLP Pipeline ---
        logger.info("Loading and configuring NLP pipeline...")
        try:
            # Step 7 (Moved): Check Transformer Dependency (before loading)
            self._check_transformer_dependency(spacy_model_to_load)

            # Step 4: Load spaCy Pipeline (handles auto-install)
            self._load_spacy_pipeline(spacy_model_to_load, _user_specified_spacy_initially)
            # self.nlp is now assigned if successful

            # Step 5: Configure spaCy Pipeline (sentencizer, etc.)
            self._configure_spacy_pipeline()

            # Step 6: Add Benepar to Pipeline (handles fallback loading)
            loaded_benepar_name = self._add_benepar_to_pipeline(benepar_model_to_load, _user_specified_benepar_initially)
            self._loaded_benepar_model_name = loaded_benepar_name # Store the actually loaded name

        except (RuntimeError, OSError, ValueError, ImportError) as e:
             # Catch specific, critical errors raised by helpers
             logger.critical(f"Fatal error during pipeline initialization: {str(e)}", exc_info=True)
             raise # Re-raise critical errors
        except Exception as e:
            # Catch any other unexpected errors during pipeline setup
            logger.critical(f"Unexpected fatal error during ANPEExtractor initialization: {str(e)}", exc_info=True)
            raise RuntimeError("Unexpected initialization failure") from e

        # --- Initialize Analyzer ---
        try:
            self.analyzer = ANPEAnalyzer(self.nlp)
        except Exception as e:
             logger.critical(f"Fatal error initializing ANPEAnalyzer: {str(e)}", exc_info=True)
             raise RuntimeError("Failed to initialize structure analyzer") from e

        logger.info("ANPEExtractor initialized successfully")

    # --------------------------------------------------------------------------
    # Private Helper Methods for Initialization
    # --------------------------------------------------------------------------

    def _initialize_config(self, config: Optional[Dict[str, Any]]) -> None:
        """Initializes the extractor configuration by merging defaults and user input."""
        logger.debug("Initializing configuration...")
        self.config = DEFAULT_CONFIG.copy()
        if config:
            logger_keys = ['log_level', 'log_file', 'log_dir']
            provided_config = {k: v for k, v in config.items() if k not in logger_keys}
            self.config.update(provided_config)
            logger.debug(f"Updated config with provided keys: {list(provided_config.keys())}")
        else:
             logger.debug("Using default config, no overrides provided.")

        # Set instance attributes from config
        self.min_length = self.config.get("min_length")
        self.max_length = self.config.get("max_length")
        self.accept_pronouns = self.config.get("accept_pronouns")
        self.structure_filters = self.config.get("structure_filters", [])
        self.newline_breaks = self.config.get("newline_breaks")
        logger.debug("Extractor attributes set from config.")

    def _setup_environment(self) -> None:
        """Sets environment variables and warning filters."""
        logger.debug("Setting up environment variables and warning filters...")
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributions")
        warnings.filterwarnings("ignore", message="Passing a tuple of `past_key_values`")
        warnings.filterwarnings("ignore", message=".*default legacy behaviour.*")
        warnings.filterwarnings("ignore", message=".*EncoderDecoderCache.*")
        logger.debug("Environment setup complete.")

    def _resolve_spacy_model_name(self) -> str:
        """Resolves the spaCy model name to load based on config, auto-detect, or default."""
        logger.debug("Resolving spaCy model name...")
        spacy_model_to_use = self.config.get("spacy_model")
        if spacy_model_to_use:
            if spacy_model_to_use not in SPACY_MODEL_MAP:
                 logger.warning(f"Specified spaCy model '{spacy_model_to_use}' is not a recognized alias/name.")
            else:
                 spacy_model_to_use = SPACY_MODEL_MAP.get(spacy_model_to_use, spacy_model_to_use)
            logger.info(f"Using specified spaCy model (resolved): {spacy_model_to_use}")
        else:
            logger.debug("No spaCy model specified, attempting auto-detection...")
            installed_spacy = find_installed_spacy_models()
            spacy_model_to_use = select_best_spacy_model(installed_spacy)
            if spacy_model_to_use:
                logger.info(f"Auto-detected best available spaCy model: {spacy_model_to_use}")
            else:
                spacy_model_to_use = SPACY_MODEL_MAP['md']
                logger.warning(f"Could not auto-detect installed spaCy models. Falling back to default: {spacy_model_to_use}")
        # Update config for reference (consistent with original logic)
        self.config["spacy_model"] = spacy_model_to_use
        logger.debug(f"Resolved spaCy model name to load: {spacy_model_to_use}")
        return spacy_model_to_use

    def _resolve_benepar_model_name(self) -> str:
        """Resolves the Benepar model name to load based on config, auto-detect, or default."""
        logger.debug("Resolving Benepar model name...")
        benepar_model_to_load = self.config.get("benepar_model")
        if not benepar_model_to_load:
            logger.debug("No Benepar model specified, attempting auto-detection...")
            installed_benepar = find_installed_benepar_models()
            benepar_model_to_load = select_best_benepar_model(installed_benepar)
            if benepar_model_to_load:
                logger.info(f"Auto-detected best available Benepar model: {benepar_model_to_load}")
            else:
                benepar_model_to_load = BENEPAR_MODEL_MAP['default']
                logger.warning(f"Could not auto-detect installed Benepar models. Falling back to default: {benepar_model_to_load}")
        else:
            if benepar_model_to_load not in BENEPAR_MODEL_MAP:
                 logger.warning(f"Specified Benepar model '{benepar_model_to_load}' is not a recognized alias/name.")
            else:
                 benepar_model_to_load = BENEPAR_MODEL_MAP.get(benepar_model_to_load, benepar_model_to_load)
            logger.info(f"Using specified Benepar model (resolved): {benepar_model_to_load}")
        # Store the *intended* model back in config for reference
        self.config["benepar_model"] = benepar_model_to_load
        self._loaded_benepar_model_name = "unknown" # Initialize; set properly after loading
        logger.debug(f"Resolved Benepar model name to load: {benepar_model_to_load}")
        return benepar_model_to_load

    def _check_transformer_dependency(self, spacy_model_name: str) -> None:
        """Checks if spacy-transformers library is needed and installed."""
        logger.debug(f"Checking transformer dependency for spaCy model: {spacy_model_name}")
        if spacy_model_name and spacy_model_name.endswith('_trf'):
            logger.info(f"Using a spaCy transformer model ('{spacy_model_name}').")
            try:
                import spacy_transformers
                logger.debug("'spacy-transformers' library found.")
            except ImportError:
                msg = f"Missing required library 'spacy-transformers' for model '{spacy_model_name}'. Please install it: pip install 'spacy[transformers]'"
                logger.critical(msg)
                raise ImportError(msg) # Raise ImportError to be caught
        else:
            logger.debug(f"Not using a spaCy transformer model (resolved model: '{spacy_model_name}').") # Changed log level

    def _load_spacy_pipeline(self, spacy_model_name: str, user_specified_spacy: bool) -> None:
        """Loads the spaCy pipeline, handling errors and potential auto-installation."""
        logger.info(f"Loading spaCy model: '{spacy_model_name}'")
        try:
            self.nlp = spacy.load(spacy_model_name)
            logger.info("spaCy model loaded successfully.")
            return # Success

        except OSError as e:
            logger.error(f"Failed to load spaCy model '{spacy_model_name}'. Is it installed? Error: {e}")
            if spacy_model_name.endswith('_trf'):
                logger.error("Ensure 'spacy-transformers' library is also installed: pip install 'spacy[transformers]'")

            # Attempt setup only if default failed AND wasn't user-specified
            is_default_md = (spacy_model_name == SPACY_MODEL_MAP['md'])
            if not user_specified_spacy and is_default_md:
                logger.info("Attempting to install default models (spaCy 'md', Benepar 'default')...")
                try:
                    if setup_models(spacy_model_alias='md', benepar_model_alias='default'):
                        logger.info("Default models installed. Attempting to reload spaCy model...")
                        self.nlp = spacy.load(spacy_model_name) # Retry load
                        logger.info("spaCy model loaded successfully after installation.")
                        return # Success after install and reload
                    else:
                        logger.critical("Automatic installation of default models failed. Cannot proceed.")
                        raise RuntimeError(f"Failed to load or install required spaCy model '{spacy_model_name}'") from e
                except Exception as setup_e:
                    logger.critical(f"Error during automatic setup or reload attempt: {setup_e}", exc_info=True)
                    raise RuntimeError(f"Failed during automatic setup/reload for spaCy model '{spacy_model_name}'") from setup_e
            else:
                 # Conditions for auto-install not met
                 logger.critical(f"SpaCy model '{spacy_model_name}' could not be loaded. Please ensure it is installed correctly.")
                 raise RuntimeError(f"Failed to load required spaCy model '{spacy_model_name}'") from e
        except Exception as load_e: # Catch other load errors
            logger.critical(f"Unexpected error loading spaCy model '{spacy_model_name}': {load_e}", exc_info=True)
            raise RuntimeError(f"Unexpected error loading spaCy model '{spacy_model_name}'") from load_e

    def _configure_spacy_pipeline(self) -> None:
        """Adds and configures components in the loaded spaCy pipeline (e.g., sentencizer)."""
        logger.debug("Configuring spaCy pipeline components...")
        if not hasattr(self, 'nlp') or self.nlp is None:
            raise RuntimeError("Cannot configure spaCy pipeline: nlp object not initialized.")

        # Add sentencizer
        if "parser" in self.nlp.pipe_names:
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer", before="parser")
                logger.debug("Added 'sentencizer' component before 'parser'.")
            else: logger.debug("'sentencizer' already in pipeline.")
        else:
            if "sentencizer" not in self.nlp.pipe_names:
                self.nlp.add_pipe("sentencizer", first=True)
                logger.debug("Added 'sentencizer' component first.")
            else: logger.debug("'sentencizer' already in pipeline.")

        # Configure sentencizer newline breaks
        try:
            sbd_pipe = self.nlp.get_pipe("sentencizer")
            sbd_component_name = "sentencizer"
            logger.debug(f"Configuring sentence boundary detection using '{sbd_component_name}'.")
            if hasattr(sbd_pipe, 'punct_chars'):
                if self.newline_breaks:
                    # Ensure newlines are treated as sentence boundaries
                    if '\n' not in sbd_pipe.punct_chars:
                        sbd_pipe.punct_chars.add('\n')
                        logger.info(f"Configured '{sbd_component_name}' to treat newlines as sentence boundaries.")
                    else: logger.debug(f"'\n' already present in punct_chars for '{sbd_component_name}'.")
                else:
                    # Ensure newlines are NOT treated as sentence boundaries
                    if '\n' in sbd_pipe.punct_chars:
                        sbd_pipe.punct_chars.remove('\n')
                        logger.info(f"Configured '{sbd_component_name}' to NOT treat newlines as sentence boundaries.")
                    else: logger.debug(f"'\n' not found in punct_chars for '{sbd_component_name}'.")
            else:
                 logger.warning(f"Component '{sbd_component_name}' lacks 'punct_chars'. Cannot configure newline breaks.")
        except KeyError:
            raise RuntimeError("Failed to get 'sentencizer' pipe after adding it.")
        except Exception as e:
            raise RuntimeError("Unexpected error during sentencizer configuration.") from e
        
        # Add custom token matching patterns to enhance sentence boundary detection based on newline_breaks
        try:
            if "token_matcher" not in self.nlp.pipe_names:
                # Add a custom pipe to further enhance sentence boundary handling
                from spacy.language import Language
                
                @Language.component("newline_handler")
                def newline_handler(doc):
                    # If newline_breaks is True, ensure every token that ends with a newline 
                    # also ends a sentence
                    if self.newline_breaks:
                        for i, token in enumerate(doc[:-1]):  # Skip the last token
                            if '\n' in token.text or token.text.endswith('\n'):
                                doc[i].is_sent_start = False  # Reset
                                if i + 1 < len(doc):
                                    doc[i + 1].is_sent_start = True
                    return doc
                
                self.nlp.add_pipe("newline_handler", after="sentencizer")
                logger.debug("Added custom 'newline_handler' pipe to enhance newline handling.")
        except Exception as e:
            logger.warning(f"Could not add custom newline handler: {str(e)}")
        
        logger.debug("spaCy pipeline component configuration complete.")

    def _add_benepar_to_pipeline(self, benepar_model_to_load: str, user_specified_benepar: bool) -> str:
        """Adds the Benepar component to the spaCy pipeline, handling fallback loading."""
        logger.debug(f"Ensuring Benepar component is added (attempting '{benepar_model_to_load}')...")
        if not hasattr(self, 'nlp') or self.nlp is None:
            raise RuntimeError("Cannot add Benepar component: spaCy nlp object not initialized.")

        if "benepar" in self.nlp.pipe_names:
            logger.info("Benepar component already found in pipeline.")
            try: # Attempt to get loaded name from existing pipe
                return self.nlp.get_pipe("benepar").cfg.get("model", "existing/unknown")
            except: return "existing/unknown"

        loaded_model_name = "unknown"
        try:
            logger.info(f"Attempting to add Benepar component with model: '{benepar_model_to_load}'")
            self.nlp.add_pipe("benepar", config={"model": benepar_model_to_load})
            logger.info(f"Benepar component ('{benepar_model_to_load}') added successfully.")
            loaded_model_name = benepar_model_to_load

        except ValueError as e:
            logger.warning(f"Failed to load Benepar model '{benepar_model_to_load}': {e}")
            e_str = str(e).lower()
            is_load_error = "can't find package" in e_str or "can't load plugin" in e_str
            default_benepar_model = BENEPAR_MODEL_MAP['default']

            if is_load_error and not user_specified_benepar:
                logger.info(f"Attempting fallback to default Benepar model: '{default_benepar_model}'")
                try:
                    self.nlp.add_pipe("benepar", config={"model": default_benepar_model})
                    logger.info(f"Benepar component ('{default_benepar_model}') added successfully via fallback.")
                    loaded_model_name = default_benepar_model
                except Exception as fallback_e: # Catch ValueError or others during fallback
                    logger.critical(f"Fallback attempt with default Benepar model '{default_benepar_model}' also failed: {fallback_e}")
                    raise RuntimeError(f"Failed to add required Benepar component. Attempted '{benepar_model_to_load}' and fallback '{default_benepar_model}'.") from fallback_e
            else: # Primary load failed, and no fallback applicable/attempted
                logger.critical(f"Failed to load Benepar model '{benepar_model_to_load}'. Cannot proceed.")
                raise RuntimeError(f"Failed to add required Benepar component with model '{benepar_model_to_load}'.") from e
        except ImportError as imp_e:
            msg = f"Missing required library 'benepar'. Please install it (e.g., pip install benepar)."
            logger.critical(msg)
            raise RuntimeError(msg) from imp_e
        except Exception as ex: # Catch other add_pipe errors
            logger.critical(f"Unexpected error adding Benepar component '{benepar_model_to_load}': {ex}")
            raise RuntimeError(f"Unexpected error adding Benepar component for model '{benepar_model_to_load}'") from ex

        # If we reach here, loading must have succeeded (directly or via fallback)
        if loaded_model_name == "unknown": # Sanity check
             raise RuntimeError("Benepar loading logic error: loaded_model_name not set after successful load.")

        return loaded_model_name

    # --------------------------------------------------------------------------
    # Public Methods (extract, export) - Kept Original
    # --------------------------------------------------------------------------

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
        processed_text = self._preprocess_text_for_benepar(text)

        # Parse the entire text with the full pipeline (spaCy + Benepar)
        try:
            logger.debug("Parsing text with spaCy+Benepar...")
            doc = self.nlp(processed_text)
            logger.debug("Text parsed successfully.")
        except AssertionError as e:
            # This is likely the Benepar retokenization assertion error
            logger.warning(f"Benepar tokenization assertion error: {str(e)}. Trying alternative preprocessing...")
            
            # Try a more aggressive preprocessing approach
            try:
                logger.debug("Attempting alternative preprocessing while respecting newline_breaks setting...")
                
                # Process text differently based on newline_breaks setting
                if self.newline_breaks:
                    # When newlines should break sentences, replace them with period+space
                    # This helps preserve sentence boundaries while fixing tokenization
                    simplified_text = text.replace('\n', '. ').replace('..', '.')
                else:
                    # When newlines shouldn't break sentences, replace with simple spaces
                    simplified_text = text.replace('\n', ' ')
                    
                # Normalize spacing and ensure a trailing space
                simplified_text = ' '.join(simplified_text.split()) + ' '
                
                logger.debug("Attempting parse with simplified text...")
                doc = self.nlp(simplified_text)
                logger.info("Successfully parsed text with alternative preprocessing.")
            except Exception as alt_e:
                logger.error(f"Alternative preprocessing also failed: {str(alt_e)}")
                # Instead of disabling Benepar, raise a helpful error
                error_msg = (
                    "ANPE could not process this text with Benepar's constituency parser despite "
                    "multiple preprocessing attempts. This is likely due to:\n"
                    "1. Irregular sentence boundaries or chaotic text structure\n"
                    "2. Unusual newline patterns or inconsistent paragraph formatting\n"
                    "3. Text that exceeds Benepar's tokenization capabilities\n\n"
                    "Please try providing more cleanly structured text with standard sentence "
                    "patterns and consistent formatting. For best results, ensure text has:\n"
                    "- Clear sentence boundaries (periods followed by spaces)\n"
                    "- Consistent paragraph breaks\n"
                    "- Standard punctuation and spacing\n\n"
                    f"Note: Your 'newline_breaks' setting is currently set to {self.newline_breaks}. "
                    "Complex text formatting may sometimes require standardizing newlines, which can "
                    "override this setting. If newline handling is critical for your use case, "
                    "consider pre-processing your text to have clearer sentence boundaries."
                )
                logger.critical(error_msg)
                raise ValueError(f"Text preprocessing failure: {error_msg}")
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

    def _normalize_text_for_matching(self, text: str) -> str:
        """Normalizes a text string (from a tree leaf or a spaCy token) for robust comparison."""
        original_text_for_logging = text # Keep the very original for logging comparison
        
        stripped_text = text.strip()

        # Comprehensive normalization dictionary
        # This maps various quote styles and PTB symbols to a canonical form.
        NORMALIZATION_MAP = {
            # PTB symbols to standard text
            "-LRB-": "(", "-RRB-": ")",
            "-LSB-": "[", "-RSB-": "]",
            "-LCB-": "{", "-RCB-": "}",
            "``":    '"',  # PTB double open
            "''":    '"',  # PTB double close
            "`":     "'",  # PTB single open/apostrophe
        }

        if stripped_text in NORMALIZATION_MAP:
            normalized_output = NORMALIZATION_MAP[stripped_text]
            # Log if the final output differs from the original input due to map lookup
            if normalized_output != original_text_for_logging:
                logger.debug(f"Text normalization (map): '{original_text_for_logging}' -> '{normalized_output}'")
            return normalized_output
        else:
            # No rule in map applied, the output is the stripped text
            # Log if stripping alone made a change compared to the original input
            if stripped_text != original_text_for_logging:
                logger.debug(f"Text normalization (strip): '{original_text_for_logging}' -> '{stripped_text}'")
            return stripped_text

    def _preprocess_text_for_benepar(self, text: str) -> str:
        """
        Preprocess text to ensure compatibility with Benepar tokenization and parsing.
        This is particularly important for handling newlines which can cause 
        retokenization errors in Benepar.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            str: Preprocessed text ready for Benepar parsing
        """
        if not text:
            return text
        
        # Always normalize line endings first
        # Original: processed_text = text.replace('\r\n', '\n')
        # Ensure both \r\n and \r are handled, standardizing to \n
        processed_text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        if not self.newline_breaks:
            # When newlines AREN'T sentence breaks, we ensure sentences continue across lines
            logger.debug("Preprocessing text with newline_breaks=False (treating newlines as spaces)")
            
            # Step 1: Identify and preserve paragraph breaks (double newlines)
            processed_text = processed_text.replace('\n\n', '\uE000')  # Unicode private use character
            
            # Step 2: Carefully handle single newlines to maintain sentence continuity
            # Convert all remaining newlines to spaces to ensure sentences continue
            processed_text = processed_text.replace('\n', ' ')
            
            # Step 3: Restore paragraph breaks with actual double newlines
            processed_text = processed_text.replace('\uE000', '\n\n')
            
            # Step 4: Fix any accidental period-space-space sequences from previous operations
            processed_text = processed_text.replace('.  ', '. ')
            
            # Step 5: Normalize multiple spaces and ensure clean spacing (original logic)
            processed_text = ' '.join(processed_text.split())

        else: # self.newline_breaks is True
            # When newlines ARE sentence breaks, we make them explicit sentence boundaries
            logger.debug("Preprocessing text with newline_breaks=True (treating newlines as sentence boundaries)")
            
            # Step 1: Ensure each newline creates a clear sentence boundary by adding a period if needed
            lines = processed_text.split('\n')
            
            # Step 2: Process each line to ensure it ends with a proper sentence boundary
            for i in range(len(lines)):
                current_line = lines[i].strip() # Use a different variable name to avoid confusion
                if not current_line:  # Skip empty lines
                    lines[i] = "" # Ensure empty lines become truly empty for join
                    continue
                    
                # If the line doesn't end with a sentence-ending punctuation, add a period
                if not current_line[-1] in '.?!':
                    lines[i] = current_line + '.'
                else:
                    lines[i] = current_line # Keep as is if already ends with punctuation
            
            # Step 3: Rejoin with newlines to preserve the line structure
            processed_text = '\n'.join(lines)
            
            # Step 4: Ensure proper spacing before and after newlines for tokenization (original logic)
            processed_text = processed_text.replace('\n', ' \n ')
            
            # Step 5: Clean up multiple spaces (original logic)
            processed_text = ' '.join(processed_text.split())

        # --- NEW: Space-padding for specific punctuation ---
        # This is applied *after* the main newline and space normalization from the blocks above.
        logger.debug(f"Text before punctuation padding (snippet): '{processed_text[:200]}...'")
        
        temp_padded_text = processed_text
        
        # For opening brackets: ensure space before (unless at start of text or already has space)
        # but don't add space after
        for char_code in ['(', '[', '{']:
            # Pattern: match char_code not preceded by space or start of string
            # Lookbehind assertion (?<!) checks what comes before without including it in match
            temp_padded_text = re.sub(f'(?<!^)(?<!\s){re.escape(char_code)}', f' {char_code}', temp_padded_text)
            
        # For closing brackets: ensure space after (unless already has space or at end of text)
        # but don't add space before
        for char_code in [')', ']', '}']:
            # Pattern: match char_code not followed by space or end of string
            # Lookahead assertion (?!) checks what comes after without including it in match
            temp_padded_text = re.sub(f'{re.escape(char_code)}(?!\s)(?!$)', f'{char_code} ', temp_padded_text)
        
        # --- Final space cleanup and ensure a single trailing space if text is not empty ---
        # This consolidates spaces from padding and normalizes overall spacing.
        final_text = ' '.join(temp_padded_text.split())

        if final_text: # Only add trailing space if the string is not empty
            final_text += ' '
        
        logger.debug(f"Preprocessed text for Benepar (snippet): '{final_text[:200]}...'")
        return final_text

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


    def _find_token_indices_for_leaves(self, leaves: List[str], sent_tokens: List[Any]) -> Optional[Tuple[int, int]]:
        """
        Attempts to find the start and end token indices in a spaCy sentence
        that correspond to a given sequence of leaves from an NLTK tree.

        Args:
            leaves: List of leaf strings from the NLTK tree node.
            sent_tokens: List of spaCy Token objects for the sentence.

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
                normalized_leaf = self._normalize_text_for_matching(leaves[j])
                normalized_token_text = self._normalize_text_for_matching(sent_tokens[i + j].text)
                if normalized_leaf != normalized_token_text:
                    match = False
                    break
            if match:
                # Found a match, return start and *inclusive* end index
                return i, i + num_leaves - 1

        # If no match found after checking all possible start positions
        logger.warning(
            f"Could not map leaves {leaves} to tokens in sentence.\n"
            f"This might be due to the presence of special characters or punctuation that are not being normalized properly.\n"
            f"Span creation might fail and related noun phrases might not be present in the output.\n"
            f"ANPE expects clean texts. Please try normalize the text manually."
        )
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
                indices = self._find_token_indices_for_leaves(leaves, sent_tokens)

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
                    indices = self._find_token_indices_for_leaves(leaves, sent_tokens)
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