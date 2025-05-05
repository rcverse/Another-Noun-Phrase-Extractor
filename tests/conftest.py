import pytest
import spacy
from spacy.tokens import Doc, Span, Token
from spacy.vocab import Vocab
from unittest.mock import Mock, MagicMock
from collections import namedtuple

# Basic spaCy Vocab fixture
@pytest.fixture(scope="session")
def spacy_vocab():
    """Provides a basic spaCy Vocab."""
    return Vocab()

# Helper function to create a mock Token (can be enhanced later)
def create_mock_token(
    text: str,
    pos: str = "",
    tag: str = "",
    dep: str = "",
    head: "Token" = None, # Note: Head is tricky, might need adjustment
    idx: int = 0,
    i: int = 0, # Token index in doc
    vocab: Vocab = None,
    doc: Doc = None,
) -> Mock:
    """Creates a mock spaCy Token with basic attributes."""
    mock_token = Mock(spec=Token)
    mock_token.text = text
    mock_token.lemma_ = text.lower() # Simple lemma
    mock_token.pos_ = pos
    mock_token.tag_ = tag
    mock_token.dep_ = dep
    mock_token.head = head if head else mock_token # Default head to self if not provided
    mock_token.idx = idx # Character offset
    mock_token.i = i # Token index
    mock_token.vocab = vocab
    mock_token.doc = doc
    mock_token.children = [] # Initialize children
    # Add text_with_ws attribute (simple mock: text + space)
    mock_token.text_with_ws = text + " " 
    # Make subtree iterable (simplistic: just the token itself)
    mock_token.subtree = [mock_token]

    # Basic string representation
    mock_token.__str__ = lambda self: self.text
    mock_token.__repr__ = lambda self: f"MockToken({self.text!r})"

    # Allow adding children (useful for head/child relationships)
    def add_child(parent, child):
        if not hasattr(parent, 'children') or parent.children is None:
            parent.children = []
        parent.children.append(child)
        child.head = parent # Set the head relationship

    mock_token.add_child = lambda child: add_child(mock_token, child)
    
    # Ensure head attribute is properly mocked if it's self
    if mock_token.head == mock_token:
        mock_token.head = mock_token

    return mock_token

# Helper fixture to create a mock Doc (useful container for Tokens/Spans)
@pytest.fixture
def mock_doc_factory(spacy_vocab):
    """Provides a factory to create mock Doc objects."""
    def _create_mock_doc(tokens: list[Mock]) -> Mock:
        mock_doc = Mock(spec=Doc)
        mock_doc.vocab = spacy_vocab
        mock_doc.tokens = tokens # Store the mock tokens
        mock_doc.text = " ".join(t.text for t in tokens)
        
        # Link tokens to the doc
        start_char = 0
        for i, token in enumerate(tokens):
            token.doc = mock_doc
            token.i = i
            token.idx = start_char
            start_char += len(token.text) + 1 # Add 1 for space

        # Make the doc iterable like a real Doc
        mock_doc.__iter__ = lambda: iter(tokens)
        mock_doc.__len__ = lambda: len(tokens)
        # mock_doc.__getitem__ = lambda self, key: tokens[key] # Incorrect: This returns a list slice

        # Basic Span creation - assumes slicing by token index
        def _create_span(key): # Changed signature to accept 'key'
             start: int | None = None
             end: int | None = None
             
             if isinstance(key, slice):
                 start = key.start
                 end = key.stop
                 # Handle None values in slice
                 if start is None:
                     start = 0
                 if end is None:
                     end = len(tokens) 
             elif isinstance(key, int):
                 # Handle single index access if needed, e.g., return a single token
                 # For now, let's raise an error as the tests seem to expect spans
                 raise TypeError(f"Mock Doc access expects a slice, got integer index: {key}")
             else:
                 raise TypeError(f"Mock Doc access expects a slice or integer, got {type(key)}")

             # Ensure start and end are valid integers after processing slice/index
             if not isinstance(start, int) or not isinstance(end, int):
                 # This shouldn't happen if logic above is correct, but as a safeguard:
                 raise TypeError(f"Calculated indices are not integers: start={start}, end={end}")
             
             # Ensure indices are within bounds (adjust end for slicing)
             if start < 0: start = 0
             if end > len(tokens): end = len(tokens)
             if start > end: start = end # Handle invalid slice like [5:2] -> empty

             # --- Original _create_span logic using valid 'start' and 'end' ---
             # Remove spec=Span to allow setting magic methods more freely
             mock_span = Mock() 
             span_tokens = tokens[start:end] # Use processed start/end
             # --- Ensure integer attributes are set --- 
             mock_span.start = int(start)
             mock_span.end = int(end)
             # --- End integer attributes --- 
             mock_span.label_ = "" # Default label, can be set later if needed
             mock_span.text = " ".join(t.text for t in span_tokens) # Use double quotes for string
             mock_span.doc = mock_doc
             mock_span.start_char = span_tokens[0].idx if span_tokens else (tokens[start].idx if start < len(tokens) else 0) # Simplified fallback for start_char
             mock_span.end_char = (span_tokens[-1].idx + len(span_tokens[-1].text)) if span_tokens else mock_span.start_char
             mock_span.root = span_tokens[0] if span_tokens else None # Simplistic root
             
             # Configure magic method behaviors using configure_mock
             mock_span.configure_mock(
                 **{
                     # Accept optional self argument
                     "__iter__": lambda self=None: iter(span_tokens),
                     "__len__": lambda self=None: len(span_tokens),
                     # __getitem__ needs careful handling as Mock might pass self implicitly
                     "__getitem__": Mock(side_effect=lambda key: span_tokens[key]), 
                     # Accept optional self argument
                     "__str__": lambda self=None: mock_span.text, 
                     # __contains__ expects 'item'
                     "__contains__": lambda item: any(item is t for t in span_tokens),
                     # Corrected: __contains__ needs self and item
                     "__contains__": lambda self, item: any(item is t for t in span_tokens),
                 }
             )
              
             return mock_span
             
        mock_doc.char_span = Mock(return_value=None) # Mock char_span initially
        # Ensure the __getitem__ attribute exists before assigning side_effect
        mock_doc.__getitem__ = Mock()
        # Correctly assign the span creation function to __getitem__ via side_effect
        mock_doc.__getitem__.side_effect = _create_span

        return mock_doc
    return _create_mock_doc

# Fixture for the Analyzer instance (using a dummy nlp object for now)
@pytest.fixture
def analyzer():
    """Provides an instance of ANPEAnalyzer for testing."""
    from anpe.utils.analyzer import ANPEAnalyzer
    # Create a dummy nlp object, as analyzer doesn't use it for parsing anymore
    mock_nlp = Mock() 
    return ANPEAnalyzer(mock_nlp) 

# --- Fixtures for Extractor Tests ---

@pytest.fixture
def default_config():
    """Provides a copy of the default ANPE configuration."""
    # Import locally to avoid circular dependency if conftest grows large
    from anpe.config import DEFAULT_CONFIG
    return DEFAULT_CONFIG.copy()

@pytest.fixture
def minimal_config():
    """Provides a minimal config dictionary."""
    return {
        # Models will be mocked usually
    } 

@pytest.fixture(autouse=True)
def mock_universal_setup(mocker):
    """Auto-used fixture for universal mocks (logging, filesystem)."""
    mocker.patch('pathlib.Path.mkdir') # Avoid creating log dirs
    # Prevent actual model setup/download attempts globally unless overridden
    mocker.patch('anpe.extractor.setup_models', return_value=False)

# Fixture moved from test_extractor.py and modified for punct_chars
@pytest.fixture
def mock_model_interactions(mocker):
    """Mocks interactions related to spaCy/Benepar model loading and selection."""
    # Mock spacy.load to return a mock nlp object
    mock_nlp = MagicMock()
    mock_nlp.pipe_names = [] # Start with an empty pipeline
    mock_nlp.config = {"nlp": {"pipeline": []}}
    mock_nlp.meta = {"name": "mock_spacy_model"}
    
    # Store mock sentencizer pipe to modify its attributes later
    mock_sentencizer_pipe_instance = MagicMock()
    # Initialize punct_chars as a real set
    mock_sentencizer_pipe_instance.punct_chars = set({'.', '?', '!'}) # Use a real set

    # Mock add_pipe to modify the mock pipe_names
    def mock_add_pipe(component_name, **kwargs):
        # Simulate adding the pipe name
        if kwargs.get('first'):
            mock_nlp.pipe_names.insert(0, component_name)
        elif kwargs.get('before') in mock_nlp.pipe_names:
            idx = mock_nlp.pipe_names.index(kwargs['before'])
            mock_nlp.pipe_names.insert(idx, component_name)
        else:
            mock_nlp.pipe_names.append(component_name)
        
        # Return the specific mock for sentencizer if requested
        if component_name == "sentencizer":
            return mock_sentencizer_pipe_instance
        else:
            # Return a generic mock for other pipes like benepar
            return MagicMock()
        
    mock_nlp.add_pipe = MagicMock(side_effect=mock_add_pipe)
    # Mock get_pipe to return the specific sentencizer mock when asked
    # Ensure it returns the *same instance* that has the mutable set
    mock_nlp.get_pipe = MagicMock(return_value=mock_sentencizer_pipe_instance)

    mocker.patch('spacy.load', return_value=mock_nlp)
    # Mock benepar import and model loading within add_pipe simulation
    mocker.patch('benepar.BeneparComponent', return_value=MagicMock())
    # Mock the find/select functions to avoid file system/network access
    mocker.patch('anpe.extractor.find_installed_spacy_models', return_value=[])
    # mocker.patch('anpe.extractor.select_best_spacy_model', return_value='mock_spacy_model') # DISABLED - Tests should mock this if needed
    mocker.patch('anpe.extractor.find_installed_benepar_models', return_value=[])
    # mocker.patch('anpe.extractor.select_best_benepar_model', return_value='mock_benepar_model') # DISABLED - Tests should mock this if needed

    # Return the mock nlp instance for potential direct assertions
    return mock_nlp

# Helper to create a mock Token with minimal fuss inside tests
MockTokenInfo = namedtuple("MockTokenInfo", ["text", "pos", "tag", "dep"])

# Updated mock_doc_factory to handle sentences and parse strings
@pytest.fixture
def mock_nlp_processing_factory(spacy_vocab):
    """
    Provides a factory to create a mock nlp object that returns a pre-defined mock Doc.
    Handles multiple sentences within a single Doc object.
    """
    
    # --- Reusable Span Creation Logic ---
    def _create_span_mock(key, all_tokens, target_doc):
        start: int | None = None
        end: int | None = None
        
        if isinstance(key, slice):
            start = key.start
            end = key.stop
            if start is None: start = 0
            if end is None: end = len(all_tokens)
        elif isinstance(key, int):
             # Return single token mock if indexed
             if 0 <= key < len(all_tokens):
                 return all_tokens[key]
             else:
                 raise IndexError(f"Mock index {key} out of range for {len(all_tokens)} tokens.")
        else:
             raise TypeError(f"Mock access expects slice or int, got {type(key)}")

        # Validate slice indices
        if not (0 <= start <= end <= len(all_tokens)):
             raise IndexError(f"Mock span slice indices ({start}:{end}) out of range for {len(all_tokens)} tokens.")

        mock_span = Mock() # No spec needed
        span_tokens = all_tokens[start:end]
        mock_span.text = " ".join(t.text for t in span_tokens)
        mock_span.start = int(start)
        mock_span.end = int(end)
        mock_span.doc = target_doc
        mock_span.vocab = spacy_vocab
        mock_span.start_char = span_tokens[0].idx if span_tokens else (all_tokens[start].idx if start < len(all_tokens) else 0)
        mock_span.end_char = (span_tokens[-1].idx + len(span_tokens[-1].text)) if span_tokens else mock_span.start_char
        mock_span.root = span_tokens[0] if span_tokens else None 
        
        # Use configure_mock for magic methods
        mock_span.configure_mock(
            **{
                "__iter__": lambda self=None: iter(span_tokens),
                "__len__": lambda self=None: len(span_tokens),
                # Slicing a span returns a sub-span relative to the *doc* tokens
                # Indexing a span should return the token at that index within the span
                "__getitem__": Mock(side_effect=lambda k: span_tokens[k] if isinstance(k, int) else _create_span_mock(k, all_tokens, target_doc)), 
                "__str__": lambda self=None: mock_span.text,
                "__contains__": lambda self, item: any(item is t for t in span_tokens),
            }
        )
        return mock_span

    # --- Factory for the main nlp mock ---
    def _create_mock_nlp(list_of_token_data_per_sent: list[list[MockTokenInfo]], list_of_parse_strings: list[str | None]):
        mock_nlp_instance = MagicMock()
        mock_nlp_instance.meta = {"name": "mock_nlp_for_extract"}

        # --- Create a SINGLE mock Doc containing ALL tokens ---
        mock_doc = MagicMock() 
        all_tokens = []
        global_token_index = 0
        char_offset = 0
        
        # Flatten token data and create all token mocks, linking to the single doc
        for sent_token_data in list_of_token_data_per_sent:
            for token_info in sent_token_data:
                token_mock = create_mock_token(
                    vocab=spacy_vocab, 
                    doc=mock_doc, 
                    i=global_token_index, 
                    idx=char_offset, 
                    **token_info._asdict()
                )
                all_tokens.append(token_mock)
                char_offset += len(token_mock.text) + 1 # Simple space addition
                global_token_index += 1
        
        mock_doc.tokens = all_tokens # For internal reference if needed
        mock_doc.__iter__ = Mock(return_value=iter(all_tokens))
        mock_doc.__len__ = Mock(return_value=len(all_tokens))
        mock_doc.text = " ".join(t.text for t in all_tokens)
        mock_doc.text_with_ws = " ".join(t.text_with_ws for t in all_tokens)
        
        # --- Mock Doc Slicing (__getitem__) ---
        # Use the _create_span_mock helper defined above
        mock_doc.__getitem__ = Mock(side_effect=lambda key: _create_span_mock(key, all_tokens, mock_doc))

        # --- Create Mock Sentences (Spans) ---
        mock_sents = []
        current_token_index = 0
        for i, sent_token_data in enumerate(list_of_token_data_per_sent):
            sent_start_token = current_token_index
            sent_end_token = current_token_index + len(sent_token_data)
            
            # Create the sentence span using the mock doc's slicing
            mock_sent = mock_doc[sent_start_token:sent_end_token]
            
            # --- Mock char_span on the sentence span --- 
            # It should return a span created using our helper, similar to slicing
            def mock_char_span_on_sent(start_char, end_char, alignment_mode="contract"): # Match signature
                # Find token indices corresponding to char span (simplistic mapping)
                # This is tricky; for tests, let's assume char spans align well with token indices
                # Find start token index within the sentence span
                start_token_sent_idx = -1
                for tok_idx_in_sent, token in enumerate(mock_sent):
                    token_start_char_in_sent = token.idx - mock_sent.start_char
                    if token_start_char_in_sent >= start_char:
                        start_token_sent_idx = tok_idx_in_sent
                        break
                if start_token_sent_idx == -1: # Not found
                    return None 
                
                # Find end token index within the sentence span (token that *ends* after end_char)
                end_token_sent_idx = -1
                for tok_idx_in_sent, token in enumerate(mock_sent):
                    token_end_char_in_sent = token.idx + len(token.text) - mock_sent.start_char
                    if token_end_char_in_sent >= end_char:
                        end_token_sent_idx = tok_idx_in_sent
                        break
                if end_token_sent_idx == -1: # Not found
                   return None 

                # Convert sentence-relative token indices to doc-relative indices
                start_token_doc_idx = mock_sent.start + start_token_sent_idx
                # Add 1 because slicing is exclusive, but char_span end is inclusive
                end_token_doc_idx = mock_sent.start + end_token_sent_idx + 1 

                # Use the main mock doc slicing to create the span
                if start_token_doc_idx < end_token_doc_idx:
                    return mock_doc[start_token_doc_idx:end_token_doc_idx]
                else:
                    return None # Invalid span

            mock_sent.char_span = Mock(side_effect=mock_char_span_on_sent)
            # ------------------------------------------

            # Add benepar extension to the sentence span
            parse_string = list_of_parse_strings[i] if i < len(list_of_parse_strings) else None
            if parse_string is not None:
                 mock_sent.has_extension = Mock(return_value=True)
                 mock_sent._ = Mock()
                 mock_sent._.parse_string = parse_string
            else:
                 mock_sent.has_extension = Mock(return_value=False)
                 
            mock_sents.append(mock_sent)
            current_token_index = sent_end_token
            
        mock_doc.sents = mock_sents

        # Configure the main mock nlp object's call behavior
        mock_nlp_instance.return_value = mock_doc # nlp(text) returns the doc
        mock_nlp_instance.pipe.return_value = [mock_doc] # nlp.pipe returns list containing the doc
        
        return mock_nlp_instance

    return _create_mock_nlp 