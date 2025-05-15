# ANPE Extractor Workflow

This document explains the step-by-step workflow of the noun phrase (NP) extraction process implemented in the `anpe/extractor.py` module of the `another-noun-phrase-extractor` project. We will follow a single example sentence through the entire process:

**Example Sentence:** `"Mr. Harrison, the lead developer known for his meticulous work, presented the system's core concepts."`

We assume the extractor is called with `metadata=True` and `include_nested=True` for this walkthrough to cover the most comprehensive path.

## Overview

The core goal is to identify noun phrases within a given text, optionally including nested phrases and metadata (length, structural analysis). The process leverages spaCy for initial text processing and tokenization, Benepar for constituency parsing (identifying grammatical structures like NP), NLTK for manipulating parse trees, and custom logic to integrate these components, validate results, and handle potential inconsistencies.

## Step 0: Initialization (`ANPEExtractor.__init__`)

* **Purpose:** Set up the extractor instance, load configuration, initialize logging, set up the environment, and load the required NLP models (spaCy and Benepar) with appropriate configurations.
* **Input:** An optional `config` dictionary overriding default settings. This dictionary can include options like `min_length`, `accept_pronouns`, `spacy_model`, `benepar_model`, etc. Logging-specific keys like `log_level` or `log_dir` from this `config` are ignored by the extractor's internal config but are typically handled by an external logging setup.
  * *Example*: `{'min_length': 2, 'accept_pronouns': False, 'spacy_model': 'lg'}`
* **Processing:**
  1. **Initialize Configuration (`_initialize_config`)**:
     * Copies the `DEFAULT_CONFIG`.
     * If a `config` dictionary is provided, it updates the defaults with the user's settings, excluding any logging-specific keys (e.g., `log_level`, `log_dir`).
     * Sets instance attributes like `self.min_length`, `self.max_length`, `self.accept_pronouns`, `self.structure_filters`, and `self.newline_breaks` from the merged configuration.
  2. **Setup Environment (`_setup_environment`)**:
     * Sets environment variables: `TRANSFORMERS_NO_ADVISORY_WARNINGS = 'true'` and `TOKENIZERS_PARALLELISM = 'false'`.
     * Applies `warnings.filterwarnings` to ignore specific `UserWarning` messages from `torch.distributions` and those related to `past_key_values`, `default legacy behaviour`, and `EncoderDecoderCache`.
  3. **Resolve Model Names**:
     * **SpaCy (`_resolve_spacy_model_name`)**:
       * If `config['spacy_model']` is specified, it's resolved (e.g., alias 'lg' to 'en_core_web_lg').
       * Otherwise, it attempts to auto-detect the best installed spaCy model using `find_installed_spacy_models()` and `select_best_spacy_model()`.
       * If auto-detection fails, it defaults to `SPACY_MODEL_MAP['md']` (e.g., 'en_core_web_md').
       * The resolved model name is stored back in `self.config['spacy_model']`.
       * *Example Log (auto-detection)*:
         ```
         INFO - Auto-detected best available spaCy model: en_core_web_md
         ```
     * **Benepar (`_resolve_benepar_model_name`)**:
       * Similar logic for `config['benepar_model']`, using `find_installed_benepar_models()`, `select_best_benepar_model()`, and defaulting to `BENEPAR_MODEL_MAP['default']` (e.g., 'benepar_en3').
       * The resolved model name is stored in `self.config['benepar_model']`.
       * *Example Log (user-specified)*:
         ```
         INFO - Using specified Benepar model (resolved): benepar_en3
         ```
  4. **Check Transformer Dependency (`_check_transformer_dependency`)**:
     * If the resolved `spacy_model_name` ends with `_trf`, it checks if the `spacy-transformers` library is installed. If not, it raises an `ImportError`.
  5. **Load spaCy Pipeline (`_load_spacy_pipeline`)**:
     * Attempts to load the resolved spaCy model using `spacy.load(spacy_model_name)`. This creates the core `self.nlp` object.
     * If `OSError` occurs (e.g., model not found):
       * If the model was the default 'md' model and *not* specified by the user, it attempts to auto-install default models (`setup_models(spacy_model_alias='md', benepar_model_alias='default')`) and retries loading.
       * Otherwise, or if auto-installation fails, it raises a `RuntimeError`.
     * *Example Log*:
       ```
       INFO - Loading spaCy model: 'en_core_web_md'
       INFO - spaCy model loaded successfully.
       ```
  6. **Configure spaCy Pipeline (`_configure_spacy_pipeline`)**:
     * Ensures `self.nlp` is initialized.
     * **Sentencizer**: Adds the `sentencizer` component to the pipeline if not already present (before "parser" or first).
     * **Newline Handling for Sentencizer**: Modifies the `sentencizer.punct_chars` based on `self.newline_breaks`. If `True`, adds `\\n`; if `False`, removes `\\n`.
     * **Custom Newline Handler**: Adds a custom spaCy component `newline_handler` after the `sentencizer`. If `self.newline_breaks` is `True`, this component ensures that tokens ending with a newline also mark the start of a new sentence.
     * *Example Log*:
       ```
       DEBUG - Added 'sentencizer' component before 'parser'.
       INFO - Configured 'sentencizer' to treat newlines as sentence boundaries.
       DEBUG - Added custom 'newline_handler' pipe to enhance newline handling.
       ```
  7. **Add Benepar to Pipeline (`_add_benepar_to_pipeline`)**:
     * Ensures `self.nlp` is initialized.
     * If "benepar" is not already in `self.nlp.pipe_names`:
       * Attempts to add Benepar: `self.nlp.add_pipe("benepar", config={"model": benepar_model_to_load})`.
       * If `ValueError` occurs (e.g., Benepar model not found) and the model was *not* specified by the user, it attempts a fallback to the default Benepar model (`BENEPAR_MODEL_MAP['default']`).
       * If fallback also fails or was not applicable, it raises a `RuntimeError`.
       * If `ImportError` occurs (Benepar library not installed), it raises a `RuntimeError`.
     * Stores the name of the actually loaded Benepar model in `self._loaded_benepar_model_name`.
     * *Example Log*:
       ```
       INFO - Attempting to add Benepar component with model: 'benepar_en3'
       INFO - Benepar component ('benepar_en3') added successfully.
       ```
  8. **Initialize Analyzer**:
     * Initializes the structural analyzer: `self.analyzer = ANPEAnalyzer(self.nlp)` (from `anpe.utils.analyzer`). This analyzer is used later for generating metadata about NP structures.
* **Output:** A configured `ANPEExtractor` instance with:
  * `self.config`: The final configuration dictionary.
    * *Example (debug run)*: `{'min_length': None, 'max_length': None, 'accept_pronouns': True, 'structure_filters': [], 'newline_breaks': True, 'spacy_model': 'en_core_web_md', 'benepar_model': 'benepar_en3', ...}`
  * `self.nlp`: A loaded spaCy `Language` object containing the full pipeline (tokenizer, tagger, parser, sentencizer, custom newline_handler, benepar, etc.).
  * `self.analyzer`: An `ANPEAnalyzer` instance ready for use.
  * `self._loaded_benepar_model_name`: The name of the Benepar model that was successfully loaded.
  * Other config attributes (`min_length`, `accept_pronouns`, etc.) set as instance variables.

## Step 1: Text Preprocessing & Parsing (`ANPEExtractor.extract`)

* **Purpose:** Prepare the input text for robust parsing by the spaCy+Benepar pipeline, execute the parsing, and handle potential parsing errors with fallback strategies.
* **Input:**
  * `text`: The raw input text string.
    * *Example*: `"Mr. Harrison, the lead developer known for his meticulous work, presented the system's core concepts."`
  * `metadata`: Boolean flag. (*Example*: `True`)
  * `include_nested`: Boolean flag. (*Example*: `True`)
* **Processing:**
  1. **Early Exit for Empty Input**: If the input `text` is empty or only whitespace, the method returns an empty result structure immediately, including a timestamp, processing duration (near zero), and basic configuration details.
  2. **Text Preprocessing (`_preprocess_text_for_benepar`)**: The input `text` is passed to `self._preprocess_text_for_benepar(text)` for normalization to improve compatibility with Benepar's tokenization.
     * Standardizes line endings (`\\r\\n`, `\\r` to `\\n`).
     * **If `self.newline_breaks` is `False` (treat newlines as spaces)**:
       * Preserves paragraph breaks (`\\n\\n`) by temporarily replacing them with a placeholder (`\\uE000`).
       * Converts remaining single newlines (`\\n`) to spaces.
       * Restores paragraph breaks.
       * Fixes accidental double spaces after periods (e.g., `.  ` to `. `).
       * Normalizes multiple spaces into single spaces.
     * **If `self.newline_breaks` is `True` (treat newlines as sentence boundaries)**:
       * Splits text into lines.
       * Ensures each non-empty line ends with sentence-ending punctuation (appends `.` if missing).
       * Rejoins lines with `\\n`.
       * Adds spaces around newlines (e.g., `text\\nmore` becomes `text \\n more`).
       * Normalizes multiple spaces into single spaces.
     * **Punctuation Padding (Applied After Newline Logic)**:
       * Uses regular expressions to ensure specific punctuation marks (`(`, `[`, `{`, `)`, `]`, `}`) are appropriately space-padded to aid tokenization. For example, `text(more)` might become `text ( more )` (actual padding is more nuanced, e.g., space before opening, space after closing).
     * **Final Cleanup**: Normalizes all spaces to single spaces and adds a single trailing space if the processed text is not empty.
     * The result is `processed_text`.
     * *Example Log*: `DEBUG - Preprocessed text for Benepar (snippet): 'Mr. Harrison , the lead developer known for his meticulous work , presented the system's core concepts. ...'`
  3. **Parse with spaCy+Benepar Pipeline**: The `processed_text` is parsed: `doc = self.nlp(processed_text)`.
     * *Example Log*: `DEBUG - Parsing text with spaCy+Benepar...`
  4. **Error Handling during Parsing**:
     * **If `AssertionError` occurs** (often a Benepar tokenization issue):
       * Logs a warning.
       * Attempts an **alternative preprocessing** strategy:
         * If `self.newline_breaks` is `True`: `text.replace('\\n', '. ').replace('..', '.')` (tries to make newlines explicit sentence ends).
         * If `self.newline_breaks` is `False`: `text.replace('\\n', ' ')` (treats newlines as simple spaces).
         * Normalizes spacing and adds a trailing space to this `simplified_text`.
         * Retries parsing: `doc = self.nlp(simplified_text)`.
         * *Example Log (if fallback attempted)*: `INFO - Successfully parsed text with alternative preprocessing.`
       * If alternative preprocessing also fails, a `ValueError` is raised with a detailed message guiding the user to provide cleaner text, explaining Benepar's limitations with irregular structures or newline patterns.
     * **If any other `Exception` occurs** during parsing, it's caught, logged, and a `RuntimeError` is raised.
     * *Example Log (successful parse on first try)*: `DEBUG - Text parsed successfully.`
* **Output:**
  * `doc`: A spaCy `Doc` object if parsing was successful. This object contains tokens, and each sentence within it (`sent in doc.sents`) will have a `._.parse_string` attribute from Benepar if parsing was successful for that sentence.
    * *Example Doc (properties)*: Contains tokens like `Mr.`, `Harrison`, `,`, `the`, `lead`, ..., `concepts`, `.`. Each token has `.text`, `.pos_`, `.tag_`, `.dep_`, etc. The `doc` object itself has a `.sents` property.
    * *Example Sentence Property*: The first (and only) sentence `Span` (`sent = list(doc.sents)[0]`) has `sent.text` = `"Mr. Harrison, the lead developer known for his meticulous work, presented the system's core concepts."` (after spaCy tokenization). Crucially, it also has `sent._.parse_string` from Benepar, e.g., `"(S (NP (NNP Mr.) (NNP Harrison)) (, ,) (NP (NP (DT the) (JJ lead) (NN developer)) (VP (VBN known) (PP (IN for) (NP (PRP$ his) (JJ meticulous) (NN work))))) (, ,) (VP (VBD presented) (NP (NP (DT the) (NN system) (POS 's)) (JJ core) (NNS concepts))) (. .))"` (This is an example parse string; actual output may vary slightly with models/versions).

## Step 2: Sentence Iteration & Tree Creation (`ANPEExtractor.extract` loop)

* **Purpose:** Process each sentence identified by spaCy to extract its constituency parse tree (from Benepar) for further noun phrase identification.
* **Input:** The spaCy `Doc` object from Step 1.
  * *Example*: The `doc` object containing our single sentence.
* **Processing (Looping through `doc.sents`):**
  1. For each `sent` (a spaCy `Span` representing a sentence) in the `doc`:
     * *Example*: Our single sentence `Span`.
     * *Example Log*: `DEBUG - [extract loop] Processing sentence 0: 'Mr. Harrison, the lead developer known for his met...'`
  2. **Check for Parse Data**: Verifies if the sentence `Span` has Benepar's parse data attached (`sent.has_extension("parse_string")`). If not, it logs a warning and skips this sentence.
  3. **Retrieve Parse String**: Gets the raw constituency parse string: `parse_string = sent._.parse_string`.
     * *Example*: `"(S (NP...) ...)"` as shown in Step 1 Output.
  4. **Validate Parse String**: Checks if `parse_string` is empty or only whitespace. If so, logs a warning and skips the sentence, as an empty parse string cannot be processed.
  5. **Create NLTK Tree**: Attempts to convert the `parse_string` into an NLTK `Tree` object: `constituents_tree = Tree.fromstring(parse_string)`.
     * **Error Handling**: If `Tree.fromstring()` raises a `ValueError` (e.g., malformed parse string) or any other `Exception`, it logs a warning/error with details about the problematic sentence and its raw parse string, then skips to the next sentence.
     * *Example Log (on success)*: `DEBUG - [extract loop] Successfully created constituents_tree. Type: <class 'nltk.tree.tree.Tree'>`
  6. Logs completion of tree creation and proceeds to NP extraction helpers (detailed in Step 3).
* **Output (Per Successfully Processed Sentence):**
  * `sent`: The current spaCy `Span` object for the sentence.
  * `constituents_tree`: An NLTK `Tree` object representing the constituency parse of the sentence.

## Step 3: NP Extraction (Conditional Logic)

The workflow now diverges based on the `include_nested` flag. Our example follows `include_nested=True`.

### Step 3a: Nested Extraction (`include_nested=True`)

#### Step 3a.1: Collect All NP Node Info (`_collect_np_nodes_info`)

* **Purpose:** Traverse the sentence's NLTK parse tree, identify *every* node labeled "NP", extract its text as yielded by its leaves, and critically, attempt to map this grammatical unit precisely to a corresponding spaCy `Span` in the original sentence. This mapping is essential for later validation and analysis using spaCy's richer token information.
* **What's an NLTK Tree Node?** When Benepar parses the sentence (Step 1), it creates a tree structure (like a family tree) representing the sentence's grammar. Think of it like diagramming a sentence. Each point in this tree is a 'node'. Nodes have labels (like 'S' for Sentence, 'VP' for Verb Phrase, 'NP' for Noun Phrase). The bottom-most nodes are the actual words (called 'leaves'). An 'NP' node in this tree represents a chunk of the sentence that Benepar identified as having the grammatical structure of a noun phrase.
* **Input:**
  * `constituents_tree`: The NLTK `Tree` from Step 2.
  * `sent`: The spaCy `Span` (representing the sentence) from Step 2.
* **Processing:** Recursively traverses the `constituents_tree`. For each node encountered:
  1. If the node is a string (a leaf), traversal stops for that branch.
  2. If the node is labeled "NP":
     * **Extract Text from Tree**: The node's plain text is extracted using `_tree_to_text(node).strip()`. This method joins all leaf (terminal) strings under the current NP node. The resulting `np_text` can sometimes lack spaces between words if the tree doesn't explicitly encode them (e.g., "Mr.Harrison"). This `np_text` is primarily for reference; the `span.text` (if mapping is successful) is preferred for final output.
     * If `np_text` is empty, the NP node is skipped.
     * **Map to spaCy Span**: This is a multi-step process to link the Benepar NP node to a spaCy `Span`:
       * Gets the leaves of the NP node: `leaves = node.leaves()` (e.g., `['Mr.', 'Harrison']`).
       * Gets the spaCy tokens for the sentence: `sent_tokens = list(sent)`.
       * **Find Token Indices (`_find_token_indices_for_leaves`)**: Attempts to find the start and end token indices within `sent_tokens` that match the sequence of `leaves`.
         * **Normalization is Key (`_normalize_text_for_matching`)**: Before comparing, both each leaf from the Benepar tree and each spaCy token's text are passed through `_normalize_text_for_matching`. This helper standardizes various representations, such as converting Penn Treebank symbols (e.g., "-LRB-", "``", "''") to their common textual equivalents (e.g., "(", '"', '"'), and strips leading/trailing whitespace. This step is critical for aligning Benepar's (often PTB-style) tokenization with spaCy's tokenization.
         * If a matching sequence of normalized tokens is found, it returns `(start_token_idx, end_token_idx)`.
         * If no match is found (logged with a warning), `indices` will be `None`.
       * **Create Span from Indices (`_get_span_from_token_indices`)**: If `indices` were found:
         * It attempts to create a spaCy `Span` using `sent.char_span(start_char, end_char, alignment_mode="contract")`. This method is robust as it uses character offsets.
         * If `char_span` returns `None` (e.g., due to alignment issues), it falls back to direct token slicing: `span = sent[start_token_idx : end_token_idx + 1]`.
         * If span creation fails by either method, `np_span` remains `None` (logged with a warning).
         * If `indices` were not found, `np_span` remains `None`.
     * **Store Information**: Appends a dictionary (`NPNodeInfo`) to `np_info_list`:
       `{'node': node, 'text': np_text, 'span': np_span}`.
       * `'node'`: A reference to the NLTK `Tree` object for this NP.
       * `'text'`: The text derived from `_tree_to_text(node).strip()`.
       * `'span'`: The corresponding spaCy `Span` object, or `None` if mapping failed.
  3. Continues traversal for all children of the current node.
* **Output:**
  * `np_nodes_info`: A `List[NPNodeInfo]`. Contains info for *all* NP nodes found in the tree, regardless of whether their mapping to a spaCy `Span` was successful.
    * *Example Log*: `DEBUG - _collect_np_nodes_info found 7 NP nodes.`
    * *Example `np_nodes_info` (showing structure for a few items based on debug run)*:

    ```python
    [
      # Info for "(NP (NNP Mr.) (NNP Harrison))"
      {'node': Tree('NP', [Tree('NNP', ['Mr.']), Tree('NNP', ['Harrison'])]),
       'text': 'Mr.Harrison', # Note: NLTK leaves() joins without space
       'span': <spaCy Span object for "Mr. Harrison">}, # Span text is correct

      # Info for "(NP (NP ...) (VP ...))" - Top level of the complex NP
      {'node': Tree('NP', [Tree('NP', [...]), Tree('VP', [...])]),
       'text': 'theleaddeveloperknownforhismeticulouswork',
       'span': <spaCy Span object for "the lead developer known for his meticulous work">},

      # Info for "(NP (DT the) (JJ lead) (NN developer))"
      {'node': Tree('NP', [Tree('DT', ['the']), Tree('JJ', ['lead']), Tree('NN', ['developer'])]),
       'text': 'theleaddeveloper',
       'span': <spaCy Span object for "the lead developer">},

      # Info for "(NP (PRP$ his) (JJ meticulous) (NN work))"
      {'node': Tree('NP', [Tree('PRP$', ['his']), Tree('JJ', ['meticulous']), Tree('NN', ['work'])]),
       'text': 'hismeticulouswork',
       'span': <spaCy Span object for "his meticulous work">},

      # Info for "(NP (NP ...) (JJ core) (NNS concepts))" - Top level system NP
       {'node': Tree('NP', [Tree('NP', [...]), Tree('JJ', ['core']), Tree('NNS', ['concepts'])]),
       'text': "thesystem'scoreconcepts",
       'span': <spaCy Span object for "the system 's core concepts">},

      # Info for "(NP (DT the) (NN system) (POS 's))"
       {'node': Tree('NP', [Tree('DT', ['the']), Tree('NN', ['system']), Tree('POS', ["'s"])]),
       'text': "thesystem's",
       'span': <spaCy Span object for "the system 's">},

      # Info for the outermost NP containing "Mr. Harrison..."
      # This depends on the exact tree structure, let's assume it includes the comma etc.
       {'node': <NLTK Tree object for the full subject phrase>,
       'text': "Mr.Harrisontheleaddeveloperknownforhismeticulouswork", # Joined leaves
       'span': <spaCy Span object for "Mr. Harrison , the lead developer known for his meticulous work ,">} # Span includes commas
    ]
    ```

    *(Note: `<...>` represent the actual objects. The `text` field shown is generated by joining the NLTK leaves (e.g., `'Mr.Harrison'`). However, the `span` object contains the text as tokenized by spaCy (e.g., `'Mr. Harrison'`), which is used for subsequent validation and becomes the final `noun_phrase` output in Step 4a.)*. Nodes where span mapping fails (`span` is `None`) are still collected but pruned later in Step 4a.

#### Step 3a.2: Create NP Info Map (In `extract` method)

* **Purpose:** Convert the list of `NPNodeInfo` dictionaries (from Step 3a.1) into a map (Python dictionary) for efficient lookup of an NP node's information using its unique ID.
* **How the map is created:** This happens directly within the `extract` method if `include_nested` is true, after `_collect_np_nodes_info` returns `np_nodes_info`. It uses a dictionary comprehension:
  `np_info_map = {id(info['node']): info for info in np_nodes_info}`
  * For each `info` dictionary in the `np_nodes_info` list, it takes the unique memory address of the NLTK `Tree` object (`id(info['node'])`) associated with that NP node.
  * This ID becomes a key in the `np_info_map`.
  * The value associated with this key is the *entire original `info` dictionary* itself (`{'node': ..., 'text': ..., 'span': ...}`).
* **Input:**
  * `np_nodes_info`: The `List[NPNodeInfo]` from Step 3a.1.
* **Output:**
  * `np_info_map`: A `Dict[int, NPNodeInfo]`. This map allows quick retrieval of an NP's collected data (its NLTK node, tree-derived text, and potentially mapped spaCy Span) using the NLTK node's unique ID.
    * *Example Log (Implied)*: If `_collect_np_nodes_info` found 7 nodes, this map will have 7 entries.

#### Step 3a.3: Build Hierarchy from Tree (`_build_hierarchy_from_tree`)

* **Purpose:** Reconstruct the hierarchical (parent-child) relationships of the noun phrases based on their structural positions within the original NLTK parse tree. This step uses the `np_info_map` (from Step 3a.2) to ensure that only NP nodes for which information was successfully collected (and a spaCy Span mapping was at least attempted) are formally included in the hierarchy.
* **How the hierarchy is built & info preserved:** This function, `_build_hierarchy_from_tree(current_processing_node, np_info_map)`, recursively navigates the NLTK tree structure of the sentence (`constituents_tree` when first called from the `extract` method).
  * **Base Case:** If `current_processing_node` is a string (a leaf), it returns an empty list (no hierarchy from a leaf).
  * **Recursive Step (for each `child_node` of `current_processing_node`):**
    1. Skips if `child_node` is a string (leaf).
    2. **If `child_node` is an NP (i.e., `child_node.label() == "NP"`):**
       * It attempts to retrieve the collected information for this `child_node` using its unique ID: `node_info = np_info_map.get(id(child_node))`.
       * **If `node_info` is found in the map:** This means the NP node was successfully processed in Step 3a.1. A new hierarchy entry is created for it.
         * It recursively calls `_build_hierarchy_from_tree(child_node, np_info_map)` on the `child_node` itself. This is crucial: it explores the children *of this current NP node* to find any NPs nested directly inside it. The result is `grandchildren`.
         * A dictionary representing this NP in the hierarchy is formed: `{'info': node_info, 'children': grandchildren}`. The `node_info` (containing the original NLTK node object, its tree-derived text, and its mapped spaCy Span or `None`) is directly embedded.
         * This new hierarchy dictionary is added to the list of hierarchies being built for the `current_processing_node`.
       * **If `node_info` is NOT found in the map:** This is an unusual case, indicating an NP node present in the tree was not captured in `np_nodes_info` (logged as a warning). Even so, the function *still* recursively calls `_build_hierarchy_from_tree(child_node, np_info_map)` on this `child_node` and extends the current list with any valid NP hierarchies found deeper within this problematic branch. This ensures that descendants of a missed NP are not automatically lost.
    3. **If `child_node` is NOT an NP (e.g., a 'VP' - Verb Phrase, 'PP' - Prepositional Phrase):**
       * It does *not* create a hierarchy entry for this non-NP `child_node` itself.
       * However, it *still recursively calls* `_build_hierarchy_from_tree(child_node, np_info_map)` on this `child_node`. This is vital because an NP might be grammatically nested inside another phrase type (e.g., an NP inside a PP like "the cat [PP in [NP the hat]]"). This ensures all branches of the NLTK tree are explored for potential NPs.
  * The function returns a list of NP hierarchy dictionaries found under `current_processing_node`.
* **Initial Call from `extract` method:** `sentence_hierarchy = self._build_hierarchy_from_tree(constituents_tree, np_info_map)`.
* **Input (for the initial call from `extract`):**
  * `constituents_tree`: The NLTK `Tree` for the entire sentence (from Step 2).
  * `np_info_map`: The map from Step 3a.2.
* **Output (from the initial call, `sentence_hierarchy`):**
  * A `List[NPNodeHierarchy]`. This list contains hierarchies starting from the *highest-level* NP nodes found directly under the sentence's root or within intermediate non-NP phrases, for which valid info was found in `np_info_map`.
    * *Example Log (from `extract` after processing all sentences and accumulating results)*: `DEBUG - [extract loop] Built hierarchy with 2 top-level nodes.` (This log refers to the number of items in `sentence_hierarchy` for one sentence).
    * *Example `sentence_hierarchy` for our single example sentence (simplified structure, actual NLTK nodes represented by placeholders):*

    ```python
    [
      # Hierarchy for the first top-level NP found (e.g., "Mr. Harrison , the lead developer known for his meticulous work ,")
      {
        'info': { # NPNodeInfo for this top-level NP
            'node': <NLTK Tree object for the full subject phrase>,
            'text': "Mr.Harrisontheleaddeveloperknownforhismeticulouswork", # Text from _tree_to_text
            'span': <Span "Mr. Harrison , the lead developer known for his meticulous work ,"> # Mapped spaCy Span
        },
        'children': [
          # Child hierarchy for "Mr. Harrison"
          {
            'info': {'node': <Tree for Mr. Harrison>, 'text': "Mr.Harrison", 'span': <Span "Mr. Harrison">},
            'children': []
          },
          # Child hierarchy for "the lead developer known for his meticulous work"
          {
            'info': {'node': <Tree for the lead dev...work>, 'text': "theleaddeveloperknownforhismeticulouswork", 'span': <Span "the lead developer known for his meticulous work">},
            'children': [
               # Grandchild for "the lead developer"
               {
                 'info': {'node': <Tree for the lead dev>, 'text': "theleaddeveloper", 'span': <Span "the lead developer">},
                 'children': []
               },
               # Grandchild for "his meticulous work"
               {
                 'info': {'node': <Tree for his work>, 'text': "hismeticulouswork", 'span': <Span "his meticulous work">},
                 'children': []
               }
            ]
          }
        ]
      },
      # Hierarchy for the second top-level NP found (e.g., "the system 's core concepts")
      {
        'info': { # NPNodeInfo for this NP
            'node': <Tree for the system...concepts>,
            'text': "thesystem'scoreconcepts",
            'span': <Span "the system 's core concepts">
        },
        'children': [
           # Child hierarchy for "the system 's"
           {
             'info': {'node': <Tree for the system 's>, 'text': "thesystem's", 'span': <Span "the system 's">},
             'children': []
           }
        ]
      }
    ]
    ```

    *(Note: The structure reflects NPs found in the tree that were also present in `np_info_map`. The text in `'info'['text']` comes from `_tree_to_text`, while `'info'['span']` is the (potentially more accurately spaced) spaCy Span object.)*

#### Rationale for Separating Collection/Mapping and Hierarchy Building

- You might notice that Step 3a.1 (`_collect_np_nodes_info`) combined with Step 3a.2 (creating `np_info_map`) handles finding all potential NP nodes and attempting the critical mapping to spaCy Spans. Then, Step 3a.3 (`_build_hierarchy_from_tree`) uses this map to reconstruct the grammatical hierarchy.

- This separation, while involving processing related to the tree structure in multiple stages, offers advantages:
  1. **Separation of Concerns:** It cleanly separates the complex task of *identifying, text-extracting, and mapping* NP nodes to spaCy Spans (which involves normalization and robust span creation logic) from the task of *reconstructing the grammatical hierarchy* based on the original NLTK tree structure and the pre-validated map.
  2. **Handling Mapping Failures Gracefully:** The spaCy Span mapping in Step 3a.1 can sometimes fail (resulting in `'span': None`). By performing this mapping first and storing all attempts in `np_info_map`, the hierarchy building step (3a.3) can directly use this information. It ensures that the `'info'` field in the hierarchy always reflects what was (or wasn't) achieved during mapping. The final decision to prune a branch due to a `None` span or other validation failures happens later, in Step 4a (`_process_np_node_info`).
  3. **Code Clarity:** This separation generally leads to more modular code where each part has a distinct responsibility, aiding in understanding and maintenance.

- While a single-pass approach is conceivable, the current method prioritizes logical clarity and robust handling of potential mapping issues over minimizing the number of tree traversals, which is generally not the main performance bottleneck.

- The design prioritizes a clear logical flow and robust handling of the critical mapping stage over minimizing distinct passes over tree-related data, as the mapping itself is a more complex operation.

#### Step 3a.4: Process NP Node Hierarchies (`_process_np_node_info`)

* **Purpose:** Convert `NPNodeHierarchy` items into the final output dictionary format, performing validation and analysis primarily on the associated spaCy `Span`. Filters out nodes where the `span` is `None` (mapping failed) or fails validation (`_is_valid_np`).
* **Input (per top-level hierarchy node):** `hierarchy_node` from `sentence_hierarchy` (Step 3a.3), `base_id` (e.g., "1"), `include_metadata`, `current_level`.
  * *Example (first call)*: The first dict in `sentence_hierarchy` list above, `base_id="1"`, `include_metadata=True`, `current_level=1`.
* **Processing (Recursive):**
  1. Retrieves `np_info` from the current `hierarchy_node`.
  2. **Validation & Pruning:**
     * Checks if `np_info['span']` exists. If not, prunes this node and its children (returns `{}`).
     * Calls `self._is_valid_np(np_info['span'])` which checks:
       * Length against `min_length`/`max_length`.
       * Whether it's a pronoun and `accept_pronouns` is `False`.
       * If invalid, prunes this node and its children (returns `{}`).
  3. **Data Extraction:**
     * Uses `np_span.text` as the primary `noun_phrase` string (more reliable than joined leaves).
     * Calculates `length` from `len(np_span)`.
  4. **Metadata (if `include_metadata`):**
     * Calls `self.analyzer.analyze_single_np(np_span)` to get structural types.
       * *Example Log (for span "Mr. Harrison...")*: `DEBUG - Analysis complete for 'Mr. Harrison, ...': Structures=['determiner', 'adjectival_modifier', 'compound', 'possessive', 'appositive', 'reduced_relative_clause']`
       * *Example Log (for span "Mr. Harrison")*: `DEBUG - Analysis complete for 'Mr. Harrison': Structures=['compound']`
       * Applies `structure_filters` if provided in config.
  5. **Recursion:** Calls `_process_np_node_info` for each `child` in `hierarchy_node['children']`, incrementing `level` and creating nested IDs (e.g., "1.1", "1.2"). Filters out empty `{}` results from pruned children.
  6. **Assembly:** Constructs the result dictionary for the current node.
* **Output (Aggregated across all top-level calls):** `processed_results`: A list of fully formatted dictionaries for the valid, top-level NPs and their valid children.
  * *Example Output (Matching the final debug script output structure)*:

  ```python
  [
    { # Top-level NP 1
      "noun_phrase": "Mr. Harrison , the lead developer known for his meticulous work ,", # From Span.text
      "id": "1",
      "level": 1,
      "metadata": {
        "length": 12, # From Span
        "structures": ["determiner", "adjectival_modifier", "compound", "possessive", "appositive", "reduced_relative_clause"] # From Analyzer
      },
      "children": [
        { # Child 1.1
          "noun_phrase": "Mr. Harrison", "id": "1.1", "level": 2,
          "metadata": {"length": 2, "structures": ["compound"]}, "children": []
        },
        { # Child 1.2
          "noun_phrase": "the lead developer known for his meticulous work", "id": "1.2", "level": 2,
          "metadata": {"length": 8, "structures": ["determiner", "adjectival_modifier", "possessive", "reduced_relative_clause"]},
          "children": [
            { # Grandchild 1.2.1
              "noun_phrase": "the lead developer", "id": "1.2.1", "level": 3,
              "metadata": {"length": 3, "structures": ["determiner", "adjectival_modifier"]}, "children": []
            },
            { # Grandchild 1.2.2
              "noun_phrase": "his meticulous work", "id": "1.2.2", "level": 3,
              "metadata": {"length": 3, "structures": ["adjectival_modifier", "possessive"]}, "children": []
            }
          ]
        }
      ]
    },
    { # Top-level NP 2
      "noun_phrase": "the system 's core concepts", "id": "2", "level": 1,
      "metadata": {"length": 5, "structures": ["determiner", "compound", "possessive"]},
      "children": [
        { # Child 2.1
          "noun_phrase": "the system 's", "id": "2.1", "level": 2,
          "metadata": {"length": 3, "structures": ["determiner"]}, "children": []
        }
      ]
    }
  ]
  ```

## Step 4: Post-Processing & Validation

Our example follows the nested path (Step 4a).

### Step 4a: Processing Hierarchies (`_process_np_node_info`) - (If `include_nested=True`)

* **Purpose:** This crucial recursive function takes a node from the NP hierarchy (built in Step 3a.3), validates it using its associated spaCy `Span`, performs structural analysis if needed, and formats it (and its valid children) into the final dictionary structure for output. It prunes branches where the NP fails validation or if its spaCy `Span` mapping was unsuccessful.
* **Input (per call, initially for each top-level hierarchy node from `all_sentence_hierarchies` collected in `extract`):**
  * `hierarchy_node`: A dictionary from the list created by `_build_hierarchy_from_tree`, e.g., `{'info': NPNodeInfo, 'children': List[NPNodeHierarchy]}`.
  * `base_id`: The ID string for the current NP (e.g., "1" for top-level, "1.1" for a child).
  * `include_metadata`: Boolean flag indicating whether to compute and include metadata.
  * `level`: Current depth in the hierarchy (e.g., 1 for top-level).
* **Processing (Recursive):**
  1. **Retrieve NP Info**: Extracts `np_info = hierarchy_node.get('info')`. If `np_info` is missing (should not happen with well-formed `hierarchy_node`), logs an error and returns an empty dictionary `{}` (pruning this node).
  2. Extracts `np_text` (tree-derived text, e.g., "Mr.Harrison") and `np_span` (mapped spaCy Span, e.g., `Span("Mr. Harrison")`) from `np_info`.
  3. **Pruning based on Span Mapping**: If `np_span` is `None` (meaning the NLTK NP node could not be successfully mapped to a spaCy Span in Step 3a.1), this node and its entire branch are pruned. A debug message `Pruning NP node (text: '{np_text}') at id {base_id} because spaCy Span mapping failed.` is logged, and an empty dictionary `{}` is returned.
  4. **Structural Analysis (if needed for validation or output)**:
     * `structures = []`.
     * Analysis is performed if `include_metadata` is `True` OR if `self.structure_filters` (from config) is non-empty. This condition is `needs_analysis = include_metadata or bool(self.structure_filters)`.
     * If `needs_analysis` is true, `structures = self.analyzer.analyze_single_np(np_span)` is called. This uses the `ANPEAnalyzer` to determine syntactic patterns (e.g., `['compound']`, `['determiner']`) within the `np_span`.
     * *Example Log (for span "Mr. Harrison", id "1.1")*: `DEBUG - Analyzed span 'Mr. Harrison' (id 1.1): ['compound']`
  5. **Validation (`_is_valid_np`)**: The `np_span` and its `structures` (which might be empty if analysis wasn't needed and no filters are set) are passed to `self._is_valid_np(np_span, structures)`.
     * This validation function checks:
       * If `np_span` itself is falsy or `np_span.text.strip()` is empty (returns `False`).
       * If `len(np_span)` (number of tokens in the spaCy Span) violates `self.min_length` (if set) or `self.max_length` (if set) from config.
       * If `self.accept_pronouns` (config) is `False`: checks if `np_span` is a single pronoun (i.e., `len(np_span) == 1 and np_span[0].pos_ == "PRON"`).
       * If `self.structure_filters` (config) is non-empty: checks if at least one of the identified `structures` is present in the `self.structure_filters` list. If `structures` is empty and filters are set, this check effectively fails.
     * If `_is_valid_np` returns `False`, a debug message like `NP span '{np_span.text}' (id {base_id}) failed validation... and was pruned.` is logged, and this node (and its entire branch) is pruned by returning an empty dictionary `{}`.
  6. **Build Output Dictionary (if valid and not pruned)**:
     * `"noun_phrase"`: Crucially, this is set to `np_text` (the potentially un-spaced text derived directly from the NLTK tree leaves in Step 3a.1, e.g., "Mr.Harrison").
     * `"id"`: Set to the current `base_id`.
     * `"level"`: Set to the current `level`.
     * **Metadata (if `include_metadata` is `True`)**: An object `"metadata"` is added:
       * `"length"`: This is `len(np_span)` (the number of tokens in the spaCy Span, e.g., 2 for the Span covering "Mr. Harrison"). This provides an accurate token count.
       * `"structures"`: The list of `structures` identified in step 4 (or an empty list if analysis wasn't performed).
  7. **Process Children Recursively**: 
     * Initializes an empty list `children_dicts`.
     * Initializes `valid_child_counter = 0`. This counter is essential for generating sequential child IDs (e.g., "1.1", "1.2") *only for children that pass validation and are not pruned*, ensuring no gaps in numbering (e.g., avoids "1.1", "1.3" if child "1.2" was filtered out).
     * For each `child_node_hierarchy` in `hierarchy_node.get('children', [])` (the children built in Step 3a.3):
       * Generates `child_id` using the `valid_child_counter`: `f"{base_id}.{valid_child_counter + 1}"`.
       * Recursively calls `processed_child = self._process_np_node_info(child_node_hierarchy, child_id, include_metadata, level + 1)`.
       * **If `processed_child` is not an empty dictionary** (i.e., the child was valid and not pruned by the recursive call):
         * Increments `valid_child_counter` (so the *next* valid child gets the next sequential number).
         * Appends `processed_child` to `children_dicts`.
       * Otherwise (if child was pruned), a debug message like `Child NP (text: '{child_text}') of parent '{np_text}' (id {base_id}) was filtered out...` is logged.
  8. **Final Assembly**: The `children_dicts` list (containing only valid, processed children) is added to the current NP's dictionary under the key `"children"`.
  9. Returns the fully assembled dictionary for the current NP node.
* **Aggregation in `extract` method**: The `extract` method iterates through `all_sentence_hierarchies` (which is a list containing the `sentence_hierarchy` lists from each sentence, effectively all top-level hierarchy nodes). For each such top-level `hierarchy_node`:
  * A global `top_level_id_counter` is incremented (e.g., 1, 2, 3...). Note: the example output in Step 3a.3 used sentence-relative IDs for illustration of hierarchy structure; this counter makes IDs unique across the document.
  * `_process_np_node_info` is called with `str(top_level_id_counter)` as the `base_id`.
  * If the result from `_process_np_node_info` is not an empty dictionary (i.e., the top-level NP itself and its branch were not entirely pruned), it's appended to the main `processed_results` list.
  * If a top-level item is pruned, a debug log `Top-level NP hierarchy starting with text '{original_text}' ... was filtered out.` is recorded.
* **Output (The `processed_results` list in the `extract` method):** A list of dictionaries. Each dictionary represents a valid top-level noun phrase and its valid nested children. The structure matches the example shown at the end of Step 3a.3, with the `id` fields reflecting unique top-level numbering (e.g., "1", "2") and dot-notation for children (e.g., "1.1", "1.1.1").

### Step 4b: Formatting Highest-Level Spans (If `include_nested=False`)

*(This path is taken if `include_nested=False` was passed to the `extract` method. It processes the `highest_level_only_spans` from Step 3b.2)*

* **Purpose:** For each highest-level, non-overlapping noun phrase `Span` identified, this step performs validation, optional structural analysis (for metadata or filtering), and formats it into the final flat dictionary structure for output.
* **Input:** `highest_level_only_spans`: A `List[Span]` from Step 3b.2.
* **Processing (within the `extract` method, if `include_nested` is `False`):**
  A `top_level_id_counter` (initialized earlier in `extract`) is used to assign unique IDs.
  For each `np_span` in `highest_level_only_spans`:
  1. **Structural Analysis (if needed for validation or output)**:
     * `structures = []`.
     * Analysis is performed if `metadata` (parameter to `extract`) is `True` OR if `self.structure_filters` (from config) is non-empty.
     * If analysis is needed, `structures = self.analyzer.analyze_single_np(np_span)`.
  2. **Validation (`_is_valid_np`)**: The `np_span` and its (potentially empty) `structures` are passed to `self._is_valid_np(np_span, structures)`.
     * This is the same validation function used in Step 4a, checking length, pronoun status (if `accept_pronouns` is `False`), and structure filters.
  3. **If `np_span` is valid**: 
     * Increments the `top_level_id_counter`.
     * `np_text = np_span.text.strip()`: The noun phrase text is taken directly from the valid spaCy Span's text and stripped of leading/trailing whitespace.
     * **Assemble Basic Dictionary**: Creates an initial dictionary for the NP:
       * `"noun_phrase"`: `np_text`
       * `"id"`: `str(top_level_id_counter)`
       * `"level"`: `1` (as these are all top-level)
     * **Add Metadata (if `metadata` parameter to `extract` is `True`)**: 
       * An object `"metadata"` is added to the dictionary.
       * If `structures` were not computed for the validation step (e.g., if `metadata` is true but no `structure_filters` were set), `structures = self.analyzer.analyze_single_np(np_span)` is called again to ensure they are available for the output.
       * `"length"`: `len(np_span)` (number of tokens in the Span).
       * `"structures"`: The list of `structures` identified.
     * **Add Empty Children List**: `"children": []` is added to maintain a consistent output structure with the nested case, even though there are no children here.
     * The complete dictionary is appended to the main `processed_results` list.
  4. **If `np_span` is not valid**, it is skipped, and no entry is added to `processed_results` for it.
* **Output (The `processed_results` list in the `extract` method):** A list of flat dictionaries. Each dictionary represents a valid, highest-level noun phrase.
  * *Example Output (Conceptual, assuming the example sentence NPs passed validation)*:

  ```python
  [
    {
      "noun_phrase": "Mr. Harrison , the lead developer known for his meticulous work ,",
      "id": "1",
      "level": 1,
      "metadata": {"length": 12, "structures": ["determiner", "adjectival_modifier", ..., "appositive"]},
      "children": []
    },
    {
      "noun_phrase": "the system 's core concepts",
      "id": "2",
      "level": 1,
      "metadata": {"length": 5, "structures": ["determiner", "compound", "possessive"]},
      "children": []
    }
    // ... potentially other highest-level NPs if found and valid ...
  ]
  ```

## Step 5: Final Result Assembly (`ANPEExtractor.extract`)

* **Purpose:** Combine the list of processed noun phrase dictionaries (`processed_results` from Step 4a or 4b) with overall metadata about the extraction process (timestamp, duration, configuration used) into the final dictionary that is returned by the `extract` method.
* **Input:** `processed_results`: The list of formatted NP dictionaries (either nested or flat) generated by the preceding steps.
* **Processing (at the end of the `extract` method):**
  1. Records the `end_time` and calculates the total `duration` of the extraction.
  2. Assembles a `config_used` dictionary to provide insights into the settings active for this specific extraction. This includes:
     * Basic configuration options like `min_length`, `max_length`, `accept_pronouns`, `structure_filters`, and `newline_breaks` (retrieved from `self`).
     * `spacy_model_used`: Determined by trying to get the name from the loaded `self.nlp.meta` attribute, falling back to `self.config.get('spacy_model')`, and then to "unknown".
     * `benepar_model_used`: Uses `self._loaded_benepar_model_name` (which stores the name of the Benepar model actually loaded), falling back to `self.config.get('benepar_model')`, and then to "unknown".
     * `metadata_requested`: The boolean value of the `metadata` parameter passed to `extract`.
     * `nested_requested`: The boolean value of the `include_nested` parameter passed to `extract`.
     * Any configuration items with a value of `None` are filtered out from the final `config_used` dictionary for cleaner output.
  3. Creates the final `result` dictionary containing:
     * `"timestamp"`: The `end_time` formatted as "YYYY-MM-DD HH:MM:SS".
     * `"processing_duration_seconds"`: The total `duration`, rounded to 3 decimal places.
     * `"configuration"`: The assembled `config_used` dictionary.
     * `"results"`: The `processed_results` list containing all the extracted and formatted noun phrases.
* **Output:** The final `Dict` returned by the `extract` method.
  * *Example Structure (assuming `include_nested=True` and `metadata=True` for the `results` part)*:

  ```python
  {
      "timestamp": "2025-05-05 00:12:32", // Example
      "processing_duration_seconds": 0.08, // Example
      "configuration": {
          // Values reflect the actual configuration used for the call
          "min_length": null, // Example, would be absent if default and None
          "max_length": null, // Example, would be absent if default and None
          "accept_pronouns": True,
          "structure_filters": [], // Empty if not specified
          "newline_breaks": True,
          "spacy_model_used": "en_core_web_md", // Actual loaded model
          "benepar_model_used": "benepar_en3",   // Actual loaded model
          "metadata_requested": True,
          "nested_requested": True
      },
      "results": [
          // The list of processed NP dictionaries as detailed in Step 4a or 4b.
          // For our example sentence with include_nested=True, this would be:
          {
            "noun_phrase": "Mr.Harrison...work", // From tree leaves
            "id": "1",
            "level": 1,
            "metadata": {"length": 12, "structures": [...]}, // Length from Span tokens
            "children": [ /* ... nested children ... */ ]
          },
          {
            "noun_phrase": "thesystem'scoreconcepts", // From tree leaves
            "id": "2",
            "level": 1,
            "metadata": {"length": 5, "structures": [...]}, // Length from Span tokens
            "children": [ /* ... nested children ... */ ]
          }
          // ... other top-level NPs ...
      ]
  }
  ```