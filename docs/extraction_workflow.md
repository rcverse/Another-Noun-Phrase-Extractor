# ANPE Extractor Workflow (Current Version)

This document explains the step-by-step workflow of the noun phrase (NP) extraction process implemented in the `anpe/extractor.py` module of the `another-noun-phrase-extractor` project. We will follow a single example sentence through the entire process:

**Example Sentence:** `"Mr. Harrison, the lead developer known for his meticulous work, presented the system's core concepts."`

We assume the extractor is called with `metadata=True` and `include_nested=True` for this walkthrough to cover the most comprehensive path.

## Overview

The core goal is to identify noun phrases within a given text, optionally including nested phrases and metadata (length, structural analysis). The process leverages spaCy for initial text processing and tokenization, Benepar for constituency parsing (identifying grammatical structures like NP), NLTK for manipulating parse trees, and custom logic to integrate these components, validate results, and handle potential inconsistencies.

## Step 0: Initialization (`ANPEExtractor.__init__`)

*   **Purpose:** Set up the extractor instance, load configuration, initialize logging, and load the required NLP models (spaCy and Benepar).
*   **Input:** An optional `config` dictionary overriding default settings (e.g., `log_level`, `min_length`, `accept_pronouns`, `spacy_model`, `benepar_model`).
    *   *Example*: `{'log_level': 'DEBUG', 'accept_pronouns': True}`
*   **Processing:**
    1.  Merges user config with defaults.
    2.  Gets the logger using `logging.getLogger(__name__)`. (Logging is configured externally, e.g., by the CLI or calling script).
    3.  Determines which spaCy and Benepar models to use (based on config or auto-detection using `anpe.utils.model_finder`). Handles model aliases (`SPACY_MODEL_MAP`, `BENEPAR_MODEL_MAP`).
        *   *Example Log (auto-detection)*:
            ```
            INFO - Auto-detected best available spaCy model: en_core_web_md
            INFO - Auto-detected best available Benepar model: benepar_en3
            ```
    4.  Loads the selected spaCy model (`spacy.load(spacy_model_to_use)`). This creates the core `self.nlp` object.
    5.  Configures the spaCy pipeline:
        *   Adds/configures the `sentencizer` component based on the `newline_breaks` setting. This component determines how sentence boundaries are detected.
        *   Adds the Benepar component (`self.nlp.add_pipe("benepar", config={"model": benepar_model_to_load})`). This integrates Benepar's parser into the spaCy pipeline. It handles potential model loading errors and fallbacks.
        *   *Example Log*:
            ```
            INFO - Configuring 'sentencizer' to treat newlines as sentence boundaries
            INFO - Attempting to add Benepar component with model: 'benepar_en3'
            INFO - Benepar component ('benepar_en3') added successfully.
            ```
    6.  Initializes the structural analyzer (`self.analyzer = ANPEAnalyzer(self.nlp)` from `anpe.utils.analyzer`), which is used later for metadata generation.
*   **Output:** A configured `ANPEExtractor` instance with:
    *   `self.config`: The final configuration dictionary.
        *   *Example (debug run)*: `{'accept_pronouns': True, 'structure_filters': [], 'newline_breaks': True, ...}`
    *   `self.logger`: A logger instance (`logging.getLogger(__name__)`).
    *   `self.nlp`: A loaded spaCy `Language` object containing the full pipeline (tokenizer, tagger, parser, sentencizer, benepar, etc.).
    *   `self.analyzer`: An `ANPEAnalyzer` instance ready for use.
    *   Other config attributes (`min_length`, `accept_pronouns`, etc.).

## Step 1: Text Preprocessing & Parsing (`ANPEExtractor.extract`)

*   **Purpose:** Prepare the input text and parse it using the full spaCy+Benepar pipeline.
*   **Input:**
    *   `text`: The raw input text string.
        *   *Example*: `"Mr. Harrison, the lead developer known for his meticulous work, presented the system's core concepts."`
    *   `metadata`: Boolean flag. (*Example*: `True`)
    *   `include_nested`: Boolean flag. (*Example*: `True`)
*   **Processing:**
    1.  Handles early exit for empty input text.
    2.  (Optional) Pre-processes the text based on `newline_breaks=False`. (Not applicable in our example).
    3.  Parses the *entire* processed text using the loaded pipeline: `doc = self.nlp(processed_text)`.
        *   *Example Log*:
            ```
            DEBUG - Parsing text with spaCy+Benepar...
            DEBUG - Text parsed successfully.
            ```
*   **Output:**
    *   `doc`: A spaCy `Doc` object.
        *   *Example Doc (properties)*: Contains tokens like `Mr.`, `Harrison`, `,`, `the`, `lead`, ..., `concepts`, `.`. Each token has `.text`, `.pos_`, `.tag_`, `.dep_`, etc. The `doc` object itself has `.sents` property.
        *   *Example Sentence Property*: The first (and only) sentence `Span` (`sent = list(doc.sents)[0]`) has `sent.text` = `"Mr. Harrison, the lead developer known for his meticulous work, presented the system's core concepts."`. Crucially, it also has `sent._.parse_string` = `"(S (NP (NNP Mr.) (NNP Harrison)) (, ,) (NP (NP (DT the) (JJ lead) (NN developer)) (VP (VBN known) (PP (IN for) (NP (PRP$ his) (JJ meticulous) (NN work))))) (, ,) (VP (VBD presented) (NP (NP (DT the) (NN system) (POS 's)) (JJ core) (NNS concepts))) (. .))"` (This is the actual parse string from the debug run, may vary slightly with models/versions).

## Step 2: Sentence Iteration & Tree Creation (`ANPEExtractor.extract` loop)

*   **Purpose:** Process each sentence identified by spaCy to extract noun phrases based on its Benepar parse tree.
*   **Input:** The spaCy `Doc` object from Step 1.
    *   *Example*: The `doc` object containing our single sentence.
*   **Processing (Looping through `doc.sents`):**
    1.  For each `sent` (a spaCy `Span`):
        *   *Example*: Our single sentence `Span`.
        *   *Example Log*: `DEBUG - [extract loop] Processing sentence 0: 'Mr. Harrison, the lead developer known for his met...'`
    2.  Retrieve the parse string: `parse_string = sent._.parse_string`.
        *   *Example*: `"(S (NP...) ...)"` as shown above.
    3.  Check if the parse string is valid.
    4.  Convert the parse string into an NLTK `Tree` object: `constituents_tree = Tree.fromstring(parse_string)`.
        *   *Example Log*: `DEBUG - [extract loop] Successfully created constituents_tree. Type: <class 'nltk.tree.tree.Tree'>`
*   **Output (Per Sentence):**
    *   `sent`: The current spaCy `Span` object for the sentence.
    *   `constituents_tree`: An NLTK `Tree` object.
        *   *Example Tree (from debug run, structure only)*:
        ```
        (S
          (NP (NNP Mr.) (NNP Harrison))
          (, ,)
          (NP
            (NP (DT the) (JJ lead) (NN developer))
            (VP (VBN known) (PP (IN for) (NP (PRP$ his) (JJ meticulous) (NN work)))))
          (, ,)
          (VP (VBD presented)
            (NP
              (NP (DT the) (NN system) (POS 's))
              (JJ core) (NNS concepts)))
          (. .))
        ```

## Step 3: NP Extraction (Conditional Logic)

The workflow now diverges based on the `include_nested` flag. Our example follows `include_nested=True`.

### Step 3a: Nested Extraction (`include_nested=True`)

#### Step 3a.1: Collect All NP Node Info (`_collect_np_nodes_info`)

*   **Purpose:** Traverse the sentence's NLTK parse tree, identify *every* node labeled "NP", extract its text, and attempt to map it precisely to a spaCy `Span`.
*   **What's an NLTK Tree Node?** When Benepar parses the sentence (Step 1), it creates a tree structure (like a family tree) representing the sentence's grammar. Think of it like diagramming a sentence. Each point in this tree is a 'node'. Nodes have labels (like 'S' for Sentence, 'VP' for Verb Phrase, 'NP' for Noun Phrase). The bottom-most nodes are the actual words (called 'leaves'). An 'NP' node in this tree represents a chunk of the sentence that Benepar identified as having the grammatical structure of a noun phrase.
*   **Input:**
    *   `constituents_tree`: The NLTK `Tree` from Step 2.
    *   `sent`: The spaCy `Span` from Step 2.
*   **Processing:** Recursively traverses the `constituents_tree`. For each node labeled "NP":
    1.  Extracts the node's plain text using `_tree_to_text(node)` (which essentially joins leaves).
    2.  Identifies the node's leaves (`node.leaves()`).
    3.  Attempts to find the corresponding token indices in the sentence's tokens using `_find_token_indices_for_leaves(sent_tokens, leaves)`.
        *   **Crucial Detail:** Within `_find_token_indices_for_leaves`, before a direct string comparison is made between a leaf from the Benepar tree and a spaCy token's text, both strings are first processed by the `_normalize_text_for_matching` helper method. This method standardizes various representations, such as converting Penn Treebank symbols (e.g., "-LRB-", "``", "''") to their common textual equivalents (e.g., "(", """), and strips leading/trailing whitespace. This normalization step is critical for accurately aligning the potentially different tokenizations and conventions of Benepar (PTB-style) and spaCy.
    4.  If token indices are found (i.e., the normalized leaf sequence matches a normalized token sequence), creates a spaCy `Span` using `_get_span_from_token_indices(sent, start_idx, end_idx)`. This mapping links Benepar's grammatical structure (the Tree node) to spaCy's token-based representation (the Span). This mapping can fail if Benepar's and spaCy's tokenization differ significantly for that phrase even after normalization (logs a warning).
    5.  Stores information about this NP node in a dictionary (`NPNodeInfo`), including:
        *   `'node'`: A reference to the *actual NLTK Tree object* representing this specific NP node in Benepar's parse tree. It's like keeping a direct pointer to that part of the grammatical structure tree.
        *   `'text'`: The plain text extracted from the node's leaves.
        *   `'span'`: The corresponding spaCy `Span` object if the mapping was successful, or `None` otherwise.
*   **Output:**
    *   `np_nodes_info`: A `List[NPNodeInfo]`. Contains info for *all* NP nodes found.
        *   *Example Log*: `DEBUG - _collect_np_nodes_info found 7 NP nodes.`
        *   *Example `np_nodes_info` (showing structure for a few items based on debug run)*:
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

#### Step 3a.2: Build NP Info Map (`_build_np_info_map`)

*   **Purpose:** Convert the list of `NPNodeInfo` dictionaries into a map (a dictionary) for efficient lookup.
*   **How the map is built:** This function iterates through the `np_nodes_info` list from Step 3a.1. For each `info` dictionary in the list, it takes the unique identifier (memory address) of the NLTK Tree object (`id(info['node'])`) and uses that ID as a key in the new map. The value associated with that key is the *entire original `info` dictionary* itself (`{'node': ..., 'text': ..., 'span': ...}`).
*   **Input:**
    *   `np_nodes_info`: The `List[NPNodeInfo]` from Step 3a.1.
*   **Output:**
    *   `np_info_map`: A `Dict[int, NPNodeInfo]`. Maps NLTK node IDs to their collected info.
        *   *Example Log*: `DEBUG - Built NP info map with 7 entries.`

#### Step 3a.3: Build Hierarchy from Tree (`_build_hierarchy_from_tree`)

*   **Purpose:** Reconstruct the hierarchical (nested) structure of the valid noun phrases based on their positions in the original NLTK parse tree.
*   **How the hierarchy is built & info preserved:** This function walks through the *original, full NLTK tree* (`constituents_tree`) generated in Step 2. It does not just look at the collected NP nodes directly, but navigates the sentence's grammatical structure.
    *   **When it encounters an NP node in the tree:**
        1.  It gets the unique ID of that NLTK NP node object.
        2.  It looks up this ID in the `np_info_map` (from Step 3a.2). This crucial step ensures that only NP nodes that were successfully found and processed (mapped to a Span, or mapping attempted) in Step 3a.1 are included in the final hierarchy.
        3.  If the ID is found, it retrieves the complete `NPNodeInfo` dictionary associated with that node. This dictionary (containing the original NLTK node reference, the text, and the mapped spaCy Span) is **preserved** by storing it under the `'info'` key within the new hierarchy node being created: `{'info': <the_retrieved_NPNodeInfo_dict>, 'children': []}`.
        4.  It then **recursively calls itself**, but importantly, it only explores the *direct children of the current NP node* within the NLTK tree. This process finds NPs that are nested immediately inside the current one.
        5.  The results of these recursive calls (which are hierarchy nodes themselves) are gathered into a list and assigned to the `'children'` key of the current node's hierarchy entry.
    *   **When it encounters a non-NP node (e.g., 'VP', 'PP'):**
        1.  It does *not* create a hierarchy entry for this node.
        2.  However, it *still recursively explores the children* of this non-NP node. This is vital because an NP might be grammatically nested inside another phrase type (e.g., `[PP in [NP the hat]]`). This ensures all potential branches of the tree are checked.
*   **Input:**
    *   `constituents_tree`: The NLTK `Tree` for the sentence.
    *   `np_info_map`: The map from Step 3a.2.
*   **Processing:** This function builds the nested structure. It works by walking through the *original full sentence NLTK tree* (`constituents_tree`) created in Step 2:
    1.  When it encounters a node in the full tree:
        *   **If the node is labeled "NP":**
            *   It checks if we successfully collected info for this *specific* NP node earlier (i.e., if its unique identifier `id(node)` exists as a key in the `np_info_map`). This ensures we only consider NPs that could be reliably mapped to spaCy Spans in Step 3a.1.
            *   If the info is found in the map, it creates the basic hierarchy dictionary: `{'info': <the collected info>, 'children': []}`.
            *   It then **only looks inside this specific NP node** for more NPs by recursively calling `_build_hierarchy_from_tree` on the children *of this current NP node* within the NLTK tree. This finds NPs nested directly inside the current one.
            *   The results of these recursive calls become the `'children'` list for the current NP node's hierarchy entry.
        *   **If the node is NOT labeled "NP"** (e.g., it's a 'VP' - Verb Phrase or 'PP' - Prepositional Phrase):
            *   It *doesn't* create a hierarchy entry for this node itself.
            *   However, it still looks inside this node's children recursively, because an NP might be nested further down (e.g., an NP inside a PP: "the cat [PP in [NP the hat]]").
*   **Output:**
    *   `sentence_hierarchy`: A `List[NPNodeHierarchy]`. This list contains only the hierarchies starting from the *highest-level* NPs found in the sentence tree (those not nested inside another NP *in that specific parse*) for which valid info was found in the `np_info_map`.
        *   *Example Log*: `DEBUG - [extract loop] Built hierarchy with 2 top-level nodes.`
        *   *Example `sentence_hierarchy` (simplified structure based on debug run):*
        ```python
        [
          # Hierarchy for the first top-level NP (Mr. Harrison...)
          # The exact NLTK node might be slightly different than the example span text
          # due to how Benepar structures the main clause vs. appositives/clauses.
          # The key is that its associated 'span' covers the correct tokens.
          {'info': {'node': <Tree for Mr. Harrison...>, # Outer NP node
                    'text': "Mr.Harrisontheleaddeveloperknownforhismeticulouswork", # Joined leaves
                    'span': <Span "Mr. Harrison , the lead developer known for his meticulous work ,">},
           'children': [
              # Child hierarchy for "Mr. Harrison"
              {'info': {'node': <Tree for Mr. Harrison>,
                        'text': "Mr.Harrison",
                        'span': <Span "Mr. Harrison">},
               'children': []},
              # Child hierarchy for "the lead developer known for his meticulous work"
              {'info': {'node': <Tree for the lead dev...>,
                        'text': "theleaddeveloperknownforhismeticulouswork",
                        'span': <Span "the lead developer known for his meticulous work">},
               'children': [
                   # Grandchild for "the lead developer"
                   {'info': {'node': <Tree for the lead dev>,
                             'text': "theleaddeveloper",
                             'span': <Span "the lead developer">},
                    'children': []},
                   # Grandchild for "his meticulous work"
                   {'info': {'node': <Tree for his work>,
                             'text': "hismeticulouswork",
                             'span': <Span "his meticulous work">},
                    'children': []}
               ]}
           ]},

          # Hierarchy for the second top-level NP ("the system 's core concepts")
          {'info': {'node': <Tree for the system...>,
                    'text': "thesystem'scoreconcepts",
                    'span': <Span "the system 's core concepts">},
           'children': [
               # Child hierarchy for "the system 's"
               {'info': {'node': <Tree for the system 's>,
                         'text': "thesystem's",
                         'span': <Span "the system 's">},
                'children': []}
           ]}
        ]
        ```
        *(Note: Structure reflects the nested NPs found in the tree and successfully mapped in `_collect_np_nodes_info`)*

#### Rationale for Two Tree Traversals (Steps 3a.1 & 3a.3)

You might notice that both Step 3a.1 (`_collect_np_nodes_info`) and Step 3a.3 (`_build_hierarchy_from_tree`) involve traversing the NLTK constituency tree. While this might seem redundant, this two-pass approach offers several advantages:

1.  **Separation of Concerns:** It cleanly separates the task of *finding and validating* potential NP nodes (including the complex mapping to spaCy Spans in 3a.1) from the task of *constructing the final hierarchy* based on the original tree structure (3a.3).
2.  **Handling Mapping Failures Gracefully:** The spaCy Span mapping in 3a.1 can sometimes fail. By performing this mapping first and storing the results (or `None`) in `np_info_map`, the hierarchy building step (3a.3) can easily check this map and only include nodes that were successfully processed, without needing complex error handling logic within its own recursive structure.
3.  **Code Clarity:** Although potentially requiring slightly more processing time, this separation often leads to code that is easier to understand, debug, and maintain, as each function has a clearly defined responsibility.

While a single-pass approach is conceivable, the current method prioritizes logical clarity and robust handling of potential mapping issues over minimizing the number of tree traversals, which is generally not the main performance bottleneck.

#### Step 3a.4: Process NP Node Hierarchies (`_process_np_node_info`)

*   **Purpose:** Convert `NPNodeHierarchy` items into the final output dictionary format, performing validation and analysis primarily on the associated spaCy `Span`. Filters out nodes where the `span` is `None` (mapping failed) or fails validation (`_is_valid_np`).
*   **Input (per top-level hierarchy node):** `hierarchy_node` from `sentence_hierarchy` (Step 3a.3), `base_id` (e.g., "1"), `include_metadata`, `current_level`.
    *   *Example (first call)*: The first dict in `sentence_hierarchy` list above, `base_id="1"`, `include_metadata=True`, `current_level=1`.
*   **Processing (Recursive):**
    1.  Retrieves `np_info` from the current `hierarchy_node`.
    2.  **Validation & Pruning:**
        *   Checks if `np_info['span']` exists. If not, prunes this node and its children (returns `{}`).
        *   Calls `self._is_valid_np(np_info['span'])` which checks:
            *   Length against `min_length`/`max_length`.
            *   Whether it's a pronoun and `accept_pronouns` is `False`.
            *   If invalid, prunes this node and its children (returns `{}`).
    3.  **Data Extraction:**
        *   Uses `np_span.text` as the primary `noun_phrase` string (more reliable than joined leaves).
        *   Calculates `length` from `len(np_span)`.
    4.  **Metadata (if `include_metadata`):**
        *   Calls `self.analyzer.analyze_single_np(np_span)` to get structural types.
            *   *Example Log (for span "Mr. Harrison...")*: `DEBUG - Analysis complete for 'Mr. Harrison, ...': Structures=['determiner', 'adjectival_modifier', 'compound', 'possessive', 'appositive', 'reduced_relative_clause']`
            *   *Example Log (for span "Mr. Harrison")*: `DEBUG - Analysis complete for 'Mr. Harrison': Structures=['compound']`
            *   Applies `structure_filters` if provided in config.
    5.  **Recursion:** Calls `_process_np_node_info` for each `child` in `hierarchy_node['children']`, incrementing `level` and creating nested IDs (e.g., "1.1", "1.2"). Filters out empty `{}` results from pruned children.
    6.  **Assembly:** Constructs the result dictionary for the current node.
*   **Output (Aggregated across all top-level calls):** `processed_results`: A list of fully formatted dictionaries for the valid, top-level NPs and their valid children.
    *   *Example Output (Matching the final debug script output structure)*:
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

### Step 3b: Non-Nested Extraction (`include_nested=False`)

*(This path is not taken in our main example, but shown for completeness)*

**Why two filtering steps?** This path involves two filtering stages because identifying the "highest-level" noun phrase can be interpreted structurally (based on the parse tree) or textually (based on which phrase contains others). The code uses both:

#### Step 3b.1: Extract Highest-Level Spans (`_extract_highest_level_np_spans`)

*   **Purpose:** Identify NPs that are structurally highest-level based on the parse tree.
*   **Input:** `constituents_tree` (the NLTK tree), `sent` (the spaCy sentence span).
*   **Processing:** This function looks at the *NLTK tree structure*. It finds NP nodes whose *immediate parent node in the tree* is *not* also an NP node. This is a purely structural definition of "highest level" based on the grammar tree produced by Benepar. It then attempts to map these structurally highest NP nodes to spaCy Spans.
*   **Output:** `candidate_np_spans`: A `List[Span]`.
    *   *Example*: Likely `[<Span for "Mr. Harrison...">, <Span for "the system's core concepts">]` (depending on exact parse tree structure).

#### Step 3b.2: Filter Contained Spans (`extract` method)

*   **Purpose:** Ensure only the outermost *textual* spans are kept.
*   **Input:** `candidate_np_spans` (the list of Spans from Step 3b.1).
*   **Processing:** Step 3b.1 might still produce multiple spaCy `Span` objects where one *textually contains* another, even if they came from different branches of the tree deemed "highest level" structurally. For example, a complex phrase like "the big dog chasing the cat" might be parsed such that "the big dog" and "the cat" are both considered structurally highest-level by Step 3b.1 (if neither had an NP parent node in that specific parse). However, for `include_nested=False`, we typically only want the longest, outermost phrase textually (in this case, arguably the whole phrase, although Benepar might not identify that as a single NP). This second filtering step iterates through the *candidate Spans* produced by 3b.1 and uses their *start and end character positions* (`span.start`, `span.end`) to explicitly remove any span that falls completely inside another span *within that candidate list*. This ensures only the truly outermost textual noun phrases (from the candidate list) are kept for the `include_nested=False` output.
*   **Output:** `highest_level_only_spans`: A `List[Span]`.
    *   *Example*: Same as input if no spans contain others: `[<Span for "Mr. Harrison...">, <Span for "the system's core concepts">]`

## Step 4: Post-Processing & Validation

Our example follows the nested path (Step 4a).

### Step 4a: Processing Hierarchies (`_process_np_node_info`) - (If `include_nested=True`)

*   **Purpose:** Convert `NPNodeHierarchy` items into the final output dictionary format, performing validation and analysis primarily on the associated spaCy `Span`. Filters out nodes where the `span` is `None` (mapping failed) or fails validation (`_is_valid_np`).
*   **Input (per top-level hierarchy node):** `hierarchy_node` from `sentence_hierarchy` (Step 3a.3), `base_id` (e.g., "1"), `include_metadata`, `current_level`.
    *   *Example (first call)*: The first dict in `sentence_hierarchy` list above, `base_id="1"`, `include_metadata=True`, `current_level=1`.
*   **Processing (Recursive):**
    1.  Retrieves `np_info` from the current `hierarchy_node`.
    2.  **Validation & Pruning:**
        *   Checks if `np_info['span']` exists. If not, prunes this node and its children (returns `{}`).
        *   Calls `self._is_valid_np(np_info['span'])` which checks:
            *   Length against `min_length`/`max_length`.
            *   Whether it's a pronoun and `accept_pronouns` is `False`.
            *   If invalid, prunes this node and its children (returns `{}`).
    3.  **Data Extraction:**
        *   Uses `np_span.text` as the primary `noun_phrase` string (more reliable than joined leaves).
        *   Calculates `length` from `len(np_span)`.
    4.  **Metadata (if `include_metadata`):**
        *   Calls `self.analyzer.analyze_single_np(np_span)` to get structural types.
            *   *Example Log (for span "Mr. Harrison...")*: `DEBUG - Analysis complete for 'Mr. Harrison, ...': Structures=['determiner', 'adjectival_modifier', 'compound', 'possessive', 'appositive', 'reduced_relative_clause']`
            *   *Example Log (for span "Mr. Harrison")*: `DEBUG - Analysis complete for 'Mr. Harrison': Structures=['compound']`
            *   Applies `structure_filters` if provided in config.
    5.  **Recursion:** Calls `_process_np_node_info` for each `child` in `hierarchy_node['children']`, incrementing `level` and creating nested IDs (e.g., "1.1", "1.2"). Filters out empty `{}` results from pruned children.
    6.  **Assembly:** Constructs the result dictionary for the current node.
*   **Output (Aggregated across all top-level calls):** `processed_results`: A list of fully formatted dictionaries for the valid, top-level NPs and their valid children.
    *   *Example Output (Matching the final debug script output structure)*:
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

### Step 4b: Formatting Highest-Level Spans (If `include_nested=False`)

*(Not used in main example)*
*   **Input:** `highest_level_only_spans` (List of Spans).
*   **Output:** A list (`processed_results`) of flat dictionaries.
    *   *Example*:
    ```python
    [
      {'noun_phrase': "Mr. Harrison...", 'id': "1", 'level': 1, 'metadata': {...}, 'children': []},
      {'noun_phrase': "the system's core concepts", 'id': "2", 'level': 1, 'metadata': {...}, 'children': []}
    ]
    ```

## Step 5: Final Result Assembly (`ANPEExtractor.extract`)

*   **Purpose:** Combine processed results with metadata.
*   **Input:** `processed_results` (The list of formatted NP dictionaries from Step 4a).
    *   *Example*: `[ {NP dict for ID "1"}, {NP dict for ID "2"}, {NP dict for ID "3"}, ... ]` containing the structured data like the example in Step 4a.
*   **Processing:** Calculates duration, assembles `config_used` dict, creates the final result dict.
*   **Output:** The final `Dict` returned by the `extract` method:
    *   *Example Structure*:
    ```python
    {
        "timestamp": "2025-05-05 00:12:32", # From debug run
        "processing_duration_seconds": 0.08, # From debug run
        "configuration": {
            "accept_pronouns": True,
            "structure_filters": [],
            "newline_breaks": True,
            "spacy_model_used": "en_core_web_md", # Resolved model (retrieved dynamically)
            "benepar_model_used": "benepar_en3", # Resolved model (retrieved dynamically)
            "metadata_requested": True,
            "nested_requested": True
        },
        "results": [
            # The list of processed NP dictionaries generated in Step 4a,
            # e.g., the dict for "Mr. Harrison", "the lead developer...", etc.
            {
              'noun_phrase': "Mr. Harrison", 'id': "1", 'level': 1, # Example ID assignment might differ
              'metadata': {'length': 2, 'structures': ['compound']}, 'children': []
            },
             {
              'noun_phrase': "the lead developer known for his meticulous work",
              'id': "2", 'level': 1,
              'metadata': {'length': 8, 'structures': [...]},
              'children': [
                 {'noun_phrase': "the lead developer", 'id': "2.1", ...},
                 {'noun_phrase': "his meticulous work", 'id': "2.2", ...}
              ]
            },
            {
              'noun_phrase': "the system 's core concepts",
              'id': "3", 'level': 1,
              'metadata': {'length': 5, 'structures': [...]},
              'children': [
                 {'noun_phrase': "the system 's", 'id': "3.1", ...}
              ]
            }
            # ... potentially more if parser identified others ...
        ]
    }
    ```

This comprehensive workflow balances the structural information from Benepar's parse trees with the token-level details and validation capabilities provided by spaCy Spans, handling potential mapping issues by filtering unreliable nodes. 