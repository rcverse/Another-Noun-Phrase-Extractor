# Summary of Span-Based Analysis Refactoring (May 2025)

This document summarizes the refactoring effort to switch ANPE's noun phrase analysis from re-parsing extracted text strings to analyzing `spaCy` `Span` objects directly within the context of the original document parse.

## 1. Goal

The primary goal was to improve the accuracy of structural analysis (POS tags, dependencies, structure labels) by leveraging the richer syntactic information available in the original, full-sentence parse performed by `spaCy` and `Benepar`. Re-parsing isolated NP strings often leads to less accurate tagging and dependency information, especially for complex or nested phrases.

## 2. Summary of Key Changes

### `anpe/utils/analyzer.py` (`ANPEAnalyzer`)

*   **Input Type Changed:** All core analysis methods (`analyze_single_np`, `_detect_...` methods) were modified to accept `spacy.tokens.Span` objects as input instead of plain text strings (`np_text`).
*   **Removed Re-Parsing:** The internal call `doc = self.nlp(np_text)` within `analyze_single_np` (and any similar implicit re-parsing) was **removed**. The analyzer now relies entirely on the linguistic features already present on the input `Span` and its constituent `Token` objects.
*   **Direct Token Access:** Detection methods (`_detect_determiner_np`, `_detect_adjectival_np`, etc.) now iterate directly over the tokens within the input `Span` (e.g., `for token in span:`). They access `token.pos_`, `token.dep_`, `token.tag_`, `token.head`, etc., directly from the tokens, utilizing the original parse information.
*   **Span-Aware Helpers:** Helper methods like `_is_complement_taking_noun_within_span` were updated to operate within the context of the provided `Span`.
*   **Imports Added:** Type hints for `Span`, `Doc`, and `Token` were added from `spacy.tokens`.
*   **Logic Simplification (`_detect_appositive_np`):** The check for appositives was simplified by removing a constraint that required the appositive's head to be near the span root. This makes detection rely primarily on the `appos` dependency relation within the span.

### `anpe/extractor.py` (`ANPEExtractor`)

*   **Retained Original `Doc`:** The main `extract` method now parses the input text *once* using `self.nlp(text)` (which includes the Benepar component) and keeps the resulting `Doc` object available.
*   **Constituency Parsing:** Sentence constituents are accessed via `sent._.parse_string` from the `Doc`'s sentences and parsed into an `nltk.Tree`.
*   **Span Extraction (`_extract_..._spans` methods):**
    *   The NP extraction helper methods (`_extract_highest_level_np_spans`, `_extract_nps_with_hierarchy_spans`) were refactored.
    *   They now traverse the `nltk.Tree` derived from `sent._.parse_string`.
    *   Crucially, they map the `Tree` leaves back to the `Token` objects in the original sentence `Span` (`sent`).
    *   They use helper functions (`find_token_indices`, `get_span_from_token_indices`) to identify the start and end token indices corresponding to the NP identified in the tree.
    *   They return `spaCy` `Span` objects (e.g., `doc[start:end]`) or, for hierarchical extraction, dictionaries containing these `Span` objects (`{'span': Span, 'children': [...]}`).
*   **Analyzer Integration:** The `extract` method now passes the extracted NP `Span` objects directly to `self.analyzer.analyze_single_np` when metadata is requested.
*   **Processing/Filtering:** The `_process_np_span_with_ordered_fields` method handles the recursive processing of the span hierarchy, calls the analyzer for metadata/structure filtering, and builds the final output dictionary. Filtering logic now operates based on the analysis of the `Span` object.
*   **Validation Logic (`_is_valid_np`):** Updated to correctly handle structure filtering. When called during initial extraction (before analysis), it only applies length/pronoun filters. The structure filter is only applied later by `_process_np_span_with_ordered_fields` after the analysis has been performed.
*   **Imports Added:** `Doc` and `Span` from `spacy.tokens`, and `ANPEAnalyzer`.
*   **Pipeline Initialization:**
    *   Refined the logic to reliably handle sentence boundary detection configuration (`newline_breaks`).
    *   It now explicitly adds the `sentencizer` component if not present (checking standard names like `sentencizer`, `senter` first, then adding `sentencizer`). This component is needed for the `newline_breaks` configuration.
    *   The `newline_breaks` setting is applied to the identified/added sentence boundary component *before* Benepar is added to the pipeline.
*   **Newline Pre-processing:**
    *   Added a pre-processing step within the `extract` method.
    *   If `newline_breaks` is set to `False`, single newline characters (`\\n`) in the input `text` are replaced with spaces *before* the text is passed to `self.nlp()`. This helps ensure constituency parsers like Benepar correctly handle phrases that span lines when those line breaks are not intended as sentence boundaries. Double newlines are preserved to potentially represent paragraph breaks.

### `anpe/utils/export.py` (`ANPEExporter`)

*   **TXT Header Fix:** Corrected the `_export_txt` method to accurately report the `metadata` and `include_nested` flags used for the specific export call, rather than potentially outdated values from the configuration dictionary.

### `anpe/cli.py`

*   **Setup Command Fix:** Corrected the `setup` command to properly pass the `cli_log_callback` function to the `setup_models` utility, fixing an issue with logging during model installation via the CLI.

## 3. Current Processing Flow (`ANPEExtractor.extract`)

Here's a breakdown of the steps involved when `ANPEExtractor.extract(text, metadata=..., include_nested=...)` is called:

1.  **Initialization:** The `ANPEExtractor` is initialized with configuration (models, filters, etc.). It loads the specified `spaCy` model, ensures a sentence boundary component (`sentencizer` or `senter`) is present (adding `sentencizer` if needed), configures it based on `newline_breaks`, and then adds the specified `Benepar` model to the pipeline. The `ANPEAnalyzer` is also initialized.
2.  **Newline Pre-processing (Conditional):** If `self.newline_breaks` is `False`, the input `text` is pre-processed to replace single newlines with spaces. This yields `processed_text`. Otherwise, `processed_text = text`.
3.  **Initial Parse:** The `processed_text` is parsed *once* by the full `spaCy` pipeline (`self.nlp(processed_text)`), which includes tokenization, tagging, dependency parsing, and Benepar constituency parsing. This produces a `Doc` object containing all linguistic features.
4.  **Sentence Iteration:** The code iterates through each sentence (`sent`) in the `doc.sents`.
5.  **Constituency Tree:** For each `sent`, the Benepar constituency parse string (`sent._.parse_string`) is retrieved and parsed into an `nltk.Tree` object (`constituents_tree`). If parsing fails, the sentence is skipped.
6.  **NP Span Identification (Conditional):**
    *   **If `include_nested=True`:**
        *   `_extract_nps_with_hierarchy_spans(constituents_tree, sent)` is called.
        *   This function traverses the `constituents_tree`, identifies all NP nodes, maps their leaves to token indices in `sent`, and creates `Span` objects.
        *   It recursively builds a list of dictionaries representing the top-level NP hierarchies, e.g., `[{'span': Span1, 'children': [{'span': Span1.1, ...}]}, {'span': Span2, ...}]`.
        *   Basic length/pronoun validation (`_is_valid_np`) is performed during this extraction to prune invalid branches early.
    *   **If `include_nested=False`:**
        *   `_extract_highest_level_np_spans(constituents_tree, sent)` is called.
        *   This function traverses the `constituents_tree`, identifies only the highest-level NP nodes (NPs not contained within another NP), maps their leaves to token indices, and creates `Span` objects.
        *   It returns a flat list of `Span` objects: `[Span1, Span2, ...]`.
7.  **Processing and Filtering (Conditional):**
    *   **If `include_nested=True`:**
        *   The code iterates through the list of top-level NP hierarchy dictionaries from step 6a.
        *   For each dictionary, `_process_np_span_with_ordered_fields` is called recursively.
        *   Inside `_process_...`:
            *   If `metadata=True`, it calls `self.analyzer.analyze_single_np(span)` to get the structure list.
            *   It applies structure filters (`self.structure_filters`) based on the analysis results. If the current span fails the filter, an empty dictionary `{}` is returned, effectively filtering it (and its children).
            *   It builds the final output dictionary for the current span (including `noun_phrase`, `id`, `level`, optional `metadata`).
            *   It recursively calls itself for children, appending only non-empty results.
        *   Only non-empty dictionaries returned by `_process_...` for the top-level spans are added to the final `processed_noun_phrases` list.
    *   **If `include_nested=False`:**
        *   The code iterates through the flat list of `Span` objects from step 6b.
        *   For each `span`:
            *   If `metadata=True`, it calls `self.analyzer.analyze_single_np(span)` to get structures.
            *   It calls `_is_valid_np(span, structures)` to perform *all* validation (length, pronoun, *and* structure filters if applicable, since analysis results are now available).
            *   If valid, it builds the output dictionary (including `noun_phrase`, `id`, `level`, optional `metadata`, empty `children`).
            *   The valid dictionary is added to `processed_noun_phrases`.
8.  **Result Formatting:** The final list `processed_noun_phrases` is packaged into the result dictionary along with timestamp, duration, and the configuration used for the extraction.
9.  **Return:** The result dictionary is returned.

*(See `structure_patterns.md` for details on specific structure detection logic)*

## 4. Incidental Bug Fixes During Refactoring

Beyond the primary goal of switching to `Span`-based analysis, the refactoring process uncovered and addressed several pre-existing bugs or functional issues in the ANPE codebase:

### 4.1. SpaCy Pipeline Initialization Error

*   **What was wrong:** The original code for adding the `sentencizer` component to the spaCy pipeline could fail with a `ValueError` under certain conditions (e.g., if a `parser` component was already present, creating conflicting arguments like `before="parser"` and `first=True`, or if the component was unexpectedly already added).
*   **What needed fixing:** Ensure the `sentencizer` component is reliably added to the pipeline regardless of the base spaCy model's default components, and handle cases where it might already exist gracefully.
*   **Possible outcome:** A `ValueError` crash during `ANPEExtractor` initialization, preventing the extractor from being used. The specific error message might mention conflicting pipeline component constraints.
*   **Fix:** The logic in `ANPEExtractor.__init__` was updated to check if `sentencizer` already exists. If not, it adds it using `first=True` if no parser is present, or `before="parser"` if a parser exists, avoiding the conflicting constraints.

### 4.2. Incorrect Token Mapping in Span Creation

*   **What was wrong:** The `find_token_indices` helper function within `ANPEExtractor`, responsible for mapping NLTK Tree leaves back to spaCy `Token` indices in the sentence `Span`, was incorrectly attempting to compare leaf strings directly with `Token` objects (`leaf == token`) instead of comparing the leaf string with the token's text (`leaf == token.text`).
*   **What needed fixing:** Correctly compare the text content of the tree leaves with the text content of the spaCy tokens to find the corresponding token indices for creating NP `Span` objects.
*   **Possible outcome:** An `AttributeError` during NP extraction when the incorrect comparison was attempted, or failure to find correct spans leading to missing NPs in the output.
*   **Fix:** The comparison logic in `find_token_indices` was changed from `leaf == token` to `leaf_texts[j] != sent_tokens[i+j].text` (checking for mismatch).

### 4.3. Flawed Filtering Logic with Nested Structures

*   **What was wrong:**
    1.  When `include_nested=True`, if a top-level NP contained *only* children that were filtered out (e.g., by length or structure filters), the parent NP itself might still have been included in the final results, even though it effectively represented an empty branch after filtering.
    2.  The `_is_valid_np` function applied `structure_filters` prematurely during the initial hierarchical extraction phase (`_extract_nps_with_hierarchy_spans`), even when analysis results (`structures`) were not yet available. This could lead to valid NPs being incorrectly filtered out early.
*   **What needed fixing:**
    1.  Ensure that if all children of an NP are filtered out, the parent NP is also removed from the results if it doesn't independently meet criteria (or if its inclusion implies its children).
    2.  Apply structure-based filtering only *after* the structural analysis has been performed.
*   **Possible outcome:**
    1.  Output containing "parent" NPs with empty `children` lists that should logically have been removed entirely.
    2.  Valid nested NPs being missing from the results because they were filtered based on structure before the structure was determined.
*   **Fix:**
    1.  The `extract` method's loop processing hierarchical results now checks if the recursive `_process_np_span_with_ordered_fields` call returns an empty dictionary (signifying the branch was filtered) and skips appending it.
    2.  `_is_valid_np` was modified to only check `structure_filters` if the `structures` list argument is actually populated (i.e., analysis has occurred). The primary structure filtering now happens definitively within `_process_np_span_with_ordered_fields` after calling the analyzer.

### 4.4. Missing `Token` Import in Analyzer

*   **What was wrong:** After refactoring the `ANPEAnalyzer` to work with `Span` and `Token` objects, a required import `from spacy.tokens import Token` was missing.
*   **What needed fixing:** Add the necessary import statement.
*   **Possible outcome:** A `NameError: name 'Token' is not defined` when the analyzer code referencing the `Token` type was executed.
*   **Fix:** Added `from spacy.tokens import Doc, Span, Token` to `anpe/utils/analyzer.py`.

### 4.5. Suboptimal Appositive Detection

*   **What was wrong:** The original logic in `_detect_appositive_np` had relatively strict conditions (involving the dependency root of the span) that sometimes failed to identify valid appositive structures, like `"the meeting, the one held yesterday"`.
*   **What needed fixing:** Make the appositive detection less strict to improve recall, focusing primarily on the presence of the `appos` dependency relation within the span.
*   **Possible outcome:** Failure to assign the `"appositive"` structure label to certain valid appositive noun phrases during analysis.
*   **Fix:** The check within `_detect_appositive_np` was simplified by removing a condition related to the span's root token, making the detection more reliant on finding any token within the span with the `dep_ == 'appos'` attribute whose head is also within the span. *(Note: Further refinement might still be needed)*. 