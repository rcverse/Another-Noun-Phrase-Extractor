# Changelog

All notable changes to the ANPE project will be documented in this file.

## [1.0.3] - 2025-05-12

### Improved

- **Further enhanced robustness for texts with complex punctuation (building on 1.0.2):**
    - Modified the `_preprocess_text_for_benepar` method to intelligently add spaces around parentheses `()`, square brackets `[]`, and curly braces `{}`.
    - This pre-processing step ensures these punctuation marks are more consistently tokenized as separate entities by both spaCy and Benepar's underlying tokenizer *before* they reach the `_normalize_text_for_matching` step.
    - The change preserves the existing sophisticated newline handling logic controlled by the `newline_breaks` setting.
- Added detailed `DEBUG` logging in `_preprocess_text_for_benepar` to show the text transformations, including the state before and after punctuation padding.

## [1.0.2] - 2025-05-09

### Improved

- ANPE is now more robust in handling various special characters (like parentheses and different types of quotation marks) within or around noun phrases
- Enhanced the robustness of matching Benepar parse tree leaves to spaCy tokens by introducing a text normalization step (`ANPEExtractor._normalize_text_for_matching`). This method standardizes various textual representations before comparison, including:
  - Penn Treebank symbols (e.g., `-LRB-`, `-RRB-`, `-LSB-`, `-RSB-`, `-LCB-`, `-RCB-`).
  - Different styles of quotation marks (e.g., PTB's ` `` ` and `''`, curly quotes `" " ' '`, and standard `" '`).
  - Stripping leading/trailing whitespace.
    This significantly improves the reliability of mapping parsed constituents to spaCy spans, especially with texts containing varied punctuation or PTB-style artifacts.
- Added detailed `DEBUG` level logging within the `_normalize_text_for_matching` method to trace the exact transformations being applied, aiding in diagnosing any future mapping discrepancies.
- Improved the clarity of the warning message logged by `_find_token_indices_for_leaves` when it fails to map a sequence of tree leaves to spaCy tokens, now providing more context about the leaves and tokens involved.

## [1.0.1] - 2025-05-07

### Fixed

- Improved the robustness of `spacy-transformers` installation verification in `anpe.utils.setup_models.install_spacy_model`.
  - The check now primarily ensures the `spacy-transformers` library is importable after `pip install`.
  - A secondary, non-blocking check for the presence of the `'transformer'` factory is performed for logging and issues a warning if not immediately found, rather than failing the installation.
  - The definitive test for model loadability (including factory availability) is now the final `spacy.load()` call after the model assets are downloaded.
  - This resolves an issue where transformer model setup could prematurely fail in certain environments (e.g., within the ANPE GUI) due to transient factory registration issues, even if the model would be loadable after a restart or with a slightly different check timing.

## [1.0.0] - 2025-05-06

### Changed

- **Breaking:** Overhauled the logging system to use Python's standard `logging` module.
  - Removed the custom `anpe.utils.anpe_logger.ANPELogger`.
  - `ANPEExtractor` no longer accepts `log_level`, `log_dir`, or `log_file` in its configuration dictionary. Logging configuration is now the responsibility of the application using the library.
- **Breaking:** Removed logging configuration parameters (`log_level`, `log_dir`) from the `anpe.extract()` and `anpe.export()` convenience functions' `**kwargs`.
- Refactored internal noun phrase processing logic within `ANPEExtractor` to primarily use spaCy `Span` objects, improving integration and maintainability.

### Fixed

- Resolved Benepar tokenization assertion errors when processing text with complex newline patterns
- Improved text preprocessing to better handle newlines while respecting the `newline_breaks` configuration setting
- Enhanced error reporting when text cannot be processed, providing clearer guidance on text formatting requirements

### Added

- Added standard `logging.NullHandler()` to the package logger (`anpe`) to prevent warnings when the library is used without application-level logging configuration.

## [0.5.0] - 2025-05-03

### Added

- Detailed configuration settings used for extraction are now included in the results dictionary (`configuration` key).
- TXT export format now includes the full configuration used in the header.

### Changed

- **Breaking:** Modified the structure of the dictionary returned by `extract()`:
  - Removed the top-level `metadata` key (which contained timestamp and output flags).
  - Added a top-level `timestamp` key.
  - Added a top-level `configuration` key containing settings and output flags (`metadata_requested`, `nested_requested`).
- Updated the header format for standard output (when running CLI without `-o`) to reflect the new structure.
- Updated internal logic and tests (`export.py`, `cli.py`, `test_utils.py`) to be compatible with the new result structure.

### Fixed

- The `benepar_model_used` field in the output configuration now correctly reports the Benepar model actually loaded (especially relevant for auto-detection/fallback scenarios), instead of just the requested model.

## [0.4.0] - 2025-04-23

### Changed

- Refactored core parsing logic to integrate Benepar directly into the spaCy pipeline, removing the need for separate NLTK resource downloads.
- Simplified model loading, error handling, and NLTK path management.

### Removed

- Explicit dependency checks and download logic for NLTK `punkt` and `punkt_tab` resources.
- References to NLTK `punkt` and `punkt_tab` from documentation, setup utilities, and cleanup logic.

## [0.3.0] - 2025-04-17

### Changed

- All tests updated to provide more comprehensive testing on code functions.

### Added

- Customization on model selection via config arguments `spacy_model` and `benepar_model`.
- Automatic model selection logic if there are multiple models installed
- Model cleanup utility script `anpe.utils.clean_models.py` to remove downloaded models and caches.
- CLI `setup` command now accepts `--spacy-model` and `--benepar-model` arguments to install specific models individually or select non-default models; and accepts `--clean-models` flag to invoke the cleanup utility.

### Fixed

- Increased subprocess timeout for large Benepar model downloads to prevent failures.
- Improved robustness of NLTK model installation by removing potentially corrupted zip files if extraction is incomplete.
- Applied similar robustness improvement to Benepar model installation.

## [0.2.0] - 2025-04-05

### Changed

- Enhanced output path handling with intelligent detection of files vs directories
- Renamed CLI parameter from `--output-dir` to `--output` for flexibility
- Improved handling of unsupported file extensions (.xlsx, etc.)

### Added

- More informative logging for file paths and extensions
- Comprehensive tests for the new file handling logic
- Better documentation explaining path handling rules

## [0.1.0] - 2025-04-02

### Added

- Initial release of ANPE (Another Noun Phrase Extractor)
- Core extraction functionality using Berkeley Neural Parser
- CLI interface with extract and version commands
- Support for nested noun phrases
- Structural analysis of noun phrases
- Metadata output including length and structure types
- Multiple output formats: TXT, CSV, and JSON
- Flexible configuration system
- Logging system with file and console output
- Automatic model download functionality

### Features

- Customize extraction with min/max length filters
- Include or exclude pronouns
- Filter by structure types
- Toggle newline treatment as sentence boundaries
- Hierarchical output of nested noun phrases
- Export directly to files or display in console
