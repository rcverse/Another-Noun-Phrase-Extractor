# Changelog

All notable changes to the ANPE project will be documented in this file.

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