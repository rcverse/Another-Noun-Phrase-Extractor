# Changelog

All notable changes to the ANPE project will be documented in this file.

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

