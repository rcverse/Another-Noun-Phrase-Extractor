"""Default configuration for ANPE."""

DEFAULT_CONFIG = {
    # Core Settings
    "accept_pronouns": True,    # Accept pronouns as valid NPs
    "min_length": None,         # Minimum NP length in tokens (None = no limit)
    "max_length": None,         # Maximum NP length in tokens (None = no limit)
    "log_level": "INFO",        # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    "log_dir": None,            # Directory path for log files (None = no logging to file)
    
    # Sentence Breaking Control
    "newline_breaks": True,     # Treat newlines as sentence boundaries
    
    # Structure Filtering
    "structure_filters": [],     # List of structure patterns to include (empty = include all)
}