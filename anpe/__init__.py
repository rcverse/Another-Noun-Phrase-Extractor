"""
ANPE: Another Noun Phrase Extractor
==================================

Accurate noun phrase extraction using the Berkeley Neural Parser.
"""

# Import necessary for NullHandler setup
import logging

# Define package version
__version__ = "1.0.3"

# Import key classes/functions for easier access
from anpe.extractor import ANPEExtractor, extract, export

# Setup library logger to prevent "No handlers found" warnings
# Applications using ANPE should configure their own logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = ['ANPEExtractor', 'extract', 'export', '__version__'] 