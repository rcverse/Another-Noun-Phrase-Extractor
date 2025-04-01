"""
ANPE: Another Noun Phrase Extractor
==================================

Accurate noun phrase extraction using the Berkeley Neural Parser.
"""

from anpe.extractor import ANPEExtractor
from anpe.extractor import extract 

__version__ = '0.1.0'

__all__ = ['ANPEExtractor', 'extract', '__version__'] 