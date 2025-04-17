"""
ANPE: Another Noun Phrase Extractor
==================================

Accurate noun phrase extraction using the Berkeley Neural Parser.
"""

from anpe.extractor import ANPEExtractor
from anpe.extractor import extract, export

__version__ = '0.3.0'

__all__ = ['ANPEExtractor', 'extract', 'export', '__version__'] 