"""
Worker for handling single text extraction in a background thread.
"""

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from anpe import ANPEExtractor
from typing import Dict, Any # Import Dict and Any
import logging

class ExtractionSignals(QObject):
    """Defines signals available from the ExtractionWorker."""
    started = pyqtSignal()
    result = pyqtSignal(dict) # Emits the extraction result dictionary
    error = pyqtSignal(str)   # Emits error message string
    finished = pyqtSignal() # Emitted when processing finishes (success or error)

class ExtractionWorker(QObject):
    """Performs ANPE extraction on a single text string using provided config."""
    
    def __init__(self, text_content: str, config: Dict[str, Any], anpe_version: str,
                 include_metadata: bool, include_nested: bool):
        super().__init__()
        # Store config and input data
        self.text_content = text_content
        self.config = config
        self.anpe_version = anpe_version # May not be needed by worker, but passed from main
        self.include_metadata = include_metadata # For output formatting
        self.include_nested = include_nested   # For output formatting
        self.signals = ExtractionSignals()

    @pyqtSlot()
    def run(self):
        """Execute the extraction process."""
        self.signals.started.emit()
        logging.debug("WORKER (Text): Starting extraction...")
        try:
            # Create extractor instance with the specific config for this run
            logging.debug(f"WORKER (Text): Creating ANPEExtractor with config: {self.config}")
            extractor = ANPEExtractor(config=self.config)
            
            # Perform extraction
            logging.debug(f"WORKER (Text): Extracting from text (len={len(self.text_content)}). Options: meta={self.include_metadata}, nested={self.include_nested}")
            result_data = extractor.extract(
                text=self.text_content,
                metadata=self.include_metadata,
                include_nested=self.include_nested
            )
            logging.info("WORKER (Text): Extraction successful.")
            self.signals.result.emit(result_data)
        except Exception as e:
            logging.error(f"WORKER (Text): Error during extraction: {e}", exc_info=True)
            self.signals.error.emit(str(e))
        finally:
            logging.info("WORKER (Text): Finishing.")
            self.signals.finished.emit() 