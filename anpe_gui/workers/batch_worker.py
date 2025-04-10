"""
Worker for handling batch file extraction in a background thread.
"""

import os
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from anpe import ANPEExtractor
from typing import Dict, Any, List, Optional 
import logging 

class BatchSignals(QObject):
    """Defines signals available from the BatchWorker."""
    started = pyqtSignal()
    status_update = pyqtSignal(str)    # NEW: For real-time status messages
    progress = pyqtSignal(int, str) # Percentage, Status message (context for % update)
    error = pyqtSignal(str)        # Emits error message string
    finished = pyqtSignal()      # Emitted when processing finishes (success or error)
    file_result = pyqtSignal(str, dict) # Emits file path and result dictionary

class BatchWorker(QObject):
    """Performs ANPE extraction on multiple files using provided config."""
    
    def __init__(self, file_paths: List[str], config: Dict[str, Any], anpe_version: str,
                 include_metadata: bool, include_nested: bool):
        super().__init__()
        # Store config and input data
        self.file_paths = file_paths
        self.config = config
        self.anpe_version = anpe_version # May not be needed by worker, but passed from main
        self.include_metadata = include_metadata # For output formatting
        self.include_nested = include_nested   # For output formatting
        self.signals = BatchSignals()
        self._is_cancelled = False

    @pyqtSlot()
    def run(self):
        """Execute the batch extraction process."""
        self.signals.started.emit()
        logging.info(f"Starting processing for {len(self.file_paths)} files...")
        results = {}
        total_files = len(self.file_paths)
        extractor: Optional[ANPEExtractor] = None # Define extractor variable
        
        try:
            # Create extractor instance ONCE with the specific config for this batch run
            logging.debug(f"Creating ANPEExtractor with config: {self.config}")
            extractor = ANPEExtractor(config=self.config)

            for i, file_path in enumerate(self.file_paths):
                if self._is_cancelled:
                    logging.info("Cancellation requested.")
                    break
                
                file_name = os.path.basename(file_path)
                # Emit status update BEFORE processing the file
                status_msg_processing = f"Processing ({i+1}/{total_files}): {file_name}"
                self.signals.status_update.emit(status_msg_processing)

                try:
                    # Log start of processing this specific file
                    logging.info(f"Processing ({i+1}/{total_files}): {file_name}")
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Use the pre-configured extractor instance
                    logging.debug(f"Extracting from file '{file_name}'. Options: meta={self.include_metadata}, nested={self.include_nested}")
                    file_result = extractor.extract(
                        text=text,
                        metadata=self.include_metadata,
                        include_nested=self.include_nested
                    )
                    results[file_path] = file_result
                    # Emit result for this single file immediately
                    self.signals.file_result.emit(file_path, file_result)
                    
                except Exception as file_e:
                    logging.error(f"Error processing file {file_path}: {file_e}", exc_info=True)
                    error_info = {"error": str(file_e)}
                    results[file_path] = error_info
                    # Emit error info for this file
                    self.signals.file_result.emit(file_path, error_info)
                    # Continue processing other files
                
                # MOVED progress calculation and emit to *after* processing the file
                progress_percent = int(((i + 1) / total_files) * 100)
                self.signals.progress.emit(progress_percent, "") # Send empty string


            if not self._is_cancelled:
                logging.info("Batch processing successful.")
                # No longer need to emit the full results dict here, handled per file
                # self.signals.result.emit(results) 
            else:
                logging.info("Batch processing cancelled.")

        except Exception as e:
            logging.error(f"Unhandled error during batch processing: {e}", exc_info=True)
            self.signals.error.emit(str(e))
        finally:
            logging.info("Finishing.")
            self.signals.finished.emit()

    def cancel(self):
        """Request cancellation of the batch process."""
        logging.info("Received cancellation request.")
        self._is_cancelled = True 