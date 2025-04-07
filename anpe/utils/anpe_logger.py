"""Logging functionality for ANPE."""

import logging as python_logging  # Renamed to avoid collision
import os
import sys
from pathlib import Path
from typing import Optional


class ANPELogger:
    """Centralized logging management for ANPE."""
    
    _instance = None
    
    def __new__(cls, log_level=None, log_file=None):
        """Implement singleton pattern for the logger."""
        if cls._instance is None:
            cls._instance = super(ANPELogger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_level: Optional[str] = None, log_file: Optional[str] = None):
        """
        Initialize the logging system.
        """
        # Skip initialization if already done (singleton pattern)
        if getattr(self, '_initialized', False):
            if log_level:
                self.set_level(log_level)
            if log_file and log_file != self.log_file:
                self.set_log_file(log_file)
            return
            
        self.logger = python_logging.getLogger('anpe')  # Updated reference
        self.log_level = log_level or os.environ.get('ANPE_LOG_LEVEL', 'INFO')
        self.log_file = log_file
        
        # Set up the logger
        self._setup_logger()
        self._initialized = True
        
        # Log the initialization
        self.logger.debug("ANPE Logger initialized")
    
    def _setup_logger(self):
        """Set up the logger with handlers and formatter."""
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Configure logger level
        self.logger.setLevel(getattr(python_logging, self.log_level.upper()))  # Updated reference
        
        # Create formatter
        formatter = python_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  # Updated reference
        
        # Create console handler
        console_handler = python_logging.StreamHandler(stream=sys.stdout)  # Updated reference
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Create file handler if log file is specified
        if self.log_file:
            try:
                file_path = Path(self.log_file)
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                file_handler = python_logging.FileHandler(file_path, encoding='utf-8')  # Updated reference
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                self.logger.info(f"File logging enabled: {file_path}")

            except (IOError, PermissionError) as e:
                # Fall back to just console logging if file can't be opened
                self.logger.warning(f"Could not set up file logging: {str(e)}")
    
    def set_level(self, level: str):
        """
        Change the logging level.
        """
        self.log_level = level.upper()
        self.logger.setLevel(getattr(python_logging, self.log_level))  # Updated reference
        
        # Log the level change
        self.logger.debug(f"Log level changed to {self.log_level}")
    
    def get_logger(self, name: Optional[str] = None):
        """
        Get a logger instance, optionally for a specific component.
        """
        if name:
            return python_logging.getLogger(f'anpe.{name}')  # Updated reference
        return self.logger

    def set_log_file(self, log_file: str):
        """
        Set a new log file.
        """
        self.log_file = log_file
        self._setup_logger()  # Reconfigure the logger with the new file
        self.logger.info(f"Logging redirected to file: {log_file}")


# Create a default logger instance
get_logger = ANPELogger().get_logger 