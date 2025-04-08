    def run(self):
        """Initialize the ANPEExtractor in the background."""
        logging.info("Initializing ANPEExtractor...")
        try:
            from anpe.utils import setup_models
            # First check if models are present
            if not setup_models.check_all_models_present():
                logging.info("Required models not found, setup needed.")
                self.models_missing.emit()
                return
                
            # Models are present, initialize extractor
            extractor = ANPEExtractor()
            logging.info("ANPEExtractor initialized successfully.")
            self.initialized.emit(extractor)
        except Exception as e:
            logging.error(f"Error initializing ANPEExtractor: {e}", exc_info=True)
            self.error.emit(str(e)) 