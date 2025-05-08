import logging
import os
import sys
import json

# Attempt to import ANPEExtractor
try:
    from anpe import ANPEExtractor
except ImportError:
    print("Failed to import ANPEExtractor. Please ensure that the ANPE_core directory (or the directory containing the 'anpe' package) is in your PYTHONPATH, or that the 'anpe' package is installed in your environment.")
    sys.exit(1)

# --- Setup Debug Logging to File ---
LOG_FILE_NAME = "normalization_test_output.log"
# Remove old log file if it exists
if os.path.exists(LOG_FILE_NAME):
    os.remove(LOG_FILE_NAME)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
    filename=LOG_FILE_NAME,
    filemode='w'
)

# Add a handler to also print INFO and above to console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger('').addHandler(console_handler)


def run_normalization_tests():
    """
    Runs a series of tests with sentences designed to challenge
    the token normalization logic in ANPEExtractor.
    """
    logging.info(f"Starting normalization tests. Detailed DEBUG logs will be in {LOG_FILE_NAME}")

    try:
        # Initialize the extractor
        # Ensure you have 'en_core_web_sm' or your desired spaCy model downloaded
        # and benepar.download('benepar_en3') has been run.
        extractor = ANPEExtractor()
        logging.info("ANPEExtractor initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize ANPEExtractor: {e}", exc_info=True)
        print(f"Critical error: Failed to initialize ANPEExtractor. Check {LOG_FILE_NAME} for details. Ensure models are downloaded.")
        return

    test_sentences = [
        "This is a state-of-the-art solution.",
        "Her well-being is important.",
        "The price is $1,234.56, not €20.00.",
        "Please visit example.com or contact test@example.org.",
        "I don't know what he'll say, it's all new.",
        "It's John's book, not the students' teacher's car.",
        "She said, \"Wait... what was that for...?\"",
        "We need item* and option A/B for the test-run.",
        "PTB uses ``double quotes'' and `single quotes` for `example`.", # Test existing quote handling
        "This sentence has no (obvious) issues.", # Control case
        "A sentence with 'typographical' quotes and a modifier apostrophe' like in some texts.", # Typographical quotes
        "Let's try this: work-(out)." # Hyphen with parenthesis (Benepar might treat '-' and '(' differently)
    ]

    logging.info(f"Processing {len(test_sentences)} test sentences...")

    for i, sentence in enumerate(test_sentences):
        logging.info(f"--- Test Case {i+1}/{len(test_sentences)} ---")
        logging.info(f'Original Sentence: "{sentence}"')
        print(f"\nProcessing sentence {i+1}: \"{sentence}\"")
        try:
            # The actual extraction will trigger the normalization and logging we want to observe
            extraction_result = extractor.extract(sentence)
            actual_nps = extraction_result.get("results", [])
            
            np_count = len(actual_nps)
            logging.info(f"Successfully processed. Extracted NP count: {np_count}")
            
            if actual_nps:
                logging.debug(f"Extracted NPs: {json.dumps([np_dict['noun_phrase'] for np_dict in actual_nps], indent=2)}")
            else:
                logging.debug("No noun phrases extracted.")
            print(f"Finished processing sentence {i+1}. NPs found: {np_count}")

        except Exception as e:
            logging.error(f'Error processing sentence: "{sentence}". Error: {e}', exc_info=True)
            print(f"Error processing sentence {i+1}. Check {LOG_FILE_NAME} for details.")
        logging.info("--- End Test Case ---")

    logging.info("Normalization tests completed.")
    print(f"\nAll normalization tests finished. Please review {LOG_FILE_NAME} for detailed DEBUG output.")

if __name__ == "__main__":
    run_normalization_tests() 