# ANPE GUI Help

Welcome to the ANPE (Another Noun Phrase Extractor) GUI! This application provides a user-friendly interface to the core ANPE library, allowing you to easily extract noun phrases from text without needing to write code.

## Usage
The application is organized into two main tabs: **Input** and **Output**.

### Input Tab
This is where you prepare your text and configure the extraction process.

#### Input Options
You can choose between two ways to provide text:

1. **Working with Files**
   * The "Add Files" button: Select individual .txt files
   * The "Add Directory" button: Add all .txt files within a selected folder
   * The "Remove/Clear All" options: Manage the list of files to process

2. **Direct Text Entry**
   * The "Paste" button: Insert text from your clipboard
   * The "Clear" button: Empty the text input area
   * You can also type text directly into the editor for quick analysis

#### Configuration
Fine-tune the extraction process:

* **General Settings**:
  * The "Include nested phrases" option: Captures noun phrases embedded within larger ones
  * The "Add metadata to output" option: Includes length and structures information for each phrase
  * The "Treat newlines as sentence boundaries" option: Controls how line breaks are interpreted

* **Filtering Options**:
  * The "Min Length" and "Max Length" settings: Specify the word count limits for phrases
  * The "Accept Pronouns" option: Controls whether single-word pronouns are included

* **Structure Filtering**:
  * Enable the main toggle to activate structure filtering
  * Select specific structures from the list (like Determiner, Compound, Appositive)

#### Control Buttons
* The "Process" button: Starts the noun phrase extraction based on your input and configuration
* The "Reset" button: Clears all input fields, file lists, and configuration settings
* The "Default" button: Reverts settings and filtering options to defaults

### Output Tab
This tab displays the results of the extraction process.

#### Viewing Results
* The main area shows the extracted noun phrases formatted according to your settings
* If processing multiple files, a dropdown menu appears above the results
* If nested phrases were included, the display will show the hierarchy

#### Exporting Results
* Click the "Export" button to save the results
* Choose an output format:
  * TXT: Human-readable, plain text list
  * CSV: Comma-Separated Values for spreadsheets
  * JSON: Best for programmatic use, preserving hierarchy
* Select a destination directory to save the file(s)

## Tips
* **Choosing Export Format**:
  * Use TXT for simple lists and easy reading
  * Use CSV if you plan to analyze the phrases in a spreadsheet
  * Use JSON if you need to feed the results into another program

* **Large Files**: Extracting from large files can take time. Monitor the status bar for progress

* **Configuration**: Start with default settings. Adjust length filters or disable pronoun inclusion if needed

* **Batch Processing**: When extracting from a directory, ANPE saves one output file per input file

## Acknowledgements
The ANPE GUI relies on several open-source libraries:

* **ANPE Core Engine**: The underlying extraction logic
* **PyQt6**: For the graphical user interface framework
* **spaCy**: For initial text processing (tokenization, sentence segmentation)
* **Benepar**: For constituency parsing (identifying phrase structures)
* **NLTK**: Used by Benepar and for certain tokenizer data

This GUI application was developed with significant assistance from AI, providing an accessible interface to the powerful ANPE library.

## Contact
* **Project Page**: Visit the [ANPE GitHub Repository](https://github.com/rcverse/anpe) for source code and documentation
* **Email**: For questions or feedback, contact rcverse6@gmail.com