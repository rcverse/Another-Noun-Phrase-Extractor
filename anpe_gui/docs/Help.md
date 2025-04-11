# ANPE GUI Help

Welcome to the ANPE (Another Noun Phrase Extractor) GUI! This application provides a user-friendly interface to the core ANPE library, allowing you to easily extract noun phrases from text without needing to write code.

## Starting the app
When you start the app, ANPE will automatically check for required language models in the background. During this initialization phase:

- The "Process" button will be temporarily disabled
- The status bar at the bottom will show "Checking ANPE models..."
- Required models include spaCy English model, Benepar parsing model, and NLTK tokenizer data

If all required models are present, the status will change to "ANPE Ready" and the "Process" button will become enabled, allowing you to begin extraction.

If any models are missing, a *setup wizard* will appear to guide you through downloading and installing the required components. This one-time setup requires an internet connection and approximately 120MB of disk space. You can also access model management later via the <button>Settings</button> icon in the top-right corner.

## Usage
The application is organized into two main tabs: <option>Input</option> and <option>Output</option>.

### Input Tab
This is where you prepare your text and configure the extraction process.

#### Input Options
You can choose between two ways to provide text:

1.  **File Input**
This tab allows you to batch process multiple files or simply process single file.
    *   The <button>Add Files</button> button: Select individual <format>.txt</format> files
    *   The <button>Add Directory</button> button: Add all <format>.txt</format> files within a selected folder
    *   The <button>Remove/Clear All</button> options: Manage the list of files to process

2.  **Text Input**
This tab allows you to type text directly into the editor for quick analysis.
    *   The <button>Paste</button> button: Insert text from your clipboard
    *   The <button>Clear</button> button: Empty the text input area
   

#### Configuration
Fine-tune the extraction process:

*   **General Settings**:
    *   The <option>Include nested phrases</option> option: Captures noun phrases embedded within larger ones
    *   The <option>Add metadata to output</option> option: Includes length and structures information for each phrase
    *   The <option>Do not treat newlines as sentence boundaries</option> option: Controls how line breaks are interpreted Check this box will ignore line breaker as sentence boundaries; suitable when processing files with irregular line breaking.

*   **Filtering Options**:
    *   The <option>Min Length</option> and <option>Max Length</option> settings: Specify the word count limits for phrases
    *   The <option>Do not accept Pronouns</option> option: Controls whether single-word pronouns are included

*   **Structure Filtering**:
    *   Enable the main toggle to activate structure filtering
    *   Select specific structures from the list
        *   **<option>Determiner</option>**: Phrases starting with articles or determiners (e.g., 'the cat', 'a house').
        *   **<option>Adjectival Modifier</option>**: Phrases containing adjectives that describe the noun (e.g., 'red car', 'beautiful day').
        *   **<option>Prepositional Modifier</option>**: Phrases where a prepositional phrase follows the noun (e.g., 'book on the table').
        *   **<option>Compound</option>**: Phrases made of multiple nouns acting as one unit (e.g., 'noun phrase extractor', 'apple pie').
        *   **<option>Possessive</option>**: Phrases showing ownership (e.g., "John's book", 'her bag').
        *   **<option>Quantified</option>**: Phrases indicating quantity or number (e.g., 'two dogs', 'many people').
        *   **<option>Coordinated</option>**: Phrases linking items with 'and' or 'or' (e.g., 'cats and dogs').
        *   **<option>Appositive</option>**: Phrases where one noun phrase renames another, often set off by commas (e.g., 'Bob, my friend').
        *   **<option>Relative Clause</option>**: Phrases containing a clause that modifies the noun (e.g., 'the man who called').
        *   **<option>Reduced Relative Clause</option>**: A relative clause modifying a noun, but without the relative pronoun (e.g., 'the book *written by him*').
        *   **<option>Nonfinite Complement</option>**: A phrase (often starting with 'to' or ending in '-ing') that completes the meaning of a noun (e.g., 'the *decision to leave*').
        *   **<option>Finite Complement</option>**: A full clause (often starting with 'that' or 'whether') that completes the meaning of a noun (e.g., 'the *idea that he left*').
        *   **<option>Pronoun</option>**: Phrases consisting only of a pronoun (e.g., 'he', 'it', 'they').
        *   **<option>Standalone Noun</option>**: Phrases consisting only of a single noun or proper noun (e.g., 'book', 'John').
        *   **<option>Others</option>**: Any other identified noun phrase structures not fitting the above categories.

#### Control Buttons
*   The <button>Process</button> button: Starts the noun phrase extraction based on your input and configuration. Processing runs in the background to keep the GUI responsive.
*   The <button>Reset</button> button: Clears all input fields, file lists, and configuration settings
*   The <button>Default</button> button: Reverts settings and filtering options to defaults

### Output Tab
This tab displays the results of the extraction process.

#### Viewing Results
*   The main area shows the extracted noun phrases formatted according to your settings
*   If processing multiple files, a dropdown menu appears above the results
*   If nested phrases were included, the display will by default collapsed all nested NP; click on the NP entry to show the nested NP.

#### Detached Results Viewer
*   Click the <button>⇱</button> button in the upper-right corner of the results to open the results in a detached window
*   The detached window provides a larger, resizable view that can be moved independently
*   All functionality from the main view is preserved, including:
    *   Search filtering to find specific phrases
    *   Sorting options (by order, length, and structure)
    *   Expanding/collapsing nested phrases
*   Keyboard shortcuts in the detached window:
    *   <kbd>Ctrl+E</kbd>: Expand all items
    *   <kbd>Ctrl+C</kbd>: Collapse all items
    *   <kbd>Ctrl+F</kbd>: Focus the search filter

#### Exporting Results
*   Click the <button>Export</button> button to save the results; if there are multiple files, it will export all results automatically.
*   Choose an output format:
    *   <format>TXT</format>: Human-readable, plain text list
    *   <format>CSV</format>: Comma-Separated Values for spreadsheets
    *   <format>JSON</format>: Best for programmatic use, preserving hierarchy
*   Select a destination directory to save the file(s)

#### Filename Structure
Exported files are named automatically to ensure uniqueness and provide context:
*   **Batch Export:**
    `[prefix_]original_filename_anpe_results_YYYYMMDD_HHMMSS.format`
*   **Single Export:**
    `[prefix_]anpe_text_results_YYYYMMDD_HHMMSS.format`

Where:
*   `[prefix_]` is the optional prefix you enter in the Export Options.
*   `original_filename` is the name of the input file (without extension).
*   `YYYYMMDD_HHMMSS` is the timestamp of the export.
*   `format` is the selected format (txt, csv, json).



## Tips
### Choose the right format
*   **<format>TXT</format> Format (.txt)**
    *   Outputs noun phrases in a human-readable, plain text format.
    *   Nested phrases are shown using indentation.
    *   **Recommendation:** Best for quick reading, simple lists, or when you just need the plain text of the phrases, especially top-level ones.

*   **<format>CSV</format> Format (.csv)**
    *   Outputs results as Comma-Separated Values, suitable for spreadsheets (like Excel).
    *   Provides a flat structure where each row is a noun phrase.
    *   Hierarchy is represented using `ID` and `Parent_ID` columns.
    *   Includes columns for `Level`, `Length`, and `Structures` if metadata is enabled.
    *   **Recommendation:** Best for importing into spreadsheets or other data analysis tools that work well with tabular data.

*   **<format>JSON</format> Format (.json)**
    *   Outputs results in JavaScript Object Notation.
    *   Maintains the full hierarchical structure using nested objects.
    *   Includes all metadata.
    *   **Recommendation:** Best for programmatic use, preserving hierarchy, or detailed inspection.

### Performance
Extracting from large files or many files can take time. ANPE processes files in the background so the interface remains usable. Monitor the status bar at the bottom for progress updates (e.g., files completed, phrases extracted) and any potential errors.

### Batch Processing
When extracting from a directory, ANPE saves one output file per input file; You may wish to set up a new folder to save these files, and add prefix to your filename to help you understand the nature of the processing.

## Acknowledgements
The ANPE GUI relies on several open-source libraries:

*   **PyQt6** (GPLv3 or Commercial): For the graphical user interface framework
*   **spaCy** (MIT License): For initial text processing (tokenization, sentence segmentation)
*   **Benepar** (MIT License): For constituency parsing (identifying phrase structures)
*   **NLTK** (Apache License 2.0): Used by Benepar and for certain tokenizer data

We are grateful for the developers of these packages that make ANPE and ANPE GUI possible. You can find more detailed license information in the application's About dialog.

## Contact
*   **Project Page**: Visit the [ANPE GitHub Repository](https://github.com/rcverse/anpe) for source code and documentation
*   **Email**: For questions or feedback, contact rcverse6@gmail.com