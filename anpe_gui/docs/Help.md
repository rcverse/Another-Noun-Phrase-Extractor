# ANPE GUI Help

This guide provides information on how to use the ANPE (Another Noun Phrase Extractor) GUI.

## Overview

The ANPE GUI allows you to extract noun phrases from text using various configuration options.

## Main Interface

-   **Header:** Displays the application title, version, and a 'Manage Models' button.
    -   **Manage Models:** Opens a dialog to download, check, or clean required language models (spaCy, Benepar).
-   **Input Tab:**
    -   **Input Mode:** Choose between 'File Input' (process one or more text files) or 'Text Input' (process text pasted directly).
    -   **File Input:** Add/Remove files, view the list of files to process.
    -   **Text Input:** Paste or type text directly into the text area.
    -   **Configuration Sections:**
        -   **General Settings:** Control output options like including nested phrases or metadata.
        -   **General Filtering:** Filter extracted phrases by minimum/maximum token length or accept/reject pronouns.
        -   **Structure Filtering:** Filter phrases based on their grammatical structure (e.g., only phrases with determiners, adjectives, etc.). Check the main box to enable/disable this section.
    -   **Reset Button:** Clears all inputs, outputs, and resets configuration to default.
    -   **Process Button:** Starts the extraction process using the current input and configuration. *This button is disabled during initialization or while processing is ongoing.*
-   **Output Tab:**
    -   **File Selector (Batch Mode):** Select which file's results to view when multiple files were processed.
    -   **Results Display:** Shows the extracted noun phrases, formatted hierarchically if nested phrases are included.
    -   **Export Options:** Export the displayed results to TXT, CSV, or JSON format.
    -   **Process New Input:** Same as the Reset button, prepares for a new task.
-   **Log Output Panel:** Displays detailed logs about the application's operations (initialization, processing steps, errors). Use the 'Filter Level' dropdown to control verbosity.
-   **Status Bar:** Shows the current application status (Ready, Processing, Error) and a progress bar during operations.

## Workflow

1.  **Initialization:** When the application starts, it checks for required models. If models are missing, the 'Manage Models' dialog may open automatically, or you can open it manually. **Pay attention to the Status Bar at the bottom.**
2.  **Input:** Select 'File Input' or 'Text Input'. Add files or paste text.
3.  **Configure:** Adjust filtering and output options as needed.
4.  **Process:** Click the 'Process' button (ensure it's enabled - wait for initialization if necessary).
5.  **View Output:** The application automatically switches to the 'Output' tab upon successful completion. Review the results.
6.  **Export (Optional):** Choose an export format and directory, then click 'Export Results'.
7.  **Reset/Process New:** Click 'Reset' or 'Process New Input' to start over for a new analysis task.

## Important Notes

*   **Check the Status Bar:** The Status Bar at the very bottom of the window is crucial. It tells you if the application is initializing models, ready for input, processing, has completed successfully, or encountered an error. Always check the status bar if things seem slow or unresponsive.
*   **Use the Reset Button:** Before starting a completely new analysis (e.g., processing different text or files with different settings), it's good practice to click the **Reset** button (or the "Process New Input" button on the Output tab). This clears previous inputs, outputs, and ensures a clean state for the next run.

## Troubleshooting

-   **Process Button Disabled:**
    -   The application might still be initializing or checking models. **Check the Status Bar** - wait for it to show 'ANPE Ready'.
    -   A previous process might still be running (**check Status Bar**/logs).
    -   Required models might be missing. Use 'Manage Models' to check and download.
    -   An error might have occurred during initialization (check logs).
    -   Try clicking **Reset** if you think the application state is stuck.
-   **Errors During Processing:** Check the Log Output panel for specific error messages.
-   **Incorrect Results:** Review your configuration settings (filters, nested phrases).

---
*Need more help? Check the project documentation or report an issue.*