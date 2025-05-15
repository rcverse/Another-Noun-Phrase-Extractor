# ANPE: Another Noun Phrase Extractor

![ANPE Banner](/pics/banner.png)

[![Build Status](https://github.com/rcverse/another-noun-phrase-extractor/actions/workflows/python-package.yml/badge.svg)](https://github.com/rcverse/another-noun-phrase-extractor/actions/workflows/python-package.yml)
[![pytest](https://img.shields.io/badge/pytest-passing-brightgreen)](https://github.com/rcverse/another-noun-phrase-extractor/actions/workflows/python-package.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/anpe)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ANPE (*Another Noun Phrase Extractor*) is a lightweight Python library for **directly extracting complete noun phrases from text**. This library leverages the [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser) (via the `benepar` package) integrated with [spaCy](https://spacy.io/) for precise parsing. The resulting constituency trees are then processed (using [NLTK](https://www.nltk.org/) tree structures) for NP extraction. 
On top of that, ANPE utilizes [spaCy](https://spacy.io/)'s dependency parsing to identify and label the syntactic structures of noun phrases, such as "`appoistive`", "`relative_clause`", or "`finite_complement`", etc.
ANPE provides flexible configuration options to **include nested NP**, **filter specific structural types of NP**, or **taget length requirements**, as well as options to **export to files** in multiple structured formats directly.

Currently, ANPE only supports **English** and is compatible with Python **3.9** through **3.12**.

## **Key Features**:

1. **✅Precision Extraction**: Accurate noun phrase identification using modern parsing techniques
2. **🏷️Structural Labelling**: Identifies and labels NPs with their different syntactic patterns
3. **✍🏻Hierarchical Analysis**: Supports both top-level and nested noun phrases
4. **⚙️Customizable Processing**: Flexible configuration options for filtering and analysis
5. **📄Flexible Output**: Multiple formats (TXT, CSV, JSON) with consistent structure
6. **⌨️CLI Integration**: Command-line interface for easy text processing

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Command-line Interface](#Command-line-interface)
- [GUI Application](#gui-application)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## TL;DR

### Quick Start

1. **Install**:

   ```bash
   pip install anpe
   ```
2. **Setup Models**:

   ```bash
   anpe setup
   ```
3. **Extract Noun Phrases**:

   ```python
   import anpe

   text = "Your texts here"
   result = anpe.extract(text)

   print(result)
   ```

   Or with CLI:

   ```bash
   anpe extract "Your text here"
   ```

### GUI App

[To be released]

## Installation

Please use pip to install.

```bash
pip install anpe
```

### Prerequisites

#### **Required Models**

ANPE relies on several pre-trained models for its functionality. The default setup uses the following:

1. **spaCy Model**: `en_core_web_md` (English language model for tokenization and sentence segmentation).
2. **Benepar Model**: `benepar_en3` (English constituency parser for syntactic analysis).

ANPE also supports using alternative spaCy models (`en_core_web_sm`, `en_core_web_lg`, `en_core_web_trf`) and a larger Benepar model (`benepar_en3_large`) for different performance/accuracy trade-offs. These can be designated for extraction via configuration.

#### **Automatic Setup**

ANPE provides a built-in tool to setup the necessary models. When you run the extractor, ANPE will automatically check if the default models are installed and install them if they're not. However, it is **recommended** to run the setup utility before you start using the extractor for the first time.
To setup the **default** models, simply run the following command in terminal (Please refer to [CLI usage](#command-line-interface) for more options):

```bash
anpe setup
```

You can also specify which models to *install* using the `--spacy-model` and `--benepar-model` flags with model aliases (e.g., `sm`, `md`, `lg`, `trf` for spaCy; `default`, `large` for Benepar; or  `all` flag to install all models). This allows for installation of non-default models or targeted installation if only one type of model is needed. For example:

```bash
anpe setup --spacy-model lg
```

Refer to the [CLI documentation](#setup-command-options) for details.

#### **Model Cleanup**

If you need to remove the downloaded models and caches (e.g., to free up space or resolve potential corruption), ANPE provides a cleanup utility.

To remove all models:

```bash
anpe setup --clean-models
```

For more fine-grained control, you can remove specific models:

```bash
# Remove a specific spaCy model
anpe setup --clean-spacy md

# Remove a specific Benepar model
anpe setup --clean-benepar default
```

All cleanup commands will prompt for confirmation before removing models. To bypass the confirmation, use the `--force` (or `-f`) flag:

```bash
anpe setup --clean-models --force
anpe setup --clean-spacy lg --force
```

> **⚠️ Warning:** Running the cleanup commands will remove the specified models from their standard locations. You will need to run `anpe setup` or let the extractor auto-download them again before using ANPE.

#### **Manual Setup**

If automatic setup fails or you prefer to manually download the models, you can run install the models manually. Below are examples for the **default** models:

```bash
# Install default spaCy model; Other options: en_core_web_sm, en_core_web_lg, en_core_web_trf
python -m spacy download en_core_web_md
# Install default benepar model; Other option: benepar_en3_large
python -m benepar.download benepar_en
```

## Usage

The primary way to use ANPE is through its Python API.

### Basic Usage

It is recommended to create your own `ANPEExtractor` instance for reusability throughout your code and better readability.

```python
import anpe

# Initialize extractor with default settings
extractor = anpe.ANPEExtractor()

# Sample text
text = """
In the summer of 1956, Stevens, a long-serving butler at Darlington Hall, decides to take a motoring trip through the West Country. The six-day excursion becomes a journey into the past of Stevens and England, a past that takes in fascism, two world wars, and an unrealised love between the butler and his housekeeper.
"""

# Extract noun phrases
result = extractor.extract(text)

# Print results
print(result)
```

### Advance Usage

By defining your configuration and controlling the parameters, you can tailor your extractor to your specific needs. Here's an example of how you might use ANPE to extract noun phrases with specific lengths and structures:

```python
from anpe import ANPEExtractor

# Create extractor with custom settings
extractor = ANPEExtractor({
    "min_length": 2,
    "max_length": 5,
    "accept_pronouns": False,
    "structure_filters": ["compound", "appositive"],
    "newline_breaks": False,
    "spacy_model": "lg",         # Use 'lg' spaCy model for this extraction
    "benepar_model": "default"   # Use default Benepar model for this extraction
})

# Sample text
text = """
In the summer of 1956, Stevens, a long-serving butler at Darlington Hall, decides to take a motoring trip through the West Country.
"""

# Extract with metadata and nested NPs
result = extractor.extract(text, metadata=True, include_nested=True)

# Print result
print(result)

```

To achieve this, you need to customize the extraction parameters and configuration.

### Extraction Parameters

The `extract()` method accepts the following parameters:

| Parameter          | Type | Default  | Description                                                 |
| ------------------ | ---- | -------- | ----------------------------------------------------------- |
| `text`           | str  | Required | Input text to process                                       |
| `metadata`       | bool | False    | Whether to include metadata (`length` and `structures`) |
| `include_nested` | bool | False    | Whether to include nested noun phrases                      |

- **Metadata**: When set to `True`, the output will include two types of additional information about each noun phrase: `length` and `structures'

  - **`length`** is the number of words that the NP contains
  - **`structures`** is the syntactic structure that the NP contains, such as `appositive`, `coordinated`, `nonfinite_complement`, etc.
- **Include Nested**: When set to `True`, the output will include nested noun phrases, allowing for a hierarchical representation of noun phrases.

> **📌 Note on Metadata:**
> **Structural analysis** is performed using the analyzer tool built into ANPE. It analyzes the NP's structure and label the NP with the structures it detected. Please refer to the [Structural Analysis](#structural-analysis) section for more details.

### Configuration Options

ANPE provides a flexible configuration system to further customize the extraction process. These options can be passed as a dictionary when initializing the extractor.

| Option                | Type          | Default  | Description                                                                                                                                                                                                                            |
| --------------------- | ------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `min_length`        | Integer       | `None` | Minimum token length for NPs. NPs with fewer tokens will be excluded.                                                                                                                                                                  |
| `max_length`        | Integer       | `None` | Maximum token length for NPs. NPs with more tokens will be excluded.                                                                                                                                                                   |
| `accept_pronouns`   | Boolean       | `True` | Whether to include single-word pronouns as valid NPs. When set to `False`, NPs that consist of a single pronoun will be excluded.                                                                                                    |
| `structure_filters` | List[str]     | `[]`   | List of structure types to include. Only NPs containing at least one of these structures will be included. If empty, all NPs are accepted.                                                                                             |
| `newline_breaks`    | Boolean       | `True` | Whether to treat newlines as sentence boundaries. Setting to `False` treats text as continuous across line breaks. See [Newline Handling](#newline-handling) for details on ANPE's newline processing behavior.                         |
| `spacy_model`       | Optional[str] | `None` | Specify the spaCy model alias/name to*use* for extraction. Accepts aliases (`"sm"`, `"md"`, `"lg"`, `"trf"`) or full names (e.g., `"en_core_web_lg"`). If `None`, ANPE attempts to auto-detect the best installed model. |
| `benepar_model`     | Optional[str] | `None` | Specify the Benepar model alias/name to*use* for extraction. Accepts aliases (`"default"`, `"large"`) or full names (e.g., `"benepar_en3_large"`). If `None`, ANPE attempts to auto-detect the best installed model.         |

Example:

```python
# Configure the extractor with multiple options
custom_extractor = ANPEExtractor({
    "min_length": 2,                # Only NPs with 2+ words
    "max_length": 5,                # Only NPs with 5 or fewer words
    "accept_pronouns": False,       # Exclude single-word pronouns
    "structure_filters": ["determiner"],  # Only include NPs with these structures
    "newline_breaks": False,         # Don't treat newlines as sentence boundaries
    "spacy_model": "lg",             # Explicitly use the large spaCy model
    "benepar_model": "default"        # Explicitly use the default Benepar model
})
```

**Minimum Length Filtering**
The `min_length` option allows you to filter out shorter noun phrases that might not be meaningful for your analysis. For example, setting `min_length=2` will exclude single-word noun phrases.

**Maximum Length Filtering**
The `max_length` option lets you limit the length of extracted noun phrases. For instance, setting `max_length=5` will exclude noun phrases with more than five words, focusing on more concise expressions.

**Pronoun Handling**
The `accept_pronouns` option controls whether pronouns like "it", "they", or "this" should be considered as valid noun phrases. When set to `False`, single-word pronouns will be excluded from the results.

**Structure Filtering**
Structure filtering allows you to target specific types of noun phrases in your extraction. You can specify a list of structure types to include in the results. When using `structure_filters`, only noun phrases that contain at least one of the specified structures will be included. This allows for targeted extraction of specific NP types.
*(Please refer to the [Structural Analysis](#structural-analysis) section for more details.)*

> 📌 **Note on Structure Filtering:**
> Note that structure filtering requires analyzing the structure of each NP, which is done automatically even if `metadata=False` in the extract call. However, the structure information will only be included in the results if `metadata=True`.

**Newline Handling**
The `newline_breaks` option determines whether newlines should be treated as sentence boundaries. When set to `True` (default), newlines are treated as sentence boundaries. When set to `False`, the text is treated as continuous, ignoring line breaks, which can be useful when processing text with irregular arbitrary line breaks (e.g., PDF extractions).

ANPE includes preprocessing to maximize compatibility with Benepar's tokenization requirements. However, it is *strongly recommended* that beforehand cleaning should be performed before processing.

**Model Selection for Usage**

When creating an `ANPEExtractor` instance or calling `anpe.extract`, ANPE determines which models to *use* based on this priority:

1. **Explicit Configuration (Highest Priority):** The model specified via the `spacy_model` or `benepar_model` configuration option (accepts aliases or full names).
2. **Default Model:** If no model is explicitly specified, the default (`en_core_web_md` for spaCy, `benepar_en3` for Benepar) is used if installed.
3. **Best Available Fallback:** If the default model isn't installed, ANPE attempts to load the best compatible model found in your environment (e.g., preferring larger or transformer models if available).
4. **Initialization Failure:** If no relevant model is specified and no suitable model can be auto-detected or loaded, extractor initialization will fail.

ANPE will log which models are being loaded at the INFO level.

### Convenient Method

For quick, one-off extractions, you may use the `anpe.extract()` function directly. This method is simpler and avoids the need to explicitly create an extractor instance.

> **Note:** While convenient for single calls, creating an `ANPEExtractor` instance (see Basic Usage) is recommended for processing multiple texts as models are loaded only once, improving performance.

Similarly, the `extract()` function accepts the following parameters:

- `text` (str): The input text to process.
- `metadata` (bool, optional): Whether to include metadata (length and structure analysis). Defaults to `False`.
- `include_nested` (bool, optional): Whether to include nested noun phrases. Defaults to `False`.
- `**kwargs`: Configuration options for the extractor (e.g., `min_length`, `max_length`, `accept_pronouns`, `log_level`, `spacy_model`).

```python
import anpe

# Extract noun phrases with custom configuration
result = anpe.extract(
    "In the summer of 1956, Stevens, a long-serving butler at Darlington Hall, decides to take a motoring trip through the West Country.",
    metadata=True,
    include_nested=True,
    min_length=2,
    max_length=5,
    accept_pronouns=False,
    spacy_model="lg"
)
print(result)
```

### Result Format

The `extract()` method returns a dictionary following this structure:

1. **`noun_phrase`**: The extracted noun phrase text
2. **`id`**: Hierarchical ID of the noun phrase
3. **`level`**: Depth level in the hierarchy
4. **`metadata`**: (*if requested*) Contains length and structures
5. **`children`**: (*if nested NPs are requested*) Always appears as the last field for readability

```python
{
    "metadata": {
        "timestamp": "2025-04-01 11:01:06",
        "includes_nested": true,
        "includes_metadata": true
    },
    "results": [
        #only demonstrate part of the result
        {
            "id": "2",
            "noun_phrase": "Stevens , a long-serving butler at Darlington Hall ,",
            "level": 1,
            "metadata": {
                "length": 9,
                "structures": [
                    "determiner",
                    "prepositional_modifier",
                    "compound",
                    "appositive"
                ]
            },
            "children": [
                {
                    "id": "2.1",
                    "noun_phrase": "Stevens",
                    "level": 2,
                    "metadata": {
                        "length": 1,
                        "structures": [
                            "standalone_noun"
                        ]
                    },
                    "children": []
                },
                {
                    "id": "2.2",
                    "noun_phrase": "a long-serving butler at Darlington Hall",
                    "level": 2,
                    "metadata": {
                        "length": 6,
                        "structures": [
                            "determiner",
                            "prepositional_modifier",
                            "compound"
                        ]
                    },
                    "children": []
                }
            ]
        }
    ]
}
```

> 📌 **Note on ID:**
> Please refer to [Hierarchical ID System](#hierarchical-id-system) for more details.

### Exporting Results

ANPE provides a quick method to extract NP and export the results of an extraction directly to a file in one go.

```python
# Export to JSON (providing a directory - timestamped filename will be generated)
extractor.export(text, format="json", output="/dir/to/exports", metadata=True, include_nested=True)

# Export to CSV (providing a specific file path - respects the path)
extractor.export(text, format="csv", output="/dir/to/exports/my_results.csv", metadata=True)

# Export to TXT (using default output - current directory, timestamped filename)
extractor.export(text, format="txt")
```

The `export()` method accepts the same parameters as `extract()` plus:

| Parameter  | Type          | Default | Description                                                                                                                      |
| ---------- | ------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `format` | str           | "txt"   | Output format ("txt", "csv", or "json")                                                                                          |
| `output` | Optional[str] | None    | Path to the output file or directory. If a directory, a timestamped file is created. If None, defaults to the current directory. |

> 📌 **Note on Output Path:** If you provide a full file path to `output` (e.g., `output='results/my_file.json'`), ANPE will use that exact path. If the file extension in the path (e.g., `.json`) doesn't match the specified `format` (e.g., `format='csv'`), ANPE will log a warning but still save the file using the provided path (`results/my_file.json`) with the content formatted according to the `format` parameter (`csv`).

**Convenient Method**
Similarly, ANPE provides a convenient method to extract NP and export files directly via `anpe.export()`. The usage is the same as `anpe.extract()` method, with the addition of the two aforementioned parameters.

> **Note:** Similar to `anpe.extract()`, if exporting results for multiple texts, using `extractor.export()` with a pre-created `ANPEExtractor` instance is more efficient.

```python
import anpe
# Export noun phrases to a text file in the specified directory
anpe.export(
    "In the summer of 1956, Stevens, a long-serving butler at Darlington Hall, decides to take a motoring trip through the West Country.",
    format="txt",
    output="./output", # Can be directory or file path
    metadata=True,
    include_nested=True,
    min_length=2,
    max_length=5,
    accept_pronouns=False,
    spacy_model="lg"
)
```

ANPE supports three output formats: **JSON, CSV, and TXT**. Each format provides different structure to present data.

#### JSON Format

The JSON output maintains a hierarchical structure:

```json
{
  "metadata": {
    "timestamp": "2025-04-01 11:01:06",
    "includes_nested": true,
    "includes_metadata": true
  },
  "results": [
    {
      "noun_phrase": "the summer of 1956",
      "id": "1",
      "level": 1,
      "metadata": {
        "length": 4,
        "structures": [
          "determiner",
          "prepositional_modifier"
        ]
      },
      "children": [
        {
          "noun_phrase": "the summer",
          "id": "1.1",
          "level": 2,
          "metadata": {
            "length": 2,
            "structures": [
              "determiner"
            ]
          },
          "children": []
        },
        {
          "noun_phrase": "1956",
          "id": "1.2",
          "level": 2,
          "metadata": {
            "length": 1,
            "structures": [
              "others"
            ]
          },
          "children": []
        }
      ]
    }
  ]
}
```

#### CSV Format

The CSV output provides a flat structure with parent-child relationships represented by additional columns:

```csv
ID,Level,Parent_ID,Noun_Phrase,Length,Structures
1,1,,the summer of 1956,4,determiner|prepositional_modifier
1.1,2,1,the summer,2,determiner
1.2,2,1,1956,1,others
2,1,,"Stevens , a long-serving butler at Darlington Hall ,",9,determiner|prepositional_modifier|compound|appositive
2.1,2,2,Stevens,1,standalone_noun
2.2,2,2,a long-serving butler at Darlington Hall,6,determiner|prepositional_modifier|compound
```

#### TXT Format

The TXT output is the most human-readable format and shows the hierarchical structure with indentation:

```
• [3] a motoring trip through the West Country
  Length: 7
  Structures: [determiner, prepositional_modifier, compound]
  ◦ [3.1] a motoring trip
    Length: 3
    Structures: [determiner, compound]
  ◦ [3.2] the West Country
    Length: 3
    Structures: [determiner, compound]

• [4] The six-day excursion
  Length: 3
  Structures: [determiner, compound, quantified]
```

> 💡We recommend use TXT if you are only intersted in top-level NPs and would like to see a plain list directly.

## Command-line Interface

ANPE provides a powerful command-line interface for text processing, providing easy access to all its features while introducing convenient methods such as batch processing and file input.

### Basic Syntax

```bash
anpe [command] [options]
```

### Available Commands

| Command     | Description                      | Example                                         |
| ----------- | -------------------------------- | ----------------------------------------------- |
| `extract` | Extract noun phrases from text   | `anpe extract "Sample text"`                  |
| `setup`   | Install or clean required models | `anpe setup` or `anpe setup --clean-models` |
| `version` | Display the ANPE version         | `anpe version`                                |

### Available Options

#### Setup Command Options

| Option                                     | Description                                                                                                                                            | Example                                |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------- |
| `--spacy-model <alias>`, `--spacy`     | Specify the spaCy model alias to*install* (`sm`, `md`, `lg`, `trf`) or `all` to install all models. If omitted, installs default (`md`). | `anpe setup --spacy lg`              |
| `--benepar-model <alias>`, `--benepar` | Specify the Benepar model alias to*install* (`default`, `large`) or `all` to install all models. If omitted, installs default (`default`).   | `anpe setup --benepar large`         |
| `--check-models`                         | Check and display current model installation status and which models would be auto-selected.                                                           | `anpe setup --check-models`          |
| `--clean-models`                         | Remove all known ANPE-related models (spaCy and Benepar).                                                                                              | `anpe setup --clean-models`          |
| `--clean-spacy <alias>`                  | Remove a specific spaCy model by alias (`sm`, `md`, `lg`, `trf`).                                                                              | `anpe setup --clean-spacy md`        |
| `--clean-benepar <alias>`                | Remove a specific Benepar model by alias (`default`, `large`).                                                                                     | `anpe setup --clean-benepar default` |
| `-f`, `--force`                        | Force removal without user confirmation when using any clean option.                                                                                   | `anpe setup --clean-models -f`       |
| `--log-level <level>`                    | Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Affects console/file output verbosity.                                                  | `anpe setup --log-level DEBUG`       |
| `--log-dir <path>`                       | Directory path for log files. If provided, logs are written to timestamped files instead of the console.                                               | `anpe setup --log-dir logs`          |

#### Input Options (for extract command)

| Option                | Description                             | Example                             |
| --------------------- | --------------------------------------- | ----------------------------------- |
| `text`              | Direct text input (positional argument) | `anpe extract "Sample text"`      |
| `-f, --file <path>` | Input file path                         | `anpe extract -f input.txt`       |
| `-d, --dir <path>`  | Input directory for batch processing    | `anpe extract -d input_directory` |

#### Processing Options (for extract command)

| Option                                    | Description                                                                                                                              | Example                                               |
| ----------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `--metadata`                            | Include metadata about each noun phrase (length and structural analysis)                                                                 | `anpe extract --metadata`                           |
| `--nested`                              | Extract nested noun phrases (maintains parent-child relationships)                                                                       | `anpe extract --nested`                             |
| `--min-length <int>`                    | Minimum NP length in tokens                                                                                                              | `anpe extract --min-length 2`                       |
| `--max-length <int>`                    | Maximum NP length in tokens                                                                                                              | `anpe extract --max-length 10`                      |
| `--no-pronouns`                         | Exclude pronouns from results                                                                                                            | `anpe extract --no-pronouns`                        |
| `--no-newline-breaks`                   | Don't treat newlines as sentence boundaries                                                                                              | `anpe extract --no-newline-breaks`                  |
| `--structures <list>`                   | Comma-separated list of structure patterns to include (e.g., "determiner,named_entity")                                                  | `anpe extract --structures "determiner,appositive"` |
| `--spacy-model <name>`, `--spacy`     | Specify spaCy model alias/name to*use* (e.g., "md", "en_core_web_lg"). Accepts aliases or full names. Overrides auto-detect.           | `anpe extract --spacy lg`                           |
| `--benepar-model <name>`, `--benepar` | Specify Benepar model alias/name to*use* (e.g., "default", "benepar_en3_large"). Accepts aliases or full names. Overrides auto-detect. | `anpe extract --benepar large`                      |

#### Output Options (for extract command)

| Option                  | Description                                                                                                           | Example                                                            |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `-o, --output <path>` | Output file path or directory. If a directory, timestamped files are created. If omitted, prints to console (stdout). | `anpe extract -o output_dir` or `anpe extract -o results.json` |
| `-t, --type <type>`   | Output format (txt, csv, json). Required if `-o` is used.                                                           | `anpe extract -o results.json -t json`                           |

#### Logging Options (for all commands)

| Option                  | Description                                                                                              | Example                            |
| ----------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `--log-level <level>` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Affects console/file output verbosity.            | `anpe extract --log-level DEBUG` |
| `--log-dir <path>`    | Directory path for log files. If provided, logs are written to timestamped files instead of the console. | `anpe extract --log-dir ./logs`  |

### Example Commands

**Setup models with logging:**

```bash
anpe setup --log-level DEBUG --log-dir logs
```

**Clean existing models (with confirmation):**

```bash
anpe setup --clean-models
```

**Clean existing models (without confirmation):**

```bash
anpe setup --clean-models -f
```

**Extract from file and output to JSON in a directory:**

```bash
anpe extract -f input.txt -o output_dir -t json
```

**Batch processing (Outputting to a directory):**

```bash
anpe extract -d input_directory --output output_directory -t json --metadata
```

**Advanced extraction with filters (Outputting to a specific CSV file):**

```bash
anpe extract -f input.txt --min-length 2 --max-length 10 --no-pronouns --structures "determiner,appositive" -o results.csv -t csv
```

**Extract from file with logging to file:**

```bash
anpe extract -f input.txt --log-dir ./logs --log-level DEBUG
```

**Check version:**

```bash
anpe version
```

## Hierarchical ID System

ANPE uses a hierarchical ID system to represent parent-child relationships between noun phrases when nested NP are captured:

- **Top-level NPs** are assigned sequential numeric IDs: "1", "2", "3", etc.
- **Child NPs** are assigned IDs that reflect their parent: "1.1", "1.2", "2.1", etc.
- **Deeper nested NPs** continue this pattern: "1.1.1", "1.1.2", etc.

This makes it easy to identify related noun phrases across different output formats.

## Structural Analysis

ANPE's structural labeling system analyzes noun phrases to identify their syntactic patterns. This is achieved through:

1. **Constituency Parsing**: Using the Berkeley Neural Parser to identify phrase structures
2. **Pattern Matching**: Applying rules based on spaCy dependency parsing to detect specific syntactic constructions within the identified NPs.

When using the `structure_filters` configuration option, use the identifier listed in the `Config Key` column below to target specific NP types.

| Type                              | Config Key                  | Description                                                                                         | Example                                                  |
| --------------------------------- | --------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Pronoun**                 | `pronoun`                 | Single pronoun (if `accept_pronouns` is True)                                                     | "it", "they"                                             |
| **Standalone Noun**         | `standalone_noun`         | Single common or proper noun                                                                        | "Stevens", "butler"                                      |
| **Determiner**              | `determiner`              | Contains determiners (the, a, an, this, etc.)                                                       | "the summer"                                             |
| **Adjectival Modifier**     | `adjectival_modifier`     | Contains adjective modifiers (or verbs acting as adjectives)                                        | "unrealised love", "intricately carved altars"           |
| **Prepositional Modifier**  | `prepositional_modifier`  | Contains prepositional phrase modifiers                                                             | "butler*at Darlington Hall*"                           |
| **Compound**                | `compound`                | Contains compound nouns forming a single conceptual unit                                            | "Darlington Hall"                                        |
| **Possessive**              | `possessive`              | Contains possessive constructions ('s marker or possessive pronouns)                                | "his housekeeper", "farmer's plot"                       |
| **Quantified**              | `quantified`              | Contains numeric quantifiers modifying a noun                                                       | "two world wars"                                         |
| **Coordinated**             | `coordinated`             | Contains coordinated elements joined by conjunctions (within the NP)                                | "Stevens and England"                                    |
| **Appositive**              | `appositive`              | Contains one NP renames or explains another                                                         | "Stevens,*a long-serving butler*"                      |
| **Relative Clause**         | `relative_clause`         | Contains a clause modifying a noun, typically introduced by a relative pronoun (who, which, that)   | "a past*that takes in fascism*"                        |
| **Reduced Relative Clause** | `reduced_relative_clause` | Contains a clause modifying a noun where the relative pronoun is omitted (often using a participle) | "a tapestry*woven with simple joys*"                   |
| **Finite Complement**       | `finite_complement`       | Contains a finite clause acting as a complement to specific types of nouns (fact, idea, etc.)       | "the idea*that he might leave*"                        |
| **Nonfinite Complement**    | `nonfinite_complement`    | Contains a nonfinite clause (infinitive or gerund phrase) acting as a complement to a noun          | "a plan*to succeed*", "the possibility *of leaving*" |
| **others**                  | `others`                  | Other valid NP structures not matching specific patterns                                            | (Various complex or simple NPs)                          |

For a comprehensive explanation of all structure patterns and their detection logic, please refer to the [structure_patterns.md](docs/structure_patterns.md).

## GUI Application

> *❗[Under development]*
>
> Please note that the gui app is now still being developed, no release is provided at the moment.

》***"Oh no, code again! I just want a quick tool, kill me already!😵"***

No worries, ANPE provides a graphical user interface (GUI) for easier interaction with the library. Best part of all - it is a standalone app and requires no environment setup. Supports Mac and Windows.

![ANPE GUI Screenshot](/pics/anpe_gui_app_windows.png)

### GUI Features

- **User-friendly interface** with distinct Input and Output tabs.
- **Input Modes**: Process text via Direct Text Input or File Input.
- **File Handling**: Add single files or entire directories; view and manage the list.
- **Batch Processing**: Automatically handles multiple files from selected directories.
- **Visual Configuration**: Easily configure all ANPE settings:
  - General: Include Nested Phrases, Include Metadata, Treat Newlines as Boundaries.
  - Filtering: Min/Max NP length, Accept Pronouns.
  - Structure Filtering: Master toggle switch and individual selection for specific NP structures (Determiner, Compound, Relative Clause, etc.).
- **Real-time Log Viewer**: Track operations and potential issues with log level filtering.
- **Results Visualization**: View formatted extraction results in the Output tab.
- **Export Options**: Export results to TXT, CSV, or JSON formats to a selected directory.

For more details on the GUI structure or building it from source, see the ANPE GUI repo.

## Contributing

Contributions are welcome! Here are some ways you can contribute:

1. **Report bugs**: Submit issues for any bugs you find
2. **Suggest features**: Submit issues for feature requests
3. **Submit pull requests**: Implement new features or fix bugs

### Testing

ANPE uses `pytest` for testing. The test suite (`tests`) includes unit tests, integration tests, and feature tests designed to verify the functionality of the package robustly.

#### Running Tests

To run the tests, first install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Then, you can run the tests from the project root directory with:

```bash
pytest tests
```

You can also run specific test files or use `pytest` markers and keywords (`-k`) to target tests.

#### Test Structure (`tests`)

The test suite is organized to separate different testing levels:

- **`unit/`**: Contains unit tests focusing on isolated components, like specific functions in `extractor.py`, `analyzer.py`, or `export.py`. These typically use mocking extensively.
- **`integration/`**: Contains integration tests checking the interaction between components, primarily focusing on the Command-Line Interface (`test_cli.py`). These tests mock external dependencies like file system operations or model downloads but test the CLI argument parsing, logging setup, and function calls.
- **`feature/`**: Contains feature tests (also known as end-to-end tests) that verify complete user workflows.
  - `test_feature_cli.py`: Tests the CLI commands (`extract`, `setup`, `clean`) by invoking the CLI entry point, mocking external actions (like actual downloads or file writes where necessary), and asserting expected outcomes or mock calls.
  - `test_feature_extractor.py`: Tests the `ANPEExtractor` API by creating instances and calling `extract` or `export` with various configurations on sample texts, asserting the correctness of the output structure and content.

## **Troubleshooting**

If you encounter issues with model setup, cleanup, or extraction:

1. **Check the Basics**: Ensure you have an active internet connection (for downloading models) and sufficient disk space (models can be large).
2. **Run with Detailed Logging**: Execute the command (e.g., `anpe setup` or `anpe extract`) with debug logging enabled using CLI arguments. Use `--log-dir` to save logs to a file for easier review:

   ```bash
   anpe extract "Some text" --log-level DEBUG --log-dir ./logs
   # or for setup:
   anpe setup --log-level DEBUG --log-dir ./logs
   ```

   Carefully examine the console output and the generated log file in the `logs` directory for specific error messages from ANPE, spaCy, or Benepar.
3. **Check File Permissions**: ANPE needs write access to install models. Ensure your user has permission to write to:

   * Your Python environment's `site-packages` directory (for spaCy models, typically handled by `pip`/`spacy download`).
   * The `~/nltk_data/models` directory (for Benepar models, NLTK attempts to create `~/nltk_data` if it doesn't exist).
     Permission issues can prevent downloading or cleanup. Running `anpe setup --clean-models` can also fail if files are locked or permissions are insufficient.
4. **Perform a Full Cleanup**: If you suspect model corruption or inconsistent state, run the cleanup command. Use `--force` (or `-f`) to skip confirmation if needed:

   ```bash
   anpe setup --clean-models --force
   ```

   Check the console output of *this* command for any errors related to file removal (e.g., permission denied). After a successful cleanup, try running `anpe setup` again.
5. **Transformer Model Issues**: If using a spaCy transformer model (i.e., alias `trf`), the setup attempts to install `spacy-transformers`. Ensure this dependency installed correctly. Transformer models also rely on underlying ML frameworks (like PyTorch or TensorFlow). Installation issues might relate to those frameworks rather than ANPE itself. Check the spaCy documentation for transformer setup.
6. **Manual Verification**: If automatic setup fails, you can manually check if the models exist in their expected locations:

   * **spaCy**: Look for model directories (e.g., `en_core_web_md`) within your Python environment's `site-packages` directory. Use `python -m spacy validate` to check installed models.
   * **Benepar**: Check for model directories (e.g., `benepar_en3`) inside `~/nltk_data/models/`.
7. **Conflicting Installations**: Ensure you don't have conflicting versions of spaCy, Benepar, NLTK, or their dependencies. Consider using a virtual environment.
8. **Refer to External Documentation**: For issues potentially related to the underlying libraries, consult their documentation:

   * [spaCy Documentation &amp; Troubleshooting](https://spacy.io/usage/troubleshooting)
   * [Benepar Issues](https://github.com/nikitakit/self-attentive-parser/issues)
   * [NLTK Data Issues](https://www.nltk.org/data.html)
9. **Report an Issue**: If the problem persists after trying these steps, please [open an issue](https://github.com/rcverse/another-noun-phrase-extractor/issues) on the GitHub repository, including:

   * Your OS and Python version.
   * The ANPE version (`anpe version`).
   * The exact command you ran.
   * The full console output and relevant logs (from step 2 or 4).

## Future Development Plans

ANPE is under active development with several features being considered for future releases. This roadmap is tentative and may change.

### 🗺️ Feature Roadmap

- [ ] **Multilingual Support**
  - [ ] Evaluate and integrate with multilingual versions of spaCy and Benepar
  - [ ] Implement language-specific structural pattern detection
  - [ ] Support for language-specific grammatical constructions

- [ ] **Enhanced Structural Analysis**
  - [ ] Add more granular structural labels for NP categorization
  - [ ] Improve detection accuracy for complex syntactic patterns
  - [ ] Support for specialized linguistic constructions (e.g., cleft sentences, extraposition)

- [ ] **Named Entity Integration**
  - [ ] Better integration with spaCy's named entity recognition
  - [ ] Special handling and labeling of named entities within NPs
  - [ ] Entity-aware filtering options

- [ ] **Custom Pattern Definitions**
  - [ ] Framework for user-defined structural patterns
  - [ ] Support for domain-specific syntactic constructions
  - [ ] Extension mechanism for custom functionality

### 💡 Contributions Welcome!

We welcome contributions of all kinds to help shape the future of ANPE:

- **Feature Suggestions**: Have an idea for a feature that would make ANPE more useful? Open an issue to discuss it!
- **Real-World Use Cases**: Sharing how you're using ANPE in real projects is especially valuable for prioritizing features
- **Code Contributions**: Pull requests for bug fixes or new features are always appreciated
- **Documentation**: Improvements to documentation, examples, or tutorials help make ANPE more accessible

If you're interested in contributing, please check the [Contributing](#contributing) section or open an issue to start a conversation about your ideas. The most valuable input often comes from users with practical applications and specific needs.

## Citation

I spent a lot of time on this project. If you use ANPE in your research or projects, please cite it as follows:

### BibTeX

```bibtex
@software{Chen_ANPE_2025,
  author = {Chen, Nuo},
  title = {{ANPE: Another Noun Phrase Extractor}},
  url = {https://github.com/rcverse/another-noun-phrase-extractor},
  version = {1.1.0},
  year = {2025}
}
```

### Plain Text (APA style)

Chen, N. (2025). *ANPE: Another Noun Phrase Extractor* (Version 1.1.0) [Computer software]. Retrieved from https://github.com/rcverse/another-noun-phrase-extractor

## Acknowledgements

ANPE builds upon several powerful open-source NLP libraries.

- **spaCy:** For industrial-strength natural language processing in Python. ([https://spacy.io/](https://spacy.io/))
- **Benepar:** For high-accuracy constituency parsing. ([https://github.com/nikitakit/self-attentive-parser](https://github.com/nikitakit/self-attentive-parser))
- **NLTK:** The Natural Language Toolkit. ([https://www.nltk.org/](https://www.nltk.org/))

Please refer to their respective websites and documentation for more information and their own citation guidelines if you are using these components directly or wish to cite their specific contributions.