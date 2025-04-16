# ANPE: Another Noun Phrase Extractor

![ANPE Banner](/pics/banner.png)

[![Build Status](https://github.com/rcverse/anpe/actions/workflows/python-package.yml/badge.svg)](https://github.com/rcverse/anpe/actions/workflows/python-package.yml)
[![pytest](https://img.shields.io/badge/pytest-passing-brightgreen)](https://github.com/rcverse/anpe/actions/workflows/python-package.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/anpe)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ANPE (*Another Noun Phrase Extractor*) is a Python library for **directly extracting complete noun phrases from text**. This library leverages the [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser) with [spaCy](https://spacy.io/) and [NLTK](https://www.nltk.org/) for precise parsing and NP extraction. On top of that, the library provides flexible configuration options to **include nested NP**, **filter specific structural types of NP**, or **taget length requirements**, as well as options to **export to files** in multiple structured formats directly. 

Currently, ANPE is only tested on **English** and compatible with Python through **3.9** to **3.12**.

## **Key Features**:
1. **✅Precision Extraction**: Accurate noun phrase identification using modern parsing techniques
2. **🏷️Structural Labelling**: Identifies and labels NPs with their different syntactic patterns
3. **✍🏻Hierarchical Analysis**: Supports both top-level and nested noun phrases
4. **📄Flexible Output**: Multiple formats (TXT, CSV, JSON) with consistent structure
5. **⚙️Customizable Processing**: Flexible configuration options for filtering and analysis
6. **⌨️CLI Integration**: Command-line interface for easy text processing

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [GUI Application](#gui-application)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)

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

text = "Sample texts"
result = anpe.extract(text)

print(result)
```

### GUI App
[To be released from Dependency HELL]

## Installation

### Using pip

```bash
pip install anpe
```

### From source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/anpe.git
   cd anpe
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Prerequisites

#### **Required Models**
ANPE relies on several pre-trained models for its functionality. The default setup uses the following:

1.  **spaCy Model**: `en_core_web_md` (English language model for tokenization and sentence segmentation).
2.  **Benepar Model**: `benepar_en3` (English constituency parser for syntactic analysis).
3.  **NLTK Models**:
    *   `punkt` (Punkt tokenizer for sentence splitting).
    *   `punkt_tab` (Language-specific tab-delimited tokenizer data required by Benepar).

ANPE also supports using alternative spaCy models (`en_core_web_sm`, `en_core_web_lg`, `en_core_web_trf`) and a larger Benepar model (`benepar_en3_large`) for different performance/accuracy trade-offs. These can be selected and managed via the GUI application or potentially future updates to the CLI/library API.

#### **Automatic Setup**

ANPE provides a built-in tool to setup the necessary **default** models (`en_core_web_md` and `benepar_en3`, plus NLTK data). When you run the extractor, the package will automatically check if the default models are installed and install them if they're not. However, it is **recommended** to run the setup utility before you start using the extractor for the first time.
To setup the **default** models, simply run the following command in terminal (Please refer to [CLI usage](#command-line-interface) for more options.):
```bash
anpe setup
```

Alteratively, you can run the script with:
```bash
python -m anpe.utils.setup_models
```

#### **Manual Setup**
If automatic setup fails or you prefer to manually download the models, you can run install the models manually. Below are examples for the **default** models:

Install spaCy English Model:
```bash
# Default model
python -m spacy download en_core_web_md
# Other options: en_core_web_sm, en_core_web_lg, en_core_web_trf
```

Install Benepar Parser Model:
```bash
# Default model
python -m benepar.download benepar_en3
# Other option: benepar_en3_large
```

Install NLTK Punkt Tokenizer (via python console):
``` python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
```

## Library API Usage

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
    "log_level": "DEBUG",
    "log_dir": "dir/to/your/log",
    "newline_breaks": False,
    "spacy_model": "lg",
    "benepar_model": "default"
})

# Sample text
text = """
In the summer of 1956, Stevens, a long-serving butler at Darlington Hall, decides to take a motoring trip through the West Country.
"""

# Extract with metadata and nested NPs
result = extractor.extract(text, metadata=True, include_nested=True)

# Process results and print
print(f"Found {len(result['results'])} top-level noun phrases:")
for np in result['results']:
    print(f"• {np['noun_phrase']}")
    if 'metadata' in np:
        print(f"  Length: {np['metadata']['length']}")
        print(f"  Structures: {', '.join(np['metadata']['structures'])}")
    if 'children' in np:
        print(f"  Contains {len(np['children'])} nested noun phrases")
```
To achieve this, you need to customize the extraction parameters and configuration.

### Extraction Parameters
The `extract()` method accepts the following parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | Required | Input text to process |
| `metadata` | bool | False | Whether to include metadata (`length` and `structures`) |
| `include_nested` | bool | False | Whether to include nested noun phrases |

- **Metadata**: When set to `True`, the output will include two types of additional information about each noun phrase: `length` and `structures'
  - **`length`** is the number of words that the NP contains
  - **`structures`** is the syntactic structure that the NP contains, such as `appositive`, `coordinated`, `nonfinite_complement`, etc. 

- **Include Nested**: When set to `True`, the output will include nested noun phrases, allowing for a hierarchical representation of noun phrases.

> **📌 Note on Metadata:**
> **Structural analysis** is performed using the analyzer tool built into ANPE. It analyzes the NP's structure and label the NP with the structures it detected. Please refer to the [Structural Analysis](#structural-analysis) section for more details.

### Configuration Options

ANPE provides a flexible configuration system to further customize the extraction process. These options can be passed as a dictionary when initializing the extractor.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `min_length` | Integer | `None` | Minimum token length for NPs. NPs with fewer tokens will be excluded. |
| `max_length` | Integer | `None` | Maximum token length for NPs. NPs with more tokens will be excluded. |
| `accept_pronouns` | Boolean | `True` | Whether to include single-word pronouns as valid NPs. When set to `False`, NPs that consist of a single pronoun will be excluded. |
| `structure_filters` | List[str] | `[]` | List of structure types to include. Only NPs containing at least one of these structures will be included. If empty, all NPs are accepted. |
| `log_level` | String | `"INFO"` | Logging level. Options: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`. |
| `log_dir` | Optional[str] | `None` | Directory to store log files. If None, logs will be printed to console. |
| `newline_breaks` | Boolean | `True` | Whether to treat newlines as sentence boundaries. This can be helpful if you are processing text resources with inconsistent line breaking. |
| `spacy_model` | Optional[str] | `None` | Specify the spaCy model alias/name to use. Accepts aliases (`"sm"`, `"md"`, `"lg"`, `"trf"`) or full names. If `None`, ANPE attempts to auto-detect the best installed model. |
| `benepar_model` | Optional[str] | `None` | Specify the Benepar model alias/name to use (e.g., "large", "benepar_en3"). Accepts aliases (`"default"`, `large"`) or full names. If `None`, ANPE attempts to auto-detect the best installed model. |


Example:

```python
# Configure the extractor with multiple options
custom_extractor = ANPEExtractor({
    "min_length": 2,                # Only NPs with 2+ words
    "max_length": 5,                # Only NPs with 5 or fewer words
    "accept_pronouns": False,       # Exclude single-word pronouns
    "structure_filters": ["determiner"],  # Only include NPs with these structures
    "log_level": "DEBUG",           # Detailed logging
    "log_dir": "dir/to/your/log",    # Enable log file by providing a dir to save the file
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

**Logging Control**  
The `log_level` option controls the verbosity of the extraction process. Use `DEBUG` for detailed logging during development or troubleshooting, and `ERROR` for production environments where you only want to see critical issues. 
The `log_dir` option controls whether to output log into a file. If provided with a directory, ANPE will output the log into a log file stored in the designated directory. If None, ANPE will by default output log into console.

**Newline Handling**  
The `newline_breaks` option determines whether newlines should be treated as sentence boundaries. When set to `True` (default), newlines are treated as sentence boundaries. You may want to disable this option if you want to treat the text as a continuous paragraph, ignoring line breaks, which can be useful when processing text with irregular formatting.


**Model Selection**

ANPE aims for a balance between ease of use and flexibility when loading spaCy and Benepar models:

1.  **Explicit Configuration (Highest Priority):** If you provide the `spacy_model` or `benepar_model` keys in the configuration dictionary when creating `ANPEExtractor` (or via the corresponding CLI flags for the `extract` command), ANPE will use the specified model(s).
2.  **Automatic Detection:** If a model is *not* explicitly specified in the configuration, ANPE will attempt to find all relevant installed models (e.g., `en_core_web_sm`, `en_core_web_md`, `en_core_web_lg` for spaCy). It will select the model based on the following priority:
    *   **Default Model Priority:** If the default model (`en_core_web_md` for spaCy or `benepar_en3` for Benepar) is installed, it will be selected.
    *   **Preference List Fallback:** If the default model is *not* installed, ANPE will use a predefined preference order (e.g., `trf` > `lg` > `sm` for spaCy; `large` for Benepar) and select the highest-priority model from that list that *is* installed.
3.  **Initialization Failure:** If no relevant model is specified and no suitable model can be auto-detected or loaded (e.g., none are installed, not even the defaults), the extractor initialization will fail.

ANPE will log which models are being loaded at the INFO level.

### Convenient Method
For quick, one-off extractions, you may use the `anpe.extract()` function directly. This method is simpler and avoids the need to explicitly create an extractor instance. 
Similarly, the `extract()` function accepts the following parameters:
- `text` (str): The input text to process.
- `metadata` (bool, optional): Whether to include metadata (length and structure analysis). Defaults to `False`.
- `include_nested` (bool, optional): Whether to include nested noun phrases. Defaults to `False`.
- `**kwargs`: Configuration options for the extractor (e.g., `min_length`, `max_length`, `accept_pronouns`, `log_level`).

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
    log_level="DEBUG"
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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | str | "txt" | Output format ("txt", "csv", or "json") |
| `output` | Optional[str] | None | Path to the output file or directory. If a directory, a timestamped file is created. If None, defaults to the current directory. |

> 📌 **Note on Output Path:** If you provide a full file path to `output` (e.g., `output='results/my_file.json'`), ANPE will use that exact path. If the file extension in the path (e.g., `.json`) doesn't match the specified `format` (e.g., `format='csv'`), ANPE will log a warning but still save the file using the provided path (`results/my_file.json`) with the content formatted according to the `format` parameter (`csv`).


**Convenient Method**
Similarly, ANPE provides a convenient method to extract NP and export files directly via `anpe.export()`. The usage is the same as `anpe.extract()` method, with the addition of the two aforementioned parameters.
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
    log_level="DEBUG"
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

| Command | Description | Example |
|---------|-------------|---------|
| `extract` | Extract noun phrases from text | `anpe extract "Sample text"` |
| `setup` | Install required models | `anpe setup` |
| `version` | Display the ANPE version | `anpe version` |

### Available Options

#### Setup Command Options

| Option | Description | Example |
|--------|-------------|---------|
| `--spacy-model` | Specify the spaCy model alias (sm, md, lg, trf). Defaults to `md`. | `anpe setup --spacy-model lg` |
| `--benepar-model` | Specify the Benepar model alias (default, large). Defaults to `default`. | `anpe setup --benepar-model large` |
| `--log-level` | Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `anpe setup --log-level DEBUG` |
| `--log-dir` | Directory path for log files | `anpe setup --log-dir logs` |

#### Input Options (for extract command)

| Option | Description | Example |
|--------|-------------|---------|
| `text` | Direct text input (positional argument) | `anpe extract "Sample text"` |
| `-f, --file` | Input file path | `anpe extract -f input.txt` |
| `-d, --dir` | Input directory for batch processing | `anpe extract -d input_directory` |

#### Processing Options (for extract command)

| Option | Description | Example |
|--------|-------------|---------|
| `--metadata` | Include metadata about each noun phrase (length and structural analysis) | `anpe extract --metadata` |
| `--nested` | Extract nested noun phrases (maintains parent-child relationships) | `anpe extract --nested` |
| `--min-length` | Minimum NP length in tokens | `anpe extract --min-length 2` |
| `--max-length` | Maximum NP length in tokens | `anpe extract --max-length 10` |
| `--no-pronouns` | Exclude pronouns from results | `anpe extract --no-pronouns` |
| `--no-newline-breaks` | Don't treat newlines as sentence boundaries | `anpe extract --no-newline-breaks` |
| `--structures` | Comma-separated list of structure patterns to include | `anpe extract --structures "determiner,named_entity"` |
| `--spacy-model` | Specify spaCy model alias/name to USE (e.g., "md", "en_core_web_lg"). Accepts aliases or full names. Overrides auto-detect. | `anpe extract --spacy-model lg` |
| `--benepar-model` | Specify Benepar model alias/name to USE (e.g., "default", "benepar_en3_large"). Accepts aliases or full names. Overrides auto-detect. | `