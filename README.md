# ANPE: Another Noun Phrase Extractor

![ANPE Banner](/pics/banner.png)

[![Build Status](https://github.com/richard20000321/anpe/actions/workflows/python-package.yml/badge.svg)](https://github.com/richard20000321/anpe/actions/workflows/python-package.yml)
[![pytest](https://img.shields.io/badge/pytest-passing-brightgreen)](https://github.com/richard20000321/anpe/actions/workflows/python-package.yml)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Python Version](https://img.shields.io/badge/python-<=3.12-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

ANPE (*Another Noun Phrase Extractor*) is a Python library for **directly extracting complete noun phrases from text**. This library leverages the [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser) with [spaCy](https://spacy.io/) and [NLTK](https://www.nltk.org/) for precise parsing and NP extraction. On top of that, the library provides flexible configuration options to **include nested NP**, **filter specific structural types of NP**, or **taget length requirements**, as well as options to **export to files** in multiple structured formats directly. 

Currently, ANPE is only tested on **English**.

## **Key Features**:
1. **✅Precision Extraction**: Accurate noun phrase identification using modern parsing techniques
2. **🏷️Structural Labelling**: Identifies and labels NPs with their different syntactic patterns
3. **✍🏻Hierarchical Analysis**: Supports both top-level and nested noun phrases
4. **📄Flexible Output**: Multiple formats (TXT, CSV, JSON) with consistent structure
5. **⚙️Customizable Processing**: Flexible configuration options for filtering and analysis
6. **⌨️CLI Integration**: Command-line interface for easy text processing

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)

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

#### **Python Version Compatibility**
ANPE is **compatible with Python 3.9 through 3.12**. It is not compatible with Python versions below 3.9 or higher than 3.12 due to dependencies requirements. Make sure you have a supported Python version installed before proceeding.

#### **Required Models**
ANPE relies on several pre-trained models for its functionality.

1. **spaCy Model**: `en_core_web_sm` (English language model for tokenization and sentence segmentation).
2. **Benepar Model**: `benepar_en3` (English constituency parser for syntactic analysis).
3. **NLTK Models**:
   - `punkt` (Punkt tokenizer for sentence splitting).
   - `punkt_tab` (Language-specific tab-delimited tokenizer data required by Benepar).

> **Note on Benepar and NLTK**: Benepar internally uses NLTK's parsing functions, which require both the `punkt` tokenizer and the `punkt_tab` language-specific data. The `punkt_tab` model is not a standard NLTK download but is automatically handled by ANPE's setup utility.

#### **Automatic Setup**

ANPE provides multiple ways to install the required models:

##### Recommended: Using the CLI
```bash
anpe setup
```

You can also specify logging options:
```bash
anpe setup --log-level DEBUG --log-dir logs
```

The setup process supports the following logging options:
- `--log-level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--log-dir`: Specify a directory for log files

Example with logging:
```bash
anpe setup --log-level DEBUG --log-dir logs
```

This will create detailed logs in the `logs/` directory.

##### Alternative: Running as a Script
```bash
python -m anpe.utils.setup_models
```

Or from within your Python code:
```python
from anpe.utils.setup_models import setup_models
setup_models()
```

This utility handles the installation of all required models.
> **Note**: When you run the extractor, the package will automatically check if the models are installed and install them if they're not. However, it is recommended to run the setup utility before you start using the extractor for the first time.


#### **Manual Setup**
If automatic setup fails or you prefer to manually download the models, you can run install the three models manually:

Install spaCy English Model:
```bash
python -m spacy download en_core_web_sm
```

Install Benepar Parser Model:
```bash
python -m benepar.download benepar_en3
```

Install NLTK Punkt Tokenizer:
``` python
import nltk
nltk.download('punkt')
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

# Configure extractor with custom settings
extractor = ANPEExtractor({
    "min_length": 2,
    "max_length": 5,
    "accept_pronouns": False,
    "structure_filters": ["compound", "appositive"]
})

# Sample text
text = """
In the summer of 1956, Stevens, a long-serving butler at Darlington Hall, decides to take a motoring trip through the West Country.
"""

# Extract with metadata and nested NPs
result = extractor.extract(text, metadata=True, include_nested=True)

# Process results
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
  - **`structures`** is the syntactic structure that the NP contains, such as `appositive`, `coodinated`, `nonfinite_complement`, etc. 

- **Include Nested**: When set to `True`, the output will include nested noun phrases, allowing for a hierarchical representation of noun phrases.

> **📌 Note on Metadata:**
> **Structural analysis** is performed using the analyzer tool built into ANPE. It analyzes the NP's structure and label the NP with the structures it detected. Please refer to the [Structural Analysis](#structural-analysis) section for more details.

### Configuration Options

ANPE provides a flexible configuration system to customize the extraction process. These options can be passed as a dictionary when initializing the extractor.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `min_length` | Integer | `None` | Minimum token length for NPs. NPs with fewer tokens will be excluded. |
| `max_length` | Integer | `None` | Maximum token length for NPs. NPs with more tokens will be excluded. |
| `accept_pronouns` | Boolean | `True` | Whether to include single-word pronouns as valid NPs. When set to `False`, NPs that consist of a single pronoun will be excluded. |
| `structure_filters` | List[str] | `[]` | List of structure types to include. Only NPs containing at least one of these structures will be included. If empty, all NPs are accepted. |
| `log_level` | String | `"INFO"` | Logging level. Options: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`. |
| `log_dir` | Optional[str] | `None` | Directory to store log files. If None, logs will be printed to console. |
| `newline_breaks` | Boolean | `True` | Whether to treat newlines as sentence boundaries. This can be helpful if you are processing text resources with inconsistent line breaking. |


Example:

```python
# Configure the extractor with multiple options
custom_extractor = ANPEExtractor({
    "min_length": 2,                # Only NPs with 2+ words
    "max_length": 5,                # Only NPs with 5 or fewer words
    "accept_pronouns": False,       # Exclude single-word pronouns
    "structure_filters": ["determiner"],  # Only include NPs with these structures
    "log_level": "DEBUG",           # Detailed logging
    "log_dir": "dir/to/your/log"    # Enable log file by providing a dir to save the file
    "newline_breaks": False         # Don't treat newlines as sentence boundaries
})
```
**Minimum Length Filtering**  
The `min_length` option allows you to filter out shorter noun phrases that might not be meaningful for your analysis. For example, setting `min_length=2` will exclude single-word noun phrases. This is particularly useful when you're only interested in more substantial noun phrases that provide richer semantic content.

**Maximum Length Filtering**  
The `max_length` option lets you limit the length of extracted noun phrases. This is helpful when dealing with long, complex noun phrases that might not be relevant to your analysis. For instance, setting `max_length=5` will exclude noun phrases with more than five words, focusing on more concise expressions.

**Pronoun Handling**  
The `accept_pronouns` option controls whether pronouns like "it", "they", or "this" should be considered as valid noun phrases. When set to `False`, single-word pronouns will be excluded from the results. This is useful when you're only interested in more descriptive noun phrases rather than referential pronouns.

**Structure Filtering**  
Structure filtering allows you to target specific types of noun phrases in your extraction. You can specify a list of structure types to include in the results. When using `structure_filters`, only noun phrases that contain at least one of the specified structures will be included. This allows for targeted extraction of specific NP types.
*(Please refer to the [Structural Analysis](#structural-analysis) section for more details.)*

> 📌 **Note on Structure Filtering:**
> Note that structure filtering requires analyzing the structure of each NP, which is done automatically even if `metadata=False` in the extract call. However, the structure information will only be included in the results if `metadata=True`.


**Logging Control**  
The `log_level` option controls the verbosity of the extraction process. Use `DEBUG` for detailed logging during development or troubleshooting, and `ERROR` for production environments where you only want to see critical issues. This helps in monitoring the extraction process and diagnosing any potential problems. The `log_dir` option controls whether to output log into a file. If provided with a directory, ANPE will output the log into a log file stored in the designated directory. If None, ANPE will by default output log into console.

**Newline Handling**  
The `newline_breaks` option determines whether newlines should be treated as sentence boundaries. When enabled, newlines are treated as sentence boundaries, which can be useful when processing text with irregular formatting. Disable this option if you want to treat the text as a continuous paragraph, ignoring line breaks.

### Convenient Method
For quick, one-off extractions, you may use the `extract()` function directly. This method is simpler and avoids the need to explicitly create an extractor instance. 
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

1. **noun_phrase**: The extracted noun phrase text
2. **id**: Hierarchical ID of the noun phrase
3. **level**: Depth level in the hierarchy
4. **metadata**: (*if requested*) Contains length and structures
5. **children**: (*if nested NPs are requested*) Always appears as the last field for readability

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

ANPE provides a convenient method to extract NP and export the results of an extraction directly to a file in one go. 

```python
# Export to JSON
extractor.export(text, format="json", export_dir="/path/to/exports", metadata=True, include_nested=True)

# Export to CSV
extractor.export(text, format="csv", export_dir="/path/to/exports" metadata=True)

# Export to TXT
extractor.export(text, format="txt", export_dir="/path/to/exports")
```

The `export()` method accepts the same parameters as `extract()` plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | str | "txt" | Output format ("txt", "csv", or "json") |
| `export_dir` | str | None | Path to save the output file (if None, returns the content as string)

Similarly, ANPE provides a convenient method to export directly via `export()`. The usage is the same as `extract()` function, with the addition of the two aforementioned parameters.
```python
import anpe
# Export noun phrases to a text file
anpe.export(
    "In the summer of 1956, Stevens, a long-serving butler at Darlington Hall, decides to take a motoring trip through the West Country.",
    format="txt",
    export_dir="./output",
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

The TXT output is human-readable and shows the hierarchical structure with indentation:

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

> We recommend use TXT if you are only intersted in top-level NPs and would like to see a plain list directly.

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

#### Output Options (for extract command)

| Option | Description | Example |
|--------|-------------|---------|
| `-o, --output-dir` | Output directory for results | `anpe extract -o output_dir` |
| `-t, --type` | Output format (txt, csv, json) | `anpe extract -t json` |

#### Logging Options (for all commands)

| Option | Description | Example |
|--------|-------------|---------|
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `anpe extract --log-level DEBUG` |
| `--log-dir` | Directory path for log files (automatically generates timestamped log files) | `anpe extract --log-dir ./logs` |

### Example Commands

**Setup models with logging:**
```bash
anpe setup --log-level DEBUG --log-dir logs
```

**Extract and output to JSON:**
```bash
anpe extract -f input.txt -o output_dir -t json
```

**Batch processing:**
```bash
anpe extract -d input_directory --output-dir output_directory -t json --metadata
```

**Advanced extraction with filters:**
```bash
anpe extract -f input.txt --min-length 2 --max-length 10 --no-pronouns --structures "determiner,named_entity" -o output_dir
```

**With logging:**
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
2. **Pattern Matching**: Applying rules to detect specific syntactic constructions
3. **Feature Extraction**: Identifying determiners, modifiers, and other grammatical features

The system categorizes patterns into fundamental types, organized from simple to complex. For a comprehensive explanation of all structure patterns and their detection logic, please refer to the [structure_patterns.md](structure_patterns.md) file included in the repository.

| Type | Description | Example | Detection Rationale |
|------|-------------|---------|---------------------|
| **Determiner** | Contains determiners (the, a, an, this, that, these, those) | "the summer" | Presence of DET tag with dependency relation "det" |
| **Adjectival Modifier** | Contains adjective modifiers | "unrealised love" | Presence of ADJ tag with dependency relation "amod" |
| **Prepositional Modifier** | Prepositional phrase modifiers | "butler at Darlington Hall" | Preposition with "prep" dependency and a prepositional object |
| **Compound** | Compound nouns forming a single conceptual unit | "Darlington Hall" | Nouns with "compound" dependency or adjacent nouns |
| **Possessive** | Possessive constructions with markers or pronouns | "his housekeeper" | Presence of POS tag or possessive pronouns (PRP$) |
| **Quantified** | Quantified NPs with numbers or quantity words | "two world wars" | Presence of NUM tag or quantifying determiners |
| **Coordinated** | Coordinated elements joined by conjunctions | "Stevens and England" | Presence of coordinating conjunction (cc) and conjoined elements |
| **Appositive** | One NP renames or explains another | "Stevens, a long-serving butler" | Presence of appositive dependency or NPs separated by comma |
| **Relative Clause** | Clause that modifies a noun | "a past that takes in fascism" | Presence of relative pronoun introducing clause |

For a comprehensive explanation of all structure patterns and their detection logic, please refer to the [structure_patterns.md](structure_patterns.md) file included in the repository. This system enables precise identification of noun phrase structures while maintaining high processing efficiency.

## Contributing

Contributions are welcome! Here are some ways you can contribute:

1. **Report bugs**: Submit issues for any bugs you find
2. **Suggest features**: Submit issues for feature requests
3. **Submit pull requests**: Implement new features or fix bugs

### Testing

ANPE uses pytest for testing. The test suite includes unit tests, integration tests, and CLI tests that verify the functionality of the package.

#### Running Tests

To run the tests, you need to have pytest installed:

```bash
pip install pytest
```

Then, you can run the tests with:

```bash
python -m pytest
```

#### Test Structure

The test suite is organized into several files that test different aspects of the package:

- **test_extractor.py**: Tests the core functionality of the `ANPEExtractor` class, including basic extraction, metadata, nested structures, and custom configurations.
- **test_cli.py**: Tests the command-line interface functionality.
- **test_integration.py**: Tests the integration between different components of the package.
- **test_utils.py**: Tests utility functions like exporting, logging, and structural analysis.

#### Adding New Tests

When contributing to ANPE, please consider adding tests for your changes. Tests should be added to the appropriate file based on what component they are testing.

For example, if you are adding a new feature to the extractor, you should add tests to `test_extractor.py`:

```python
def test_my_new_feature(self):
    """Test the new feature."""
    extractor = ANPEExtractor()
    # Test setup and assertions
```

If you are adding a new utility function, you should add tests to `test_utils.py`:

```python
def test_my_utility_function(self):
    """Test the utility function."""
    # Test setup and assertions
```

## **Troubleshooting**
If you encounter issues with model setup:
1. Ensure you have an active internet connection.
2. Run the `setup_models` utility manually and check the logs for errors.
3. If the issue persists, delete the model directories and retry the setup.
4. For specific model installation issues, refer to the documentation for [spaCy](https://spacy.io/), [Benepar](https://github.com/nikitakit/self-attentive-parser), and [NLTK](https://www.nltk.org/).

## Citation

If you use ANPE in your research or projects, please cite it as follows:

### BibTeX
```bibtex
@software{Chen_ANPE_2024,
  author = {Chen, Nuo},
  title = {{ANPE: Another Noun Phrase Extractor}},
  url = {https://github.com/richard20000321/anpe},
  version = {0.1.0},
  year = {2025}
}
```

### Plain Text (APA style)
Chen, N. (2025). *ANPE: Another Noun Phrase Extractor* (Version 0.1.0) [Computer software]. Retrieved from https://github.com/richard20000321/anpe
