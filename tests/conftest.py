import pytest
from pathlib import Path
import tempfile
import os

from anpe import ANPEExtractor

@pytest.fixture(scope="session")
def sample_texts():
    """Return a dictionary of sample texts for testing."""
    return {
        "simple": "The team of scientists published their exciting research on climate change.",
        "complex": """
        The big brown dog chased the small black cat. 
        John's new car, which is very expensive, was parked in the garage.
        The team of scientists published their exciting research on climate change.
        """,
        "nested": """The president of the United States, who was elected last year, 
        gave a speech at the annual conference on climate change in Paris.
        The team of scientists published their exciting research on climate change.""",
        "structures": """
        The team of scientists (determiner, prepositional_modifier)
        Their exciting research on climate change (adjectival_modifier, prepositional_modifier, possessive, compound)
        Climate change (compound)
        John's car (possessive)
        Many books (quantified)
        Dogs and cats (coordinated)
        The president, a skilled orator (appositive)
        The book that I read (standard relative)
        The book written by Smith (reduced relative)
        The idea that we should leave (finite complement)
        The plan to visit Paris (nonfinite complement)
        """
    }

@pytest.fixture(scope="session")
def extractor():
    """Return an ANPEExtractor instance with default configuration."""
    return ANPEExtractor()

@pytest.fixture(scope="session")
def custom_extractor():
    """Return an ANPEExtractor instance with custom configuration."""
    return ANPEExtractor({
        "min_length": 2,
        "max_length": 5,
        "accept_pronouns": False,
        "log_level": "DEBUG"
    })

@pytest.fixture(scope="function")
def temp_directory():
    """Create and return a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="function")
def temp_input_file(temp_directory, sample_texts):
    """Create and return a temporary input file with sample text."""
    input_file = temp_directory / "input.txt"
    with open(input_file, 'w', encoding='utf-8') as f:
        f.write(sample_texts["complex"])
    return input_file

@pytest.fixture(scope="function")
def temp_output_dir(temp_directory):
    """Create and return a temporary output directory."""
    output_dir = temp_directory / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir 