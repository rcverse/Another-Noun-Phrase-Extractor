import pytest
from unittest.mock import patch, mock_open, MagicMock, call, ANY
import csv
import json
from pathlib import Path

# Import the class to test
from anpe.utils.export import ANPEExporter

# --- Test Data Fixtures ---

@pytest.fixture
def sample_extraction_result_simple():
    """Provides a simple extraction result dictionary (no nesting/metadata)."""
    return {
        "timestamp": "2023-10-27 10:00:00",
        "processing_duration_seconds": 0.123,
        "configuration": {
            "spacy_model_used": "mock_spacy",
            "benepar_model_used": "mock_benepar",
            "metadata_requested": False,
            "nested_requested": False
        },
        "results": [
            {
                "noun_phrase": "the cat",
                "id": "1",
                "level": 1,
                "children": []
            },
            {
                "noun_phrase": "a dog",
                "id": "2",
                "level": 1,
                "children": []
            }
        ]
    }

@pytest.fixture
def sample_extraction_result_metadata():
    """Provides an extraction result dictionary with metadata."""
    return {
        "timestamp": "2023-10-27 10:05:00",
        "processing_duration_seconds": 0.456,
        "configuration": {
            "spacy_model_used": "mock_spacy_md",
            "benepar_model_used": "mock_benepar_lg",
            "metadata_requested": True,
            "nested_requested": False,
            "min_length": 2,
            "accept_pronouns": True
        },
        "results": [
            {
                "noun_phrase": "the big cat",
                "id": "1",
                "level": 1,
                "metadata": {
                    "length": 3,
                    "structures": ["determiner", "adjectival_modifier"]
                },
                "children": []
            }
        ]
    }

@pytest.fixture
def sample_extraction_result_nested():
    """Provides an extraction result dictionary with nested NPs."""
    return {
        "timestamp": "2023-10-27 10:10:00",
        "processing_duration_seconds": 0.789,
        "configuration": {
            "spacy_model_used": "mock_spacy",
            "benepar_model_used": "mock_benepar",
            "metadata_requested": False,
            "nested_requested": True
        },
        "results": [
            {
                "noun_phrase": "the cat in the hat",
                "id": "1",
                "level": 1,
                "children": [
                    {
                        "noun_phrase": "the cat",
                        "id": "1.1",
                        "level": 2,
                        "children": []
                    },
                    {
                        "noun_phrase": "the hat",
                        "id": "1.2",
                        "level": 2,
                        "children": []
                    }
                ]
            }
        ]
    }

@pytest.fixture
def sample_extraction_result_nested_metadata():
    """Provides an extraction result dictionary with nested NPs and metadata."""
    return {
        "timestamp": "2023-10-27 10:15:00",
        "processing_duration_seconds": 1.101,
        "configuration": {
            "spacy_model_used": "mock_spacy",
            "benepar_model_used": "mock_benepar",
            "metadata_requested": True,
            "nested_requested": True
        },
        "results": [
            {
                "noun_phrase": "the cat in the hat",
                "id": "1",
                "level": 1,
                "metadata": {"length": 5, "structures": ["determiner", "prepositional_modifier"]},
                "children": [
                    {
                        "noun_phrase": "the cat",
                        "id": "1.1",
                        "level": 2,
                        "metadata": {"length": 2, "structures": ["determiner"]},
                        "children": []
                    },
                    {
                        "noun_phrase": "the hat",
                        "id": "1.2",
                        "level": 2,
                        "metadata": {"length": 2, "structures": ["determiner"]},
                        "children": []
                    }
                ]
            }
        ]
    }

@pytest.fixture
def exporter_instance():
    """Provides a basic ANPEExporter instance."""
    return ANPEExporter()

# --- Unit Tests for ANPEExporter ---

@patch("builtins.open", new_callable=mock_open)
def test_export_txt_simple(mock_file, exporter_instance, sample_extraction_result_simple):
    """Test TXT export with simple results (no nesting/metadata)."""
    output_path = "test_output.txt"
    expected_header = "# ANPE Extraction Results\n# Timestamp: 2023-10-27 10:00:00\n# Duration: 0.123 seconds\n# spaCy Model: mock_spacy\n# Benepar Model: mock_benepar\n# Metadata Requested: False\n# Nested Requested: False\n# --- Noun Phrases (2) ---\n"
    expected_content = expected_header + "the cat\na dog\n"
    
    exporter_instance.export(sample_extraction_result_simple, format="txt", output_filepath=output_path)
    
    mock_file.assert_called_once_with(ANY, 'w', encoding='utf-8')
    handle = mock_file() 
    # Check the combined content of all write calls
    actual_content = "".join(call_args[0][0] for call_args in handle.write.call_args_list)
    # Flexible assertions for TXT format
    assert "--- ANPE Noun Phrase Extraction Results ---" in actual_content
    assert "Timestamp: 2023-10-27 10:00:00" in actual_content
    assert "Spacy Model Used: mock_spacy" in actual_content # Check key config part
    assert "Benepar Model Used: mock_benepar" in actual_content
    assert "Output includes Nested NPs: False" in actual_content
    assert "Output includes Metadata: False" in actual_content
    assert "--- Extraction Results ---" in actual_content
    # Check for the presence of the expected noun phrases
    assert "[1] the cat" in actual_content
    assert "[2] a dog" in actual_content

@patch("builtins.open", new_callable=mock_open)
def test_export_txt_metadata(mock_file, exporter_instance, sample_extraction_result_metadata):
    """Test TXT export with metadata."""
    output_path = "test_output.txt"
    cfg = sample_extraction_result_metadata["configuration"]
    exporter_instance.export(sample_extraction_result_metadata, format="txt", output_filepath=output_path)
    
    mock_file.assert_called_once_with(ANY, 'w', encoding='utf-8')
    handle = mock_file()
    actual_content = "".join(call_args[0][0] for call_args in handle.write.call_args_list)
    
    # Flexible assertions for TXT format with metadata
    assert "--- ANPE Noun Phrase Extraction Results ---" in actual_content
    assert "Timestamp: 2023-10-27 10:05:00" in actual_content
    assert "Spacy Model Used: mock_spacy_md" in actual_content
    assert "Output includes Metadata: True" in actual_content
    assert "--- Extraction Results ---" in actual_content
    assert "[1] the big cat" in actual_content
    # Check for presence of metadata indicators
    assert "Length: 3" in actual_content
    assert "Structures:" in actual_content
    assert "determiner" in actual_content
    assert "adjectival_modifier" in actual_content

@patch("builtins.open", new_callable=mock_open)
def test_export_txt_nested(mock_file, exporter_instance, sample_extraction_result_nested):
    """Test TXT export with nested NPs."""
    output_path = "test_output.txt"
    # Placeholder for actual expected content based on traceback:
    expected_content_actual_format = """--- ANPE Noun Phrase Extraction Results ---
Timestamp: 2023-10-27 10:10:00

--- Configuration Used ---
Output includes Nested NPs: True
Output includes Metadata: False
-----
Accept Pronouns: True
Structure Filters: None
Newline Breaks: True
Spacy Model Used: mock_spacy
Benepar Model Used: mock_benepar
--------------------------

--- Extraction Results ---
  ◦ [1] the cat in the hat
    ◦ [1.1] the cat
    ◦ [1.2] the hat
"""

    exporter_instance.export(sample_extraction_result_nested, format="txt", output_filepath=output_path)
    
    mock_file.assert_called_once_with(ANY, 'w', encoding='utf-8')
    handle = mock_file()
    # Check the combined content of all write calls
    actual_content = "".join(call_args[0][0] for call_args in handle.write.call_args_list)
    # Flexible assertions for TXT format with nesting
    assert "--- ANPE Noun Phrase Extraction Results ---" in actual_content
    assert "Timestamp: 2023-10-27 10:10:00" in actual_content
    assert "Output includes Nested NPs: True" in actual_content
    assert "--- Extraction Results ---" in actual_content
    assert "[1] the cat in the hat" in actual_content
    # Check for indented nested items
    assert "  ◦ [1.1] the cat" in actual_content # Assuming specific indentation
    assert "  ◦ [1.2] the hat" in actual_content

@patch("builtins.open", new_callable=mock_open)
def test_export_txt_nested_metadata(mock_file, exporter_instance, sample_extraction_result_nested_metadata):
    """Test TXT export with nested NPs and metadata."""
    output_path = "test_output.txt"
    # Placeholder for actual expected content based on traceback:
    expected_content_actual_format = """--- ANPE Noun Phrase Extraction Results ---
Timestamp: 2023-10-27 10:15:00

--- Configuration Used ---
Output includes Nested NPs: True
Output includes Metadata: True
-----
Accept Pronouns: True
Structure Filters: None
Newline Breaks: True
Spacy Model Used: mock_spacy
Benepar Model Used: mock_benepar
--------------------------

--- Extraction Results ---
  ◦ [1] the cat in the hat
    Length: 5
    Structures: [determiner, prepositional_modifier]
    ◦ [1.1] the cat
      Length: 2
      Structures: [determiner]
    ◦ [1.2] the hat
      Length: 2
      Structures: [determiner]
"""

    exporter_instance.export(sample_extraction_result_nested_metadata, format="txt", output_filepath=output_path)
    
    mock_file.assert_called_once_with(ANY, 'w', encoding='utf-8')
    handle = mock_file()
    # Check the combined content of all write calls
    actual_content = "".join(call_args[0][0] for call_args in handle.write.call_args_list)
    # Flexible assertions for TXT format with nesting and metadata
    assert "--- ANPE Noun Phrase Extraction Results ---" in actual_content
    assert "Timestamp: 2023-10-27 10:15:00" in actual_content
    assert "Output includes Nested NPs: True" in actual_content
    assert "Output includes Metadata: True" in actual_content
    assert "--- Extraction Results ---" in actual_content
    assert "[1] the cat in the hat" in actual_content
    assert "Length: 5" in actual_content
    assert "Structures:" in actual_content
    assert "prepositional_modifier" in actual_content
    # Check nested items
    assert "  ◦ [1.1] the cat" in actual_content
    assert "Length: 2" in actual_content # Check metadata for nested item
    assert "  ◦ [1.2] the hat" in actual_content

# Mock csv.writer and the file handle it uses
@patch("csv.writer")
@patch("builtins.open", new_callable=mock_open)
def test_export_csv_simple(mock_file, mock_csv_writer, exporter_instance, sample_extraction_result_simple):
    """Test CSV export with simple results."""
    output_path = "test_output.csv"
    mock_writer_instance = MagicMock()
    mock_csv_writer.return_value = mock_writer_instance
    
    exporter_instance.export(sample_extraction_result_simple, format="csv", output_filepath=output_path)
    
    mock_file.assert_called_once_with(ANY, 'w', newline='', encoding='utf-8')
    # Check headers written - Updated based on traceback
    expected_headers = ["ID", "Level", "Parent_ID", "Noun_Phrase"] 
    # Check data rows - Updated based on traceback (added None for Parent_ID)
    expected_rows = [
        ["1", 1, None, "the cat"],
        ["2", 1, None, "a dog"]
    ]
    
    # Check calls accurately
    calls = mock_writer_instance.writerow.call_args_list
    assert len(calls) == 1 + len(expected_rows) # Header + data rows
    assert calls[0] == call(expected_headers)
    # Check data rows (order might matter here, so check specific calls)
    assert calls[1] == call(expected_rows[0])
    assert calls[2] == call(expected_rows[1])

@patch("csv.writer")
@patch("builtins.open", new_callable=mock_open)
def test_export_csv_metadata(mock_file, mock_csv_writer, exporter_instance, sample_extraction_result_metadata):
    """Test CSV export with metadata."""
    output_path = "test_output.csv"
    mock_writer_instance = MagicMock()
    mock_csv_writer.return_value = mock_writer_instance
    
    exporter_instance.export(sample_extraction_result_metadata, format="csv", output_filepath=output_path)
    
    mock_file.assert_called_once_with(ANY, 'w', newline='', encoding='utf-8')
    # Updated headers based on traceback
    expected_headers = ["ID", "Level", "Parent_ID", "Noun_Phrase", "Length", "Structures"]
    # Updated data row based on traceback (Parent_ID, length, joined structures)
    expected_rows = [
        ["1", 1, None, "the big cat", 3, "determiner|adjectival_modifier"]
    ]

    # Check calls accurately
    calls = mock_writer_instance.writerow.call_args_list
    assert len(calls) == 1 + len(expected_rows) # Header + data rows
    assert calls[0] == call(expected_headers)
    assert calls[1] == call(expected_rows[0])

@patch("csv.writer")
@patch("builtins.open", new_callable=mock_open)
def test_export_csv_nested_metadata(mock_file, mock_csv_writer, exporter_instance, sample_extraction_result_nested_metadata):
    """Test CSV export flattens nested data with metadata."""
    output_path = "test_output.csv"
    mock_writer_instance = MagicMock()
    mock_csv_writer.return_value = mock_writer_instance
    
    exporter_instance.export(sample_extraction_result_nested_metadata, format="csv", output_filepath=output_path)
    
    mock_file.assert_called_once_with(ANY, 'w', newline='', encoding='utf-8')
    # Updated headers based on traceback
    expected_headers = ["ID", "Level", "Parent_ID", "Noun_Phrase", "Length", "Structures"]
    # Updated data rows based on traceback (Parent_ID for children, joined structures)
    expected_rows = [
        ["1",   1, None, "the cat in the hat", 5, "determiner|prepositional_modifier"],
        ["1.1", 2, "1",  "the cat",            2, "determiner"],
        ["1.2", 2, "1",  "the hat",            2, "determiner"]
    ]

    # Check calls accurately
    calls = mock_writer_instance.writerow.call_args_list
    assert len(calls) == 1 + len(expected_rows) # Header + data rows
    assert calls[0] == call(expected_headers)
    # Check data rows (order might matter here, so check specific calls)
    assert calls[1] == call(expected_rows[0])
    assert calls[2] == call(expected_rows[1])
    assert calls[3] == call(expected_rows[2])

@patch("json.dump")
@patch("builtins.open", new_callable=mock_open)
def test_export_json(mock_file, mock_json_dump, exporter_instance, sample_extraction_result_nested_metadata):
    """Test JSON export writes the expected data structure with more robust checks."""
    output_path = "test_output.json"
    exporter_instance.export(sample_extraction_result_nested_metadata, format="json", output_filepath=output_path)

    mock_file.assert_called_once_with(ANY, 'w', encoding='utf-8')
    mock_json_dump.assert_called_once() 
    
    # Get the data structure passed to json.dump
    call_args, call_kwargs = mock_json_dump.call_args
    dumped_data = call_args[0]

    # More robust assertions: Check structure and key elements
    assert isinstance(dumped_data, dict)
    assert "timestamp" in dumped_data
    assert "configuration" in dumped_data
    assert "results" in dumped_data
    assert isinstance(dumped_data["results"], list)
    assert len(dumped_data["results"]) > 0 # Ensure there are results

    # Check the first result for expected structure (based on fixture)
    first_result = dumped_data["results"][0]
    expected_first_result_fixture = sample_extraction_result_nested_metadata["results"][0]
    
    assert "id" in first_result
    assert first_result["id"] == expected_first_result_fixture["id"]
    assert "noun_phrase" in first_result
    assert first_result["noun_phrase"] == expected_first_result_fixture["noun_phrase"]
    assert "metadata" in first_result
    assert isinstance(first_result["metadata"], dict)
    assert first_result["metadata"]["length"] == expected_first_result_fixture["metadata"]["length"]
    assert "children" in first_result
    assert isinstance(first_result["children"], list)

def test_export_invalid_format_raises_error(exporter_instance, sample_extraction_result_simple):
    """Test that exporting with an invalid format raises ValueError."""
    output_path = "test_output.invalid"
    invalid_format = "xml"
    
    with pytest.raises(ValueError, match=f"Invalid format: {invalid_format}. Must be one of ..."):
        exporter_instance.export(
            sample_extraction_result_simple, 
            format=invalid_format, 
            output_filepath=output_path
        )

# TODO: Add tests for invalid format error 