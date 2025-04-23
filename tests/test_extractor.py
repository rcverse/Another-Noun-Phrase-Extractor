import unittest
import tempfile
from pathlib import Path
import os
import json
import csv
import time
from anpe import ANPEExtractor
from unittest.mock import patch, MagicMock

class TestANPEExtractor(unittest.TestCase):
    """Test cases for ANPEExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ANPEExtractor()
        self.test_text = "The team of scientists published their exciting research on climate change."
        self.pronoun_text = "He saw it. They liked the big red car."
        self.newline_text = "First sentence.\nSecond sentence with\n a line break."
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)
    
    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_initialization(self):
        """Test that the extractor initializes correctly."""
        self.assertIsInstance(self.extractor, ANPEExtractor)
    
    def test_basic_extraction(self):
        """Test basic noun phrase extraction."""
        result = self.extractor.extract(self.test_text)
        self.assertIn("results", result)
        self.assertIsInstance(result["results"], list)
        
        # Expected top-level NPs in our test text:
        # - "The team of scientists"
        # - "their exciting research on climate change"
        # Note: "climate change" is a nested NP, not a top-level one
        self.assertEqual(len(result["results"]), 2)
        
        # Verify these expected noun phrases are found
        expected_nps = [
            "The team of scientists",
            "their exciting research on climate change"
        ]
        found_nps = [np["noun_phrase"].lower() for np in result["results"]]
        for expected_np in expected_nps:
            self.assertTrue(any(expected_np.lower() == found_np for found_np in found_nps),
                           f"Expected NP '{expected_np}' not found in results")
    
    def test_extraction_with_metadata(self):
        """Test extraction with metadata."""
        result = self.extractor.extract(self.test_text, metadata=True)
        self.assertIn("results", result)
        for np in result["results"]:
            self.assertIn("metadata", np)
            self.assertIn("length", np["metadata"])
            self.assertIn("structures", np["metadata"])
            
            # Verify specific structures for each noun phrase
            if np["noun_phrase"].lower() == "the team of scientists":
                self.assertIn("determiner", np["metadata"]["structures"])
                self.assertIn("prepositional_modifier", np["metadata"]["structures"])
            elif np["noun_phrase"].lower() == "their exciting research on climate change":
                self.assertIn("adjectival_modifier", np["metadata"]["structures"])
                self.assertIn("prepositional_modifier", np["metadata"]["structures"])
                self.assertIn("possessive", np["metadata"]["structures"])
                self.assertIn("compound", np["metadata"]["structures"])
    
    def test_extraction_with_nested(self):
        """Test extraction with nested noun phrases."""
        result = self.extractor.extract(self.test_text, include_nested=True)
        self.assertIn("results", result)
        
        # In our sentence, "climate change" should be a child of 
        # "their exciting research on climate change"
        found_parent_child = False
        for np in result["results"]:
            self.assertIn("children", np)
            if np["noun_phrase"].lower() == "their exciting research on climate change":
                for child in np["children"]:
                    if child["noun_phrase"].lower() == "climate change":
                        found_parent_child = True
                        break
        
        # The parent-child relationship should exist
        self.assertTrue(found_parent_child, 
                       "Expected parent-child relationship not found")
    
    def test_extraction_with_config(self):
        """Test extraction with custom length configuration."""
        custom_extractor = ANPEExtractor({
            "min_length": 2,
            "max_length": 4
        })
        result = custom_extractor.extract(self.test_text)
        
        # With min_length=2 and max_length=4, we should only get phrases with 2-4 words
        for np in result["results"]:
            tokens = np["noun_phrase"].split()
            self.assertGreaterEqual(len(tokens), 2)
            self.assertLessEqual(len(tokens), 4)
            
        # "The team of scientists" should be included (4 words)
        # "their exciting research on climate change" should be excluded (6 words)
        found_nps = [np["noun_phrase"].lower() for np in result["results"]]
        self.assertTrue(any("the team of scientists" in found_np for found_np in found_nps),
                        "Expected NP 'The team of scientists' not found")
        
        # Make sure the longer phrase is not included
        self.assertFalse(any("their exciting research on climate change" in found_np 
                              for found_np in found_nps),
                         "Phrase 'their exciting research on climate change' should be excluded by max_length")

    def test_config_no_pronouns(self):
        """Test extraction with accept_pronouns=False configuration."""
        no_pronoun_extractor = ANPEExtractor({"accept_pronouns": False})
        result = no_pronoun_extractor.extract(self.pronoun_text)
        found_nps = [np['noun_phrase'].lower() for np in result["results"]]
        
        # Ensure single-word pronouns are excluded
        self.assertNotIn("he", found_nps)
        self.assertNotIn("it", found_nps)
        self.assertNotIn("they", found_nps)
        # Ensure other NPs are still found
        self.assertTrue(any("the big red car" in found_np for found_np in found_nps))

    def test_config_structure_filters(self):
        """Test extraction with structure_filters configuration."""
        # Extract with metadata to check structures
        # Filter for NPs that have a determiner
        filter_extractor = ANPEExtractor({"structure_filters": ["determiner"]})
        result = filter_extractor.extract(self.test_text, metadata=True)
        
        # All results should have 'determiner' in their structures
        self.assertGreater(len(result["results"]), 0) # Ensure we got some results
        for np in result["results"]:
            self.assertIn("metadata", np)
            self.assertIn("structures", np["metadata"])
            self.assertIn("determiner", np["metadata"]["structures"],
                          f"NP '{np['noun_phrase']}' missing 'determiner' structure despite filter")
            
        # Example check: 'the team of scientists' should be present
        found_nps = [np['noun_phrase'].lower() for np in result["results"]]
        self.assertTrue(any("the team of scientists" in found_np for found_np in found_nps))
        
        # Now filter for something not present in the top-level NPs 
        # (e.g., 'appositive' which isn't in the main NPs of self.test_text)
        filter_extractor_appos = ANPEExtractor({"structure_filters": ["appositive"]})
        result_appos = filter_extractor_appos.extract(self.test_text, metadata=True)
        # We expect no results based on the filter and the test text
        self.assertEqual(len(result_appos["results"]), 0)

    def test_config_no_newline_breaks(self):
        """Test extraction with newline_breaks=False configuration."""
        # Default behavior (newline_breaks=True)
        # Uses self.extractor initialized in setUp which has default config (newline_breaks=True)
        result_default = self.extractor.extract(self.newline_text)
        nps_default = [np['noun_phrase'] for np in result_default["results"]]
        print(f"--- NPs with newline_breaks=True: {nps_default} ---")
        
        # Check the actual output based on debug script findings
        # The parser includes the trailing period and might have whitespace variations
        expected_np_text = "second sentence with a line break ."
        # Normalize extracted text for comparison (lower, strip)
        normalized_nps_default = [' '.join(np.lower().split()) for np in nps_default]
        self.assertTrue(
            any(expected_np_text == norm_np for norm_np in normalized_nps_default),
            f"Expected to find normalized NP '{expected_np_text}' when newline_breaks=True. Found: {normalized_nps_default}"
        )
        
        # Custom behavior (newline_breaks=False)
        extractor_no_break = ANPEExtractor({"newline_breaks": False})
        result_no_break = extractor_no_break.extract(self.newline_text)
        nps_no_break = [np['noun_phrase'] for np in result_no_break["results"]]
        print(f"--- NPs with newline_breaks=False: {nps_no_break} ---")

        # Ensure *some* results are found in both cases
        self.assertGreater(len(nps_default), 0, "Expected some NPs when newline_breaks=True")
        self.assertGreater(len(nps_no_break), 0, "Expected some NPs when newline_breaks=False")

    def test_config_explicit_model_selection(self):
        """Test that explicitly providing models in config uses them."""
        
        # Dictionary to store the mock nlp object created by the side effect
        mock_nlp_holder = {'instance': None}
        
        # Use patch as a context manager - remove benepar.Parser patch
        with patch('anpe.extractor.spacy.load') as mock_spacy_load:
            
            # Define a side effect function for spacy.load mock
            def spacy_load_side_effect(model_name, *args, **kwargs):
                print(f"--- MOCK spacy.load called with: {model_name} ---") # Debug print
                if model_name == "en_core_web_md":
                    # If called with the default model, raise error immediately
                    raise OSError("Mock detected attempt to load default spaCy model!")
                elif model_name == "en_core_web_lg":
                    # If called with the intended model, return a MagicMock
                    # Configure the mock to simulate the nlp object
                    mock_nlp = MagicMock()
                    # Simulate pipe_names attribute (needed for check)
                    mock_nlp.pipe_names = [] 
                    # Simulate add_pipe method (returns None by default)
                    mock_nlp.add_pipe = MagicMock(return_value=None)
                    # Store the mock nlp instance
                    mock_nlp_holder['instance'] = mock_nlp 
                    return mock_nlp
                else:
                    # For any other unexpected call, maybe return default mock
                    return MagicMock()

            # Assign the side effect
            mock_spacy_load.side_effect = spacy_load_side_effect
            
            # Config under test
            config = {
                "spacy_model": "en_core_web_lg", # Explicitly use lg
                "benepar_model": "benepar_en3_large" # Explicitly use large
            }

            # Action under test: Initialize extractor with specific config
            # This should NOT raise the OSError defined in the side_effect
            try:
                # We also need to mock nltk.data.find for the punkt check
                with patch('anpe.extractor.nltk.data.find') as mock_nltk_find:
                    mock_nltk_find.return_value = True # Assume punkt is found
                    explicit_extractor = ANPEExtractor(config)
            except OSError as e:
                # If the specific OSError we defined is raised, the test fails
                if "Mock detected attempt to load default SpaCy model!" in str(e):
                    self.fail(f"ANPEExtractor incorrectly tried to load default spaCy model: {e}")
                else:
                    # Re-raise other OS errors if necessary
                    raise e
            
            # Verification: Check spacy.load calls *after* successful initialization
            calls = mock_spacy_load.call_args_list
            print(f"--- FINAL MOCK spacy.load CALLS: {calls} ---") # Debug print
            
            # Assert that the intended model was called
            intended_spacy_call = unittest.mock.call("en_core_web_lg")
            self.assertIn(intended_spacy_call, calls)
            
            # The side effect already prevents the unwanted call, so no need for assertNotIn
            
            # --- Verify benepar integration via nlp.add_pipe ---
            # Retrieve the mock nlp object that should have been created
            mock_nlp_instance = mock_nlp_holder.get('instance')
            self.assertIsNotNone(mock_nlp_instance, "Mock nlp object was not created by spacy.load side effect")
            
            # Assert that add_pipe was called correctly on the mock nlp object
            # Use assert_any_call since 'sentencizer' might also be added
            mock_nlp_instance.add_pipe.assert_any_call(
                "benepar", 
                config={"model": "benepar_en3_large"}
            )
            print("--- MOCK nlp.add_pipe called correctly. ---") # Debug print

    # --- Tests for Export Functionality --- 

    def test_export_to_directory(self):
        """Test exporting to a directory (generates timestamped filename)."""
        # Test for each format
        for fmt in ["txt", "csv", "json"]:
            # Export providing the directory path
            exported_file_path_str = self.extractor.export(
                self.test_text, 
                output=str(self.output_dir),
                format=fmt,
                metadata=True,
                include_nested=True
            )
            exported_file_path = Path(exported_file_path_str)

            # Verify the returned path is within the directory and has the correct extension
            self.assertEqual(exported_file_path.resolve().parent, self.output_dir.resolve())
            self.assertTrue(exported_file_path.name.startswith("anpe_export_"))
            self.assertTrue(exported_file_path.name.endswith(f".{fmt}"))
            self.assertTrue(exported_file_path.resolve().exists())
            self.assertTrue(exported_file_path.resolve().is_file())

            # Basic content check (e.g., non-empty for txt)
            if fmt == "txt":
                self.assertGreater(exported_file_path.resolve().stat().st_size, 0)

    def test_export_to_specific_file(self):
        """Test exporting to a specific file path."""
        output_filepath = self.output_dir / "my_specific_results.json"
        exported_file_path_str = self.extractor.export(
            self.test_text,
            output=str(output_filepath),
            format="json", # Format matches extension
            metadata=True,
            include_nested=True
        )
        exported_file_path = Path(exported_file_path_str)

        # Verify the exact file was created and returned
        self.assertEqual(exported_file_path.resolve(), output_filepath.resolve())
        self.assertTrue(exported_file_path.resolve().exists())
        
        # Verify content
        with open(exported_file_path.resolve(), 'r', encoding='utf-8') as f:
            result = json.load(f)
        self.assertIn("results", result)
        self.assertTrue(len(result["results"]) > 0)

    def test_export_creates_parent_directory(self):
        """Test that export creates non-existent parent directories."""
        nested_output_filepath = self.output_dir / "new_subdir" / "nested_file.txt"
        
        # Ensure parent does not exist initially
        self.assertFalse(nested_output_filepath.parent.exists())

        exported_file_path_str = self.extractor.export(
            self.test_text,
            output=str(nested_output_filepath),
            format="txt"
        )
        exported_file_path = Path(exported_file_path_str)

        # Verify the file was created in the new subdirectory
        self.assertEqual(exported_file_path.resolve(), nested_output_filepath.resolve())
        self.assertTrue(exported_file_path.resolve().parent.exists())
        self.assertTrue(exported_file_path.resolve().parent.is_dir())
        self.assertTrue(exported_file_path.resolve().exists())
        self.assertGreater(exported_file_path.resolve().stat().st_size, 0)
        
    def test_export_extension_mismatch(self):
        """Test export behavior when file extension mismatches format."""
        output_filepath = self.output_dir / "mismatch_test.txt" # .txt extension
        exported_file_path_str = self.extractor.export(
            self.test_text,
            output=str(output_filepath),
            format="json", # but format is json
            metadata=True,
            include_nested=True
        )
        exported_file_path = Path(exported_file_path_str)
        
        # Verify the exact file (with .txt extension) was created
        self.assertEqual(exported_file_path.resolve(), output_filepath.resolve())
        self.assertTrue(exported_file_path.resolve().exists())

        # Verify the content is JSON despite the .txt extension
        try:
            with open(exported_file_path.resolve(), 'r', encoding='utf-8') as f:
                result = json.load(f)
            self.assertIn("metadata", result)
            self.assertIn("results", result)
            self.assertTrue(len(result["results"]) > 0)
        except json.JSONDecodeError:
            self.fail("File mismatch_test.txt did not contain valid JSON content.")

    def test_export_unsupported_extension(self):
        """Test export behavior with unsupported file extension."""
        # Create a path with an unsupported extension (.xlsx)
        unsupported_path = self.output_dir / "data.xlsx"
        
        exported_file_path_str = self.extractor.export(
            self.test_text,
            output=str(unsupported_path),
            format="json",
            metadata=True
        )
        exported_file_path = Path(exported_file_path_str)
        
        # Verify that the file was created in the same directory but with a timestamped name
        self.assertEqual(exported_file_path.resolve().parent, unsupported_path.resolve().parent)
        self.assertTrue(exported_file_path.name.startswith("anpe_export_"))
        self.assertTrue(exported_file_path.name.endswith(".json"))  # Should have the format extension
        self.assertNotEqual(exported_file_path.name, "data.xlsx")   # Should not be the original name
        
        # Verify the file exists and contains valid JSON
        self.assertTrue(exported_file_path.resolve().exists())
        with open(exported_file_path.resolve(), 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.assertIn("results", data)
            self.assertTrue(len(data["results"]) > 0)

if __name__ == "__main__":
    unittest.main() 