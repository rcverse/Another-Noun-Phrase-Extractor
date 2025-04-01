import unittest
from anpe import ANPEExtractor

class TestANPEExtractor(unittest.TestCase):
    """Test cases for ANPEExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = ANPEExtractor()
        self.test_text = "The team of scientists published their exciting research on climate change."
    
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
        """Test extraction with custom configuration."""
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

if __name__ == "__main__":
    unittest.main() 