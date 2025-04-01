"""
Analyzer module for detecting structural patterns in noun phrases.

This module implements refined detection methods based on a comprehensive analysis
of spaCy's dependency parsing patterns for various noun phrase structures.
"""

from typing import List
import spacy
from anpe.utils.logging import get_logger


class ANPEAnalyzer:
    """Analyzes the structure of noun phrases with improved detection methods."""
    
    def __init__(self):
        """Initialize the analyzer with improved detection methods and logging."""
        self.logger = get_logger('analyzer')
        self.logger.debug("Initializing ANPEAnalyzer")
        
        # Load spaCy model
        try:
            self.logger.debug("Loading spaCy model for analysis")
            self.nlp = spacy.load("en_core_web_sm")
            self.logger.debug("spaCy model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading spaCy model: {str(e)}")
            raise
    
    def analyze(self, nps_with_length: List[tuple]) -> List[tuple]:
        """
        Analyze the structure of noun phrases.
        
        Args:
            nps_with_length: List of (np, length) tuples
            
        Returns:
            List of (np, length, structures) tuples where structures is a list
            of structural patterns found in the NP
        """
        self.logger.info(f"Analyzing structure of {len(nps_with_length)} noun phrases")
        analyzed_nps = []
        
        for np, length in nps_with_length:
            self.logger.debug(f"Analyzing structure of '{np}'")
            structures = self.analyze_single_np(np)
            self.logger.debug(f"Found structures in '{np}': {structures}")
            analyzed_nps.append((np, length, structures))
            
        self.logger.info(f"Completed structural analysis for {len(analyzed_nps)} noun phrases")
        return analyzed_nps
    
    def analyze_single_np(self, np_text: str) -> List[str]:
        """
        Analyze the structure of a single noun phrase.
        
        Args:
            np_text: The noun phrase text
            
        Returns:
            List[str]: List of structure labels for this NP
        """
        self.logger.debug(f"Analyzing structure of single NP: '{np_text}'")
        return self._analyze_structure(np_text)
    
    def is_standalone_pronoun(self, np_text: str) -> bool:
        """
        Just a wrapper for the internal _detect_pronoun method.
        """
        tokens = np_text.strip().split()
        
        if len(tokens) != 1:
            return False
            
        try:
            doc = self.nlp(np_text)
            return self._detect_pronoun(doc)
        except Exception as e:
            self.logger.error(f"Error in pronoun detection for '{np_text}': {str(e)}")
            return False
            
    def _analyze_structure(self, np_text: str) -> List[str]:
        """
        Enhanced structural analysis with improved detection methods.
        
        Args:
            np_text: A noun phrase as a string
            
        Returns:
            List of structural patterns found in the NP
        """
        try:
            doc = self.nlp(np_text)
            structures = []
            
            # First check basic structure types for simple NPs
            self.logger.debug(f"Checking if '{np_text}' is a pronoun")
            if self._detect_pronoun(doc):
                structures.append("pronoun")
                self.logger.debug(f"Found pronoun: '{np_text}'")
                
            self.logger.debug(f"Checking if '{np_text}' is a standalone noun")
            if self._detect_standalone_noun(doc):
                structures.append("standalone_noun")
                self.logger.debug(f"Found standalone noun: '{np_text}'")
            
            # Check each structural type with detailed logging
            self.logger.debug(f"Checking determiner NP for '{np_text}'")
            if self._detect_determiner_np(doc):
                structures.append("determiner")
                self.logger.debug(f"Found determiner NP in '{np_text}'")
                
            self.logger.debug(f"Checking adjectival NP for '{np_text}'")
            if self._detect_adjectival_np(doc):
                structures.append("adjectival_modifier")
                self.logger.debug(f"Found adjectival NP in '{np_text}'")
                
            self.logger.debug(f"Checking prepositional NP for '{np_text}'")
            if self._detect_prepositional_np(doc):
                structures.append("prepositional_modifier")
                self.logger.debug(f"Found prepositional NP in '{np_text}'")
                
            self.logger.debug(f"Checking compound noun for '{np_text}'")
            if self._detect_compound_noun(doc):
                structures.append("compound")
                self.logger.debug(f"Found compound noun in '{np_text}'")
                
            self.logger.debug(f"Checking possessive NP for '{np_text}'")
            if self._detect_possessive_np(doc):
                structures.append("possessive")
                self.logger.debug(f"Found possessive NP in '{np_text}'")
                
            self.logger.debug(f"Checking quantified NP for '{np_text}'")
            if self._detect_quantified_np(doc):
                structures.append("quantified")
                self.logger.debug(f"Found quantified NP in '{np_text}'")
                
            self.logger.debug(f"Checking coordinate NP for '{np_text}'")
            if self._detect_coordinate_np(doc):
                structures.append("coordinated")
                self.logger.debug(f"Found coordinate NP in '{np_text}'")
                
            self.logger.debug(f"Checking appositive NP for '{np_text}'")
            if self._detect_appositive_np(doc):
                structures.append("appositive")
                self.logger.debug(f"Found appositive NP in '{np_text}'")

            # Check for clausal structures
            # First check for nonfinite complements
            self.logger.debug(f"Checking nonfinite complement for '{np_text}'")
            if self._detect_nonfinite_complement(doc):
                structures.append("nonfinite_complement")
                self.logger.debug(f"Found nonfinite complement in '{np_text}'")
            
            # Then check for finite complements - prioritize over relative clauses
            # for disambiguating 'that'
            self.logger.debug(f"Checking finite complement clause for '{np_text}'")
            if self._detect_finite_complement(doc):
                structures.append("finite_complement")
                self.logger.debug(f"Found finite complement in '{np_text}'")
                
            # First check for relative clause
            self.logger.debug(f"Checking relative clause for '{np_text}'")
            if self._detect_relative_clause(doc):
                structures.append("relative_clause")
                self.logger.debug(f"Found relative clause in '{np_text}'")
            
            # Then check for reduced relative clause
            self.logger.debug(f"Checking reduced relative clause for '{np_text}'")
            if self._detect_reduced_relative_clause(doc):
                structures.append("reduced_relative_clause")
                self.logger.debug(f"Found reduced relative clause in '{np_text}'")
            
            # Apply post-processing validation to remove inconsistent patterns
            structures = self._validate_structures(structures, np_text)
                
            return structures
            
        except Exception as e:
            self.logger.error(f"Error analyzing structure of '{np_text}': {str(e)}")
            return []
    
    def _validate_structures(self, structures: List[str], np_text: str) -> List[str]:
        """
        Apply validation rules to remove inconsistent or implausible combinations.
        
        Args:
            structures: List of identified structures
            np_text: The noun phrase text
            
        Returns:
            List[str]: Validated list of structures
        """
        # Resolve conflict between finite complement and relative clause
        if "finite_complement" in structures and "relative_clause" in structures:
            doc = self.nlp(np_text)
            
            # Check if 'that' is used as a relative pronoun (object or subject)
            that_as_rel_pronoun = any(token.text.lower() == "that" and 
                                      token.tag_ == "WDT" and
                                      token.dep_ in ["dobj", "nsubj"] 
                                      for token in doc)
            
            # Check if head noun is a complement-taking noun
            has_complement_noun = any(
                self._is_complement_taking_noun(token)
                for token in doc if token.pos_ in ["NOUN", "PROPN"]
            )
            
            # Resolution logic:
            if that_as_rel_pronoun and not has_complement_noun:
                # If 'that' is clearly a relative pronoun, favor relative clause
                self.logger.debug(f"Removing finite_complement in favor of relative_clause in '{np_text}'")
                structures.remove("finite_complement")
            elif has_complement_noun and not that_as_rel_pronoun:
                # If head noun strongly suggests complement, favor finite complement
                self.logger.debug(f"Removing relative_clause in favor of finite_complement in '{np_text}'")
                structures.remove("relative_clause")
            else:
                # Default to relative clause in ambiguous cases
                self.logger.debug(f"Defaulting to relative_clause in ambiguous case: '{np_text}'")
                structures.remove("finite_complement")
        
        # Ensure at least one structure label
        if not structures and len(np_text.split()) > 0:
            # If no structures detected but valid text, label as base_np
            structures.append("others")
            self.logger.debug(f"Assigned others label to '{np_text}' as fallback")
                
        return structures
            
    def _detect_pronoun(self, doc) -> bool:
        """Detect if the NP is a standalone pronoun."""
        if len(doc) == 1 and doc[0].pos_ == "PRON":
            self.logger.debug(f"Found pronoun: '{doc[0].text}'")
            return True
        return False
        
    def _detect_standalone_noun(self, doc) -> bool:
        """Detect if the NP is a single noun with no modifiers."""
        if len(doc) == 1 and doc[0].pos_ in ["NOUN", "PROPN"]:
            # Check no modifiers are present
            self.logger.debug(f"Found standalone noun: '{doc[0].text}'")
            return True
        return False
    
    def _detect_determiner_np(self, doc) -> bool:
        """Detect if the NP contains a determiner structure."""
        for token in doc:
            if token.pos_ == "DET" and token.dep_ == "det" and token.head.pos_ in ["NOUN", "PROPN"]:
                self.logger.debug(f"Found determiner '{token.text}' modifying noun '{token.head.text}'")
                return True
        return False
        
    def _detect_adjectival_np(self, doc) -> bool:
        """Detect if the NP contains an adjectival modifier."""
        for token in doc:
            if token.pos_ == "ADJ" and token.dep_ == "amod" and token.head.pos_ in ["NOUN", "PROPN"]:
                self.logger.debug(f"Found adjective '{token.text}' modifying noun '{token.head.text}'")
                return True
        return False
        
    def _detect_prepositional_np(self, doc) -> bool:
        """Detect if the NP contains a prepositional modifier."""
        for token in doc:
            if token.pos_ == "ADP" and token.dep_ == "prep" and token.head.pos_ in ["NOUN", "PROPN"]:
                # Verify preposition has an object
                if any(child.dep_ == "pobj" for child in token.children):
                    self.logger.debug(f"Found preposition '{token.text}' modifying noun '{token.head.text}'")
                    return True
        return False
        
    def _detect_compound_noun(self, doc) -> bool:
        """Detect if the NP contains a compound noun."""
        for token in doc:
            if token.dep_ == "compound" and token.head.pos_ in ["NOUN", "PROPN"]:
                self.logger.debug(f"Found compound noun: '{token.text} {token.head.text}'")
                return True
        return False
        
    def _detect_possessive_np(self, doc) -> bool:
        """Detect if the NP contains a possessive structure."""
        # Check for possessive markers and pronouns
        for token in doc:
            if token.tag_ == "POS" or token.tag_ == "PRP$" or token.dep_ == "poss":
                self.logger.debug(f"Found possessive marker: '{token.text}'")
                return True
            # Check for unmarked possessives (e.g., "Rousseaus insights")
            if token.pos_ == "PROPN" and any(child.dep_ == "poss" for child in token.children):
                self.logger.debug(f"Found unmarked possessive: '{token.text}'")
                return True
        return False
        
    def _detect_quantified_np(self, doc) -> bool:
        """Detect if the NP contains a quantifier."""
        for token in doc:
            # Check for numeric modifiers
            if token.pos_ == "NUM" or token.dep_ == "nummod":
                if token.head.pos_ in ["NOUN", "PROPN"]:
                    self.logger.debug(f"Found numeric quantifier '{token.text}' for noun '{token.head.text}'")
                    return True
        return False
        
    def _detect_coordinate_np(self, doc) -> bool:
        """Detect if the NP contains coordination."""
        # Check for coordinating conjunctions
        for token in doc:
            if token.dep_ == "cc" and token.head.pos_ in ["NOUN", "PROPN", "ADJ"]:
                self.logger.debug(f"Found conjunction '{token.text}' joining elements")
                return True
            
            # Check for conjoined elements
            if token.dep_ == "conj" and token.head.pos_ in ["NOUN", "PROPN", "ADJ"]:
                self.logger.debug(f"Found coordinated element: '{token.text}'")
                return True
                
        return False
        
    def _detect_appositive_np(self, doc) -> bool:
        """Detect if the NP contains an appositive structure."""
        # Check for appositive dependency
        for token in doc:
            if token.dep_ == "appos":
                self.logger.debug(f"Found appositive: '{token.text}' explaining '{token.head.text}'")
                return True
        return False
        
    def _detect_relative_clause(self, doc) -> bool:
        """
        Detect if the NP contains a relative clause (standard or non-restrictive).
        """
        # Check for relative pronouns, relcl dependency, or acl with SCONJ
        relative_pronouns = [token for token in doc if token.tag_ in ["WDT", "WP", "WP$", "WRB"]]
        relcl_deps = [token for token in doc if token.dep_ == "relcl"]
        acl_with_sconj = any(
            token.dep_ == "acl" and any(child.dep_ == "mark" for child in token.children)
            for token in doc
        )
        
        # Check for non-restrictive relative clauses
        for token in doc:
            if token.dep_ == "relcl" and any(
                child.tag_ in ["WDT", "WP", "WP$", "WRB"] and child.dep_ in ["nsubj", "nsubjpass"]
                for child in token.children
            ):
                # Check if the clause is separated by commas
                prev_token = doc[token.i - 1] if token.i > 0 else None
                next_token = doc[token.i + 1] if token.i < len(doc) - 1 else None
                if (prev_token and prev_token.text == ",") or (next_token and next_token.text == ","):
                    return True
        
        return bool(relative_pronouns or relcl_deps or acl_with_sconj)
    
    def _detect_reduced_relative_clause(self, doc) -> bool:
        """
        Detect if the NP contains a reduced relative clause.
        """
        # Check for acl dependency without a relative pronoun
        for token in doc:
            if token.dep_ in ["acl", "relcl"] and token.head.pos_ in ["NOUN", "PROPN"]:
                # Make sure there's no relative pronoun
                no_rel_pron = not any(t.tag_ in ["WDT", "WP", "WP$", "WRB"] for t in doc)
                # Make sure it's not an infinitive (to + verb)
                no_infinitive = not any(child.tag_ == "TO" for child in token.children)
                # Make sure it's not part of a finite complement
                not_complement = not self._is_complement_taking_noun(token.head)
                
                if no_rel_pron and no_infinitive and not_complement:
                    self.logger.debug(f"Found reduced relative clause with acl: '{token.head.text} ... {token.text}'")
                    return True
        
        # Enhanced: Check for verb phrases modifying a noun without a relative pronoun
        for token in doc:
            if token.pos_ == "VERB" and token.head.pos_ in ["NOUN", "PROPN"]:
                # Ensure no relative pronoun is present
                no_rel_pron = not any(t.tag_ in ["WDT", "WP", "WP$", "WRB"] for t in doc)
                # Ensure it's not an infinitive or part of a finite complement
                no_infinitive = not any(child.tag_ == "TO" for child in token.children)
                not_complement = not self._is_complement_taking_noun(token.head)
                
                # Check if the verb has a subject (e.g., "I like") or is passive (e.g., "under construction")
                has_subject = any(child.dep_ == "nsubj" for child in token.children)
                is_passive = token.tag_ == "VBN" and any(child.dep_ == "auxpass" for child in token.children)
                
                if no_rel_pron and no_infinitive and not_complement and (has_subject or is_passive):
                    self.logger.debug(f"Found reduced relative clause with verb phrase: '{token.head.text} ... {token.text}'")
                    return True
        
        # New: Check for subject pronoun introducing a reduced relative clause
        for token in doc:
            if token.pos_ == "PRON" and token.dep_ == "nsubj":
                # Check if the pronoun is part of a clause modifying a noun
                head = token.head
                if head.pos_ == "VERB" and head.head.pos_ in ["NOUN", "PROPN"]:
                    # Ensure no relative pronoun is present
                    no_rel_pron = not any(t.tag_ in ["WDT", "WP", "WP$", "WRB"] for t in doc)
                    if no_rel_pron:
                        self.logger.debug(f"Found reduced relative clause with subject pronoun: '{token.head.head.text} {token.text} {head.text}'")
                        return True
                
        return False
    
    
        
    def _detect_finite_complement(self, doc) -> bool:
        """
        Detect if the NP contains a finite complement clause.
        
        Finite complement clauses:
        - Introduced by a complementizer (that, whether, if)
        - The complementizer has the 'mark' dependency
        - Often has a subject-verb structure following the complementizer
        """
        # Define common complement-taking nouns
        if not self._is_complement_taking_noun(doc):
            return False
        
        # Check for complementizers with mark dependency
        complementizers = [token for token in doc if token.text.lower() in ["that", "whether", "if"] 
                          and token.dep_ == "mark"]
        
        if complementizers:
            # Find the verb this complementizer is attached to
            for comp in complementizers:
                verb = comp.head if comp.head.pos_ == "VERB" else None
                
                if verb:
                    # Check if there's a subject for this verb
                    has_subject = any(child.dep_ == "nsubj" for child in verb.children)
                    
                    # Look for a noun before the complementizer that the clause modifies
                    for token in doc:
                        if (token.pos_ in ["NOUN", "PROPN"] and 
                            token.i < comp.i and 
                            self._is_complement_taking_noun(token)):
                            
                            # Verify there's a subject-verb structure after the complementizer
                            if has_subject:
                                self.logger.debug(f"Found finite complement with '{comp.text}': '{token.text} {comp.text}...'")
                                return True
        
        # Alternative pattern: acl dependency with mark
        for token in doc:
            if token.dep_ == "acl" and token.head.pos_ in ["NOUN", "PROPN"] and self._is_complement_taking_noun(token):
                # Check if there's a complementizer
                has_comp = any(child.dep_ == "mark" and 
                              child.text.lower() in ["that", "whether", "if"] 
                              for child in token.children)
                              
                if has_comp:
                    self.logger.debug(f"Found finite complement with acl: '{token.head.text} that...'")
                    return True
                    
        return False
    
    def _is_complement_taking_noun(self, token) -> bool:
        """
        Check if a token is a noun that typically takes a complement clause.
        Args:
            token: A spaCy token to check.
        Returns:
            bool: True if the token is a complement-taking noun, False otherwise.
        """
        # Ensure the input is a Token object
        if not hasattr(token, 'pos_'):
            return False
        
        # Check if the token is a noun and has a complement clause dependency
        if token.pos_ in ["NOUN", "PROPN"] and any(child.dep_ in ["ccomp", "acl"] for child in token.children):
            self.logger.debug(f"Found complement-taking noun: '{token.text}'")
            return True
        return False
        
    def _detect_nonfinite_complement(self, doc) -> bool:
        """
        Detect if the NP contains a nonfinite complement (infinitive or gerund).
        
        Nonfinite complement clauses:
        - Contain 'to' followed by a verb (infinitive)
        - Contain a preposition followed by a verb (gerund)
        - The infinitive or gerund is attached to a head noun
        """
        # Check for 'to' with infinitive verb
        for token in doc:
            if token.tag_ == "TO":
                # Find the verb this 'to' introduces
                verb = None
                for potential_verb in doc:
                    if (potential_verb.pos_ == "VERB" and 
                        potential_verb.i > token.i and
                        (potential_verb.head.i == token.i or token.head.i == potential_verb.i)):
                        verb = potential_verb
                        break
                
                if verb:
                    # Find the noun this infinitive modifies
                    for i in range(token.i):
                        if doc[i].pos_ in ["NOUN", "PROPN"]:
                            # Verify this is the actual head
                            is_head = False
                            # Check if the verb or 'to' is directly attached to this noun
                            if verb.head.i == doc[i].i or (verb.head == token and token.head.i == doc[i].i):
                                is_head = True
                            # Check if immediately preceding (e.g., "time to go")
                            elif i == token.i - 1:
                                is_head = True
                                
                            if is_head:
                                self.logger.debug(f"Found nonfinite complement: '{doc[i].text} to {verb.text}'")
                                return True
        
        # Check for prepositional phrases with gerunds (e.g., "the possibility of winning")
        for token in doc:
            if token.pos_ == "ADP" and token.dep_ == "prep" and token.head.pos_ in ["NOUN", "PROPN"]:
                # Check if the preposition has a gerund as its complement
                for child in token.children:
                    if child.pos_ == "VERB" and child.tag_ == "VBG":
                        self.logger.debug(f"Found nonfinite complement: '{token.head.text} {token.text} {child.text}'")
                        return True
        
        # Check for acl or relcl with 'to'
        for token in doc:
            if token.dep_ in ["acl", "relcl"] and token.pos_ == "VERB":
                # Look for 'to' among this verb's children
                to_child = next((child for child in token.children if child.tag_ == "TO"), None)
                
                if to_child and token.head.pos_ in ["NOUN", "PROPN"]:
                    self.logger.debug(f"Found nonfinite complement with acl/relcl: '{token.head.text} to {token.text}'")
                    return True
        
        return False 