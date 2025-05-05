"""
Analyzer module for detecting structural patterns in noun phrases.

This module implements refined detection methods based on a comprehensive analysis
of spaCy's dependency parsing patterns for various noun phrase structures.
"""

# --- Standard Logging Setup ---
import logging
logger = logging.getLogger(__name__)
# --- End Standard Logging ---

from typing import List
import spacy
from spacy.tokens import Doc, Span, Token


class ANPEAnalyzer:
    """Analyzes the structure of noun phrases with improved detection methods."""
    
    def __init__(self, nlp):
        """Initialize the analyzer with improved detection methods and logging."""
        logger.debug("Initializing ANPEAnalyzer")
        
        # Store the nlp object for potential future use (e.g., validation), but avoid re-parsing NPs
        if nlp is None:
             logger.error("ANPEAnalyzer initialized without a valid spaCy nlp object.")
             raise ValueError("ANPEAnalyzer requires a valid spaCy nlp object.")
        # self.nlp = nlp # We might not need this if we always get Spans
        logger.debug("ANPEAnalyzer initialized.")
    
    def analyze(self, nps_with_length: List[tuple]) -> List[tuple]:
        """
        Analyze the structure of noun phrases.
        
        Args:
            nps_with_length: List of (np, length) tuples
            
        Returns:
            List of (np, length, structures) tuples where structures is a list
            of structural patterns found in the NP
        """
        logger.info(f"Analyzing structure of {len(nps_with_length)} noun phrases")
        analyzed_nps = []
        
        for np, length in nps_with_length:
            logger.debug(f"Analyzing structure of '{np}'")
            structures = self.analyze_single_np(np)
            logger.debug(f"Found structures in '{np}': {structures}")
            analyzed_nps.append((np, length, structures))
            
        logger.info(f"Completed structural analysis for {len(analyzed_nps)} noun phrases")
        return analyzed_nps
    
    def analyze_single_np(self, span: Span) -> List[str]:
        """
        Analyze the structure of a single noun phrase Span using its existing tokens.
        
        Args:
            span: The spaCy Span object representing the noun phrase.
            
        Returns:
            List[str]: List of structure labels for this NP Span.
        """
        np_text = span.text # Get text for logging if needed
        logger.debug(f"Analyzing NP span: '{np_text}'")

        structures = []
        
        # First check basic structure types for simple NPs
        logger.debug(f"Checking if '{np_text}' is a pronoun")
        if self._detect_pronoun(span):
            structures.append("pronoun")
            logger.debug(f"Found pronoun: '{np_text}'")
            
        logger.debug(f"Checking if '{np_text}' is a standalone noun")
        if self._detect_standalone_noun(span):
            structures.append("standalone_noun")
            logger.debug(f"Found standalone noun: '{np_text}'")
        
        # Check each structural type with detailed logging
        logger.debug(f"Checking determiner NP for '{np_text}'")
        if self._detect_determiner_np(span):
            structures.append("determiner")
            logger.debug(f"Found determiner NP in '{np_text}'")
            
        logger.debug(f"Checking adjectival NP for '{np_text}'")
        if self._detect_adjectival_np(span):
            structures.append("adjectival_modifier")
            logger.debug(f"Found adjectival NP in '{np_text}'")
            
        logger.debug(f"Checking prepositional NP for '{np_text}'")
        if self._detect_prepositional_np(span):
            structures.append("prepositional_modifier")
            logger.debug(f"Found prepositional NP in '{np_text}'")
            
        logger.debug(f"Checking compound noun for '{np_text}'")
        if self._detect_compound_noun(span):
            structures.append("compound")
            logger.debug(f"Found compound noun in '{np_text}'")
            
        logger.debug(f"Checking possessive NP for '{np_text}'")
        if self._detect_possessive_np(span):
            structures.append("possessive")
            logger.debug(f"Found possessive NP in '{np_text}'")
            
        logger.debug(f"Checking quantified NP for '{np_text}'")
        if self._detect_quantified_np(span):
            structures.append("quantified")
            logger.debug(f"Found quantified NP in '{np_text}'")
            
        logger.debug(f"Checking coordinate NP for '{np_text}'")
        if self._detect_coordinate_np(span):
            structures.append("coordinated")
            logger.debug(f"Found coordinate NP in '{np_text}'")
            
        logger.debug(f"Checking appositive NP for '{np_text}'")
        if self._detect_appositive_np(span):
            structures.append("appositive")
            logger.debug(f"Found appositive NP in '{np_text}'")

        # Check for clausal structures
        # First check for nonfinite complements
        logger.debug(f"Checking nonfinite complement for '{np_text}'")
        if self._detect_nonfinite_complement(span):
            structures.append("nonfinite_complement")
            logger.debug(f"Found nonfinite complement in '{np_text}'")
        
        # Then check for finite complements - prioritize over relative clauses
        # for disambiguating 'that'
        logger.debug(f"Checking finite complement clause for '{np_text}'")
        if self._detect_finite_complement(span):
            structures.append("finite_complement")
            logger.debug(f"Found finite complement in '{np_text}'")
            
        # First check for relative clause
        logger.debug(f"Checking relative clause for '{np_text}'")
        if self._detect_relative_clause(span):
            structures.append("relative_clause")
            logger.debug(f"Found relative clause in '{np_text}'")
        
        # Then check for reduced relative clause
        logger.debug(f"Checking reduced relative clause for '{np_text}'")
        if self._detect_reduced_relative_clause(span):
            structures.append("reduced_relative_clause")
            logger.debug(f"Found reduced relative clause in '{np_text}'")
        
        # Apply post-processing validation to remove inconsistent patterns
        structures = self._validate_structures(structures, span)
            
        logger.debug(f"Analysis complete for '{np_text}': Structures={structures}")
        return structures
    
    def is_standalone_pronoun(self, np_span: Span) -> bool:
        """
        Checks if the Span represents a standalone pronoun.
        """
        # No need to re-parse, just call the detection method
        if len(np_span) == 1:
             return self._detect_pronoun(np_span)
        return False
            
    def _validate_structures(self, structures: List[str], span: Span) -> List[str]:
        """
        Apply validation rules to remove inconsistent or implausible combinations.
        
        Args:
            structures: List of identified structures
            span: The spaCy Span object representing the noun phrase.
            
        Returns:
            List[str]: Validated list of structures
        """
        np_text = span.text # For logging
        # Resolve conflict between finite complement and relative clause
        if "finite_complement" in structures and "relative_clause" in structures:
            # Use the existing span, no need to re-parse
            # doc = self.nlp(np_text) 
            
            # Check if 'that' is used as a relative pronoun (object or subject)
            that_as_rel_pronoun = any(token.text.lower() == "that" and 
                                      token.tag_ == "WDT" and
                                      token.dep_ in ["dobj", "nsubj"] 
                                      for token in span) # Use span here
            
            # Check if head noun is a complement-taking noun
            has_complement_noun = any(
                self._is_complement_taking_noun_within_span(token, span) # Use the span-aware version\
                for token in span if token.pos_ in ["NOUN", "PROPN"]\
            )
            
            # Resolution logic remains the same, just operating on span context
            if that_as_rel_pronoun and not has_complement_noun:
                # If 'that' is clearly a relative pronoun, favor relative clause
                logger.debug(f"Removing finite_complement in favor of relative_clause in '{np_text}'")
                structures.remove("finite_complement")
            elif has_complement_noun and not that_as_rel_pronoun:
                # If head noun strongly suggests complement, favor finite complement
                logger.debug(f"Removing relative_clause in favor of finite_complement in '{np_text}'")
                structures.remove("relative_clause")
            else:
                # Default to relative clause in ambiguous cases
                logger.debug(f"Defaulting to relative_clause in ambiguous case: '{np_text}'")
                structures.remove("finite_complement")
        
        # Ensure at least one structure label
        if not structures and len(span) > 0: # Use len(span)
            # If no structures detected but valid text, label as base_np
            structures.append("others")
            logger.debug(f"Assigned others label to '{np_text}' as fallback")
                
        return structures
            
    def _detect_pronoun(self, span: Span) -> bool:
        """Detect if the NP Span is a standalone pronoun."""
        if len(span) == 1 and span[0].pos_ == "PRON":
            logger.debug(f"Found pronoun: '{span[0].text}'")
            return True
        return False
        
    def _detect_standalone_noun(self, span: Span) -> bool:
        """Detect if the NP Span is a single noun with no modifiers."""
        if len(span) == 1 and span[0].pos_ in ["NOUN", "PROPN"]:
            # Check no modifiers are present
            logger.debug(f"Found standalone noun: '{span[0].text}'")
            return True
        return False
    
    def _detect_determiner_np(self, span: Span) -> bool:
        """Detect if the NP Span contains a determiner structure."""
        for token in span:
            if token.pos_ == "DET" and token.dep_ == "det" and token.head.pos_ in ["NOUN", "PROPN"]:
                # Ensure the head is also within the span
                if token.head in span:
                    logger.debug(f"Found determiner '{token.text}' modifying noun '{token.head.text}'")
                    return True
        return False
        
    def _detect_adjectival_np(self, span: Span) -> bool:
        """Detect if the NP Span contains an adjectival modifier."""
        for token in span:
            # Include adjectives (ADJ) and participles (VERB) used adjectivally (amod)
            if token.dep_ == "amod" and token.head.pos_ in ["NOUN", "PROPN"]:
                # Ensure the head is also within the span
                if token.head in span:
                    # Check if the modifier is tagged ADJ or VERB
                    if token.pos_ in ["ADJ", "VERB"]:
                         logger.debug(f"Found adjectival modifier ({token.pos_}) '{token.text}' modifying noun '{token.head.text}'")
                         return True
        return False
        
    def _detect_prepositional_np(self, span: Span) -> bool:
        """Detect if the NP Span contains a prepositional modifier."""
        for token in span:
            if token.pos_ == "ADP" and token.dep_ == "prep" and token.head.pos_ in ["NOUN", "PROPN"]:
                # Ensure the head is also within the span
                if token.head in span:
                    # Verify preposition has an object *within the span*
                    if any(child.dep_ == "pobj" and child in span for child in token.children):
                        logger.debug(f"Found preposition '{token.text}' modifying noun '{token.head.text}'")
                        return True
        return False
        
    def _detect_compound_noun(self, span: Span) -> bool:
        """Detect if the NP Span contains a compound noun."""
        for token in span:
            if token.dep_ == "compound" and token.head.pos_ in ["NOUN", "PROPN"]:
                # Ensure the head is also within the span
                if token.head in span:
                    logger.debug(f"Found compound noun: '{token.text} {token.head.text}'")
                    return True
        return False
        
    def _detect_possessive_np(self, span: Span) -> bool:
        """Detect if the NP Span contains a possessive structure."""
        # Check for possessive markers and pronouns
        for token in span:
            # Check for 's case marker: token is 's, head is possessor (X), head.head is possessed (Y)
            # Ensure possessor (X) and possessed (Y) are within the span
            if token.tag_ == "POS" and token.dep_ == "case":
                possessor = token.head
                if possessor in span and hasattr(possessor, 'head') and possessor.head in span:
                    logger.debug(f"Found possessive ('s): '{possessor.text}'s {possessor.head.text}' within span '{span.text}'")
                    return True
                    
            # Check for possessive pronoun: token is PRP$, head is possessed (Y)
            # Ensure possessed (Y) is within the span
            if token.tag_ == "PRP$" and token.dep_ == "poss":
                 if token.head in span:
                     logger.debug(f"Found possessive (pronoun): '{token.text} {token.head.text}' within span '{span.text}'")
                     return True
                     
            # Check for 'poss' dependency: token is possessor (X), head is possessed (Y)
            # Ensure possessed (Y) is within the span
            if token.dep_ == "poss": # Exclude PRP$ handled above
                 if token.head in span:
                     logger.debug(f"Found possessive (poss dep): '{token.text} {token.head.text}' within span '{span.text}'")
                     return True
                     
            # Check for unmarked possessive (e.g., "Rousseau insights") 
            # token is head (Y), child has poss dep and is possessor (X)
            # Ensure possessor (X) is within the span
            if token.pos_ in ["NOUN", "PROPN"] and any(
                child.dep_ == "poss" and child in span for child in token.children
            ):
                poss_child = next((child for child in token.children if child.dep_ == "poss" and child in span), None)
                if poss_child: # Should always be true if any() passed
                    logger.debug(f"Found possessive (unmarked): '{poss_child.text} {token.text}' within span '{span.text}'")
                    return True
                    
        return False
        
    def _detect_quantified_np(self, span: Span) -> bool:
        """Detect if the NP Span contains a quantifier."""
        for token in span:
            # Check for numeric modifiers
            if (token.pos_ == "NUM" or token.dep_ == "nummod") and token.head.pos_ in ["NOUN", "PROPN"]:
                # Ensure the head is also within the span
                if token.head in span:
                    logger.debug(f"Found numeric quantifier '{token.text}' for noun '{token.head.text}'")
                    return True
        return False
        
    def _detect_coordinate_np(self, span: Span) -> bool:
        """Detect if the NP Span contains coordination."""
        # Check for coordinating conjunctions linking elements within the span
        has_cc = False
        has_conj = False
        for token in span:
            # Check for 'cc' where the head is within the span
            if token.dep_ == "cc" and token.head in span:
                # Check if the element being coordinated is also in the span
                conjunct = next((sibling for sibling in token.head.children if sibling.dep_ == "conj" and sibling in span), None)
                if conjunct:
                    logger.debug(f"Found conjunction '{token.text}' joining '{token.head.text}' and '{conjunct.text}'")
                    has_cc = True
            
            # Check for 'conj' where the head is within the span
            if token.dep_ == "conj" and token.head in span:
                 # Check if the coordinating conjunction is also in the span
                conjunction = next((sibling for sibling in token.head.children if sibling.dep_ == "cc" and sibling in span), None)
                if conjunction:
                    logger.debug(f"Found coordinated element '{token.text}' linked to '{token.head.text}' via '{conjunction.text}'")
                    has_conj = True

        return has_cc and has_conj # Require both conjunction and conjunct within the span
        
    def _detect_appositive_np(self, span: Span) -> bool:
        """Detect if the NP Span contains an appositive structure."""
        # Check for appositive dependency where both head and modifier are within the span
        for token in span:
            if token.dep_ == "appos" and token.head in span:
                # Check if the head itself is the root of the span OR directly linked to the span's root
                # span_root = span.root
                # if token.head == span_root or token.head.head == span_root:
                #      logger.debug(f"Found appositive: '{token.text}' explaining '{token.head.text}' linked near span root '{span_root.text}'")
                #      return True
                # else:
                #      # Added check: Log if appos is found but head is not near span root
                #      logger.debug(f"Found appos relation ('{token.text}' -> '{token.head.text}') but head '{token.head.text}' not near span root '{span_root.text}'.")
                # Simpler check: if appos dependency and head are in span, consider it appositive NP structure.
                logger.debug(f"Found appositive: '{token.text}' explaining '{token.head.text}'")
                return True
        return False
        
    def _detect_relative_clause(self, span: Span) -> bool:
        """
        Detect if the NP Span contains a relative clause (standard or reduced).
        """
        # Check 1: Find relative pronouns (WDT, WP, WP$, WRB) within the span
        relative_pronouns_in_span = [token for token in span if token.tag_ in ["WDT", "WP", "WP$", "WRB"]] 
        for pron in relative_pronouns_in_span:
            # If pronoun found, check its head (the verb of the clause) is also in the span
            if hasattr(pron, 'head') and pron.head in span:
                logger.debug(f"Found relative clause (pronoun-based): '{pron.text} ...' within span '{span.text}'")
                return True
                
        # Check 2: Find verbs with 'relcl' dependency within the span
        relcl_verbs_in_span = [token for token in span if token.dep_ == "relcl"] 
        for verb in relcl_verbs_in_span:
            # If relcl verb found, check its head (the noun being modified) is also in the span
            if hasattr(verb, 'head') and verb.head in span:
                logger.debug(f"Found relative clause (relcl-dep): '{verb.head.text} ... {verb.text}' within span '{span.text}'")
                return True
                
        # If neither check passes, it's not a standard relative clause within this span
        return False
    
    def _detect_reduced_relative_clause(self, span: Span) -> bool:
        """
        Detect if the NP Span contains a reduced relative clause.
        Operates on the tokens within the provided Span.
        """
        # Check for acl dependency modifying a noun/propn, where both are in the span
        for token in span:
            # Focus on 'acl' for reduced relatives
            if token.dep_ == "acl" and token.head.pos_ in ["NOUN", "PROPN"] and token.head in span:
                # Ensure no explicit relative pronouns are also present in the span
                has_rel_pronoun = any(t.tag_ in ["WDT", "WP", "WP$", "WRB"] for t in span)
                if has_rel_pronoun:
                    continue # Skip if explicit pronoun found, it's likely a standard relative clause

                # Basic check: if the main verb of the clause is in the span, consider it valid
                # The token *is* the main verb here, and it's already confirmed to be in the span by the loop
                # We also implicitly check it's reduced because _detect_relative_clause runs first
                # and would catch standard relatives with explicit markers/relcl dep.
                logger.debug(f"Found reduced relative clause (acl-dep, no rel pron): '{token.head.text} ... {token.text}' within span '{span.text}'")
                return True
        return False

    def _detect_finite_complement(self, span: Span) -> bool:
        """
        Detect if the NP Span contains a finite complement clause.
        Operates on the tokens within the provided Span.
        """
        # Check for complementizers with mark dependency within the span
        complementizers = [token for token in span if token.text.lower() in ["that", "whether", "if"] 
                          and token.dep_ == "mark" and token.head in span]
        
        found_complement = False
        if complementizers:
            for comp in complementizers:
                verb = comp.head if comp.head.pos_ == "VERB" else None
                
                if verb: # Verb must be within the span
                    # Check if there's a subject for this verb within the span
                    has_subject = any(child.dep_ == "nsubj" and child in span for child in verb.children)
                    
                    # Look for a noun *within the span* before the complementizer that the clause modifies
                    potential_head_nouns = [token for token in span \
                                            if token.pos_ in ["NOUN", "PROPN"] and token.i < comp.i]
                    
                    for head_noun in potential_head_nouns:
                        if self._is_complement_taking_noun_within_span(head_noun, span):
                            # Verify there's a subject-verb structure *within the span* after the complementizer
                            if has_subject:
                                logger.debug(f"Found finite complement with '{comp.text}': '{head_noun.text} {comp.text}...' within span '{span.text}'")
                                found_complement = True
                                break # Found for this complementizer
                    if found_complement:
                        break # Found for the span

        # Alternative pattern: acl dependency with mark, all within span
        if not found_complement:
            for token in span:
                 # Check for a verb with acl dependency whose head is a noun/propn in the span
                if token.dep_ == "acl" and token.pos_ == "VERB" and \
                   token.head.pos_ in ["NOUN", "PROPN"] and token.head in span:
                    
                    # Check if the head noun takes complements within the span context
                    if self._is_complement_taking_noun_within_span(token.head, span):
                        # Check if there's a complementizer mark within the span attached to this verb
                        has_comp = any(child.dep_ == "mark" and \
                                      child.text.lower() in ["that", "whether", "if"] and \
                                      child in span \
                                      for child in token.children)
                                      
                        if has_comp:
                            logger.debug(f"Found finite complement (acl with mark): '{token.head.text} that...' within span '{span.text}'")
                            found_complement = True
                            break # Found

        return found_complement

    def _is_complement_taking_noun_within_span(self, token: Token, span: Span) -> bool:
        """
        Check if a token (noun) within a specific span appears to take a complement clause
        based on its children *within that span*.
        """
        # Check if the token is a noun/propn within the span
        if token not in span or token.pos_ not in ["NOUN", "PROPN"]:
            return False
            
        # Check if it has a ccomp or acl child *that is also within the span*
        takes_complement = any(child.dep_ in ["ccomp", "acl"] and child in span for child in token.children)
        if takes_complement:
             logger.debug(f"Noun '{token.text}' takes complement based on children within span '{span.text}'")
        return takes_complement
        
    def _detect_nonfinite_complement(self, span: Span) -> bool:
        """
        Detect if the NP Span contains a nonfinite complement (infinitive or gerund).
        Operates on the tokens within the provided Span.
        """
        # Check for 'to' + VERB infinitive structure within the span
        for token in span:
            if token.tag_ == "TO" and token.head.pos_ == "VERB" and token.head in span:
                 # Check if the verb's head (or 'to's head) is a noun/propn within the span
                verb = token.head
                ultimate_head = verb.head # Could be the noun or 'to'
                if ultimate_head == token: # If verb attached to 'to'
                    ultimate_head = token.head # Get the head of 'to'
                
                if ultimate_head in span and ultimate_head.pos_ in ["NOUN", "PROPN"]:
                     logger.debug(f"Found nonfinite complement (to + verb): '{ultimate_head.text} to {verb.text}' within span '{span.text}'")
                     return True

        # Check for prepositional phrases with gerunds within the span
        for token in span:
            if token.pos_ == "ADP" and token.dep_ == "prep" and token.head.pos_ in ["NOUN", "PROPN"] and token.head in span:
                # Check if the preposition has a gerund object within the span
                gerund_child = next((child for child in token.children \
                                     if child.pos_ == "VERB" and child.tag_ == "VBG" and \
                                     child.dep_ == "pcomp" and child in span), None) # pcomp is more specific
                if gerund_child:
                     logger.debug(f"Found nonfinite complement (prep + gerund): '{token.head.text} {token.text} {gerund_child.text}' within span '{span.text}'")
                     return True

        # Check for acl/relcl verb with a 'to' child, all within the span
        for token in span:
            if token.dep_ in ["acl", "relcl"] and token.pos_ == "VERB" and token.head in span and token.head.pos_ in ["NOUN", "PROPN"]:
                # Look for 'to' among this verb's children that is also in the span
                to_child = next((child for child in token.children if child.tag_ == "TO" and child in span), None)
                if to_child:
                    logger.debug(f"Found nonfinite complement (acl/relcl with to): '{token.head.text} to {token.text}' within span '{span.text}'")
                    return True
        
        return False 