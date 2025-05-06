# NP Structure Detection Patterns

This document summarizes the patterns used by `ANPEAnalyzer` for detecting various noun phrase (NP) structures based on spaCy's dependency parsing.

> **Disclaimer**: While we strive for accuracy in detecting noun phrase structures, please keep in mind that the results may vary due to **the limitations of spaCy's dependency parsing** and the **complex, sometimes ambiguous nature of English syntax**. This labeling system is a helpful tool, but it **may not be 100% accurate**. If you find any inaccuracies or have suggestions for additional structures to detect, please feel free to raise an issue on GitHub. I'll do my best to address them!

## Table of Contents

1.  [Pronoun (`pronoun`)](#pronoun-pronoun)
2.  [Standalone Noun (`standalone_noun`)](#standalone-noun-standalone_noun)
3.  [Determiner (`determiner`)](#determiner-determiner)
4.  [Adjectival Modifier (`adjectival_modifier`)](#adjectival-modifier-adjectival_modifier)
5.  [Prepositional Modifier (`prepositional_modifier`)](#prepositional-modifier-prepositional_modifier)
6.  [Compound Noun (`compound`)](#compound-noun-compound)
7.  [Possessive (`possessive`)](#possessive-possessive)
8.  [Quantified (`quantified`)](#quantified-quantified)
9.  [Coordinated (`coordinated`)](#coordinated-coordinated)
10. [Appositive (`appositive`)](#appositive-appositive)
11. [Relative Clause (`relative_clause`)](#relative-clause-relative_clause)
12. [Reduced Relative Clause (`reduced_relative_clause`)](#reduced-relative-clause-reduced_relative_clause)
13. [Finite Complement (`finite_complement`)](#finite-complement-finite_complement)
14. [Nonfinite Complement (`nonfinite_complement`)](#nonfinite-complement-nonfinite_complement)
15. [Validation and Fallback](#validation-and-fallback)
    - [Structure Validation (`_validate_structures`)](#structure-validation-_validate_structures)
    - [Helper: Complement-Taking Noun (`_is_complement_taking_noun_within_span`)](#helper-complement-taking-noun-_is_complement_taking_noun_within_span)

## Basic Structures

### Pronoun (`pronoun`)
- **Detection Logic**: Checks if the NP `Span` consists of exactly one token and that token's `pos_` is `PRON`.
- **Explanation**: Identifies NPs that are just a single pronoun (e.g., "it", "they", "he").

### Standalone Noun (`standalone_noun`)
- **Detection Logic**: Checks if the NP `Span` consists of exactly one token and that token's `pos_` is `NOUN` or `PROPN`.
- **Explanation**: Identifies NPs that are just a single common or proper noun without any modifiers (e.g., "cat", "John").

## Modifier Structures

### Determiner (`determiner`)
- **Detection Logic**: Looks for a token within the NP `Span` that has `pos_ == 'DET'` and `dep_ == 'det'`, and whose `head` token (which must have `pos_` in `['NOUN', 'PROPN']`) is also within the `Span`.
- **Explanation**: Identifies NPs containing a determiner (e.g., "the book", "a car", "this idea").

### Adjectival Modifier (`adjectival_modifier`)
- **Detection Logic**: Looks for a token within the NP `Span` that has `dep_ == 'amod'` and `pos_` in `['ADJ', 'VERB']`, and whose `head` token (which must have `pos_` in `['NOUN', 'PROPN']`) is also within the `Span`.
- **Explanation**: Identifies NPs where a noun is modified by an adjective (e.g., "red ball", "beautiful scenery").

### Prepositional Modifier (`prepositional_modifier`)
- **Detection Logic**: Looks for a token within the NP `Span` that has `pos_ == 'ADP'` and `dep_ == 'prep'`, whose `head` token (which must have `pos_` in `['NOUN', 'PROPN']`) is also within the `Span`, and which has at least one child token with `dep_ == 'pobj'` that is also within the `Span`.
- **Explanation**: Identifies NPs containing a prepositional phrase that modifies the noun (e.g., "man in the hat", "box of chocolates").

### Compound Noun (`compound`)
- **Detection Logic**: Looks for a token within the NP `Span` that has `dep_ == 'compound'`, and whose `head` token (which must have `pos_` in `['NOUN', 'PROPN']`) is also within the `Span`.
- **Explanation**: Identifies NPs formed by multiple nouns working together as a single unit (e.g., "coffee shop", "computer science").

### Possessive (`possessive`)
- **Detection Logic**: Detects possessive structures within the NP `Span` by looking for any of the following conditions where all involved tokens are within the `Span`:
    1. A token with `tag_ == 'POS'` and `dep_ == 'case'`, whose `head` (the possessor) and the `head`'s `head` (the possessed) are both within the `Span`.
    2. A token with `tag_ == 'PRP$'` (possessive pronoun) and `dep_ == 'poss'`, whose `head` (the possessed) is within the `Span`.
    3. A token with `dep_ == 'poss'`, whose `head` (the possessed) is within the `Span`.
    4. A token (the possessed) which has a child token with `dep_ == 'poss'` (the possessor), where both are within the `Span`.
- **Explanation**: Identifies NPs indicating possession (e.g., "John's car", "her idea", "its tail").

### Quantified (`quantified`)
- **Detection Logic**: Looks for a token within the NP `Span` that has `pos_ == 'NUM'` or `dep_ == 'nummod'`, and whose `head` token (which must have `pos_` in `['NOUN', 'PROPN']`) is also within the `Span`.
- **Explanation**: Identifies NPs that include a number or quantifier (e.g., "three dogs", "several people").

## Complex Structures

### Coordinated (`coordinated`)
- **Detection Logic**: Detects coordination within the NP `Span` by requiring both:
    1. A token with `dep_ == 'cc'` whose `head` is within the `Span` and has a conjunct child (`dep_ == 'conj'`) also within the `Span`.
    2. A token with `dep_ == 'conj'` whose `head` is within the `Span` and has a conjunction child (`dep_ == 'cc'`) also within the `Span`.
- **Explanation**: Identifies NPs containing elements joined by conjunctions like "and", "or" (e.g., "cats and dogs", "the red or blue car").

### Appositive (`appositive`)
- **Detection Logic**: Looks for a token within the NP `Span` that has `dep_ == 'appos'`, and whose `head` token is also within the `Span`.
- **Explanation**: Identifies NPs where one NP element renames or explains another (e.g., "John, my brother", "Paris, the capital of France").

### Relative Clause (`relative_clause`)
- **Detection Logic**: Detects relative clauses within the NP `Span` by looking for either:
    1. Tokens tagged as relative pronouns (`WDT`, `WP`, `WP$`, `WRB`) whose `head` (verb of the clause) is also within the `Span`.
    2. Tokens (verbs) with the dependency relation `relcl` whose `head` (noun being modified) is also within the `Span`.
- **Explanation**: Identifies NPs modified by a clause that provides additional information, often starting with "who", "which", "that", etc. (e.g., "the man who lives next door", "the book that I read"). Note: Disambiguation with finite complements for 'that' clauses occurs during validation.

### Reduced Relative Clause (`reduced_relative_clause`)
- **Detection Logic**: Detects relative clauses where the relative pronoun is omitted. Looks for a token within the NP `Span` that has `dep_ == 'acl'`, whose `head` (with `pos_` in `['NOUN', 'PROPN']`) is also within the `Span`, and ensures no explicit relative pronoun (`WDT`, `WP`, etc.) is present within the entire `Span`.
- **Explanation**: Identifies NPs modified by a clause where the relative pronoun and potentially the auxiliary verb are omitted (e.g., "the book written by John", "the man talking to the police"). Assumes standard relative clauses are checked first.

### Finite Complement (`finite_complement`)
- **Detection Logic**: Detects finite complement clauses attached to nouns within the NP `Span`. Looks for either:
    1. A complementizer token ("that", "whether", "if") with `dep_ == 'mark'`, whose `head` (verb) is within the `Span`. It then requires a preceding noun within the `Span` identified as complement-taking (using `_is_complement_taking_noun_within_span`) and confirms the verb has a subject (`dep_ == 'nsubj'`) also within the `Span`.
    2. A verb token with `dep_ == 'acl'`, whose `head` (noun) is within the `Span` and identified as complement-taking. Additionally, requires this verb to have a complementizer child (`dep_ == 'mark'`, text "that"/"whether"/"if") also within the `Span`.
- **Explanation**: Identifies NPs containing a clause that completes the meaning of a noun, often expressing a fact, possibility, or belief (e.g., "the idea that he might leave", "the question whether it's true"). Requires specific noun types.

### Nonfinite Complement (`nonfinite_complement`)
- **Detection Logic**: Detects nonfinite complements (infinitives or gerunds) attached to nouns within the NP `Span`. Looks for any of:
    1. A token with `tag_ == 'TO'` whose `head` (verb) is within the `Span`. The ultimate head (either the verb's head or the 'TO' token's head) must also be within the `Span` and have `pos_` in `['NOUN', 'PROPN']`.
    2. A preposition token (`pos_ == 'ADP'`, `dep_ == 'prep'`) whose `head` (noun) is within the `Span`. The preposition must have a gerund child (`pos_ == 'VERB'`, `tag_ == 'VBG'`, `dep_ == 'pcomp'`) that is also within the `Span`.
    3. A verb token (`pos_ == 'VERB'`) with `dep_` in `['acl', 'relcl']`, whose `head` (noun) is within the `Span`. The verb must have a child token with `tag_ == 'TO'` that is also within the `Span`.
- **Explanation**: Identifies NPs containing an infinitive ("to" + verb) or gerund (-ing form used as noun) phrase that completes the meaning of the noun (e.g., "a chance to win", "the possibility of leaving", "time to go").

## Validation and Fallback

### Structure Validation (`_validate_structures`)
- **Conflict Resolution**: Specifically addresses conflicts between `finite_complement` and `relative_clause` when both are detected for the same `Span`. It prioritizes `relative_clause` if a token like "that" within the span is tagged `WDT` (relative pronoun) with dependency `dobj` or `nsubj`, and the head noun isn't complement-taking. It prioritizes `finite_complement` if the head noun *is* complement-taking (checked within the span) and "that" isn't clearly a relative pronoun based on its tag/dep. Defaults to `relative_clause` in ambiguous cases.
- **Fallback Label**: If no specific structure is detected for a non-empty `Span`, it assigns the label `others`.
- **Explanation**: Cleans up the detected labels by resolving known ambiguities (especially around the word "that") based on token-level evidence within the span, and ensures every NP gets at least one label, using "others" if no specific pattern matches.

### Helper: Complement-Taking Noun (`_is_complement_taking_noun_within_span`)
- **Detection Logic**: Checks if a given `NOUN` or `PROPN` token within a specific `Span` has a child token with dependency `ccomp` or `acl` that is *also* within that same `Span`.
- **Explanation**: Identifies nouns within the context of the current NP span that appear to introduce complement clauses (like "fact", "idea", "belief", "possibility"). Used as a prerequisite for detecting `finite_complement` structures accurately within the span.