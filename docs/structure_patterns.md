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
    - [Helper: Complement-Taking Noun (`_is_complement_taking_noun`)](#helper-complement-taking-noun-_is_complement_taking_noun)

## Basic Structures

### Pronoun (`pronoun`)
- **Detection Logic**: Checks if the NP consists of exactly one token and that token's Part-of-Speech (POS) tag is `PRON`.
- **Explanation**: Identifies NPs that are just a single pronoun (e.g., "it", "they", "he").

### Standalone Noun (`standalone_noun`)
- **Detection Logic**: Checks if the NP consists of exactly one token and that token's POS tag is `NOUN` or `PROPN` (Proper Noun).
- **Explanation**: Identifies NPs that are just a single common or proper noun without any modifiers (e.g., "cat", "John").

## Modifier Structures

### Determiner (`determiner`)
- **Detection Logic**: Looks for a token within the NP that has the POS tag `DET` (Determiner) and the dependency relation `det`, modifying a head token tagged as `NOUN` or `PROPN`.
- **Explanation**: Identifies NPs containing a determiner (e.g., "the book", "a car", "this idea").

### Adjectival Modifier (`adjectival_modifier`)
- **Detection Logic**: Looks for a token tagged as `ADJ` (Adjective) with the dependency relation `amod` (adjectival modifier), modifying a head token tagged as `NOUN` or `PROPN`.
- **Explanation**: Identifies NPs where a noun is modified by an adjective (e.g., "red ball", "beautiful scenery").

### Prepositional Modifier (`prepositional_modifier`)
- **Detection Logic**: Looks for a token tagged as `ADP` (Adposition/Preposition) with the dependency relation `prep`, modifying a head token tagged as `NOUN` or `PROPN`. It also verifies that the preposition has a child token with the dependency `pobj` (object of preposition).
- **Explanation**: Identifies NPs containing a prepositional phrase that modifies the noun (e.g., "man in the hat", "box of chocolates").

### Compound Noun (`compound`)
- **Detection Logic**: Looks for a token with the dependency relation `compound`, modifying a head token tagged as `NOUN` or `PROPN`.
- **Explanation**: Identifies NPs formed by multiple nouns working together as a single unit (e.g., "coffee shop", "computer science").

### Possessive (`possessive`)
- **Detection Logic**: Detects possessive structures by looking for:
    1. Tokens with the fine-grained tag `POS` (possessive ending, like 's)
    2. Tokens with the fine-grained tag `PRP$` (possessive pronoun, like "my", "his")
    3. Tokens with the dependency relation `poss`.
    4. Proper nouns (`PROPN`) that have a child with the `poss` dependency (handles unmarked possessives like "James book").
- **Explanation**: Identifies NPs indicating possession (e.g., "John's car", "her idea", "its tail").

### Quantified (`quantified`)
- **Detection Logic**: Looks for tokens tagged as `NUM` (Numeral) or having the dependency relation `nummod` (numeric modifier), modifying a head token tagged as `NOUN` or `PROPN`.
- **Explanation**: Identifies NPs that include a number or quantifier (e.g., "three dogs", "several people").

## Complex Structures

### Coordinated (`coordinated`)
- **Detection Logic**: Detects coordination by looking for:
    1. Tokens with the dependency relation `cc` (coordinating conjunction) whose head is a `NOUN`, `PROPN`, or `ADJ`.
    2. Tokens with the dependency relation `conj` (conjunct) whose head is a `NOUN`, `PROPN`, or `ADJ`.
- **Explanation**: Identifies NPs containing elements joined by conjunctions like "and", "or" (e.g., "cats and dogs", "the red or blue car").

### Appositive (`appositive`)
- **Detection Logic**: Looks for a token with the dependency relation `appos` (appositional modifier).
- **Explanation**: Identifies NPs where one NP element renames or explains another (e.g., "John, my brother", "Paris, the capital of France").

### Relative Clause (`relative_clause`)
- **Detection Logic**: Detects relative clauses (both standard and non-restrictive) by looking for:
    1. Tokens tagged as relative pronouns (`WDT`, `WP`, `WP$`, `WRB`).
    2. Tokens with the dependency relation `relcl` (relative clause modifier).
    3. Tokens with the dependency `acl` (adjectival clause) that have a child with the `mark` dependency (subordinating conjunction).
    4. Specifically checks for non-restrictive clauses indicated by `relcl` dependency potentially separated by commas.
- **Explanation**: Identifies NPs modified by a clause that provides additional information, often starting with "who", "which", "that", etc. (e.g., "the man who lives next door", "the book that I read").

### Reduced Relative Clause (`reduced_relative_clause`)
- **Detection Logic**: Detects relative clauses where the relative pronoun is omitted. It looks for:
    1. Tokens with `acl` or `relcl` dependency modifying a `NOUN` or `PROPN`, ensuring no relative pronoun (`WDT`, `WP`, etc.) is present, it's not an infinitive (`TO` tag), and the head noun isn't a complement-taking noun.
    2. Verb phrases (`VERB` POS) modifying a `NOUN` or `PROPN`, ensuring no relative pronoun, no infinitive, not a complement-taking noun head, and the verb either has a subject (`nsubj`) or is passive (`VBN` tag with `auxpass` child).
    3. A subject pronoun (`PRON` POS, `nsubj` dep) whose head is a verb, and *that* verb's head is a `NOUN` or `PROPN`, ensuring no relative pronoun is present in the NP.
- **Explanation**: Identifies NPs modified by a clause where the relative pronoun and potentially the auxiliary verb are omitted (e.g., "the book written by John" instead of "the book that was written by John", "the man talking to the police").

### Finite Complement (`finite_complement`)
- **Detection Logic**: Detects finite complement clauses attached to nouns. Requires the presence of a complement-taking noun (identified by `_is_complement_taking_noun` helper). Looks for:
    1. Complementizers ("that", "whether", "if") with the `mark` dependency, attached to a verb which has a subject (`nsubj`), and modifying a preceding complement-taking noun.
    2. Tokens with `acl` dependency modifying a complement-taking noun, where the `acl` token has a child with the `mark` dependency ("that", "whether", "if").
- **Explanation**: Identifies NPs containing a clause that completes the meaning of a noun, often expressing a fact, possibility, or belief (e.g., "the idea that he might leave", "the question whether it's true").

### Nonfinite Complement (`nonfinite_complement`)
- **Detection Logic**: Detects nonfinite complements (infinitives or gerunds) attached to nouns. Looks for:
    1. An infinitive marker (`TO` tag) followed by a verb (`VERB` POS), where the "to" or the verb is attached to or immediately follows a preceding `NOUN` or `PROPN`.
    2. A preposition (`ADP` POS, `prep` dep) modifying a `NOUN` or `PROPN`, where the preposition's object (`pobj`) is a gerund (`VERB` POS, `VBG` tag).
    3. A verb (`VERB` POS) with `acl` or `relcl` dependency modifying a `NOUN` or `PROPN`, where the verb has a child tagged `TO`.
- **Explanation**: Identifies NPs containing an infinitive ("to" + verb) or gerund (-ing form used as noun) phrase that completes the meaning of the noun (e.g., "a chance to win", "the possibility of leaving", "time to go").

## Validation and Fallback

### Structure Validation (`_validate_structures`)
- **Conflict Resolution**: Specifically addresses conflicts between `finite_complement` and `relative_clause` when both are detected. It prioritizes `relative_clause` if "that" is clearly used as a relative pronoun (`WDT` tag, `dobj`/`nsubj` dep) and the head noun isn't typically complement-taking. It prioritizes `finite_complement` if the head noun *is* complement-taking and "that" isn't clearly a relative pronoun. Defaults to `relative_clause` in ambiguous cases.
- **Fallback Label**: If no specific structure is detected for a non-empty NP, it assigns the label `others`.
- **Explanation**: Cleans up the detected labels by resolving known ambiguities (especially around the word "that") and ensures every NP gets at least one label, using "others" if no specific pattern matches.

### Helper: Complement-Taking Noun (`_is_complement_taking_noun`)
- **Detection Logic**: Checks if a token is a `NOUN` or `PROPN` and has a child with a dependency relation indicating a clausal complement (`ccomp` or `acl`).
- **Explanation**: Identifies nouns that commonly introduce complement clauses (like "fact", "idea", "belief", "possibility"). Used as a prerequisite for detecting `finite_complement` structures.

---

This document serves as a comprehensive guide to the patterns and logic used in the `ANPEAnalyzer` for detecting noun phrase structures. It reflects all current logic implemented in the analyzer.