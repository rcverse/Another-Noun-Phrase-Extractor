# NP Structure Detection Patterns

This document summarizes the patterns for detecting various noun phrase structures based on spaCy's dependency parsing.

> **Disclaimer**: While we strive for accuracy in detecting noun phrase structures, please keep in mind that the results may vary due to **the limitations of spaCy's dependency parsing** and the **complex, sometimes ambiguous nature of English syntax**. This labeling system is a helpful tool, but it **may not be 100% accurate**. If you find any inaccuracies or have suggestions for additional structures to detect, please feel free to raise an issue on GitHub. I'll do my best to address them!

## Table of Contents

1. [Determiner Structure](#1-determiner-structure)
2. [Adjectival Modifier Structure](#2-adjectival-modifier-structure)
3. [Prepositional Modifier Structure](#3-prepositional-modifier-structure)
4. [Compound Noun Structure](#4-compound-noun-structure)
5. [Possessive Structure](#5-possessive-structure)
6. [Quantified Structure](#6-quantified-structure)
7. [Coordinated Structure](#7-coordinated-structure)
8. [Appositive Structure](#8-appositive-structure)
9. [Standard Relative Clause Structure](#9-standard-relative-clause-structure)
10. [Reduced Relative Clause Structure](#10-reduced-relative-clause-structure)
11. [Non-restrictive Relative Clause Structure](#11-non-restrictive-relative-clause-structure)
12. [Finite Complement Structure](#12-finite-complement-structure)
13. [Nonfinite Complement Structure](#13-nonfinite-complement-structure)

## 1. Determiner Structure

### Definition
A noun phrase with a determiner (the, a, an, this, that, these, those, etc.) preceding the head noun.

### Reliable Patterns
- Token with `pos="DET"` and `dep="det"`
- Determiner is a left child of a noun with `pos="NOUN"` or `pos="PROPN"`

### Logic
We check each token in the document to see if it is a determiner and if it modifies a noun.

---

## 2. Adjectival Modifier Structure

### Definition
A noun phrase with one or more adjectives modifying the head noun.

### Reliable Patterns
- Token with `pos="ADJ"` and typically `dep="amod"`
- Adjective is a left child of a noun with `pos="NOUN"` or `pos="PROPN"`

### Logic
We identify adjectives that modify nouns by checking their part of speech and dependency relations.

---

## 3. Prepositional Modifier Structure

### Definition
A noun phrase with a prepositional phrase modifying the head noun.

### Reliable Patterns
- Token with `pos="ADP"` and `dep="prep"` as a child of a noun
- Preposition has a child with `dep="pobj"` (prepositional object)

### Logic
We look for prepositions that modify nouns and ensure they have an object to confirm their role as modifiers.

---

## 4. Compound Noun Structure

### Definition
A noun phrase where multiple nouns combine to form a single conceptual unit.

### Reliable Patterns
- Token with `dep="compound"` and `pos="NOUN"` or `pos="PROPN"` modifying another noun
- Two adjacent nouns where the first modifies the second

### Logic
We check for compound dependencies and adjacent nouns to identify compound structures.

---

## 5. Possessive Structure

### Definition
A noun phrase where a possessor is indicated, either by an explicit possessive marker (apostrophe + s) or a possessive pronoun.

### Reliable Patterns
- Token with `tag="POS"` (possessive marker)
- Token with `tag="PRP$"` (possessive pronoun: my, your, his, her, etc.)
- Token with `dep="poss"` (possessive modifier)

### Logic
We identify possessive structures by checking for possessive markers, pronouns, and unmarked possessives.

---

## 6. Quantified Structure

### Definition
A noun phrase with a quantifier (number, quantity word) modifying the head noun.

### Reliable Patterns
- Token with `pos="NUM"` or `dep="nummod"` modifying a noun
- Quantifying determiners: many, few, several, etc.

### Logic
We look for numeric modifiers and quantifying determiners that modify nouns.

---

## 7. Coordinated Structure

### Definition
A noun phrase containing coordinated elements joined by conjunctions like "and", "or", etc.

### Reliable Patterns
- Token with `dep="cc"` (coordinating conjunction)
- Token with `dep="conj"` (conjoined element)

### Logic
We check for conjunctions and conjoined elements to identify coordinated structures.

---

## 8. Appositive Structure

### Definition
A noun phrase containing an appositive construction, where one noun phrase renames or explains another.

### Reliable Patterns
- Token with `dep="appos"` (appositive)
- Two noun phrases separated by a comma

### Logic
We identify appositives by checking for appositive dependencies and potential appositives following commas.

---

## 9. Standard Relative Clause Structure

### Definition
A noun phrase containing a standard relative clause that modifies the head noun, introduced by a relative pronoun.

### Reliable Patterns
- Contains a relative pronoun (who, which, that, etc.) with `tag=WDT`, `tag=WP`, `tag=WP$`, or `tag=WRB`
- The verb in the clause often has `dep="relcl"` (relative clause)

### Logic
We check for relative pronouns and their associated verbs to identify standard relative clauses.

---

## 10. Reduced Relative Clause Structure

### Definition
A noun phrase containing a reduced relative clause (without a relative pronoun), where the clause modifies the head noun.

### Reliable Patterns
- No relative pronoun
- Subject immediately follows the head noun
- Verb follows the subject

### Logic
We identify reduced relative clauses by checking for the presence of a subject and verb sequence after the head noun. We also check for `acl` dependencies without relative pronouns.

---

## 11. Non-restrictive Relative Clause Structure

### Definition
A noun phrase containing a non-restrictive relative clause, which adds additional information about the head noun but is not essential for identifying it. The clause is separated from the head noun by a comma.

### Reliable Patterns
- Contains a relative pronoun (who, which, that, etc.)
- Has a comma separating the head noun from the relative clause

### Logic
We check for both a relative pronoun and a comma to identify non-restrictive relative clauses.

---

## 12. Finite Complement Structure

### Definition
A noun phrase with a finite clause complement introduced by a complementizer like "that", "whether", or "if".

### Reliable Patterns
- Contains a complementizer (that, whether, if) with `dep="mark"` (marker)

### Logic
We identify finite complements by checking for complementizers and their associated verbs, ensuring the structure is valid.

---

## 13. Nonfinite Complement Structure

### Definition
A noun phrase with a nonfinite clause complement, typically an infinitive (to-clause).

### Reliable Patterns
- Token "to" with `tag="TO"` introducing an infinitive

### Logic
We check for the presence of "to" followed by a verb to identify nonfinite complements.

---

## Special Case: Handling Ambiguities in Clausal Structures

### Strategies for Resolving Ambiguities
1. Distinguishing Relative Clauses from Finite Complements
2. Resolving Conflicts Between Different Clause Types

### Implementation for Conflict Resolution
We apply validation rules to remove inconsistent or implausible combinations of structures detected in a noun phrase. If no structures are detected, we assign the label `others` to avoid empty returns.

---

This document serves as a comprehensive guide to the patterns and logic used in the `ANPEAnalyzer` for detecting noun phrase structures. It reflects all current logic implemented in the analyzer.