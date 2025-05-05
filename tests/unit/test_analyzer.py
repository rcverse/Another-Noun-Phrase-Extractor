import pytest
from unittest.mock import MagicMock, patch
import spacy

from anpe.utils.analyzer import ANPEAnalyzer
from tests.conftest import create_mock_token # Corrected import path

# Sample Doc setup (could use fixtures if more complex setup needed)
nlp = spacy.blank("en")

# --- Fixtures for detect_* tests ---

@pytest.fixture
def pronoun_span(spacy_vocab, mock_doc_factory):
    """Creates a mock Span representing a single pronoun."""
    token_he = create_mock_token(text="He", pos="PRON", tag="PRP", dep="nsubj", vocab=spacy_vocab)
    mock_doc = mock_doc_factory([token_he])
    return mock_doc[0:1] # Create a Span from the mock Doc

@pytest.fixture
def non_pronoun_span(spacy_vocab, mock_doc_factory):
    """Creates a mock Span representing a single non-pronoun (noun)."""
    token_cat = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="nsubj", vocab=spacy_vocab)
    mock_doc = mock_doc_factory([token_cat])
    return mock_doc[0:1]

@pytest.fixture
def multi_token_span(spacy_vocab, mock_doc_factory):
    """Creates a mock Span with multiple tokens (not a standalone pronoun)."""
    token_the = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_cat = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="nsubj", vocab=spacy_vocab, i=1)
    # Establish dependency: DET -> NOUN (det head is noun)
    token_the.head = token_cat
    token_cat.children.append(token_the) # Add 'the' as child of 'cat'
    mock_doc = mock_doc_factory([token_the, token_cat])
    return mock_doc[0:2]

@pytest.fixture
def standalone_noun_span(spacy_vocab, mock_doc_factory):
    """Creates a mock Span representing a single standalone noun."""
    token_cat = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab)
    mock_doc = mock_doc_factory([token_cat])
    return mock_doc[0:1]

@pytest.fixture
def standalone_propn_span(spacy_vocab, mock_doc_factory):
    """Creates a mock Span representing a single standalone proper noun."""
    token_london = create_mock_token(text="London", pos="PROPN", tag="NNP", dep="ROOT", vocab=spacy_vocab)
    mock_doc = mock_doc_factory([token_london])
    return mock_doc[0:1]

@pytest.fixture
def determiner_np_span(spacy_vocab, mock_doc_factory):
    """Creates a mock Span representing a determiner + noun NP."""
    # Setup tokens with relationships
    token_the = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_cat = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    
    # Establish dependency: DET -> NOUN (det head is noun)
    token_the.head = token_cat
    token_cat.children.append(token_the) # Add 'the' as child of 'cat'
    
    mock_doc = mock_doc_factory([token_the, token_cat])
    return mock_doc[0:2]

@pytest.fixture
def determiner_np_propn_span(spacy_vocab, mock_doc_factory):
    """Creates a mock Span representing a determiner + proper noun NP."""
    token_the = create_mock_token(text="The", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_joneses = create_mock_token(text="Joneses", pos="PROPN", tag="NNPS", dep="ROOT", vocab=spacy_vocab, i=1)
    token_the.head = token_joneses
    token_joneses.children.append(token_the)
    mock_doc = mock_doc_factory([token_the, token_joneses])
    return mock_doc[0:2]

@pytest.fixture
def no_determiner_span(spacy_vocab, mock_doc_factory):
    """Creates a mock Span with a noun but no determiner."""
    token_cat = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0)
    token_sat = create_mock_token(text="sat", pos="VERB", tag="VBD", dep="ROOT", vocab=spacy_vocab, i=1)
    mock_doc = mock_doc_factory([token_cat, token_sat])
    return mock_doc[0:1] # Span just covering 'cat'

@pytest.fixture
def determiner_outside_span(spacy_vocab, mock_doc_factory):
    """Creates a mock Doc where determiner modifies noun, but span only covers noun."""
    token_the = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_cat = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_the.head = token_cat
    token_cat.children.append(token_the)
    mock_doc = mock_doc_factory([token_the, token_cat])
    return mock_doc[1:2] # Span covers only 'cat'
    
@pytest.fixture
def determiner_head_outside_span(spacy_vocab, mock_doc_factory):
    """Creates a mock Doc where determiner modifies noun, but span only covers determiner."""
    token_the = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_cat = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_the.head = token_cat # Head ('cat') is outside the span
    token_cat.children.append(token_the)
    mock_doc = mock_doc_factory([token_the, token_cat])
    return mock_doc[0:1] # Span covers only 'the'


# --- Fixtures for Adjectival Modifier Tests ---

@pytest.fixture
def adjectival_np_span(spacy_vocab, mock_doc_factory):
    """Span: adj + noun (e.g., 'happy cat')"""
    token_adj = create_mock_token(text="happy", pos="ADJ", tag="JJ", dep="amod", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_adj.head = token_noun
    token_noun.children.append(token_adj)
    mock_doc = mock_doc_factory([token_adj, token_noun])
    return mock_doc[0:2]

@pytest.fixture
def participle_modifier_np_span(spacy_vocab, mock_doc_factory):
    """Span: verb (participle) + noun (e.g., 'running water')"""
    token_verb = create_mock_token(text="running", pos="VERB", tag="VBG", dep="amod", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="water", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_verb.head = token_noun
    token_noun.children.append(token_verb)
    mock_doc = mock_doc_factory([token_verb, token_noun])
    return mock_doc[0:2]

@pytest.fixture
def adjectival_modifier_outside_span(spacy_vocab, mock_doc_factory):
    """Adj modifies noun, but span only covers noun."""
    token_adj = create_mock_token(text="happy", pos="ADJ", tag="JJ", dep="amod", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_adj.head = token_noun
    token_noun.children.append(token_adj)
    mock_doc = mock_doc_factory([token_adj, token_noun])
    return mock_doc[1:2] # Span only has 'cat'

@pytest.fixture
def adjectival_head_outside_span(spacy_vocab, mock_doc_factory):
    """Adj modifies noun, but span only covers adj."""
    token_adj = create_mock_token(text="happy", pos="ADJ", tag="JJ", dep="amod", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_adj.head = token_noun # Head 'cat' is outside span
    token_noun.children.append(token_adj)
    mock_doc = mock_doc_factory([token_adj, token_noun])
    return mock_doc[0:1] # Span only has 'happy'

@pytest.fixture
def non_amod_modifier_span(spacy_vocab, mock_doc_factory):
    """Span: word + noun, but dependency is not 'amod'."""
    token_adv = create_mock_token(text="quickly", pos="ADV", tag="RB", dep="advmod", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_adv.head = token_noun
    token_noun.children.append(token_adv)
    mock_doc = mock_doc_factory([token_adv, token_noun])
    return mock_doc[0:2]

# --- Fixtures for Prepositional Modifier Tests ---

@pytest.fixture
def prepositional_np_span(spacy_vocab, mock_doc_factory):
    """Span: noun + prep + noun_obj (e.g., 'cat in the hat')"""
    token_noun1 = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0)
    token_prep = create_mock_token(text="in", pos="ADP", tag="IN", dep="prep", vocab=spacy_vocab, i=1)
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=2)
    token_noun2 = create_mock_token(text="hat", pos="NOUN", tag="NN", dep="pobj", vocab=spacy_vocab, i=3) # Object of preposition

    token_prep.head = token_noun1      # 'in' modifies 'cat'
    token_noun1.children.append(token_prep)
    token_noun2.head = token_prep      # 'hat' is object of 'in'
    token_prep.children.append(token_noun2)
    token_det.head = token_noun2       # 'the' modifies 'hat'
    token_noun2.children.append(token_det)

    mock_doc = mock_doc_factory([token_noun1, token_prep, token_det, token_noun2])
    return mock_doc[0:4] # Span covers the whole phrase

@pytest.fixture
def prepositional_prep_outside_span(spacy_vocab, mock_doc_factory):
    """Prep modifies noun, but span excludes prep."""
    token_noun1 = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0)
    token_prep = create_mock_token(text="in", pos="ADP", tag="IN", dep="prep", vocab=spacy_vocab, i=1)
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=2)
    token_noun2 = create_mock_token(text="hat", pos="NOUN", tag="NN", dep="pobj", vocab=spacy_vocab, i=3)
    token_prep.head = token_noun1
    token_noun1.children.append(token_prep)
    token_noun2.head = token_prep
    token_prep.children.append(token_noun2)
    token_det.head = token_noun2
    token_noun2.children.append(token_det)
    mock_doc = mock_doc_factory([token_noun1, token_prep, token_det, token_noun2])
    return mock_doc[0:1] # Span only 'cat'

@pytest.fixture
def prepositional_head_outside_span(spacy_vocab, mock_doc_factory):
    """Prep modifies noun, but span excludes head noun."""
    token_noun1 = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0)
    token_prep = create_mock_token(text="in", pos="ADP", tag="IN", dep="prep", vocab=spacy_vocab, i=1)
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=2)
    token_noun2 = create_mock_token(text="hat", pos="NOUN", tag="NN", dep="pobj", vocab=spacy_vocab, i=3)
    token_prep.head = token_noun1 # Head 'cat' is outside span
    token_noun1.children.append(token_prep)
    token_noun2.head = token_prep
    token_prep.children.append(token_noun2)
    token_det.head = token_noun2
    token_noun2.children.append(token_det)
    mock_doc = mock_doc_factory([token_noun1, token_prep, token_det, token_noun2])
    return mock_doc[1:4] # Span 'in the hat'

@pytest.fixture
def prepositional_obj_outside_span(spacy_vocab, mock_doc_factory):
    """Prep modifies noun, object of prep is outside span."""
    token_noun1 = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0)
    token_prep = create_mock_token(text="in", pos="ADP", tag="IN", dep="prep", vocab=spacy_vocab, i=1)
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=2)
    token_noun2 = create_mock_token(text="hat", pos="NOUN", tag="NN", dep="pobj", vocab=spacy_vocab, i=3) # Object 'hat' is outside
    token_prep.head = token_noun1
    token_noun1.children.append(token_prep)
    token_noun2.head = token_prep # 'hat' is object of 'in'
    token_prep.children.append(token_noun2)
    token_det.head = token_noun2 # 'the' modifies 'hat'
    token_noun2.children.append(token_det)
    mock_doc = mock_doc_factory([token_noun1, token_prep, token_det, token_noun2])
    return mock_doc[0:3] # Span 'cat in the'

@pytest.fixture
def prepositional_no_pobj_span(spacy_vocab, mock_doc_factory):
    """Span includes prep modifying noun, but prep has no pobj child within span."""
    token_noun1 = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0)
    token_prep = create_mock_token(text="in", pos="ADP", tag="IN", dep="prep", vocab=spacy_vocab, i=1)
    # No pobj token linked to prep
    token_prep.head = token_noun1
    token_noun1.children.append(token_prep)
    mock_doc = mock_doc_factory([token_noun1, token_prep])
    return mock_doc[0:2] # Span 'cat in'


# --- Fixtures for Compound Noun Tests ---

@pytest.fixture
def compound_noun_span(spacy_vocab, mock_doc_factory):
    """Span: noun + noun (e.g., 'apple pie')"""
    token_noun1 = create_mock_token(text="apple", pos="NOUN", tag="NN", dep="compound", vocab=spacy_vocab, i=0)
    token_noun2 = create_mock_token(text="pie", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_noun1.head = token_noun2
    token_noun2.children.append(token_noun1)
    mock_doc = mock_doc_factory([token_noun1, token_noun2])
    return mock_doc[0:2]

@pytest.fixture
def compound_modifier_outside_span(spacy_vocab, mock_doc_factory):
    """Compound modifier noun outside span."""
    token_noun1 = create_mock_token(text="apple", pos="NOUN", tag="NN", dep="compound", vocab=spacy_vocab, i=0)
    token_noun2 = create_mock_token(text="pie", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_noun1.head = token_noun2
    token_noun2.children.append(token_noun1)
    mock_doc = mock_doc_factory([token_noun1, token_noun2])
    return mock_doc[1:2] # Span only 'pie'

@pytest.fixture
def compound_head_outside_span(spacy_vocab, mock_doc_factory):
    """Compound head noun outside span."""
    token_noun1 = create_mock_token(text="apple", pos="NOUN", tag="NN", dep="compound", vocab=spacy_vocab, i=0)
    token_noun2 = create_mock_token(text="pie", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_noun1.head = token_noun2 # Head 'pie' is outside span
    token_noun2.children.append(token_noun1)
    mock_doc = mock_doc_factory([token_noun1, token_noun2])
    return mock_doc[0:1] # Span only 'apple'

@pytest.fixture
def non_compound_dep_span(spacy_vocab, mock_doc_factory):
    """Noun + Noun but dependency is not 'compound'."""
    token_noun1 = create_mock_token(text="apple", pos="NOUN", tag="NN", dep="nsubj", vocab=spacy_vocab, i=0)
    token_noun2 = create_mock_token(text="pie", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_noun1.head = token_noun2 # Example relationship, dep is not compound
    token_noun2.children.append(token_noun1)
    mock_doc = mock_doc_factory([token_noun1, token_noun2])
    return mock_doc[0:2]

# --- Fixtures for Possessive NP Tests ---

@pytest.fixture
def possessive_s_span(spacy_vocab, mock_doc_factory):
    """Span: noun + 's + noun (e.g., 'cat's toy')"""
    token_noun1 = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="poss", vocab=spacy_vocab, i=0)
    token_poss = create_mock_token(text="'s", pos="PART", tag="POS", dep="case", vocab=spacy_vocab, i=1)
    token_noun2 = create_mock_token(text="toy", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=2)

    token_noun1.head = token_noun2 # 'cat' modifies 'toy' via possessive
    token_noun2.children.append(token_noun1)
    token_poss.head = token_noun1  # 's case marker attaches to 'cat'
    token_noun1.children.append(token_poss)
    
    mock_doc = mock_doc_factory([token_noun1, token_poss, token_noun2])
    return mock_doc[0:3]

@pytest.fixture
def possessive_pronoun_span(spacy_vocab, mock_doc_factory):
    """Span: possessive_pronoun + noun (e.g., 'its toy')"""
    token_pron = create_mock_token(text="its", pos="PRON", tag="PRP$", dep="poss", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="toy", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_pron.head = token_noun
    token_noun.children.append(token_pron)
    mock_doc = mock_doc_factory([token_pron, token_noun])
    return mock_doc[0:2]

@pytest.fixture
def possessive_dep_span(spacy_vocab, mock_doc_factory):
    """Span: noun + noun with 'poss' dep (e.g., 'government intervention')"""
    # Note: spaCy might analyze this differently, but testing the code's check
    token_noun1 = create_mock_token(text="government", pos="NOUN", tag="NN", dep="poss", vocab=spacy_vocab, i=0)
    token_noun2 = create_mock_token(text="intervention", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_noun1.head = token_noun2
    token_noun2.children.append(token_noun1)
    mock_doc = mock_doc_factory([token_noun1, token_noun2])
    return mock_doc[0:2]

@pytest.fixture
def possessive_unmarked_dep_span(spacy_vocab, mock_doc_factory):
    """Span: Noun child has 'poss' dep pointing to Noun head (Rousseau insights)"""
    token_noun1 = create_mock_token(text="Rousseau", pos="PROPN", tag="NNP", dep="poss", vocab=spacy_vocab, i=0)
    token_noun2 = create_mock_token(text="insights", pos="NOUN", tag="NNS", dep="ROOT", vocab=spacy_vocab, i=1)
    # 'Rousseau' has 'poss' dep and modifies 'insights'
    token_noun1.head = token_noun2
    # 'insights' has 'Rousseau' as a child with 'poss' dep
    token_noun2.children.append(token_noun1)
    mock_doc = mock_doc_factory([token_noun1, token_noun2])
    return mock_doc[0:2]

@pytest.fixture
def possessive_marker_outside_span(spacy_vocab, mock_doc_factory):
    """Possessive marker ('s) outside span."""
    token_noun1 = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="poss", vocab=spacy_vocab, i=0)
    token_poss = create_mock_token(text="'s", pos="PART", tag="POS", dep="case", vocab=spacy_vocab, i=1) # Outside
    token_noun2 = create_mock_token(text="toy", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=2)
    token_noun1.head = token_noun2
    token_noun2.children.append(token_noun1)
    token_poss.head = token_noun1
    token_noun1.children.append(token_poss)
    mock_doc = mock_doc_factory([token_noun1, token_poss, token_noun2])
    return mock_doc[0:1] # Span 'cat'
    
@pytest.fixture
def possessive_head_outside_span(spacy_vocab, mock_doc_factory):
    """Possessive head noun outside span."""
    token_noun1 = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="poss", vocab=spacy_vocab, i=0)
    token_poss = create_mock_token(text="'s", pos="PART", tag="POS", dep="case", vocab=spacy_vocab, i=1)
    token_noun2 = create_mock_token(text="toy", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=2) # Outside
    token_noun1.head = token_noun2 # Head 'toy' is outside
    token_noun2.children.append(token_noun1)
    token_poss.head = token_noun1
    token_noun1.children.append(token_poss)
    mock_doc = mock_doc_factory([token_noun1, token_poss, token_noun2])
    return mock_doc[0:2] # Span "cat 's"

# --- Fixtures for Quantified NP Tests ---

@pytest.fixture
def quantified_np_span(spacy_vocab, mock_doc_factory):
    """Span: number + noun (e.g., 'two cats')"""
    token_num = create_mock_token(text="two", pos="NUM", tag="CD", dep="nummod", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cats", pos="NOUN", tag="NNS", dep="ROOT", vocab=spacy_vocab, i=1)
    token_num.head = token_noun
    token_noun.children.append(token_num)
    mock_doc = mock_doc_factory([token_num, token_noun])
    return mock_doc[0:2]

@pytest.fixture
def quantified_nummod_span(spacy_vocab, mock_doc_factory):
    """Span: word (dep=nummod) + noun (e.g., 'many cats')"""
    token_quant = create_mock_token(text="many", pos="ADJ", tag="JJ", dep="nummod", vocab=spacy_vocab, i=0) # Note: might be ADJ/DET, test dep
    token_noun = create_mock_token(text="cats", pos="NOUN", tag="NNS", dep="ROOT", vocab=spacy_vocab, i=1)
    token_quant.head = token_noun
    token_noun.children.append(token_quant)
    mock_doc = mock_doc_factory([token_quant, token_noun])
    return mock_doc[0:2]

@pytest.fixture
def quantified_quantifier_outside_span(spacy_vocab, mock_doc_factory):
    """Quantifier (NUM) outside span."""
    token_num = create_mock_token(text="two", pos="NUM", tag="CD", dep="nummod", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cats", pos="NOUN", tag="NNS", dep="ROOT", vocab=spacy_vocab, i=1)
    token_num.head = token_noun
    token_noun.children.append(token_num)
    mock_doc = mock_doc_factory([token_num, token_noun])
    return mock_doc[1:2] # Span 'cats'

@pytest.fixture
def quantified_head_outside_span(spacy_vocab, mock_doc_factory):
    """Quantified head noun outside span."""
    token_num = create_mock_token(text="two", pos="NUM", tag="CD", dep="nummod", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cats", pos="NOUN", tag="NNS", dep="ROOT", vocab=spacy_vocab, i=1)
    token_num.head = token_noun # Head 'cats' is outside
    token_noun.children.append(token_num)
    mock_doc = mock_doc_factory([token_num, token_noun])
    return mock_doc[0:1] # Span 'two'

@pytest.fixture
def non_quantified_span(spacy_vocab, mock_doc_factory):
    """Span without NUM or nummod dep."""
    token_adj = create_mock_token(text="happy", pos="ADJ", tag="JJ", dep="amod", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_adj.head = token_noun
    token_noun.children.append(token_adj)
    mock_doc = mock_doc_factory([token_adj, token_noun])
    return mock_doc[0:2]


# --- Fixtures for Coordinate NP Tests ---

@pytest.fixture
def coordinate_np_span(spacy_vocab, mock_doc_factory):
    """Span: noun + cc + noun (e.g., 'cats and dogs')"""
    token_noun1 = create_mock_token(text="cats", pos="NOUN", tag="NNS", dep="ROOT", vocab=spacy_vocab, i=0)
    token_cc = create_mock_token(text="and", pos="CCONJ", tag="CC", dep="cc", vocab=spacy_vocab, i=1)
    token_noun2 = create_mock_token(text="dogs", pos="NOUN", tag="NNS", dep="conj", vocab=spacy_vocab, i=2)

    token_cc.head = token_noun1 # 'and' attaches to first element
    token_noun1.children.append(token_cc)
    token_noun2.head = token_noun1 # 'dogs' is conjunct of 'cats'
    token_noun1.children.append(token_noun2)
    
    mock_doc = mock_doc_factory([token_noun1, token_cc, token_noun2])
    return mock_doc[0:3]

@pytest.fixture
def coordinate_np_multiple_span(spacy_vocab, mock_doc_factory):
    """Span: noun, noun, cc + noun (e.g., 'cats, mice, and dogs')"""
    token_noun1 = create_mock_token(text="cats", pos="NOUN", tag="NNS", dep="ROOT", vocab=spacy_vocab, i=0)
    token_punc = create_mock_token(text=",", pos="PUNCT", tag=",", dep="punct", vocab=spacy_vocab, i=1)
    token_noun2 = create_mock_token(text="mice", pos="NOUN", tag="NNS", dep="conj", vocab=spacy_vocab, i=2)
    token_punc2 = create_mock_token(text=",", pos="PUNCT", tag=",", dep="punct", vocab=spacy_vocab, i=3)
    token_cc = create_mock_token(text="and", pos="CCONJ", tag="CC", dep="cc", vocab=spacy_vocab, i=4)
    token_noun3 = create_mock_token(text="dogs", pos="NOUN", tag="NNS", dep="conj", vocab=spacy_vocab, i=5)
    
    token_punc.head = token_noun1
    token_noun1.children.append(token_punc)
    token_noun2.head = token_noun1 # mice conj cats
    token_noun1.children.append(token_noun2)
    token_punc2.head = token_noun2
    token_noun2.children.append(token_punc2)
    token_cc.head = token_noun1 # cc attached to head of coordination
    token_noun1.children.append(token_cc)
    token_noun3.head = token_noun1 # dogs conj cats (or mice, depends on parser model)
    token_noun1.children.append(token_noun3)
    
    mock_doc = mock_doc_factory([token_noun1, token_punc, token_noun2, token_punc2, token_cc, token_noun3])
    return mock_doc[0:6]

@pytest.fixture
def coordinate_cc_outside_span(spacy_vocab, mock_doc_factory):
    """Coordinating conjunction outside span."""
    token_noun1 = create_mock_token(text="cats", pos="NOUN", tag="NNS", dep="ROOT", vocab=spacy_vocab, i=0)
    token_cc = create_mock_token(text="and", pos="CCONJ", tag="CC", dep="cc", vocab=spacy_vocab, i=1) # Outside
    token_noun2 = create_mock_token(text="dogs", pos="NOUN", tag="NNS", dep="conj", vocab=spacy_vocab, i=2)
    token_cc.head = token_noun1
    token_noun1.children.append(token_cc)
    token_noun2.head = token_noun1
    token_noun1.children.append(token_noun2)
    mock_doc = mock_doc_factory([token_noun1, token_cc, token_noun2])
    return mock_doc[0:1] # Span 'cats'

@pytest.fixture
def coordinate_conj_outside_span(spacy_vocab, mock_doc_factory):
    """Conjunct outside span."""
    token_noun1 = create_mock_token(text="cats", pos="NOUN", tag="NNS", dep="ROOT", vocab=spacy_vocab, i=0)
    token_cc = create_mock_token(text="and", pos="CCONJ", tag="CC", dep="cc", vocab=spacy_vocab, i=1)
    token_noun2 = create_mock_token(text="dogs", pos="NOUN", tag="NNS", dep="conj", vocab=spacy_vocab, i=2) # Outside
    token_cc.head = token_noun1
    token_noun1.children.append(token_cc)
    token_noun2.head = token_noun1
    token_noun1.children.append(token_noun2)
    mock_doc = mock_doc_factory([token_noun1, token_cc, token_noun2])
    return mock_doc[0:2] # Span 'cats and'

@pytest.fixture
def coordinate_no_cc_span(spacy_vocab, mock_doc_factory):
    """Span has conjunct but no cc."""
    token_noun1 = create_mock_token(text="cats", pos="NOUN", tag="NNS", dep="ROOT", vocab=spacy_vocab, i=0)
    token_noun2 = create_mock_token(text="dogs", pos="NOUN", tag="NNS", dep="conj", vocab=spacy_vocab, i=1)
    token_noun2.head = token_noun1
    token_noun1.children.append(token_noun2)
    mock_doc = mock_doc_factory([token_noun1, token_noun2])
    return mock_doc[0:2] # Span 'cats dogs'

@pytest.fixture
def coordinate_no_conj_span(spacy_vocab, mock_doc_factory):
    """Span has cc but no conjunct."""
    token_noun1 = create_mock_token(text="cats", pos="NOUN", tag="NNS", dep="ROOT", vocab=spacy_vocab, i=0)
    token_cc = create_mock_token(text="and", pos="CCONJ", tag="CC", dep="cc", vocab=spacy_vocab, i=1)
    token_cc.head = token_noun1
    token_noun1.children.append(token_cc)
    mock_doc = mock_doc_factory([token_noun1, token_cc])
    return mock_doc[0:2] # Span 'cats and'

# --- Fixtures for Appositive NP Tests ---

@pytest.fixture
def appositive_np_span(spacy_vocab, mock_doc_factory):
    """Span: noun + , + noun(appos) (e.g., 'my friend, Bob')"""
    token_det = create_mock_token(text="my", pos="PRON", tag="PRP$", dep="poss", vocab=spacy_vocab, i=0)
    token_noun1 = create_mock_token(text="friend", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_punc = create_mock_token(text=",", pos="PUNCT", tag=",", dep="punct", vocab=spacy_vocab, i=2)
    token_noun2 = create_mock_token(text="Bob", pos="PROPN", tag="NNP", dep="appos", vocab=spacy_vocab, i=3)

    token_det.head = token_noun1
    token_noun1.children.append(token_det)
    token_punc.head = token_noun1
    token_noun1.children.append(token_punc)
    token_noun2.head = token_noun1 # 'Bob' is appositive to 'friend'
    token_noun1.children.append(token_noun2)
    
    mock_doc = mock_doc_factory([token_det, token_noun1, token_punc, token_noun2])
    return mock_doc[0:4]

@pytest.fixture
def appositive_modifier_outside_span(spacy_vocab, mock_doc_factory):
    """Appositive modifier outside span."""
    token_det = create_mock_token(text="my", pos="PRON", tag="PRP$", dep="poss", vocab=spacy_vocab, i=0)
    token_noun1 = create_mock_token(text="friend", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_punc = create_mock_token(text=",", pos="PUNCT", tag=",", dep="punct", vocab=spacy_vocab, i=2)
    token_noun2 = create_mock_token(text="Bob", pos="PROPN", tag="NNP", dep="appos", vocab=spacy_vocab, i=3) # Outside
    token_det.head = token_noun1
    token_noun1.children.append(token_det)
    token_punc.head = token_noun1
    token_noun1.children.append(token_punc)
    token_noun2.head = token_noun1
    token_noun1.children.append(token_noun2)
    mock_doc = mock_doc_factory([token_det, token_noun1, token_punc, token_noun2])
    return mock_doc[0:3] # Span 'my friend ,'

@pytest.fixture
def appositive_head_outside_span(spacy_vocab, mock_doc_factory):
    """Appositive head outside span."""
    token_det = create_mock_token(text="my", pos="PRON", tag="PRP$", dep="poss", vocab=spacy_vocab, i=0) # Outside
    token_noun1 = create_mock_token(text="friend", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1) # Outside
    token_punc = create_mock_token(text=",", pos="PUNCT", tag=",", dep="punct", vocab=spacy_vocab, i=2)
    token_noun2 = create_mock_token(text="Bob", pos="PROPN", tag="NNP", dep="appos", vocab=spacy_vocab, i=3)
    token_det.head = token_noun1
    token_noun1.children.append(token_det)
    token_punc.head = token_noun1
    token_noun1.children.append(token_punc)
    token_noun2.head = token_noun1 # Head 'friend' is outside
    token_noun1.children.append(token_noun2)
    mock_doc = mock_doc_factory([token_det, token_noun1, token_punc, token_noun2])
    return mock_doc[2:4] # Span ', Bob'

@pytest.fixture
def non_appositive_dep_span(spacy_vocab, mock_doc_factory):
    """Span with two nouns but no 'appos' dependency."""
    token_det = create_mock_token(text="my", pos="PRON", tag="PRP$", dep="poss", vocab=spacy_vocab, i=0)
    token_noun1 = create_mock_token(text="friend", pos="NOUN", tag="NN", dep="nsubj", vocab=spacy_vocab, i=1)
    token_verb = create_mock_token(text="likes", pos="VERB", tag="VBZ", dep="ROOT", vocab=spacy_vocab, i=2)
    token_noun2 = create_mock_token(text="Bob", pos="PROPN", tag="NNP", dep="dobj", vocab=spacy_vocab, i=3)
    token_det.head = token_noun1
    token_noun1.children.append(token_det)
    token_noun1.head = token_verb
    token_verb.children.append(token_noun1)
    token_noun2.head = token_verb
    token_verb.children.append(token_noun2)
    mock_doc = mock_doc_factory([token_det, token_noun1, token_verb, token_noun2])
    # Create a span covering the first NP 'my friend'
    return mock_doc[0:2]


# --- Fixtures for Relative Clause Tests ---

@pytest.fixture
def relative_clause_wdt_span(spacy_vocab, mock_doc_factory):
    """Span: noun + WDT + verb (e.g., 'the cat that slept')"""
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="nsubj", vocab=spacy_vocab, i=1)
    token_rel = create_mock_token(text="that", pos="PRON", tag="WDT", dep="nsubj", vocab=spacy_vocab, i=2) # Can be PRON/SCONJ, use tag WDT
    token_verb = create_mock_token(text="slept", pos="VERB", tag="VBD", dep="relcl", vocab=spacy_vocab, i=3)

    token_det.head = token_noun
    token_noun.children.append(token_det)
    token_verb.head = token_noun # 'slept' clause modifies 'cat'
    token_noun.children.append(token_verb)
    token_rel.head = token_verb # 'that' is subject of 'slept'
    token_verb.children.append(token_rel)
    
    mock_doc = mock_doc_factory([token_det, token_noun, token_rel, token_verb])
    return mock_doc[0:4] # Span 'the cat that slept'

@pytest.fixture
def relative_clause_relcl_span(spacy_vocab, mock_doc_factory):
    """Span: noun + verb(relcl) (implicit relative e.g., 'the toy (that) I bought')"""
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_noun1 = create_mock_token(text="toy", pos="NOUN", tag="NN", dep="dobj", vocab=spacy_vocab, i=1) # toy is obj of bought
    token_pron = create_mock_token(text="I", pos="PRON", tag="PRP", dep="nsubj", vocab=spacy_vocab, i=2)
    token_verb = create_mock_token(text="bought", pos="VERB", tag="VBD", dep="relcl", vocab=spacy_vocab, i=3)

    token_det.head = token_noun1
    token_noun1.children.append(token_det)
    token_verb.head = token_noun1 # 'bought' clause modifies 'toy'
    token_noun1.children.append(token_verb)
    token_pron.head = token_verb # 'I' is subject of 'bought'
    token_verb.children.append(token_pron)
    
    # Note: in a real parse, toy might be dobj of bought, and the whole clause attached elsewhere.
    # Here we simplify attachment to noun1 for testing the detector's relcl check.
    mock_doc = mock_doc_factory([token_det, token_noun1, token_pron, token_verb])
    return mock_doc[0:4] # Span 'the toy I bought'
    
@pytest.fixture
def relative_clause_pronoun_outside(spacy_vocab, mock_doc_factory):
    """Relative pronoun WDT outside span."""
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="nsubj", vocab=spacy_vocab, i=1)
    token_rel = create_mock_token(text="that", pos="PRON", tag="WDT", dep="nsubj", vocab=spacy_vocab, i=2) # Outside
    token_verb = create_mock_token(text="slept", pos="VERB", tag="VBD", dep="relcl", vocab=spacy_vocab, i=3)
    token_det.head = token_noun
    token_noun.children.append(token_det)
    token_verb.head = token_noun
    token_noun.children.append(token_verb)
    token_rel.head = token_verb
    token_verb.children.append(token_rel)
    mock_doc = mock_doc_factory([token_det, token_noun, token_rel, token_verb])
    return mock_doc[0:2] # Span 'the cat'
    
@pytest.fixture
def relative_clause_verb_outside(spacy_vocab, mock_doc_factory):
    """Relative clause verb (relcl) outside span."""
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="nsubj", vocab=spacy_vocab, i=1)
    token_rel = create_mock_token(text="that", pos="PRON", tag="WDT", dep="nsubj", vocab=spacy_vocab, i=2)
    token_verb = create_mock_token(text="slept", pos="VERB", tag="VBD", dep="relcl", vocab=spacy_vocab, i=3) # Outside
    token_det.head = token_noun
    token_noun.children.append(token_det)
    token_verb.head = token_noun # Attached to noun in span
    token_noun.children.append(token_verb)
    token_rel.head = token_verb
    token_verb.children.append(token_rel)
    mock_doc = mock_doc_factory([token_det, token_noun, token_rel, token_verb])
    return mock_doc[0:3] # Span 'the cat that'

# --- Fixtures for Reduced Relative Clause Tests ---

@pytest.fixture
def reduced_relative_clause_acl_span(spacy_vocab, mock_doc_factory):
    """Span: noun + verb(acl) (e.g., 'book written by Bob')"""
    token_noun1 = create_mock_token(text="book", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0)
    token_verb = create_mock_token(text="written", pos="VERB", tag="VBN", dep="acl", vocab=spacy_vocab, i=1)
    token_prep = create_mock_token(text="by", pos="ADP", tag="IN", dep="agent", vocab=spacy_vocab, i=2)
    token_noun2 = create_mock_token(text="Bob", pos="PROPN", tag="NNP", dep="pobj", vocab=spacy_vocab, i=3)

    token_verb.head = token_noun1 # 'written' modifies 'book'
    token_noun1.children.append(token_verb)
    token_prep.head = token_verb # 'by' agent of 'written'
    token_verb.children.append(token_prep)
    token_noun2.head = token_prep # 'Bob' object of 'by'
    token_prep.children.append(token_noun2)
    
    mock_doc = mock_doc_factory([token_noun1, token_verb, token_prep, token_noun2])
    return mock_doc[0:4]

@pytest.fixture
def reduced_relative_clause_acl_outside_span(spacy_vocab, mock_doc_factory):
    """Reduced relative verb (acl) outside span."""
    token_noun1 = create_mock_token(text="book", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0)
    token_verb = create_mock_token(text="written", pos="VERB", tag="VBN", dep="acl", vocab=spacy_vocab, i=1) # Outside
    token_prep = create_mock_token(text="by", pos="ADP", tag="IN", dep="agent", vocab=spacy_vocab, i=2)
    token_noun2 = create_mock_token(text="Bob", pos="PROPN", tag="NNP", dep="pobj", vocab=spacy_vocab, i=3)
    token_verb.head = token_noun1
    token_noun1.children.append(token_verb)
    token_prep.head = token_verb
    token_verb.children.append(token_prep)
    token_noun2.head = token_prep
    token_prep.children.append(token_noun2)
    mock_doc = mock_doc_factory([token_noun1, token_verb, token_prep, token_noun2])
    return mock_doc[0:1] # Span 'book'

@pytest.fixture
def reduced_relative_clause_head_outside_span(spacy_vocab, mock_doc_factory):
    """Reduced relative head noun outside span."""
    token_noun1 = create_mock_token(text="book", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0) # Outside
    token_verb = create_mock_token(text="written", pos="VERB", tag="VBN", dep="acl", vocab=spacy_vocab, i=1)
    token_prep = create_mock_token(text="by", pos="ADP", tag="IN", dep="agent", vocab=spacy_vocab, i=2)
    token_noun2 = create_mock_token(text="Bob", pos="PROPN", tag="NNP", dep="pobj", vocab=spacy_vocab, i=3)
    token_verb.head = token_noun1 # Head 'book' is outside
    token_noun1.children.append(token_verb)
    token_prep.head = token_verb
    token_verb.children.append(token_prep)
    token_noun2.head = token_prep
    token_prep.children.append(token_noun2)
    mock_doc = mock_doc_factory([token_noun1, token_verb, token_prep, token_noun2])
    return mock_doc[1:4] # Span 'written by Bob'

@pytest.fixture
def reduced_relative_clause_with_rel_pron(spacy_vocab, mock_doc_factory):
    """Looks like reduced relative, but has a relative pronoun (so it's not reduced)."""
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="nsubj", vocab=spacy_vocab, i=1)
    token_rel = create_mock_token(text="that", pos="PRON", tag="WDT", dep="nsubj", vocab=spacy_vocab, i=2) # Relative pronoun present
    token_verb = create_mock_token(text="slept", pos="VERB", tag="VBD", dep="acl", vocab=spacy_vocab, i=3) # Using acl dep for test
    token_det.head = token_noun
    token_noun.children.append(token_det)
    token_verb.head = token_noun # 'slept' modifies 'cat'
    token_noun.children.append(token_verb)
    token_rel.head = token_verb # 'that' is subject of 'slept'
    token_verb.children.append(token_rel)
    mock_doc = mock_doc_factory([token_det, token_noun, token_rel, token_verb])
    return mock_doc[0:4] # Span 'the cat that slept'

# --- Fixtures for Finite Complement Clause Tests ---

# Mock _is_complement_taking_noun_within_span for some tests
@pytest.fixture
def mock_is_complement_noun(mocker):
    # This fixture allows mocking the helper directly if needed for specific complement tests
    return mocker.patch('anpe.utils.analyzer.ANPEAnalyzer._is_complement_taking_noun_within_span')

@pytest.fixture
def finite_complement_span(spacy_vocab, mock_doc_factory, mocker):
    """Span: noun + that(mark) + verb(ccomp/acl) (e.g., 'the fact that he left')"""
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="fact", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_comp = create_mock_token(text="that", pos="SCONJ", tag="IN", dep="mark", vocab=spacy_vocab, i=2)
    token_subj = create_mock_token(text="he", pos="PRON", tag="PRP", dep="nsubj", vocab=spacy_vocab, i=3)
    token_verb = create_mock_token(text="left", pos="VERB", tag="VBD", dep="acl", vocab=spacy_vocab, i=4) # Often acl/ccomp
    
    token_det.head = token_noun
    token_noun.children.append(token_det)
    token_verb.head = token_noun # Clause modifies 'fact'
    token_noun.children.append(token_verb)
    token_comp.head = token_verb # 'that' marks 'left' verb
    token_verb.children.append(token_comp)
    token_subj.head = token_verb # 'he' subject of 'left'
    token_verb.children.append(token_subj)

    # Mock the helper to return True for 'fact' within this specific span instance
    mocker.patch('anpe.utils.analyzer.ANPEAnalyzer._is_complement_taking_noun_within_span', return_value=True)

    mock_doc = mock_doc_factory([token_det, token_noun, token_comp, token_subj, token_verb])
    span = mock_doc[0:5] 
    
    # Ensure the patch applies only when checking the specific 'fact' token within the span
    original_helper = ANPEAnalyzer._is_complement_taking_noun_within_span
    def side_effect(token, current_span):
        if token is token_noun and current_span == span:
            return True
        # Call the original function for other cases if necessary (or just return False)
        # return original_helper(self, token, current_span) 
        return False
    mocker.patch('anpe.utils.analyzer.ANPEAnalyzer._is_complement_taking_noun_within_span', side_effect=side_effect)

    return span

@pytest.fixture
def finite_complement_comp_outside(spacy_vocab, mock_doc_factory, mocker):
    """Complementizer 'that' outside span."""
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="fact", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_comp = create_mock_token(text="that", pos="SCONJ", tag="IN", dep="mark", vocab=spacy_vocab, i=2) # Outside
    token_subj = create_mock_token(text="he", pos="PRON", tag="PRP", dep="nsubj", vocab=spacy_vocab, i=3)
    token_verb = create_mock_token(text="left", pos="VERB", tag="VBD", dep="acl", vocab=spacy_vocab, i=4)
    token_det.head = token_noun
    token_noun.children.append(token_det)
    token_verb.head = token_noun
    token_noun.children.append(token_verb)
    token_comp.head = token_verb
    token_verb.children.append(token_comp)
    token_subj.head = token_verb
    token_verb.children.append(token_subj)
    mocker.patch('anpe.utils.analyzer.ANPEAnalyzer._is_complement_taking_noun_within_span', return_value=True)
    mock_doc = mock_doc_factory([token_det, token_noun, token_comp, token_subj, token_verb])
    return mock_doc[0:2] # Span 'the fact'

@pytest.fixture
def finite_complement_verb_outside(spacy_vocab, mock_doc_factory, mocker):
    """Complement verb outside span."""
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="fact", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_comp = create_mock_token(text="that", pos="SCONJ", tag="IN", dep="mark", vocab=spacy_vocab, i=2)
    token_subj = create_mock_token(text="he", pos="PRON", tag="PRP", dep="nsubj", vocab=spacy_vocab, i=3)
    token_verb = create_mock_token(text="left", pos="VERB", tag="VBD", dep="acl", vocab=spacy_vocab, i=4) # Outside
    token_det.head = token_noun
    token_noun.children.append(token_det)
    token_verb.head = token_noun # Attached to noun in span
    token_noun.children.append(token_verb)
    token_comp.head = token_verb
    token_verb.children.append(token_comp)
    token_subj.head = token_verb
    token_verb.children.append(token_subj)
    mocker.patch('anpe.utils.analyzer.ANPEAnalyzer._is_complement_taking_noun_within_span', return_value=True)
    mock_doc = mock_doc_factory([token_det, token_noun, token_comp, token_subj, token_verb])
    return mock_doc[0:4] # Span 'the fact that he'

@pytest.fixture
def finite_complement_non_comp_noun(spacy_vocab, mock_doc_factory, mocker):
    """Structure looks like complement, but head noun doesn't take complements."""
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_noun = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1) # 'cat' assumed not comp-taking
    token_comp = create_mock_token(text="that", pos="SCONJ", tag="IN", dep="mark", vocab=spacy_vocab, i=2)
    token_subj = create_mock_token(text="he", pos="PRON", tag="PRP", dep="nsubj", vocab=spacy_vocab, i=3)
    token_verb = create_mock_token(text="left", pos="VERB", tag="VBD", dep="acl", vocab=spacy_vocab, i=4)
    token_det.head = token_noun
    token_noun.children.append(token_det)
    token_verb.head = token_noun 
    token_noun.children.append(token_verb)
    token_comp.head = token_verb
    token_verb.children.append(token_comp)
    token_subj.head = token_verb
    token_verb.children.append(token_subj)
    
    # Mock helper to return False for 'cat'
    mocker.patch('anpe.utils.analyzer.ANPEAnalyzer._is_complement_taking_noun_within_span', return_value=False)
    mock_doc = mock_doc_factory([token_det, token_noun, token_comp, token_subj, token_verb])
    return mock_doc[0:5]

# --- Fixtures for Non-finite Complement Clause Tests ---

@pytest.fixture
def nonfinite_infinitive_span(spacy_vocab, mock_doc_factory):
    """Span: noun + to + verb (e.g., 'plan to leave')"""
    token_noun = create_mock_token(text="plan", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0)
    token_to = create_mock_token(text="to", pos="PART", tag="TO", dep="mark", vocab=spacy_vocab, i=1) # Can be aux/mark
    token_verb = create_mock_token(text="leave", pos="VERB", tag="VB", dep="acl", vocab=spacy_vocab, i=2) # acl/xcomp
    
    token_verb.head = token_noun # Verb modifies 'plan'
    token_noun.children.append(token_verb)
    token_to.head = token_verb # 'to' marks 'leave'
    token_verb.children.append(token_to)
    
    mock_doc = mock_doc_factory([token_noun, token_to, token_verb])
    return mock_doc[0:3]
    
@pytest.fixture
def nonfinite_gerund_span(spacy_vocab, mock_doc_factory):
    """Span: noun + prep + verb(gerund) (e.g., 'fear of flying')"""
    token_noun1 = create_mock_token(text="fear", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0)
    token_prep = create_mock_token(text="of", pos="ADP", tag="IN", dep="prep", vocab=spacy_vocab, i=1)
    token_verb = create_mock_token(text="flying", pos="VERB", tag="VBG", dep="pcomp", vocab=spacy_vocab, i=2)

    token_prep.head = token_noun1 # 'of' modifies 'fear'
    token_noun1.children.append(token_prep)
    token_verb.head = token_prep # 'flying' is pcomp of 'of'
    token_prep.children.append(token_verb)
    
    mock_doc = mock_doc_factory([token_noun1, token_prep, token_verb])
    return mock_doc[0:3]
    
@pytest.fixture
def nonfinite_infinitive_verb_outside(spacy_vocab, mock_doc_factory):
    """Infinitive verb outside span."""
    token_noun = create_mock_token(text="plan", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0)
    token_to = create_mock_token(text="to", pos="PART", tag="TO", dep="mark", vocab=spacy_vocab, i=1)
    token_verb = create_mock_token(text="leave", pos="VERB", tag="VB", dep="acl", vocab=spacy_vocab, i=2) # Outside
    token_verb.head = token_noun
    token_noun.children.append(token_verb)
    token_to.head = token_verb
    token_verb.children.append(token_to)
    mock_doc = mock_doc_factory([token_noun, token_to, token_verb])
    return mock_doc[0:2] # Span 'plan to'
    
@pytest.fixture
def nonfinite_gerund_outside(spacy_vocab, mock_doc_factory):
    """Gerund verb outside span."""
    token_noun1 = create_mock_token(text="fear", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0)
    token_prep = create_mock_token(text="of", pos="ADP", tag="IN", dep="prep", vocab=spacy_vocab, i=1)
    token_verb = create_mock_token(text="flying", pos="VERB", tag="VBG", dep="pcomp", vocab=spacy_vocab, i=2) # Outside
    token_prep.head = token_noun1
    token_noun1.children.append(token_prep)
    token_verb.head = token_prep
    token_prep.children.append(token_verb)
    mock_doc = mock_doc_factory([token_noun1, token_prep, token_verb])
    return mock_doc[0:2] # Span 'fear of'
    
@pytest.fixture
def nonfinite_head_outside(spacy_vocab, mock_doc_factory):
    """Head noun outside span for nonfinite clause."""
    token_noun = create_mock_token(text="plan", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=0) # Outside
    token_to = create_mock_token(text="to", pos="PART", tag="TO", dep="mark", vocab=spacy_vocab, i=1)
    token_verb = create_mock_token(text="leave", pos="VERB", tag="VB", dep="acl", vocab=spacy_vocab, i=2)
    token_verb.head = token_noun # Head 'plan' outside
    token_noun.children.append(token_verb)
    token_to.head = token_verb
    token_verb.children.append(token_to)
    mock_doc = mock_doc_factory([token_noun, token_to, token_verb])
    return mock_doc[1:3] # Span 'to leave'


# --- Fixtures for analyze_single_np Tests (Combinations) ---

@pytest.fixture
def det_adj_noun_span(spacy_vocab, mock_doc_factory):
    """Span: DET + ADJ + NOUN (e.g., 'the happy cat')"""
    token_det = create_mock_token(text="the", pos="DET", tag="DT", dep="det", vocab=spacy_vocab, i=0)
    token_adj = create_mock_token(text="happy", pos="ADJ", tag="JJ", dep="amod", vocab=spacy_vocab, i=1)
    token_noun = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=2)

    token_det.head = token_noun
    token_noun.children.append(token_det)
    token_adj.head = token_noun
    token_noun.children.append(token_adj)

    mock_doc = mock_doc_factory([token_det, token_adj, token_noun])
    return mock_doc[0:3]

@pytest.fixture
def compound_possessive_span(spacy_vocab, mock_doc_factory):
    """Span: NOUN(compound) + NOUN(poss) + 's + NOUN (e.g., 'apple pie's price')"""
    token_noun1 = create_mock_token(text="apple", pos="NOUN", tag="NN", dep="compound", vocab=spacy_vocab, i=0)
    token_noun2 = create_mock_token(text="pie", pos="NOUN", tag="NN", dep="poss", vocab=spacy_vocab, i=1)
    token_poss = create_mock_token(text="'s", pos="PART", tag="POS", dep="case", vocab=spacy_vocab, i=2)
    token_noun3 = create_mock_token(text="price", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=3)

    token_noun1.head = token_noun2 # apple -> pie
    token_noun2.children.append(token_noun1)
    token_noun2.head = token_noun3 # pie -> price (possessive link)
    token_noun3.children.append(token_noun2)
    token_poss.head = token_noun2 # 's -> pie
    token_noun2.children.append(token_poss)

    mock_doc = mock_doc_factory([token_noun1, token_noun2, token_poss, token_noun3])
    return mock_doc[0:4]

@pytest.fixture
def adj_noun_prep_span(spacy_vocab, mock_doc_factory):
    """Span: ADJ + NOUN + PREP + NOUN (e.g., 'happy cat in hat')"""
    token_adj = create_mock_token(text="happy", pos="ADJ", tag="JJ", dep="amod", vocab=spacy_vocab, i=0)
    token_noun1 = create_mock_token(text="cat", pos="NOUN", tag="NN", dep="ROOT", vocab=spacy_vocab, i=1)
    token_prep = create_mock_token(text="in", pos="ADP", tag="IN", dep="prep", vocab=spacy_vocab, i=2)
    token_noun2 = create_mock_token(text="hat", pos="NOUN", tag="NN", dep="pobj", vocab=spacy_vocab, i=3)

    token_adj.head = token_noun1
    token_noun1.children.append(token_adj)
    token_prep.head = token_noun1
    token_noun1.children.append(token_prep)
    token_noun2.head = token_prep
    token_prep.children.append(token_noun2)

    mock_doc = mock_doc_factory([token_adj, token_noun1, token_prep, token_noun2])
    return mock_doc[0:4]

@pytest.fixture
def quantified_compound_span(spacy_vocab, mock_doc_factory):
    """Span: NUM + NOUN(compound) + NOUN (e.g., 'two apple pies')"""
    token_num = create_mock_token(text="two", pos="NUM", tag="CD", dep="nummod", vocab=spacy_vocab, i=0)
    token_noun1 = create_mock_token(text="apple", pos="NOUN", tag="NN", dep="compound", vocab=spacy_vocab, i=1)
    token_noun2 = create_mock_token(text="pies", pos="NOUN", tag="NNS", dep="ROOT", vocab=spacy_vocab, i=2)

    token_num.head = token_noun2
    token_noun2.children.append(token_num)
    token_noun1.head = token_noun2
    token_noun2.children.append(token_noun1)

    mock_doc = mock_doc_factory([token_num, token_noun1, token_noun2])
    return mock_doc[0:3]

# Use relative_clause_wdt_span which also has a determiner
@pytest.fixture
def det_noun_relative_clause_span(relative_clause_wdt_span):
    """Reuse relative_clause_wdt_span for combined test (det + relcl)"""
    return relative_clause_wdt_span

# Use coordinate_np_span fixture
@pytest.fixture
def simple_coordinate_span(coordinate_np_span):
    """Reuse coordinate_np_span for analyze test."""
    return coordinate_np_span

# Use standalone_noun_span fixture
@pytest.fixture
def simple_noun_span(standalone_noun_span):
    """Reuse standalone_noun_span for analyze test."""
    return standalone_noun_span

# Use pronoun_span fixture
@pytest.fixture
def simple_pronoun_span(pronoun_span):
    """Reuse pronoun_span for analyze test."""
    return pronoun_span


# --- Unit Tests for ANPEAnalyzer ---

def test_detect_pronoun_positive(analyzer, pronoun_span):
    """Test that _detect_pronoun correctly identifies a pronoun span."""
    assert analyzer._detect_pronoun(pronoun_span) is True

def test_detect_pronoun_negative_non_pronoun(analyzer, non_pronoun_span):
    """Test that _detect_pronoun rejects a non-pronoun span."""
    assert analyzer._detect_pronoun(non_pronoun_span) is False

def test_detect_pronoun_negative_multi_token(analyzer, multi_token_span):
    """Test that _detect_pronoun rejects a multi-token span."""
    assert analyzer._detect_pronoun(multi_token_span) is False

def test_detect_standalone_noun_positive(analyzer, standalone_noun_span):
    """Test that _detect_standalone_noun identifies a single noun span."""
    assert analyzer._detect_standalone_noun(standalone_noun_span) is True

def test_detect_standalone_propn_positive(analyzer, standalone_propn_span):
    """Test that _detect_standalone_noun identifies a single proper noun span."""
    assert analyzer._detect_standalone_noun(standalone_propn_span) is True

def test_detect_standalone_noun_negative_pronoun(analyzer, pronoun_span):
    """Test that _detect_standalone_noun rejects a pronoun span."""
    assert analyzer._detect_standalone_noun(pronoun_span) is False

def test_detect_standalone_noun_negative_multi_token(analyzer, multi_token_span):
    """Test that _detect_standalone_noun rejects a multi-token span."""
    assert analyzer._detect_standalone_noun(multi_token_span) is False

def test_detect_determiner_np_positive_noun(analyzer, determiner_np_span):
    """Test _detect_determiner_np identifies DET + NOUN within span."""
    assert analyzer._detect_determiner_np(determiner_np_span) is True

def test_detect_determiner_np_positive_propn(analyzer, determiner_np_propn_span):
    """Test _detect_determiner_np identifies DET + PROPN within span."""
    assert analyzer._detect_determiner_np(determiner_np_propn_span) is True

def test_detect_determiner_np_negative_no_det(analyzer, no_determiner_span):
    """Test _detect_determiner_np rejects span with noun but no determiner."""
    assert analyzer._detect_determiner_np(no_determiner_span) is False

def test_detect_determiner_np_negative_det_outside(analyzer, determiner_outside_span):
    """Test _detect_determiner_np rejects span where DET is outside."""
    # The span only contains 'cat', the DET 'the' is outside.
    assert analyzer._detect_determiner_np(determiner_outside_span) is False
    
def test_detect_determiner_np_negative_head_outside(analyzer, determiner_head_outside_span):
    """Test _detect_determiner_np rejects span where DET's head (NOUN) is outside."""
    # The span contains 'the', its head 'cat' is outside.
    assert analyzer._detect_determiner_np(determiner_head_outside_span) is False

# --- Tests for _detect_adjectival_np ---

def test_detect_adjectival_np_positive_adj(analyzer, adjectival_np_span):
    """Test _detect_adjectival_np identifies ADJ + NOUN within span."""
    assert analyzer._detect_adjectival_np(adjectival_np_span) is True

def test_detect_adjectival_np_positive_participle(analyzer, participle_modifier_np_span):
    """Test _detect_adjectival_np identifies VERB(amod) + NOUN within span."""
    assert analyzer._detect_adjectival_np(participle_modifier_np_span) is True

def test_detect_adjectival_np_negative_modifier_outside(analyzer, adjectival_modifier_outside_span):
    """Test _detect_adjectival_np rejects span where modifier is outside."""
    assert analyzer._detect_adjectival_np(adjectival_modifier_outside_span) is False

def test_detect_adjectival_np_negative_head_outside(analyzer, adjectival_head_outside_span):
    """Test _detect_adjectival_np rejects span where head noun is outside."""
    assert analyzer._detect_adjectival_np(adjectival_head_outside_span) is False

def test_detect_adjectival_np_negative_wrong_dep(analyzer, non_amod_modifier_span):
    """Test _detect_adjectival_np rejects span where dependency is not 'amod'."""
    assert analyzer._detect_adjectival_np(non_amod_modifier_span) is False

# --- Tests for _detect_prepositional_np ---

def test_detect_prepositional_np_positive(analyzer, prepositional_np_span):
    """Test _detect_prepositional_np identifies NOUN + PREP + POBJ within span."""
    assert analyzer._detect_prepositional_np(prepositional_np_span) is True

def test_detect_prepositional_np_negative_prep_outside(analyzer, prepositional_prep_outside_span):
    """Test _detect_prepositional_np rejects span where PREP is outside."""
    assert analyzer._detect_prepositional_np(prepositional_prep_outside_span) is False

def test_detect_prepositional_np_negative_head_outside(analyzer, prepositional_head_outside_span):
    """Test _detect_prepositional_np rejects span where head NOUN is outside."""
    assert analyzer._detect_prepositional_np(prepositional_head_outside_span) is False

def test_detect_prepositional_np_negative_pobj_outside(analyzer, prepositional_obj_outside_span):
    """Test _detect_prepositional_np rejects span where POBJ is outside."""
    assert analyzer._detect_prepositional_np(prepositional_obj_outside_span) is False

def test_detect_prepositional_np_negative_no_pobj(analyzer, prepositional_no_pobj_span):
    """Test _detect_prepositional_np rejects span where PREP has no POBJ child within span."""
    assert analyzer._detect_prepositional_np(prepositional_no_pobj_span) is False

# --- Tests for _detect_compound_noun ---

def test_detect_compound_noun_positive(analyzer, compound_noun_span):
    """Test _detect_compound_noun identifies NOUN(compound) + NOUN."""
    assert analyzer._detect_compound_noun(compound_noun_span) is True

def test_detect_compound_noun_negative_modifier_outside(analyzer, compound_modifier_outside_span):
    """Test _detect_compound_noun rejects span where modifier noun is outside."""
    assert analyzer._detect_compound_noun(compound_modifier_outside_span) is False

def test_detect_compound_noun_negative_head_outside(analyzer, compound_head_outside_span):
    """Test _detect_compound_noun rejects span where head noun is outside."""
    assert analyzer._detect_compound_noun(compound_head_outside_span) is False

def test_detect_compound_noun_negative_wrong_dep(analyzer, non_compound_dep_span):
    """Test _detect_compound_noun rejects span where dependency is not 'compound'."""
    assert analyzer._detect_compound_noun(non_compound_dep_span) is False

# --- Tests for _detect_possessive_np ---

def test_detect_possessive_np_positive_s(analyzer, possessive_s_span):
    """Test _detect_possessive_np identifies possessive with 's."""
    assert analyzer._detect_possessive_np(possessive_s_span) is True

def test_detect_possessive_np_positive_pronoun(analyzer, possessive_pronoun_span):
    """Test _detect_possessive_np identifies possessive with PRP$."""
    assert analyzer._detect_possessive_np(possessive_pronoun_span) is True

def test_detect_possessive_np_positive_poss_dep(analyzer, possessive_dep_span):
    """Test _detect_possessive_np identifies possessive via 'poss' dependency."""
    assert analyzer._detect_possessive_np(possessive_dep_span) is True
    
def test_detect_possessive_np_positive_unmarked(analyzer, possessive_unmarked_dep_span):
    """Test _detect_possessive_np identifies unmarked possessive via children check."""
    assert analyzer._detect_possessive_np(possessive_unmarked_dep_span) is True

def test_detect_possessive_np_negative_marker_outside(analyzer, possessive_marker_outside_span):
    """Test _detect_possessive_np rejects span where 's marker is outside."""
    assert analyzer._detect_possessive_np(possessive_marker_outside_span) is False
    
def test_detect_possessive_np_negative_head_outside(analyzer, possessive_head_outside_span):
    """Test _detect_possessive_np rejects span where head noun is outside."""
    assert analyzer._detect_possessive_np(possessive_head_outside_span) is False

# --- Tests for _detect_quantified_np ---

def test_detect_quantified_np_positive_num(analyzer, quantified_np_span):
    """Test _detect_quantified_np identifies NUM + NOUN."""
    assert analyzer._detect_quantified_np(quantified_np_span) is True

def test_detect_quantified_np_positive_nummod(analyzer, quantified_nummod_span):
    """Test _detect_quantified_np identifies dep=nummod + NOUN."""
    assert analyzer._detect_quantified_np(quantified_nummod_span) is True

def test_detect_quantified_np_negative_quantifier_outside(analyzer, quantified_quantifier_outside_span):
    """Test _detect_quantified_np rejects span where quantifier is outside."""
    assert analyzer._detect_quantified_np(quantified_quantifier_outside_span) is False

def test_detect_quantified_np_negative_head_outside(analyzer, quantified_head_outside_span):
    """Test _detect_quantified_np rejects span where head noun is outside."""
    assert analyzer._detect_quantified_np(quantified_head_outside_span) is False

def test_detect_quantified_np_negative_non_quantifier(analyzer, non_quantified_span):
    """Test _detect_quantified_np rejects span without relevant quantifier/dep."""
    assert analyzer._detect_quantified_np(non_quantified_span) is False

# --- Tests for _detect_coordinate_np ---

def test_detect_coordinate_np_positive(analyzer, coordinate_np_span):
    """Test _detect_coordinate_np identifies simple coordination Noun+CC+Noun."""
    assert analyzer._detect_coordinate_np(coordinate_np_span) is True

def test_detect_coordinate_np_positive_multiple(analyzer, coordinate_np_multiple_span):
    """Test _detect_coordinate_np identifies multiple coordination."""
    assert analyzer._detect_coordinate_np(coordinate_np_multiple_span) is True

def test_detect_coordinate_np_negative_cc_outside(analyzer, coordinate_cc_outside_span):
    """Test _detect_coordinate_np rejects span if CC is outside."""
    assert analyzer._detect_coordinate_np(coordinate_cc_outside_span) is False

def test_detect_coordinate_np_negative_conj_outside(analyzer, coordinate_conj_outside_span):
    """Test _detect_coordinate_np rejects span if CONJ is outside."""
    assert analyzer._detect_coordinate_np(coordinate_conj_outside_span) is False

def test_detect_coordinate_np_negative_no_cc(analyzer, coordinate_no_cc_span):
    """Test _detect_coordinate_np rejects span if CC is missing."""
    assert analyzer._detect_coordinate_np(coordinate_no_cc_span) is False

def test_detect_coordinate_np_negative_no_conj(analyzer, coordinate_no_conj_span):
    """Test _detect_coordinate_np rejects span if CONJ is missing."""
    assert analyzer._detect_coordinate_np(coordinate_no_conj_span) is False

# --- Tests for _detect_appositive_np ---

def test_detect_appositive_np_positive(analyzer, appositive_np_span):
    """Test _detect_appositive_np identifies apposition within span."""
    assert analyzer._detect_appositive_np(appositive_np_span) is True

def test_detect_appositive_np_negative_modifier_outside(analyzer, appositive_modifier_outside_span):
    """Test _detect_appositive_np rejects span if appositive modifier is outside."""
    assert analyzer._detect_appositive_np(appositive_modifier_outside_span) is False

def test_detect_appositive_np_negative_head_outside(analyzer, appositive_head_outside_span):
    """Test _detect_appositive_np rejects span if head noun is outside."""
    assert analyzer._detect_appositive_np(appositive_head_outside_span) is False

def test_detect_appositive_np_negative_wrong_dep(analyzer, non_appositive_dep_span):
    """Test _detect_appositive_np rejects span if 'appos' dependency is missing."""
    assert analyzer._detect_appositive_np(non_appositive_dep_span) is False

# --- Tests for _detect_relative_clause ---

def test_detect_relative_clause_positive_wdt(analyzer, relative_clause_wdt_span):
    """Test _detect_relative_clause finds clause with WDT pronoun."""
    assert analyzer._detect_relative_clause(relative_clause_wdt_span) is True

def test_detect_relative_clause_positive_relcl(analyzer, relative_clause_relcl_span):
    """Test _detect_relative_clause finds clause via relcl dependency."""
    assert analyzer._detect_relative_clause(relative_clause_relcl_span) is True

def test_detect_relative_clause_negative_pronoun_outside(analyzer, relative_clause_pronoun_outside):
    """Test _detect_relative_clause rejects if relative pronoun is outside span."""
    assert analyzer._detect_relative_clause(relative_clause_pronoun_outside) is False

def test_detect_relative_clause_negative_verb_outside(analyzer, relative_clause_verb_outside):
    """Test _detect_relative_clause rejects if relcl verb is outside span."""
    assert analyzer._detect_relative_clause(relative_clause_verb_outside) is False

# --- Tests for _detect_reduced_relative_clause ---

def test_detect_reduced_relative_clause_positive(analyzer, reduced_relative_clause_acl_span):
    """Test _detect_reduced_relative_clause finds acl-based reduced relative clause."""
    assert analyzer._detect_reduced_relative_clause(reduced_relative_clause_acl_span) is True

def test_detect_reduced_relative_clause_negative_acl_outside(analyzer, reduced_relative_clause_acl_outside_span):
    """Test _detect_reduced_relative_clause rejects if acl verb is outside span."""
    assert analyzer._detect_reduced_relative_clause(reduced_relative_clause_acl_outside_span) is False

def test_detect_reduced_relative_clause_negative_head_outside(analyzer, reduced_relative_clause_head_outside_span):
    """Test _detect_reduced_relative_clause rejects if head noun is outside span."""
    assert analyzer._detect_reduced_relative_clause(reduced_relative_clause_head_outside_span) is False

def test_detect_reduced_relative_clause_negative_has_rel_pron(analyzer, reduced_relative_clause_with_rel_pron):
    """Test _detect_reduced_relative_clause rejects if relative pronoun is present."""
    assert analyzer._detect_reduced_relative_clause(reduced_relative_clause_with_rel_pron) is False

# --- Tests for _detect_finite_complement ---

def test_detect_finite_complement_positive(analyzer, finite_complement_span):
    """Test _detect_finite_complement finds finite complement with comp-taking noun."""
    assert analyzer._detect_finite_complement(finite_complement_span) is True

def test_detect_finite_complement_negative_comp_outside(analyzer, finite_complement_comp_outside):
    """Test _detect_finite_complement rejects if complementizer is outside span."""
    assert analyzer._detect_finite_complement(finite_complement_comp_outside) is False

def test_detect_finite_complement_negative_verb_outside(analyzer, finite_complement_verb_outside):
    """Test _detect_finite_complement rejects if complement verb is outside span."""
    assert analyzer._detect_finite_complement(finite_complement_verb_outside) is False

def test_detect_finite_complement_negative_non_comp_noun(analyzer, finite_complement_non_comp_noun):
    """Test _detect_finite_complement rejects if head noun is not complement-taking."""
    assert analyzer._detect_finite_complement(finite_complement_non_comp_noun) is False

# --- Tests for _detect_nonfinite_complement ---

def test_detect_nonfinite_complement_positive_infinitive(analyzer, nonfinite_infinitive_span):
    """Test _detect_nonfinite_complement finds infinitive complement."""
    assert analyzer._detect_nonfinite_complement(nonfinite_infinitive_span) is True

def test_detect_nonfinite_complement_positive_gerund(analyzer, nonfinite_gerund_span):
    """Test _detect_nonfinite_complement finds gerund complement."""
    assert analyzer._detect_nonfinite_complement(nonfinite_gerund_span) is True

def test_detect_nonfinite_complement_negative_verb_outside(analyzer, nonfinite_infinitive_verb_outside):
    """Test _detect_nonfinite_complement rejects if infinitive verb is outside span."""
    assert analyzer._detect_nonfinite_complement(nonfinite_infinitive_verb_outside) is False

def test_detect_nonfinite_complement_negative_gerund_outside(analyzer, nonfinite_gerund_outside):
    """Test _detect_nonfinite_complement rejects if gerund verb is outside span."""
    assert analyzer._detect_nonfinite_complement(nonfinite_gerund_outside) is False
    
def test_detect_nonfinite_complement_negative_head_outside(analyzer, nonfinite_head_outside):
    """Test _detect_nonfinite_complement rejects if head noun is outside span."""
    assert analyzer._detect_nonfinite_complement(nonfinite_head_outside) is False

# --- Tests for analyze_single_np (Combinations) ---

def test_analyze_single_np_det_adj_noun(analyzer, det_adj_noun_span):
    """Test analyze_single_np identifies determiner and adjectival modifier."""
    expected = {"determiner", "adjectival_modifier"}
    result = analyzer.analyze_single_np(det_adj_noun_span)
    assert set(result) == expected

def test_analyze_single_np_compound_possessive(analyzer, compound_possessive_span):
    """Test analyze_single_np identifies compound and possessive."""
    # Structure: [compound(apple pie)]'s price -> possessive
    # Should detect compound within 'apple pie' part? NO, detector looks for token with 'compound' dep.
    # Should detect possessive based on 'pie' having poss dep or 's tag?
    # Let's check the possessive logic again: checks for POS tag or PRP$ or 'poss' dep *OR* noun child with 'poss' dep.
    # Here 'pie' has 'poss' dep relative to 'price'. 's has POS tag.
    expected = {"compound", "possessive"}
    result = analyzer.analyze_single_np(compound_possessive_span)
    assert set(result) == expected

def test_analyze_single_np_adj_noun_prep(analyzer, adj_noun_prep_span):
    """Test analyze_single_np identifies adjectival and prepositional modifiers."""
    expected = {"adjectival_modifier", "prepositional_modifier"}
    result = analyzer.analyze_single_np(adj_noun_prep_span)
    assert set(result) == expected

def test_analyze_single_np_quantified_compound(analyzer, quantified_compound_span):
    """Test analyze_single_np identifies quantified and compound structures."""
    expected = {"quantified", "compound"}
    result = analyzer.analyze_single_np(quantified_compound_span)
    assert set(result) == expected

def test_analyze_single_np_det_relative_clause(analyzer, det_noun_relative_clause_span):
    """Test analyze_single_np identifies determiner and relative clause."""
    expected = {"determiner", "relative_clause"}
    result = analyzer.analyze_single_np(det_noun_relative_clause_span)
    # Note: This might pick up other things depending on exact tags/deps, focus on main ones
    assert expected.issubset(set(result))

def test_analyze_single_np_coordinate(analyzer, simple_coordinate_span):
    """Test analyze_single_np identifies coordination."""
    expected = {"coordinated"}
    result = analyzer.analyze_single_np(simple_coordinate_span)
    assert set(result) == expected

def test_analyze_single_np_standalone_noun(analyzer, simple_noun_span):
    """Test analyze_single_np identifies standalone noun."""
    expected = {"standalone_noun"}
    result = analyzer.analyze_single_np(simple_noun_span)
    assert set(result) == expected
    
def test_analyze_single_np_pronoun(analyzer, simple_pronoun_span):
    """Test analyze_single_np identifies pronoun structure."""
    assert analyzer.analyze_single_np(simple_pronoun_span) == ["pronoun"]

