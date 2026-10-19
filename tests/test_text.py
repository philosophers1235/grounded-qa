from groundedqa.text import query_terms, sentence_spans, stem, tokenize


def test_stem_conflates_common_inflections():
    assert stem("installed") == stem("installation") == stem("install")
    assert stem("boilers") == stem("boiler")
    assert stem("serviced") == stem("servicing") == stem("service")
    assert stem("fitted") == stem("fitting") == stem("fit")
    assert stem("warranties") == stem("warranty")


def test_stem_leaves_short_words_and_numbers_alone():
    assert stem("gas") == "gas"
    assert stem("e10") == "e10"
    assert stem("1.5") == "1.5"


def test_tokenize_drops_stopwords_and_question_phrasing():
    assert tokenize("How long is the warranty on the Linea 24?") == ["warranty", "linea", "24"]


def test_query_terms_are_unique_and_ordered():
    assert query_terms("pressure pressure valve") == ["pressur", "valv"]


def test_sentence_spans_split_sentences_and_lines():
    text = "First sentence. Second one at 1.5 bar.\n- a list item\n\nLast."
    sentences = [text[s:e] for s, e in sentence_spans(text)]
    assert sentences == ["First sentence.", "Second one at 1.5 bar.", "- a list item", "Last."]
