import sys
import types

import numpy as np
import pytest

from groundedqa.bm25 import BM25
from groundedqa.dense import load_embedder, reciprocal_rank_fusion
from groundedqa.index import Index, IndexNotFound, build_index
from groundedqa.retrieval import Retriever


@pytest.fixture
def corpus(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "warranty.md").write_text(
        "# Warranty\n\n## Period\n\nThe boiler warranty lasts 24 months from installation.\n\n"
        "## Exclusions\n\nFrost damage and limescale are not covered.\n",
        encoding="utf-8",
    )
    (docs / "returns.md").write_text(
        "# Returns\n\n## Unused parts\n\nUnused parts can be returned within 30 days. A restocking fee of 15% applies.\n",
        encoding="utf-8",
    )
    return docs


def test_bm25_prefers_documents_with_the_rarer_term():
    bm25 = BM25([["boiler", "pressure"], ["boiler", "warranty"], ["boiler", "flue"]])
    ranked = bm25.top(["boiler", "warranty"], k=3)
    assert ranked[0][0] == 1
    assert ranked[0][1] > ranked[1][1]


def test_bm25_normalises_for_document_length():
    short = ["warranty", "months"]
    long = ["warranty"] + ["filler"] * 50
    bm25 = BM25([short, long, ["other"]])
    scores = bm25.scores(["warranty"])
    assert scores[0] > scores[1] > 0


def test_bm25_ignores_unknown_terms_and_keeps_idf_positive():
    bm25 = BM25([["a1", "b1"], ["a1"], ["a1", "c1"]])
    assert bm25.top(["zzz"], k=3) == []
    assert bm25.idf("a1") > 0


def test_question_answered_in_corpus_is_supported(corpus):
    retriever = Retriever(build_index(corpus), min_support=0.5)
    result = retriever.search("How long is the boiler warranty?")
    assert result.supported
    assert result.hits[0].chunk.heading == "Period"
    assert result.hits[0].support == pytest.approx(1.0)


def test_question_outside_corpus_is_refused_by_threshold(corpus):
    retriever = Retriever(build_index(corpus), min_support=0.5)
    result = retriever.search("Do you sell solar panels for the boiler?")
    assert result.hits, "'boiler' still matches something"
    assert result.best_support < 0.5
    assert not result.supported


def test_question_with_no_matching_terms_has_no_hits(corpus):
    result = Retriever(build_index(corpus)).search("What is the meaning of life?")
    assert result.hits == []
    assert not result.supported


def test_index_survives_a_round_trip(corpus, tmp_path):
    index = build_index(corpus)
    index.save(tmp_path / "idx")
    loaded = Index.load(tmp_path / "idx")
    assert loaded.chunks == index.chunks
    question = "restocking fee for returned parts"
    assert Retriever(loaded).search(question).hits == Retriever(index).search(question).hits


def test_loading_a_missing_index_says_what_to_do(tmp_path):
    with pytest.raises(IndexNotFound, match="gqa index"):
        Index.load(tmp_path / "nothing-here")


class KeywordEmbedder:
    """Maps texts onto a few hand-picked concepts, so 'guarantee' lands near 'warranty'."""

    name = "keywords"
    concepts = [("warranty", "guarantee"), ("return", "refund"), ("frost", "freeze")]

    def embed(self, texts):
        return np.array(
            [[float(any(word in text.lower() for word in group)) for group in self.concepts] for text in texts]
        )


def test_dense_retrieval_finds_synonyms_bm25_misses(corpus):
    index = build_index(corpus, embedder=KeywordEmbedder(), embedder_spec="tests:KeywordEmbedder")
    assert index.vectors.shape == (len(index.chunks), 3)

    question = "Is there a guarantee?"
    lexical_only = Retriever(index).search(question)
    hybrid = Retriever(index, embedder=KeywordEmbedder()).search(question)
    assert lexical_only.hits == []
    assert hybrid.hits[0].chunk.doc_id == "warranty"


def test_reciprocal_rank_fusion_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    assert [item for item, _ in fused][:2] == [1, 3]


def test_load_embedder_from_module_spec(monkeypatch):
    module = types.ModuleType("fake_embedders")
    module.make = KeywordEmbedder
    monkeypatch.setitem(sys.modules, "fake_embedders", module)
    assert load_embedder("fake_embedders:make").name == "keywords"
    with pytest.raises(ValueError):
        load_embedder("no-colon-here")
