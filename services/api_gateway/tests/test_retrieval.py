import numpy as np

from app.services.sparse_index import BM25Okapi, SparseIndex, tokenize
from app.services.vector_store import LocalVectorStore, VectorRecord
from app.services.embeddings import HashingEmbedder
from app.services.retrieval import reciprocal_rank_fusion, answer_question
from tests.conftest import make_document


# ---- BM25 ----------------------------------------------------------------
def test_bm25_ranks_relevant_doc_first():
    corpus = [tokenize(t) for t in [
        "the invoice total amount is fifteen hundred dollars",
        "a contract governing law clause in delaware",
        "claim filed by the policy holder",
    ]]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize("invoice total amount"))
    assert int(np.argmax(scores)) == 0
    assert scores[0] > 0


def test_sparse_index_search():
    recs = [
        VectorRecord(id="1", document_id="d1", chunk_index=0, filename="a", text="invoice total amount due"),
        VectorRecord(id="2", document_id="d2", chunk_index=0, filename="b", text="governing law delaware contract"),
    ]
    idx = SparseIndex()
    idx.build(recs, signature=(2, "2"))
    hits = idx.search("invoice amount", top_k=2)
    assert hits[0][0].document_id == "d1"


# ---- Vector store --------------------------------------------------------
def test_local_vector_store_roundtrip(tmp_path):
    emb = HashingEmbedder(dim=128)
    store = LocalVectorStore(dim=128, index_dir=tmp_path)
    texts = ["invoice total amount", "contract governing law", "insurance claim amount"]
    recs = [VectorRecord(id=f"c{i}", document_id=f"d{i}", chunk_index=0, filename=f"f{i}", text=t)
            for i, t in enumerate(texts)]
    store.upsert(recs, emb.embed(texts))
    assert store.count() == 3

    hits = store.search(emb.embed_one("what is the invoice total"), top_k=1)
    assert hits[0].record.document_id == "d0"

    # persistence
    reloaded = LocalVectorStore(dim=128, index_dir=tmp_path)
    assert reloaded.count() == 3

    # dedup on re-upsert + delete
    store.upsert([recs[0]], emb.embed([texts[0]]))
    assert store.count() == 3
    store.delete_document("d1")
    assert store.count() == 2


# ---- Fusion --------------------------------------------------------------
class _Hit:
    def __init__(self, rid, score):
        self.record = VectorRecord(id=rid, document_id=rid, chunk_index=0, filename=rid, text=rid)
        self.score = score


def test_rrf_rewards_agreement():
    dense = [_Hit("a", 0.9), _Hit("b", 0.5)]
    sparse = [("_", 0.0)]  # placeholder, replaced below
    sparse = [(_Hit("a", 0).record, 3.0), (_Hit("c", 0).record, 1.0)]
    fused = reciprocal_rank_fusion(dense, sparse, k=60, w_dense=1.0, w_sparse=1.0)
    ids = [e["record"].id for e in fused]
    # "a" appears top of both lists -> must rank first
    assert ids[0] == "a"
    assert set(ids) == {"a", "b", "c"}


# ---- End to end ----------------------------------------------------------
def test_answer_question_grounds_in_correct_document(db):
    make_document(db, "invoice.txt",
                  "Invoice Number INV-1024. Vendor Nova Industrial Supplies. Total Amount 1375.00 due soon.",
                  doc_type="invoice")
    make_document(db, "contract.md",
                  "This agreement effective 2026. Governing law is Delaware. Renewal term twelve months.",
                  doc_type="contract")

    res = answer_question(db, "What is the invoice total amount?", top_k=2)
    assert res.evidence
    assert res.evidence[0].filename == "invoice.txt"
    assert "1375" in res.answer
    assert res.strategy.embedding_backend == "hashing"
    assert res.strategy.vector_backend == "local"
    assert res.strategy.fused_candidates >= 1


def test_answer_question_empty_index(db):
    res = answer_question(db, "anything", top_k=3)
    assert res.evidence == []
    assert "No documents" in res.answer
