from app.services.embeddings import get_embedder
from app.services.vector_store import VectorRecord, get_vector_store, total_indexed_vectors


def test_per_tenant_vector_isolation(db):
    emb = get_embedder()
    acme = get_vector_store("acme")
    globex = get_vector_store("globex")
    acme.clear()
    globex.clear()

    recs = [VectorRecord(id="c1", document_id="d1", chunk_index=0, filename="f", text="invoice total", tenant="acme")]
    acme.upsert(recs, emb.embed(["invoice total"]))

    assert acme.count() == 1
    assert globex.count() == 0
    assert total_indexed_vectors() >= 1

    hits = acme.search(emb.embed_one("invoice total"), top_k=1)
    assert hits and hits[0].record.tenant == "acme"
