import numpy as np

from app.services.embeddings import HashingEmbedder


def test_hashing_embedder_shape_and_norm():
    emb = HashingEmbedder(dim=128)
    matrix = emb.embed(["hello world", "another document about invoices"])
    assert matrix.shape == (2, 128)
    assert matrix.dtype == np.float32
    norms = np.linalg.norm(matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_hashing_embedder_is_deterministic():
    a = HashingEmbedder(dim=256).embed_one("total amount due on invoice")
    b = HashingEmbedder(dim=256).embed_one("total amount due on invoice")
    assert np.allclose(a, b)


def test_hashing_embedder_semantic_ordering():
    emb = HashingEmbedder(dim=512)
    q = emb.embed_one("invoice total amount")
    related = emb.embed_one("the invoice total amount is due")
    unrelated = emb.embed_one("a poem about the ocean and the sky")
    assert float(q @ related) > float(q @ unrelated)


def test_empty_text_is_safe():
    vec = HashingEmbedder(dim=64).embed_one("")
    assert vec.shape == (64,)
    assert not np.isnan(vec).any()
