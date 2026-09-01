import numpy as np
from sonicdj.db.vector_store import VectorStore


def test_vector_store_crud_and_search(temp_db):
    store = VectorStore(temp_db)

    # 1. Upsert vectors
    v1 = np.ones(512, dtype=np.float32) / np.sqrt(512)
    v2 = np.zeros(512, dtype=np.float32); v2[0] = 1.0
    v3 = np.zeros(512, dtype=np.float32); v3[1] = 1.0

    store.upsert_embedding(1, v1)
    store.upsert_embedding(2, v2)
    store.upsert_embedding(3, v3)

    # 2. Get embedding
    ret_v2 = store.get_embedding(2)
    assert ret_v2 is not None
    assert np.allclose(ret_v2, v2, atol=1e-4)

    # 3. Search k-NN
    # Query matching v2
    results = store.search_knn(v2, top_k=2)
    assert len(results) == 2
    assert results[0][0] == 2  # Best match is track 2
    assert abs(results[0][1] - 1.0) < 1e-4

    # 4. Search with candidate filter
    results_filtered = store.search_knn(v2, top_k=5, candidate_track_ids={1, 3})
    assert len(results_filtered) == 2
    assert 2 not in [r[0] for r in results_filtered]

    # 5. Batch upsert
    store.batch_upsert_embeddings([(4, v1), (5, v2)])
    assert store.get_embedding(4) is not None
