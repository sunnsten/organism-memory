import numpy as np
from unittest.mock import patch, MagicMock


def test_embed_single_text():
    """VLLMEmbedder calls /v1/embeddings and returns L2-normalized vector."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3] * 341 + [0.1]}],  # 1024d
        "model": "Qwen/Qwen3-Embedding-0.6B",
    }

    with patch("httpx.Client.post", return_value=mock_response):
        from organism.core.embedding.vllm_embedder import VLLMEmbedder
        embedder = VLLMEmbedder(
            base_url="http://localhost:8002/v1",
            model_name="Qwen/Qwen3-Embedding-0.6B",
            dim=1024,
        )
        vec = embedder.embed("hello world")

    assert vec.shape == (1024,)
    assert vec.dtype == np.float32
    norm = float(np.linalg.norm(vec))
    assert abs(norm - 1.0) < 0.01  # L2-normalized


def test_embed_batch():
    """embed_batch returns list of L2-normalized vectors."""
    raw_vecs = [[0.5] * 1024, [0.3] * 1024]
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": v} for v in raw_vecs]
    }

    with patch("httpx.Client.post", return_value=mock_response):
        from organism.core.embedding.vllm_embedder import VLLMEmbedder
        embedder = VLLMEmbedder(
            base_url="http://localhost:8002/v1",
            model_name="Qwen/Qwen3-Embedding-0.6B",
            dim=1024,
        )
        vecs = embedder.embed_batch(["text1", "text2"])

    assert len(vecs) == 2
    for v in vecs:
        assert v.shape == (1024,)
        assert abs(float(np.linalg.norm(v)) - 1.0) < 0.01


def test_embed_batch_empty():
    """embed_batch with empty input returns empty list."""
    from organism.core.embedding.vllm_embedder import VLLMEmbedder
    embedder = VLLMEmbedder(
        base_url="http://localhost:8002/v1",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        dim=1024,
    )
    assert embedder.embed_batch([]) == []


def test_dim_property():
    from organism.core.embedding.vllm_embedder import VLLMEmbedder
    embedder = VLLMEmbedder(
        base_url="http://localhost:8002/v1",
        model_name="Qwen/Qwen3-Embedding-0.6B",
        dim=1024,
    )
    assert embedder.dim == 1024
