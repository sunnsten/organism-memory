from __future__ import annotations

import pytest
import torch
from torch import Tensor

from organism.shared.utils.embeddings import (
    cosine_similarity_pairwise,
    cosine_similarity_batch,
    pairwise_cosine,
    jaccard_similarity,
)
from organism.shared.utils import (
    cosine_similarity_pairwise as cs_pairwise,
    jaccard_similarity as jaccard,
)
from organism.shared.utils.attention_utils import (
    MemSpan,
    aggregate_mem_attention,
    pick_heads,
    sample_steps,
)


@pytest.mark.unit
def test_cosine_similarity_pairwise():
    """cosine_similarity_pairwise from shared.utils.embeddings and similarity."""
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([1.0, 0.0, 0.0])
    assert abs(cosine_similarity_pairwise(a, b) - 1.0) < 1e-6
    assert cs_pairwise is cosine_similarity_pairwise

    c = torch.tensor([0.0, 1.0, 0.0])
    assert abs(cosine_similarity_pairwise(a, c) - 0.0) < 1e-6


@pytest.mark.unit
def test_jaccard_similarity():
    """jaccard_similarity from shared.utils."""
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1, 2}, set()) == 0.0
    assert jaccard({1, 2}, {1, 2}) == 1.0
    assert abs(jaccard({1, 2}, {2, 3}) - 1.0 / 3.0) < 1e-6  # 1 in intersection, 3 in union


@pytest.mark.unit
def test_cosine_similarity_batch_and_pairwise_cosine():
    """cosine_similarity_batch and pairwise_cosine."""
    query = torch.randn(8)
    keys = torch.randn(5, 8)
    batch_sims = cosine_similarity_batch(query, keys)
    assert batch_sims.shape == (5,)

    a_batch = torch.randn(3, 8)
    b_batch = torch.randn(4, 8)
    mat = pairwise_cosine(a_batch, b_batch)
    assert mat.shape == (3, 4)


@pytest.mark.unit
def test_text_embedding_from_encoded():
    """text_embedding_from_encoded requires EncodedText (backbone)."""
    from organism.shared.utils.embeddings import text_embedding_from_encoded
    from organism.backbone.base import EncodedText

    # EncodedText: hidden_states [1, L, D], attention_mask [1, L], seq_len, d_model
    hidden = torch.randn(1, 10, 32)
    mask = torch.ones(1, 10)
    encoded = EncodedText(
        hidden_states=hidden,
        attention_mask=mask,
        seq_len=10,
        d_model=32,
    )
    emb = text_embedding_from_encoded(encoded)
    assert emb.shape == (32,)
    assert abs(emb.norm().item() - 1.0) < 1e-5


@pytest.mark.unit
def test_attention_utils_mem_span_and_aggregate():
    """MemSpan and aggregate_mem_attention."""
    attn = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    spans = [MemSpan(mem_id=1, start=0, end=1), MemSpan(mem_id=2, start=2, end=4)]
    scores = aggregate_mem_attention(attn, spans)
    assert 1 in scores
    assert 2 in scores
    assert scores[1] == pytest.approx(0.1 + 0.2)
    assert scores[2] == pytest.approx(0.3 + 0.4 + 0.5)


@pytest.mark.unit
def test_pick_heads():
    """pick_heads for 2D and 3D attention."""
    # [heads, prompt_len]
    attn_2d = torch.randn(4, 20)
    out = pick_heads(attn_2d)
    assert out.shape == (20,)
    out_selected = pick_heads(attn_2d, heads_to_use=[0, 1])
    assert out_selected.shape == (20,)

    # [layers, heads, prompt_len]
    attn_3d = torch.randn(2, 4, 20)
    out3 = pick_heads(attn_3d)
    assert out3.shape == (20,)


@pytest.mark.unit
def test_sample_steps():
    """sample_steps."""
    steps = sample_steps(10, collect_every=2)
    assert steps == [0, 2, 4, 6, 8]
    steps_lim = sample_steps(100, collect_every=1, max_collect_steps=5)
    assert len(steps_lim) == 5
