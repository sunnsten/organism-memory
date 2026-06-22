from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from organism.backbone.base import EncodedText

logger = logging.getLogger(__name__)


def text_embedding_from_encoded(encoded: EncodedText) -> Tensor:
    """
    Extract a normalised embedding from EncodedText.

    Uses mean pooling over hidden_states and L2-normalises the result.
    This is the single canonical way to obtain an embedding from text.

    Args:
        encoded: EncodedText with hidden_states

    Returns:
        Normalised embedding [D]
    """
    emb = encoded.hidden_states[0].mean(dim=0).detach().cpu()
    emb = emb / (emb.norm() + 1e-8)
    return emb


def cosine_similarity_pairwise(a: Tensor, b: Tensor) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: first vector [D]
        b: second vector [D]

    Returns:
        Cosine similarity in [-1, 1].
    """
    a_norm = a / (a.norm() + 1e-8)
    b_norm = b / (b.norm() + 1e-8)
    return float(torch.dot(a_norm, b_norm).item())


def cosine_similarity_batch(query: Tensor, keys: Tensor, dim: int = 1) -> Tensor:
    """
    Compute cosine similarity between a query vector and a batch of keys.

    Args:
        query: query vector [D] or [1, D]
        keys: key batch [K, D]
        dim: dimension along which to compute similarity

    Returns:
        Similarity tensor [K].
    """
    if query.dim() == 1:
        query = query.unsqueeze(0)  # [1, D]
    return F.cosine_similarity(query, keys, dim=dim)


def pairwise_cosine(a: Tensor, b: Tensor) -> Tensor:
    """
    Compute pairwise cosine similarity between two batches of vectors.

    Args:
        a: first batch [N, D]
        b: second batch [M, D]

    Returns:
        Similarity matrix [N, M].
    """
    a_norm = a / (a.norm(dim=-1, keepdim=True) + 1e-8)
    b_norm = b / (b.norm(dim=-1, keepdim=True) + 1e-8)
    return a_norm @ b_norm.T


def jaccard_similarity(set1: set[int], set2: set[int]) -> float:
    """
    Compute Jaccard similarity between two sets.

    Used to compare used_memories overlaps between experience blocks.

    Args:
        set1: first set (e.g. slot_indices from the first block)
        set2: second set (e.g. slot_indices from the second block)

    Returns:
        Jaccard similarity in [0, 1].
    """
    if not set1 and not set2:
        return 1.0  # both empty → treat as identical
    if not set1 or not set2:
        return 0.0  # one empty, one non-empty → no overlap

    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def serialize_embedding(emb: Tensor, dtype: str = "float16") -> tuple[bytes, int, bool, str]:
    """
    Serialise an embedding to binary format with metadata.
    Normalises before storing if not already close to unit norm (threshold 1e-2).

    Returns:
        Tuple (blob, dim, l2norm_flag, storage_dtype)
    """
    emb_cpu = emb.detach()
    if emb_cpu.device.type != "cpu":
        emb_cpu = emb_cpu.to("cpu")
    if emb_cpu.ndim != 1:
        emb_cpu = emb_cpu.flatten()
    dim = emb_cpu.numel()
    emb_original = emb_cpu.clone()

    norm = emb_cpu.norm().item()
    if not np.isfinite(norm) or norm < 1e-12:
        logger.warning("Invalid embedding norm=%s; storing as-is", norm)
    else:
        if abs(norm - 1.0) > 1e-2:
            emb_cpu = emb_cpu / (norm + 1e-8)
            if not torch.isfinite(emb_cpu).all():
                logger.warning("Embedding became non-finite after normalization; storing original")
                emb_cpu = emb_original

    DTYPE_MAP = {
        "float16": (torch.float16, np.float16, "float16"),
        "float32": (torch.float32, np.float32, "float32"),
        "bfloat16": (torch.float16, np.float16, "float16"),
    }
    torch_dt, np_dt, storage_dtype = DTYPE_MAP.get(dtype, (torch.float16, np.float16, "float16"))
    if dtype not in DTYPE_MAP:
        logger.warning("Unknown dtype '%s', falling back to float16", dtype)

    if emb_cpu.dtype != torch_dt:
        emb_cpu = emb_cpu.to(torch_dt)
    arr = emb_cpu.numpy().astype(np_dt, copy=False)
    final_norm = float(np.linalg.norm(arr.astype(np.float32)))
    l2norm_flag = abs(final_norm - 1.0) < 1e-2
    blob = arr.tobytes()
    return blob, dim, l2norm_flag, storage_dtype


def deserialize_embedding(
    blob: bytes,
    dim: int,
    dtype: str = "float16",
    l2norm: bool = False,
) -> Tensor:
    """
    Deserialise an embedding from binary format.

    Returns:
        Embedding tensor [D].
    """
    if dtype in ("float16", "bfloat16"):
        np_dtype = np.float16
    else:
        np_dtype = np.float32

    arr = np.frombuffer(blob, dtype=np_dtype).reshape(dim)
    emb = torch.from_numpy(arr.copy())

    if not l2norm:
        norm = emb.norm().item()
        if not np.isfinite(norm) or norm < 1e-12:
            logger.warning("Invalid embedding norm=%s during deserialization; skipping normalization", norm)
        elif abs(norm - 1.0) > 1e-2:
            emb = emb / (norm + 1e-8)
            if not torch.isfinite(emb).all():
                logger.warning("Embedding became non-finite after normalization during deserialization")
                arr = np.frombuffer(blob, dtype=np_dtype).reshape(dim)
                emb = torch.from_numpy(arr.copy())
    return emb


def normalize_embedding(emb: Tensor) -> Tensor:
    """Normalise an embedding to unit norm."""
    norm = emb.norm().item()
    if not np.isfinite(norm) or norm < 1e-12:
        logger.warning("Invalid embedding norm=%s in normalize_embedding; returning original", norm)
        return emb
    if abs(norm - 1.0) < 1e-2:
        return emb
    normalized = emb / (norm + 1e-8)
    if not torch.isfinite(normalized).all():
        logger.warning("Embedding became non-finite after normalization; returning original")
        return emb
    return normalized


__all__ = [
    "text_embedding_from_encoded",
    "cosine_similarity_pairwise",
    "cosine_similarity_batch",
    "pairwise_cosine",
    "jaccard_similarity",
    "serialize_embedding",
    "deserialize_embedding",
    "normalize_embedding",
]
