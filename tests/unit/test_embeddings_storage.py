from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from organism.shared.utils.embeddings import (
    deserialize_embedding,
    normalize_embedding,
    serialize_embedding,
)


class TestSerializeEmbedding:
    """Tests for serialize_embedding()."""

    def test_basic_serialization_float16(self):
        """Test basic serialization with float16."""
        emb = torch.randn(128)
        emb = emb / emb.norm()  # Normalize
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="float16")
        
        assert dim == 128
        assert l2norm_flag is True
        assert storage_dtype == "float16"
        assert isinstance(blob, bytes)
        assert len(blob) == 128 * 2  # float16 = 2 bytes

    def test_basic_serialization_float32(self):
        """Test basic serialization with float32."""
        emb = torch.randn(64)
        emb = emb / emb.norm()
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="float32")
        
        assert dim == 64
        assert l2norm_flag is True
        assert storage_dtype == "float32"
        assert len(blob) == 64 * 4  # float32 = 4 bytes

    def test_bfloat16_conversion(self):
        """Test bfloat16 conversion to float16."""
        emb = torch.randn(32, dtype=torch.bfloat16)
        emb = emb / emb.norm()
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="bfloat16")
        
        assert dim == 32
        assert storage_dtype == "float16"  # bfloat16 → float16
        assert len(blob) == 32 * 2

    def test_non_normalized_embedding(self):
        """Test serialization of non-normalized embedding."""
        emb = torch.randn(64) * 5.0  # Not normalized
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="float16")
        
        # Should normalize and set flag
        assert l2norm_flag is True
        # Verify it's actually normalized after deserialization
        restored = deserialize_embedding(blob, dim, storage_dtype, l2norm_flag)
        norm = restored.norm().item()
        assert abs(norm - 1.0) < 1e-2

    def test_already_normalized(self):
        """Test serialization of already normalized embedding."""
        emb = torch.randn(64)
        emb = emb / emb.norm()
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="float16")
        
        assert l2norm_flag is True

    def test_nan_embedding(self):
        """Test handling of NaN embedding."""
        emb = torch.tensor([float('nan')] * 64)
        
        # Should not crash, should store as-is
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="float16")
        
        assert dim == 64
        assert l2norm_flag is False  # Not normalized due to NaN

    def test_inf_embedding(self):
        """Test handling of Inf embedding."""
        emb = torch.tensor([float('inf')] * 64)
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="float16")
        
        assert dim == 64
        assert l2norm_flag is False

    def test_zero_norm_embedding(self):
        """Test handling of zero norm embedding."""
        emb = torch.zeros(64)
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="float16")
        
        assert dim == 64
        assert l2norm_flag is False  # Not normalized due to zero norm

    def test_very_small_norm(self):
        """Test handling of very small norm (< 1e-12)."""
        emb = torch.ones(64) * 1e-13
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="float16")
        
        assert dim == 64
        assert l2norm_flag is False

    def test_2d_tensor_flattening(self):
        """Test that 2D tensors are flattened."""
        emb = torch.randn(1, 128)  # [1, D]
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="float16")
        
        assert dim == 128  # Should be flattened

    def test_unknown_dtype_fallback(self):
        """Test fallback for unknown dtype."""
        emb = torch.randn(32)
        emb = emb / emb.norm()
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="unknown_dtype")
        
        assert storage_dtype == "float16"  # Fallback (see storage.py DTYPE_MAP)


class TestDeserializeEmbedding:
    """Tests for deserialize_embedding()."""

    def test_basic_deserialization_float16(self):
        """Test basic deserialization with float16."""
        emb = torch.randn(64)
        emb = emb / emb.norm()
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="float16")
        restored = deserialize_embedding(blob, dim, storage_dtype, l2norm_flag)
        
        assert restored.shape == (64,)
        # Convert to float32 for comparison (float16 vs float32)
        assert torch.allclose(restored.float(), emb.float(), atol=1e-3)

    def test_basic_deserialization_float32(self):
        """Test basic deserialization with float32."""
        emb = torch.randn(64)
        emb = emb / emb.norm()
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="float32")
        restored = deserialize_embedding(blob, dim, storage_dtype, l2norm_flag)
        
        assert restored.shape == (64,)
        assert torch.allclose(restored, emb, atol=1e-5)

    def test_deserialize_without_l2norm_flag(self):
        """Test deserialization of non-normalized embedding."""
        emb = torch.randn(64) * 3.0  # Not normalized
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="float16")
        # l2norm_flag should be True after serialization (normalized)
        restored = deserialize_embedding(blob, dim, storage_dtype, l2norm_flag)
        
        # Should be normalized
        norm = restored.norm().item()
        assert abs(norm - 1.0) < 1e-2

    def test_bfloat16_deserialization(self):
        """Test deserialization of bfloat16 (stored as float16)."""
        emb = torch.randn(32, dtype=torch.bfloat16)
        emb = emb / emb.norm()
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb, dtype="bfloat16")
        # Deserialize as bfloat16 (will read as float16)
        restored = deserialize_embedding(blob, dim, "bfloat16", l2norm_flag)
        
        assert restored.shape == (32,)

    def test_nan_handling_deserialization(self):
        """Test handling of NaN during deserialization."""
        # Create a blob with NaN values
        arr = np.array([float('nan')] * 32, dtype=np.float16)
        blob = arr.tobytes()
        
        # Should not crash
        restored = deserialize_embedding(blob, 32, "float16", l2norm=False)
        
        assert restored.shape == (32,)
        # Should skip normalization due to NaN norm

    def test_zero_norm_deserialization(self):
        """Test handling of zero norm during deserialization."""
        arr = np.zeros(32, dtype=np.float16)
        blob = arr.tobytes()
        
        restored = deserialize_embedding(blob, 32, "float16", l2norm=False)
        
        assert restored.shape == (32,)
        # Should skip normalization due to zero norm


class TestNormalizeEmbedding:
    """Tests for normalize_embedding()."""

    def test_basic_normalization(self):
        """Test basic normalization."""
        emb = torch.randn(64) * 5.0
        
        normalized = normalize_embedding(emb)
        
        norm = normalized.norm().item()
        assert abs(norm - 1.0) < 1e-2

    def test_already_normalized(self):
        """Test normalization of already normalized embedding."""
        emb = torch.randn(64)
        emb = emb / emb.norm()
        
        normalized = normalize_embedding(emb)
        
        # Should return same (or very close)
        assert torch.allclose(normalized, emb, atol=1e-5)

    def test_nan_handling(self):
        """Test handling of NaN in normalize_embedding."""
        emb = torch.tensor([float('nan')] * 64)
        
        normalized = normalize_embedding(emb)
        
        # Should return original (not crash)
        # torch.equal doesn't work with NaN, check shape and that all are NaN
        assert normalized.shape == emb.shape
        assert torch.isnan(normalized).all()
        assert torch.isnan(emb).all()

    def test_inf_handling(self):
        """Test handling of Inf in normalize_embedding."""
        emb = torch.tensor([float('inf')] * 64)
        
        normalized = normalize_embedding(emb)
        
        # Should return original
        assert torch.equal(normalized, emb)

    def test_zero_norm_handling(self):
        """Test handling of zero norm in normalize_embedding."""
        emb = torch.zeros(64)
        
        normalized = normalize_embedding(emb)
        
        # Should return original
        assert torch.equal(normalized, emb)

    def test_very_small_norm(self):
        """Test handling of very small norm (< 1e-12)."""
        emb = torch.ones(64) * 1e-13
        
        normalized = normalize_embedding(emb)
        
        # Should return original
        assert torch.equal(normalized, emb)


class TestRoundtrip:
    """Tests for roundtrip serialization/deserialization."""

    def test_roundtrip_float16(self):
        """Test roundtrip with float16."""
        emb_original = torch.randn(128)
        emb_original = emb_original / emb_original.norm()
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb_original, dtype="float16")
        emb_restored = deserialize_embedding(blob, dim, storage_dtype, l2norm_flag)
        
        # Should be close (float16 precision)
        # Convert to float32 for comparison
        assert torch.allclose(emb_restored.float(), emb_original.float(), atol=1e-3)

    def test_roundtrip_float32(self):
        """Test roundtrip with float32."""
        emb_original = torch.randn(128)
        emb_original = emb_original / emb_original.norm()
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb_original, dtype="float32")
        emb_restored = deserialize_embedding(blob, dim, storage_dtype, l2norm_flag)
        
        # Should be very close (float32 precision)
        assert torch.allclose(emb_restored, emb_original, atol=1e-5)

    def test_roundtrip_bfloat16(self):
        """Test roundtrip with bfloat16 (converted to float16)."""
        emb_original = torch.randn(64, dtype=torch.bfloat16)
        emb_original = emb_original / emb_original.norm()
        
        blob, dim, l2norm_flag, storage_dtype = serialize_embedding(emb_original, dtype="bfloat16")
        emb_restored = deserialize_embedding(blob, dim, "bfloat16", l2norm_flag)
        
        # Should be close (float16 precision after conversion)
        assert emb_restored.shape == (64,)
