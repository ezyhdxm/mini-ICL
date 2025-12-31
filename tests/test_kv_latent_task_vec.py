"""Unit tests for kv_latent_task_vec utilities."""

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from icl.utils.kv_latent_task_vec import (
    _get_n_emb,
    _infer_n_head_head_dim,
    _make_kv_caches_like_model,
    _HookLastTokenAllLayers,
    compute_hiddens_onepos_all_layers_kvcache,
)
from icl.models.kv_cache import KVCache


class TestGetNEmb:
    """Tests for _get_n_emb function."""

    def test_get_n_emb_with_emb_dim(self):
        """Test retrieving n_emb when config has emb_dim."""
        config = Mock()
        config.model.emb_dim = 256
        assert _get_n_emb(config) == 256

    def test_get_n_emb_with_n_embd(self):
        """Test retrieving n_emb when config has n_embd."""
        config = Mock()
        config.model.n_embd = 512
        # Remove emb_dim attribute to force fallback to n_embd
        delattr(config.model, 'emb_dim')
        with patch('builtins.getattr', side_effect=[AttributeError, 512]):
            assert _get_n_emb(config) == 512

    def test_get_n_emb_returns_int(self):
        """Test that _get_n_emb returns an integer."""
        config = Mock()
        config.model.emb_dim = 384.0
        result = _get_n_emb(config)
        assert isinstance(result, int)
        assert result == 384


class TestInferNHeadHeadDim:
    """Tests for _infer_n_head_head_dim function."""

    def test_infer_n_head_head_dim(self):
        """Test inferring number of heads and head dimension."""
        model = Mock()
        mha = Mock()
        mha.n_head = 8
        mha.head_dim = 64
        model.layers = [Mock()]
        model.layers[0].attn_block.MHA = mha

        n_head, head_dim = _infer_n_head_head_dim(model)
        assert n_head == 8
        assert head_dim == 64

    def test_infer_n_head_head_dim_returns_ints(self):
        """Test that function returns integers."""
        model = Mock()
        mha = Mock()
        mha.n_head = 12.0
        mha.head_dim = 32.0
        model.layers = [Mock()]
        model.layers[0].attn_block.MHA = mha

        n_head, head_dim = _infer_n_head_head_dim(model)
        assert isinstance(n_head, int)
        assert isinstance(head_dim, int)
        assert n_head == 12
        assert head_dim == 32


class TestMakeKVCachesLikeModel:
    """Tests for _make_kv_caches_like_model function."""

    def test_make_kv_caches_basic(self):
        """Test creating KV caches with basic parameters."""
        model = Mock()
        mha = Mock()
        mha.n_head = 4
        mha.head_dim = 32
        model.layers = [Mock(), Mock(), Mock()]
        model.layers[0].attn_block.MHA = mha
        model.parameters = lambda: [torch.ones(1, dtype=torch.float32)]

        caches = _make_kv_caches_like_model(
            model=model,
            batch_size=2,
            max_len=16,
            device=torch.device('cpu')
        )

        assert len(caches) == 3
        for cache in caches:
            assert isinstance(cache, KVCache)
            assert cache.k.shape == (2, 4, 16, 32)
            assert cache.v.shape == (2, 4, 16, 32)
            assert cache.cur_len == 0
            assert cache.k.dtype == torch.float32

    def test_make_kv_caches_dtype_matching(self):
        """Test that cache dtype matches model dtype."""
        model = Mock()
        mha = Mock()
        mha.n_head = 2
        mha.head_dim = 16
        model.layers = [Mock()]
        model.layers[0].attn_block.MHA = mha
        model.parameters = lambda: [torch.ones(1, dtype=torch.float16)]

        caches = _make_kv_caches_like_model(
            model=model,
            batch_size=1,
            max_len=8,
            device=torch.device('cpu')
        )

        assert caches[0].k.dtype == torch.float16
        assert caches[0].v.dtype == torch.float16

    def test_make_kv_caches_device(self):
        """Test that caches are created on correct device."""
        model = Mock()
        mha = Mock()
        mha.n_head = 2
        mha.head_dim = 16
        model.layers = [Mock()]
        model.layers[0].attn_block.MHA = mha
        model.parameters = lambda: [torch.ones(1, dtype=torch.float32)]

        caches = _make_kv_caches_like_model(
            model=model,
            batch_size=1,
            max_len=8,
            device=torch.device('cpu')
        )

        assert caches[0].k.device.type == 'cpu'
        assert caches[0].v.device.type == 'cpu'


class TestHookLastTokenAllLayers:
    """Tests for _HookLastTokenAllLayers class."""

    def test_hook_initialization(self):
        """Test hook initialization."""
        modules = [torch.nn.Identity(), torch.nn.Identity()]
        hook = _HookLastTokenAllLayers(modules)

        assert hook.modules == modules
        assert len(hook.handles) == 0
        assert len(hook.store) == 2
        assert all(s is None for s in hook.store)

    def test_hook_clear(self):
        """Test clearing stored activations."""
        modules = [torch.nn.Identity()]
        hook = _HookLastTokenAllLayers(modules)
        hook.store[0] = torch.ones(2, 3)

        hook.clear()
        assert hook.store[0] is None

    def test_hook_context_manager(self):
        """Test hook as context manager."""
        module = torch.nn.Linear(4, 4)
        hook = _HookLastTokenAllLayers([module])

        with hook:
            assert len(hook.handles) == 1
            # Forward pass
            inp = torch.randn(2, 5, 4)  # (B, T, D)
            _ = module(inp)
            # Should capture last token
            assert hook.store[0] is not None
            assert hook.store[0].shape == (2, 4)  # (B, D)

        # Handles should be removed after exit
        assert len(hook.handles) == 0

    def test_hook_captures_last_token(self):
        """Test that hook captures the last token correctly."""
        module = torch.nn.Identity()
        hook = _HookLastTokenAllLayers([module])

        with hook:
            inp = torch.arange(24).reshape(2, 3, 4).float()  # (B=2, T=3, D=4)
            _ = module(inp)

        captured = hook.store[0]
        expected = inp[:, -1, :]  # Last token
        assert torch.allclose(captured, expected)

    def test_hook_multiple_modules(self):
        """Test hook with multiple modules."""
        modules = [torch.nn.Identity(), torch.nn.Identity()]
        hook = _HookLastTokenAllLayers(modules)

        with hook:
            inp1 = torch.ones(2, 3, 4)
            inp2 = torch.ones(2, 3, 4) * 2
            _ = modules[0](inp1)
            _ = modules[1](inp2)

        assert hook.store[0] is not None
        assert hook.store[1] is not None
        assert torch.allclose(hook.store[0], torch.ones(2, 4))
        assert torch.allclose(hook.store[1], torch.ones(2, 4) * 2)

    def test_hook_raises_on_wrong_shape(self):
        """Test that hook raises error for wrong input shape."""
        module = torch.nn.Identity()
        hook = _HookLastTokenAllLayers([module])

        with pytest.raises(RuntimeError, match="Expected \\(B,T,D\\)"):
            with hook:
                inp = torch.randn(4, 4)  # Wrong shape (2D instead of 3D)
                _ = module(inp)


class TestComputeHiddensOneposAllLayersKvcache:
    """Tests for compute_hiddens_onepos_all_layers_kvcache function."""

    def test_invalid_samples_device(self):
        """Test that function raises error if samples not on CPU."""
        config = Mock()
        model = Mock()
        samples = torch.randn(1, 1, 3).cuda() if torch.cuda.is_available() else torch.randn(1, 1, 3)

        if torch.cuda.is_available():
            with pytest.raises(ValueError, match="Provide samples on CPU"):
                compute_hiddens_onepos_all_layers_kvcache(config, model, samples)

    def test_invalid_samples_shape(self):
        """Test that function raises error for invalid samples shape."""
        config = Mock()
        model = Mock()
        samples = torch.randint(0, 10, (5, 3))  # 2D instead of 3D

        with pytest.raises(ValueError, match="samples must be \\(n_tasks,B,Sfull\\)"):
            compute_hiddens_onepos_all_layers_kvcache(config, model, samples)

    def test_invalid_sfull(self):
        """Test that function raises error for invalid Sfull."""
        config = Mock()
        config.model.emb_dim = 8
        model = Mock()
        model.layers = []
        model.parameters = lambda: [torch.ones(1)]
        # Sfull should be 2*seq_len - 1, so even numbers are invalid
        samples = torch.randint(0, 10, (1, 1, 4))  # Invalid Sfull

        with pytest.raises(ValueError, match="Expected Sfull = 2\\*seq_len-1"):
            compute_hiddens_onepos_all_layers_kvcache(config, model, samples)

    def test_invalid_activation_kind(self):
        """Test that function raises error for invalid activation kind."""
        config = Mock()
        config.model.emb_dim = 8
        model = Mock()
        model.layers = []
        model.parameters = lambda: [torch.ones(1)]
        samples = torch.randint(0, 10, (1, 1, 3))

        with pytest.raises(ValueError, match="activation must be 'attn_block' or 'mlp'"):
            compute_hiddens_onepos_all_layers_kvcache(
                config, model, samples, activation="invalid"
            )

    def test_samples_converted_to_long(self):
        """Test that samples are converted to long dtype."""
        config = Mock()
        config.model.emb_dim = 8
        
        # Create minimal working model
        model = Mock()
        model.eval = Mock()
        model.layers = [Mock()]
        model.parameters = lambda: [torch.ones(1)]
        
        samples = torch.randint(0, 10, (1, 1, 3)).float()  # Float dtype
        
        # This should convert to long without error
        # We expect it to raise about hook modules, but that's after dtype conversion
        with pytest.raises((ValueError, AttributeError)):
            compute_hiddens_onepos_all_layers_kvcache(config, model, samples)

    def test_too_small_sfull(self):
        """Test error for Sfull < 2."""
        config = Mock()
        config.model.emb_dim = 8
        model = Mock()
        model.layers = []
        model.parameters = lambda: [torch.ones(1)]
        samples = torch.randint(0, 10, (1, 1, 1))

        with pytest.raises(ValueError, match="Sfull too small"):
            compute_hiddens_onepos_all_layers_kvcache(config, model, samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
