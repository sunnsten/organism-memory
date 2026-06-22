import pytest


class TestNewConfigsInstantiation:
    """All domain configs instantiate with sensible defaults."""

    def test_slots_config_defaults(self):
        from organism.core.config.slots_config import SlotsConfig, MemoryGatingConfig, ImportanceWeights
        cfg = SlotsConfig()
        assert cfg.d_state == 512
        assert cfg.d_compressed == 256
        assert cfg.memory_size == 1000
        assert cfg.gate_hidden_size == 64
        assert cfg.memory_threshold == 0.5
        assert isinstance(cfg.gating, MemoryGatingConfig)
        assert isinstance(cfg.importance_weights, ImportanceWeights)
        assert cfg.merge_sim_threshold == 0.85
        assert cfg.enable_value_merge is True
        assert cfg.max_slot_text_len == 200
        assert cfg.retrieve_top_k == 4
        assert cfg.enable_retrieve_slots is True

    def test_rag_config_defaults(self):
        from organism.core.config.rag_config import RAGConfig
        cfg = RAGConfig()
        assert cfg.context_window_enabled is True
        assert cfg.context_window_max_history_tokens == 900
        assert cfg.context_window_overflow_trigger_tokens == 1200
        assert cfg.memory_extraction_enabled is True
        assert cfg.enable_retrieve_db is True

    def test_consolidation_config_defaults(self):
        mod = pytest.importorskip("organism.research.config.consolidation_config")
        cfg = mod.ConsolidationConfig()
        assert cfg.consolidation_sim_threshold == 0.8
        assert cfg.consolidation_min_importance == 0.15
        assert cfg.clustering_embedding_weight == 0.7
        assert len(cfg.consolidation_allow_kinds) == 6
        assert "personal" in cfg.namespace_routing

    def test_sleep_config_defaults(self):
        mod = pytest.importorskip("organism.research.config")
        cfg = mod.SleepConfig()
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 32
        assert cfg.min_experience == 5
        assert cfg.max_steps == 20
        assert cfg.learning_rate == 1e-4

    def test_importance_weights_defaults(self):
        from organism.core.config.slots_config import ImportanceWeights
        w = ImportanceWeights()
        assert w.attn == 0.4
        assert w.surprisal == 0.3
        assert w.length == 0.2
        assert w.mem == 0.1
        assert abs(w.attn + w.surprisal + w.length + w.mem - 1.0) < 1e-6


class TestOrganismConfigIntegration:
    """OrganismConfig holds typed sub-configs as direct fields."""

    def test_organism_config_has_slots(self):
        from organism.config import OrganismConfig
        cfg = OrganismConfig()
        assert cfg.slots.d_state == 512

    def test_organism_config_has_rag(self):
        from organism.config import OrganismConfig
        cfg = OrganismConfig()
        assert cfg.rag.context_window_enabled is True

    def test_organism_config_sub_configs_are_typed(self):
        from organism.config import OrganismConfig
        from organism.core.config import SlotsConfig, RAGConfig
        cfg = OrganismConfig()
        assert isinstance(cfg.slots, SlotsConfig)
        assert isinstance(cfg.rag, RAGConfig)

    def test_organism_config_custom_slots(self):
        from organism.config import OrganismConfig
        from organism.core.config import SlotsConfig
        cfg = OrganismConfig(slots=SlotsConfig(d_state=1024, memory_size=500))
        assert cfg.slots.d_state == 1024
        assert cfg.slots.memory_size == 500


class TestConvenienceReExports:
    """Shared types are re-exported from organism.config."""

    def test_memory_gating_config_import(self):
        from organism.config import MemoryGatingConfig
        cfg = MemoryGatingConfig()
        assert cfg.decay_lambda == 0.99

    def test_importance_weights_import(self):
        from organism.config import ImportanceWeights
        w = ImportanceWeights()
        assert w.attn == 0.4

    def test_gating_config_is_same_class(self):
        """MemoryGatingConfig from organism.config and canonical location is the same class."""
        from organism.config import MemoryGatingConfig as FromTop
        from organism.core.config.slots_config import MemoryGatingConfig as FromCanonical
        assert FromTop is FromCanonical

    def test_sleep_config_from_research_config(self):
        mod = pytest.importorskip("organism.research.config")
        cfg = mod.SleepConfig()
        assert cfg.lora_r == 8
        assert cfg.min_experience == 5

    def test_slots_config_from_core_config(self):
        from organism.core.config import SlotsConfig
        cfg = SlotsConfig()
        assert cfg.d_state == 512


class TestYAMLRoundTrip:
    """OrganismConfig serializes and deserializes correctly."""

    def test_to_dict_has_expected_keys(self):
        from organism.config import OrganismConfig
        d = OrganismConfig().to_dict()
        assert "memory_mode" in d
        assert "base_model" in d
        assert "slots" in d
        assert "rag" in d
        assert "consolidation" not in d
        assert "sleep" not in d

    def test_from_yaml_new_format(self, tmp_path):
        import yaml
        from organism.config import OrganismConfig
        yaml_content = {
            "base_model": {"type": "llama31", "model_name": "meta-llama/Llama-3.1-8B-Instruct"},
            "slots": {"d_state": 1024, "importance_weights": {"attn": 0.5, "surprisal": 0.3, "length": 0.1, "mem": 0.1}},
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(yaml_content))
        cfg = OrganismConfig.from_yaml(path)
        assert cfg.base_model.type == "llama31"
        assert cfg.slots.d_state == 1024
        assert cfg.slots.importance_weights.attn == 0.5

    def test_from_yaml_legacy_memory_format(self, tmp_path):
        """Legacy 'memory:' flat section still loads correctly (research keys silently dropped)."""
        import yaml
        from organism.config import OrganismConfig
        yaml_content = {
            "base_model": {"type": "qwen3", "model_name": "Qwen/Qwen3-8B"},
            "memory": {
                "memory_size": 500,
                "d_state": 256,
                "consolidation_sim_threshold": 0.9,
                "sleep_min_experience": 10,
                "importance_weights": {"attn": 0.6, "surprisal": 0.3, "length": 0.05, "mem": 0.05},
            },
        }
        path = tmp_path / "legacy.yaml"
        path.write_text(yaml.dump(yaml_content))
        cfg = OrganismConfig.from_yaml(path)
        assert cfg.base_model.type == "qwen3"
        assert cfg.slots.memory_size == 500
        assert cfg.slots.d_state == 256
        assert cfg.slots.importance_weights.attn == 0.6

    def test_from_yaml_drops_legacy_consolidation_and_sleep(self, tmp_path):
        """consolidation/sleep top-level keys are silently dropped on load."""
        import yaml
        from organism.config import OrganismConfig
        yaml_content = {
            "base_model": {"type": "qwen3"},
            "consolidation": {"consolidation_sim_threshold": 0.9},
            "sleep": {"lora_r": 16},
        }
        path = tmp_path / "legacy_research.yaml"
        path.write_text(yaml.dump(yaml_content))
        cfg = OrganismConfig.from_yaml(path)
        assert cfg.base_model.type == "qwen3"
        assert not hasattr(cfg, "consolidation")
        assert not hasattr(cfg, "sleep")


class TestMemoryMode:
    """OrganismConfig.memory_mode parsing and validation."""

    def test_default_is_t2(self):
        from organism.config import OrganismConfig
        assert OrganismConfig().memory_mode == "t2"

    def test_valid_modes_parse(self, tmp_path):
        import yaml
        from organism.config import OrganismConfig
        for mode in ("t1", "t2"):
            path = tmp_path / f"cfg_{mode}.yaml"
            path.write_text(yaml.dump({"memory_mode": mode}))
            cfg = OrganismConfig.from_yaml(path)
            assert cfg.memory_mode == mode

    def test_t3_clamped_to_t2(self, tmp_path):
        """Legacy t3 mode is silently clamped to t2 (research tier removed)."""
        import yaml
        from organism.config import OrganismConfig
        path = tmp_path / "cfg_t3.yaml"
        path.write_text(yaml.dump({"memory_mode": "t3"}))
        cfg = OrganismConfig.from_yaml(path)
        assert cfg.memory_mode == "t2"

    def test_invalid_mode_clamped(self, tmp_path):
        """Invalid memory_mode is silently clamped to t2."""
        import yaml
        from organism.config import OrganismConfig
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump({"memory_mode": "t9"}))
        cfg = OrganismConfig.from_yaml(path)
        assert cfg.memory_mode == "t2"

    def test_memory_mode_roundtrips_through_dict(self):
        from organism.config import OrganismConfig
        cfg = OrganismConfig(memory_mode="t1")
        assert cfg.to_dict()["memory_mode"] == "t1"


class TestFactLLMConfig:
    """fact_llm optional dedicated backend for FactExtractor."""

    def test_fact_llm_defaults_to_none(self):
        from organism.config import OrganismConfig
        assert OrganismConfig().fact_llm is None

    def test_fact_llm_parsed_from_yaml(self, tmp_path):
        import yaml
        from organism.config import OrganismConfig
        from organism.backbone.config import BackboneConfig
        yaml_content = {
            "base_model": {"type": "llama_cpp", "model_path": "/main/model.gguf"},
            "fact_llm": {
                "type": "llama_cpp",
                "model_path": "/fact/model.gguf",
                "n_gpu_layers": 0,
                "n_ctx": 8192,
                "max_new_tokens": 300,
                "temperature": 0.1,
            },
        }
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump(yaml_content))
        cfg = OrganismConfig.from_yaml(path)
        assert cfg.fact_llm is not None
        assert isinstance(cfg.fact_llm, BackboneConfig)
        assert cfg.fact_llm.type == "llama_cpp"
        assert cfg.fact_llm.model_path == "/fact/model.gguf"
        assert cfg.fact_llm.n_gpu_layers == 0
        assert cfg.fact_llm.temperature == 0.1

    def test_fact_llm_absent_when_not_in_yaml(self, tmp_path):
        import yaml
        from organism.config import OrganismConfig
        path = tmp_path / "cfg.yaml"
        path.write_text(yaml.dump({"base_model": {"type": "llama_cpp", "model_path": "/x.gguf"}}))
        cfg = OrganismConfig.from_yaml(path)
        assert cfg.fact_llm is None

    def test_fact_llm_included_in_to_dict_when_set(self):
        from organism.config import OrganismConfig
        from organism.backbone.config import BackboneConfig
        cfg = OrganismConfig(
            fact_llm=BackboneConfig(type="llama_cpp", model_path="/fact.gguf", n_gpu_layers=0)
        )
        d = cfg.to_dict()
        assert "fact_llm" in d
        assert d["fact_llm"]["type"] == "llama_cpp"
        assert d["fact_llm"]["model_path"] == "/fact.gguf"

    def test_fact_llm_absent_from_to_dict_when_none(self):
        from organism.config import OrganismConfig
        d = OrganismConfig().to_dict()
        assert "fact_llm" not in d

    def test_fact_llm_roundtrip_through_yaml(self, tmp_path):
        import yaml
        from organism.config import OrganismConfig
        from organism.backbone.config import BackboneConfig
        cfg = OrganismConfig(
            fact_llm=BackboneConfig(type="llama_cpp", model_path="/fact.gguf", n_gpu_layers=0, n_ctx=4096)
        )
        path = tmp_path / "rt.yaml"
        cfg.to_yaml(path)
        cfg2 = OrganismConfig.from_yaml(path)
        assert cfg2.fact_llm is not None
        assert cfg2.fact_llm.model_path == "/fact.gguf"
        assert cfg2.fact_llm.n_gpu_layers == 0
        assert cfg2.fact_llm.n_ctx == 4096


class TestFactLLMBackboneFactory:
    """create_lm_backend_from_backbone accepts BackboneConfig directly."""

    def test_create_from_backbone_openai(self):
        from organism.backbone import create_lm_backend_from_backbone
        from organism.backbone.config import BackboneConfig
        from organism.backbone.openai_backend import OpenAIBackend
        cfg = BackboneConfig(type="openai", model_name="gpt-4o", base_url="http://localhost/v1")
        backend = create_lm_backend_from_backbone(cfg)
        assert isinstance(backend, OpenAIBackend)

    def test_create_from_backbone_llama_cpp(self):
        from unittest.mock import patch
        from organism.backbone import create_lm_backend_from_backbone
        from organism.backbone.config import BackboneConfig
        from organism.backbone.llama_cpp_backend import LlamaCppBackend
        cfg = BackboneConfig(type="llama_cpp", model_path="/fake.gguf", n_gpu_layers=0)
        with patch.object(LlamaCppBackend, "_start_worker"):
            backend = create_lm_backend_from_backbone(cfg)
        assert isinstance(backend, LlamaCppBackend)

    def test_unknown_type_raises(self):
        from organism.backbone import create_lm_backend_from_backbone
        from organism.backbone.config import BackboneConfig
        with pytest.raises(ValueError, match="Unknown base_model.type"):
            create_lm_backend_from_backbone(BackboneConfig(type="unknown_xyz"))


class TestFactExtractorDualBackend:
    """FactExtractor uses fact_lm when provided, falls back to main lm."""

    def _make_extractor(self, fact_lm=None):
        from unittest.mock import MagicMock
        from organism.core.stores import UnifiedStore
        from organism.core.memory.service.fact_extractor import FactExtractor
        main_lm = MagicMock()
        embedder = MagicMock()
        embedder.embed_batch.return_value = []
        store = MagicMock()
        store.add_or_supersede.return_value = ("id1", True)
        lm_for_facts = fact_lm if fact_lm is not None else main_lm
        extractor = FactExtractor(lm_backend=lm_for_facts, embedder=embedder, fact_store=store)
        return extractor, main_lm, fact_lm

    def test_uses_provided_lm(self):
        from unittest.mock import MagicMock
        dedicated_lm = MagicMock()
        dedicated_lm.generate.return_value = '[{"content": "User likes Python", "category": "preference"}]'
        extractor, main_lm, _ = self._make_extractor(fact_lm=dedicated_lm)
        extractor.extract_and_store("s1", "u1", "t1", [{"role": "user", "content": "I like Python"}])
        dedicated_lm.generate.assert_called_once()
        main_lm.generate.assert_not_called()

    def test_fallback_to_main_lm_when_no_dedicated(self):
        from unittest.mock import MagicMock
        extractor, main_lm, _ = self._make_extractor(fact_lm=None)
        main_lm.generate.return_value = '[{"content": "User likes Go", "category": "preference"}]'
        extractor.extract_and_store("s1", "u1", "t1", [{"role": "user", "content": "I like Go"}])
        main_lm.generate.assert_called_once()
