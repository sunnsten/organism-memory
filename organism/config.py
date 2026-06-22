from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from organism.backbone.config import BackboneConfig
from organism.core.config import (
    SlotsConfig,
    RAGConfig,
    MemoryGatingConfig,
    ImportanceWeights,
)


_VALID_MEMORY_MODES = ("t1", "t2")


@dataclass
class OrganismConfig:
    """
    Top-level config for the Organism project.

    Sub-config locations:
      - base_model  → organism.backbone.config.BackboneConfig
      - fact_llm    → organism.backbone.config.BackboneConfig (optional dedicated model for FactExtractor)
      - slots       → organism.core.config.SlotsConfig
      - rag         → organism.core.config.RAGConfig

    memory_mode controls which memory tiers are active:
      "t1" — RAG chunks only (WriteService chunks, no FactExtractor)
      "t2" — RAG chunks + Facts (default; FactExtractor async daemon)
    """
    base_model: BackboneConfig = field(default_factory=BackboneConfig)
    fact_llm: Optional[BackboneConfig] = None
    slots: SlotsConfig = field(default_factory=SlotsConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    memory_mode: str = "t2"

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "OrganismConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data or {})

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(
                self.to_dict(),
                f,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False,
            )

    def to_dict(self) -> Dict[str, Any]:
        def _dc(obj: Any) -> Dict[str, Any]:
            result: Dict[str, Any] = {}
            for k, v in obj.__dict__.items():
                if hasattr(v, "__dataclass_fields__"):
                    result[k] = _dc(v)
                elif isinstance(v, list):
                    result[k] = v
                elif isinstance(v, dict):
                    result[k] = {
                        dk: _dc(dv) if hasattr(dv, "__dataclass_fields__") else dv
                        for dk, dv in v.items()
                    }
                else:
                    result[k] = v
            return result

        d: Dict[str, Any] = {
            "memory_mode": self.memory_mode,
            "base_model": _dc(self.base_model),
            "slots": _dc(self.slots),
            "rag": _dc(self.rag),
        }
        if self.fact_llm is not None:
            d["fact_llm"] = _dc(self.fact_llm)
        return d

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "OrganismConfig":
        data = dict(data)
        data.pop("personal", None)
        data.pop("consolidation", None)  # silently drop legacy research key
        data.pop("sleep", None)          # silently drop legacy research key

        base_model = BackboneConfig(**data.get("base_model", {}))

        if "memory" in data:
            return _from_legacy_memory(base_model, data["memory"])

        memory_mode = data.get("memory_mode", "t2")
        if memory_mode not in _VALID_MEMORY_MODES:
            memory_mode = "t2"  # silently clamp unknown modes (e.g. legacy "t3")

        fact_llm: Optional[BackboneConfig] = None
        if "fact_llm" in data and data["fact_llm"]:
            fact_llm = BackboneConfig(**data["fact_llm"])

        return cls(
            memory_mode=memory_mode,
            base_model=base_model,
            fact_llm=fact_llm,
            slots=_parse_slots(data.get("slots", {})),
            rag=RAGConfig(**_filter(data.get("rag", {}), RAGConfig)),
        )


def _filter(data: dict, cls: type) -> dict:
    """Keep only keys that exist in the dataclass."""
    known = set(cls.__dataclass_fields__.keys())
    return {k: v for k, v in data.items() if k in known}


def _parse_slots(data: dict) -> SlotsConfig:
    data = _filter(data, SlotsConfig)
    if "importance_weights" in data and isinstance(data["importance_weights"], dict):
        data["importance_weights"] = ImportanceWeights(**data["importance_weights"])
    if "gating" in data and isinstance(data["gating"], dict):
        data["gating"] = MemoryGatingConfig(**data["gating"])
    return SlotsConfig(**data)


def _from_legacy_memory(base_model: BackboneConfig, memory: dict) -> "OrganismConfig":
    """Map flat legacy 'memory:' YAML section to typed sub-configs."""
    slots_data = _filter(memory, SlotsConfig)
    if "importance_weights" in slots_data and isinstance(slots_data["importance_weights"], dict):
        slots_data["importance_weights"] = ImportanceWeights(**slots_data["importance_weights"])
    if "gating" in slots_data and isinstance(slots_data["gating"], dict):
        slots_data["gating"] = MemoryGatingConfig(**slots_data["gating"])
    rag_data = _filter(memory, RAGConfig)
    return OrganismConfig(
        base_model=base_model,
        slots=SlotsConfig(**slots_data) if slots_data else SlotsConfig(),
        rag=RAGConfig(**rag_data) if rag_data else RAGConfig(),
    )


__all__ = [
    "OrganismConfig",
    "BackboneConfig",
    "SlotsConfig",
    "RAGConfig",
    "MemoryGatingConfig",
    "ImportanceWeights",
]
