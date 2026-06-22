import pytest
from organism.core.memory.rag.context_assembler import ContextAssembler, ContextAssemblerConfig
from organism.core.memory.rag.hybrid_retriever import HybridResult


def _make_fact(content, event_time=None, rrf_score=1.0):
    return HybridResult(
        id=1, content=content, rrf_score=rrf_score,
        tier="memory_item", sources=["facts"],
        valid_from=event_time,
    )


def test_facts_sorted_chronologically():
    """Facts with event_time are shown oldest-first so LLM sees timeline."""
    assembler = ContextAssembler()
    facts = [
        _make_fact("User moved to Berlin", event_time=1700000000),   # Nov 2023
        _make_fact("User started Python", event_time=1600000000),    # Sep 2020 (older)
        _make_fact("User has a cat"),                                   # no date
    ]
    ctx = assembler.assemble(
        hybrid_results=facts, working_memory=[], user_question="Q",
    )
    lines = ctx.memory_block.split("\n")
    # Sep 2020 fact should appear before Nov 2023 fact
    idx_python = next(i for i, l in enumerate(lines) if "Python" in l)
    idx_berlin = next(i for i, l in enumerate(lines) if "Berlin" in l)
    assert idx_python < idx_berlin, "Older fact should appear first"


def test_facts_without_date_appended_after_dated():
    """Facts without event_time appear after chronological facts."""
    assembler = ContextAssembler()
    facts = [
        _make_fact("User has a cat"),             # no date, comes first in input
        _make_fact("User moved to Berlin", event_time=1700000000),
    ]
    ctx = assembler.assemble(
        hybrid_results=facts, working_memory=[], user_question="Q",
    )
    lines = ctx.memory_block.split("\n")
    idx_cat = next(i for i, l in enumerate(lines) if "cat" in l)
    idx_berlin = next(i for i, l in enumerate(lines) if "Berlin" in l)
    assert idx_berlin < idx_cat, "Dated fact before undated fact"
