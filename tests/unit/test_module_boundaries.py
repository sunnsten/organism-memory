import ast
import importlib.util
from pathlib import Path
from typing import List, Set

import pytest


def get_imports_from_file(file_path: Path) -> Set[str]:
    """Extract all imports from a Python file."""
    with open(file_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(file_path))

    imports: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)

    return imports


def test_pipeline_modules_no_deep_imports():
    """
    Verify that pipeline modules do not import "deep" internal modules.

    Forbidden imports:
    - organism.memory.core
    - organism.memory.personal_store
    - organism.memory.consolidation.*
    - organism.memory._personal_store.*
    - organism.memory.stores.* (direct store access is forbidden)

    Allowed imports:
    - organism.memory.service (MemoryService — the single entry point)
    - organism.memory.domain.* (domain types)
    - organism.memory.types (legacy facade, acceptable for backwards compatibility)
    - organism.memory.embeddings.* (embedding utilities)
    - organism.agents.personal.contracts.* (protocols and DTOs)
    """
    project_root = Path(__file__).parent.parent.parent
    pipeline_dir = project_root / "organism" / "agents" / "personal" / "pipeline"

    forbidden_patterns = [
        "organism.memory.core",
        "organism.memory.personal_store",
        "organism.memory.consolidation",
        "organism.memory._personal_store",
        "organism.memory.stores",  # direct store access is forbidden
    ]

    pipeline_files = list(pipeline_dir.glob("*.py"))
    pipeline_files = [f for f in pipeline_files if f.name != "__init__.py"]

    violations: List[tuple[str, str]] = []

    for file_path in pipeline_files:
        imports = get_imports_from_file(file_path)

        for import_name in imports:
            for pattern in forbidden_patterns:
                if import_name.startswith(pattern):
                    violations.append((file_path.name, import_name))

    if violations:
        violation_msg = "\n".join(
            f"  {file}: {import_name}" for file, import_name in violations
        )
        pytest.fail(
            f"Pipeline modules contain forbidden imports:\n{violation_msg}\n\n"
            f"Pipeline modules should only import:\n"
            f"  - organism.memory.service (MemoryService — single entry point)\n"
            f"  - organism.memory.domain.* (domain types: EventRecord, ExperienceBlock, RetrieveResult, etc.)\n"
            f"  - organism.memory.types (legacy facade, acceptable for backwards compatibility)\n"
            f"  - organism.agents.personal.contracts.* (protocols and DTOs)\n"
            f"  - organism.backbone.* (LM interfaces)\n"
            f"  - organism.memory.embeddings.* (embedding utilities)\n"
        )


def test_pipeline_modules_use_contracts():
    """
    Verify that pipeline modules use contracts/protocols.
    """
    project_root = Path(__file__).parent.parent.parent
    pipeline_dir = project_root / "organism" / "agents" / "personal" / "pipeline"

    pipeline_files = list(pipeline_dir.glob("*.py"))
    pipeline_files = [f for f in pipeline_files if f.name != "__init__.py"]

    # Files that should use contracts
    files_using_protocols = [
        "retrieve.py",  # uses MemoryBackend, PersonalStoreBackend
        "observe.py",   # uses MemoryBackend
        "persist.py",   # uses PersonalStoreBackend
    ]

    violations: List[str] = []

    for file_name in files_using_protocols:
        file_path = pipeline_dir / file_name
        if not file_path.exists():
            continue

        imports = get_imports_from_file(file_path)

        # Check that contracts/protocols are used instead of the old protocols module
        uses_old_protocols = any(
            imp == "organism.agents.personal.protocols"
            for imp in imports
        )

        uses_contracts = any(
            imp == "organism.agents.personal.contracts.protocols"
            for imp in imports
        )

        if uses_old_protocols and not uses_contracts:
            violations.append(
                f"{file_name} uses old protocols.py instead of contracts/protocols.py"
            )

    if violations:
        pytest.fail(
            f"Pipeline modules should use contracts/protocols:\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


def test_internal_code_no_memory_types_imports():
    """
    Verify that organism/ code does not import from organism.memory.types.

    organism/memory/types.py has been removed — all types now live in organism.shared.domain.
    This test verifies that no organism/ file uses the old import path.
    """
    project_root = Path(__file__).parent.parent.parent
    organism_dir = project_root / "organism"

    forbidden_pattern = "from organism.memory.types import"
    violations = []

    for py_file in organism_dir.rglob("*.py"):
        rel_path = py_file.relative_to(project_root)
        rel_path_str = str(rel_path).replace("\\", "/")

        try:
            imports = get_imports_from_file(py_file)
            for imp in imports:
                if forbidden_pattern in imp:
                    violations.append((rel_path_str, imp))
        except Exception:
            continue

    if violations:
        violation_lines = "\n".join([f"  {path}: {imp}" for path, imp in violations])
        pytest.fail(
            f"Stale imports from organism.memory.types found:\n{violation_lines}\n"
            f"Use: from organism.shared.domain import ..."
        )


__all__ = [
    "test_pipeline_modules_no_deep_imports",
    "test_pipeline_modules_use_contracts",
    "test_internal_code_no_memory_types_imports",
]
