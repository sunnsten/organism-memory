import subprocess
import sys


def test_organism_has_version():
    import organism
    assert hasattr(organism, "__version__")
    ver = organism.__version__
    parts = ver.split(".")
    assert len(parts) == 3, f"Expected X.Y.Z, got {ver!r}"
    assert all(p.isdigit() for p in parts), f"Non-numeric version part in {ver!r}"


def test_organism_importable_without_torch():
    result = subprocess.run(
        [sys.executable, "-c", "from organism.config import OrganismConfig; print('ok')"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout