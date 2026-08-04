from pathlib import Path


def project_root() -> Path:
    """
    Returns the repository root by walking upward until it finds a marker file/folder.
    Works no matter where the script is run from.
    """
    p = Path(__file__).resolve()

    # Walk up parents looking for a marker that only exists at repo root
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists():
            return parent
        if (parent / ".git").exists():
            return parent
        if (parent / "README.md").exists() and (parent / "environment").exists():
            return parent

    # Fallback: 2 levels up from utils/paths.py (repo/utils/paths.py)
    return Path(__file__).resolve().parents[1]

def data_dir() -> Path:
    return project_root() / "data"

def raw_data_dir() -> Path:
    return data_dir() / "raw"
