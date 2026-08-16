"""Environment verification and diagnostic script for Amazon Semantic Search."""

import importlib
import os
import platform
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    ("fastapi", "FastAPI"),
    ("pydantic", "Pydantic"),
    ("uvicorn", "Uvicorn"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("sklearn", "scikit-learn"),
    ("yaml", "PyYAML"),
]

OPTIONAL_ML_PACKAGES = [
    ("torch", "PyTorch"),
    ("sentence_transformers", "Sentence Transformers"),
    ("transformers", "Transformers"),
    ("faiss", "FAISS"),
]


def check_python_version() -> bool:
    print(f"[*] Python version: {platform.python_version()} on {platform.platform()}")
    if sys.version_info < (3, 10):
        print("[!] Warning: Python 3.10+ is recommended.")
        return False
    print("[+] Python version is compatible.")
    return True


def check_packages(packages: list, category: str) -> None:
    print(f"\n[*] Checking {category} dependencies:")
    for module_name, display_name in packages:
        try:
            mod = importlib.import_module(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [+] {display_name} ({module_name}): {ver}")
        except ImportError:
            print(f"  [-] {display_name} ({module_name}): NOT INSTALLED")


def check_directories() -> None:
    print("\n[*] Checking repository directory structure:")
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "indexes",
        "experiments/baseline",
        "experiments/retrieval",
        "experiments/reranking",
        "experiments/recommendation",
        "experiments/results",
        "backend/app",
        "evaluation",
        "frontend",
    ]
    for d in directories:
        p = Path(d)
        if p.exists():
            print(f"  [+] {d}/ exists")
        else:
            print(f"  [-] {d}/ missing -> creating directory")
            p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("=" * 70)
    print(" Amazon-Scale Semantic Product Search - Environment Diagnostic")
    print("=" * 70)
    check_python_version()
    check_packages(REQUIRED_PACKAGES, "Core & API")
    check_packages(OPTIONAL_ML_PACKAGES, "Machine Learning & Vector Search")
    check_directories()
    print("\n" + "=" * 70)
    print(" Diagnostic Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
