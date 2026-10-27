import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make `python -m personalized_recommender` subprocesses (and in-process imports) work from a fresh
# checkout without installing the package. An installed copy is unaffected.
_SRC = str(Path(__file__).parents[1] / "src")
os.environ["PYTHONPATH"] = os.pathsep.join(filter(None, [_SRC, os.environ.get("PYTHONPATH", "")]))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

FIXTURES = Path(__file__).parent / "fixtures"
PKG_FIXTURE = Path(__file__).parents[1] / "src" / "personalized_recommender" / "fixtures" / "synthetic_ml"


def ev(*rows):
    """Build an (N,4) int64 event array from (user, movie, rating, ts) tuples."""
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def write_ratings(dir_: Path, rows) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / "ratings.dat").write_text("".join(f"{u}::{m}::{r}::{t}\n" for u, m, r, t in rows))
    return dir_


@pytest.fixture
def ml1m_dir():
    """Official MovieLens-1M directory supplied via P02_ML1M_DIR (real-data tests)."""
    d = os.environ.get("P02_ML1M_DIR")
    if not d:
        pytest.skip("requires official MovieLens-1M (obtain from GroupLens, not shipped): set P02_ML1M_DIR to the extracted ml-1m directory")
    return Path(d)


@pytest.fixture
def ml1m_archive():
    a = os.environ.get("P02_ML1M_ARCHIVE")
    if not a:
        pytest.skip("requires official ml-1m.zip (not shipped): set P02_ML1M_ARCHIVE")
    return Path(a)
