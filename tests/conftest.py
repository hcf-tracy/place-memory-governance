from __future__ import annotations
import pytest
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
DATA=ROOT/'data/longitudinal'
from hf13.artifact_registry import build_shared_registry
from hf13.factorial import run_factorial
from hf13.evaluator import evaluate_rows

@pytest.fixture(scope='session')
def reference_core():
    reg=build_shared_registry(DATA,mode='reference')
    rows=run_factorial(data_root=DATA,backend='reference',answer_mode='reference',artifact_registry=reg,run_id='pytest-prof-final')
    evals={c:evaluate_rows(rows[c],DATA,artifact_registry=reg) for c in rows}
    return {'registry':reg,'rows':rows,'evals':evals}
