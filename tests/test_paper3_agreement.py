"""The library must agree with an independent re-implementation of the original code.

``scripts/verify_paper3_pipeline.py`` holds a literal copy of the fold and
representation code of the original study. Here it is run on the synthetic data.
"""

import importlib.util
import os

import pytest

from cavity_encoding_ml.datasets import load_tip_example, tip_example_config

_PATH = os.path.join(os.path.dirname(__file__), "..", "scripts", "verify_paper3_pipeline.py")


def _load():
    spec = importlib.util.spec_from_file_location("verify_paper3_pipeline", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("rep", ["R0", "R1", "R2", "R3", "R4"])
def test_pooled_rmse_identical(rep):
    mod = _load()
    out = mod.compare(load_tip_example(), tip_example_config(), reps=[rep])
    assert out[rep]["difference"] == 0.0  # random forest: exactly equal


@pytest.mark.parametrize("model", ["GB", "SVR", "MLP"])
def test_other_learners_identical(model):
    mod = _load()
    out = mod.compare(load_tip_example(), tip_example_config(), reps=["R0", "R3", "R4"], model=model)
    assert all(v["difference"] < 1e-12 for v in out.values())
