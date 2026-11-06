import json

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from cavity_encoding_ml import (CavityConfig, CavityWorkflow, LeakageWarning, ShotGroupKFold)
from cavity_encoding_ml.datasets import load_tip_example, tip_example_config


def test_toy_workflow_keeps_label(toy, toy_cfg, ridge):
    r = CavityWorkflow(toy_cfg, ridge).run(toy)
    assert r.verdict == "use_cavity" and r.diagnostic.passed
    assert "STEP 1" in r.report() and "STEP 4" in r.report()


def test_no_effect_skips_validation(noeffect):
    from sklearn.linear_model import Ridge

    cfg = CavityConfig("shot", "cav", "y", shared=["x1"])
    r = CavityWorkflow(cfg, Ridge()).run(noeffect)
    assert r.verdict == "no_cavity_effect" and r.validation is None
    assert "Skipped" in r.report()


def test_always_validate_option(noeffect):
    from sklearn.linear_model import Ridge

    cfg = CavityConfig("shot", "cav", "y", shared=["x1"])
    r = CavityWorkflow(cfg, Ridge(), always_validate=True).run(noeffect)
    assert r.validation is not None


def test_ordinary_kfold_triggers_alert(toy, toy_cfg, ridge):
    with pytest.warns(LeakageWarning):
        r = CavityWorkflow(toy_cfg, ridge, cv=KFold(4, shuffle=True, random_state=1)).run(toy)
    assert "ALERT" in r.report()


def test_audit_text(toy, toy_cfg):
    rf = RandomForestRegressor(n_estimators=20, random_state=0, n_jobs=1)
    r = CavityWorkflow(toy_cfg, rf, audit_kfold=True).run(toy)
    assert "Leakage audit" in r.report()


def test_auto_encoding(toy, toy_cfg, ridge):
    assert CavityWorkflow(toy_cfg, ridge, encoding="auto").run(toy).encoding == "onehot"


def test_from_config(tmp_path, toy, toy_cfg, ridge):
    p = tmp_path / "cfg.json"
    toy_cfg.to_json(str(p))
    assert CavityWorkflow.from_config(str(p), ridge).run(toy).verdict == "use_cavity"


def test_to_dict_is_json_serialisable(toy, toy_cfg, ridge):
    json.dumps(CavityWorkflow(toy_cfg, ridge).run(toy).to_dict())


def test_margin_changes_verdict(toy, toy_cfg, ridge):
    r = CavityWorkflow(toy_cfg, ridge, margin=1e6).run(toy)
    assert r.verdict == "adds_noise"


def test_bundled_example_data():
    df, cfg = load_tip_example(), tip_example_config()
    assert df.shape == (384, 16) and df.ID_Zk.nunique() == 96
    assert set(cfg.required_columns()) <= set(df.columns)
    assert (df.groupby("ID_Zk").size() == 4).all()


def test_bundled_example_passes_diagnostic():
    from cavity_encoding_ml import cavity_diagnostic

    df = load_tip_example()
    d = cavity_diagnostic(df.Value, df.ID_Zk, df.CavityNumber)
    assert d.passed and d.F > 100 and 0.4 < d.separation < 0.65


def test_workflow_default_has_readings(toy, toy_cfg):
    rf = RandomForestRegressor(n_estimators=20, random_state=0)
    r = CavityWorkflow(toy_cfg, rf).run(toy)
    assert set(r.validation.arms) == {"blind", "readings", "encoded"}
    assert "R1 cavity readings" in r.report() and "Leakage audit" not in r.report()
    assert "Recommendation:" in r.report() and "Result:" in r.report()


def test_workflow_lean(toy, toy_cfg):
    rf = RandomForestRegressor(n_estimators=20, random_state=0)
    r = CavityWorkflow(toy_cfg, rf, arms=("blind", "encoded")).run(toy)
    assert set(r.validation.arms) == {"blind", "encoded"} and "R1" not in r.report()


def test_workflow_with_readings(toy, toy_cfg):
    rf = RandomForestRegressor(n_estimators=20, random_state=0)
    r = CavityWorkflow(toy_cfg, rf, arms=("blind", "readings", "encoded")).run(toy)
    assert "R1 cavity readings" in r.report()


def test_workflow_needs_blind_and_encoded(toy_cfg):
    import pytest
    with pytest.raises(ValueError):
        CavityWorkflow(toy_cfg, RandomForestRegressor(), arms=("encoded",))
