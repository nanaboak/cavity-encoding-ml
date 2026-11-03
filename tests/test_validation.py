import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from cavity_encoding_ml import (FunctionModel, LeakageWarning, RepeatedRowKFold, ShotGroupKFold,
                                compare_encodings, leakage_audit, validate_cavity_encoding)


@pytest.fixture
def rf():
    return RandomForestRegressor(n_estimators=30, random_state=0, n_jobs=1)


def test_three_arms_and_folds(toy, toy_cfg, ridge):
    v = validate_cavity_encoding(toy, toy_cfg, ridge)
    assert set(v.arms) == {"blind", "readings", "encoded"} and v.n_folds == 12
    assert v.grouped and v.leakage_share == 0.0


def test_label_helps_on_toy_data(toy, toy_cfg, ridge):
    v = validate_cavity_encoding(toy, toy_cfg, ridge)
    assert v.encoded.rmse < v.blind.rmse and v.gain() > 0.1


def test_oof_has_no_nan(toy, toy_cfg, ridge):
    v = validate_cavity_encoding(toy, toy_cfg, ridge)
    assert np.isfinite(v.encoded.oof).all()


def test_jitter_definition(toy, toy_cfg, ridge):
    v = validate_cavity_encoding(toy, toy_cfg, ridge)
    f = v.blind.fold_rmse
    assert v.jitter == pytest.approx(np.std(f, ddof=1) / np.mean(f))


@pytest.mark.parametrize("enc", ["onehot", "target", "integer", "auto"])
def test_all_encodings_run(toy, toy_cfg, ridge, enc):
    v = validate_cavity_encoding(toy, toy_cfg, ridge, encoding=enc, cv=ShotGroupKFold(4, 1))
    assert v.encoded.n_inputs >= 4


def test_bad_encoding(toy, toy_cfg, ridge):
    with pytest.raises(ValueError):
        validate_cavity_encoding(toy, toy_cfg, ridge, encoding="hash")


def test_alert_for_ordinary_kfold(toy, toy_cfg, ridge):
    with pytest.warns(LeakageWarning, match="Prefer ShotGroupKFold"):
        v = validate_cavity_encoding(toy, toy_cfg, ridge, cv=KFold(4, shuffle=True, random_state=0))
    assert not v.grouped and v.leakage_share > 0.8
    assert "ALERT" in v.report()


def test_no_alert_for_group_kfold(toy, toy_cfg, ridge, recwarn):
    from sklearn.model_selection import GroupKFold

    v = validate_cavity_encoding(toy, toy_cfg, ridge, cv=GroupKFold(4))
    assert v.grouped
    assert not [w for w in recwarn if issubclass(w.category, LeakageWarning)]


def test_warn_false_silences(toy, toy_cfg, ridge, recwarn):
    validate_cavity_encoding(toy, toy_cfg, ridge, cv=RepeatedRowKFold(4), warn=False)
    assert not [w for w in recwarn if issubclass(w.category, LeakageWarning)]


def test_ordinary_kfold_looks_better(toy, toy_cfg, rf):
    g = validate_cavity_encoding(toy, toy_cfg, rf, cv=ShotGroupKFold(4, 2))
    with pytest.warns(LeakageWarning):
        k = validate_cavity_encoding(toy, toy_cfg, rf, cv=RepeatedRowKFold(4, 2))
    assert k.blind.rmse < g.blind.rmse


def test_function_model(toy, toy_cfg):
    def fit(X, y):
        A = np.c_[X, np.ones(len(X))]
        return np.linalg.lstsq(A, y, rcond=None)[0]

    fm = FunctionModel(fit, lambda w, X: np.c_[X, np.ones(len(X))] @ w)
    v = validate_cavity_encoding(toy, toy_cfg, fm)
    assert v.encoded.rmse < v.blind.rmse


def test_factory_model(toy, toy_cfg):
    from sklearn.linear_model import Ridge

    v = validate_cavity_encoding(toy, toy_cfg, lambda: Ridge(alpha=2.0))
    assert np.isfinite(v.blind.rmse)


def test_original_data_not_modified(toy, toy_cfg, ridge):
    before = toy.copy()
    validate_cavity_encoding(toy, toy_cfg, ridge)
    assert before.equals(toy)


def test_report_text(toy, toy_cfg, ridge):
    t = validate_cavity_encoding(toy, toy_cfg, ridge, response_unit="mm").report()
    assert "STEP 3" in t and "R0 cavity-blind" in t and "jitter" in t


def test_compare_encodings(toy, toy_cfg, ridge):
    out = compare_encodings(toy, toy_cfg, ridge, cv=ShotGroupKFold(4, 1))
    assert set(out) == {"onehot", "target", "integer"}


def test_leakage_audit(toy, toy_cfg, rf):
    a = leakage_audit(toy, toy_cfg, rf, n_repeats=2)
    assert a["sibling_share"] > 0.9 and a["understatement"] > 0


def test_same_folds_for_all_arms(toy, toy_cfg, ridge):
    v = validate_cavity_encoding(toy, toy_cfg, ridge)
    assert len(v.blind.fold_rmse) == len(v.readings.fold_rmse) == len(v.encoded.fold_rmse)


def test_pooled_rmse_matches_oof(toy, toy_cfg, ridge):
    v = validate_cavity_encoding(toy, toy_cfg, ridge)
    resid = toy.y.to_numpy() - v.blind.oof
    assert v.blind.rmse == pytest.approx(np.sqrt(np.mean(resid**2)))


def test_arms_encoded_only(toy, toy_cfg, rf):
    v = validate_cavity_encoding(toy, toy_cfg, rf, arms=("encoded",))
    assert list(v.arms) == ["encoded"] and v.n_folds == 12
    assert v.leakage_share == 0.0 and v.grouped
    assert np.isnan(v.jitter) and np.isnan(v.gain())
    text = v.report()
    assert "onehot label" in text and "gain vs blind" not in text and "jitter" not in text


def test_arms_blind_and_encoded(toy, toy_cfg, rf):
    v = validate_cavity_encoding(toy, toy_cfg, rf, arms=("blind", "encoded"))
    assert list(v.arms) == ["blind", "encoded"]
    assert "readings" not in v.report() and v.jitter > 0


def test_arms_match_full_run(toy, toy_cfg, rf):
    full = validate_cavity_encoding(toy, toy_cfg, rf)
    lean = validate_cavity_encoding(toy, toy_cfg, rf, arms=("blind", "encoded"))
    assert lean.blind.rmse == full.blind.rmse and lean.encoded.rmse == full.encoded.rmse


def test_arms_invalid(toy, toy_cfg, rf):
    with pytest.raises(ValueError):
        validate_cavity_encoding(toy, toy_cfg, rf, arms=("blind", "nonsense"))
    with pytest.raises(ValueError):
        validate_cavity_encoding(toy, toy_cfg, rf, arms=())
