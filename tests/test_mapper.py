import numpy as np
import pandas as pd
import pytest

from cavity_encoding_ml import CavityConfig, CavityMapper, LeakageError, REPRESENTATIONS


def test_config_default_cavities():
    c = CavityConfig("s", "c", "y", sensors={"P": ["a", "b", "c", "d"]})
    assert c.cavities == [1, 2, 3, 4]


def test_config_length_check():
    with pytest.raises(ValueError):
        CavityConfig("s", "c", "y", sensors={"P": ["a", "b"]}, cavities=[1, 2, 3])


def test_config_json_roundtrip(tmp_path, toy_cfg):
    p = tmp_path / "c.json"
    toy_cfg.to_json(str(p))
    assert CavityConfig.from_json(str(p)) == toy_cfg


def test_required_columns(toy_cfg):
    assert "shot" in toy_cfg.required_columns() and "p4" in toy_cfg.required_columns()


@pytest.mark.parametrize("rep,width", [("R0", 3), ("R1", 3), ("R2", 4), ("R3", 7), ("R4", 4)])
def test_widths(toy, toy_cfg, rep, width):
    Z = CavityMapper(toy_cfg, rep).fit_transform(toy, toy.y)
    assert Z.shape == (len(toy), width)


def test_r0_is_shot_mean(toy, toy_cfg):
    Z = CavityMapper(toy_cfg, "R0").fit_transform(toy)
    expected = toy[["p1", "p2", "p3", "p4"]].mean(axis=1).to_numpy()
    assert np.allclose(Z[:, -1], expected)


def test_r1_picks_producing_cavity(toy, toy_cfg):
    Z = CavityMapper(toy_cfg, "R1").fit_transform(toy)
    for i in (0, 1, 2, 3, 50):
        c = int(toy.cav.iloc[i])
        assert Z[i, -1] == toy[f"p{c}"].iloc[i]


def test_r0_identical_inside_shot(toy, toy_cfg):
    Z = CavityMapper(toy_cfg, "R0").fit_transform(toy)
    first = toy.shot.to_numpy() == toy.shot.iloc[0]
    assert np.ptp(Z[first], axis=0).max() == 0


def test_r3_columns_one_hot(toy, toy_cfg):
    Z = CavityMapper(toy_cfg, "R3").fit_transform(toy)
    assert (Z[:, -4:].sum(axis=1) == 1).all()


def test_aliases(toy, toy_cfg):
    a = CavityMapper(toy_cfg, "onehot").fit_transform(toy)
    b = CavityMapper(toy_cfg, "R3").fit_transform(toy)
    assert np.array_equal(a, b)


def test_unknown_representation(toy, toy_cfg):
    with pytest.raises(ValueError):
        CavityMapper(toy_cfg, "R9").fit(toy)


def test_missing_column(toy, toy_cfg):
    with pytest.raises(KeyError):
        CavityMapper(toy_cfg, "R3").fit(toy.drop(columns=["p2"]))


def test_unknown_cavity_value(toy, toy_cfg):
    bad = toy.copy()
    bad.loc[0, "cav"] = 9
    with pytest.raises(ValueError):
        CavityMapper(toy_cfg, "R1").fit_transform(bad)


def test_target_uses_training_only(toy, toy_cfg):
    tr, te = toy.iloc[:80], toy.iloc[80:]
    m = CavityMapper(toy_cfg, "R4", strict=False).fit(tr, tr.y)
    Z = m.transform(te)
    assert np.isfinite(Z).all()


def test_strict_blocks_shared_shots(toy, toy_cfg):
    m = CavityMapper(toy_cfg, "R3").fit(toy.iloc[:100])
    with pytest.raises(LeakageError):
        m.transform(toy.iloc[50:120])


def test_strict_allows_new_shots(toy, toy_cfg):
    sh = toy.shot.unique()
    tr, te = toy[toy.shot.isin(sh[:30])], toy[toy.shot.isin(sh[30:])]
    Z = CavityMapper(toy_cfg, "R3").fit(tr).transform(te)
    assert len(Z) == len(te)


def test_not_strict_allows_overlap(toy, toy_cfg):
    m = CavityMapper(toy_cfg, "R3", strict=False).fit(toy)
    assert len(m.transform(toy)) == len(toy)


def test_feature_names(toy, toy_cfg):
    m = CavityMapper(toy_cfg, "R3").fit(toy)
    assert list(m.get_feature_names_out())[-4:] == ["Cavity_1", "Cavity_2", "Cavity_3", "Cavity_4"]


def test_needs_dataframe(toy_cfg):
    with pytest.raises(TypeError):
        CavityMapper(toy_cfg, "R3").fit(np.zeros((4, 4)))


def test_no_sensors_config(toy):
    cfg = CavityConfig("shot", "cav", "y", shared=["x1"])
    assert CavityMapper(cfg, "R0").fit_transform(toy).shape[1] == 1
    assert CavityMapper(cfg, "R3", strict=False).fit_transform(toy).shape[1] == 5


def test_representation_table():
    assert list(REPRESENTATIONS) == ["R0", "R1", "R2", "R3", "R4"]
