import numpy as np
import pandas as pd
import pytest

from cavity_encoding_ml import cavity_diagnostic


def test_clear_effect_passes(toy):
    d = cavity_diagnostic(toy.y, toy.shot, toy.cav)
    assert d.passed and d.p_value < 1e-6
    assert d.n_shots == 40 and d.n_cavities == 4 and d.n_parts == 160
    assert d.df1 == 3 and d.df2 == 160 - 40 - 3


def test_no_effect_fails(noeffect):
    d = cavity_diagnostic(noeffect.y, noeffect.shot, noeffect.cav)
    assert not d.passed


def test_floor_identity_balanced(toy):
    d = cavity_diagnostic(toy.y, toy.shot, toy.cav)
    assert d.F == pytest.approx(d.F_from_floors, rel=1e-9)


def test_separation_formula(toy):
    d = cavity_diagnostic(toy.y, toy.shot, toy.cav)
    assert d.separation == pytest.approx(1 - d.sigma_noise / d.sigma_blind)
    assert 0 < d.separation < 1


def test_offsets_sum_to_zero(toy):
    d = cavity_diagnostic(toy.y, toy.shot, toy.cav)
    assert sum(d.offsets.values()) == pytest.approx(0, abs=1e-6)


def test_missing_parts_allowed(toy):
    sub = toy.drop(index=[3, 10, 77])
    d = cavity_diagnostic(sub.y, sub.shot, sub.cav)
    assert d.n_parts == 157 and d.df2 == 157 - 40 - 3 and d.passed


def test_nan_response_is_dropped(toy):
    y = toy.y.copy()
    y.iloc[0] = np.nan
    assert cavity_diagnostic(y, toy.shot, toy.cav).n_parts == 159


def test_strings_as_shot_ids(toy):
    d1 = cavity_diagnostic(toy.y, toy.shot, toy.cav)
    d2 = cavity_diagnostic(toy.y, toy.shot.astype(str) + "x", toy.cav)
    assert d1.F == pytest.approx(d2.F)


def test_min_eta_blocks_small_effect(toy):
    d = cavity_diagnostic(toy.y, toy.shot, toy.cav, min_partial_eta2=0.999)
    assert not d.passed


def test_length_mismatch():
    with pytest.raises(ValueError):
        cavity_diagnostic([1, 2, 3], [1, 1], [1, 2, 3])


def test_one_cavity_rejected():
    with pytest.raises(ValueError):
        cavity_diagnostic(np.arange(20.0), np.repeat(np.arange(10), 2), np.ones(20))


def test_too_few_parts():
    with pytest.raises(ValueError):
        cavity_diagnostic([1, 2, 3, 4], [1, 1, 2, 2], [1, 2, 1, 2])


def test_single_part_shots_rejected():
    with pytest.raises(ValueError):
        cavity_diagnostic(np.arange(10.0), np.arange(10), np.tile([1, 2], 5))


def test_report_text(toy):
    txt = cavity_diagnostic(toy.y, toy.shot, toy.cav).report("mm")
    assert "STEP 1" in txt and "PASSED" in txt and "F(3" in txt


def test_to_dict(toy):
    d = cavity_diagnostic(toy.y, toy.shot, toy.cav).to_dict()
    assert set(["F", "separation", "passed", "offsets"]) <= set(d)


def test_scale_invariance_of_F(toy):
    a = cavity_diagnostic(toy.y, toy.shot, toy.cav)
    b = cavity_diagnostic(toy.y * 1000 + 5, toy.shot, toy.cav)
    assert a.F == pytest.approx(b.F, rel=1e-6)
    assert a.separation == pytest.approx(b.separation, rel=1e-6)
