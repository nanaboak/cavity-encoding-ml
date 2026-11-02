import numpy as np
import pytest

from cavity_encoding_ml import CavityEncoder, choose_encoding


def test_onehot_shape_and_values():
    e = CavityEncoder("onehot").fit([1, 2, 3, 4])
    Z = e.transform([2, 4, 1])
    assert Z.shape == (3, 4) and Z.sum(axis=1).tolist() == [1, 1, 1]
    assert Z[0].tolist() == [0, 1, 0, 0]


def test_onehot_fixed_cavities_when_one_missing():
    e = CavityEncoder("onehot", cavities=[1, 2, 3, 4]).fit([1, 2, 3])
    assert e.transform([4]).tolist() == [[0, 0, 0, 1]]


def test_integer_is_identity():
    Z = CavityEncoder("integer").fit([1, 2, 3]).transform([3, 1])
    assert Z.tolist() == [[3.0], [1.0]]


def test_target_needs_y():
    with pytest.raises(ValueError):
        CavityEncoder("target").fit([1, 2])


def test_target_values_smoothing():
    cav = np.array([1, 1, 2, 2])
    y = np.array([0.0, 2.0, 10.0, 12.0])
    e = CavityEncoder("target", smoothing=2.0).fit(cav, y)
    prior = y.mean()
    assert e.transform([1])[0, 0] == pytest.approx((2 * 1.0 + 2 * prior) / 4)
    assert e.transform([2])[0, 0] == pytest.approx((2 * 11.0 + 2 * prior) / 4)


def test_target_unknown_uses_prior():
    e = CavityEncoder("target").fit([1, 2], [1.0, 3.0])
    assert e.transform([9])[0, 0] == pytest.approx(2.0)


def test_target_large_smoothing_goes_to_prior():
    e = CavityEncoder("target", smoothing=1e9).fit([1, 1, 2, 2], [0, 0, 10, 10.0])
    assert e.transform([1])[0, 0] == pytest.approx(5.0, abs=1e-3)


def test_feature_names():
    assert list(CavityEncoder("onehot").fit([1, 2]).get_feature_names_out()) == ["Cavity_1", "Cavity_2"]
    assert list(CavityEncoder("integer").fit([1, 2]).get_feature_names_out()) == ["CavityID_integer"]


def test_bad_method():
    with pytest.raises(ValueError):
        CavityEncoder("nope").fit([1, 2])


def test_choose_encoding():
    assert choose_encoding(4) == "onehot"
    assert choose_encoding(12) == "onehot"
    assert choose_encoding(32) == "target"


def test_sklearn_clone():
    from sklearn.base import clone

    e = clone(CavityEncoder("target", smoothing=5.0))
    assert e.smoothing == 5.0
