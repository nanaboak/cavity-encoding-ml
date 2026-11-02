import numpy as np

from cavity_encoding_ml import screen_outliers


def test_flags_extreme_value():
    rng = np.random.default_rng(0)
    y = rng.normal(100, 1, 200)
    cav = np.tile([1, 2, 3, 4], 50)
    y[7] = 140
    keep = screen_outliers(y, cav)
    assert not keep[7] and keep.sum() >= 190


def test_clean_data_kept():
    rng = np.random.default_rng(1)
    y = rng.normal(100, 1, 400)
    assert screen_outliers(y, np.tile([1, 2, 3, 4], 100), threshold=6).all()


def test_raw_scale_option():
    y = np.r_[np.random.default_rng(2).normal(0, 1, 99), 30.0]
    keep = screen_outliers(y, np.ones(100), boxcox=False)
    assert not keep[-1]


def test_per_cavity_scope():
    rng = np.random.default_rng(3)
    y = np.r_[rng.normal(100, 1, 100), rng.normal(120, 1, 100)]
    cav = np.r_[np.ones(100), 2 * np.ones(100)]
    assert screen_outliers(y, cav, boxcox=False).all()
