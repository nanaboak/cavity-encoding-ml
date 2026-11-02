import numpy as np
import pytest
from sklearn.model_selection import KFold

from cavity_encoding_ml import (LeakageWarning, RepeatedRowKFold, ShotGroupKFold, sibling_share)
from cavity_encoding_ml.splitting import check_splits

G = np.repeat(np.arange(24), 4)


def test_no_shot_shared_between_train_and_test():
    for tr, te in ShotGroupKFold(4, 2, 7).split(groups=G):
        assert not set(G[tr]) & set(G[te])


def test_every_part_tested_once_per_repeat():
    te_all = np.concatenate([te for _, te in ShotGroupKFold(4, 1).split(groups=G)])
    assert sorted(te_all.tolist()) == list(range(len(G)))


def test_number_of_splits():
    cv = ShotGroupKFold(4, 3)
    assert cv.get_n_splits() == 12 == len(list(cv.split(groups=G)))


def test_reproducible_and_seed_dependent():
    a = [te.tolist() for _, te in ShotGroupKFold(4, 1, 1).split(groups=G)]
    b = [te.tolist() for _, te in ShotGroupKFold(4, 1, 1).split(groups=G)]
    c = [te.tolist() for _, te in ShotGroupKFold(4, 1, 2).split(groups=G)]
    assert a == b and a != c


def test_repeats_differ():
    sp = [te.tolist() for _, te in ShotGroupKFold(4, 2).split(groups=G)]
    assert sp[:4] != sp[4:]


def test_groups_required():
    with pytest.raises(ValueError):
        list(ShotGroupKFold().split(X=np.zeros(10)))


def test_fold_sizes_balanced():
    sizes = [len(te) for _, te in ShotGroupKFold(4).split(groups=G)]
    assert max(sizes) - min(sizes) <= 4


def test_integer_and_string_ids_agree():
    a = [te.tolist() for _, te in ShotGroupKFold().split(groups=G)]
    b = [te.tolist() for _, te in ShotGroupKFold().split(groups=G.astype(str))]
    assert a == b


def test_few_shots_reduce_folds():
    g = np.repeat([1, 2, 3], 4)
    assert len(list(ShotGroupKFold(10).split(groups=g))) == 3


def test_sibling_share_zero_for_grouped():
    assert sibling_share(G, ShotGroupKFold(4, 2).split(groups=G)) == 0.0


def test_sibling_share_high_for_kfold():
    share = sibling_share(G, RepeatedRowKFold(4, 3).split(np.zeros(len(G))))
    assert share > 0.9


def test_row_kfold_matches_sklearn():
    ours = [te.tolist() for _, te in RepeatedRowKFold(4, 1, 42).split(np.zeros(40))]
    ref = [te.tolist() for _, te in KFold(4, shuffle=True, random_state=42).split(np.zeros(40))]
    assert ours == ref


def test_check_splits_warns():
    with pytest.warns(LeakageWarning, match="ShotGroupKFold"):
        check_splits(G, list(RepeatedRowKFold(4).split(np.zeros(len(G)))))


def test_check_splits_silent_for_grouped(recwarn):
    check_splits(G, list(ShotGroupKFold(4).split(groups=G)))
    assert not [w for w in recwarn if issubclass(w.category, LeakageWarning)]
