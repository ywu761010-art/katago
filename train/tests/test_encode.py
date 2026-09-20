import numpy as np

from katago_train.encode import encode, policy_size


def test_empty_board_shape():
    t = encode(np.zeros((9, 9), dtype=np.float32), 1)
    assert tuple(t.shape) == (1, 3, 9, 9)


def test_policy_size():
    assert policy_size(9) == 82


def test_empty_board_channels():
    t = encode(np.zeros((9, 9), dtype=np.float32), 1)
    assert float(t[0, 0].sum()) == 0
    assert float(t[0, 1].sum()) == 0
    assert float(t[0, 2].sum()) == 81