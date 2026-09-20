import numpy as np
import torch


def encode(board, to_play, size=9):
    b = np.asarray(board, dtype=np.float32).reshape((size, size))
    own = (b == to_play).astype(np.float32)
    opp = (b == -to_play).astype(np.float32)
    turn = np.ones((size, size), dtype=np.float32)
    return torch.from_numpy(np.stack([own, opp, turn])).unsqueeze(0)


def policy_size(size):
    return size * size + 1
