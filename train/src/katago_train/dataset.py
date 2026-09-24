import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from katago_train.encode import encode, policy_size

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "selfplay"


class GoDataset(Dataset):
    def __init__(self, root=DEFAULT_DATA_DIR, size=9):
        self.size = size
        self.samples = []
        for path in sorted(Path(root).glob("*.json")):
            self.samples.extend(self._load_game(path))

    def _load_game(self, path):
        data = json.loads(path.read_text(encoding="utf-8"))
        size = data["size"]
        result = data["result"]
        board = np.zeros((size, size), dtype=np.float32)
        samples = []
        for move in data["moves"]:
            color = move["color"]
            pos = move["pos"]
            samples.append((
                encode(board, color, size)[0],
                self._policy_target(move, pos, size),
                torch.tensor([result * color], dtype=torch.float32),
            ))
            if pos != size * size:
                y, x = divmod(pos, size)
                board[y, x] = color
        return samples

    def _policy_target(self, move, pos, size):
        policy = move.get("policy")
        if policy is not None:
            policy = np.asarray(policy, dtype=np.float32)
            if policy.shape != (policy_size(size),):
                raise ValueError(
                    f"{move} 的 policy 长度应为 {policy_size(size)}，实际 {len(policy)}"
                )
            return torch.from_numpy(policy)
        target = np.zeros(policy_size(size), dtype=np.float32)
        target[pos] = 1.0
        return torch.from_numpy(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]