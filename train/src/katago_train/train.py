import argparse

import torch
from torch.utils.data import DataLoader

from katago_train.dataset import DEFAULT_DATA_DIR, GoDataset
from katago_train.model import Net


def policy_loss(logits, target):
    log_probs = torch.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--size", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GoDataset(args.data, args.size)
    if len(dataset) == 0:
        raise SystemExit(f"没有棋谱：{args.data}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Net(args.size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, p_target, v_target in loader:
            x = x.to(device)
            p_target = p_target.to(device)
            v_target = v_target.to(device)
            p_logits, v = model(x)
            loss = policy_loss(p_logits, p_target) + torch.nn.functional.mse_loss(v, v_target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x)
        print(f"epoch {epoch:3d}  loss {total / len(dataset):.4f}")

    torch.save(model.state_dict(), args.checkpoint)
    print(f"saved {args.checkpoint}")


if __name__ == "__main__":
    main()