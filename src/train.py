"""Minimal training loop for the set-DEQ (CPU-friendly).

Layer 1 backpropagates through the unrolled solver iterations -- adequate for
small N and few iterations. A later layer replaces this with implicit / phantom
gradients via torchdeq so train-time depth is decoupled from memory.
"""

import torch
import torch.nn.functional as F


def train(model, dataset, epochs=10, batch_size=64, lr=1e-3, seed=0, log_every=1,
          device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for X, y in dataset.iter_batches(batch_size, shuffle=True, seed=seed + epoch):
            X, y = X.to(device), y.to(device)
            logits, _ = model(X)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += int((logits.argmax(-1) == y).sum())
            total += len(y)
        acc = correct / total
        history.append({"epoch": epoch, "loss": loss_sum / total, "acc": acc})
        if log_every and epoch % log_every == 0:
            print(f"epoch {epoch:3d}  loss {loss_sum / total:.4f}  acc {acc:.3f}")
    return history


@torch.no_grad()
def evaluate(model, dataset, batch_size=128, device="cpu"):
    model.to(device).eval()
    correct, total = 0, 0
    for X, y in dataset.iter_batches(batch_size, shuffle=False):
        X, y = X.to(device), y.to(device)
        logits, _ = model(X)
        correct += int((logits.argmax(-1) == y).sum())
        total += len(y)
    return correct / total
