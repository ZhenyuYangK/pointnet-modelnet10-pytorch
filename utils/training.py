"""One-epoch training and validation with sample-weighted metrics."""

import torch


def train_one_epoch(model, loader, optimizer, criterion, device, progress_every=0):
    """Update model parameters on training batches; criterion uses mean reduction."""
    model.train()
    loss_sum, correct, count = 0.0, 0, 0
    for batch_index, (points, labels) in enumerate(loader, start=1):
        points, labels = points.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(points)
        loss = criterion(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        # 按样本数加权，避免最后一个较小 batch 与完整 batch 占相同权重。
        count += len(labels)
        loss_sum += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        if progress_every and batch_index % progress_every == 0:
            print(f"  train batch {batch_index}/{len(loader)}: samples={count}, "
                  f"loss={loss_sum / count:.4f}, acc={correct / count:.1%}", flush=True)
    if count == 0:
        raise ValueError("Training loader yielded no samples")
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate held-out samples without gradients or updates to BatchNorm buffers."""
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for points, labels in loader:
        points, labels = points.to(device), labels.to(device)
        logits = model(points)
        loss = criterion(logits, labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite validation loss")
        count += len(labels)
        loss_sum += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
    if count == 0:
        raise ValueError("Validation loader yielded no samples")
    return {"loss": loss_sum / count, "accuracy": correct / count, "samples": count}
