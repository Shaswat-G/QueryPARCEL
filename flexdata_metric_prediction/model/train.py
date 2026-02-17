import torch
from tqdm import tqdm


def train_step(model, optimizer, criterion, dataloader, metrics, device):
    model.train()
    model.to(device=device, dtype=torch.float64)
    total_loss = []
    for _, datapoint in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        datapoint.to(device=device)
        out = model(datapoint)
        # print(out)
        # print(out)
        loss = criterion(out, datapoint.y)
        metrics.update(out, datapoint.y, "train")
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0)
        optimizer.step()
        total_loss.append(loss)
    return torch.mean(torch.tensor(total_loss))


def train_step_multihead(
    model, optimizer, criterion, dataloader, metrics, device, dtype, grad_clip: float | None = None
):
    """Unified training step for multi-head models.

    Handles both:
    - Single-metric per head: criterion applied per-head with NaN masking
    - Multi-metric per head: criterion (e.g. MultiMetricLoss) applied to concatenated outputs

    Args:
        grad_clip: If set, clip gradients to this max norm. Useful for multi-metric with different scales.
    """
    model.train()
    model.to(device=device, dtype=dtype)
    criterion.to(dtype=dtype)
    total_loss = []

    # Check if this is a multi-metric loss (has num_metrics_per_engine attribute)
    is_multi_metric = hasattr(criterion, "num_metrics_per_engine")

    for datapoint in tqdm(dataloader):
        optimizer.zero_grad()
        datapoint.to(device=device)
        out = model(datapoint)

        if is_multi_metric:
            # Multi-metric: concatenate all head outputs and apply loss once
            all_preds = torch.cat(out, dim=1)
            loss = criterion(all_preds, datapoint.y)
            # Update tracker with concatenated predictions
            if metrics is not None:
                metrics.update(all_preds, datapoint.y, "train")
        else:
            # Single-metric per head: apply criterion per-head with NaN masking
            losses = []
            for head_idx, head_out in enumerate(out):
                mask = torch.isnan(datapoint.y[:, head_idx].reshape(-1, 1)).squeeze(-1)
                losses.append(criterion(head_out[~mask], datapoint.y[~mask, head_idx].reshape(-1, 1)))
            loss = sum(losses) / len(losses)
            if metrics is not None:
                metrics.update(out, datapoint.y, "train")

        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        total_loss.append(loss.detach())  # Detach to release computation graph

    return torch.mean(torch.stack(total_loss))


# Alias for backward compatibility
train_step_multi_metric = train_step_multihead


def train_step_set(models, optimizers, criterion, dataloader, metrics, device, dtype):
    total_loss = []
    for model, _ in zip(models, optimizers):
        model.train()
        model.to(device=device, dtype=dtype)
        criterion.to(dtype=dtype)

    for _, datapoint in tqdm(enumerate(dataloader)):
        outs = []
        losses = []
        cur_model = 0
        for model, optimizer in zip(models, optimizers):
            optimizer.zero_grad()
            datapoint.to(device=device)
            out = model(datapoint)
            outs.append(out)
            loss = criterion(out, datapoint.y[:, cur_model].reshape(-1, 1))
            losses.append(loss)
            loss.backward()
            optimizer.step()

        metrics.update(outs, datapoint.y, "train")

        total_loss.append(sum(losses) / len(losses))
    return torch.mean(torch.tensor(total_loss))
