import torch


def evaluate(model, dataloader, criterion, metrics, device, split="val"):
    model.eval()
    model.to(device=device, dtype=torch.float64)
    total_loss = []
    with torch.no_grad():
        for datapoint in dataloader:
            datapoint.to(device=device)
            out = model(datapoint)
            loss = criterion(out, datapoint.y.reshape(out.shape))
            total_loss.append(loss)
            metrics.update(out.cpu(), datapoint.y.cpu(), split)
    return torch.mean(torch.tensor(total_loss))


def evaluate_multihead(model, dataloader, criterion, metrics, device, dtype, split="val"):
    """Unified evaluation for multi-head models.

    Handles both:
    - Single-metric per head: criterion applied per-head with NaN masking
    - Multi-metric per head: criterion (e.g. MultiMetricLoss) applied to concatenated outputs
    """
    model.eval()
    model.to(device=device, dtype=dtype)
    total_loss = []

    # Check if this is a multi-metric loss (has num_metrics_per_engine attribute)
    is_multi_metric = hasattr(criterion, "num_metrics_per_engine")

    with torch.no_grad():
        for datapoint in dataloader:
            datapoint.to(device=device)
            out = model(datapoint)

            if is_multi_metric:
                # Multi-metric: concatenate all head outputs and apply loss once
                all_preds = torch.cat(out, dim=1)
                loss = criterion(all_preds, datapoint.y)
                # Update tracker with concatenated predictions
                if metrics is not None:
                    metrics.update(all_preds, datapoint.y, split)
            else:
                # Single-metric per head: apply criterion per-head with NaN masking
                losses = []
                for head_idx, head_out in enumerate(out):
                    mask = torch.isnan(datapoint.y[:, head_idx].reshape(-1, 1)).squeeze(-1)
                    losses.append(criterion(head_out[~mask], datapoint.y[~mask, head_idx].reshape(-1, 1)))
                loss = sum(losses) / len(losses)
                if metrics is not None:
                    metrics.update(out, datapoint.y, split)

            total_loss.append(loss)

    return torch.mean(torch.tensor(total_loss))


def evaluate_multihead_detailed(model, dataloader, criterion, device, dtype) -> dict:
    """Evaluate with per-metric loss breakdown.

    Returns:
        Dict with "total" loss and "per_metric" breakdown (if criterion supports it)
    """
    model.eval()
    model.to(device=device, dtype=dtype)

    # Accumulate all predictions and targets
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for datapoint in dataloader:
            datapoint.to(device=device)
            out = model(datapoint)
            preds = torch.cat(out, dim=1)
            all_preds.append(preds)
            all_targets.append(datapoint.y)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Get detailed loss if criterion supports it
    if hasattr(criterion, "forward_detailed"):
        return criterion.forward_detailed(all_preds, all_targets)
    else:
        return {"total": criterion(all_preds, all_targets), "per_metric": {}}


# Alias for backward compatibility
evaluate_multi_metric = evaluate_multihead
