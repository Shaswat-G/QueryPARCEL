from torch import nn


def model_summary(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Estimate model size in bytes (using parameter element size)
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024**2)

    out_string = "Model shape:\n" + str(model) + "\n"
    out_string += f"Total parameters: {total_params}\n"
    out_string += f"Trainable parameters: {trainable_params}\n"
    out_string += f"Estimated model size: {model_size_mb:.2f} MB"
    return out_string
