import logging
from typing import Any

import torch
from torch import nn


class MultiHeadModel(nn.Module):
    def __init__(self, encoder_model: nn.Module, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.heads = nn.ModuleDict({})
        self.encoder = encoder_model

    def register_head(
        self,
        engine: str,
        head: nn.Module,
        overwrite: bool = False,
        dtype: torch.dtype = torch.float64,
    ):
        head = head.to(dtype)
        if engine not in self.heads or overwrite:
            self.heads[engine] = head
        else:
            logging.warning(
                f"Engine's predictor head already initialized for engine: {engine} and overwrite was not set... Skipping"
            )

    def freeze_encoder_model(self):
        logging.info("Freezing parameters of encoder model!")
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, data):
        embedding = self.encoder(data)
        outputs = [self.heads[engine](embedding) for engine in self.heads]
        return outputs

    def predict_engines(self, data) -> dict[str, Any]:
        """Return predictions for each engine as a dictionary.
        Used for flexdata_metric_prediction_service.

        Args:
            data (_type_): Input GNN HeteroData datapoint(s)

        Returns:
            dict[str, Any]: dictionary containing predictions for each registered engine
        """
        outputs = self.forward(data)
        return {engine: outputs[i] for i, engine in enumerate(self.heads)}

    def forward_embeddings(self, data):
        embedding = self.encoder(data)
        return embedding
