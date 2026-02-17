import logging
import torch
from importlib import resources
from typing import List

from flexdata_metric_prediction.tree.tree_nodes import NODE_TYPES
from flexdata_metric_prediction.encoder.hint_encoder import HintEncoder
from flexdata_metric_prediction.model.bottom_up_gnn import MLP, BottomUpGNN
from flexdata_metric_prediction.model.multi_head_model import MultiHeadModel
from flexdata_metric_prediction.model.set_aggregation_model import SetAggregationModel


class ModelBuilder:
    def __init__(
        self,
        net_cfg: dict,
        model_cfg: dict,
        train_cfg: dict,
    ):
        self.net_cfg = net_cfg
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg

    def build_encoder(self) -> HintEncoder:
        if "encoder" in self.net_cfg:
            if self.net_cfg["encoder"] == "hintEncoder":
                with resources.path("flexdata_metric_prediction.encoder", "opMappingNew.json") as fpath:
                    op_mappings = str(fpath)
                with resources.path("flexdata_metric_prediction.encoder", "relMapping.json") as fpath:
                    rel_mappings = str(fpath)
                with resources.path("flexdata_metric_prediction.encoder", "typeMapping.json") as fpath:
                    type_mappings = str(fpath)
                encoder = HintEncoder(op_mappings, rel_mappings, type_mappings)
            else:
                raise RuntimeError("Unknown encoder: " + self.net_cfg["encoder"])
        else:
            raise NotImplementedError("net_cfg objects needs to have an encoder key")
        return encoder

    def build_model(self, input_dims: dict, engines: List) -> MultiHeadModel:
        for nodeType in NODE_TYPES:
            if nodeType.__name__ not in input_dims:
                logging.warning(
                    f"{nodeType.__name__} was not among the node types in the training set. Assuming absent and setting feature dimension to one."
                )
                input_dims[nodeType.__name__] = 1

        if "model_type" in self.model_cfg:
            if self.model_cfg["model_type"] == "BottomUpGNN":
                encoder = BottomUpGNN(input_dims, self.model_cfg)
                heads = [
                    MLP(self.model_cfg["final_mlp"]["output_dim"], self.model_cfg["head"]) for _ in range(len(engines))
                ]
            elif self.model_cfg["model_type"] == "Set":
                set_encoders = torch.nn.ModuleDict(
                    {
                        node_type: MLP(dim, self.model_cfg["encoder"]).to(dtype=self.train_cfg["dtype"])
                        for node_type, dim in input_dims
                    }
                )
                encoder = SetAggregationModel(set_encoders=set_encoders)
                heads = [
                    MLP(
                        len(set_encoders) * self.model_cfg["encoder"]["output_dim"],
                        self.model_cfg["head"],
                    ).to(dtype=self.train_cfg["dtype"])
                    for _ in range(len(engines))
                ]
            else:
                raise NotImplementedError(f"Unknown model type: {self.model_cfg['model_type']}")
        else:
            raise NotImplementedError("model_cfg needs to have a model_type key")

        model = MultiHeadModel(encoder_model=encoder)
        for head, engine in zip(heads, engines):
            model.register_head(engine, head)

        return model
