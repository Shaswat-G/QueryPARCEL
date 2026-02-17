import torch
from torch import nn


class SetAggregationModel(nn.Module):
    def __init__(self, set_encoders, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.set_encoders = set_encoders
        self.embedding_out_shape = 256  # HACK
        # self.output_model = output_model

    def forward(self, data):
        node_type_embeddings = {}
        dtype = data.node_stores[0].x.dtype
        if hasattr(data, "num_graphs"):
            batch_size = data.num_graphs
        else:
            batch_size = 1
        for nodeType in self.set_encoders:
            node_type_embedding = torch.zeros((batch_size, self.embedding_out_shape), dtype=dtype)
            if len(data[nodeType]) == 0:
                node_type_embeddings[nodeType] = node_type_embedding
                continue
            converted_embeddings = self.set_encoders[nodeType](data[nodeType].x)

            i_graph = 0
            for node_start, node_end in zip(data[nodeType].ptr[:-1], data[nodeType].ptr[1:]):
                if 0 < node_end - node_start:
                    node_type_embedding[i_graph] = torch.mean(converted_embeddings[node_start:node_end], dim=0)
                i_graph += 1
            node_type_embeddings[nodeType] = node_type_embedding
        # for nodeType in self.set_encoders:
        # out = self.output_model(torch.cat(node_type_embeddings))
        return torch.cat(list(node_type_embeddings.values()), dim=1)
