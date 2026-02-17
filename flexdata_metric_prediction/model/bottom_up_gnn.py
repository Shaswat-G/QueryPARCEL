import logging

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, MessagePassing
from torch_geometric.utils import degree


class LinearAggregate(MessagePassing):
    """Linear aggregation for GNN"""

    def __init__(self, normalize=False):
        super().__init__(aggr="sum")
        self.normalize = normalize

    def forward(self, x, edge_index):
        if self.normalize:
            # Normalize w.r.t. number of incoming edges
            if isinstance(x, tuple):
                size = x[1].size(0)
            else:
                size = x.size(0)
            in_degree_norm = 1 / torch.sqrt(degree(edge_index[1], size))[edge_index[1]]
        else:
            in_degree_norm = None
        return self.propagate(edge_index, x=x, norm=in_degree_norm)

    def message(self, x_j, norm):  # type: ignore
        if self.normalize and x_j.shape[0] != 0:
            return norm.view(-1, 1) * x_j
        return x_j


class MLP(nn.Module):
    def __init__(self, input_dim, mlp_config):
        super().__init__()
        num_layers = mlp_config["num_layers"]
        width_factor = mlp_config["width_factor"]
        hidden_dim = mlp_config["hidden_dim"]
        output_dim = mlp_config["output_dim"]
        activation = mlp_config["activation"]

        if activation == "GELU":
            self.activation = nn.GELU()
        elif activation == "LeakyRelu":
            self.activation = nn.LeakyReLU()
        else:
            logging.warning(f"Unknown activation function ({activation}) passed to MLP. Defaulting to LeakyRelu.")
            self.activation = nn.LeakyReLU()

        self.layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [
                nn.Linear(
                    int(hidden_dim * (width_factor**i)),
                    int(hidden_dim * (width_factor ** (i + 1))),
                )
                for i in range(num_layers)
            ]
            + [nn.Linear(int(hidden_dim * (width_factor ** (num_layers))), output_dim)]
        )
        self.dropout = nn.Dropout(mlp_config["dropout"])

    def forward(self, x):
        out = x
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))
            out = self.dropout(out)
        out = self.layers[-1](out)
        return out


class BottomUpGNN(torch.nn.Module):
    def __init__(
        self,
        input_dims: dict,
        model_config: dict,
        shared_out_mlp: bool = False,
        normalize_mp: bool = False,
    ):
        """Bottom-up GNN implementation. For details, see Hilprecth and Binnig (2022).

        Args:
            input_dims (dict): Dictionary containing the encoding dimensions for each node type
            model_config (dict): Model-specific configs
            shared_out_mlp (bool, optional): Whether the MLP applied to the hidden embeddings is shared or node type-specific. Defaults to False.
            normalize_mp (bool, optional): Whether normalization is applied after message aggregation. Defaults to False.
            debug (bool, optional): If set, the per-node representations are also returned. Defaults to False.
        """
        super().__init__()

        # Naming of mlps:
        # - hidden_mlp: first mlp appliead to all original node encodings
        # - out_mlp: mlp applied after the message passing layer
        # - final_mlp: mlp to produce final estimate based on last rel node's embedding

        # Set root weight false as we manually sum after collecting information from all edge types
        self.conv = HeteroConv(
            {
                ("RelNode", "edge", "RelNode"): LinearAggregate(normalize_mp),
                ("RelNode", "edge", "OPNode"): LinearAggregate(normalize_mp),
                ("OPNode", "edge", "RelNode"): LinearAggregate(normalize_mp),
                ("OPNode", "edge", "OPNode"): LinearAggregate(normalize_mp),
                ("FieldNode", "edge", "OPNode"): LinearAggregate(normalize_mp),
                ("FieldNode", "edge", "RelNode"): LinearAggregate(normalize_mp),
                ("TableNode", "edge", "FieldNode"): LinearAggregate(normalize_mp),
                ("LiteralNode", "edge", "OPNode"): LinearAggregate(normalize_mp),
            },
            aggr="sum",
        )

        self.hidden_state_mlps = nn.ModuleDict(
            {
                "TableNode": MLP(input_dims["TableNode"], model_config["hidden_mlp"]),
                "FieldNode": MLP(input_dims["FieldNode"], model_config["hidden_mlp"]),
                "OPNode": MLP(input_dims["OPNode"], model_config["hidden_mlp"]),
                "RelNode": MLP(input_dims["RelNode"], model_config["hidden_mlp"]),
                "LiteralNode": MLP(input_dims["LiteralNode"], model_config["hidden_mlp"]),
            }
        )

        # We concatenate the current node's hidden and sum(children_out):
        out_input_dim = model_config["hidden_mlp"]["output_dim"] * 2
        # Also, the hidden embedding size must stay constant:
        model_config["out_mlp"]["output_dim"] = model_config["hidden_mlp"]["output_dim"]
        if shared_out_mlp:
            out_mlp = MLP(out_input_dim, model_config["out_mlp"])
            self.out_mlps = nn.ModuleDict(
                {
                    "TableNode": out_mlp,
                    "FieldNode": out_mlp,
                    "OPNode": out_mlp,
                    "RelNode": out_mlp,
                    "LiteralNode": out_mlp,
                }
            )
        else:
            self.out_mlps = nn.ModuleDict(
                {
                    "TableNode": MLP(out_input_dim, model_config["out_mlp"]),
                    "FieldNode": MLP(out_input_dim, model_config["out_mlp"]),
                    "OPNode": MLP(out_input_dim, model_config["out_mlp"]),
                    "RelNode": MLP(out_input_dim, model_config["out_mlp"]),
                    "LiteralNode": MLP(out_input_dim, model_config["out_mlp"]),
                }
            )

        # The input to the final MLP is the final embedding of the last rel node with dimension:
        classifier_input_dim = model_config["out_mlp"]["output_dim"]
        self.classifier = nn.Sequential(MLP(classifier_input_dim, model_config["final_mlp"]))
        self.mean_pool_rel_embeddings = model_config["mean_pool"]

    def forward(self, data: HeteroData):  # noqa: C901
        """Forward pass trough the bottom-up message passing GNN.

        Args:
            data (HeteroData): Batch of datapoints.

        Returns:
            Tensor: computed embedding tensor
        """

        max_depth = max(max(data[key].depth) for key in data.node_types)
        edge_subgraphs = [{k: [] for k in data.edge_types} for _ in range(max_depth)]
        active_nodes = [{k: set() for k in data.node_types} for _ in range(max_depth)]

        # Collect all edges that lead to a active node at that level
        for node_type in data.node_types:
            for node_id, d in enumerate(data[node_type].depth):
                for edge_type in data.edge_types:
                    if edge_type[2] == node_type:
                        in_edges = torch.where(data[edge_type].edge_index[1].detach().cpu() == node_id)[0]
                        if 0 < len(in_edges):
                            edge_subgraphs[max_depth - d][edge_type] += in_edges.tolist()
                            active_nodes[max_depth - d][node_type].add(node_id)

        # First we apply MLPs to create hidden states for each node
        for node_type in data.node_types:
            data[node_type].x = self.hidden_state_mlps[node_type](data[node_type].x)

        # Do message passing bottom up (starting with max depth)
        for depth, edge_subgraph in enumerate(edge_subgraphs):
            # NEW VERSION
            num_edges = 0
            for edge in edge_subgraph:
                num_edges += len(edge_subgraph[edge])
            if num_edges == 0:
                continue

            edge_indices_subgraph = {}
            for edge in edge_subgraph:
                edge_indices_subgraph[edge] = data.edge_index_dict[edge].T[edge_subgraph[edge]].T.reshape(2, -1)
            child_aggregates = self.conv(data.x_dict, edge_indices_subgraph)

            for node_type in data.node_types:
                # NOTE: we skip if there are no nodes of this type that receives a message
                cur_actives = list(active_nodes[depth][node_type])
                if len(cur_actives) == 0:
                    continue

                # Apply MLP on the hidden state (used if node has parents)
                # MLP should be applied to the concat of own embedding and
                # sum of child embeddings
                data[node_type].x[cur_actives] = self.out_mlps[node_type](
                    torch.concat(
                        (
                            child_aggregates[node_type][cur_actives],
                            data[node_type].x[cur_actives],
                        ),
                        dim=1,
                    )
                )
        # Last node is rel node with smallest depth (ie depth == 1)
        # AVG pooling over RelNodes
        if self.mean_pool_rel_embeddings:
            rel_nodes_per_graph = [list(range(i, j)) for i, j in zip(data["RelNode"].ptr[:-1], data["RelNode"].ptr[1:])]
            last_repr = torch.stack(
                [
                    torch.mean(
                        data["RelNode"].x[graph_ind] * (1 / data["RelNode"].depth[graph_ind].unsqueeze(1)),
                        dim=0,
                    )
                    for graph_ind in rel_nodes_per_graph
                ]
            )
            out = self.classifier(last_repr)
        else:
            outputNodes = torch.where(data["RelNode"].depth == 1)[0]
            out = self.classifier(data["RelNode"].x[outputNodes])
        return out
