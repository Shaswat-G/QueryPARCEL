import logging
from typing import Any, List, Union

import torch
from torch_geometric.data import HeteroData

from flexdata_metric_prediction.tree.tree_nodes import (
    FieldNode,
    LiteralNode,
    OPNode,
    RelNode,
    RootNode,
    TableNode,
    TreeNode,
    create_node,
)


class Tree:
    def __init__(self, dtype: torch.dtype = torch.float64) -> None:
        self.root: TreeNode | None = None
        self.gnnData: HeteroData | None = None
        self.gnnDtype = dtype
        # This contains a GNNData node type -> treeNode ->GNNData node id mapping for convenience
        self.treeToGNNmap: dict | None = None

    def create_tree(self, plan: Any, root: TreeNode | None = None):
        self.root = root or RootNode()
        self.traverse_substrait_plan(self.root, plan.relations[0], [])

    def to_gnn_data(self, label: float | None = None):  # noqa: C901
        """Compute a HeteroData object for the tree.

        Args:
            label (float | None, optional): Optional label attached to the hetero data object as field "y". Defaults to None.

        Raises:
            AttributeError: If invoked before tree is computed.
            RuntimeError: If no relation is contained in the tree.
        """
        if self.root is None:
            raise AttributeError("Tree must be first created before transforming it to GNN data.")
        # Graph to output
        self.gnnData = HeteroData()
        # Node types considered
        nodeTypes = [TableNode, FieldNode, RelNode, OPNode, LiteralNode]

        # This will contain node_class_name -> node -> ID
        self.treeToGNNmap = {}
        curID = {}
        for nodeType in nodeTypes:
            self.treeToGNNmap[nodeType.__name__] = {}
            curID[nodeType.__name__] = 0

        # First assign per node type IDs and keep the mapping
        curNodes = [self.root]
        while 0 < len(curNodes):
            newCurNodes = []
            for node in curNodes:
                nodeClass = type(node).__name__
                if not isinstance(node, RootNode) and node not in self.treeToGNNmap[nodeClass]:
                    self.treeToGNNmap[nodeClass][node] = curID[nodeClass]
                    curID[nodeClass] += 1
                for child in node.children:
                    newCurNodes.append(child)
            curNodes = newCurNodes

        # NOTE: Literal nodes may not exist in the tree
        # We handle that here by removing it from the nodeTypes
        if not self.treeToGNNmap[RelNode.__name__]:
            raise RuntimeError("At least one RelNode must be contained in the graph")

        for nodeType in nodeTypes[:]:
            if not self.treeToGNNmap[nodeType.__name__]:
                logging.debug(f"Removing {nodeType.__name__} from the considered node types for GNN data")
                nodeTypes.remove(nodeType)

        # Node encoding can now be initialized
        # NOTE: we have to ensure that the position corresponds to the nodeID!
        for nodeType in nodeTypes:
            self.gnnData[nodeType.__name__].x = [None for _ in range(len(self.treeToGNNmap[nodeType.__name__]))]
            for node, pos in self.treeToGNNmap[nodeType.__name__].items():
                self.gnnData[nodeType.__name__].x[pos] = node.encoding
            self.gnnData[nodeType.__name__].x = torch.tensor(self.gnnData[nodeType.__name__].x, dtype=self.gnnDtype)
            # torch.tensor([node.encoding for node in self.treeToGNNmap[nodeType.__name__]], dtype=torch.float32)

        # We attach the depth map to the HeteroData object as the depth attribute
        # NOTE: the order matters (e.g. first entry corresponds to firs node)
        self.compute_depth()
        for nodeType in nodeTypes:
            self.gnnData[nodeType.__name__].depth = [None for _ in range(len(self.treeToGNNmap[nodeType.__name__]))]
            for node, pos in self.treeToGNNmap[nodeType.__name__].items():
                self.gnnData[nodeType.__name__].depth[pos] = node.depth
            self.gnnData[nodeType.__name__].depth = torch.tensor(
                self.gnnData[nodeType.__name__].depth, dtype=torch.int8
            )
            assert 0 < len(self.gnnData[nodeType.__name__].depth), "Failed to create depth map"

        # Add edges using the mapping created
        visited = set()
        curNodes = [self.root]
        while 0 < len(curNodes):
            newCurNodes = []
            for node in curNodes:
                if node not in visited:
                    visited.add(node)
                    v_type = type(node).__name__
                    for child in node.children:
                        if not isinstance(node, RootNode):
                            u_type = type(child).__name__
                            if not hasattr(self.gnnData[u_type, "edge", v_type], "edge_index"):
                                self.gnnData[u_type, "edge", v_type].edge_index = [
                                    [],
                                    [],
                                ]
                            self.gnnData[u_type, "edge", v_type].edge_index[0].append(self.treeToGNNmap[u_type][child])
                            self.gnnData[u_type, "edge", v_type].edge_index[1].append(self.treeToGNNmap[v_type][node])
                        newCurNodes.append(child)
            curNodes = newCurNodes

        for edgeType in self.gnnData.edge_types:
            self.gnnData[edgeType].edge_index = torch.tensor(self.gnnData[edgeType].edge_index, dtype=torch.int64)

        if label is not None:
            self.gnnData.y = torch.tensor(label, dtype=self.gnnDtype)

    def compute_depth(self):
        if self.root is None:
            raise AttributeError("Tree must be first created before computing its depth.")
        if self.root.depth is not None:
            return

        curDepth = 0
        curNodes = [self.root]
        while 0 < len(curNodes):
            nextCurNodes = []
            for node in curNodes:
                node.depth = max(node.depth, curDepth) if node.depth else curDepth
                for child in node.children:
                    nextCurNodes.append(child)
            curDepth += 1
            curNodes = nextCurNodes

    def get_depth_map(self):
        if self.root is None:
            raise AttributeError("Tree must be first created before computing depth map.")
        if self.gnnData is None or self.treeToGNNmap is None:
            raise AttributeError("GNN representation must be first computed before computing depth map.")
        if self.root.depth is None:
            self.compute_depth()
        if self.gnnData is None:
            self.to_gnn_data()  # TODO: this dependency is unnecessary
        # Contains nodeType->NodeID->Depth mapping
        mapping = {}
        for nodeType in self.gnnData.node_types:
            mapping[nodeType] = {}

        for nodeType in self.treeToGNNmap:
            for node in self.treeToGNNmap[nodeType]:
                nodeID = self.treeToGNNmap[nodeType][node]
                mapping[nodeType][nodeID] = node.depth

        return mapping

    def get_gnn_data(self, label=None):
        if self.gnnData is None:
            self.to_gnn_data(label)
        return self.gnnData

    def is_tree_created(self) -> bool:
        return self.root is not None

    def is_gnn_data_create(self) -> bool:
        return self.gnnData is not None

    def degree(self) -> Union[None, int]:
        if self.root is None:
            raise AttributeError("Tree must be first created before its degree is computed.")
        nodes = set()
        queue = [self.root]
        while 0 < len(queue):
            new_queue = []
            for node in queue:
                nodes.add(node)
                for child in node.children:
                    new_queue.append(child)
            queue = new_queue
        return len(nodes)

    def traverse_substrait_plan(  # noqa: C901
        self,
        curNode: TreeNode,
        curPlan: Any,
        fieldsAccessed: List[FieldNode],
    ) -> None:
        """
        Traverse the DAG consisting of the relational nodes.
        Expressions are parsed separately after calling TreeNode.init_from_plan().
        Args:
            - curNode: the node representing the root of the current subtree
            - curPlan: Substrait plan corresponding to the current subtree
            - fieldsAccessed: contains the current intermediate schema. Indirect references are handled by this list.
        Returns:
            None
        """
        # curPlan's root is either a RelRoot or a Rel msg
        if curPlan.DESCRIPTOR.name == "RelRoot":
            nodeName = "RelRoot"
        else:
            nodeName = curPlan.WhichOneof("rel_type")
            curPlan = getattr(curPlan, nodeName)

        # Traverse child relations recursively
        if "input" in curPlan.DESCRIPTOR.fields_by_name.keys():
            # Unary relation
            newNode = create_node(curPlan.input.WhichOneof("rel_type"), curPlan.input, fieldsAccessed)
            curNode.add_children(newNode)
            if isinstance(newNode, RelNode):
                self.traverse_substrait_plan(newNode, curPlan.input, fieldsAccessed)
        elif "left" in curPlan.DESCRIPTOR.fields_by_name.keys() and "right" in curPlan.DESCRIPTOR.fields_by_name.keys():
            # Binary relation
            # Note: must start with left to keep correct intermediate schema
            newNodeLeft = create_node(curPlan.left.WhichOneof("rel_type"), curPlan.left, fieldsAccessed)
            curNode.add_children(newNodeLeft)
            if isinstance(newNodeLeft, RelNode):
                # assert len(fieldsAccessed) == 0, print(fieldsAccessed)
                self.traverse_substrait_plan(newNodeLeft, curPlan.left, fieldsAccessed)

            newNodeRight = create_node(curPlan.right.WhichOneof("rel_type"), curPlan.right, fieldsAccessed)
            curNode.add_children(newNodeRight)
            fields_accessed_right = []
            if isinstance(newNodeRight, RelNode):
                self.traverse_substrait_plan(newNodeRight, curPlan.right, fields_accessed_right)
            for f in fields_accessed_right:
                fieldsAccessed.append(f)
        else:
            # Leaf node - TODO: check for other options for child
            # TODO: SetRel has inputs
            logging.warning(f"Potentially unhandled input fields of a rel node: {curNode.name}")

        # After processing child relations, we can init current level
        if isinstance(curNode, RelNode):
            curNode.init_from_plan(curPlan, fieldsAccessed)
        elif not isinstance(curNode, RootNode):
            raise TypeError(f"Non-rel node encountered inside traversing rel nodes: {curNode.name}")

        # We also need to adjust the current intermediate schema
        if not isinstance(curNode, RootNode) and curPlan.common.WhichOneof("emit_kind") == "emit":
            output_mapping = curPlan.common.emit.output_mapping
            curLen = len(fieldsAccessed)
            for i in output_mapping:
                fieldsAccessed.append(fieldsAccessed[i])
            # HACK: but must be in-place!
            del fieldsAccessed[:curLen]
