import time
from typing import Set, Tuple

import graphviz
from torch_geometric.data import HeteroData

from flexdata_metric_prediction.tree.tree_nodes import FieldNode, LiteralNode, OPNode, RelNode, RootNode, TableNode, TreeNode

nodeColor = {
    RootNode: "black",
    OPNode: "#6C8EBF",
    RelNode: "#B85450",
    TableNode: "#D6B656",
    FieldNode: "#D79B00",
    LiteralNode: "#82B366",
    "OPNode": "#6C8EBF",
    "RelNode": "#B85450",
    "TableNode": "#D6B656",
    "FieldNode": "#D79B00",
    "LiteralNode": "#82B366",
}

nodeFillColor = {
    RootNode: "black",
    OPNode: "#EEF3FC",
    RelNode: "#F8E1E0",
    TableNode: "#FFF2CC",
    FieldNode: "#FFE6CC",
    LiteralNode: "#D5E8D4",
    "OPNode": "#EEF3FC",
    "RelNode": "#F8E1E0",
    "TableNode": "#FFF2CC",
    "FieldNode": "#FFE6CC",
    "LiteralNode": "#D5E8D4",
}


def createGraphViz(  # noqa: C901
    root: TreeNode,
    graph: graphviz.Digraph,
    edges: Set[Tuple[str, str]],
    node_attr=None,
    flip: bool = False,
    extension_mapping=None,
):
    if node_attr is not None and hasattr(root, node_attr):
        if node_attr == "encoding":
            encodings = []
            for val in root.encoding:
                if isinstance(val, int):
                    encodings.append(val)
                else:
                    encodings.append(float(val))
            nodeName = root.name + "\n" + str(encodings)
        else:
            nodeName = root.name + "\n" + str(getattr(root, node_attr))
    else:
        if hasattr(root, "function_reference"):
            filename, funcname = extension_mapping[root.function_reference]
            nodeName = funcname
        else:
            if isinstance(root, LiteralNode):
                nodeName = "Literal:\n" + root.name
            elif isinstance(root, TableNode):
                nodeName = "Table: " + root.name.lower()
            elif isinstance(root, FieldNode):
                nodeName = "Field: " + root.name.lower()
            else:
                nodeName = root.name.title()

        # if root.name == "aggregate":
        #     nodeName = "<<B>h1</B>>"
        # nodeName = root.name

        if "precision" in nodeName or "scale" in nodeName:
            nodeName = "Decimal val"
    graph.node(
        str(id(root)),
        nodeName,
        color=nodeColor[root.__class__],
        fillcolor=nodeFillColor[root.__class__],
        style="filled",
    )
    for child in root.children:
        createGraphViz(
            child,
            graph,
            edges,
            node_attr=node_attr,
            flip=flip,
            extension_mapping=extension_mapping,
        )
        u, v = str(id(child)), str(id(root))
        if (u, v) not in edges:
            if flip:
                graph.edge(u, v)
            else:
                graph.edge(v, u)
            edges.add((u, v))


def _process_extensions(extensionURIs, extensions):
    """
    Returns:
        - Mapping from functionAnchor -> (FileName, FuncName)
    """
    extension_mapping = {}
    for extension in extensions:
        name = extension.extension_function.name.split(":")[0].strip()
        fileName = ""
        # We cannot assume any order among extensionURIs
        for extensionURI in extensionURIs:
            if extensionURI.extension_uri_anchor == extension.extension_function.extension_uri_reference:
                fileName = extensionURI.uri[1:]  # starts with /
                break
        extension_mapping[extension.extension_function.function_anchor] = (
            fileName,
            name,
        )
    return extension_mapping


def createAndSaveGraphViz(tree, plan, fname, dirname, node_attr=None, label="", flip=False):
    extension_mapping = _process_extensions(plan.extension_uris, plan.extensions)
    graph = graphviz.Digraph()
    createGraphViz(
        tree.root,
        graph,
        set(),
        node_attr=node_attr,
        flip=flip,
        extension_mapping=extension_mapping,
    )
    graph.attr(label=label, labelloc="t", fontsize="32")
    graph.render(fname, dirname, format="pdf")


def creteAndShowGraphViz(root, node_attr=None, label="", flip=False):
    graph = graphviz.Digraph()
    createGraphViz(root, graph, set(), node_attr=node_attr, flip=flip)
    graph.attr(label=label, labelloc="t", fontsize="14")
    graph.view(filename="graph_" + str(time.time()), cleanup=True)


def createAndShowGNNGraphViz(gnnData: HeteroData, node_attr="encoding", label="", flip=False):
    graph = graphviz.Digraph()
    for node_type in gnnData.node_types:
        node_cntr = 0
        for node in gnnData[node_type].x:
            nodeName = node_type + "_" + str(node_cntr)
            nodeID = nodeName
            if node_attr == "encoding":
                nodeName += "\n" + str(node.numpy().tolist())
            graph.node(nodeID, nodeName, color=nodeColor[node_type])
            node_cntr += 1

    for edge_type in gnnData.edge_types:
        src, trgt = str(edge_type[0]), str(edge_type[2])
        for edge in gnnData[edge_type].edge_index.T:
            graph.edge(src + "_" + str(edge[0].item()), trgt + "_" + str(edge[1].item()))
    graph.attr(label=label, labelloc="t", fontsize="14")
    graph.view(filename="graph_" + str(time.time()), cleanup=True)
