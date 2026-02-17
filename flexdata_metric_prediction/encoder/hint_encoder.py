import logging
from typing import Any

import torch

from flexdata_metric_prediction.encoder.encoder import Encoder
from flexdata_metric_prediction.tree.tree_nodes import RootNode, TreeNode
from flexdata_metric_prediction.utils.read_config import read_json


class HintEncoder(Encoder):
    def __init__(self, op_mapping: str | dict, rel_mapping: str | dict, type_mapping: str | dict) -> None:
        self.op_mapping = op_mapping
        if isinstance(self.op_mapping, str):
            self.op_mapping = read_json(op_mapping)

        self.op_mapping = self.parse_op_mapping(self.op_mapping)
        self.op_enc_length = 0
        for op_or_file in self.op_mapping:
            if isinstance(self.op_mapping[op_or_file], int):
                self.op_enc_length = max(self.op_enc_length, self.op_mapping[op_or_file])
            else:
                for elem in self.op_mapping[op_or_file]:
                    self.op_enc_length = max(self.op_enc_length, self.op_mapping[op_or_file][elem])
        self.op_enc_length += 1

        # Mapping of rel nodes
        self.rel_mapping = rel_mapping
        if isinstance(self.rel_mapping, str):
            self.rel_mapping = read_json(rel_mapping)
        self.rel_enc_length = 0
        for k in self.rel_mapping:
            self.rel_enc_length = max(self.rel_enc_length, self.rel_mapping[k])
        self.rel_enc_length += 1  # Accommodate for last rel

        # mapping for types
        if isinstance(type_mapping, str):
            self.type_mapping = read_json(type_mapping)
        else:
            self.type_mapping = type_mapping

        self.type_enc_length = 0
        for k in self.type_mapping:
            self.type_enc_length = max(self.type_enc_length, self.type_mapping[k])
        self.type_enc_length += 1  # Accommodate for last rel

    def _process_extensions(self, extensionURIs, extensions):
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
                    uri = extensionURI.uri
                    # Handle both URI formats:
                    # Old format: "/functions_comparison.yaml"
                    # New format: "extension:io.substrait:functions_comparison"
                    if uri.startswith("/"):
                        # Old format: remove leading slash
                        fileName = uri[1:]
                    elif ":" in uri:
                        # New format: extract the last part and add .yaml extension
                        # "extension:io.substrait:functions_comparison" -> "functions_comparison.yaml"
                        fileName = uri.split(":")[-1] + ".yaml"
                    else:
                        # Fallback: use as-is
                        fileName = uri
                    break
            extension_mapping[extension.extension_function.function_anchor] = (
                fileName,
                name,
            )
        return extension_mapping

    def encode_tree(self, root: RootNode, plan: Any) -> None:
        # Map extensionFuncAnchors to (filename, funcname)
        extension_mapping = self._process_extensions(plan.extension_uris, plan.extensions)

        self._encode(root, plan, extension_mapping)
        self.validate_encoding(root)

    def _encode(self, root: TreeNode, plan: Any, extension_mapping: dict):
        from flexdata_metric_prediction.tree.tree_nodes import FieldNode, LiteralNode, OPNode, RelNode, TableNode

        for child in root.children:
            if not child.encoded:
                self._encode(child, plan, extension_mapping)
        if isinstance(root, RelNode):
            self._encode_rel_node(root)
        elif isinstance(root, OPNode):
            self.encode_op_node(root, extension_mapping)
        elif isinstance(root, FieldNode):
            self._encode_field_node(root)
        elif isinstance(root, TableNode):
            self._encode_table_node(root)
        elif isinstance(root, LiteralNode):
            self._encode_literal_node(root)
        elif isinstance(root, RootNode):
            if len(root.children) != 1:
                raise AttributeError("Root node must have a single child.")
            self._encode(root.children[0], plan, extension_mapping)
        else:
            raise AttributeError("Unknown node type: " + str(root))

    def _encode_rel_node(self, node):
        node.encoded = True
        if node.name != "join":
            hotIdx = self.rel_mapping[node.name]
        elif hasattr(node, "join_type") and node.join_type in self.rel_mapping:
            hotIdx = self.rel_mapping[node.join_type]
        else:
            logging.warning(f"Unexpected rel node type {node.name}, encoding as unspecified")
            hotIdx = self.rel_mapping["JOIN_TYPE_UNSPECIFIED"]
        node.encoding = self.to_one_hot(self.rel_enc_length, hotIdx)
        # Ensure consistent encoding length by using default hints if missing
        # Expected hints: avgSize, rowCount
        if "avgSize" not in node.hints:
            node.hints["avgSize"] = "0"
        if "rowCount" not in node.hints:
            node.hints["rowCount"] = "0"
        node.encoding += [torch.log1p(torch.tensor(float(v), dtype=torch.float64)) for v in node.hints.values()]
        # node.encoding += [float(v) for v in node.hints.values()]

    def encode_op_node(self, node, extension_mapping):
        if hasattr(node, "function_reference"):
            filename, funcname = extension_mapping[node.function_reference]
            hot_bit = self.op_mapping[filename][funcname]
            node.encoding = self.to_one_hot(self.op_enc_length, hot_bit)
        elif hasattr(node, "subquery_type"):
            hot_bit = self.op_mapping[node.subquery_type]
            node.encoding = self.to_one_hot(self.op_enc_length, hot_bit)
        else:
            hot_bit = self.op_mapping[node.name]
            node.encoding = self.to_one_hot(self.op_enc_length, hot_bit)
        node.encoded = True

    def _encode_field_node(self, node):
        def type_conversion(val: str):
            val = val.lower()
            if "(" in val:
                val = val.split("(")[0]
            if "not null" in val:
                val = val.split("not null")[0]
            return self.type_mapping.get(val.strip(), 0)  # Default to 0 for unknown types

        assert len(node.children) == 1, "Field associated with multiple nodes"
        node.encoded = True

        # Ensure consistent encoding length with default hints for subquery fields
        # Expected hints: colType, numNulls, numDVs, avgColLen, maxColLen
        if "colType" not in node.hints:
            node.hints["colType"] = "bigint"  # Default type
        if "numNulls" not in node.hints:
            node.hints["numNulls"] = "0"
        if "numDVs" not in node.hints:
            node.hints["numDVs"] = "0"
        if "avgColLen" not in node.hints:
            node.hints["avgColLen"] = "8"
        if "maxColLen" not in node.hints:
            node.hints["maxColLen"] = "8"

        node.encoding = self.to_one_hot(self.type_enc_length, type_conversion(node.hints["colType"]))
        node.encoding += [
            torch.log1p(torch.tensor(float(v), dtype=torch.float64)) for k, v in node.hints.items() if k != "colType"
        ]

    def _encode_table_node(self, node):
        node.encoded = True
        # Ensure consistent encoding length with default hints for subquery tables
        # Expected hints: avgSize, rowCount
        if "avgSize" not in node.hints:
            node.hints["avgSize"] = "0"
        if "rowCount" not in node.hints:
            node.hints["rowCount"] = "0"
        node.encoding = [torch.log1p(torch.tensor(float(v), dtype=torch.float64)) for v in node.hints.values()]

    def _encode_literal_node(self, node):
        node.encoded = True
        node.encoding = self.to_one_hot(self.type_enc_length, self.type_mapping[node.var_type])
        node.encoding += [
            float(node.length),
            float(node.is_casted),
        ]
