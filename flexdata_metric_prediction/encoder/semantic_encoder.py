import json
import os
from typing import Any

import torch

# from sentence_transformers import SentenceTransformer
from flexdata_metric_prediction.encoder.encoder import Encoder
from flexdata_metric_prediction.tree.tree_nodes import TreeNode
from flexdata_metric_prediction.utils.read_config import read_yaml_config

model_path = "ibm-granite/granite-embedding-107m-multilingual"
# encode_model = SentenceTransformer(model_path)

embedding_cache = {}

function_yamls = {}
extension_defs_path = "/Users/rau/MasterThesis/substrait-java/substrait/extensions"
for file in os.listdir(extension_defs_path):
    function_yamls[file] = read_yaml_config(os.path.join(extension_defs_path, file))


def getSemanticEncoder(plan):
    with open("./encoder/relMapping.json") as f:
        RelMapping = json.load(f)
    with open("./encoder/typeMapping.json") as f:
        TypeMapping = json.load(f)

    path_extension_defs = "/Users/rau/MasterThesis/substrait-java/substrait/extensions"

    return SemanticEncoder(
        plan.extension_uris,
        plan.extensions,
        RelMapping,
        TypeMapping,
        path_extension_defs,
        None,
    )


class SemanticEncoder(Encoder):
    def __init__(
        self,
        extensionURIs,
        extensions,
        RelMapping,
        TypeMapping,
        extensionDefsDir,
        model,
    ) -> None:
        """
        Args:
            - functional_references:
            - OPMapping: a dictionary containing the mapping from file->operator->idx (see createOPMappingFile())
        """
        # Map extensionFuncAnchors to (filename, funcname)
        self.extension_mapping = self._process_extensions(extensionURIs, extensions)

        # Mapping of OP nodes
        self.extensionDefsDir = extensionDefsDir

        # Load the Sentence Transformer model
        self.model = model
        self.encoding_cache = {}

        # Mapping of rel nodes
        self.RelMapping = RelMapping
        self.relEncLength = 0
        for k in RelMapping:
            self.relEncLength = max(self.relEncLength, RelMapping[k])
        self.relEncLength += 1  # Accommodate for last rel

        # mapping for types
        self.TypeMapping = TypeMapping
        self.typeEncLength = 0
        for k in TypeMapping:
            self.typeEncLength = max(self.typeEncLength, TypeMapping[k])
        self.typeEncLength += 1  # Accommodate for last rel
        raise NotImplementedError("Encoder deprecated!")

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
                    fileName = extensionURI.uri[1:]  # starts with /
                    break
            extension_mapping[extension.extension_function.function_anchor] = (
                fileName,
                name,
            )
        return extension_mapping

    def encodeTree(self, tree, plan: Any) -> None:
        self._encode(tree, plan)
        self.validate_encoding(tree)

    def _encode(self, root, plan):
        from flexdata_metric_prediction.tree.tree import FieldNode, LiteralNode, OPNode, RelNode, TableNode

        for child in root.children:
            if not child.encoded:
                self._encode(child, plan)
        if isinstance(root, RelNode):
            self._encodeRelNode(root)
        elif isinstance(root, OPNode):
            self._encodeOPNode(root)
        elif isinstance(root, FieldNode):
            self._encodeFieldNode(root)
        elif isinstance(root, TableNode):
            self._encodeTableNode(root)
        elif isinstance(root, LiteralNode):
            self._encodeLiteralNode(root)

    def _encodeRelNode(self, node):
        node.encoded = True
        if node.name != "join":
            hotIdx = self.RelMapping[node.name]
        elif hasattr(node, "join_type") and node.join_type in self.RelMapping:
            hotIdx = self.RelMapping[node.join_type]
        else:
            hotIdx = self.RelMapping["JOIN_TYPE_UNSPECIFIED"]
        node.encoding = self.to_one_hot(self.relEncLength, hotIdx)
        node.encoding += [torch.log1p(torch.tensor(float(v), dtype=torch.float64)) for v in node.hints.values()]
        # node.encoding += [float(v) for v in node.hints.values()]

    def get_function_description(self, filename, funcname) -> dict:
        # extension = read_yaml_config(os.path.join(self.extensionDefsDir, filename))
        extension = function_yamls[filename]
        for superclass in extension:
            for function in extension[superclass]:
                if function["name"] == funcname:
                    return function
        raise AttributeError("Function description not found")

    def _encodeOPNode(self, node):
        if hasattr(node, "function_reference"):
            filename, funcname = self.extension_mapping[node.function_reference]
            func_description = self.get_function_description(filename, funcname)

            if "description" in func_description:
                if func_description["description"] not in embedding_cache:
                    encoding = self.model.encode(func_description["description"])
                    embedding_cache[func_description["description"]] = encoding
                else:
                    # logging.info("OP embedding already cached")
                    encoding = embedding_cache[func_description["description"]]
            else:
                encoding = self.model.encode(f"This is a function called {func_description['name']} with arguments ")
            node.encoding = encoding

        elif node.name == "cast":
            encoding = self.model.encode("This node casts the field")
            node.encoding = encoding
            # node.encoding = self.toOneHot(self.opEncLength, self.opEncLength - 1)
        elif node.name == "if_clause":
            node.encoding = self.model.encode("This operation represents an if clause")
        elif node.name == "if_then":
            node.encoding = self.model.encode("This operation represents a if else condition")
        elif node.name == "subquery":
            node.encoding = self.model.encode("This operation represents a subquery")
        else:
            raise RuntimeError("OP node not encoded")
            node.encoding = self.toOneHot(self.opEncLength, -1)
        node.encoded = True

    def _encodeFieldNode(self, node):
        def type_conversion(val):
            val = val.lower()
            if "(" in val:
                val = val.split("(")[0]
            return self.TypeMapping[val]

        assert len(node.children) == 1, "Field associated with multiple nodes"
        node.encoded = True
        node.encoding = self.to_one_hot(self.typeEncLength, type_conversion(node.hints["colType"]))
        node.encoding += [
            torch.log1p(torch.tensor(float(v), dtype=torch.float64)) for k, v in node.hints.items() if k != "colType"
        ]

    def _encodeTableNode(self, node):
        node.encoded = True
        node.encoding = [torch.log1p(torch.tensor(float(v), dtype=torch.float64)) for v in node.hints.values()]

    def _encodeLiteralNode(self, node):
        node.encoded = True
        node.encoding = self.to_one_hot(self.typeEncLength, self.TypeMapping[node.var_type])
        node.encoding += [
            float(node.length),
            float(node.is_casted),
        ]

    def encode_tree(self, root: TreeNode, plan: Any) -> None:
        raise NotImplementedError
