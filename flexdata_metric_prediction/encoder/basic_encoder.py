import json
import os
from typing import Any

import yaml
from encoder.encoder import Encoder


def createOPMappingFile(expressionDefPath, outPath):
    # Create OP to "class" mapping based on the extension files
    mapping = {}
    opGroup = 0
    for file in os.listdir(expressionDefPath):
        if "function" in file:
            with open(os.path.join(expressionDefPath, file)) as f:
                yaml_extensions = yaml.safe_load(f)
            for k in yaml_extensions.keys():
                if "function" in k:
                    ops = [ext["name"] for ext in yaml_extensions[k]]
                    if file not in mapping:
                        mapping[file] = {}
                    for op in ops:
                        mapping[file][op] = opGroup
                    opGroup += 1
    with open(os.path.join(outPath, "opMapping.json"), "w") as f:
        json.dump(mapping, f, indent=4)


def createRelMappingFile(outPath):
    # Create the RelMapping based on the fixed list of handled relations
    from tree.nodeNameMapping import RELATIONAL_NODE

    relCnt = 0
    mapping = {}
    for relNode in RELATIONAL_NODE:
        mapping[relNode] = relCnt
        relCnt += 1
    with open(os.path.join(outPath, "relMapping.json"), "w") as f:
        json.dump(mapping, f, indent=4)


# createOPMappingFile(
#     "/Users/rau/MasterThesis/substrait-java/substrait/extensions",
#     "/Users/rau/MasterThesis/SubstraitForML/encoder",
# )

# createRelMappingFile("/Users/rau/MasterThesis/SubstraitForML/encoder")


def getBasicEncoder(plan, **kwargs):
    with open("./encoder/opMapping.json") as f:
        OPmapping = json.load(f)
    with open("./encoder/relMapping.json") as f:
        RelMapping = json.load(f)
    return BasicEncoder(plan.extension_uris, plan.extensions, OPmapping, RelMapping)


class BasicEncoder(Encoder):
    def __init__(self, extensionURIs, extensions, OPMapping, RelMapping) -> None:
        """
        Args:
            - functional_references:
            - OPMapping: a dictionary containing the mapping from file->operator->idx (see createOPMappingFile())
        """
        # Map extensionFuncAnchors to (filename, funcname)
        self.extension_mapping = self._process_extensions(extensionURIs, extensions)

        # Mapping of OP nodes
        self.OPMapping = OPMapping
        self.opEncLength = 0
        for k in OPMapping:
            for kk in OPMapping[k]:
                self.opEncLength = max(self.opEncLength, OPMapping[k][kk])
        self.opEncLength += 2  # TODO: how to handle non-func OPs? (also accommodate for last OP)

        # Mapping of rel nodes
        self.RelMapping = RelMapping
        self.relEncLength = 0
        for k in RelMapping:
            self.relEncLength = max(self.relEncLength, RelMapping[k])
        self.relEncLength += 1  # Accommodate for last rel

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
        self.validateEncoding(tree)

    def _encode(self, root, plan):
        from tree.treeNodes import FieldNode, OPNode, RelNode, TableNode

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

    def _encodeRelNode(self, node):
        node.encoded = True
        node.encoding = [0 for _ in range(self.relEncLength)]
        node.encoding[self.RelMapping[node.name]] = 1

    def _encodeOPNode(self, node):
        node.encoding = [0 for _ in range(self.opEncLength)]
        if hasattr(node, "function_reference"):
            filename, funcname = self.extension_mapping[node.function_reference]
            hot_bit = self.OPMapping[filename][funcname]
            node.encoding[hot_bit] = 1
        else:
            node.encoding[-1] = 1
        node.encoded = True

    def _encodeFieldNode(self, node):
        node.encoded = True
        node.encoding = [0]

    def _encodeTableNode(self, node):
        node.encoded = True
        node.encoding = [0]
