import json
import logging
import os
from typing import Any

import sql_metadata
from encoder.encoder import Encoder
from tree.treeNodes import FieldNode


def createIMDBMappingFile(schemaPath, outPath):
    with open(schemaPath) as f:
        # sql = sqlparse.parse(f)
        creates = f.read()

    tableMapping = {}
    tableCnt = 0
    fieldsMapping = {}
    creates = creates.split(";")

    for create in creates:
        create = create.strip()
        parsed = sql_metadata.Parser(create)
        if parsed.tables:
            table_name = parsed.tables[0]
            fields = parsed.columns
            if table_name == "link_type":
                fields.append("link")  # BUG: this field does not get parsed by sql_metadata
            tableMapping[table_name] = tableCnt
            tableCnt += 1
            fieldsMapping[table_name] = {f: i for i, f in enumerate(fields)}

    with open(os.path.join(outPath, "tableMapping.json"), "w") as f:
        json.dump(tableMapping, f, indent=4)

    with open(os.path.join(outPath, "fieldMapping.json"), "w") as f:
        json.dump(fieldsMapping, f, indent=4)


# createIMDBMappingFile("./test/resources/imdb/schema.sql", "./test/resources/imdb/")
def getIMDBEncoder(plan, query_statistics):
    with open("./encoder/opMapping.json") as f:
        OPmapping = json.load(f)
    with open("./encoder/relMapping.json") as f:
        RelMapping = json.load(f)
    with open("./data/imdb/tableMapping.json") as f:
        TableMapping = json.load(f)
    with open("./data/imdb/fieldMapping.json") as f:
        FieldMapping = json.load(f)

    return IMDBEncoder(
        plan.extension_uris, plan.extensions, OPmapping, RelMapping, TableMapping, FieldMapping, query_statistics
    )


class IMDBEncoder(Encoder):
    def __init__(
        self, extensionURIs, extensions, OPMapping, RelMapping, TableMapping, FieldMapping, query_statistics
    ) -> None:
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

        # Mapping of table nodes
        self.TableMapping = TableMapping
        self.tableEncLength = len(TableMapping.keys())

        # Mapping of field nodes
        self.FieldMapping = FieldMapping
        self.fieldEncLength = 0
        for k in FieldMapping:
            self.fieldEncLength = max(self.fieldEncLength, len(FieldMapping[k].keys()))

        # Query statistics provided by brad paper
        self.query_statistics = query_statistics

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
        from tree.treeNodes import OPNode, RelNode, TableNode

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
        node.encoding = self.toOneHot(self.relEncLength, self.RelMapping[node.name])

    def _encodeOPNode(self, node):
        if hasattr(node, "function_reference"):
            filename, funcname = self.extension_mapping[node.function_reference]
            hot_bit = self.OPMapping[filename][funcname]
            node.encoding = self.toOneHot(self.opEncLength, hot_bit)
        else:
            node.encoding = self.toOneHot(self.opEncLength, -1)
        node.encoded = True

    def _encodeFieldNode(self, node):
        assert len(node.children) == 1, "Field associated with multiple nodes"
        table_name = node.children[0].name.lower()
        node.encoded = True
        node.encoding = self.toOneHot(self.fieldEncLength, self.FieldMapping[table_name][node.name.lower()])

    def _encodeTableNode(self, node):
        node.encoded = True
        node.encoding = self.toOneHot(self.tableEncLength, self.TableMapping[node.name.lower()])
        if node.name.lower() in self.query_statistics["selectivity"]:
            node.encoding.extend(
                [
                    self.query_statistics["selectivity"][node.name.lower()],
                    self.query_statistics["access_width"][node.name.lower()],
                    self.query_statistics["est_cardinality"][node.name.lower()],
                    self.query_statistics["access_width_pct"][node.name.lower()],
                ]
            )
        else:
            node.encoding.extend([0, 0, 0, 0])
            logging.warning(f"{node.name} was missing from statistics")
