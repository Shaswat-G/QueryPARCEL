import math
from abc import ABC, abstractmethod
from typing import Any

from flexdata_metric_prediction.tree.tree_nodes import TreeNode


class Encoder(ABC):
    """
    This class should be used to transform a tree of TreeNodes into a tree that also contains encodings for each node.
    The tree should still be represented by it's root, but all nodes should include an encoding field.
    For each node of the same class the size of the encoding must be equal.
    """

    @abstractmethod
    def encode_tree(self, root: TreeNode, plan: Any) -> None:
        """
        Receives a tree and includes an "encoding" field for each node.
        After having called encodeTree all nodes have the encoded field set to True.

        Args:
            - tree: Root node of the tree
            - plan: Plan containing all relations
        """
        pass

    def validate_encoding(self, root):
        def _validate(curNode, encodingSizes):
            nodeClassName = curNode.__class__.__name__
            if nodeClassName != "RootNode":
                assert curNode.encoded, f"Node {curNode.name} was reachable but not encoded."
                if encodingSizes[nodeClassName] == -1:
                    encodingSizes[nodeClassName] = len(curNode.encoding)
                else:
                    assert len(curNode.encoding) == encodingSizes[nodeClassName], (
                        f"Node {curNode.name} had an encoding length different to other nodes of same type"
                    )
                assert not any(math.isnan(x) for x in curNode.encoding), (
                    f"The encoding of {curNode.name} contain NaN values"
                )
            for child in curNode.children:
                _validate(child, encodingSizes)

        encodingSizes = {
            "RelNode": -1,
            "OPNode": -1,
            "FieldNode": -1,
            "TableNode": -1,
            "LiteralNode": -1,
        }
        _validate(root, encodingSizes)
        # logging.debug("Tree's encoding successfully validated.")

    def to_one_hot(self, size, hot):
        oneHot = [0 for _ in range(size)]
        oneHot[hot] = 1
        return oneHot

    def parse_op_mapping(self, op_mapping: dict):
        # Structure:
        # Each key refers to a high level group, only for naming:
        #  Each high level group contains a list of elements, which are distinct groups
        #  Ie we will have sum(len(group) for group in highLevelGroups) many categories
        #    Each element in a high level group may be:
        #     1) string -> matched based on name
        #     2) Dict of
        #       2.1) sub_group_name: list of OPs -> matched based on name
        #       2.2) sub_grou_name: {file_name: list of OPs} -> these OPs come from substrait def yamls

        mapping = {}
        enc_num = 0
        for high_level_cat in op_mapping:
            assert isinstance(op_mapping[high_level_cat], list)
            for enc_group in op_mapping[high_level_cat]:
                if isinstance(enc_group, str):
                    mapping[enc_group] = enc_num
                else:
                    for sub_group in enc_group:
                        if isinstance(enc_group[sub_group], list):
                            for sub_elem in enc_group[sub_group]:
                                mapping[sub_elem] = enc_num
                        else:
                            for yaml_def in enc_group[sub_group]:
                                if yaml_def not in mapping:
                                    mapping[yaml_def] = {}
                                mapping[yaml_def].update(dict.fromkeys(enc_group[sub_group][yaml_def], enc_num))

                enc_num += 1
        return mapping
