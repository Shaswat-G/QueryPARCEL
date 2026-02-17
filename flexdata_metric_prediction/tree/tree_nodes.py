import logging
import math
from abc import ABC, abstractmethod
from typing import Any, List

from flexdata_metric_prediction.tree.node_name_mapping import OPERATOR_NODE, READ_NODE, RELATIONAL_NODE

# Distinguish between:
# - read node (transformed to table and field)
# - relational node
# - operator node
# - root node


# Abstract class for every node in the tree
class TreeNode(ABC):
    def __init__(self, name) -> None:
        self.name = name
        self.encoded = False
        # Depth of node in the tree. Used for BottomUp modeling
        self.depth = None  # i.e. uninitialized
        self.children = []
        self.hints = {}

    def add_single_child(self, child) -> None:
        if child is None:
            logging.warning(f"Node: {self.name} received None children.")
        elif child in self.children:
            # logging.warning(
            #     f"Node: {self.name} received the same child ({child}) twice"
            # )
            pass
        else:
            self.children.append(child)

    def add_children(self, children) -> None:
        if isinstance(children, list):
            for child in children:
                self.add_single_child(child)
        else:
            self.add_single_child(children)

    def _extract_hints(self, common):
        # TODO: the name mapping is artificial here
        hints = {}
        if hasattr(common, "hint") and 0 < len(common.hint.alias):
            hintNames = common.hint.alias.split(",")
            for i, hintName in enumerate(hintNames):
                hints[hintName] = common.hint.output_names[
                    i
                ]  # NOTE: the hints are kept as strings and left for the encoder to transform them to numeric form
        return hints

    @abstractmethod
    def __str__(self) -> str:
        pass


# -------------------
# Root node
class RootNode(TreeNode):
    def __init__(self) -> None:
        super().__init__("root")

    def __str__(self) -> str:
        return "Root: " + self.name


class TableNode(TreeNode):
    def __init__(self, curPlan) -> None:
        name = ".".join(curPlan.named_table.names)  # TODO: The source could not just be a table
        super().__init__(name)

        # NOTE: field and table hints are in the same Substrait rel node,
        # here we only keep those that don't belong to a specific field
        # (not in the format: fieldName:hintType)
        all_hints = self._extract_hints(curPlan.common)
        for k, v in all_hints.items():
            if ":" not in k:
                self.hints[k] = v

    def __str__(self) -> str:
        # TODO
        return "TableNode: " + str(self.name)


class FieldNode(TreeNode):
    def __init__(self, name, curPlan) -> None:
        super().__init__(name)
        self.numRows = None  # TODO: copy all attributes from paper to here
        # NOTE: we only keep field specific hints at this node
        all_hints = self._extract_hints(curPlan.common)
        for k, v in all_hints.items():
            if name.lower() in k.lower():
                # Trim the field name from the hint as well
                self.hints[k.split(":")[1]] = v

    def __str__(self) -> str:
        # TODO
        return "FieldNode: " + str(self.name)


# -------------------
class RelNode(TreeNode):
    def __init__(self, name) -> None:
        super().__init__(name)

    def init_from_plan(self, plan, fieldsAccessed):  # noqa: C901
        # Extract hints if exist
        self.hints = self._extract_hints(plan.common)
        # Create the corresponding OPnodes
        if self.name == "filter":
            opNode = create_op_node(plan.condition, fieldsAccessed)
            self.add_children(opNode)
        elif self.name == "fetch":
            pass  # no expression
        elif self.name == "aggregate":
            # Handle grouping expressions
            # NOTE: the output order is the order of groupings followed by the order of measures!
            fieldLen = len(fieldsAccessed)
            opsToAdd = []
            # How is it possible to have multiple groupings, each with multiple groupbys???
            for grouping in plan.groupings:
                # TODO: handle the pointers to grouping expressions (see docs)
                # This is now the deprecated version
                for group in grouping.grouping_expressions:
                    opNode = create_op_node(group, fieldsAccessed)
                    self.add_children(opNode)
                    opsToAdd.append(opNode)
            for measure in plan.measures:
                # Each measure msg has a measure (aggregate func) and an optional filter (Expr)
                opNode = create_op_node(measure.measure, fieldsAccessed)
                self.add_children(opNode)
                opsToAdd.append(opNode)
                if measure.HasField("filter"):
                    opNode = create_op_node(measure.filter, fieldsAccessed)
                    self.add_children(opNode)
            for op in opsToAdd:
                fieldsAccessed.append(op)
            del fieldsAccessed[:fieldLen]
        elif self.name == "sort":
            # List of fields to sort on is a lest of selects!
            for sort in plan.sorts:
                opNode = create_op_node(sort.expr, fieldsAccessed)
                self.add_children(opNode)
        elif self.name == "join":
            if plan.HasField("expression"):
                opNode = create_op_node(plan.expression, fieldsAccessed)
                self.add_children(opNode)
            if plan.HasField("post_join_filter"):
                logging.warning("Post-join filter encountered but it's not implemented!")
            # if plan.HasField("type"):
            self.join_type = plan.DESCRIPTOR.fields_by_name["type"].enum_type.values_by_number[plan.type].name
        elif self.name == "project":
            for expression in plan.expressions:
                opNode = create_op_node(expression, fieldsAccessed)
                self.add_children(opNode)
                # NOTE: Every expression in a project adds to the intermediate schema
                fieldsAccessed.append(opNode)
        elif self.name == "set":
            # TODO: what to do with the set operations?
            pass
        elif self.name == "cross":
            pass  # no expression
        elif self.name == "reference":
            pass  # no expression
        else:
            raise NotImplementedError("Unhandled RelNode type: " + self.name)

    def __str__(self) -> str:
        # TODO
        return "RelNode: " + str(self.name)


# -----------------
class OPNode(TreeNode):
    def __init__(self, name) -> None:
        super().__init__(name)
        self.numRows = None  # TODO: copy all attributes from paper to here

    def init_from_plan(self, plan, fieldsAccessed):
        # Store the function reference if present
        if "function_reference" in plan.DESCRIPTOR.fields_by_name.keys():
            self.function_reference = plan.function_reference
        elif self.name == "cast":
            pass
        elif self.name == "subquery":
            self.subquery_type = plan.WhichOneof("subquery_type")
        elif self.name in ["if_clause", "if_then"]:
            pass
        else:
            logging.warning(f"Unhandled OP node name encountered: {self.name}")
        # pass

    def __str__(self) -> str:
        # TODO
        return "OPNode: " + str(self.name)


class LiteralNode(TreeNode):
    _var_type_len = {
        "i8": 1,
        "boolean": 1,
        "i16": 2,
        "i32": 4,
        "date": 4,
        "fp32": 4,
        "null": 4,
        "i64": 8,
        "timestamp": 8,
        "time": 8,
        "fp64": 8,
        "intervalYearToMonth": 8,
        "decimal": 16,
        "intervalDayToSecond": 16,
    }

    _var_type_dynamic = {
        "string": lambda value: math.ceil(len(str(value).encode())),
        "fixed_char": lambda value: math.ceil(len(str(value).encode())),
        "binary": lambda value: math.ceil(len(value)),
        "fixed_binary": lambda value: math.ceil(len(value)),
        "var_char": lambda value: value.length,
    }

    def __init__(self, name) -> None:
        if name is None:
            # Handle enum arguments - name will be set later
            super().__init__("enum")
            return
        varType = name.WhichOneof("literal_type")
        super().__init__(str(getattr(name, varType)))

    def __str__(self) -> str:
        # TODO
        return "Literal: " + str(self.name)

    def init_from_plan(self, plan, fieldsAccessed):
        # Handle enum literals (created with name=None)
        if plan is None or self.name.startswith("enum_"):
            self.is_casted = False
            self.var_type = "string"  # Treat enums as string type for encoding
            self.length = 8  # Default length for enums
            return

        self.is_casted = "input" in plan.DESCRIPTOR.fields_by_name.keys()
        if self.is_casted:
            self.var_type = plan.input.literal.WhichOneof("literal_type")
            value = getattr(plan.input.literal, self.var_type)
        else:
            self.var_type = plan.WhichOneof("literal_type")
            value = getattr(plan, self.var_type)

        if self.var_type in self._var_type_len:
            self.length = self._var_type_len[self.var_type]
        elif self.var_type in self._var_type_dynamic:
            self.length = self._var_type_dynamic[self.var_type](value)
        else:
            logging.warning(f"Unknown datatype encountered{self.var_type}, using default value")
            self.length = 16


NODE_TYPES = [RelNode, OPNode, TableNode, FieldNode, LiteralNode]


# -------------
def create_node(
    name: str,
    curPlan: Any,
    fieldsAccessed: List[FieldNode],
) -> List[TreeNode] | TreeNode | None:
    """
    Create a node based on the name.
    If it's a read node it returns all accessed fields as field nodes. A corresponding TableNode is also created.

    Args:
        - name
        - curPlan
        - fieldsAccessedLocal
        - fieldsAccessedGlobal

    Returns:
        - Single TreeNode if the name corresponds to a relational or operator node
        - List of TreeNodes if the name corresponds to a read
        - None if the name is unrecognized
    """
    if name in RELATIONAL_NODE:
        return RelNode(name)
    elif name in OPERATOR_NODE:
        return create_op_node(getattr(curPlan, name), fieldsAccessed)
    elif name in READ_NODE:
        return create_read_node(getattr(curPlan, name), fieldsAccessed)
    else:
        raise NotImplementedError(f"Node type {name} is not mapped to any general type.")


def create_read_node(
    curPlan: Any,
    fieldsAccessed: List[FieldNode],
) -> List[TreeNode]:
    """
    Given the accessed field indices in the proto plan, if the fields are not contained in the global list
    it creates corresponding field nodes (as well as table node if the table was unseen too)
    and adjust the local list.

    Returns:
        - List of FieldNodes corresponding to the fields indexed by local indices in the proto plan.
    """
    # TODO: Is it possible that the same table is referenced (by name) by two distinct reads?
    # Parse read node (leaf of the tree)

    fields = curPlan.base_schema.names

    tableNode = TableNode(curPlan)
    opAccessedFields = []
    for field in fields:
        fnode = FieldNode(field, curPlan)
        fnode.add_children(tableNode)
        fieldsAccessed.append(fnode)
        opAccessedFields.append(fnode)
    return opAccessedFields


def create_op_node(curPlan, fieldsAccessed) -> OPNode | LiteralNode | None:  # noqa: C901
    """
    Create an OP node from an Expression proto message.

    Args:
        - curPlan: the expression message (one of it's field will point to the actual expression) or a measure message
        - fieldsAccessed: current intermediate schema

    Return:
        The OPNode (with possible other opNode descendants) created from the current message
    """

    # TODO: improve this
    expressionType = None
    if curPlan.DESCRIPTOR.name == "Expression":
        expressionType = curPlan.WhichOneof("rex_type")
        curPlan = getattr(curPlan, expressionType)  # type: ignore
    elif curPlan.DESCRIPTOR.name == "AggregateFunction":
        # HACK: as measures contains two expressions, we have to separately handle this
        # Here we will receive already the aggregate_function msg and not the expr msg containing it
        expressionType = "aggregate_function"
    elif curPlan.DESCRIPTOR.name == "IfClause":
        expressionType = "if_clause"
    else:
        NotImplementedError("Which OP is this??? " + curPlan.DESCRIPTOR.name)

    opNode = None

    if expressionType in ["scalar_function", "aggregate_function"]:
        # Functions may be nested, we create an OPNode for each expr node
        # A AggregateFunction will be parsed the same way as other functions
        opNode = OPNode(expressionType)
        opNode.init_from_plan(curPlan, fieldsAccessed)
        for argument in curPlan.arguments:
            if argument.HasField("value"):
                # Expression argument (most common case: columns, literals, nested functions)
                opChild = create_op_node(argument.value, fieldsAccessed)
                opNode.add_children(opChild)
            elif argument.HasField("enum"):
                # Enumeration argument (e.g., rounding mode for ROUND, trim direction for TRIM)
                # Represent as a literal node to capture the enum semantics in the tree
                enum_literal = LiteralNode(None)
                enum_literal.name = f"enum_{argument.enum}"
                enum_literal.init_from_plan(None, fieldsAccessed)  # Initialize with defaults
                opNode.add_children(enum_literal)
            elif argument.HasField("type"):
                # Type argument (e.g., target type for CAST operations)
                # Types don't contribute to the computational graph, so we skip them
                # They represent metadata about the operation rather than data flow
                pass
            else:
                # This should never happen if Substrait spec is followed
                logging.warning(
                    f"Unhandled function field encountered. Fields: {argument.DESCRIPTOR.fields_by_name.keys()}"
                )
    elif expressionType == "cast":
        # TODO: should we create an OPNode for each literal?
        opNode = OPNode(expressionType)
        opNode.init_from_plan(curPlan, fieldsAccessed)
        if curPlan.HasField("input"):
            opChild = create_op_node(curPlan.input, fieldsAccessed)
            opNode.add_children(opChild)
    elif expressionType == "literal":
        opNode = LiteralNode(curPlan)
    elif expressionType == "selection":
        # Direct read argument of the function, handle directly
        # Is always a leaf node
        # TODO: what if this isn't a direct reference?
        # TODO: the field access should be parsed better (handle diff. ways of access)
        # TODO: handle the case with 0 <- or is this transformed back by prot?
        fieldAccessID = curPlan.direct_reference.struct_field.field
        return fieldsAccessed[fieldAccessID]
    elif expressionType == "subquery":
        from flexdata_metric_prediction.tree.tree import (
            Tree,
        )  # This has to be imported here to avoid circular imports...

        # Here we have a subquery <- treat it as a new query without root node and parse it
        # we insert a subquery OP node anyway
        # The changes to the intermediate schema inside the subquery should not affect
        # the current intermediate schema
        fieldsAccessedSubquery = list(fieldsAccessed)
        subquery_type = curPlan.WhichOneof("subquery_type")
        subqueryPlan = getattr(curPlan, subquery_type)
        opNode = OPNode(expressionType)
        opNode.subquery_type = subquery_type  # Store subquery type for encoding

        if subquery_type == "scalar":
            # Here we have a single REL input (like the last aggregate of the subquery)
            relType = subqueryPlan.input.WhichOneof("rel_type")
            relNode = create_node(relType, subqueryPlan.input, fieldsAccessedSubquery)
            subTree = Tree()
            subTree.traverse_substrait_plan(relNode, subqueryPlan.input, fieldsAccessedSubquery)
            opNode.add_children(relNode)
        elif subquery_type == "in_predicate":
            # "needles" are a list of expressions
            for needle in subqueryPlan.needles:
                opNeedle = create_op_node(needle, fieldsAccessedSubquery)
                if opNeedle:
                    opNode.add_children(opNeedle)
            # Haystack (actual subquery) is then a rel
            relType = subqueryPlan.haystack.WhichOneof("rel_type")
            relNode = create_node(relType, subqueryPlan.haystack, fieldsAccessedSubquery)
            subTree = Tree()
            subTree.traverse_substrait_plan(relNode, subqueryPlan.haystack, fieldsAccessedSubquery)
            opNode.add_children(relNode)
        elif subquery_type == "set_predicate":
            relType = subqueryPlan.tuples.WhichOneof("rel_type")
            relNode = create_node(relType, subqueryPlan.tuples, fieldsAccessedSubquery)
            subTree = Tree()
            subTree.traverse_substrait_plan(relNode, subqueryPlan.tuples, fieldsAccessedSubquery)
            opNode.add_children(relNode)
        else:
            # This is a SetComparison
            # left is an expression
            leftOP = create_op_node(subqueryPlan.left, fieldsAccessedSubquery)
            if leftOP:
                opNode.add_children(leftOP)
            # right is a rel
            relType = subqueryPlan.right.WhichOneof("rel_type")
            relNode = create_node(relType, subqueryPlan.right, fieldsAccessedSubquery)
            subTree = Tree()
            subTree.traverse_substrait_plan(relNode, subqueryPlan.right, fieldsAccessedSubquery)
            opNode.add_children(relNode)
            logging.warning("Set comparison subquery encountered.")

    elif expressionType == "if_then":
        # NOTE: this encoding will loose information as branching cannot be encoded
        # We encode this as a v shape: parent:if_then, children:conditions
        opNode = OPNode(expressionType)
        # opNode.init_from_plan(curPlan, fieldsAccessed)
        for if_clasue in curPlan.ifs:
            if_op = create_op_node(if_clasue, fieldsAccessed)
            opNode.add_children(if_op)
        else_op = create_op_node(getattr(curPlan, "else"), fieldsAccessed)
        opNode.add_children(else_op)

    elif expressionType == "if_clause":
        opNode = OPNode(expressionType)
        if_clause = create_op_node(getattr(curPlan, "if"), fieldsAccessed)
        opNode.add_children(if_clause)
        then_clause = create_op_node(curPlan.then, fieldsAccessed)
        opNode.add_children(then_clause)

    else:
        logging.error(f"OP node encountered with unknown expression type: {expressionType}", exc_info=True)

    if opNode:
        opNode.init_from_plan(curPlan, fieldsAccessed)
    return opNode
