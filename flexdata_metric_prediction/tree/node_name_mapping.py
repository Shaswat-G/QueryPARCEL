# TODO: the naming could come from a json as well to keep it static

# -------------------
# Read nodes -> Table and Field nodes
# TODO: consider other types of reads as well
READ_NODE = ["read"]


# -------------------
# Realational node
RELATIONAL_NODE = [
    "filter",
    "sort",
    "project",
    "cross",
    "join",
    # "JOIN_TYPE_UNSPECIFIED",
    # "JOIN_TYPE_INNER",
    # "JOIN_TYPE_OUTER",
    # "JOIN_TYPE_LEFT",
    # "JOIN_TYPE_RIGHT",
    "set",
    "fetch",
    "aggregate",
]

# -------------------
# Operator node <- group operators as they are grouped in substrait for now

OPERATOR_NODE = [
    "extract",
    "extract_boolean",
    "add",
    "multiply",
    "add_intervals",
    "subtract",
    "lte",
    "lt",
    "gte",
    "gt",
    "assume_timezone",
    "local_timestamp",
    "strptime_time",
    "strptime_date",
    "strptime_timestamp",
    "strftime",
    "round_temporal",
    "round_calendar",
    "min",
    "max",
    "not_equal",
    "equal",
    "is_not_distinct_from",
    "is_distinct_from",
    "between",
    "is_null",
    "is_not_null",
    "is_nan",
    "is_finite",
    "is_infinite",
    "nullif",
    "coalesce",
    "least",
    "least_skip_null",
    "greatest",
    "greatest_skip_null",
    "point",
    "make_line",
    "x_coordinate",
    "y_coordinate",
    "num_points",
    "is_empty",
    "is_closed",
    "is_simple",
    "is_ring",
    "geometry_type",
    "envelope",
    "dimension",
    "is_valid",
    "collection_extract",
    "flip_coordinates",
    "remove_repeated_points",
    "buffer",
    "centroid",
    "minimum_bounding_circle",
    "approx_count_distinct",
    "count",
    "ln",
    "log10",
    "log2",
    "logb",
    "log1p",
    "concat",
    "like",
    "substring",
    "regexp_match_substring",
    "regexp_match_substring_all",
    "starts_with",
    "ends_with",
    "contains",
    "strpos",
    "regexp_strpos",
    "count_substring",
    "regexp_count_substring",
    "replace",
    "concat_ws",
    "repeat",
    "reverse",
    "replace_slice",
    "lower",
    "upper",
    "swapcase",
    "capitalize",
    "title",
    "initcap",
    "char_length",
    "bit_length",
    "octet_length",
    "regexp_replace",
    "ltrim",
    "rtrim",
    "trim",
    "lpad",
    "rpad",
    "center",
    "left",
    "right",
    "string_split",
    "regexp_string_split",
    "string_agg",
    "divide",
    "negate",
    "modulus",
    "power",
    "sqrt",
    "exp",
    "cos",
    "sin",
    "tan",
    "cosh",
    "sinh",
    "tanh",
    "acos",
    "asin",
    "atan",
    "acosh",
    "asinh",
    "atanh",
    "atan2",
    "radians",
    "degrees",
    "abs",
    "sign",
    "factorial",
    "bitwise_not",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "sum",
    "sum0",
    "avg",
    "product",
    "std_dev",
    "variance",
    "corr",
    "mode",
    "median",
    "quantile",
    "row_number",
    "rank",
    "dense_rank",
    "percent_rank",
    "cume_dist",
    "ntile",
    "first_value",
    "last_value",
    "nth_value",
    "lead",
    "lag",
    "ceil",
    "floor",
    "round",
    "any_value",
    "or",
    "and",
    "and_not",
    "xor",
    "not",
    "bool_and",
    "bool_or",
    "index_in",
]


# To extract OPnode names
# FUNCTION_TYPES = [
#     "scalarFunction",
#     "aggregateFunction",
# ]  # TODO: consider other types as well

# # Read operator names from extensions
# path_to_function_extensions = (
#     "/Users/rau/MasterThesis/substrait-java/substrait/extensions"
# )
# opGroups = {}
# for file in os.listdir(path_to_function_extensions):
#     if "function" in file:
#         with open(os.path.join(path_to_function_extensions, file), "r") as f:
#             yaml_extensions = yaml.safe_load(f)
#         for k in yaml_extensions.keys():
#             if "function" in k:
#                 opGroupName = (
#                     file.replace(".yaml", "").replace("functions_", "") + "_" + k
#                 )
#                 ops = [ext["name"] for ext in yaml_extensions[k]]
#                 opGroups[opGroupName] = ops

# OPERATOR_CLASSES = list(opGroups.keys())
# OPERATOR_MAPPING = {}
# for k in OPERATOR_CLASSES:
#     OPERATOR_MAPPING.update({op: k for op in opGroups[k]})
# OPERATOR_NODE = list(
#     OPERATOR_MAPPING.keys()
# )
