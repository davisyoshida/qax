# WARNING: This file is obviously super incomplete, and is
# currently just for convenience in testing.

COMMUTATIVE_OPS = frozenset(
    [
        "add",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "eq",
        "max",
        "min",
        "mul",
        "ne",
    ]
)

ELEMENTWISE_UNOPS = frozenset(
    [
        "abs",
        "acos",
        "acosh",
        "asin",
        "asinh",
        "atan",
        "atanh",
        "bessel_i0e",
        "bessel_i1e",
        "cbrt",
        "ceil",
        "clz",
        "conj",
        "convert_element_type",
        "copy",
        "cos",
        "cosh",
        "digamma",
        "erf_inv",
        "erf",
        "erfc",
        "exp",
        "expm1",
        "floor",
        "imag",
        "integer_pow",
        "is_finite",
        "lgamma",
        "log1p",
        "log",
        "logistic",
        "neg",
        "not",
        "population_count",
        "real",
        "reduce_precision",
        "round",
        "rsqrt",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    ]
)

ELEMENTWISE_BINOPS = frozenset(
    [
        "add",
        "and",
        "atan2",
        "complex",
        "div",
        "eq",
        "ge",
        "gt",
        "igamma_grad_a",
        "igamma",
        "igammac",
        "le",
        "lt",
        "max",
        "min",
        "mul",
        "ne",
        "nextafter",
        "or",
        "pow",
        "random_gamma_grad",
        "rem",
        "shift_left",
        "shift_right_arithmetic",
        "shift_right_logical",
        "sub",
        "xor",
    ]
)

REDUCTION_OPS = frozenset(
    [
        "argmax",
        "argmin",
        "reduce_and",
        "reduce_max",
        "reduce_min",
        "reduce_or",
        "reduce_prod",
        "reduce_sum",
        "reduce_xor",
    ]
)

CUMULATIVE_REDUCTION_OPS = frozenset(
    [
        "cumlogsumexp",
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
    ]
)
