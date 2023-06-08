# WARNING: This file is obviously super incomplete, and is
# currently just for convenience in testing.

COMMUTATIVE_OPS = frozenset([
    'add',
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    'eq',
    'max',
    'min',
    'mul',
    'ne',
])

ELEMENTWISE_UNOPS = frozenset([
    'abs',
    'neg',
    'convert_element_type'
])

ELEMENTWISE_BINOPS = frozenset([
    'abs',
    'add',
    'sub',
    'mul',
])

REDUCTION_OPS = frozenset([
    'reduce_sum',
    'reduce_max'
])
