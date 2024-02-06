from functools import partial, wraps
from itertools import chain

import jax
from jax.api_util import flatten_fun_nokwargs
from jax import core
import jax.extend.linear_util as lu
from jax import tree_util

from . import implicit_array as ia

class _EmptyNodeCls:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

EmptyNode = _EmptyNodeCls()

tree_util.register_pytree_node(
    _EmptyNodeCls,
    lambda node: ((), None),
    lambda _, __: EmptyNode
)

def combine_leaf_predicate(base_fn, is_leaf):
    @wraps(base_fn)
    def new_fn(*args, new_is_leaf=None):
        if new_is_leaf is None:
            combined_is_leaf = is_leaf
        else:
            def combined_is_leaf(arg):
                return is_leaf(arg) or new_is_leaf(arg)
        return base_fn(*args, is_leaf=combined_is_leaf)
    return new_fn

def leaf_predicate(x):
    return isinstance(x, (ia.ImplicitArray, _EmptyNodeCls))

tree_map_with_implicit = combine_leaf_predicate(jax.tree_map, leaf_predicate)
tree_map_with_path_with_implicit = combine_leaf_predicate(tree_util.tree_map_with_path, leaf_predicate)
tree_flatten_with_implicit = combine_leaf_predicate(tree_util.tree_flatten, leaf_predicate)
tree_flatten_with_path_with_implicit = combine_leaf_predicate(tree_util.tree_flatten_with_path, leaf_predicate)
tree_leaves_with_implicit = combine_leaf_predicate(tree_util.tree_leaves, leaf_predicate)
tree_structure_with_implicit = combine_leaf_predicate(tree_util.tree_structure, leaf_predicate)

def flatten_one_implicit_layer(tree):
    def is_leaf_below_node(node, x):
        return isinstance(x, ia.ImplicitArray) and x is not node

    def replace_subtree_implicits(node):
        return tree_util.tree_map(lambda _: 1, node, is_leaf=partial(is_leaf_below_node, node))

    prototype = tree_map_with_implicit(replace_subtree_implicits, tree)
    struct = tree_util.tree_structure(prototype)

    leaves = tree_leaves_with_implicit(tree)
    leaves = list(chain.from_iterable(
        tree_util.tree_leaves(leaf, is_leaf=partial(is_leaf_below_node, leaf))
        if isinstance(leaf, ia.ImplicitArray) else
        [leaf] for leaf in leaves
    ))
    return leaves, struct

def implicit_depth(tree):
    leaves = tree_leaves_with_implicit(tree)
    depth = 0
    while True:
        next_leaves = []
        any_implicit = False
        for leaf in leaves:
            if not isinstance(leaf, ia.ImplicitArray):
                continue
            any_implicit = True
            next_leaves.extend(flatten_one_implicit_layer(leaf)[0])

        if not any_implicit:
            return depth

        depth += 1
        leaves = next_leaves

def _map_leaves_with_implicit_path(f, leaves, is_leaf, path_prefix=()):
    mapped_leaves = []
    for idx, leaf in enumerate(leaves):
        path = path_prefix + (idx,)
        if not isinstance(leaf, ia.ImplicitArray) or is_leaf(path, leaf):
            mapped_leaves.append(f(leaf))
            continue

        subtree, substruct = flatten_one_implicit_layer(leaf)
        mapped_subtree = _map_leaves_with_implicit_path(
            f,
            subtree,
            is_leaf=is_leaf,
            path_prefix=path
        )
        mapped_leaves.append(tree_util.tree_unflatten(substruct, mapped_subtree))
    return mapped_leaves

def _get_pruning_transform(tree, materialization_paths):
    if not materialization_paths:
        return lambda x: x
    def is_leaf(path, leaf):
        return path in materialization_paths

    def materialize_subtrees(tree):
        leaves, struct = tree_flatten_with_implicit(tree)

        mapped_leaves =  _map_leaves_with_implicit_path(partial(materialize_nested, full=True), leaves, is_leaf)
        return tree_util.tree_unflatten(struct, mapped_leaves)

    return materialize_subtrees

def get_common_prefix_transforms(trees):
    """
    Given an iterable of pytrees which have the same structure after all
    ImplicitArray instances are materialized, return a list of callables
    which will transform each tree into the largest common structure
    obtainable via materialization of ImplicitArrays.
    """
    if len(trees) <= 1:
        return [lambda x: x for _ in trees]

    all_leaves, structures = zip(*(tree_flatten_with_implicit(tree) for tree in trees))
    post_materialization_avals = [core.get_aval(leaf) for leaf in all_leaves[0]]
    for i, (leaves, structure) in enumerate(zip(all_leaves[1:], structures[1:]), 1):
        if structure != structures[0]:
            raise ValueError('Trees do not have the same structure after materialization')

        for leaf, expected_aval in zip(leaves, post_materialization_avals):
            aval = core.get_aval(leaf)
            if not (aval.shape == expected_aval.shape and aval.dtype == expected_aval.dtype):
                raise ValueError(
                    f'Trees do not have the same avals after materialization. Tree 0: {expected_aval}, Tree {i}: {aval}'
                )

    # Stack will contain tuples of (path, nodes)
    # path = a sequence of integers specifying which child
    # was taken at each _flatten_one_implicit_layer call
    # or the first flatten_with_implicit call
    # nodes = one node from each tree
    stack = []

    all_leaves = []
    for tree in trees:
        all_leaves.append(tree_leaves_with_implicit(tree))

    for i, nodes in enumerate(zip(*all_leaves)):
        stack.append(((i,), nodes))

    materialization_paths = set()
    while stack:
        path_prefix, nodes = stack.pop()
        if not any(isinstance(node, ia.ImplicitArray) for node in nodes):
               continue

        all_leaves, all_structures = zip(*(
            flatten_one_implicit_layer(node) for node in nodes
        ))
        node_structures = set(all_structures)
        if len(node_structures) > 1:
            materialization_paths.add(path_prefix)
            continue

        aval_diff = False
        for leaves in zip(*all_leaves):
            first_aval = core.get_aval(leaves[0])
            shape = first_aval.shape
            dtype = first_aval.dtype
            for leaf in leaves[1:]:
                aval = core.get_aval(leaf)
                if not (aval.shape == shape and aval.dtype == dtype):
                    materialization_paths.add(path_prefix)
                    aval_diff = True
            if aval_diff:
                break

        if aval_diff:
            continue

        for i, leaf_group in enumerate(zip(*all_leaves)):
            stack.append((path_prefix + (i,), leaf_group))

    return [_get_pruning_transform(tree, materialization_paths) for tree in trees]

def materialize_nested(implicit_arr, full=False):
    """
    Materialize an ImplicitArray instance, handling the case where implicit_arr.materialize()
    involves further ImplicitArray instances.
    Arguments:
        implicit_arr: An ImplicitArray instance
        full: If True, repeatedly materialize until the result is a concrete array
    Returns:
        The materialized array
    """
    while isinstance(implicit_arr, ia.ImplicitArray):
        wrapped = lu.wrap_init(type(implicit_arr).materialize)
        flat, in_tree = flatten_one_implicit_layer((implicit_arr,))
        flat_fn, out_tree = flatten_fun_nokwargs(wrapped, in_tree)
        out_flat = ia.use_implicit_args(flat_fn.call_wrapped)(*flat)
        implicit_arr = jax.tree_util.tree_unflatten(out_tree(), out_flat)

        if not full:
            break

    return implicit_arr

