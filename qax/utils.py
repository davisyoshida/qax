from functools import wraps

import jax

from . import implicit_array

def vmap_all_but_one(f, axis, out_ndim=0):
    """
    Repeatedly calls vmap to map over all axes except for `axis.`
    All args will be mapped on the same dimensions.
    """
    @wraps(f)
    def inner(*args):
        n_dim = args[0].ndim
        if axis >= n_dim:
            raise ValueError(f'Axis {axis} is out of bounds for array of dimension {n_dim}')
        fn = f
        vmap_dim = 1
        out_dim = out_ndim
        for i in reversed(range(n_dim)):
            if i == axis:
                vmap_dim = 0
                out_dim = 0
            else:
                fn = jax.vmap(fn, vmap_dim, out_dim)
        return fn(*args)
    return inner

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

leaf_predicate = lambda x: isinstance(x, implicit_array.ImplicitArray)
tree_map_with_implicit = combine_leaf_predicate(jax.tree_map, leaf_predicate)
tree_flatten_with_implicit = combine_leaf_predicate(jax.tree_util.tree_flatten, leaf_predicate)
tree_leaves_with_implicit = combine_leaf_predicate(jax.tree_util.tree_leaves, leaf_predicate)
