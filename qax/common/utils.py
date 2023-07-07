from functools import wraps

import jax
import jax.numpy as jnp
from jax import tree_util
from jax.dtypes import float0
import optax

from ..implicit.implicit_array import use_implicit_args
from ..symbols import SymbolicConstant

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

def freeze_subtrees(optimizer : optax.GradientTransformation, label_fn, use_scalar_zeros=False):
    """
    Utility which wraps an optimizer such that subtrees specified by
    label_fn will receive zeros as updates.
    Subtrees to be frozen should be labeled with "freeze"
    and all other subtrees should be labeled with "train"
    """
    multi_transformed_optimizer = optax.multi_transform(
        {
            'freeze': set_to_zero_scalar() if use_scalar_zeros else optax.set_to_zero(),
            'train': optimizer
        },
        label_fn
    )

    def new_update(grads, opt_state, params):
        def map_float0(param, grad):
            if grad.dtype == float0:
                return jnp.zeros((), param.dtype) if use_scalar_zeros else jnp.zeros_like(param)
            return grad

        fixed_grads = jax.tree_map(map_float0, params, grads)
        return multi_transformed_optimizer.update(fixed_grads, opt_state, params)

    return optax.GradientTransformation(
        multi_transformed_optimizer.init,
        new_update
    )

def freeze_keys(optimizer : optax.GradientTransformation, arr_type, keys, use_scalar_zeros=False) -> optax.GradientTransformation:
    keys = set(keys)
    def label_leaf(leaf):
        if not isinstance(leaf, arr_type):
            return 'train'

        children, aux_data = leaf.tree_flatten_with_keys()
        labels = ['freeze' if key in keys else 'train' for key, _ in children]
        struct = leaf.tree_unflatten(aux_data, labels)
        return struct

    def label_fn(root):
        return jax.tree_map(label_leaf, root, is_leaf=lambda x: isinstance(x, arr_type))

    return freeze_subtrees(optimizer, label_fn, use_scalar_zeros=use_scalar_zeros)

def apply_updates(params : optax.Params, updates : optax.Updates) -> optax.Params:
    """
    Like optax.apply_updates, but updates can be SymbolicConstant instances
    """
    updates_flat, update_struct = tree_util.tree_flatten(updates, is_leaf=lambda x: isinstance(x, SymbolicConstant))
    semi_flat_params = update_struct.flatten_up_to(params)

    updated_flat = use_implicit_args(optax.apply_updates)(semi_flat_params, updates_flat)
    updated = update_struct.unflatten(updated_flat)
    return updated

def set_to_zero_scalar() -> optax.GradientTransformation:
    """
    Returns a gradient transformation that sets all gradients to 0 in order to
    make downstream constant folding cheaper.
    """
    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        return jax.tree_map(lambda x: jnp.zeros((), x.dtype), updates), state

    return optax.GradientTransformation(init_fn, update_fn)
