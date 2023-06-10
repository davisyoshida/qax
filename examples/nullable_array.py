from typing import Union

import jax
import jax.numpy as jnp

from qax import ImplicitArray, default_handler, primitive_handler, use_implicit_args

# WARNING: These constants are currently incomplete! They only include a few ops each
from qax.constants import ELEMENTWISE_BINOPS, ELEMENTWISE_UNOPS, REDUCTION_OPS

class NullableArray(ImplicitArray):
    def __init__(self, val, mask):
        super().__init__(shape=val.shape, dtype=val.dtype)
        self.val = val
        self.mask = mask

    def flatten(self):
        return [('val', self.val), ('mask', self.mask)], ()

    def unflatten(self, aux_data, children):
        self.val, self.mask = children

    def materialize(self):
        return self.val

@primitive_handler(ELEMENTWISE_UNOPS)
def handle_unop(primitive, nullable_val : NullableArray, **params):
    val = default_handler(primitive, nullable_val.val, **params)
    return NullableArray(val, nullable_val.mask)

@primitive_handler(ELEMENTWISE_BINOPS)
def handle_binop(primitive, lhs : Union[NullableArray, jnp.ndarray], rhs : Union[NullableArray, jnp.ndarray], **params):
    lhs_is_nullable = isinstance(lhs, NullableArray)
    rhs_is_nullable = isinstance(rhs, NullableArray)
    mask = lhs.mask if lhs_is_nullable else None

    if lhs_is_nullable:
        lhs = lhs.val

    if rhs_is_nullable:
        mask = rhs.mask if mask is None else mask & rhs.mask
        rhs = rhs.val

    out_val = default_handler(primitive, lhs, rhs, **params)
    return NullableArray(out_val, mask)

@primitive_handler(REDUCTION_OPS)
def handle_reduction(primitive, null_arr : NullableArray, **params):
    new_val = default_handler(primitive, null_arr.val, **params)
    new_mask = default_handler(jax.lax.reduce_and_p, null_arr.mask, **params)
    return NullableArray(new_val, new_mask)

@jax.jit
@use_implicit_args
def f(x, y):
    return jnp.sum(-x * y, axis=0)

if __name__ == '__main__':
    x = NullableArray(
        val=jnp.ones((2, 3)),
        mask=jnp.asarray(
            [[True, False, True],
             [False, True, True]]
        )
    )

    y = NullableArray(
        val=jnp.full((2, 3), 3),
        mask=jnp.asarray(
            [[False, True, True],
             [True, True, True]]
        )
    )

    output = f(x, y)
    print(f'Result: {output.val}')
    print(f'Mask: {output.mask}')
