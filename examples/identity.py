from dataclasses import dataclass, InitVar
from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
import qax

from examples.minimal_lora import LoraMatrix

@dataclass
class Eye(qax.ImplicitArray):
    dim : InitVar[int]

    def __init__(self, dim, dtype=jnp.float32):
        super().__init__(shape=(dim, dim), dtype=dtype)

    def materialize(self):
        return jnp.eye(self.shape[0], dtype=self.dtype)

@qax.primitive_handler(jax.lax.dot_general_p)
def dot_handler(primitive, lhs : Union[Eye,jax.Array], rhs : Union[Eye,jax.Array], *, dimension_numbers, **kwargs):
    lhs_aval = jax.core.get_aval(lhs)
    rhs_aval = jax.core.get_aval(rhs)

    out_aval = jax.eval_shape(
        partial(qax.default_handler, primitive, dimension_numbers=dimension_numbers, **kwargs),
        lhs_aval, rhs_aval
    )

    lhs_is_eye = isinstance(lhs, Eye)
    rhs_is_eye = isinstance(rhs, Eye)

    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers
    # It's 1 AM and I can only conceptualize dot_generals during the PM hours
    # so I will only be implementing 1-2D x 2D matmuls
    if not (
        1 <= lhs.aval.ndim <= 2
        and rhs.aval.ndim <= 2
        and len(lhs_contract) == len(rhs_contract) == 1
        and lhs_batch == rhs_batch == ()
    ):
        return NotImplemented

    if lhs_is_eye and rhs_is_eye:
        return Eye(out_aval.shape[0], dtype=out_aval.dtype)

    result = rhs if lhs_is_eye else lhs
    return result.astype(out_aval.dtype)

def main():
    @qax.use_implicit_args
    def f(a, b):
        return a @ b

    w = Eye(3)
    x = jnp.arange(39, dtype=jnp.float32).reshape(3, 13)


    print(f(w, x))

    dim = 128
    rank = 16
    eye_plus_low_rank = LoraMatrix(
        W=Eye(dim),
        A=jax.random.normal(jax.random.PRNGKey(0), (dim, rank)),
        B=jnp.zeros((dim, rank))
    )

    x = jax.random.normal(jax.random.PRNGKey(1), (73, dim))
    print(jnp.sum(f(x, eye_plus_low_rank)))

if __name__ == '__main__':
    main()
