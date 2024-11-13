from typing import Union

import jax
import jax.numpy as jnp

from examples.const import ImplicitConst
from examples.zero import ImplicitZeros
from qax import primitive_handler, use_implicit_args


@use_implicit_args
def f(x, y):
    return x * y


def main():
    shape = (2, 3)
    zeros = ImplicitZeros(shape=shape, dtype=jnp.float32)
    const = ImplicitConst(1.0, shape=shape)

    assert isinstance(f(const, zeros), jax.Array)

    @primitive_handler("mul")
    def heterogenous_handler(
        primitive,
        x: Union[ImplicitZeros, ImplicitConst],
        y: Union[ImplicitZeros, ImplicitConst],
    ):
        out_shape = jnp.broadcast_shapes(x.shape, y.shape)
        out_dtype = jnp.result_type(x.dtype, y.dtype)
        return ImplicitZeros(shape=out_shape, dtype=out_dtype)

    assert isinstance(f(const, zeros), ImplicitZeros)


if __name__ == "__main__":
    main()
