import jax
import jax.numpy as jnp

from qax import use_implicit_args
from examples.nullable_array import NullableArray
from examples.zero import ImplicitZeros

@jax.jit
@use_implicit_args
def f(x, y):
    return (x * y) + 3

shape = (2, 3)
x = NullableArray(
    val=ImplicitZeros(shape, jnp.float32),
    mask=jnp.asarray(
        [[True, False, True],
         [False, True, True]],
    )
)

y = NullableArray(
    val=jnp.ones(shape),
    mask=jnp.asarray(
        [[True, True, True],
         [False, False, True]]
    )
)

result = f(x, y)
print(f'Result:\n{result.val}')
print(f'Mask:\n{result.mask}')
