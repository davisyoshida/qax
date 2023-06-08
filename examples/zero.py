import jax
import jax.numpy as jnp

from qax.implicit_array import ImplicitArray, implicit_op, use_implicit_args

class ImplicitZeros(ImplicitArray):
    def materialize(self):
        return jnp.zeros(self.shape, dtype=self.dtype)

    @implicit_op('mul')
    def mul(primitive, *args):
        result = jax.eval_shape(
            (lambda x, y: x * y),
            *(x.aval for x in args)
        )
        return ImplicitZeros(result.shape, result.dtype)

def main():
    @use_implicit_args
    def f(x, y):
        return jnp.sum(x * y)

    zeros = ImplicitZeros((2, 3), dtype=jnp.float32)

    result = jax.jit(f)(zeros, jnp.ones(zeros.shape))
    # UserWarning: Primitive reduce_sum was not handled by class ImplicitZeros, so implicit args will be materialized

    assert result == 0

if __name__ == '__main__':
    main()
