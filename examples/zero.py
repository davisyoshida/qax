from typing import Union
import jax
import jax.numpy as jnp

from qax import ImplicitArray, primitive_handler, use_implicit_args

class ImplicitZeros(ImplicitArray):
    def materialize(self):
        return jnp.zeros(self.shape, dtype=self.dtype)

def get_binop_out_aval(x, y):
    shape = jnp.broadcast_shapes(x.shape, y.shape)
    dtype = jnp.result_type(x.dtype, y.dtype)
    return shape, dtype

@primitive_handler(jax.lax.mul_p)
def do_mul(primitive, x : ImplicitZeros, y : jax.Array):
    print('Invoked do_mul')
    return ImplicitZeros(*get_binop_out_aval(x, y))

@primitive_handler([jax.lax.add_p, jax.lax.mul_p])
def handle_both_implicit(primitive, x : ImplicitZeros, y : ImplicitZeros):
    print('Invoked handle_both_implicit')
    return ImplicitZeros(*get_binop_out_aval(x, y))

@primitive_handler(jax.lax.add_p)
def handle_add_general(primitive, x : ImplicitZeros, y : jax.Array):
    print('Invoked handle_add')
    out_shape, out_dtype = get_binop_out_aval(x, y)
    return jnp.broadcast_to(y, out_shape).astype(out_dtype)

def main():
    @jax.jit
    @use_implicit_args
    def f(x, y):
                                 # If x and y are both of type ImplicitZeros, the result will be:

        z = x + y                # z: ImplicitZeros  output: Invoked handle_both_implicit
        w = z * jnp.ones_like(z) # w: ImplicitZeros  output: Invoked do_mul
        a = jnp.sum(w)           # a: jax.Array      output: UserWarning: Primitive reduce_sum was not
                                 #                           handled by class ImplicitZeros, so implicit
                                 #                           args will be materialized
        b =  w + a               # b: jax.Array      output: Invoked handle_add
        return b

    zeros = ImplicitZeros((2, 3), dtype=jnp.float32)

    result = jax.jit(f)(zeros, zeros)

    assert isinstance(result, jax.Array)
    assert result.shape == zeros.shape
    assert jnp.all(result == 0)

    # The decorated f will also work with mixed arguments or non-implicit arguments
    jnp_ones = jnp.ones(zeros.shape)
    f(zeros, jnp_ones)
    f(jnp_ones, zeros)
    f(jnp_ones, jnp_ones)



if __name__ == '__main__':
    main()
