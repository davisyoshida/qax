import jax
import jax.numpy as jnp

from qax.implicit_array import ImplicitArray, implicit_op, use_implicit_args

class Outer(ImplicitArray):
    def __init__(self, x):
        super().__init__(x.shape, x.dtype)
        self.x = x

    def flatten(self):
        return [('x', self.x)], None

    def unflatten(self, aux_data, children):
        self.x, = children

    @classmethod
    def materialize(cls, outer):
        return 2 * (outer.x ** 1)

    @implicit_op('mul')
    def mul(primitive, arg, other):
        return Outer(arg.x * other)

class Inner(ImplicitArray):
    def __init__(self, value, shape, dtype):
        super().__init__(shape, dtype)
        self.value = value

    @classmethod
    def materialize(cls, inner):
        return jnp.full(inner.shape, inner.value, dtype=inner.dtype)

    @implicit_op('integer_pow')
    def pow(primitive, arg, params):
        new_value = arg.value ** params['y']
        return Inner(new_value, arg.shape, arg.dtype)

    def flatten(self):
        return [('value', self.value)], None

    def unflatten(self, aux_data, children):
        self.value, = children

def test_nested():
    @use_implicit_args
    def f(x):
        return jnp.sum(x)

    inner = Inner(3, shape=(2, 3), dtype=jnp.float32)
    nested = Outer(inner)
    result = f(nested)
    assert result == 36

def test_nested_with_operation():
    @use_implicit_args
    def f(x):
        return jnp.sum(x * jnp.ones(x.shape))

    inner = Inner(3, shape=(2, 3), dtype=jnp.float32)
    nested = Outer(inner)
    result = f(nested)
    assert result == 36
