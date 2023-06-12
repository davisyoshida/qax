import jax
import jax.numpy as jnp

from qax import use_implicit_args, ImplicitArray, primitive_handler

class TwoMatricesInATrenchcoat(ImplicitArray):
    def __init__(self, a, b):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        super().__init__(a.shape, a.dtype)

        self.a = a
        self.b = b

    def flatten(self):
        return [('a', self.a), ('b', self.b)], ()

    def unflatten(self, aux, children):
        self.a, self.b = children

    def materialize(self):
        return self.a + self.b


@primitive_handler('mul')
def handler(primitive, x : TwoMatricesInATrenchcoat, y : jax.Array):
    return TwoMatricesInATrenchcoat(x.a * y, x.b * y)

def test_grad():
    shape = (5, 7)
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    x = TwoMatricesInATrenchcoat(
        jax.random.normal(k1, shape),
        jax.random.normal(k2, shape),
    )
    y = jax.random.normal(k3, shape)

    @use_implicit_args
    def f(x, y):
        return jnp.sum(x * y)

    def explicit_f(a, b, y):
        return jnp.sum((a * y) + (b * y))

    x_grad = jax.grad(f)(x, y)
    y_grad = jax.grad(f, 1)(x, y)

    a_grad_expected = jax.grad(explicit_f)(x.a, x.b, y)
    b_grad_expected = jax.grad(explicit_f, 1)(x.b, x.a, y)
    y_grad_expected = jax.grad(explicit_f, 2)(x.a, x.b, y)

    assert jnp.allclose(x_grad.a, a_grad_expected)
    assert jnp.allclose(x_grad.b, b_grad_expected)
    assert jnp.allclose(y_grad, y_grad_expected)

