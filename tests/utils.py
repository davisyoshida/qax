import jax
import jax.numpy as jnp
from jax.tree_util import tree_structure
import pytest

from qax import utils, ImplicitArray

class Container(ImplicitArray):
    def __init__(self, a, b):
        super().__init__(a.shape, a.dtype)
        self.a = a
        self.b = b

    def flatten(self):
        return [('a', self.a), ('b', self.b)], ()

    def unflatten(self, aux_data, children):
        self.a, self.b = children

    def materialize(self):
        return self.a

    def __str__(self):
        return f'Container(a={self.a}, b={self.b})'

    def __repr__(self):
        return str(self)

@pytest.fixture(scope='module', params=[0, 1, 2, 3])
def container_with_depth(request):
    a = jnp.arange(10)
    for d in range(request.param):
        a = Container(a, jnp.zeros(d))

    return a, request.param

def test_count_depth(container_with_depth):
    container, depth = container_with_depth
    assert utils.implicit_depth(container) == depth

def test_flatten_one_layer(container_with_depth):
    container, depth = container_with_depth
    pytree = [{'x': container}, {'y': container}]
    flat, struct = utils.flatten_one_implicit_layer(pytree)

    unflattened = jax.tree_util.tree_unflatten(struct, flat)
    assert jax.tree_util.tree_structure(unflattened) == jax.tree_util.tree_structure(pytree)
    assert utils.implicit_depth(flat) == max(depth - 1, 0)

def _get_prefix(*containers):
    return [transform(c) for c, transform in zip(containers, utils.get_common_prefix_transforms(containers))]

def test_prefix():
    c1 = Container(
        a=Container(jnp.zeros(10), jnp.zeros(10)),
        b=Container(jnp.zeros(3), jnp.zeros(13))
    )
    c2 = Container(
        a=Container(jnp.zeros(10), jnp.zeros(10)),
        b=jnp.zeros(3)
    )

    full_materialized_c1, _ = _get_prefix(c1, jnp.ones(10))
    assert isinstance(full_materialized_c1, jnp.ndarray)
    assert jnp.all(full_materialized_c1 == jnp.zeros(10))

    c3 = Container(
        a=Container(jnp.zeros(10), jnp.zeros(3)),
        b=Container(jnp.zeros(3), jnp.zeros(13))
    )

    prefix_c1, prefix_c3 = _get_prefix(c1, c3)
    expected = Container(a=jnp.zeros(10), b=Container(jnp.zeros(3), jnp.zeros(13)))
    assert tree_structure(prefix_c1) == tree_structure(prefix_c3) == tree_structure(expected)

    c4 = Container(
        a=Container(
            a=Container(jnp.ones(10), jnp.zeros(3)),
            b=jnp.zeros(3)
        ),
        b=jnp.zeros(10)
    )

    c5 = Container(
        a=jnp.zeros(10),
        b=Container(
            Container(jnp.zeros(10), jnp.zeros(3)),
            Container(jnp.zeros(3), jnp.zeros(13))
        )
    )

    prefix_c4, prefix_c5 = _get_prefix(c4, c5)
    expected = Container(a=jnp.zeros(10), b=jnp.zeros(10))
    assert tree_structure(prefix_c4) == tree_structure(prefix_c5) == tree_structure(expected)
