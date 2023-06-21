from functools import partial
from typing import Union
import warnings

import jax
from jax.core import Primitive
from jax.experimental import pjit
import jax.numpy as jnp
import numpy as np
import pytest

from qax import ImplicitArray, use_implicit_args, primitive_handler

WARN_PATTERN = '.*implicit args will be materialized'

class ImplicitConst(ImplicitArray):
    def __init__(self, value, dummy_val, shape, dtype=jnp.float32):
        super().__init__(shape=shape, dtype=dtype)
        self.value = value
        self.dummy_val = dummy_val

    def materialize(self):
        return jnp.full(self.shape, self.value, dtype=self.dtype)

    def flatten(self):
        return [('value', self.value), ('dummy_val', self.dummy_val)], ()

    def unflatten(self, aux_data, children):
        self.value, self.dummy_val = children

    def __str__(self):
        return f'ImplicitConst(value={self.value}, shape={self.shape})'

    def __repr__(self):
        return str(self)

@primitive_handler([jax.lax.mul_p, jax.lax.sub_p])
def mul_handler(primitive : Primitive, a : ImplicitConst, b : Union[ImplicitConst, jax.Array], **params):
    def op(lhs, rhs):
        return lhs * rhs if primitive.name == 'mul' else lhs - rhs
    assert not params
    if isinstance(b, ImplicitConst):
        return ImplicitConst(op(a.value, b.value), a.dummy_val, a.shape, a.dtype)
    if b.shape == ():
        new_value = op(a.value, b)
        return ImplicitConst(new_value, a.dummy_val, a.shape, a.dtype)
    return op(a.value, b)

@pytest.fixture
def const():
    shape = (2, 3)
    return ImplicitConst(2, -173, shape, dtype=jnp.float32)

def test_transform(const):
    @use_implicit_args
    def f(x, y):
        return x * y

    assert f(const, jnp.ones(const.shape))[0, 0] == const.value

def test_pjit(const):
    @use_implicit_args
    @pjit.pjit
    def f(x, y):
        return x * y

    assert f(const, jnp.ones(const.shape))[0, 0] == const.value

def test_remat(const):
    @use_implicit_args
    @jax.checkpoint
    def f(x, y):
        return x * y

    result = f(const, jnp.ones(const.shape))
    assert result.shape == const.shape
    assert result[0, 0] == const.value

def test_materialize(const):
    def f(x):
        return 3 + x
    with pytest.warns(UserWarning, match=WARN_PATTERN):
        use_implicit_args(f)(const)


def test_cond(const):
    @use_implicit_args
    def f(x, y):
        def true_fn(x):
            return x * jnp.ones(x.shape)
        def false_fn(x):
            return x * jnp.zeros(x.shape) + 5
        return jnp.sum(jax.lax.cond(y, true_fn, false_fn, x))

    with warnings.catch_warnings():
        warnings.filterwarnings('error', message=WARN_PATTERN)
        assert f(const, True) == const.value * np.prod(const.shape)
        assert f(const, False) == 5 * np.prod(const.shape)

def test_cond_materialize_branch(const):
    @use_implicit_args
    def f(x, y):
        def true_fn(x):
            return x
        def false_fn(x):
            return jnp.ones(x.shape)
        return jax.lax.cond(y, true_fn, false_fn, x)

    result = f(const, True)
    assert isinstance(result, jax.Array)
    assert result.shape == const.shape
    assert jnp.all(result == const.value)

def test_cond_partial_materialize_branch():
    @use_implicit_args
    def f(x, y, z):
        def true_fn(x, y):
            return y * y

        def false_fn(x, y):
            return x * x

        return jax.lax.cond(z, true_fn, false_fn, x, y)

    shape = (2, 3)
    x = ImplicitConst(2., -173, shape)
    y = ImplicitConst(
            value=ImplicitConst(1., -173, ()),
            dummy_val=-173,
            shape=shape
    )
    #y._materialize()

    result = f(x, y, True)
    assert isinstance(result, ImplicitConst)
    assert isinstance(result.value, jax.Array)
    assert result.shape == (2, 3)
    assert jnp.all(result.value == 1)

def test_switch(const):
    @use_implicit_args
    def f(x, i):
        branch_fn = lambda a, x: jnp.sum(a * x)
        branches = [partial(branch_fn, jnp.asarray(i)) for i in range(3)]

        return jax.lax.switch(i, branches, x)

    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='.*switch was not handled')
        assert f(const, 0) == 0

def test_no_implicit_args():
    def f(x):
        return jnp.sum(x ** 2)
    assert use_implicit_args(f)(jnp.ones((3, 3))) == 9

def test_vmap():
    def f(x, y):
        return jnp.sum(x * y)

    xs = ImplicitConst(jnp.arange(3), jnp.arange(-100, -97), (7, 11))
    ys = jax.random.normal(jax.random.PRNGKey(0), (7, 11))

    x_value = jnp.tile(jnp.arange(3)[:, None, None], (1, 7, 11))

    vmapped_f = jax.vmap(f, in_axes=(0, None))
    implicit_f = jax.vmap(use_implicit_args(f), in_axes=(0, None))

    result = implicit_f(xs, ys)
    expected_result = vmapped_f(x_value, ys)

    assert jnp.allclose(result, expected_result)
