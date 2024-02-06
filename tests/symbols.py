import itertools
import operator as fn

import pytest

import jax
import jax.numpy as jnp
import numpy as np

import qax
from qax.symbols import SymbolicConstant

add = qax.use_implicit_args(fn.add)
mul = qax.use_implicit_args(fn.mul)

default_shape = (3, 5)

@pytest.fixture
def arr():
    return jax.random.normal(jax.random.PRNGKey(0), default_shape)

@pytest.fixture
def zeros():
    return SymbolicConstant(0, shape=default_shape, dtype=jnp.float32)

@pytest.fixture
def ones():
    return SymbolicConstant(1, shape=default_shape, dtype=jnp.float32)

@pytest.fixture
def pis():
    return SymbolicConstant(jnp.pi, shape=default_shape, dtype=jnp.float32)

def test_add(zeros, arr, pis):
    z_plus_z = add(zeros, zeros)

    assert isinstance(z_plus_z, SymbolicConstant)
    assert z_plus_z.value == 0

    assert jnp.allclose(add(zeros, arr), arr)

    pi_plus_pi = add(pis, pis)
    assert isinstance(pi_plus_pi, SymbolicConstant)
    assert jnp.isclose(pi_plus_pi.value, 2 * jnp.pi)

    pi_plus_arr = add(pis, arr)
    assert isinstance(pi_plus_arr, jnp.ndarray)
    assert jnp.allclose(pi_plus_arr, arr + jnp.pi)


def test_mul(zeros, ones, arr, pis):
    zero_times_one = mul(zeros, ones)

    assert isinstance(zero_times_one, SymbolicConstant)
    assert zero_times_one.value == 0

    assert jnp.all(mul(ones, arr) == arr)

    pi_times_pi = mul(pis, pis)
    assert isinstance(pi_times_pi, SymbolicConstant)
    assert jnp.isclose(pi_times_pi.value, jnp.pi ** 2)

    assert mul(pis, ones).value == jnp.pi

_names = ['zeros', 'ones', 'pis']
@pytest.mark.parametrize('fn,lhs,rhs', itertools.product(
    [jax.lax.add, jax.lax.mul, jax.lax.sub, jax.lax.atan2, jax.lax.max],
    _names,
    _names
))
def test_binop(fn, lhs, rhs, request):
    lhs = request.getfixturevalue(lhs)
    rhs = request.getfixturevalue(rhs)
    expected = fn(lhs.materialize(), rhs.materialize())
    result = qax.use_implicit_args(fn)(lhs, rhs)
    assert isinstance(result, SymbolicConstant)
    assert jnp.allclose(result.value, expected)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype

@pytest.mark.parametrize('fn,arg', itertools.product(
    [jnp.sum, jnp.prod, jnp.all, jnp.any, jnp.sin, jnp.isfinite],
    _names
))
def test_unop(fn, arg, request):
    value = request.getfixturevalue(arg)
    expected = fn(value.materialize())
    result = qax.use_implicit_args(fn)(value)
    assert isinstance(result, SymbolicConstant)
    assert jnp.allclose(result.value, expected)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype

def test_select_n(zeros, ones):
    @qax.use_implicit_args
    def f(c, x, y):
        return jax.lax.select_n(c, x, y)

    assert isinstance(f(True, zeros, ones), jnp.ndarray)
    assert isinstance(f(False, zeros, zeros), SymbolicConstant)
