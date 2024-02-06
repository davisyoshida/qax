from dataclasses import dataclass
from functools import partial
import warnings

import jax
import jax.numpy as jnp

import qax
from qax.symbols import SymbolicConstant

def test_scan_empty_tree():
    a = SymbolicConstant(1, shape=(3, 5), dtype=jnp.float32)

    def body(carry, x):
        return [carry[0] + jnp.sum(x), carry[1]], x[1]

    qax_scan = partial(qax.use_implicit_args(jax.lax.scan), body, [0, jnp.ones(11)])

    materialized_output = qax_scan(a.materialize())

    with warnings.catch_warnings():
        warnings.filterwarnings('error', message='Primitive scan was not')
        output = qax_scan(a)

    assert jax.tree_util.tree_all(jax.tree_map(
        lambda x, y: jnp.all(x == y),
        materialized_output,
        output
    ))

def test_scan_arr_with_data():
    @dataclass
    class SumArray(qax.ImplicitArray):
        a : jax.Array
        b : jax.Array

        def materialize(self):
            return 2 * (self.a + self.b)

    @qax.primitive_handler('add')
    def add(primitive, lhs : SumArray, rhs : jax.Array):
        return 2 * lhs.a + 2 * lhs.b + rhs

    a = SumArray(jnp.ones((3, 5)), jnp.ones((3, 5)))

    def body(carry, x):
        return jnp.sum(x + carry), None

    qax_scan = partial(qax.use_implicit_args(jax.lax.scan), body, 0.)

    expected_output = qax_scan(a.materialize())
    output = qax_scan(a)

    assert expected_output == output

def test_output_implicit():
    @dataclass
    class Wrapper(qax.ImplicitArray):
        a : jax.Array

        def materialize(self):
            return self.a

    @qax.primitive_handler('add')
    def add(primitive, lhs : Wrapper, rhs : jax.Array):
        return Wrapper(a=lhs.a + rhs)

    @qax.primitive_handler('reduce_sum')
    def reduce_sum(primitive, x : Wrapper, **kwargs):
        return Wrapper(qax.default_handler(primitive, x.a, **kwargs))

    def body(carry, x):
        return jnp.sum(x + carry), None

    qax_scan = partial(qax.use_implicit_args(jax.lax.scan), body, 0.)

    a = Wrapper(jnp.arange(6).reshape((3, 2)).astype(jnp.float32))

    expected_output = qax_scan(a.materialize())
    output = qax_scan(a)

    assert expected_output == output
