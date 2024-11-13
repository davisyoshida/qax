from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from qax import (
    ArrayValue,
    ImplicitArray,
    default_handler,
    primitive_handler,
    use_implicit_args,
)
from qax.constants import (
    CUMULATIVE_REDUCTION_OPS,
    ELEMENTWISE_BINOPS,
    ELEMENTWISE_UNOPS,
    REDUCTION_OPS,
)

primitive_example_params = {
    "convert_element_type": {"new_dtype": jnp.float32, "weak_type": False},
    "integer_pow": {"y": 3},
    "reduce_precision": {"exponent_bits": 4, "mantissa_bits": 12},
    "round": {"rounding_method": jax.lax.RoundingMethod.AWAY_FROM_ZERO},
}

for op in REDUCTION_OPS:
    primitive_example_params[op] = {"axes": (1,)}

for op in CUMULATIVE_REDUCTION_OPS:
    primitive_example_params[op] = {"axis": 1, "reverse": False}

primitive_example_params["argmax"] = primitive_example_params["argmin"] = {
    "axes": (1,),
    "index_dtype": jnp.int32,
}

input_dtypes = {
    "clz": jnp.int32,
    "not": jnp.uint8,
    "population_count": jnp.int32,
    "imag": jnp.complex64,
    "real": jnp.complex64,
    "shift_right_logical": (jnp.int32, jnp.int32),
    "shift_right_arithmetic": (jnp.int32, jnp.int32),
    "shift_left": (jnp.int32, jnp.int32),
    "or": (jnp.int32, jnp.int32),
    "and": (jnp.int32, jnp.int32),
    "xor": (jnp.int32, jnp.int32),
}


def make_class_for_primitive(primitive):
    @dataclass
    class StackedArray(ImplicitArray):
        a: ArrayValue
        b: ArrayValue

        def materialize(self):
            return jnp.concatenate((self.a, self.b), axis=0)

        def __repr__(self):
            return f"StackedArray({self.a}, {self.b})"

    return StackedArray


@pytest.mark.parametrize("primitive", ELEMENTWISE_UNOPS)
def test_unop(primitive):
    StackedArray = make_class_for_primitive(primitive)

    @primitive_handler(primitive)
    def handler(primitive, arg: StackedArray, **kwargs):
        new_a = default_handler(primitive, arg.a, **kwargs)
        new_b = default_handler(primitive, arg.b, **kwargs)
        return StackedArray(
            new_a,
            new_b,
        )

    lax_primitive = getattr(jax.lax, f"{primitive}_p")

    def f(x):
        params = primitive_example_params.get(primitive, {})
        return lax_primitive.bind(x, **params)

    to_type = input_dtypes.get(primitive, jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 7)).astype(to_type)
    y = jax.random.normal(jax.random.PRNGKey(1), (9, 7)).astype(to_type)
    stacked = StackedArray(x, y)
    expected = f(stacked.materialize())

    with_implicit = use_implicit_args(f)(stacked).materialize()

    close = jnp.isclose(with_implicit, expected)
    nan_agree = jnp.logical_and(jnp.isnan(with_implicit), jnp.isnan(expected))
    assert jnp.all(close | nan_agree)


@pytest.mark.parametrize("primitive", ELEMENTWISE_BINOPS)
def test_binop(primitive):
    StackedArray = make_class_for_primitive(primitive)

    @primitive_handler(primitive)
    def handler(primitive, arg1: StackedArray, arg2: StackedArray, **kwargs):
        new_a = default_handler(primitive, arg1.a, arg2.a, **kwargs)
        new_b = default_handler(primitive, arg1.b, arg2.b, **kwargs)
        return StackedArray(new_a, new_b)

    lax_primitive = getattr(jax.lax, f"{primitive}_p")

    def f(x, y):
        params = primitive_example_params.get(primitive, {})
        return lax_primitive.bind(x, y, **params)

    lhs_type, rhs_type = input_dtypes.get(primitive, (jnp.float32, jnp.float32))
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 7)).astype(lhs_type)
    y = jax.random.normal(jax.random.PRNGKey(1), (9, 7)).astype(lhs_type)
    stacked1 = StackedArray(x, y)

    z = jax.random.normal(jax.random.PRNGKey(2), (3, 7)).astype(rhs_type)
    w = jax.random.normal(jax.random.PRNGKey(3), (9, 7)).astype(rhs_type)
    stacked2 = StackedArray(z, w)

    expected = f(stacked1.materialize(), stacked2.materialize())

    with_implicit = use_implicit_args(f)(stacked1, stacked2).materialize()

    close = jnp.isclose(with_implicit, expected)
    nan_agree = jnp.logical_and(jnp.isnan(with_implicit), jnp.isnan(expected))
    assert jnp.all(close | nan_agree)


@pytest.mark.parametrize("primitive", REDUCTION_OPS)
def test_reduction(primitive):
    StackedArray = make_class_for_primitive(primitive)

    @primitive_handler(primitive)
    def handler(primitive, arg: StackedArray, *, axes, **params):
        if 0 in axes:
            raise ValueError("Tests should use axis 1")
        a_reduced = default_handler(primitive, arg.a, axes=axes, **params)
        b_reduced = default_handler(primitive, arg.b, axes=axes, **params)
        return StackedArray(a_reduced, b_reduced)

    lax_primitive = getattr(jax.lax, f"{primitive}_p")

    def f(x):
        params = primitive_example_params.get(primitive, {})
        return lax_primitive.bind(x, **params)

    to_type = input_dtypes.get(primitive, jnp.int32)
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 7)).astype(to_type)
    y = jax.random.normal(jax.random.PRNGKey(1), (9, 7)).astype(to_type)
    stacked = StackedArray(x, y)
    expected = f(stacked.materialize())

    with_implicit = use_implicit_args(f)(stacked).materialize()

    close = jnp.isclose(with_implicit, expected)
    nan_agree = jnp.logical_and(jnp.isnan(with_implicit), jnp.isnan(expected))
    assert jnp.all(close | nan_agree)


@pytest.mark.parametrize("primitive", CUMULATIVE_REDUCTION_OPS)
def test_cumulative_reduction(primitive):
    StackedArray = make_class_for_primitive(primitive)

    @primitive_handler(primitive)
    def handler(primitive, arg: StackedArray, *, axis, **params):
        if axis != 1:
            raise ValueError("Tests should use axis 1")

        a_reduced = default_handler(primitive, arg.a, axis=axis, **params)
        b_reduced = default_handler(primitive, arg.b, axis=axis, **params)

        return StackedArray(a_reduced, b_reduced)

    lax_primitive = getattr(jax.lax, f"{primitive}_p")

    def f(x):
        params = primitive_example_params.get(primitive, {})
        return lax_primitive.bind(x, **params)

    to_type = input_dtypes.get(primitive, jnp.float32)
    x = jax.random.normal(jax.random.PRNGKey(0), (3, 7)).astype(to_type)
    y = jax.random.normal(jax.random.PRNGKey(1), (9, 7)).astype(to_type)
    stacked = StackedArray(x, y)
    expected = f(stacked.materialize())

    with_implicit = use_implicit_args(f)(stacked).materialize()

    close = jnp.isclose(with_implicit, expected)
    nan_agree = jnp.logical_and(jnp.isnan(with_implicit), jnp.isnan(expected))
    assert jnp.all(close | nan_agree)
