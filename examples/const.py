from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from qax import (
    ArrayValue,
    ImplicitArray,
    aux_field,
    default_handler,
    primitive_handler,
    use_implicit_args,
)

# To define the behavior we want, we subclass qax.ImplicitArray
# To define additional fields we also need to mark this class as
# a dataclass


@dataclass
class ImplicitConst(ImplicitArray):
    # Dataclass attributes may be used to define the arrays which
    # determine the concrete array being represented
    # In this case it's a single JAX scalar
    value: ArrayValue

    # ImplicitArrays are pytrees, so all attributes are automatically
    # marked as pytree children. To instead mark one as auxiliary data
    # use the qax.aux_field decorator
    my_aux_value: str = aux_field(default="some_metadata")

    # There are several ways to control the shape and dtype of an ImplicitArray
    # They are:
    # 1. Pass shape/dtype kwargs to the constructor
    # 2. Override the compute_shape/commute_dtype methods
    # 3. Override the default_shape/default_dtype class attributes
    # 4. Manually override __post_init__ and set the self.shape/self.dtype values yourself
    # 5. Do none of the above, in which case materialize() will be abstractly evaluated
    # in an attempt to derive the values. That won't work in this case since we need
    # to know them in order to call jnp.full

    default_dtype = jnp.float32

    def compute_dtype(self):
        # We're doing this instead of just self.value.dtype since we might get
        # a python scalar
        return jax.core.get_aval(self.value).dtype

    # The way we can guarantee that our ImplicitArrays will work
    # with pre-existing code is that whenever we hit an op
    # that we haven't written custom behavior for, the
    # ImplicitArray instance will be materialized into a
    # dense array and the default behavior will be used
    def materialize(self):
        return jnp.full(self.shape, self.value, dtype=self.dtype)


# The way we define custom behavior is by writing a function
# and decorating it with the primitive_handler decorator
# The type annotations are used for multiple dispatch with
# plum so make sure to get them right!
#
# For commutative ops, the ImplicitArray instance will always be made the
# lhs, but this isn't true for non-commutative ops as we'll see below
@primitive_handler("mul")
def mul(primitive, a: ImplicitConst, b: jax.Array):
    """
    Arguments:
        - primitive: A JAX primitive
        - const: An argument guaranteed to be an ImplicitConst instance
        - other: An argument which will either be an ImplicitConst or a JAX typj
    """

    # Get the output shape in case there's any broadcasting going on
    out_shape = jnp.broadcast_shapes(a.shape, b.shape)
    if b.size == 1:
        # If we get multiplied by a scalar, we can
        # output another ImplicitConst instance
        # rather than materializing the dense array
        return ImplicitConst(a.value * b.reshape(1)[0], shape=out_shape)

    # In the general case we just multiply our constant value by the other array
    result = b * a.value
    return jnp.broadcast_to(result, out_shape)


# We can also define the case where both arguments are ImplicitConsts
@primitive_handler("mul")
def mul(primitive, a: ImplicitConst, b: ImplicitConst):
    out_shape = jnp.broadcast_shapes(a.shape, b.shape)
    return ImplicitConst(a.value * b.value, shape=out_shape)


# You can use one handler for multiple primitives by passing an iterable to the decorator
@primitive_handler(["sin", "cos", "exp"])
def elementwise_unop(primitive, arg: ImplicitConst):
    # In a lot of cases the logic doesn't have anything
    # to do with the exact primitive being used so
    # we can just use qax.default_handler to execute
    result = default_handler(primitive, arg.value)
    return ImplicitConst(result, shape=arg.shape)


# If the primitive has any params (such as reduction axes) the handler will receive
# them as a param kwarg
#
# The above handlers were registered using the primitive name, which is
# is using the actual lax primitive under the hood. You can also use
# the actual primitive, which is done here
@primitive_handler(jax.lax.reduce_sum_p)
def reduce_sum(primitive, a: ImplicitConst, *, axes):
    sum_result = np.prod([a.shape[i] for i in axes]) * a.value
    new_shape = tuple(d for d in a.shape if d not in axes)
    return ImplicitConst(sum_result, shape=new_shape)


# This decorator makes it so that `f` can handle inputs which are instances
# of ImplicitArray subclasses (or pytrees containing such instances)
# You can also still call it with ordinary JAX inputs
@use_implicit_args
def f(a, b):
    c = a * b
    d = jnp.sin(c)
    return jnp.sum(d)


def main():
    shape = (5, 7)

    a_full = jnp.full(shape, 3.0)
    a_implicit = ImplicitConst(3.0, shape=shape)

    b_full = jnp.full(shape, 2.0)
    b_implicit = ImplicitConst(2.0, shape=shape)

    result = f(a_full, b_full)

    full_implicit_result = f(a_implicit, b_implicit)
    mixed_result = f(a_implicit, b_full)

    # We get the same result each time (other than some floating point error)
    # In the second case, we were able to avoid materializing the ImplicitConst
    # so we get an ImplicitConst as an output
    print(result)  # -9.779543
    print(full_implicit_result)  # ImplicitConst(-9.779541969299316, (5, 7))
    print(mixed_result)  # -9.779543

    # We can also nest ImplicitArray instances (even if they're different subclasses)
    nested_b = ImplicitConst(value=ImplicitConst(2.0, shape=()), shape=shape)

    nested_result = f(a_implicit, nested_b)
    print(nested_result)  # ImplicitConst(ImplicitConst(-9.779541969299316, ()), (5, 7))


if __name__ == "__main__":
    main()
