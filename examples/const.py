import jax
import jax.numpy as jnp
import numpy as np

from qax import ImplicitArray, primitive_handler, use_implicit_args, default_handler

# To define the behavior we want, we subclass qax.ImplicitArray
class ImplicitConst(ImplicitArray):
    def __init__(self, value, shape):
        # The ImplicitArray constructor needs a shape and dtype
        # so that we can pretend to be a tensor of that description

        # This is in case we get a python scalar as an argument
        dtype = jax.core.get_aval(value).dtype

        super().__init__(shape=shape, dtype=dtype)
        self.value = value

    # The way we can guarantee that our ImplicitArrays will work
    # with pre-existing code is that whenever we hit an op
    # that we haven't written custom behavior for, the
    # ImplicitArray instance will be materialized into a
    # dense array and the default behavior will be used
    def materialize(self):
        return jnp.full(self.shape, self.value, dtype=self.dtype)

    # ImplicitArray subclasses automatically get registered as pytrees
    # so they need to define flattening and unflattening behavior
    # This allows them to be easily used with other transforms
    # such as jax.grad
    def flatten(self):
        """
        Should return:
            - an iterable of (name, child) tuples
            - Any auxiliary data (shape and dtype are handled by the base class we 
        """
        return [('value', self.value)], ()

    def unflatten(self, aux_data, children):
        """
        This will be called on an _uninitialized_ instance of ImplicitConst.
        This is because things like `tree_map` use unflatten but not __init__, and we 
        want to be able to be able to produce ImplicitConst's with invalid internal data
        via tree_map if necessary.
        """
        self.value, = children


    def __str__(self):
        return f'ImplicitConst({self.value}, {self.shape})'

# The way we define custom behavior is by writing a function
# and decorating it with the primitive_handler decorator
# The type annotations are used for multiple dispatch with
# plum so make sure to get them right!
#
# For commutative ops, the ImplicitArray instance will always be made the
# lhs, but this isn't true for non-commutative ops as we'll see below
@primitive_handler('mul')
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
        return ImplicitConst(a.value * b.reshape(1)[0], out_shape)

    # In the general case we just multiply our constant value by the other array
    result = b * a.value
    return jnp.broadcast_to(result, out_shape)

# We can also define the case where both arguments are ImplicitConsts
@primitive_handler('mul')
def mul(primitive, a: ImplicitConst, b: ImplicitConst):
    out_shape = jnp.broadcast_shapes(a.shape, b.shape)
    return ImplicitConst(a.value * b.value, out_shape)

# You can use one handler for multiple primitives by passing an iterable to the decorator
@primitive_handler(['sin', 'cos', 'exp'])
def elementwise_unop(primitive, arg : ImplicitConst):
    # In a lot of cases the logic doesn't have anything
    # to do with the exact primitive being used so 
    # we can just use qax.default_handler to execute
    result = default_handler(primitive, arg.value)
    return ImplicitConst(result, arg.shape)

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
    return ImplicitConst(sum_result, new_shape)


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

    a_full = jnp.full(shape, 3.)
    a_implicit = ImplicitConst(3., shape)

    b_full = jnp.full(shape, 2.)
    b_implicit = ImplicitConst(2., shape)

    result = f(a_full, b_full)

    full_implicit_result = f(a_implicit, b_implicit)
    mixed_result = f(a_implicit, b_full)


    # We get the same result each time (other than some floating point error)
    # In the second case, we were able to avoid materializing the ImplicitConst
    # so we get an ImplicitConst as an output
    print(result)               # -9.779543
    print(full_implicit_result) # ImplicitConst(-9.779541969299316, (5, 7))
    print(mixed_result)         # -9.779543

    # We can also nest ImplicitArray instances (even if they're different subclasses)
    nested_b = ImplicitConst(
        value=ImplicitConst(2., ()),
        shape=shape
    )

    nested_result = f(a_implicit, nested_b)
    print(nested_result) # ImplicitConst(ImplicitConst(-9.779541969299316, ()), (5, 7))

if __name__ == '__main__':
    main()
