# Qax: If it quacks like a tensor...

This is a WIP/experimental JAX transform which aims at reducing how much researchers need to modify model code when working on ideas which can be implemented by changing tensor representation. The example use cases I had in mind were:

* LoRA-like transformations: Represent a matrix as a full matrix + some small auxiliary data
* Quantization: Represent a matrix as a uint8 array + some floating point offsets/scales

As a toy example to demonstrate how this works, here's how you'd represent an arbitrarily shaped tensor with all entries being equal:

```python
import jax
import jax.numpy as jnp
import numpy as np
from qax import ImplicitArray, implicit_op, use_implicit_args

# To define the behavior we want, we subclass qax.ImplicitArray
class ImplicitConst(ImplicitArray):
    def __init__(self, value, shape):
        # Get the dtype in a way which will work for JAX types or python scalars
        dtype = jax.core.get_aval(value).dtype

        # The ImplicitArray constructor needs a shape and dtype
        # so that we can pretend to be a tensor of that description
        super().__init__(shape=shape, dtype=dtype)
        self.value = value

    # ImplicitArray subclasses automatically get registered as pytrees
    # so they need to define flattening and unflattening behavior
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

    # The way we define custom behavior is by writing a method and 
    # decorating it with and implicit_op('<primitive_name>') decorator
    # for commutative ops, the ImplicitArray instance will always be made the
    # lhs, but this isn't true for non-commutative ops as we'll see below
    # If the primitive involves some params (such as what axis to reduce on)
    @implicit_op('mul')
    @classmethod
    def mul(cls, primitive, const, other):
        """
        Arguments:
            - primitive: A JAX primitive
            - const: An argument guaranteed to be an ImplicitConst instance
            - other: An argument which will either be an ImplicitConst or a JAX typj
        """

        # Get the output shape in case there's any broadcasting going on
        out_shape = jax.eval_shape(
            lambda x, y: x * y,
            const.aval, # ImplicitArray defines an aval property which you can override if necessary
            other.aval
        ).shape

        if isinstance(other, ImplicitArray):
            new_val = const.value * other.value
            return ImplicitConst(new_val, out_shape)
        if other.size == 1:
            # If we get multiplied by a scalar, we can 
            # output another ImplicitConst instance
            # rather than materializing the dense array
            new_value = const.value * other
            return ImplicitConst(new_value, out_shape)

        # In the general case we just multiply our constant value by the other array
        return const.value * other

    # You can use one handler for multiple primitives by passing an iterable to the decorator
    @implicit_op(['sin', 'cos', 'exp'])
    @classmethod
    def elementwise_unop(cls, primitive, const):
        # The ImplicitArray class has a `default_handler`
        # method which we can use to execute the primitive
        # on our value
        result = cls.default_handler(primitive, const.value)
        return ImplicitConst(result, const.shape)

    # This next one is showing two things:
    # 1. If you leave off the @classmethod decorator you just won't receive the class as an argument
    # 2. If the primitive has any params (such as reduction axes) the handler will receive
    # them as a param kwarg
    @implicit_op('reduce_sum')
    def reduce_sum(primitive, const, *, params):
        sum_result = np.prod([const.shape[i] for i in params['axes']]) * const.value
        new_shape = tuple(d for d in const.shape if d not in params['axes'])
        return ImplicitConst(sum_result, new_shape)

    # If we hit an operation we haven't implemented such as 'sqrt'
    # we need to tell Qax how to materialize our ImplicitConst
    # into an actual JAX array
    def materialize(self):
        return jnp.full(self.shape, self.value, dtype=self.dtype)

    def __str__(self):
        return f'ImplicitConst({self.value}, {self.shape})'

# This decorator makes it so that `f` can handle inputs which are instances
# of ImplicitArray subclasses (or pytrees containing such instances)
# You can also still call it with ordinary JAX inputs
@use_implicit_args
def f(a, b):
    c = a * b
    d = jnp.sin(c)
    return jnp.sum(d)

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
```

The API might change at any time since there's a high chance there are issues I'm not aware of or preferable ways to implement things. That being said, it would be super helpful if people could try it and give feedback, or tell me about possible use case they find.

## Installation

```bash
git clone https://github.com/davisyoshida/qax.git
cd qax
pip install -e .
```
