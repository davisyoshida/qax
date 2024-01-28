[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/davisyoshida/qax/blob/master/examples/How_to_Qax.ipynb)

# Qax: If it quacks like a tensor...
🦆[Qax](https://github.com/davisyoshida/qax)🦆 is a tool for implementing types which represent tensors, but may or may not be instantiated as a single dense array on your GPU. Examples of this include:
* Quantization: A 4-bit array of integers + a small number of scale values are used to represent a full 16/32-bit array
* LoRA: An array $W$ is replaced by the array $(W + BA^T)$ so that $A$ and $B$ may be trained while leaving $W$ frozen
* Symbolic zeros/constants: For arrays which will consist entirely of a single repeated value, simply store that single value and the shape of the array
* Custom kernels: If you have a custom kernel and want to use it with existing models without modifying them, Qax is an easy way to do so
* Hopefully many more things!

The goal of Qax is to make implementing custom JAX behavior much easier, so that users won't need to deal with all the details of writing a full JAX transform. All you need to do to get custom representations is:

1. Define what data/metadata your datatype should contain
2. Optionally write any number of handlers which specify how your type behaves under JAX primitives such as multiplication
3. Write a function which constructs a dense array from your implicit representation

Both of the above are written in pure JAX, so no need for custom gradients (unless you want to of course!).

## Installation
```
pip install qax
```

## Example 1: A symbolic zero
The way you specify custom behavior with Qax is to subclass the `qax.ImplicitArray` abstract class. One of the simplest things we could implement is a symbolic zero: A data type which represents an arbitrary tensor full of zeros without actually instantiating them on the GPU.


```python
class Zeros(qax.ImplicitArray):
    default_dtype = jnp.float32

    def materialize(self):
        # self.shape and self.dtype will be
        # populated by the ImplicitArray constructor
        return jnp.zeros(self.shape, self.dtype)

    def __str__(self):
        return f'Zeros({self.shape}, {self.dtype})'
```

The only mandatory method to implement when subclassing `ImplicitArray` is `materialize()`.
`materialize()` specifies how to turn our _implicitly_ represented array into an _explicit_ one, i.e. a single dense JAX array. In the case of `Zeros`, we can just call `jnp.zeros`.

Let's instantiate a `Zeros` instance to try it out:


```python
z = Zeros(shape=(2, 3))
```

ImplicitArrays are [dataclasses](https://docs.python.org/3/library/dataclasses.html), which by default have two keyword only attributes: `shape` and `dtype`.

By default JAX won't know how to use our new type. In order to use it in functions, we apply the `@use_implicit_args` decorator:


```python
@qax.use_implicit_args
def f(x, y):
    return (x + y)[0, 0]
```


```python
with warnings.catch_warnings():
    warnings.simplefilter('always')
    print(f(z, jnp.ones(3)))
```

    /home/davis/src/qax/qax/implicit/implicit_array.py:303: UserWarning: Primitive add was not handled by class Zeros, so implicit args will be materialized.
      warnings.warn(f'Primitive {primitive.name} was not handled by class {vals[implicit_idx].__class__.__name__}, so implicit args will be materialized.')


    1.0


The cool thing is that `f` doesn't need to have any idea that it will be called with `ImplicitArray` instances, so we can use this with any pre-existing model. Right now this isn't much use, since all `z` is being materialized into a dense array as soon as it's needed for a JAX operation.

To make our `Zeros` do something productive, let's implement the fact that $x + 0$ is always equal to $x$. We do this using the `@qax.primitive_handler` decorator:


```python
def get_binop_result_shape_dtype(a, b):
    out_shape = jnp.broadcast_shapes(jnp.shape(a), jnp.shape(b))
    out_dtype = jnp.result_type(a.dtype, b.dtype)
    return out_shape, out_dtype

# primitive_handler() takes a string, JAX primitive, or a list of those types
# strings are used to find the corresponding primitive from `jax.lax`
@qax.primitive_handler('add')
def my_add_handler(primitive, a : Zeros, b):
    # Handlers will receive as arguments:
    # - primitive: a jax.core.Primitive instance (often can be ignored if the handler is just for one op)
    # Any number of arguments which are either JAX values or ImplicitArrays
    # Keyword arguments specifying parameters of the operation (e.g. axes for reduction operations)

    out_shape, out_dtype = get_binop_result_shape_dtype(a, b)

    if isinstance(b, Zeros):
        # We can return further ImplicitArray instances if we want
        return Zeros(shape=out_shape, dtype=out_dtype)

    # Return b, possibly modifying its shape or dtype
    return jnp.broadcast_to(b, out_shape).astype(out_dtype)
```

The type annotation `a : Zeros` is actually important, Qax uses [Plum](https://github.com/beartype/plum) for multiple dispatch. You can even use this to define how different subclasses of ImplicitArray should interact with each other.

(For convenience, commutative binary ops like $+$ and $\times$ will automatically get their argument order switched so that the `ImplicitArray` instance comes first.)

Now when we call `f`, we no longer see the materialization log message, since our add handler is skipping over ever instantiating the array of zeros:


```python
print(f(z, jnp.ones(3)))
```

    1.0


Let's define a multiplication handler as well, since $x \cdot 0 = 0$ for all $x$:


```python
@qax.primitive_handler('mul')
def handle_mul(primitive, a : Zeros, b):
    out_shape, out_dtype = get_binop_result_shape_dtype(a, b)

    return Zeros(shape=out_shape, dtype=out_dtype)


@jax.jit
@qax.use_implicit_args
def g(x, y):
    return (1 + x) * y

print(g(z, z))
```

    Zeros((2, 3), float32)


The output of `use_implicit_args` is a function which is compatible with all the usual JAX transformations such as `jit`, `vmap`, `grad`, etc.

Even this simple implementation is enough to let us modify the behavior of models which were written without knowing about Qax. Let's try replacing all the biases in HuggingFace's GPT-2 with zeros:


```python
@qax.primitive_handler('broadcast_in_dim')
def broadcast(primitive, a : Zeros, *, shape, broadcast_dimensions):
    # The biases get broadcast in order to add them to the activations
    # so we need to handle that case
    # Sometimes the simplest thing to do is use jax.eval_shape
    # to figure out what shape to return
    result_shape = jax.eval_shape(
        partial(jax.lax.broadcast_in_dim, shape=shape, broadcast_dimensions=broadcast_dimensions),
        a.aval # ImplicitArray has an aval property which will get an abstract shape/dtype
    ).shape
    return Zeros(shape=result_shape, dtype=a.dtype)


model, params = transformers.FlaxAutoModelForCausalLM.from_pretrained('gpt2', _do_init=False)

inputs = jnp.arange(1, 10)[None]

# Helper function to switch all the biases
# in the params out for some other value
def replace_biases(params, replacer):
    def maybe_replace_val(path, val):
        if val.ndim != 1:
            return val

        # Skip layernorms
        if any(
            isinstance(p, jax.tree_util.DictKey) and p.key.startswith('ln')
            for p in path
        ):
            return val
        return replacer(shape=val.shape, dtype=val.dtype)
    return jax.tree_util.tree_map_with_path(maybe_replace_val, params)


# Replace the biases with dense zero arrays:
params_with_zeros = replace_biases(params, jnp.zeros)
print('New bias:', params['transformer']['h']['0']['attn']['c_attn']['bias'])

output = model(inputs, params=params_with_zeros).logits
print('Last logit average:', jnp.mean(output[0, -1]))
```

    New bias: [ 0.48033914 -0.5254326  -0.42926455 ...  0.01257301 -0.04987717
      0.00324764]
    Last logit average: -105.25595


Now let's try replacing them with our symbolic zeros instead:


```python
params_with_zeros = replace_biases(params, Zeros)
print('New bias:', params['transformer']['h']['0']['attn']['c_attn']['bias'])

# In this case since we're calling the model directly, we need to
# wrap it so we can pass params in a positional argument
# This usually won't be an issue since the call to the model will
# be inside a loss function or some other function

output = qax.use_implicit_args(model)(inputs, params=params_with_zeros).logits
print('Last logit average:', jnp.mean(output[0, -1]))
```

    New bias: [ 0.48033914 -0.5254326  -0.42926455 ...  0.01257301 -0.04987717
      0.00324764]
    Last logit average: -105.25595



```python
del model
del params
```

We got the same result, but using 0 FLOPs for adding the biases! If you really wanted to flesh out the behavior of `Zeros`, you could also add handlers defining its output for primitives such as `sin`, `cos`, etc. Let's move on to something more interesting though.

## Example 2: LoRA
In this example we'll implement [LoRA](https://arxiv.org/abs/2106.09685) in just a few lines of code. Unlike the `Zeros` example from the previous section, our `ImplicitArray` subclass will actually contain data this time. As such we'll need to implement flattening/unflattening logic, since all `ImplicitArray` subclasses are pytrees. This also means you can use `tree_map` and friends to manipulate them.

To add child pytrees to a subclass, we just add them as dataclass attributes. To add auxilary data, you can wrap a field with `qax.aux_field` which is just a wrapper around `dataclass.field`.

LoRA replaces a matrix $W$ with the matrix $W_0 + AB^T$, so we'll have three arrays as new attributes.


```python
@dataclass
class LoraMatrix(qax.ImplicitArray):
    """Represent W + A B^T"""
    w : qax.ArrayValue
    a : qax.ArrayValue
    b : qax.ArrayValue

    # auxiliary data example
    is_array_happy : bool = qax.aux_field(default=True)

    def __post_init__(self):
        # If you need to do any validation, you can override the __post_init__ method
        # This example is purely for error checking, but you can also
        # add manipulations of the attributes
        super().__post_init__()
        w_aval = jax.core.get_aval(self.w)
        a_aval = jax.core.get_aval(self.a)
        b_aval = jax.core.get_aval(self.b)
        assert w_aval.ndim == a_aval.ndim == b_aval.ndim == 2
        assert a_aval.shape[1] == b_aval.shape[1]
        assert a_aval.shape[0] == w_aval.shape[0]
        assert b_aval.shape[0] == w_aval.shape[1]
        assert a_aval.dtype == b_aval.dtype == w_aval.dtype

    def materialize(self):
        return self.w + self.a @ self.b.T

@qax.primitive_handler('dot_general')
def f(primitive, x : jax.Array, w : LoraMatrix, *, dimension_numbers, **kwargs):
    # For this example, we'll only handle the simple case of of x @ w, rather than
    # all possible dot_general invocations
    (lhs_contract, rhs_contract), (lhs_batch, rhs_batch) = dimension_numbers

    # This check just makes sure that all that's happening is a simple matmul
    if not (
        len(w.shape) == 2
        and lhs_contract == (x.ndim - 1,)
        and rhs_contract == (0,)
        and lhs_batch == ()
        and rhs_batch == ()
    ):
        # If we want to only partially handle a particular primitive,
        # we can fall back to the default logic by returning NotImplemented
        return NotImplemented

    kwargs = {**kwargs, 'dimension_numbers': dimension_numbers}
    # In order to defer to the default implementation of the primitive,
    # use the qax.default_handler helper:
    result = qax.default_handler(
        primitive, # pass the primitive
        x, w.w,    # Any number of positional arguments,
        **kwargs   # Then the primitive's keyword args
    )

    xa = qax.default_handler(primitive, x, w.a, **kwargs)

    xab = qax.default_handler(primitive, xa, w.b.T, **kwargs)

    result += xab
    return result

def lora_from_tree(tree, key, lora_dim=8):
    """
    Helper function for replacing non-embedding weight
    matrices in T5 with LoraMatrix instances.
    """
    def iter_keys(key):
        while True:
            key, k2 = jax.random.split(key)
            yield k2

    key_it = iter_keys(key)
    def map_fn(path, val):
        if val.ndim != 2:
            return val

        # Skip embedding params
        if any(isinstance(p, jax.tree_util.DictKey) and p.key == 'embedding' for p in path):
            return val

        a = jax.random.normal(next(key_it), (val.shape[0], lora_dim), val.dtype)
        b = jnp.zeros((val.shape[1], lora_dim), val.dtype)
        return LoraMatrix(val, a, b)

    return jax.tree_util.tree_map_with_path(map_fn, tree)
```

Let's try it out on a T5 model:


```python
t5, params = transformers.FlaxAutoModelForSeq2SeqLM.from_pretrained('t5-small', _do_init=False)
tokenizer = transformers.AutoTokenizer.from_pretrained('t5-small')
encoder_inputs = jnp.asarray(tokenizer.encode('Some input'))[None]
decoder_inputs = jnp.asarray([0] + tokenizer.encode('Some output'))[None]

lora_params = lora_from_tree(params, jax.random.PRNGKey(1234))
```


```python
orig_output = t5(input_ids=encoder_inputs, decoder_input_ids=decoder_inputs, params=params).logits
```


```python
lora_output = qax.use_implicit_args(t5)(
    input_ids=encoder_inputs,
    decoder_input_ids=decoder_inputs,
    params=lora_params
).logits
print(jnp.max(jnp.abs(lora_output - orig_output)))
```

    0.0


The LoRA result is identical to the execution of the unmodified network, and we didn't get any materialization warnings so we successfully made a LoRA forward pass without ever calculating $W + AB^T$!

## Training
So far we haven't looked at how to train a model when using Qax. The main thing to understand is that you should apply `qax.use_implicit_args` first, _then_ differentiate the resulting function. `use_implicit_args` transforms the function into one which goes from pytrees to pytrees, so all the standard JAX autodiff machinery will work.

If you need to update only a subset of the elements of an ImplicitArray instance (e.g. only `a` and `b` for LoRA), Qax provides `qax.utils.freeze_keys` to make this easier. Here's an end-to-end example training T5 to memorize the input/output pair from above:


```python
optimizer = optax.adam(3e-4)
# freeze_keys_in_optimizer takes an optax optimizer, the ImplicitArray subclass to freeze for,
# and an iterable of the keys to be frozen
optimizer = qax.utils.freeze_keys(optimizer, LoraMatrix, ['w'])

# We're only using a single example so we'll just close over the training data
# There are no code changes from an ordinary training loop other than decorating
# loss_fn with @use_implicit_args

@qax.use_implicit_args
def loss_fn(params):
    decoder_ids = decoder_inputs[:, :-1]
    targets = decoder_inputs[:, 1:]
    logits = t5(
        input_ids=encoder_inputs,
        decoder_input_ids=decoder_ids,
        params=params
    ).logits

    logprobs = jax.nn.log_softmax(logits)
    target_logprobs = jnp.take_along_axis(logprobs, targets[:, :, None], axis=-1)
    loss = -jnp.sum(target_logprobs)
    return loss

grad_fn = jax.value_and_grad(loss_fn)

@jax.jit
def update(params, opt_state):
    loss, grads = grad_fn(params)
    updates, new_opt_state = optimizer.update(grads, opt_state, params=params)
    new_params = optax.apply_updates(updates, params)
    return loss, new_params, new_opt_state

opt_state = optimizer.init(lora_params)
for step in range(20):
    loss, lora_params, opt_state = update(lora_params, opt_state)
    print(f'{step}. {loss:.3f}')
```

    0. 8.882
    1. 5.375
    2. 3.787
    3. 2.524
    4. 1.491
    5. 0.723
    6. 0.242
    7. 0.062
    8. 0.022
    9. 0.013
    10. 0.011
    11. 0.009
    12. 0.008
    13. 0.007
    14. 0.007
    15. 0.006
    16. 0.005
    17. 0.004
    18. 0.003
    19. 0.003


That's all you need to know to get started using Qax!

## Example 3: Nesting
Qax supports arbitrary nesting of `ImplicitArray` instances without. Here's a quick demo combining the previous two examples:


```python
@qax.use_implicit_args
def g(w, x):
    return jnp.sum(x @ w)

w = jnp.ones((3, 5))
x = jnp.arange(3, dtype=jnp.float32)

lora_with_symbolic_zero = LoraMatrix(
    w=w,
    a=Zeros(shape=(w.shape[0], 6)),
    b=Zeros(shape=(w.shape[1], 6))
)
print(f'Original: {g(w, x)}')
with warnings.catch_warnings():
    warnings.simplefilter('always')
    print(f'With lora: {g(lora_with_symbolic_zero, x)}')
```

    Original: 15.0
    With lora: 15.0


    UserWarning: Primitive dot_general was not handled by class Zeros, so implicit args will be materialized.
      warnings.warn(f'Primitive {primitive.name} was not handled by class {vals[implicit_idx].__class__.__name__}, so implicit args will be materialized.')
    UserWarning: Primitive transpose was not handled by class Zeros, so implicit args will be materialized.
      warnings.warn(f'Primitive {primitive.name} was not handled by class {vals[implicit_idx].__class__.__name__}, so implicit args will be materialized.')


If we wanted we could write a `dot_general` handler to avoid the materialization as well, but the main point is just to illustrate that it's easy to mix and match different `ImplicitArray` subclasses. A more useful example might be using a symbolic zero as the offset for a quantization datatypes which expects both an offset and a scale.

## Other examples
[Here's](https://github.com/davisyoshida/abnormal-floats/blob/master/transform.py) an example of using Qax to implement a 4-bit quantized matrix representation.
