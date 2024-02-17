"""
Implementation of SymbolicConstant, an ImplicitArray subclass representing jnp.full(shape, value, dtype).
The data is stored in pytree auxilary data, and all supported operations are run at compile time.
This is useful for forcing constant folding in cases where the JIT would not know that an argument
is constant, or for lowering the cost of constant folding on large constant arrays.
These do not respect NaNs in certain ways, e.g. 0 * NaN = 0, max(inf, NaN) = inf
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .implicit.implicit_array import ArrayValue, ImplicitArray, UninitializedAval, aux_field, use_implicit_args
from .implicit import implicit_utils as iu
from .primitives import get_primitive_handler, primitive_handler, default_handler
from .common.type_utils import Complement
from .constants import ELEMENTWISE_BINOPS, ELEMENTWISE_UNOPS

_GENERAL = -2
_SPECIALIZED = -1

def _get_shape_dtype(x, shape, dtype):
    if shape is None:
        shape = np.shape(x)
    else:
        shape = jax.core.canonicalize_shape(shape)

    if dtype is None:
        jax.lax.dtype(x)
    return shape, dtype

def _out_shape_dtype(primitive, *args, **kwargs):
    out_aval = jax.eval_shape(
        partial(default_handler, primitive, **kwargs),
        *(jax.core.get_aval(x) for x in args)
    )
    return jax.tree_map(
        lambda x: (x.shape, x.dtype),
        out_aval
    )

def symbolic_zero_like(x, shape=None, dtype=None):
    dtype = jax.lax.dtype(x) if dtype is None else dtype
    return symbolic_full_like(x, 0, shape=shape, dtype=dtype)

def symbolic_full_like(x, fill_value, shape=None, dtype=None):
    shape, _ = _get_shape_dtype(x, shape, None)
    if dtype is None:
        dtype = jax.lax.dtype(fill_value)

    return SymbolicConstant(fill_value, shape=shape, dtype=dtype)

@dataclass
class SymbolicConstant(ImplicitArray):
    value : Any = aux_field()
    weak_type : bool = aux_field(default=False)

    def __post_init__(self):
        super().__post_init__()
        with jax.ensure_compile_time_eval():
            self.value = jnp.asarray(self.value, dtype=self.dtype)

    def compute_dtype(self):
        return jax.lax.dtype(self.value)

    def materialize(self):
        return jnp.full(self.shape, self.value, dtype=self.dtype)

    def copy(self):
        return jax.tree_map(lambda x: x, self)

@use_implicit_args
def broadcast_to(val, shape):
    return jnp.broadcast_to(val, shape)

@use_implicit_args
def astype(val, dtype):
    return val.astype(dtype)

@primitive_handler([
    'reshape',
    'broadcast_in_dim',
    'reduce_min',
    'reduce_max',
    'reduce_or',
    'reduce_and'
])
def unchanged_value_op(primitive, sym : SymbolicConstant, **kwargs):
    out_shape, out_dtype = _out_shape_dtype(primitive, sym, **kwargs)
    return SymbolicConstant(sym.value, shape=out_shape, dtype=out_dtype)

def _op_and_reshape(primitive, lhs, rhs, flip=False):
    """
    Close over one arg so we can do math at tracing time, but let the other one get traced
    """
    if flip:
        lhs, rhs = (rhs, lhs)
    @use_implicit_args
    def inner(arg):
        other = lhs
        if flip:
            arg, other = (other, arg)

        result = default_handler(primitive, arg, other)
        return result
    return inner(rhs)

def special_case_binop(name, identity=None, annihilator=None, flip=False):
    lhs_type = SymbolicConstant
    rhs_type = Complement[ArrayValue, SymbolicConstant]
    if flip:
        lhs_type, rhs_type = rhs_type, lhs_type

    @primitive_handler(name, precedence=_SPECIALIZED)
    def handler(primitive, lhs : lhs_type, rhs : rhs_type, **kwargs):
        out_shape, out_dtype = _out_shape_dtype(primitive, lhs, rhs, **kwargs)
        with jax.ensure_compile_time_eval():
            if lhs.value == identity:
                return broadcast_to(astype(rhs, out_dtype), out_shape)

            if lhs.value == annihilator:
                return SymbolicConstant(lhs.value, shape=out_shape, dtype=out_dtype)

            print(f'{primitive} {lhs.value} {rhs}')
            return _op_and_reshape(primitive, lhs.value, rhs)

special_case_binop('add', identity=0)
special_case_binop('mul', identity=1, annihilator=0)
special_case_binop('and', annihilator=0)
special_case_binop('or', identity=0)
special_case_binop('xor', identity=0)

special_case_binop('sub', identity=0, flip=True)
special_case_binop('div', identity=1, flip=True)
special_case_binop('exp', identity=1, flip=True)

special_case_binop('min', identity=float('inf'), annihilator=float('-inf'))
special_case_binop('max', identity=float('-inf'), annihilator=float('inf'))

def eval_default_handler(primitive, *args, **kwargs):
    with jax.ensure_compile_time_eval():
        result = primitive.bind(*args, **kwargs)
    return result

@primitive_handler(ELEMENTWISE_UNOPS, precedence=_GENERAL)
def handle_unop(primitive, sym : SymbolicConstant, **kwargs):
    print(f'Handling {primitive} with {sym}')
    new_val = eval_default_handler(primitive, sym.value, **kwargs)
    return symbolic_full_like(sym, new_val)

@primitive_handler(ELEMENTWISE_BINOPS, precedence=_GENERAL)
def handle_binop(primitive, lhs : SymbolicConstant, rhs : SymbolicConstant, **kwargs):
    out_shape, out_dtype = _out_shape_dtype(primitive, lhs, rhs, **kwargs)
    new_val = eval_default_handler(primitive, lhs.value, rhs.value, **kwargs)
    return symbolic_full_like(lhs, new_val, shape=out_shape, dtype=out_dtype)

@primitive_handler(['reduce_sum', 'reduce_prod'])
def reduce_sum(primitive, sym : SymbolicConstant, *, axes):
    out_shape, out_dtype = _out_shape_dtype(primitive, sym, axes=axes)
    with jax.ensure_compile_time_eval():
        if sym.value == 0:
            return SymbolicConstant(0, shape=out_shape, dtype=out_dtype)

        orig_size = np.prod(sym.shape)
        new_size = np.prod(out_shape)

        n_combined = orig_size // new_size

        new_val = sym.value
        if primitive.name == 'reduce_sum':
            new_val = new_val * n_combined
        else:
            new_val = new_val ** n_combined

    return SymbolicConstant(new_val, shape=out_shape, dtype=out_dtype)

@primitive_handler('select_n')
def handle_select_n(primitive, cond_val, *arg_vals : SymbolicConstant):
    if len(set(val.value.item() for val in arg_vals)) != 1:
        return NotImplemented

    out_shape, out_dtype = _out_shape_dtype(primitive, cond_val, *arg_vals)
    return SymbolicConstant(arg_vals[0].value, shape=out_shape, dtype=out_dtype)

