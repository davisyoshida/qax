from abc import ABC, abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields, is_dataclass
from functools import partial, wraps
from itertools import chain
from typing import ClassVar, Optional
import warnings

import jax
from jax.api_util import flatten_fun, flatten_fun_nokwargs
from jax import core
import jax.linear_util as lu
import jax.interpreters.partial_eval as pe
from jax.tree_util import register_pytree_with_keys_class
import jax.numpy as jnp

from jax._src.typing import DTypeLike, Shape

from . import constants
from .primitives import ArrayValue, get_primitive_handler
from . import utils

def _use_implicit_flat(f_flat):
    @wraps(f_flat)
    def inner(*flat_args):
        with core.new_main(ImplicitArrayTrace) as main:
            trace = main.with_cur_sublevel()
            tracers_in = [ImplicitArrayTracer(trace, val) if isinstance(val, ImplicitArray) else val for val in flat_args]

            results = f_flat.call_wrapped(*tracers_in)
            tracers_out = [trace.full_raise(t).value for t in results]
        return tracers_out
    return inner

def use_implicit_args(f):
    """
    Decorator which allows a function to accept arguments which subclass ImplicitArray, possibly
    including further ImplicitArray instances as children.
    Any number of arguments (including 0) may be ImplicitArrays.
    """
    @wraps(f)
    def inner(*args, **kwargs):
        flat_args, in_tree = utils.tree_flatten_with_implicit((args, kwargs))
        f_flat, out_tree = flatten_fun(lu.wrap_init(f), in_tree)
        outs_flat = _use_implicit_flat(f_flat)(*flat_args)
        return jax.tree_util.tree_unflatten(out_tree(), outs_flat)
    return inner

def aux_field(metadata=None, **kwargs):
    metadata = dict(metadata) if metadata else {}
    metadata['implicit_array_aux'] = True
    return field(metadata=metadata, **kwargs)

class UninitializedAval(Exception):
    def __init__(self, kind):
        super().__init__(_AVAL_ERROR_MESSAGE.format(kind))

# This descriptor and the below context manager support discovering the aval
# of an ImplicitArray. We don't want to throw an error just because a shape
# wasn't passed, since it may be possible to infer it via materialization
class _AvalDescriptor:
    def __set_name__(self, owner, name):
        self._name = f'_{name}'

    def __get__(self, obj, owner=None):
        if obj is None:
            return None
        result = getattr(obj, self._name, None)
        if result is None:
            raise UninitializedAval(kind=self._name[1:])
        return result

    def __set__(self, obj, value):
        setattr(obj, self._name, value)

# Context manager used for disabling UninitializedAval errors
# during tree flattening only
_aval_discovery = ContextVar('aval_discovery', default=False)
@contextmanager
def _aval_discovery_context():
    token = _aval_discovery.set(True)
    try:
        yield
    finally:
        _aval_discovery.reset(token)

@dataclass
class _ImplicitArrayBase(ArrayValue,ABC):
    commute_ops : ClassVar[bool] = True
    default_shape : ClassVar[Optional[Shape]] = None
    default_dtype : ClassVar[Optional[DTypeLike]] = None

    shape : Optional[Shape] = aux_field(kw_only=True, default=None)
    dtype : DTypeLike = aux_field(kw_only=True, default=None)

@dataclass
class ImplicitArray(_ImplicitArrayBase):
    """
    Abstract class for representing an abstract array of a given shape/dtype without actually instantiating it.
    Subclasses must implement the materialize method, which defines the relationship between the implicit array
    and the value it represents. Subclasses are valid arguments to functions decorated with qax.use_implicit_args.

    All subclasses are automatically registered as pytrees using jax.tree_util.register_pytree_with_keys_class.
    Any dataclass attributes added will be included as children, unless they are decorated with qax.aux_field
    in which case they are passed as auxiliary data during flattening.

    The represented shape and dtype may be defined in any of the following ways:
        - Explicitly passing shape/dtype keyword arguments at initialization
        - Overriding the default_shape/default_dtype class variables
        - Overriding the compute_shape/compute_dtype methods, which are called during __post_init__
        - Overriding __post_init__ and manually setting shape/dtype before calling super().__post_init__
        - None of the above, in which case an shape/dtype will be inferred by by running jax.eval_shape()
          on the subclass's materialize method.
    """

    shape = _AvalDescriptor()
    dtype = _AvalDescriptor()

    def __post_init__(self):
        try:
            aval = _get_materialization_aval(self)
        except UninitializedAval:
            # Materialization depends on currently uninitialized shape/dtype
            aval = None

        shape = None
        try:
            shape = self.shape
        except UninitializedAval as e:
            shape = self.shape = self.compute_shape()

        if aval is not None:
            if shape is None:
                self.shape = aval.shape
            elif shape != aval.shape:
                warnings.warn(f'ImplicitArray shape {shape} does not match materialization shape {aval.shape}')
        elif shape is None:
            raise UninitializedAval('shape')

        dtype = None
        try:
            dtype = self.dtype
        except UninitializedAval as e:
            dtype = self.dtype = self.compute_dtype()

        if dtype is None and aval is None:
            # We have a shape but not a dtype, try once again to infer the dtype
            aval = _get_materialization_aval(self)

        if aval is not None:
            if dtype is None:
                self.dtype = aval.dtype
            elif dtype != aval.dtype:
                warnings.warn(f'ImplicitArray dtype {dtype} does not match materialization dtype {aval.dtype}')
        elif dtype is None:
            raise UninitializedAval('dtype')



    def compute_shape(self):
        """
        Override this method if the subclass instance's shape should be computed based on its other properties.
        Returns: shape
        """
        return self.default_shape

    def compute_dtype(self):
        """
        Override this method if the subclass instance's dtype should be computed based on its other properties.
        Returns: dtype
        """
        return self.default_dtype

    @property
    def aval(self):
        return core.ShapedArray(self.shape, self.dtype)

    @classmethod
    def default_handler(cls, primitive, *args, params=None):
        if params is None:
            params = {}
        return materialize_handler(primitive, *args, params=params)

    @abstractmethod
    def materialize(self):
        pass

    def tree_flatten_with_keys(self):
        children = []
        aux_data = []
        for name, is_aux in _get_names_and_aux(self):
            try:
                value = getattr(self, name)
            except UninitializedAval:
                if not _aval_discovery.get():
                    raise
                value = None
            if is_aux:
                aux_data.append(value)
            else:
                children.append((name, value))

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        child_it = iter(children)
        aux_it = iter(aux_data)
        obj = cls.__new__(cls)
        for name, is_aux in _get_names_and_aux(cls):
            value = next(aux_it if is_aux else child_it)
            setattr(obj, name, value)

        return obj

    def handle_primitive(self, primitive, *args, params):
        handler = lu.wrap_init(partial(get_primitive_handler(primitive), primitive))
        use_params = params

        if len(args) == 2 and self.commute_ops:
            args, use_params = _maybe_swap_args(primitive.name, args, use_params)

        #maybe_kwargs = {'params': params} if params else {}
        flat_args, in_tree = utils.flatten_one_implicit_layer((args, params))
        flat_handler, out_tree = flatten_fun(handler, in_tree)

        result = use_implicit_args(flat_handler.call_wrapped)(*flat_args)
        return jax.tree_util.tree_unflatten(out_tree(), result)

    def __init_subclass__(cls, commute_ops=True, **kwargs):
        super().__init_subclass__(**kwargs)

        if not is_dataclass(cls):
            raise TypeError(f'{cls.__name__} must be a dataclass')
        core.pytype_aval_mappings[cls] = lambda x: x.aval
        register_pytree_with_keys_class(cls)
        return cls

def _get_names_and_aux(obj):
    for val in fields(obj):
        yield val.name, bool(val.metadata.get('implicit_array_aux'))

def _materialize_all(it):
    return [utils.materialize_nested(val) if isinstance(val, ImplicitArray) else val for val in it]

def _maybe_swap_args(op_name, args, params):
    if isinstance(args[0], ImplicitArray):
        return args, params
    if op_name in constants.COMMUTATIVE_OPS:
        return args[::-1], params

    return args, params

class ImplicitArrayTracer(core.Tracer):
    def __init__(self, trace, value):
        super().__init__(trace)
        self.value = value

    @property
    def aval(self):
        if isinstance(self.value, ImplicitArray):
            return self.value.aval
        return core.get_aval(self.value)

    def full_lower(self):
        if isinstance(self.value, ImplicitArray):
            return self

        return core.full_lower(self.value)

class ImplicitArrayTrace(core.Trace):
    pure = lift = lambda self, val: ImplicitArrayTracer(self, val)

    def process_primitive(self, primitive, tracers, params):
        outs = NotImplemented
        vals = [t.value for t in tracers]
        implicit_idx = next(i for i, v in enumerate(vals) if isinstance(v, ImplicitArray))

        # First try to handle the primitive using custom handlers
        outs = vals[implicit_idx].handle_primitive(primitive, *vals, params=params)

        if outs is NotImplemented:
            # For higher order primitives most users won't implement custom
            # logic, so there shouldn't be a warning
            if primitive.name in _default_handlers:
                outs = _default_handlers[primitive.name](primitive, *vals, params=params)
            else:
                warnings.warn(f'Primitive {primitive.name} was not handled by class {vals[implicit_idx].__class__.__name__}, so implicit args will be materialized.')

        if outs is NotImplemented:
            outs = vals[implicit_idx].default_handler(primitive, *vals, params=params)

        if primitive.multiple_results:
            return [ImplicitArrayTracer(self, out) for out in outs]
        return ImplicitArrayTracer(self, outs)

def wrap_jaxpr(jaxpr, vals_with_implicits, return_closed=True):
    if isinstance(jaxpr, jax.core.ClosedJaxpr):
        literals = jaxpr.literals
        jaxpr = jaxpr.jaxpr
    else:
        literals = []

    wrapped_fn = lu.wrap_init(use_implicit_args(partial(core.eval_jaxpr, jaxpr)))
    flat_args, in_tree = jax.tree_util.tree_flatten((literals, *vals_with_implicits))
    flat_fn, out_tree = flatten_fun_nokwargs(wrapped_fn, in_tree)

    new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fn, [core.get_aval(v) for v in flat_args])

    ret = (jax.core.ClosedJaxpr(new_jaxpr, consts),) if return_closed else (new_jaxpr, consts)
    return *ret, flat_args, out_tree()

def _transform_jaxpr_output(jaxpr, jaxpr_args, orig_out_struct, out_transform):
    def eval_fn(literals, *args):
        output = use_implicit_args(partial(core.eval_jaxpr, jaxpr.jaxpr))(literals, *args)
        unflattened_output = orig_out_struct.unflatten(output)
        return out_transform(unflattened_output)

    wrapped = lu.wrap_init(eval_fn)

    flat_args, in_tree = jax.tree_util.tree_flatten((jaxpr.literals, *jaxpr_args))
    flat_fn, out_tree = flatten_fun_nokwargs(wrapped, in_tree)
    new_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fn, [core.get_aval(v) for v in flat_args])

    return jax.core.ClosedJaxpr(new_jaxpr, consts), out_tree()

def _match_branches(branches, arg_vals):
    out_avals = []
    new_jaxprs = []
    flat_inputs = None
    branch_out_struct = None
    for branch in branches:
        new_jaxpr, flat_inputs, branch_out_struct = wrap_jaxpr(branch, arg_vals)
        new_jaxprs.append((new_jaxpr, branch_out_struct))
        out_avals.append(
            branch_out_struct.unflatten(
                jax.eval_shape(
                    partial(core.eval_jaxpr, new_jaxpr.jaxpr), new_jaxpr.literals, *flat_inputs
                )
            )
        )

    out_transforms = utils.get_common_prefix_transforms(out_avals)
    new_branches = []
    out_struct = None
    for (new_jaxpr, orig_out_struct), transform in zip(new_jaxprs, out_transforms):
        new_jaxpr, out_struct = _transform_jaxpr_output(new_jaxpr, flat_inputs, orig_out_struct, transform)
        new_branches.append(new_jaxpr)

    return tuple(new_branches), out_struct, flat_inputs

def _handle_cond(primitive, *vals, params):
    cond_val, *arg_vals = vals
    subfuns, bind_params = primitive.get_bind_params(params)

    new_branches, out_struct, flat_inputs = _match_branches(params['branches'], arg_vals)
    bind_params['branches'] = new_branches
    bind_params['linear'] = _broadcast_tuple(params['linear'], arg_vals)

    outs = primitive.bind(*subfuns, cond_val, *flat_inputs, **bind_params)
    return jax.tree_util.tree_unflatten(out_struct, outs)

def _handle_remat2(primitive, *vals, params):
    subfuns, bind_params = primitive.get_bind_params(params)
    new_jaxpr, consts, flat_inputs, out_tree = wrap_jaxpr(bind_params['jaxpr'], vals, return_closed=False)
    new_jaxpr = pe.convert_constvars_jaxpr(new_jaxpr)
    bind_params['jaxpr'] = new_jaxpr
    outs = primitive.bind(*subfuns, *consts, *flat_inputs, **bind_params)
    return jax.tree_util.tree_unflatten(out_tree, outs)


def _handle_pjit(primitive, *vals, params):
    new_jaxpr, flat_inputs, out_tree = wrap_jaxpr(params['jaxpr'], vals)
    donated_invars = _broadcast_tuple(params['donated_invars'], vals)
    in_shardings = _broadcast_tuple(params['in_shardings'], vals)
    out_shardings = _broadcast_tuple(params['out_shardings'], out_tree)

    subfuns, bind_params = primitive.get_bind_params(params)
    bind_params['jaxpr'] = new_jaxpr
    bind_params['donated_invars'] = donated_invars
    bind_params['in_shardings'] = in_shardings
    bind_params['out_shardings'] = out_shardings
    outs = primitive.bind(*subfuns, *flat_inputs, **bind_params)
    return jax.tree_util.tree_unflatten(out_tree, outs)

_default_handlers = {
    'cond': _handle_cond,
    'remat2': _handle_remat2,
    'pjit': _handle_pjit,
}

def materialize_handler(primitive, *vals, params):
    vals = _materialize_all(vals)
    subfuns, bind_params = primitive.get_bind_params(params)
    result = use_implicit_args(primitive.bind)(*subfuns, *vals, **bind_params)
    return result

def _broadcast_tuple(t, trees):
    if isinstance(trees, jax.tree_util.PyTreeDef):
        trees = jax.tree_util.tree_unflatten(trees, range(trees.num_leaves))
    assert len(t) == len(trees)
    return tuple(chain.from_iterable(
        (tuple_val for _ in jax.tree_util.tree_leaves(tree))
        for tuple_val, tree in zip(t, trees)
    ))

def _get_materialization_aval(imp_arr):
    with _aval_discovery_context():
        result = jax.eval_shape(
            partial(utils.materialize_nested, full=True),
            imp_arr
        )
    return result

_AVAL_ERROR_MESSAGE = (
    '{} was not set during initialization. Shape and dtype may be set by:'
    '\n\t1. Directly passing them as keyword arguments to ImplicitArray instances'
    '\n\t2. Overriding the default_shape/default_dtype class attributes'
    '\n\t3. Overriding the compute_shape/compute_dtype methods'
    '\n\t4. Overriding __post_init__ and setting their values there'
    '\n\t5. None of the above, in which case `materialize()` will be called in an attempt to infer them.'
    ' If their values are required in order to compute the materialization this will be unsuccessful.'
)
