from abc import ABC, abstractmethod
from functools import partial, wraps
from itertools import chain
import warnings

import jax
from jax.api_util import flatten_fun, flatten_fun_nokwargs
from jax import core
import jax.linear_util as lu
import jax.interpreters.partial_eval as pe
from jax.tree_util import register_pytree_with_keys_class

from . import constants
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
    @wraps(f)
    def inner(*args, **kwargs):
        flat_args, in_tree = utils.tree_flatten_with_implicit((args, kwargs))
        f_flat, out_tree = flatten_fun(lu.wrap_init(f), in_tree)
        outs_flat = _use_implicit_flat(f_flat)(*flat_args)
        return jax.tree_util.tree_unflatten(out_tree(), outs_flat)
    return inner

class ImplicitArray(ABC):
    op_registry = None
    commute_ops = True

    def __init__(self, shape, dtype):
        self._shape = shape
        self._dtype = dtype

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, new_shape):
        self._shape = new_shape

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, new_dtype):
        self._dtype = new_dtype

    @property
    def aval(self):
        return core.ShapedArray(self.shape, self.dtype)

    @classmethod
    def default_handler(cls, primitive, *args, params=None):
        if params is None:
            params = {}
        return materialize_handler(primitive, *args, params=params)

    def _materialize(self):
        wrapped = lu.wrap_init(type(self).materialize)
        flat, in_tree = _flatten_one_implicit_layer((self,))
        flat_fn, out_tree = flatten_fun_nokwargs(wrapped, in_tree)
        out_flat = use_implicit_args(flat_fn.call_wrapped)(*flat)
        result = jax.tree_util.tree_unflatten(out_tree(), out_flat)
        return result

    @abstractmethod
    def materialize(self):
        pass

    def flatten(self):
        """
        To control tree flattening, override this method rather than `tree_flatten`.
        The auxiliary data returned will have the shape and dtype added to it.
        """
        return [], ()

    def unflatten(self, aux_data, children):
        """
        To control unflattening override this method rather than `tree_unflatten`.
        This will avoid __init__ being called during tree unflattening.
        """
        pass

    def tree_flatten_with_keys(self):
        children, aux_data = self.flatten()
        aux_data = (self.shape, self.dtype, aux_data)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        shape, dtype, aux_data = aux_data
        obj.shape = shape
        obj.dtype = dtype
        obj.unflatten(aux_data, children)
        return obj

    def handle_primitive(self, primitive, *args, params):
        if primitive.name not in self.op_registry:
            return NotImplemented

        handler = lu.wrap_init(partial(self.op_registry[primitive.name], primitive))
        use_params = params
        maybe_kwargs = {'params': params} if params else {}

        if len(args) == 2 and self.commute_ops:
            args, use_params = _maybe_swap_args(primitive.name, args, use_params)

        flat_args, in_tree = _flatten_one_implicit_layer((args, maybe_kwargs))
        flat_handler, out_tree = flatten_fun(handler, in_tree)

        result = use_implicit_args(flat_handler.call_wrapped)(*flat_args)
        return jax.tree_util.tree_unflatten(out_tree(), result)

    def __init_subclass__(cls, commute_ops=True, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.op_registry = {}

        if 'tree_unflatten' in cls.__dict__:
            raise TypeError(
                'In order to define an ImplicitArray\'s unflattening behavior,'
                ' define the `unflatten` method instead of `tree_unflatten`.'
            )

        if 'tree_flatten_with_keys' in cls.__dict__:
            raise TypeError(
                'In order to define an ImplicitArray\'s flattening behavior,'
                ' define the `flatten` method instead of `tree_flatten_with_keys`.'
            )

        core.pytype_aval_mappings[cls] = lambda x: x.aval

        register_pytree_with_keys_class(cls)
        for imp_op in cls.__dict__.values():
            if not isinstance(imp_op, _ImplicitOp):
                continue
            for name in imp_op.names:
                cls.op_registry[name] = partial(imp_op, cls)

def _materialize_all(it):
    return [val._materialize() if isinstance(val, ImplicitArray) else val for val in it]

class _ImplicitOp:
    def __init__(self, names, op_fn):
        self.names = names
        self.op_fn = op_fn

    def __call__(self, cls, *args, **kwargs):
        if isinstance(self.op_fn, classmethod):
            return self.op_fn.__func__(cls, *args, **kwargs)
        return self.op_fn(*args, **kwargs)

def implicit_op(op_names):
    if isinstance(op_names, str):
        op_names = [op_names]
    def decorator(op_fn):
        return _ImplicitOp(op_names, op_fn)
    return decorator

def _maybe_swap_args(op_name, args, params):
    if isinstance(args[0], ImplicitArray):
        return args, params
    if op_name in constants.COMMUTATIVE_OPS:
        return args[::-1], params

    if op_name != 'dot_general':
        return args, params

    new_params = {**params}
    new_params['dimension_numbers'] = (

        params['dimension_numbers'][0][::-1],
        params['dimension_numbers'][1][::-1],
    )

    return args[::-1], new_params

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

class ImplicitArrayTrace(core.Trace):
    pure = lift = lambda self, val: ImplicitArrayTracer(self, val)

    def process_primitive(self, primitive, tracers, params):
        outs = NotImplemented
        vals = [t.value for t in tracers]
        implicit_idx = next(i for i, v in enumerate(vals) if isinstance(v, ImplicitArray))
        if primitive.name in _default_handlers:
            outs = _default_handlers[primitive.name](primitive, *vals, params=params)
        else:
            kind = type(vals[implicit_idx])
            for idx, val in enumerate(vals[implicit_idx + 1:], implicit_idx + 1):
                if isinstance(val, ImplicitArray) and not isinstance(val, kind):
                    warnings.warn(f'Encountered {primitive.name} with heterogenous implicit inputs. Argument of type {type(val)} will be materialized.')
                    vals[idx] = val._materialize()

            outs = vals[implicit_idx].handle_primitive(primitive, *vals, params=params)
            if outs is NotImplemented:
                warnings.warn(f'Primitive {primitive.name} was not handled by class {vals[implicit_idx].__class__.__name__}, so implicit args will be materialized.')

        if outs is NotImplemented:
            outs = vals[implicit_idx].default_handler(primitive, *vals, params=params)

        if primitive.multiple_results:
            return [ImplicitArrayTracer(self, out) for out in outs]
        return ImplicitArrayTracer(self, outs)

def _handle_cond(primitive, *vals, params):
    cond_val, *arg_vals = vals
    subfuns, bind_params = primitive.get_bind_params(params)

    new_branches = []
    out_struct = None
    flat_inputs = []
    for branch in params['branches']:
        new_jaxpr, flat_inputs, branch_out_struct = wrap_jaxpr(branch, arg_vals)
        new_branches.append(new_jaxpr)
        if out_struct is not None and (branch_out_struct != out_struct):
            warnings.warn(f'Branches of cond have different output shapes, so implicit args will be materialized.')
            return NotImplemented
        out_struct = branch_out_struct

    bind_params['branches'] = tuple(new_branches)
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

def _flatten_one_implicit_layer(tree):
    def is_leaf_below_node(node, x):
        return isinstance(x, ImplicitArray) and x is not node

    def replace_subtree_implicits(node):
        return jax.tree_util.tree_map(lambda _: 1, node, is_leaf=partial(is_leaf_below_node, node))

    prototype = utils.tree_map_with_implicit(replace_subtree_implicits, tree)
    struct = jax.tree_util.tree_structure(prototype)

    leaves = utils.tree_leaves_with_implicit(tree)
    leaves = list(chain.from_iterable(
        jax.tree_util.tree_leaves(leaf, is_leaf=partial(is_leaf_below_node, leaf))
        if isinstance(leaf, ImplicitArray) else
        [leaf] for leaf in leaves
    ))
    return leaves, struct
