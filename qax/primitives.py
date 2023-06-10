from collections import defaultdict
from functools import wraps
from itertools import count

import jax

from plum import Dispatcher, Function

_dispatch = Dispatcher()

_primitive_ids = count()

def get_lax_primitive_by_name(name):
    return getattr(jax.lax, f'{name}_p')

def get_primitive_handler(primitive):
    handler = _dispatch.functions.get(primitive)
    if handler is None:
        def _not_impl_handler(*args, **kwargs):
            return NotImplemented
        _not_impl_handler.__doc__ = 'Default handler for {primitive.name}'
        handler = Function(_not_impl_handler)
        handler.register(_not_impl_handler)
        handler.__name__ = f'{primitive.name}_{next(_primitive_ids)}'
        _dispatch.functions[primitive] = handler
    return handler

def primitive_handler(primitives):
    if isinstance(primitives, (str, jax.core.Primitive)):
        primitives = [primitives]
    def decorator(fn):
        for primitive in primitives:
            if isinstance(primitive, str):
                primitive = get_lax_primitive_by_name(primitive)
            handler = get_primitive_handler(primitive)
            if isinstance(fn, Function):
                fn = fn._f
            handler.register(fn)

    return decorator

def default_handler(primitive, *args, **params):
    subfuns, bind_params = primitive.get_bind_params(params)
    return primitive.bind(*subfuns, *args, **bind_params)
