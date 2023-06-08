import warnings

import jax
from jax.core import get_aval, Tracer, full_lower

from .implicit_array import ImplicitArray



class ImplicitArrayTracer(Tracer):
    def __init__(self, trace, value):
        super().__init__(trace)
        self.value = value

    @property
    def aval(self):
        if isinstance(self.value, ImplicitArray):
            return jax.ShapedArray(self.value.shape, self.value.dtype)
        return get_aval(self.value)

    def full_lower(self):
        if isinstance(self.value, ImplicitArray):
            return self

        return full_lower(self.value)

class ImplicitArrayTrace(Trace):
    pure = lift = lambda self, val: ImplicitArrayTracer(self, val)

    def process_primitive(self, primitive, tracers, params):
        vals = [t.value for t in tracers]
        n_implicit = sum(isinstance(v, ImplicitArray) for v in vals)
        assert 1 <= n_implicit <= 2
        if n_implicit == 2:
            warnings.warn(f'Encountered op {primitive.name} with two implicit inputs so second will be materialized.')
            vals[1] = vals[1].materialize()
