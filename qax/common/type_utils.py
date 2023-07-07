from typing import Any, Optional, Tuple

from beartype.vale import IsInstance
from plum import dispatch, parametric

class _ComplementMeta(type):
    def __instancecheck__(self, x):
        a, b = self.type_parameter
        return (
            a is None or (
                isinstance(x, a) and not isinstance(x, b)
            )
        )

@parametric
class Complement(metaclass=_ComplementMeta):
    """
    Relative complement
    I.e. Complement[A, B] = A - B
    """
    @classmethod
    @dispatch
    def __init_type_parameter__(
        cls,
        a: Optional[Any],
        b: Optional[Any],
    ):
        return a, b

    @classmethod
    @dispatch
    def __le_type_parameter__(
        cls,
        left: Tuple[Optional[Any], Optional[Any]],
        right: Tuple[Optional[Any], Optional[Any]],
    ):
        a_left, b_left = left
        a_right, b_right = right

        return issubclass(a_left, a_right) and issubclass(b_right, b_left)

