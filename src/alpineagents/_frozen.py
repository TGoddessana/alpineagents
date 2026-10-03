"""Read-only JSON containers: ``FrozenDict`` and ``FrozenList``.

History is the only source of truth for a State, so the values inside its entries (a tool call's arguments, a
``finish`` answer, ``state.extra_data``) must not change after they are recorded. A frozen dataclass only stops
rebinding a field, not ``entry.content["x"] = 1``, so these containers refuse to be changed.

- ``FrozenDict`` is a ``dict`` and ``FrozenList`` is a ``list``. So ``isinstance(x, dict)``, ``==`` with plain
  values, ``json.dumps``, ``**args``, iteration and indexing all work as before. Only the methods that change them
  raise ``TypeError``.
- ``freeze`` converts a value deeply (dict to ``FrozenDict``, list and tuple to ``FrozenList``) and ``thaw`` converts
  it back to plain ``dict`` and ``list``.
- Pickle and ``copy.deepcopy`` work and give frozen containers again, so a deep copy of an entry keeps its values
  read-only. To get an editable copy, use ``thaw``.
- Reading a slice, ``+``, ``*``, ``|`` and ``.copy()`` give plain (editable) ``list``/``dict`` values: they are new
  values, not the frozen one.

Not exported from the package: users meet these containers as the type of ``call.args``, ``state.extra_data`` and
so on, and only read them. ``JSON`` below is what such a value is made of.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, NoReturn, TypeAlias, TypeVar

from .errors import fix_message

__all__ = ["FrozenDict", "FrozenList", "JSON", "freeze", "thaw"]

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")

#: What a JSON value is made of. The frozen containers are ``list``/``dict`` subclasses, so a plain value and a
#: frozen one both fit this type.
JSON: TypeAlias = "None | bool | int | float | str | list[Any] | dict[str, Any]"


def _read_only(kind: str, operation: str) -> TypeError:
    """The error every mutating method raises."""
    plain = "dict" if kind == "FrozenDict" else "list"
    return TypeError(
        fix_message(
            f"this {kind} is read-only (it was recorded in a State's history), so {operation} is not allowed",
            f"make a plain copy and change that: {plain}(value) copies the top level, and "
            "json.loads(json.dumps(value)) copies every level. "
            "A State's extra_data is changed with state.edit_extra_data()",
            "with state.edit_extra_data() as data:\n    data['notes'] = ['checked b.py']",
        )
    )


class FrozenDict(dict[K, V]):
    """A ``dict`` that cannot be changed, with every value frozen too (``FrozenDict({"a": [1]})["a"]`` is a
    ``FrozenList``). Reading works as with ``dict``; changing raises ``TypeError``.

    Pickle and ``copy.deepcopy`` rebuild a ``FrozenDict``, so a deep copy is still read-only. ``thaw`` makes an
    editable copy.
    """

    __slots__ = ()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # dict.__init__ and dict.__setitem__ do not call our overrides, so this fills it. Values are frozen here
        # so a FrozenDict is deep-frozen however it is built.
        super().__init__(*args, **kwargs)
        for key, value in dict.items(self):
            frozen = freeze(value)
            if frozen is not value:
                dict.__setitem__(self, key, frozen)

    @classmethod
    def fromkeys(cls, iterable: Iterable[K], value: Any = None) -> FrozenDict[K, Any]:  # type: ignore[override]
        return cls(dict.fromkeys(iterable, value))

    def __reduce__(self) -> Any:
        # The default pickle protocol would fill a new object with __setitem__, which raises. Rebuilding from a
        # plain dict also makes copy.copy and copy.deepcopy give a frozen result.
        return (type(self), (dict(self),))

    def __setitem__(self, key: K, value: V) -> NoReturn:
        raise _read_only("FrozenDict", f"setting key {key!r}")

    def __delitem__(self, key: K) -> NoReturn:
        raise _read_only("FrozenDict", f"deleting key {key!r}")

    def __ior__(self, other: Any) -> NoReturn:  # type: ignore[override,misc]
        raise _read_only("FrozenDict", "|=")

    def update(self, *args: Any, **kwargs: Any) -> NoReturn:  # type: ignore[override]
        raise _read_only("FrozenDict", "update()")

    def pop(self, *args: Any, **kwargs: Any) -> NoReturn:  # type: ignore[override]
        raise _read_only("FrozenDict", "pop()")

    def popitem(self) -> NoReturn:
        raise _read_only("FrozenDict", "popitem()")

    def clear(self) -> NoReturn:
        raise _read_only("FrozenDict", "clear()")

    def setdefault(self, *args: Any, **kwargs: Any) -> NoReturn:  # type: ignore[override]
        raise _read_only("FrozenDict", "setdefault()")


class FrozenList(list[T]):
    """A ``list`` that cannot be changed, with every item frozen too. Reading works as with ``list``; changing
    raises ``TypeError``.

    Pickle and ``copy.deepcopy`` rebuild a ``FrozenList``, so a deep copy is still read-only. ``thaw`` makes an
    editable copy.
    """

    __slots__ = ()

    def __init__(self, iterable: Iterable[T] = (), /) -> None:
        super().__init__(iterable)
        for index, item in enumerate(list.__iter__(self)):
            frozen = freeze(item)
            if frozen is not item:
                list.__setitem__(self, index, frozen)

    def __reduce__(self) -> Any:
        return (type(self), (list(self),))

    def __setitem__(self, index: Any, value: Any) -> NoReturn:
        raise _read_only("FrozenList", "assigning to an item or a slice")

    def __delitem__(self, index: Any) -> NoReturn:
        raise _read_only("FrozenList", "deleting an item or a slice")

    def __iadd__(self, other: Any) -> NoReturn:  # type: ignore[override,misc]
        raise _read_only("FrozenList", "+=")

    def __imul__(self, other: Any) -> NoReturn:  # type: ignore[override,misc]
        raise _read_only("FrozenList", "*=")

    def append(self, item: Any) -> NoReturn:
        raise _read_only("FrozenList", "append()")

    def extend(self, items: Any) -> NoReturn:
        raise _read_only("FrozenList", "extend()")

    def insert(self, index: Any, item: Any) -> NoReturn:
        raise _read_only("FrozenList", "insert()")

    def remove(self, item: Any) -> NoReturn:
        raise _read_only("FrozenList", "remove()")

    def pop(self, index: Any = -1) -> NoReturn:
        raise _read_only("FrozenList", "pop()")

    def clear(self) -> NoReturn:
        raise _read_only("FrozenList", "clear()")

    def sort(self, *args: Any, **kwargs: Any) -> NoReturn:
        raise _read_only("FrozenList", "sort() (use sorted(value) for a sorted copy)")

    def reverse(self) -> NoReturn:
        raise _read_only("FrozenList", "reverse() (use reversed(value) or value[::-1])")


def freeze(value: Any) -> Any:
    """``value`` with every dict (any ``Mapping``) turned into a ``FrozenDict`` and every list or tuple into a
    ``FrozenList``, at every level. Scalars and other objects are returned as they are. A value that is already
    frozen is returned itself (frozen containers are deep-frozen by construction)."""
    if isinstance(value, (FrozenDict, FrozenList)):
        return value
    if isinstance(value, Mapping):
        return FrozenDict(value)
    if isinstance(value, (list, tuple)):
        return FrozenList(value)
    return value


def thaw(value: Any) -> Any:
    """``value`` with every ``Mapping`` turned into a plain ``dict`` and every list or tuple into a plain ``list``,
    at every level: an editable deep copy of a frozen value."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw(item) for item in value]
    return value
