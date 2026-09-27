"""@tool: turns Python functions and methods into tools the model can call.

The contract is in ARCHITECTURE.md "Tool contract". This module only does schema generation, argument
validation, State injection and result-to-string conversion. Execution order, threads and exception handling
belong to Agent (``_runner.py``).
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import inspect
import json
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints, overload

from pydantic import BaseModel, ConfigDict, Field, PydanticUserError, ValidationError, create_model
from pydantic_core import to_jsonable_python

# On 3.11 and 3.12, typing.is_typeddict does not recognize classes made with typing_extensions.TypedDict.
# typing_extensions is a required dependency of pydantic.
from typing_extensions import is_typeddict

from .errors import ToolError, ToolInputError, fix_message
from .state import State
from .types import INVALID_ARGS_KEY, TRUNCATED_ARGS_MESSAGE, ToolSpec

__all__ = ["tool", "Tool", "ToolMap", "collect_tools", "hint_values", "SUPPORTED_TYPES_TEXT", "DONE"]

#: Text sent to the model when a tool returns ``None``.
DONE = "(done)"

#: List shown in the unsupported-type error.
SUPPORTED_TYPES_TEXT = (
    "str, int, float, bool, Literal[...], Enum, list[T], dict[str, T], T | None, "
    "dataclass, TypedDict, Pydantic model"
)

_BAD_PARAM_KINDS = (
    inspect.Parameter.VAR_POSITIONAL,
    inspect.Parameter.VAR_KEYWORD,
    inspect.Parameter.POSITIONAL_ONLY,
)

_ARG_LINE = re.compile(r"^(\s*)([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*:\s*(.*)$")
#: Google-style section header after Args: (Returns:, Raises:, etc.). Starts with a capital letter
#: to tell it apart from a parameter line.
_SECTION_HEADER = re.compile(r"^\s*[A-Z][A-Za-z ]*:\s*$")

#: Tool names providers accept. The overlap of the OpenAI FunctionDefinition.name rule (a-z, A-Z, 0-9, _, -;
#: up to 64 chars) and the Anthropic tool name rule.
_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Extracts the docstring's first paragraph as the description and the ``Args:`` section as parameter
    descriptions (Google style)."""
    if not doc or not doc.strip():
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()

    # Find the Args: header line by line, regardless of paragraph breaks. This keeps it out of the description
    # when the docstring starts with Args: without a summary, or Args: comes right after the summary line.
    args_at: int | None = None
    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        paragraph_start = i == 0 or not lines[i - 1].strip()
        if stripped == "args:" or (stripped == "args" and paragraph_start):
            args_at = i
            break

    head = lines if args_at is None else lines[:args_at]
    first_paragraph: list[str] = []
    for line in head:
        if not line.strip():
            if first_paragraph:
                break
            continue
        first_paragraph.append(line.strip())
    description = " ".join(first_paragraph).strip()

    param_docs: dict[str, str] = {}
    if args_at is None:
        return description, param_docs

    header_indent = _indent(lines[args_at])
    entry_indent: int | None = None
    current_name: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        if current_name is not None:
            param_docs[current_name] = " ".join(current_lines).strip()

    after_blank = False
    for line in lines[args_at + 1 :]:
        if not line.strip():
            after_blank = True
            continue
        indent = _indent(line)
        # The section ends at another section header at the Args: indent (Returns: etc.),
        # or at a dedented line after a blank line.
        if indent <= header_indent and (_SECTION_HEADER.match(line) or after_blank):
            break
        after_blank = False
        match = _ARG_LINE.match(line)
        if match and (entry_indent is None or indent <= entry_indent):
            _flush()
            entry_indent = indent
            current_name = match.group(2)
            current_lines = [match.group(3)]
        else:
            current_lines.append(line.strip())
    _flush()
    return description, param_docs


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_supported_type(tp: Any, _seen: frozenset[Any] = frozenset()) -> bool:
    """Recursively checks whether ``@tool`` supports the type."""
    if tp in (str, int, float, bool):
        return True
    if tp in _seen:
        return True  # recursive type: pass if already being checked
    origin = get_origin(tp)
    if origin is Literal:
        return True
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        return True
    if origin is list:
        args = get_args(tp)
        return len(args) == 1 and _is_supported_type(args[0], _seen)
    if origin is dict:
        args = get_args(tp)
        return len(args) == 2 and args[0] is str and _is_supported_type(args[1], _seen)
    if origin is Union or origin is UnionType:
        args = get_args(tp)
        non_none = [a for a in args if a is not type(None)]
        return type(None) in args and len(non_none) == 1 and _is_supported_type(non_none[0], _seen)
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        try:
            hints = get_type_hints(tp)
        except NameError:
            return False
        seen = _seen | {tp}
        return all(_is_supported_type(hints[f.name], seen) for f in dataclasses.fields(tp))
    if is_typeddict(tp):
        try:
            hints = get_type_hints(tp)
        except NameError:
            return False
        seen = _seen | {tp}
        return all(_is_supported_type(v, seen) for v in hints.values())
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        seen = _seen | {tp}
        return all(_is_supported_type(f.annotation, seen) for f in tp.model_fields.values())
    return False


def _check_tool_name(name: Any, *, explicit: bool) -> str:
    """Checks the tool name shown to the model. Names a provider would reject are stopped here instead of
    as a 400 on the first think."""
    if isinstance(name, str) and _TOOL_NAME.fullmatch(name):
        return name
    where = f"name={name!r}" if explicit else f"function name {name!r}"
    raise TypeError(
        fix_message(
            f"{where} cannot be used as a tool name. It must be 1-64 characters of letters, digits, _ and -",
            "Give it a valid name with @tool(name=...)",
            '@tool(name="web_search")',
        )
    )


def _resolve_hints(fn: Callable[..., Any]) -> dict[str, Any]:
    """``get_type_hints(fn)``. So that modules importing ``State`` only under ``TYPE_CHECKING`` still resolve,
    if the module has no name ``State``, it retries with the framework's State."""
    try:
        return get_type_hints(fn)
    except NameError:
        if "State" in getattr(fn, "__globals__", {}):
            raise
        return get_type_hints(fn, localns={"State": State})


def _check_stdlib_typeddict(tool_name: str, param: str, tp: Any) -> None:
    """Below Python 3.12, Pydantic rejects ``typing.TypedDict``. Gives a fix instead of that error."""
    if sys.version_info >= (3, 12):
        return
    stack, seen = [tp], set()
    while stack:
        cur = stack.pop()
        if id(cur) in seen:
            continue
        seen.add(id(cur))
        stack.extend(get_args(cur))
        if is_typeddict(cur):
            if type(cur).__module__ == "typing":
                raise TypeError(
                    fix_message(
                        f"Tool {tool_name}, parameter {param}: typing.TypedDict ({cur.__name__}) cannot be used "
                        "on Python 3.11",
                        "Use typing_extensions.TypedDict",
                        "from typing_extensions import TypedDict",
                    )
                )
            stack.extend(get_type_hints(cur).values())
        elif isinstance(cur, type) and dataclasses.is_dataclass(cur):
            stack.extend(get_type_hints(cur).values())
        elif isinstance(cur, type) and issubclass(cur, BaseModel):
            stack.extend(f.annotation for f in cur.model_fields.values())


def hint_values(
    read_only: bool | None,
    destructive: bool | None,
    idempotent: bool | None,
    open_world: bool | None,
) -> tuple[bool, bool, bool, bool]:
    """The four hints, with the values not given filled in. Same rules as MCP tool annotations: a hint not given
    assumes the worst (not read-only, destructive, not idempotent, open world), except that a read-only tool is
    never destructive and always idempotent."""
    read_only = bool(read_only)
    if read_only:
        destructive, idempotent = False, True
    return (
        read_only,
        True if destructive is None else bool(destructive),
        False if idempotent is None else bool(idempotent),
        True if open_world is None else bool(open_world),
    )


def _check_hints(tool_name: str, **hints: Any) -> tuple[bool, bool, bool, bool]:
    """``hint_values`` for ``@tool``: each hint must be a bool or left out, and a read-only tool cannot also be
    destructive or not idempotent."""
    for hint, value in hints.items():
        if value is not None and not isinstance(value, bool):
            raise TypeError(
                fix_message(
                    f"Tool {tool_name}: {hint}= takes True or False (got: {value!r})",
                    "Pass a bool, or leave it out to assume the worst",
                    "@tool(read_only=True, open_world=False)",
                )
            )
    if hints["read_only"]:
        for hint, contradiction in (("destructive", True), ("idempotent", False)):
            if hints[hint] is contradiction:
                raise ValueError(
                    fix_message(
                        f"Tool {tool_name}: read_only=True and {hint}={contradiction} contradict each other. "
                        "A read-only tool changes nothing, so it is never destructive and always idempotent",
                        f"Leave {hint} out (read_only=True sets it), or set read_only=False",
                        "@tool(read_only=True)",
                    )
                )
    return hint_values(**hints)


_HANDLER_EXAMPLE = (
    "def http_errors(error: httpx.HTTPError) -> str:\n"
    '    return f"request failed: {error}"\n'
    "\n"
    "@tool(exception_handler=http_errors)\n"
    "def fetch_url(url: str) -> str: ..."
)


def _handled_types(handler: Any, tool_name: str) -> tuple[type[Exception], ...]:
    """Checks an ``exception_handler`` and returns the exception types it takes, read from the type hint of its
    one parameter (a class or a union of classes, each a subclass of ``Exception``)."""
    where = f"Tool {tool_name}: exception_handler={handler!r}"
    if not callable(handler) or isinstance(handler, type):
        raise TypeError(
            fix_message(
                f"{where} is not a function",
                "Pass a function that takes the exception and returns the message for the model",
                _HANDLER_EXAMPLE,
            )
        )
    if inspect.iscoroutinefunction(handler) or inspect.iscoroutinefunction(getattr(handler, "__call__", None)):
        raise TypeError(
            fix_message(
                f"{where} is async",
                "Write it as a regular def. It only turns the exception into a message",
                _HANDLER_EXAMPLE,
            )
        )
    try:
        params = list(inspect.signature(handler).parameters.values())
    except (TypeError, ValueError):
        params = []
    required = [
        p for p in params
        if p.default is inspect.Parameter.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]
    if len(required) != 1 or required[0].kind is inspect.Parameter.KEYWORD_ONLY:
        raise TypeError(
            fix_message(
                f"{where} must take exactly one argument, the exception",
                "Give it one parameter, with a type hint naming the exceptions it handles",
                _HANDLER_EXAMPLE,
            )
        )
    param = required[0]
    target = handler if inspect.isfunction(handler) or inspect.ismethod(handler) else type(handler).__call__
    try:
        hint = get_type_hints(target).get(param.name)
    except NameError as e:
        raise TypeError(
            fix_message(
                f"{where}: cannot resolve the type hint of {param.name}: {e}",
                "Check that the exception types are imported in the handler's module",
            )
        ) from e
    if hint is None:
        raise TypeError(
            fix_message(
                f"{where} has no type hint on its parameter {param.name}",
                "Annotate it with the exceptions it handles. Use Exception to handle every exception",
                _HANDLER_EXAMPLE,
            )
        )
    types_ = get_args(hint) if get_origin(hint) in (Union, UnionType) else (hint,)
    for tp in types_:
        if not (isinstance(tp, type) and issubclass(tp, BaseException)):
            raise TypeError(
                fix_message(
                    f"{where}: the type hint of {param.name} must name exception classes (got: {hint!r})",
                    "Use an exception class, or several joined with |",
                    "def file_errors(error: FileNotFoundError | PermissionError) -> str: ...",
                )
            )
        if not issubclass(tp, Exception):
            raise TypeError(
                fix_message(
                    f"{where}: {tp.__name__} cannot be handled. KeyboardInterrupt, SystemExit and other "
                    "exceptions that are not Exception subclasses always stop the run",
                    "Handle subclasses of Exception only",
                    _HANDLER_EXAMPLE,
                )
            )
    return tuple(types_)


_MISUSE_KIND_TEXT = {
    "missing": "required argument is missing",
    "extra_forbidden": "unknown argument",
}


def _describe_validation_error(err: dict[str, Any]) -> str:
    loc = ".".join(str(p) for p in err["loc"]) or "(value)"
    text = _MISUSE_KIND_TEXT.get(err["type"], err["msg"])
    return f"{loc}: {text}"


class Tool:
    """A tool made by ``@tool``. Calling it runs the original function as is, without validation.

    A ``@tool`` method in a class becomes a bound tool when accessed on an object (``fs.read_file``).
    The attributes do not change after creation.
    """

    name: str
    """The name the model calls. Defaults to the function name."""
    description: str
    """What the model reads. Defaults to the docstring's first paragraph, or an empty string."""
    parallel: bool
    """``True`` runs the tool together with the turn's other calls. ``False`` runs it alone after they finish."""
    input_schema: dict[str, Any]
    """The JSON Schema object shown to the model. ``State`` parameters and ``self`` are left out, ``Args:``
    descriptions become property descriptions, and parameters with defaults are optional."""
    is_async: bool
    """``True`` if the original function is ``async def``."""
    exception_handler: Callable[[Any], str] | None
    """Turns the exceptions named in its parameter's type hint into error results the model sees. ``None`` if not
    given."""
    read_only: bool
    """The tool does not change its environment. ``False`` unless given. The model does not see the hints; they are
    for your own code, such as a loop that asks before calls that are not read-only."""
    destructive: bool
    """The tool may delete or overwrite something. ``True`` unless given, and ``False`` for a read-only tool."""
    idempotent: bool
    """Calling it again with the same arguments has no further effect. ``False`` unless given, and ``True`` for a
    read-only tool."""
    open_world: bool
    """The tool reaches outside its own domain, such as the web, a shell or another system. ``True`` unless given."""
    # Internal: the original function, whether it is a method in a class (first parameter is ``self`` with no
    # type hint), and the bound object (None if unbound).
    fn: Callable[..., Any]
    needs_self: bool
    bound_to: Any

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
        parallel: bool = True,
        exception_handler: Callable[[Any], str] | None = None,
        read_only: bool | None = None,
        destructive: bool | None = None,
        idempotent: bool | None = None,
        open_world: bool | None = None,
    ) -> None:
        """Usually created with ``@tool``. A bad tool raises here, before any model call.

        Args:
            fn: The function or method. Every parameter needs a type hint.
            name: The tool name. Must match ``[A-Za-z0-9_-]{1,64}``. Defaults to the function name.
            description: Defaults to the docstring's first paragraph.
            parallel: ``False`` runs the tool alone after the turn's other calls finish.
            exception_handler: See ``tool``.
            read_only: See ``tool``.
            destructive: See ``tool``.
            idempotent: See ``tool``.
            open_world: See ``tool``.

        Raises:
            TypeError: The name is invalid, a parameter has no type hint or an unsupported type, the function
                takes ``*args``, ``**kwargs`` or positional-only parameters, ``exception_handler`` is not a
                function with one parameter whose type hint names ``Exception`` subclasses, or a hint is not a bool.
            ValueError: ``read_only=True`` with ``destructive=True`` or ``idempotent=False``.
        """
        # - If the first parameter is named ``self`` with no type hint, it is a method (``needs_self=True``, left
        #   out of the schema).
        # - Unsupported types are checked recursively; the error lists SUPPORTED_TYPES_TEXT.
        # - Hints are resolved with ``typing.get_type_hints(fn)`` (string hints and
        #   ``from __future__ import annotations`` supported). If that fails because the module has no ``State``
        #   (``TYPE_CHECKING``-only import), it resolves again with the framework's State.
        # - Parameters typed ``State`` (subclasses included) are hidden from the model and injected at run time.
        # - The Pydantic model for argument validation is built once here (``pydantic.create_model``,
        #   ``extra="forbid"``). Field names are internal names and parameter names are aliases (so they do not
        #   clash with BaseModel attribute names).
        self.fn = fn
        self.name = _check_tool_name(fn.__name__ if name is None else name, explicit=name is not None)
        self.parallel = parallel
        self.is_async = inspect.iscoroutinefunction(fn)
        self.bound_to = None
        # The options as given, for copy.
        self._options: dict[str, Any] = {
            "name": name, "description": description, "parallel": parallel, "exception_handler": exception_handler,
            "read_only": read_only, "destructive": destructive, "idempotent": idempotent, "open_world": open_world,
        }
        self.read_only, self.destructive, self.idempotent, self.open_world = _check_hints(
            self.name, read_only=read_only, destructive=destructive, idempotent=idempotent, open_world=open_world
        )
        self.exception_handler = exception_handler
        self._handles: tuple[type[Exception], ...] = (
            () if exception_handler is None else _handled_types(exception_handler, self.name)
        )

        doc_description, arg_docs = _parse_docstring(fn.__doc__)
        self.description = description if description is not None else doc_description

        sig = inspect.signature(fn)
        params = list(sig.parameters.values())

        self.needs_self = (
            bool(params) and params[0].name == "self" and params[0].annotation is inspect.Parameter.empty
        )
        positional = params[1:] if self.needs_self else params

        for p in positional:
            if p.kind in _BAD_PARAM_KINDS:
                raise TypeError(
                    fix_message(
                        f"Tool {self.name}: parameter {p.name} is of an unsupported kind",
                        "Use only named parameters, without *args, **kwargs or positional-only parameters",
                    )
                )

        try:
            hints = _resolve_hints(fn)
        except NameError as e:
            raise TypeError(
                fix_message(
                    f"Tool {self.name}: cannot resolve type hints: {e}",
                    "Check that the types are imported in this function's module",
                )
            ) from e

        state_params: list[str] = []
        fields: dict[str, tuple[Any, Any]] = {}
        param_names: list[tuple[str, str]] = []
        for p in positional:
            if p.name not in hints:
                raise TypeError(
                    fix_message(
                        f"Tool {self.name}: parameter {p.name} has no type hint",
                        "Add a type hint. To receive State, use state: State",
                        f"@tool\ndef {self.name}({p.name}: str): ...",
                    )
                )
            tp = hints[p.name]
            if isinstance(tp, type) and issubclass(tp, State):
                state_params.append(p.name)
                continue
            if not _is_supported_type(tp):
                raise TypeError(
                    fix_message(
                        f"Tool {self.name}: parameter {p.name} has unsupported type {tp!r}",
                        f"Only these types can be used: {SUPPORTED_TYPES_TEXT}",
                    )
                )
            _check_stdlib_typeddict(self.name, p.name, tp)
            # Pydantic field names are internal names (a0, a1, ...) and parameter names are aliases.
            # This keeps parameter names like model_config, model_dump, _x or json from clashing with
            # BaseModel attributes. The schema shown to the model and validation use the alias (parameter name).
            field_kwargs: dict[str, Any] = {"alias": p.name}
            if p.name in arg_docs:
                field_kwargs["description"] = arg_docs[p.name]
            field_kwargs["default"] = ... if p.default is inspect.Parameter.empty else p.default
            internal = f"a{len(fields)}"
            fields[internal] = (tp, Field(**field_kwargs))
            param_names.append((p.name, internal))

        self._state_params: tuple[str, ...] = tuple(state_params)
        #: (parameter name, internal Pydantic field name) pairs.
        self._param_names: tuple[tuple[str, str], ...] = tuple(param_names)
        try:
            self._model: type[BaseModel] = create_model(
                f"{self.name}_input",
                __config__=ConfigDict(extra="forbid"),
                **fields,
            )
            schema = self._model.model_json_schema(by_alias=True)
        except (PydanticUserError, NameError, ValueError, TypeError) as e:
            raise TypeError(
                fix_message(
                    f"Tool {self.name}: cannot build the input schema: {e}",
                    f"Pick parameter types from: {SUPPORTED_TYPES_TEXT}",
                )
            ) from e
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            if isinstance(prop, dict):
                prop.pop("title", None)
        self.input_schema = schema

    def __get__(self, obj: Any, objtype: type | None = None) -> Tool:
        """Descriptor. Accessed as ``obj.method``, returns a new Tool bound to ``obj`` (same schema).

        If ``obj`` is ``None`` (accessed on the class), returns itself (unbound).
        """
        if obj is None:
            return self
        bound = copy.copy(self)
        bound.bound_to = obj
        return bound

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the original function as is (filling in ``self`` if bound). No validation."""
        if self.needs_self and self.bound_to is not None:
            return self.fn(self.bound_to, *args, **kwargs)
        return self.fn(*args, **kwargs)

    @property
    def spec(self) -> ToolSpec:
        """The tool as the model sees it: ``ToolSpec(name, description, input_schema)``."""
        return ToolSpec(self.name, self.description, self.input_schema)

    def prepare(self, args: Mapping[str, Any]) -> dict[str, Any]:
        """Validates the model's arguments and converts them to Python values (State parameters excluded).

        - On mismatch, ``ToolInputError(message)``. The message is short so the model can fix it: which
          argument is wrong and why (a Pydantic error summary). Covers unknown arguments, missing required
          arguments and type mismatches.
          If ``args`` has ``INVALID_ARGS_KEY``: "arguments are not valid JSON: {raw}" --
          except when the value is ``TRUNCATED_ARGS_MESSAGE`` (the reply was cut off so the arguments are
          incomplete), which is used as is without wrapping.
        - On success, a dict of parameter name → value. Values are taken straight from the validated Pydantic
          model's attributes (dataclass and Pydantic model arguments as objects of that type, omitted arguments
          as their defaults).
        """
        if INVALID_ARGS_KEY in args:
            raw = args[INVALID_ARGS_KEY]
            if raw == TRUNCATED_ARGS_MESSAGE:
                raise ToolInputError(raw)
            raise ToolInputError(f"arguments are not valid JSON: {raw}")
        try:
            validated = self._model.model_validate(dict(args))
        except ValidationError as e:
            message = "; ".join(_describe_validation_error(err) for err in e.errors())
            raise ToolInputError(message) from e
        return {name: getattr(validated, internal) for name, internal in self._param_names}

    def invoke(self, kwargs: Mapping[str, Any], state: Any) -> Any:
        """Fills State parameters (``state``) into ``prepare``'s result and calls the original function.

        Also fills ``self`` if bound. Returns the return value for a sync function, or a coroutine for
        ``async def`` (awaiting it is Agent's job). An exception that ``exception_handler`` takes becomes a
        ``ToolError`` with the handler's message (the original in ``__cause__``); any other exception propagates
        as is.
        """
        full_kwargs = dict(kwargs)
        for state_param in self._state_params:
            full_kwargs[state_param] = state
        args = (self.bound_to,) if self.needs_self else ()
        if self.is_async:
            coroutine = self.fn(*args, **full_kwargs)
            return self._handle_async(coroutine) if self._handles else coroutine
        try:
            return self.fn(*args, **full_kwargs)
        except Exception as error:
            self._handle(error)
            raise

    async def _handle_async(self, coroutine: Any) -> Any:
        try:
            return await coroutine
        except Exception as error:
            self._handle(error)
            raise

    def _handle(self, error: Exception) -> None:
        """Called in an ``except`` block. Raises ``ToolError`` if ``exception_handler`` takes ``error``; returns
        otherwise, and the caller re-raises ``error``. A ``ToolError`` is never passed to the handler."""
        if isinstance(error, ToolError) or not isinstance(error, self._handles):
            return
        assert self.exception_handler is not None
        message = self.exception_handler(error)
        if not isinstance(message, str) or not message.strip():
            raise TypeError(
                fix_message(
                    f"Tool {self.name}: exception_handler {self.exception_handler!r} returned {message!r} for "
                    f"{type(error).__name__}, not a message",
                    "Return a non-empty string, the message the model is told. To let the exception stop the run, "
                    "raise it instead",
                    _HANDLER_EXAMPLE,
                )
            ) from error
        raise ToolError(message) from error

    def copy(self, **changes: Any) -> Tool:
        """Returns a new Tool with some options changed. The original does not change, and a bound tool stays bound.

        Args:
            **changes: Any ``@tool`` option: ``name``, ``description``, ``parallel``, ``exception_handler``,
                ``read_only``, ``destructive``, ``idempotent`` or ``open_world``.

        Returns:
            The new Tool.

        Raises:
            TypeError: A name in ``changes`` is not an option, or a new value fails the same checks as ``@tool``.
            ValueError: The new hints contradict each other, as in ``@tool``.

        Example:
            ```python
            strict_fetch = fetch_url.copy(exception_handler=None)  # every exception stops the run
            ```
        """
        unknown = [key for key in changes if key not in self._options]
        if unknown:
            raise TypeError(
                fix_message(
                    f"copy() got unknown options: {', '.join(unknown)}",
                    f"Options you can change: {', '.join(self._options)}",
                    "fetch_url.copy(exception_handler=None)",
                )
            )
        new = Tool(self.fn, **{**self._options, **changes})
        new.bound_to = self.bound_to
        return new

    @staticmethod
    def format_result(value: Any) -> str:
        """Converts a return value into the string sent to the model.

        - ``str`` as is, ``None`` as ``"(done)"``
        - Anything else (dict, list, numbers, bool, dataclass, Pydantic model, combinations of these) goes
          through ``pydantic_core.to_jsonable_python`` and then ``json.dumps(..., ensure_ascii=False)``
        - If it cannot become JSON, ``TypeError`` (fix: return a str, dict, list, dataclass or Pydantic model).
          This exception is treated like any tool exception.
        """
        if value is None:
            return DONE
        if isinstance(value, str):
            return value
        try:
            jsonable = to_jsonable_python(value)
            return json.dumps(jsonable, ensure_ascii=False)
        except Exception as e:
            raise TypeError(
                fix_message(
                    f"Cannot convert tool return value {value!r} to JSON",
                    "Return a str, dict, list, dataclass or Pydantic model",
                )
            ) from e

    @staticmethod
    def is_error_result(value: Any) -> bool:
        """Whether a return value is an error result (sent to the model with ``is_error``). Always false for
        ``@tool`` functions: a failure the model should handle is returned as a plain string."""
        return False

    def __repr__(self) -> str:
        bound = f", bound_to={self.bound_to!r}" if self.bound_to is not None else ""
        return f"Tool({self.name!r}{bound})"


@overload
def tool(fn: Callable[..., Any], /) -> Tool: ...
@overload
def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    parallel: bool = True,
    exception_handler: Callable[[Any], str] | None = None,
    read_only: bool | None = None,
    destructive: bool | None = None,
    idempotent: bool | None = None,
    open_world: bool | None = None,
) -> Callable[[Callable[..., Any]], Tool]: ...


def tool(
    fn: Callable[..., Any] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
    parallel: bool = True,
    exception_handler: Callable[[Any], str] | None = None,
    read_only: bool | None = None,
    destructive: bool | None = None,
    idempotent: bool | None = None,
    open_world: bool | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turns a function or method into a tool the model can call.

    Type hints become the input schema and the docstring becomes the description. The ``Args:`` section of the
    docstring describes each parameter. A parameter typed ``State`` is hidden from the model and receives the
    current State. The return value is sent to the model: a ``str`` as is, ``None`` as ``(done)``, anything else
    as JSON.

    Use it bare (``@tool``) or with options (``@tool(name=..., description=..., parallel=...,
    exception_handler=..., read_only=..., destructive=..., idempotent=..., open_world=...)``).

    An exception from the tool stops the run, because it is usually a bug. For a failure the model should see and
    handle, raise ``ToolError("message")``, or name the exceptions in ``exception_handler``.

    ``read_only``, ``destructive``, ``idempotent`` and ``open_world`` describe what the tool does, with the meaning of
    MCP tool annotations. The model does not see them: they are for your code, such as a loop that asks before calls
    that are not read-only, and it reads them through ``agent.tool_map``. A hint left out assumes the worst: not
    read-only, destructive, not idempotent, open world. ``@tool(read_only=True, open_world=False)`` fits a tool that
    reads local files.

    Example:
        ```python
        @tool
        def read_file(path: str) -> str:
            \"\"\"Read a file's contents

            Args:
                path: Path relative to the repository root
            \"\"\"
            return Path(path).read_text()
        ```

    Args:
        fn: The function or method. Every parameter needs a type hint.
        name: The tool name. Must match ``[A-Za-z0-9_-]{1,64}``. Defaults to the function name.
        description: Defaults to the docstring's first paragraph.
        parallel: ``False`` runs the tool alone after the turn's other calls finish.
        exception_handler: A function that takes one exception and returns the message the model is told. The
            type hint of its parameter names the exceptions it takes (``httpx.HTTPError``, or
            ``FileNotFoundError | PermissionError``); those become error results and the run continues. Other
            exceptions stop the run. To let one of them stop the run anyway, raise it in the handler.

            ```python
            def http_errors(error: httpx.HTTPError) -> str:
                return f"request failed: {error}"

            @tool(exception_handler=http_errors)
            def fetch_url(url: str) -> str:
                return httpx.get(url).raise_for_status().text
            ```
        read_only: The tool does not change its environment. Sets ``destructive=False`` and ``idempotent=True``.
        destructive: The tool may delete or overwrite something (``False`` for one that only adds).
        idempotent: Calling it again with the same arguments has no further effect.
        open_world: The tool reaches outside its own domain, such as the web, a shell or another system.

    Returns:
        A ``Tool``, or a decorator that makes one when options are given.

    Raises:
        TypeError: ``@tool`` is applied twice or to something that is not a function, the function cannot be a
            tool, ``exception_handler`` is not a function with one parameter whose type hint names
            ``Exception`` subclasses, or a hint is not a bool (see ``Tool``).
        ValueError: ``read_only=True`` with ``destructive=True`` or ``idempotent=False``.
    """
    options: dict[str, Any] = {
        "name": name, "description": description, "parallel": parallel, "exception_handler": exception_handler,
        "read_only": read_only, "destructive": destructive, "idempotent": idempotent, "open_world": open_world,
    }
    if fn is not None:
        if isinstance(fn, Tool):
            raise TypeError(
                fix_message(
                    f"@tool was applied twice to {fn.name!r}",
                    "Apply @tool only once per function",
                )
            )
        if not inspect.isfunction(fn):
            raise TypeError(
                fix_message(
                    f"{fn!r} is not a function, so @tool cannot be applied to it",
                    "Apply @tool to a plain function or method",
                )
            )
        return Tool(fn, **options)

    def decorate(inner: Callable[..., Any]) -> Tool:
        return tool(inner, **options)

    return decorate


def collect_tools(items: Iterable[Any]) -> dict[str, Tool]:
    """Flattens the items of ``Agent(tools=...)`` or ``think(tools=...)`` into a name → Tool dict (insertion order).

    Rules per item (Agent instances are filtered out by Agent first, so they never get here):

    - A bound Tool, or a Tool with ``needs_self=False`` → as is
    - An unbound method Tool (``FileSystem.read_file``) → ``TypeError``: create an object and pass that
      (``tools=[FileSystem()]`` or ``tools=[fs.read_file]``)
    - A function or method without ``@tool`` → ``TypeError``: add ``@tool``
    - Any other object → finds every Tool attribute in the MRO of ``type(obj)`` and binds it with
      ``getattr(obj, attr)``. If there are none, ``TypeError``: an object with no ``@tool`` methods
    - Duplicate names → ``ValueError``: the duplicate name and both sources (repr), fix ``@tool(name="...")``
    """
    result: dict[str, Tool] = {}
    sources: dict[str, Any] = {}

    def _add(t: Tool, source: Any) -> None:
        if t.name in result:
            raise ValueError(
                fix_message(
                    f"Duplicate tool name {t.name!r}: {sources[t.name]!r} and {source!r}",
                    "Rename one of them with @tool(name=\"...\")",
                )
            )
        result[t.name] = t
        sources[t.name] = source

    for item in items:
        if isinstance(item, Tool):
            if item.needs_self and item.bound_to is None:
                raise TypeError(
                    fix_message(
                        f"{item!r} is an unbound method tool",
                        "Create an object and pass that",
                        f"tools=[MyClass()] or tools=[obj.{item.name}]",
                    )
                )
            _add(item, item)
            continue

        if inspect.isfunction(item) or inspect.ismethod(item):
            raise TypeError(
                fix_message(
                    f"{item!r} does not have @tool",
                    "Add @tool to the function or method definition",
                )
            )

        if isinstance(item, type):
            if any(isinstance(attr, Tool) for klass in item.__mro__ for attr in vars(klass).values()):
                raise TypeError(
                    fix_message(
                        f"tools= got the class {item.__name__}. @tool methods are tools only on an object",
                        "Create an object and pass that, not the class",
                        f"tools=[{item.__name__}()]",
                    )
                )
        found = False
        seen_names: set[str] = set()
        for klass in type(item).__mro__:
            for attr_name, attr in vars(klass).items():
                if attr_name in seen_names:
                    continue
                if isinstance(attr, Tool):
                    seen_names.add(attr_name)
                    found = True
                    bound_tool = getattr(item, attr_name)
                    _add(bound_tool, item)
        if not found:
            raise TypeError(
                fix_message(
                    f"{item!r} has no methods with @tool",
                    "Pass an object that has @tool methods, or add @tool to a function",
                )
            )

    return result


class ToolMap(Mapping[str, Any]):
    """``agent.tool_map``: a read-only name -> tool mapping. Looking up a missing name while MCP servers are not
    connected raises a ``KeyError`` that says so."""

    __slots__ = ("_tools", "_waiting")

    def __init__(self, tools: Mapping[str, Any], waiting: tuple[str, ...]) -> None:
        self._tools = dict(tools)
        #: Names of the MCP servers that are not connected, so their tools are missing.
        self._waiting = waiting

    def __getitem__(self, name: str) -> Any:
        try:
            return self._tools[name]
        except KeyError:
            if not self._waiting:
                raise
        raise _MissingToolError(
            fix_message(
                f"No tool named {name!r}. This Agent's MCP servers ({', '.join(self._waiting)}) are not connected, "
                "so their tools are not listed yet",
                "Read tool_map during a run (in the loop, a block or a Reporter), or inside `with agent:`",
                "with agent:\n    print(list(agent.tool_map))",
            )
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return repr(self._tools)


class _MissingToolError(KeyError):
    """A ``KeyError`` whose message prints as written (``KeyError`` itself prints the repr of its argument)."""

    def __str__(self) -> str:
        return str(self.args[0])


# Tracebacks name the class by module and qualname: show it as the KeyError it is, not an internal class.
_MissingToolError.__module__ = "builtins"
_MissingToolError.__qualname__ = "KeyError"
