"""The Store role: keeps States outside the process, so a run can be continued later (``--resume``).

A Store only moves JSON values. ``State`` turns itself into them (``_serial``), and ``store.save(state)`` (called by
the Agent at its save points) hands them over (ARCHITECTURE.md "Store"). A saved State is two things under its id:

- **entries**: history, one dict per entry, in order. Each has ``seq``, its index in history. Writes only add
  entries, and a store skips entries whose ``seq`` it already has, so writing the same batch again is harmless.
- **info**: one small dict with what a listing shows (``StateInfo``: the first message, times, turn, why it
  stopped) and the format version. Each write replaces it.

History is the only source of truth: ``load`` replays every entry through the same ``fold`` that built the State
(``State(history=...)``), so nothing else needs saving.

An implementation provides ``write``/``read``/``list``/``delete``, or their ``async`` versions (``awrite``, ...),
or both, like Human. ``save`` and ``load`` are built on them.
"""

from __future__ import annotations

import asyncio
import builtins
import json
import os
import shutil
import threading
import uuid
from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Self

from . import _serial
from ._async import ASYNC_RUN, run_in_thread
from .errors import fix_message
from .state import State, _Batch, check_id
from .types import Stopped

__all__ = ["Store", "FileStore", "Record", "StateInfo"]


@dataclass(frozen=True)
class Record:
    """What a store has for one State id (``Store.read``)."""

    entries: list[dict[str, Any]]
    """History entries in ``seq`` order, starting at 0."""
    info: dict[str, Any]
    """The latest info dict, exactly as ``write`` got it (it has the format version under ``"v"``)."""


@dataclass(frozen=True)
class StateInfo:
    """One saved State, as ``Store.list`` shows it."""

    id: str
    """The State id. Continue it with ``store.load(id)``."""
    first_message: str | None
    """The text of the first user message, or ``None`` if the State has none (or it has no text)."""
    created_at: datetime | None
    """When the first thing was recorded (UTC), or ``None`` for a State saved with no history."""
    updated_at: datetime | None
    """When the last saved step was recorded (UTC), or ``None`` for a State saved with no history."""
    turn: int
    """Turns so far."""
    stopped: Stopped | None
    """Why its last run stopped (``state.stopped``: ``StoppedByUntil``, ``StoppedByLimit``, ``StoppedByFinish``
    or ``StoppedByPermission``), or ``None`` if it ended with an exception or was still running."""
    finished: bool
    """Whether ``finish()`` was called. A finished State cannot run again."""

    @classmethod
    def from_info(cls, state_id: str, info: Mapping[str, Any]) -> StateInfo:
        """Builds one from an info dict, for Store implementations.

        Raises:
            ValueError: The info was saved by another format version, or is missing keys.
        """
        return cls(id=state_id, **_serial.info_from_dict(state_id, info))


def _overrides(store: Store, name: str) -> bool:
    return getattr(type(store), name) is not getattr(Store, name)


class Store(ABC):
    """Keeps States outside the process. Used by ``Agent(store=...)``, which saves as the run goes, and directly
    with ``store.save(state)`` and ``store.load(id)``.

    A subclass implements ``write`` and ``read`` (and ``list``, ``delete`` if it can), or their ``async``
    versions, or both. ``save`` calls ``write`` from one thread at a time for a given State, outside the State
    lock. ``FileStore`` is the implementation that ships with alpineagents.

    A State is saved in one store only. ``save`` compares stores with ``==``, so define ``__eq__`` if two
    objects can stand for the same storage.

    A store that implements only the ``async`` methods works only with ``arun`` and the ``a*`` methods
    (``asave``, ``aload``); a sync store also works with ``arun``, where its methods run on a worker thread.
    Sync ``save`` called on the event loop thread of an async run raises ``TypeError``. The saved format has a
    version: ``load`` refuses a State saved by another version (alpineagents 0.4 States do not load in 0.5).

    Raises:
        TypeError: When creating an instance of a subclass that implements neither ``write`` nor ``awrite``, or
            neither ``read`` nor ``aread``.

    Example:
        ```python
        class RedisStore(Store):
            async def awrite(self, state_id, entries, info, *, create=False): ...
            async def aread(self, state_id): ...
        ```
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        for sync, asynchronous in (("write", "awrite"), ("read", "aread")):
            inherited = getattr(cls, sync) is getattr(Store, sync)
            if inherited and getattr(cls, asynchronous) is getattr(Store, asynchronous):
                raise TypeError(
                    fix_message(
                        f"{cls.__name__} implements neither {sync} nor {asynchronous}",
                        f"implement {sync}(...), or async {asynchronous}(...) for a store reached asynchronously",
                        "class RedisStore(Store):\n"
                        "    async def awrite(self, state_id, entries, info, *, create=False): ...\n"
                        "    async def aread(self, state_id): ...",
                    )
                )
        return super().__new__(cls)

    # ------------------------------------------------------------ the role (implemented by subclasses)

    def write(
        self,
        state_id: str,
        entries: Sequence[dict[str, Any]],
        info: Mapping[str, Any],
        *,
        create: bool = False,
    ) -> None:
        """Adds history entries and replaces the info dict of one State.

        Args:
            state_id: The State id.
            entries: New history entries in ``seq`` order. Skip those whose ``seq`` you already have. They
                continue right after what you have (no gaps). May be empty (a State with no history yet).
            info: The new info dict: a small JSON dict with the ``StateInfo`` fields (all but ``id``) and the
                format version under ``"v"``. No messages, no history. Replace the stored one with it, after the
                entries. Give it back as it is in ``Record.info``.
            create: The first write of a new State. If ``state_id`` already exists, write nothing and raise
                ``ValueError``. Either everything is written or nothing is.

        Raises:
            ValueError: ``create`` is true and the id exists.
            LookupError: ``create`` is false and the id does not exist (deleted while in use).
        """
        raise self._async_only("write", "awrite")

    async def awrite(
        self,
        state_id: str,
        entries: Sequence[dict[str, Any]],
        info: Mapping[str, Any],
        *,
        create: bool = False,
    ) -> None:
        """The async version of ``write``. By default it runs ``write`` on a worker thread."""
        await run_in_thread(lambda: self.write(state_id, entries, info, create=create))

    def read(self, state_id: str) -> Record | None:
        """Everything saved for one State, or ``None`` if the id does not exist."""
        raise self._async_only("read", "aread")

    async def aread(self, state_id: str) -> Record | None:
        """The async version of ``read``. By default it runs ``read`` on a worker thread."""
        return await run_in_thread(self.read, state_id)

    def delete(self, state_id: str) -> None:
        """Deletes one State. Does nothing if the id does not exist."""
        if _overrides(self, "adelete"):
            raise self._async_only("delete", "adelete")
        raise NotImplementedError(f"{type(self).__name__} does not implement delete")

    async def adelete(self, state_id: str) -> None:
        """The async version of ``delete``. By default it runs ``delete`` on a worker thread."""
        if not _overrides(self, "delete"):
            raise NotImplementedError(f"{type(self).__name__} does not implement delete")
        await run_in_thread(self.delete, state_id)

    def list(self) -> builtins.list[StateInfo]:
        """Every saved State as a ``StateInfo``, most recently updated first."""
        if _overrides(self, "alist"):
            raise self._async_only("list", "alist")
        raise NotImplementedError(f"{type(self).__name__} does not implement list")

    async def alist(self) -> builtins.list[StateInfo]:
        """The async version of ``list``. By default it runs ``list`` on a worker thread."""
        if not _overrides(self, "list"):
            raise NotImplementedError(f"{type(self).__name__} does not implement list")
        return await run_in_thread(self.list)

    # ------------------------------------------------------------ built on write and read

    def save(self, state: State) -> None:
        """Writes what this store does not have yet of ``state``: its new history entries and its info.

        The Agent calls it at its save points (``Agent(store=...)``). Call it yourself to save something else right
        away, such as the person's next message in a chat loop, before the next ``run``. The first save of a State
        refuses an id the store already has, and a State is saved in one store only. Call it from one thread at a
        time for a given State.

        Args:
            state: The State to save.

        Raises:
            ValueError: The State is saved in another store, or it is new and this store already has a State
                with its id.
            TypeError: A value in the State cannot be saved as JSON, the store only works asynchronously
                (use ``asave``), or it was called on the event loop thread of an async run (use ``asave``).

        Example:
            ```python
            state.add_message(Message.user(input("> ")))
            store.save(state)
            ```
        """
        _check_sync_call("save")
        batch = _batch(self, state, "save")
        if batch is None:
            return
        self.write(state.id, batch.entries, batch.info, create=batch.create)
        state._mark_saved(self, batch)

    async def asave(self, state: State) -> None:
        """The async version of ``save``."""
        batch = _batch(self, state, "asave")
        if batch is None:
            return
        await self.awrite(state.id, batch.entries, batch.info, create=batch.create)
        state._mark_saved(self, batch)

    def load(self, state_id: str) -> State:
        """Rebuilds a saved State, ready to continue with ``agent.run(state)``.

        Every saved entry is replayed. A State saved in the middle of a turn is closed: tool calls whose results
        were never saved get a result telling the model that they may or may not have run, and a request that was
        waiting on the model is taken back. The first ``run`` warns with ``ResumeWarning`` if its Agent differs
        from the one that saved the State.

        Raises:
            LookupError: No State with this id.
            ValueError: The saved State is damaged, or was saved by another version of alpineagents.

        Example:
            ```python
            state = store.load("bug-hunt-1")
            agent.run(state)
            ```
        """
        return self._rebuild(state_id, self.read(check_id(state_id, "load()")))

    async def aload(self, state_id: str) -> State:
        """The async version of ``load``."""
        return self._rebuild(state_id, await self.aread(check_id(state_id, "aload()")))

    def _rebuild(self, state_id: str, record: Record | None) -> State:
        if record is None:
            raise LookupError(f"no saved State with id {state_id!r} in {self!r}")
        if not isinstance(record.info, Mapping):
            raise ValueError(f"saved State {state_id!r} is damaged: its info is not a dict")
        # First, so a State saved by another version says so instead of failing on an entry it cannot read.
        _serial.check_version(state_id, record.info)
        for index, entry in enumerate(record.entries):
            if entry.get("seq") != index:
                raise ValueError(
                    f"saved State {state_id!r} is damaged: history entry {index} has seq {entry.get('seq')!r}"
                )
        history = [_serial.entry_from_dict(entry) for entry in record.entries]
        return State._from_store(state_id, history, self)

    def _async_only(self, sync: str, asynchronous: str) -> TypeError:
        return TypeError(
            fix_message(
                f"{type(self).__name__} works only asynchronously (it implements {asynchronous}, not {sync})",
                "run the agent with await agent.arun(...), and use the a* methods of the store",
                "state = await store.aload(state_id)\nawait agent.arun(state)",
            )
        )


# ---------------------------------------------------------------- FileStore

_LOG = "log.jsonl"
_INFO = "info.json"
# The 0.4 layout kept the rest of the State in this file instead. It is only read to say why it cannot be loaded.
_LEGACY = "snapshot.json"
_EPOCH = datetime.min.replace(tzinfo=timezone.utc)


class FileStore(Store):
    """Saves each State in a folder of its own: ``{path}/{id}/log.jsonl`` (history, one entry per line) and
    ``info.json`` (what ``list`` shows).

    Files are readable only by their owner (history holds tool results and messages), and each write is flushed to
    disk (``fsync``) before it returns. One process writes a State at a time: two processes running the same State
    id are not supported. Two ``FileStore`` objects for the same folder are equal, so a State loaded with one can
    be saved with the other. Images are saved as base64, so a run with many screenshots makes large files.

    Args:
        path: The folder to keep States in. Created on the first write.

    Example:
        ```python
        store = FileStore(".agent-runs")
        agent = Agent(model="claude-sonnet-5", store=store)
        agent.run(State(messages=[Message.user("Find the bug")], id="bug-hunt-1"))
        ```
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._root = Path(path)
        self._lock = threading.Lock()
        # Per State id: the seq the log continues with and the log's size when this object last wrote it. The count
        # is read from the log once; a log whose size is not what this object left (another FileStore object for the
        # same folder wrote to it) is counted again.
        self._next: dict[str, tuple[int, int]] = {}

    def __repr__(self) -> str:
        return f"FileStore({str(self._root)!r})"

    def __eq__(self, other: object) -> bool:
        # Two FileStores for the same folder are the same store (a State loaded with one can be saved with the
        # other).
        if not isinstance(other, FileStore):
            return NotImplemented
        return self._root.resolve() == other._root.resolve()

    def __hash__(self) -> int:
        return hash(self._root.resolve())

    def write(
        self,
        state_id: str,
        entries: Sequence[dict[str, Any]],
        info: Mapping[str, Any],
        *,
        create: bool = False,
    ) -> None:
        """Appends the new entries to ``{state_id}/log.jsonl`` and replaces ``info.json``, flushing both to disk.

        With ``create=True`` the folder is built under a temporary name and renamed into place, so creating is
        all or nothing and never overwrites another State.

        Raises:
            ValueError: ``state_id`` is not a valid id, ``create`` is true and the id exists, or the entries do not
                continue the log.
            LookupError: ``create`` is false and the State was deleted.
        """
        check_id(state_id, "FileStore.write()")
        with self._lock:
            if create:
                self._create(state_id, entries, info)
            else:
                self._add(state_id, entries, info)

    def read(self, state_id: str) -> Record | None:
        """The entries and info saved for ``state_id``, or ``None`` if there is no such folder.

        Raises:
            ValueError: ``state_id`` is not a valid id, or the saved State is damaged (no ``info.json``).
        """
        state_id = check_id(state_id, "FileStore.read()")
        folder = self._root / state_id
        if not folder.is_dir():
            return None
        with self._lock:
            entries = [json.loads(line) for line in _log_lines(folder / _LOG, repair=False)]
            for name in (_INFO, _LEGACY):
                path = folder / name
                if path.exists():
                    # A legacy file has the old format version in it, so ``load`` refuses it with the right words.
                    return Record(entries, json.loads(path.read_text("utf-8")))
        raise ValueError(f"saved State {state_id!r} is damaged: {folder} has no {_INFO}")

    def list(self) -> builtins.list[StateInfo]:
        """Every saved State, most recently updated first. A folder without ``info.json`` is not listed: that
        includes States saved by alpineagents 0.4 (``load`` says why it cannot open them)."""
        if not self._root.is_dir():
            return []
        found = []
        for folder in self._root.iterdir():
            if folder.name.startswith(".") or not folder.is_dir():
                continue
            info_path = folder / _INFO
            if info_path.exists():
                found.append(StateInfo.from_info(folder.name, json.loads(info_path.read_text("utf-8"))))
        found.sort(key=lambda info: info.updated_at or _EPOCH, reverse=True)
        return found

    def delete(self, state_id: str) -> None:
        """Deletes the folder of ``state_id``. The folder is renamed first, so a half-deleted State is never listed.
        Does nothing if the id does not exist."""
        folder = self._root / check_id(state_id, "FileStore.delete()")
        with self._lock:
            self._next.pop(state_id, None)
            if not folder.is_dir():
                return
            # Rename first, so a half-deleted folder never looks like a saved State.
            doomed = self._root / f".deleted-{state_id}-{uuid.uuid4().hex}"
            os.rename(folder, doomed)
        shutil.rmtree(doomed, ignore_errors=True)

    # ------------------------------------------------------------ internal (inside self._lock)

    def _create(self, state_id: str, entries: Sequence[dict[str, Any]], info: Mapping[str, Any]) -> None:
        """Builds the folder under a temporary name, then renames it into place: renaming onto an existing
        folder fails, so creating is all or nothing and never overwrites another State."""
        folder = self._root / state_id
        self._root.mkdir(parents=True, exist_ok=True)
        if folder.exists():
            raise _exists(state_id, self)
        staging = self._root / f".new-{state_id}-{uuid.uuid4().hex}"
        os.mkdir(staging, 0o700)
        try:
            _write_file(staging / _LOG, _lines(entries), append=False)
            _write_file(staging / _INFO, _dump(info).encode("utf-8"), append=False)
            try:
                os.rename(staging, folder)
            except OSError:
                if folder.exists():
                    raise _exists(state_id, self) from None
                raise
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        self._next[state_id] = (len(entries), (folder / _LOG).stat().st_size)

    def _add(self, state_id: str, entries: Sequence[dict[str, Any]], info: Mapping[str, Any]) -> None:
        folder = self._root / state_id
        if not folder.is_dir():
            self._next.pop(state_id, None)
            raise LookupError(f"no saved State with id {state_id!r} in {self!r} (was it deleted?)")
        log = folder / _LOG
        known = self._next.get(state_id)
        if known is None or known[1] != _size_of(log):
            known = (len(_log_lines(log, repair=True)), 0)
        start = known[0]
        new = [entry for entry in entries if entry["seq"] >= start]
        if new:
            if new[0]["seq"] != start:
                raise ValueError(
                    f"history entries for {state_id!r} start at seq {new[0]['seq']}, but {self!r} has {start}"
                )
            _write_file(log, _lines(new), append=True)
            start += len(new)
        self._next[state_id] = (start, _size_of(log))
        # The info is replaced after the entries, through a temporary file, so a reader never sees half of it.
        staging = folder / f".{_INFO}.{uuid.uuid4().hex}"
        try:
            _write_file(staging, _dump(info).encode("utf-8"), append=False)
            os.replace(staging, folder / _INFO)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise


def _size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def _check_sync_call(method: str) -> None:
    """``TypeError`` if ``store.{method}()`` (sync) is called on the event loop thread of an async run, where
    its write would block every other task."""
    run_loop = ASYNC_RUN.get()
    if run_loop is None:
        return
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return
    if running is run_loop:
        raise TypeError(
            fix_message(
                f"store.{method}() was called inside an async run, where it would block the event loop",
                f"Use the async version, await store.a{method}(...)",
                f"await store.a{method}(state)",
            )
        )


def _batch(store: Store, state: State, method: str) -> _Batch | None:
    """What ``store`` still needs of ``state`` (``State._unsaved``), or ``None``. ``TypeError`` if ``state`` is not
    a State."""
    if not isinstance(state, State):
        raise TypeError(
            fix_message(
                f"{method}() takes a State (got: {state!r})",
                "pass the State to save",
                f"store.{method}(state)",
            )
        )
    return state._unsaved(store)


def _exists(state_id: str, store: FileStore) -> ValueError:
    return ValueError(
        fix_message(
            f"a State with id {state_id!r} is already saved in {store!r}",
            "continue it with store.load(id), or give the new State another id (or none, for a random one)",
            f'state = store.load("{state_id}")\nagent.run(state)',
        )
    )


def _dump(value: Any) -> str:
    """``value`` as one line of JSON, in readable UTF-8. A lone surrogate (e.g. from a file name read with
    ``surrogateescape``) cannot be encoded as UTF-8, so for that text the escaped ASCII form is written instead;
    ``json.loads`` reads it back as the same string."""
    text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        text = json.dumps(value, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    return text


def _lines(entries: Sequence[dict[str, Any]]) -> bytes:
    return "".join(_dump(entry) + "\n" for entry in entries).encode("utf-8")


def _write_file(path: Path, data: bytes, *, append: bool) -> None:
    """Writes ``data`` with one ``os.write`` loop and ``fsync``. New files are readable only by the owner."""
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    try:
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def _log_lines(path: Path, *, repair: bool) -> builtins.list[str]:
    """The complete lines of a log. A last line cut off by a crash is ignored, and with ``repair`` cut from the
    file, so the next append starts on a fresh line."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return []
    end = data.rfind(b"\n") + 1
    if end < len(data) and repair:
        with open(path, "r+b") as file:
            file.truncate(end)
            file.flush()
            os.fsync(file.fileno())
    # Split on "\n" only: str.splitlines() also splits on U+2028, U+2029 and U+0085, which JSON may hold raw.
    return data[:end].decode("utf-8").split("\n")[:-1]
