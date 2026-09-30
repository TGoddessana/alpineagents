"""The Store role: keeps States outside the process, so a run can be continued later (``--resume``).

A Store only moves JSON values. State turns itself into them (``_serial``), and the Agent decides when to save
(ARCHITECTURE.md "Store"). A saved State is two things under its id:

- **entries**: history, one dict per entry, in order. Each has ``seq``, its index in history. Writes only add
  entries, and a store skips entries whose ``seq`` it already has, so writing the same batch again is harmless.
- **snapshot**: one dict with the rest of the State (context, turn, usage, data, the Agent summary, ...). Each
  write replaces it. ``history_len`` says how many entries it covers.

An implementation provides ``write``/``read``/``list``/``delete``, or their ``async`` versions (``awrite``, ...),
or both, like Human. ``load`` rebuilds a State from ``read``.
"""

from __future__ import annotations

import builtins
import json
import os
import shutil
import threading
import uuid
from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Self

from . import _serial
from ._async import run_in_thread
from .errors import fix_message
from .state import State, check_id
from .types import Stopped

__all__ = ["Store", "FileStore", "Record", "SavedState"]


@dataclass(frozen=True)
class Record:
    """What a store has for one State id (``Store.read``)."""

    entries: list[dict[str, Any]]
    """History entries in ``seq`` order, starting at 0."""
    snapshot: dict[str, Any] | None
    """The latest snapshot, or ``None`` if none was written yet."""


@dataclass(frozen=True)
class SavedState:
    """One saved State, as ``Store.list`` shows it."""

    id: str
    """The State id. Continue it with ``store.load(id)``."""
    task: str
    """The task the State was created with."""
    created_at: datetime
    """When the State was created (UTC)."""
    updated_at: datetime
    """When the last saved step was recorded (UTC)."""
    turn: int
    """Turns so far."""
    stopped: Stopped | None
    """Why its last run stopped (``state.stopped``: ``StoppedByUntil``, ``StoppedByLimit``, ``StoppedByFinish``
    or ``StoppedByPermission``), or ``None`` if it ended with an exception or was still running."""
    finished: bool
    """Whether ``finish()`` was called. A finished State cannot run again."""

    @classmethod
    def from_snapshot(cls, state_id: str, snapshot: Mapping[str, Any]) -> SavedState:
        """Builds one from a snapshot dict, for Store implementations."""
        return cls(
            id=state_id,
            task=snapshot["task"],
            created_at=_serial.time_from_str(snapshot["created_at"]),
            updated_at=_serial.time_from_str(snapshot["updated_at"]),
            turn=snapshot["turn"],
            stopped=_serial.stopped_from_snapshot(snapshot),
            finished=snapshot["finished"],
        )


def _overrides(store: Store, name: str) -> bool:
    return getattr(type(store), name) is not getattr(Store, name)


class Store(ABC):
    """Keeps States outside the process. Used by ``Agent(store=...)``, which saves as the run goes.

    A subclass implements ``write`` and ``read`` (and ``list``, ``delete`` if it can), or their ``async``
    versions, or both. The Agent calls ``write`` from one thread at a time for a given State, outside the State
    lock. ``FileStore`` is the implementation that ships with alpineagents.

    A State is saved in one store only. The Agent compares stores with ``==``, so define ``__eq__`` if two
    objects can stand for the same storage.

    Raises:
        TypeError: When creating an instance of a subclass that implements neither ``write`` nor ``awrite``, or
            neither ``read`` nor ``aread``.

    Example:
        ```python
        class RedisStore(Store):
            async def awrite(self, state_id, entries, snapshot, *, create=False): ...
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
                        "    async def awrite(self, state_id, entries, snapshot, *, create=False): ...\n"
                        "    async def aread(self, state_id): ...",
                    )
                )
        return super().__new__(cls)

    # ------------------------------------------------------------ the role (implemented by subclasses)

    def write(
        self,
        state_id: str,
        entries: Sequence[dict[str, Any]],
        snapshot: dict[str, Any] | None,
        *,
        create: bool = False,
    ) -> None:
        """Adds history entries and replaces the snapshot of one State.

        Args:
            state_id: The State id.
            entries: New history entries in ``seq`` order. Skip those whose ``seq`` you already have. They
                continue right after what you have (no gaps).
            snapshot: The new snapshot, or ``None`` to keep the current one. Write it after the entries.
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
        snapshot: dict[str, Any] | None,
        *,
        create: bool = False,
    ) -> None:
        """The async version of ``write``. By default it runs ``write`` on a worker thread."""
        await run_in_thread(lambda: self.write(state_id, entries, snapshot, create=create))

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

    def list(self) -> builtins.list[SavedState]:
        """Every saved State, most recently updated first."""
        if _overrides(self, "alist"):
            raise self._async_only("list", "alist")
        raise NotImplementedError(f"{type(self).__name__} does not implement list")

    async def alist(self) -> builtins.list[SavedState]:
        """The async version of ``list``. By default it runs ``list`` on a worker thread."""
        if not _overrides(self, "list"):
            raise NotImplementedError(f"{type(self).__name__} does not implement list")
        return await run_in_thread(self.list)

    # ------------------------------------------------------------ built on read

    def load(self, state_id: str) -> State:
        """Rebuilds a saved State, ready to continue with ``agent.run(state)``.

        Tool calls whose results were never saved (the process stopped while they ran) get a result telling the
        model that they may or may not have run. The first ``run`` warns with ``ResumeWarning`` if its Agent
        differs from the one that saved the State.

        Raises:
            LookupError: No State with this id.
            ValueError: The saved State is damaged, or was saved by a newer alpineagents.

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
        return State._from_record(state_id, record.entries, record.snapshot, self)

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
_SNAPSHOT = "snapshot.json"


class FileStore(Store):
    """Saves each State in a folder of its own: ``{path}/{id}/log.jsonl`` (history, one entry per line) and
    ``snapshot.json``.

    Files are readable only by their owner, and each write is flushed to disk (``fsync``) before it returns.
    One process writes a State at a time: two processes running the same State id are not supported.

    Args:
        path: The folder to keep States in. Created on the first write.

    Example:
        ```python
        store = FileStore(".agent-runs")
        agent = Agent(model="claude-sonnet-5", store=store)
        agent.run(State("Find the bug", id="bug-hunt-1"))
        ```
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self._root = Path(path)
        self._lock = threading.Lock()
        # Per State id: the seq the log continues with, read from the log once per process.
        self._next: dict[str, int] = {}

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
        snapshot: dict[str, Any] | None,
        *,
        create: bool = False,
    ) -> None:
        check_id(state_id, "FileStore.write()")
        with self._lock:
            if create:
                self._create(state_id, entries, snapshot)
            else:
                self._add(state_id, entries, snapshot)

    def read(self, state_id: str) -> Record | None:
        folder = self._root / check_id(state_id, "FileStore.read()")
        if not folder.is_dir():
            return None
        with self._lock:
            entries = [json.loads(line) for line in _log_lines(folder / _LOG, repair=False)]
            snapshot_path = folder / _SNAPSHOT
            snapshot = json.loads(snapshot_path.read_text("utf-8")) if snapshot_path.exists() else None
        return Record(entries, snapshot)

    def list(self) -> builtins.list[SavedState]:
        if not self._root.is_dir():
            return []
        found = []
        for folder in self._root.iterdir():
            if folder.name.startswith(".") or not folder.is_dir():
                continue
            snapshot_path = folder / _SNAPSHOT
            if snapshot_path.exists():
                found.append(SavedState.from_snapshot(folder.name, json.loads(snapshot_path.read_text("utf-8"))))
                continue
            lines = _log_lines(folder / _LOG, repair=False)
            if lines:
                first = json.loads(lines[0])
                at = _serial.time_from_str(first["at"])
                found.append(SavedState(folder.name, first["content"], at, at, 0, None, False))
        found.sort(key=lambda saved: saved.updated_at, reverse=True)
        return found

    def delete(self, state_id: str) -> None:
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

    def _create(self, state_id: str, entries: Sequence[dict[str, Any]], snapshot: dict[str, Any] | None) -> None:
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
            if snapshot is not None:
                _write_file(staging / _SNAPSHOT, _dump(snapshot).encode("utf-8"), append=False)
            try:
                os.rename(staging, folder)
            except OSError:
                if folder.exists():
                    raise _exists(state_id, self) from None
                raise
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        self._next[state_id] = len(entries)

    def _add(self, state_id: str, entries: Sequence[dict[str, Any]], snapshot: dict[str, Any] | None) -> None:
        folder = self._root / state_id
        if not folder.is_dir():
            self._next.pop(state_id, None)
            raise LookupError(f"no saved State with id {state_id!r} in {self!r} (was it deleted?)")
        log = folder / _LOG
        if state_id not in self._next:
            self._next[state_id] = len(_log_lines(log, repair=True))
        start = self._next[state_id]
        new = [entry for entry in entries if entry["seq"] >= start]
        if new:
            if new[0]["seq"] != start:
                raise ValueError(
                    f"history entries for {state_id!r} start at seq {new[0]['seq']}, but {self!r} has {start}"
                )
            _write_file(log, _lines(new), append=True)
            self._next[state_id] = start + len(new)
        if snapshot is not None:
            staging = folder / f".{_SNAPSHOT}.{uuid.uuid4().hex}"
            try:
                _write_file(staging, _dump(snapshot).encode("utf-8"), append=False)
                os.replace(staging, folder / _SNAPSHOT)
            except BaseException:
                staging.unlink(missing_ok=True)
                raise


def _exists(state_id: str, store: FileStore) -> ValueError:
    return ValueError(
        fix_message(
            f"a State with id {state_id!r} is already saved in {store!r}",
            "continue it with store.load(id), or give the new State another id (or none, for a random one)",
            f'state = store.load("{state_id}")\nagent.run(state)',
        )
    )


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


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
    return data[:end].decode("utf-8").splitlines()
