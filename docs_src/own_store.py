from datetime import datetime, timezone

from alpineagents import Agent, Message, State, Store
from alpineagents.store import Record, StateInfo


class MemoryStore(Store):
    """Keeps States in a dict. A real store writes to a database."""

    def __init__(self):
        self.records: dict[str, Record] = {}

    def write(self, state_id, entries, info, *, create=False):
        old = self.records.get(state_id)
        if create and old is not None:
            raise ValueError(f"{state_id} already exists")
        if not create and old is None:
            raise LookupError(state_id)
        kept = old.entries if old else []
        new = [entry for entry in entries if entry["seq"] >= len(kept)]  # skip entries sent again
        self.records[state_id] = Record(kept + new, dict(info))

    def read(self, state_id):
        return self.records.get(state_id)

    def list(self):
        infos = [StateInfo.from_info(state_id, record.info) for state_id, record in self.records.items()]
        return sorted(infos, key=lambda i: i.updated_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    def delete(self, state_id):
        self.records.pop(state_id, None)


store = MemoryStore()
agent = Agent(model="claude-sonnet-5", store=store)

state = State(id="demo", messages=[Message.user("Say hello")])
agent.run(state)
print(store.load("demo").answer)
