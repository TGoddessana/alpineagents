from alpineagents import Agent, FileStore, Message, State
from alpineagents.testing import FakeModel


def test_a_saved_conversation_loads_as_it_was(tmp_path):
    store = FileStore(tmp_path)
    agent = Agent(model=FakeModel(["Done"]), reporter=None, store=store)
    state = State(id="t1", messages=[Message.user("Fix it")])

    agent.run(state)

    loaded = store.load("t1")
    assert loaded.answer == "Done"
    assert loaded.snapshot() == state.snapshot()  # a loaded State equals the saved one
