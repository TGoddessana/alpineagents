from alpineagents.testing import FakeModel, tool_call

fake = FakeModel([
    tool_call("read_file", path="README.md"),
    "README.md describes a Python agent framework.",
])
print(agent.copy(model=fake).run("Summarize README.md"))
