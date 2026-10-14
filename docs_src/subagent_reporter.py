from alpineagents import Agent, Reporter, State


class TreeReporter(Reporter):
    def __init__(self):
        self.depth: dict[str, int] = {}  # State id -> how many parents it has

    def on_run_start(self, state: State):
        # The parent's run started first, so its depth is known. A run without a parent has depth 0.
        self.depth[state.id] = self.depth.get(state.parent_id, -1) + 1

    def on_tool_start(self, state: State, call):
        print("  " * self.depth.get(state.id, 0) + f"{call.name}({call.args})")

    def on_run_end(self, state: State, error):
        print("  " * self.depth.get(state.id, 0) + f"done: {state.stopped}")


lead = Agent(model="claude-sonnet-5", tools=[researcher], reporter=TreeReporter())
lead.run("Which Python versions does this project support? Ask the researcher to read pyproject.toml")
