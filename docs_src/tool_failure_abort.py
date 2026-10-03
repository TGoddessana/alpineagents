import json

from alpineagents import Agent, Message, State, tool

LEGACY_PAGE = "<html>Moved</html>"


@tool
def get_json(path: str) -> dict:
    """GET a JSON endpoint of the API

    Args:
        path: A path such as /users/42
    """
    return json.loads(LEGACY_PAGE)  # the legacy API answers with a web page


agent = Agent(model="claude-sonnet-5", tools=[get_json])

state = State(messages=[Message.user("Get the settings from the legacy API")])
try:
    agent.run(state)
except Exception as error:
    print(error.__notes__)  # ['exception raised in tool get_json(path="/legacy")']
    print(state)
