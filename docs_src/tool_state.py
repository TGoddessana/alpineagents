from alpineagents import State, tool


@tool
def remember(key: str, value: str, state: State) -> str:
    """Save a note for later in this run"""
    with state.edit_extra_data() as data:
        data.setdefault("notes", {})[key] = value
    return f"Saved {key}"


@tool
def recall(key: str, state: State) -> str:
    """Read a note saved with remember"""
    return state.extra_data.get("notes", {}).get(key, f"No note named {key}")
