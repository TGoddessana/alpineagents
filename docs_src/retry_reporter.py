from alpineagents import Reporter


class LiveText(Reporter):
    """Keeps the text of the reply being written, for a UI that redraws it."""

    def __init__(self):
        self.text = ""

    def on_think_start(self, state):
        self.text = ""

    def on_text(self, state, chunk):
        self.text += chunk

    def on_model_event(self, state, event):
        if event.kind == "retry":
            self.text = ""  # the reply starts over
