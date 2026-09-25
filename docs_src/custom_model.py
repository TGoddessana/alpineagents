from alpineagents import Message, Model, Reply, Usage
from alpineagents.types import TextBlock


class Echo(Model):
    """Replies with the last user message. Uses no network."""

    name = "echo"
    provider = "echo"

    @property
    def context_window(self) -> int:
        return 100_000

    def respond(self, request, on_text=None, on_event=None):
        text = request.messages[-1].text
        if on_text:
            on_text(text)
        return Reply(
            message=Message("assistant", (TextBlock(text),)),
            usage=Usage(requests=1),
            context_tokens=self.count_tokens(request),
        )
