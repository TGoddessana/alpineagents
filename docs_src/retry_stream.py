from alpineagents import Agent, Anthropic, AuthError, ContextTooLongError, Model, ModelEvent, ProviderError


class _Watch:
    """Passes text chunks on and remembers whether any arrived."""

    def __init__(self, on_text):
        self.on_text = on_text
        self.started = False

    def __call__(self, chunk):
        self.started = True
        if self.on_text is not None:
            self.on_text(chunk)


class RetryDroppedStream(Model):
    """Wraps a Model. Starts a reply over when its stream breaks after text arrived."""

    def __init__(self, model: Model, attempts: int = 3):
        self.model = model
        self.attempts = attempts
        self.name = model.name
        self.provider = model.provider
        self.supports = model.supports
        self.price = model.price
        self.max_tokens = model.max_tokens

    @property
    def context_window(self):
        return self.model.context_window

    def count_tokens(self, request):
        return self.model.count_tokens(request)

    def mark_cache(self, request):
        return self.model.mark_cache(request)

    def compact(self, request, instructions=None, on_event=None):
        return self.model.compact(request, instructions, on_event)

    async def acompact(self, request, instructions=None, on_event=None):
        return await self.model.acompact(request, instructions, on_event)

    def respond(self, request, on_text=None, on_event=None):
        for attempt in range(1, self.attempts + 1):
            watch = _Watch(on_text)
            try:
                return self.model.respond(request, watch, on_event)
            except ProviderError as error:
                if not self._retry(error, watch, attempt, on_event):
                    raise

    async def arespond(self, request, on_text=None, on_event=None):
        for attempt in range(1, self.attempts + 1):
            watch = _Watch(on_text)
            try:
                return await self.model.arespond(request, watch, on_event)
            except ProviderError as error:
                if not self._retry(error, watch, attempt, on_event):
                    raise

    def _retry(self, error, watch, attempt, on_event) -> bool:
        if isinstance(error, (AuthError, ContextTooLongError)):
            return False  # retrying cannot fix these
        if not watch.started or attempt == self.attempts:
            return False  # before the stream, the SDK already retried
        if on_event is not None:
            on_event(ModelEvent("retry", f"stream dropped ({error}), starting the reply over", {"attempt": attempt}))
        return True


agent = Agent(model=RetryDroppedStream(Anthropic("claude-sonnet-5")))
