# Use other models

## Model strings

`Agent(model=...)` accepts a string:

| String | Model | API key |
| --- | --- | --- |
| `"claude-sonnet-5"`, any name starting with `claude-` | `Anthropic` | `ANTHROPIC_API_KEY` |
| `"anthropic/<name>"` | `Anthropic` | `ANTHROPIC_API_KEY` |
| `"gpt-5"`, any name starting with `gpt-` | `OpenAICompatible` at api.openai.com | `OPENAI_API_KEY` |
| `"openai/<name>"` | `OpenAICompatible` at api.openai.com | `OPENAI_API_KEY` |
| `"ollama/<name>"`, for example `"ollama/llama3:8b"` | `OpenAICompatible` at `http://localhost:11434/v1` | Not needed |

Any other string raises `ValueError` with the strings you can use instead.

## Options with a model string

`resolve_model` turns a model string into a Model, the same way `Agent(model=...)` does. Keyword options go to the
Model it picks:

```python
from alpineagents import Agent
from alpineagents.models import resolve_model

model = resolve_model("anthropic/claude-sonnet-5", max_tokens=16_000, thinking=True)
# the same as Anthropic("claude-sonnet-5", max_tokens=16_000, thinking=True)

agent = Agent(model=model)
print(agent.run("Say hello"))
```

- `"ollama/<name>"` uses `http://localhost:11434/v1` unless you pass `base_url=`, for an Ollama server on another
  machine.
- An option the picked Model does not take raises `TypeError` that lists the options it takes.

With `base_url=`, the string is not split into a provider and a name. The model is always `OpenAICompatible`, and
the whole string is the model name the server receives. Routers such as OpenRouter name their models this way:

```python
import os

from alpineagents import Agent
from alpineagents.models import resolve_model

model = resolve_model(
    "anthropic/claude-sonnet-5",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
# the same as OpenAICompatible("anthropic/claude-sonnet-5", base_url=..., api_key=...)

agent = Agent(model=model)
print(agent.run("Say hello"))
```

A proxy that speaks the Anthropic API rather than the OpenAI one needs `Anthropic` itself:
`Anthropic("claude-sonnet-5", base_url="https://my-proxy/anthropic")`.

## Model settings

Create the Model yourself to change its settings:

```python
--8<-- "docs_src/model_settings.py"
```

- `OpenAICompatible` works with any server that speaks the OpenAI Chat Completions API: OpenAI, Ollama, vLLM,
  OpenRouter and others.
- Set `context_window` to the model's real window. `compact_if_full` and `agent.context_used(state)` use it.
- Creating a Model uses no network and needs no API key. The key is read at the first request.
- Retries are left to the provider SDK: `retries=2` by default.

See the [Models API](../api/models.md) for every setting.

## Cost

`state.usage.cost` is `None` unless the Model has prices. Prices are dollars per million tokens:

```python
from alpineagents import Agent, Anthropic, Message, Price, State

model = Anthropic("claude-sonnet-5", price=Price(input=3, output=15, cache_read=0.3))
agent = Agent(model=model)
state = State(messages=[Message.user("Say hello")])
agent.run(state)
print(state.usage.cost)
```

Each `Reply` records the model that answered in `reply.model`, as the provider reported it. It can differ from the
requested name: an alias resolved to a dated id, or a router that picked another model. The cost still uses the
requested Model's price.

## Switch models in one conversation

A State does not belong to the Agent that ran it. Any Agent may run, think, use tools and compact the same State, so a
conversation can start on a fast model and go on with a strong one. `agent.copy(model=...)` makes the second Agent
with every other setting the same:

```python
--8<-- "docs_src/switch_model.py"
```

- Both runs add to `state`. The second model sees the whole conversation, including the replies of the first.
- `state.history` records the model of every `think` request in a `model_request` entry, so you can see which model
  wrote each reply. Tokens of `ask` and `compact` requests count in `state.usage`, but they write no `model_request`
  entry. `usage.cost` is `None` when any model used has no price.
- Thinking blocks and other provider-specific blocks are sent back only to the provider that wrote them.
- Providers cache a long context per model, so the first request after a switch pays for the whole context again.
- The windows differ. Check `strong.context_used(state)` before you switch to a model with a smaller window, and
  compact first if needed: `strong.compact(state)`.
- Switching never warns. `ResumeWarning` is only for the first run after `store.load`
  ([State and history](../concepts/state.md#a-state-saved-by-another-agent)).

## Write a Model

Subclass `Model` to add a provider. Implement `respond` and `context_window`:

```python
--8<-- "docs_src/custom_model.py"
```

Rules for a Model:

- Return a `Reply`. Do not change the State. The Agent records the reply.
- User messages can hold `Image` blocks. Convert them to the provider's image format, or raise a clear error if the
  provider cannot read images.
- Put the model that actually answered in `Reply.model`: the name the provider reported, or the requested name if it
  did not say.
- Use no network and no credentials in `__init__`. Create the SDK client at the first request.
- Call `on_text` with each piece of text as it streams in, if `on_text` is not `None`.
- Wrap provider exceptions that finally fail as `RateLimitError`, `AuthError`, `ContextTooLongError` or
  `ProviderError`, with `raise ... from e`.
- Print nothing. Report events such as a fallback to another model with `on_event(ModelEvent(...))`.

`Model` has working defaults for everything else, including `arespond` for async code.

## Related

- [Test your agent](../learn/testing.md): `FakeModel` replaces the model in tests
- [Handle unreliable models](unreliable-models.md): wrap a Model to retry
