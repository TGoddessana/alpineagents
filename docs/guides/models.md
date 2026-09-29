# Models

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

`resolve_model` turns a model string into a Model, the same way `Agent(model=...)` does. Keyword options go to
the Model it picks:

```python
from alpineagents import Agent
from alpineagents.models import resolve_model

model = resolve_model("anthropic/claude-sonnet-5", max_tokens=16_000, thinking=True)
# the same as Anthropic("claude-sonnet-5", max_tokens=16_000, thinking=True)

agent = Agent(model=model)
```

- `"ollama/<name>"` uses `http://localhost:11434/v1` unless you pass `base_url=`, for an Ollama server on
  another machine.
- An option the picked Model does not take raises `TypeError` that lists the options it takes. For example,
  `thinking=` with an `"openai/..."` string.
- Passing a Model object together with options raises `TypeError`. Set the options when you create the Model.

### `base_url` sends the whole string to an OpenAI-compatible server

With `base_url=`, the string is not split into a provider and a name. The model is always `OpenAICompatible`,
and the whole string is the model name the server receives. Routers such as OpenRouter name their models this
way:

```python
import os

from alpineagents.models import resolve_model

model = resolve_model(
    "anthropic/claude-sonnet-5",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
# the same as OpenAICompatible("anthropic/claude-sonnet-5", base_url=..., api_key=...)
```

`"ollama/<name>"` is the one exception: the `ollama/` prefix is still removed.

A proxy that speaks the Anthropic API rather than the OpenAI one needs `Anthropic` itself. Create it directly:
`Anthropic("claude-sonnet-5", base_url="https://my-proxy/anthropic")`.

### When to create the Model directly

A model string with options is enough for the usual cases. Create `Anthropic(...)` or `OpenAICompatible(...)`
yourself when:

- the server speaks the Anthropic API at another URL (see above),
- you want the Model class to be obvious to the reader, or type checkers to check its settings,
- you [subclass or wrap a Model](#write-a-model).

## Model settings

Pass a Model object to change its settings:

```python
--8<-- "docs_src/model_settings.py"
```

- `OpenAICompatible` works with any server that speaks the OpenAI Chat Completions API: OpenAI, Ollama, vLLM,
  OpenRouter and others.
- Set `context_window` to the model's real window. `compact_if_full` and `state.context_used` use it.
- Creating a Model uses no network and needs no API key. The key is read at the first request.
- Retries are left to the provider SDK: `retries=2` by default.

See the [Models API](../api/models.md) for every setting.

## Cost

`state.usage.cost` is `None` unless the Model has prices. Prices are dollars per million tokens:

```python
from alpineagents import Anthropic, Price

model = Anthropic("claude-sonnet-5", price=Price(input=3, output=15, cache_read=0.3))
```

Each `Reply` records the model that answered in `reply.model`, as the provider reported it. It can differ from the
requested name: an alias resolved to a dated id, or a router that picked another model. The cost still uses the
requested Model's price.

## Write a Model

Subclass `Model` to add a provider. Implement `respond` and `context_window`:

```python
--8<-- "docs_src/custom_model.py"
```

Rules for a Model:

- Return a `Reply`. Do not change the State. The Agent records the reply.
- Put the model that actually answered in `Reply.model`: the name the provider reported, or the requested name
  if it did not say. A Model that falls back to another model puts the fallback's name there.
- Use no network and no credentials in `__init__`. Create the SDK client at the first request.
- Call `on_text` with each piece of text as it streams in, if `on_text` is not `None`.
- Wrap provider exceptions that finally fail as `RateLimitError`, `AuthError`, `ContextTooLongError` or
  `ProviderError`, with `raise ... from e`.
- Print nothing. Report events such as a fallback to another model with `on_event(ModelEvent(...))`.

`Model` has working defaults for everything else, including `arespond` for async code.

## Related

- [Testing](testing.md): `FakeModel` replaces the model in tests
