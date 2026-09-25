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

## Write a Model

Subclass `Model` to add a provider. Implement `respond` and `context_window`:

```python
--8<-- "docs_src/custom_model.py"
```

Rules for a Model:

- Return a `Reply`. Do not change the State. The Agent records the reply.
- Use no network and no credentials in `__init__`. Create the SDK client at the first request.
- Call `on_text` with each piece of text as it streams in, if `on_text` is not `None`.
- Wrap provider exceptions that finally fail as `RateLimitError`, `AuthError`, `ContextTooLongError` or
  `ProviderError`, with `raise ... from e`.
- Print nothing. Report events such as a fallback to another model with `on_event(ModelEvent(...))`.

`Model` has working defaults for everything else, including `arespond` for async code.

## Related

- [Testing](testing.md): `FakeModel` replaces the model in tests
