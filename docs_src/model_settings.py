import os

from alpineagents import Agent, Anthropic, OpenAICompatible

claude = Anthropic("claude-sonnet-5", thinking=True, max_tokens=16_000)

ollama = OpenAICompatible(
    "qwen3:8b",
    base_url="http://localhost:11434/v1",
    context_window=32_000,
)

openrouter = OpenAICompatible(
    "moonshotai/kimi-k2",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

agent = Agent(model=claude)
