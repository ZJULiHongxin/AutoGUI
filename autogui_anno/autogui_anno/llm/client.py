# autogui_anno/autogui_anno/llm/client.py
"""One OpenAI-compatible LLM client with bounded retry and token accounting."""
from __future__ import annotations
import tiktoken
import openai
from autogui_anno.config import ModelSpec
from autogui_anno.llm.registry import resolve_api_key

_ENCODING = tiktoken.encoding_for_model("gpt-3.5-turbo")


class LLMClient:
    def __init__(self, spec: ModelSpec, *, max_tokens: int = 4096,
                 max_retries: int = 3, client_factory=None):
        self.model = spec.model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        if client_factory is not None:
            self.client = client_factory()
        else:
            self.client = openai.OpenAI(base_url=spec.base_url,
                                        api_key=resolve_api_key(spec))
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._query_count = 0

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    @property
    def query_count(self) -> int:
        return self._query_count

    def num_tokens(self, text: str) -> int:
        return len(_ENCODING.encode(text))

    def query(self, messages, *, temperature: float = 1.0, repeat: int = 1,
              stop=None, do_sample: bool = True) -> list:
        last_exc = None
        for _ in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=temperature, n=repeat,
                    max_tokens=self.max_tokens, stop=stop,
                )
                self._prompt_tokens += resp.usage.prompt_tokens
                self._completion_tokens += resp.usage.completion_tokens
                self._query_count += 1
                return [c.message.content for c in resp.choices]
            except openai.BadRequestError:
                raise
            except Exception as e:  # transient
                last_exc = e
        raise RuntimeError(
            f"LLM query failed after {self.max_retries} attempts: {last_exc}"
        )
