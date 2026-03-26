from __future__ import annotations

class StreamGenerationConfig:
    """Minimal compatibility shim for legacy Qwen streaming helpers."""

    def __init__(self, *args, do_stream: bool = False, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.do_stream = do_stream


class NewGenerationMixin:
    """Compatibility shim expected by legacy Qwen modeling code.

    The training stack here does not use the legacy `chat_stream()` path.
    We only need these symbols to exist so `trust_remote_code=True` models
    can be imported successfully.
    """

    def generate(self, *args, **kwargs):
        raise NotImplementedError(
            "Legacy stream generation is not supported in this environment."
        )

    def sample_stream(self, *args, **kwargs):
        raise NotImplementedError(
            "Legacy stream generation is not supported in this environment."
        )
