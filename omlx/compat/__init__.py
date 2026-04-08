"""Compatibility helpers for projects migrating onto oMLX."""

from .vlm import (
    apply_chat_template,
    load,
    make_prompt_cache,
    quantize_prompt_cache,
    stream_generate,
)

__all__ = [
    "apply_chat_template",
    "load",
    "make_prompt_cache",
    "quantize_prompt_cache",
    "stream_generate",
]
