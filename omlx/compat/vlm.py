"""mlx-vlm surface wrappers pinned through oMLX's dependency set.

This module gives downstream projects a stable import target while oMLX keeps
tracking the exact mlx-vlm / mlx-lm revisions it is validated against.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from types import ModuleType
from typing import Any


@lru_cache(maxsize=1)
def _modules() -> tuple[ModuleType, ModuleType, ModuleType, ModuleType]:
    try:
        api_module = import_module("mlx_vlm")
        generate_module = import_module("mlx_vlm.generate")
        cache_module = import_module("mlx_vlm.models.cache")
        prompt_utils_module = import_module("mlx_vlm.prompt_utils")
    except ImportError as exc:
        raise ImportError(
            "oMLX VLM compatibility requires the bundled mlx-vlm dependency. "
            "Install omlx from a branch or release that includes mlx-vlm."
        ) from exc

    return api_module, generate_module, cache_module, prompt_utils_module


def load(
    model_path: str,
    *,
    adapter_path: str | None = None,
    revision: str | None = None,
    trust_remote_code: bool = False,
) -> tuple[Any, Any]:
    api_module, _, _, _ = _modules()
    return api_module.load(
        model_path,
        adapter_path=adapter_path,
        revision=revision,
        trust_remote_code=trust_remote_code,
    )


def stream_generate(
    model: Any,
    processor: Any,
    prompt: str,
    *,
    image: list[str] | None = None,
    **kwargs: Any,
):
    _, generate_module, _, _ = _modules()
    return generate_module.stream_generate(
        model,
        processor,
        prompt,
        image=image,
        **kwargs,
    )


def make_prompt_cache(model: Any, max_kv_size: int | None = None) -> list[Any]:
    _, _, cache_module, _ = _modules()
    return cache_module.make_prompt_cache(model, max_kv_size=max_kv_size)


def quantize_prompt_cache(
    prompt_cache: list[Any],
    *,
    quantized_kv_start: int,
    kv_group_size: int,
    kv_bits: float | None,
    kv_quant_scheme: str = "uniform",
) -> None:
    _, generate_module, _, _ = _modules()
    generate_module.maybe_quantize_kv_cache(
        prompt_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
        kv_quant_scheme=kv_quant_scheme,
    )


def apply_chat_template(
    processor: Any,
    model_config: Any,
    messages: list[dict[str, Any]],
    *,
    num_images: int,
    enable_thinking: bool,
) -> str:
    _, _, _, prompt_utils_module = _modules()
    return prompt_utils_module.apply_chat_template(
        processor,
        model_config,
        messages,
        num_images=num_images,
        enable_thinking=enable_thinking,
    )


__all__ = [
    "apply_chat_template",
    "load",
    "make_prompt_cache",
    "quantize_prompt_cache",
    "stream_generate",
]
