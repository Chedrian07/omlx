# SPDX-License-Identifier: Apache-2.0
"""Fallback attention path for batched uniform KV quantization."""

from __future__ import annotations

import logging

import mlx.core as mx

from ..batch_quantized_kv import BatchQuantizedKVCache

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_uniform_kv_attention_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from mlx_lm.models import base as mlx_base
    from mlx_vlm.models import base as vlm_base

    original_mlx_sdpa = mlx_base.scaled_dot_product_attention
    original_vlm_sdpa = vlm_base.scaled_dot_product_attention

    def _unwrap_cache(cache):
        if hasattr(cache, "_cache"):
            return cache._cache
        return cache

    def _mlx_patched(queries, keys, values, cache, scale, mask, sinks=None):
        real_cache = _unwrap_cache(cache)
        if isinstance(real_cache, BatchQuantizedKVCache):
            if sinks is not None:
                raise ValueError(
                    "BatchQuantizedKVCache does not support attention sinks."
                )
            dq_keys, dq_values = real_cache.dequantize(keys, values)
            return mx.fast.scaled_dot_product_attention(
                queries,
                dq_keys.astype(queries.dtype),
                dq_values.astype(queries.dtype),
                scale=scale,
                mask=mask,
            )
        return original_mlx_sdpa(queries, keys, values, cache, scale, mask, sinks)

    def _vlm_patched(queries, keys, values, cache, scale, mask, sinks=None):
        real_cache = _unwrap_cache(cache)
        if isinstance(real_cache, BatchQuantizedKVCache):
            if sinks is not None:
                raise ValueError(
                    "BatchQuantizedKVCache does not support attention sinks."
                )
            dq_keys, dq_values = real_cache.dequantize(keys, values)
            return mx.fast.scaled_dot_product_attention(
                queries,
                dq_keys.astype(queries.dtype),
                dq_values.astype(queries.dtype),
                scale=scale,
                mask=mask,
            )
        return original_vlm_sdpa(queries, keys, values, cache, scale, mask, sinks)

    mlx_base.scaled_dot_product_attention = _mlx_patched
    vlm_base.scaled_dot_product_attention = _vlm_patched

    import sys

    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (
            mod_name.startswith("mlx_lm.models.") or mod_name.startswith("mlx_vlm.models.")
        ):
            continue
        if hasattr(mod, "scaled_dot_product_attention"):
            func = getattr(mod, "scaled_dot_product_attention")
            if func is original_mlx_sdpa or func is original_vlm_sdpa:
                setattr(mod, "scaled_dot_product_attention", _vlm_patched)

    _PATCHED = True
    logger.info("Uniform KV attention patch applied")
