# SPDX-License-Identifier: Apache-2.0
"""Patch scaled_dot_product_attention to support TurboQuantKVCache.

When TurboQuantKVCache is detected, routes attention to:
  - Decode (L=1): cache.decode_attention() — Metal kernel, no dequant
  - Prefill (L>1): cache.prefill_attention() fast path, fallback to
    dequantize + mx.fast.scaled_dot_product_attention
"""

import logging
from typing import Optional

import mlx.core as mx

logger = logging.getLogger(__name__)

_PATCHED = False


def apply_turboquant_attention_patch() -> bool:
    """Monkey-patch mlx-lm's scaled_dot_product_attention for TurboQuant."""
    global _PATCHED
    if _PATCHED:
        return False

    try:
        from mlx_lm.models import base as mlx_base
    except ImportError:
        return False

    original_sdpa = mlx_base.scaled_dot_product_attention

    def patched_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask: Optional[mx.array],
        sinks: Optional[mx.array] = None,
    ) -> mx.array:
        from mlx_vlm.turboquant import TurboQuantKVCache as _TQCache
        from ..turboquant_kv import BatchTurboQuantKVCache

        # Unwrap VLM _IntOffsetCacheProxy to detect underlying TQ cache
        real_cache = cache
        if hasattr(cache, "_cache") and not isinstance(
            cache, (_TQCache, BatchTurboQuantKVCache)
        ):
            real_cache = cache._cache

        if isinstance(real_cache, (_TQCache, BatchTurboQuantKVCache)):
            if isinstance(real_cache, BatchTurboQuantKVCache):
                # Continuous batching uses left-padded merged caches. The fused
                # single-request decode kernel in mlx-vlm cannot consume that
                # layout directly, and fully dequantizing the merged batch cache
                # creates very large transient tensors on long VLM prompts.
                #
                # For stability, split the batch back into per-request caches
                # and use the single-request TurboQuant path for decode. This
                # keeps GPU working set bounded by one request at a time.
                if queries.shape[0] > 1:
                    outputs = []
                    for batch_idx in range(queries.shape[0]):
                        single_cache = real_cache.extract(batch_idx)
                        single_query = queries[batch_idx : batch_idx + 1]
                        single_mask = None
                        if isinstance(mask, str):
                            single_mask = mask
                        elif isinstance(mask, mx.array) and mask.shape[0] == queries.shape[0]:
                            single_mask = mask[batch_idx : batch_idx + 1]

                        if single_query.shape[-2] == 1:
                            # Avoid TurboQuant's fused decode kernels for the
                            # batched VLM path. On long prompts they can still
                            # trip Metal command-buffer failures, even after
                            # splitting the merged batch cache back into
                            # single-request caches. Dequantizing one request at
                            # a time keeps the transient footprint bounded while
                            # preserving quantization effects in the cache.
                            dequantized_keys, dequantized_values = single_cache.dequantize()
                            outputs.append(
                                mx.fast.scaled_dot_product_attention(
                                    single_query,
                                    dequantized_keys.astype(single_query.dtype),
                                    dequantized_values.astype(single_query.dtype),
                                    scale=scale,
                                    mask=single_mask,
                                )
                            )
                            continue

                        result = single_cache.prefill_attention(
                            single_query,
                            scale=scale,
                            mask=single_mask,
                        )
                        if result is None:
                            dequantized_keys, dequantized_values = single_cache.dequantize()
                            result = mx.fast.scaled_dot_product_attention(
                                single_query,
                                dequantized_keys.astype(single_query.dtype),
                                dequantized_values.astype(single_query.dtype),
                                scale=scale,
                                mask=single_mask,
                            )
                        outputs.append(result)

                    return mx.concatenate(outputs, axis=0)

                # B=1 merged caches can also use the extracted single-request
                # path to avoid materializing full fp16 KV tensors.
                single_cache = real_cache.extract(0)
                if queries.shape[-2] == 1:
                    dequantized_keys, dequantized_values = single_cache.dequantize()
                    return mx.fast.scaled_dot_product_attention(
                        queries,
                        dequantized_keys.astype(queries.dtype),
                        dequantized_values.astype(queries.dtype),
                        scale=scale,
                        mask=mask if isinstance(mask, mx.array) else None,
                    )
                result = single_cache.prefill_attention(
                    queries,
                    scale=scale,
                    mask=mask,
                )
                if result is not None:
                    return result
                dequantized_keys, dequantized_values = single_cache.dequantize()
                return mx.fast.scaled_dot_product_attention(
                    queries,
                    dequantized_keys.astype(queries.dtype),
                    dequantized_values.astype(queries.dtype),
                    scale=scale,
                    mask=mask,
                )

            if queries.shape[-2] == 1:
                return real_cache.decode_attention(
                    queries,
                    keys_state=keys,
                    values_state=values,
                    scale=scale,
                    mask=mask,
                )
            # Prefill: try quantized fast path, fallback to dequantize+SDPA
            result = real_cache.prefill_attention(
                queries, scale=scale, mask=mask,
            )
            if result is not None:
                return result
            dequantized_keys, dequantized_values = real_cache.dequantize()
            return mx.fast.scaled_dot_product_attention(
                queries,
                dequantized_keys.astype(queries.dtype),
                dequantized_values.astype(queries.dtype),
                scale=scale,
                mask=mask,
            )

        return original_sdpa(queries, keys, values, cache, scale, mask, sinks)

    # Patch the module attribute
    mlx_base.scaled_dot_product_attention = patched_sdpa

    # Also patch any model modules that already imported it locally
    # Covers both mlx_lm (LLM) and mlx_vlm (VLM) model modules
    import sys
    for mod_name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        if not (mod_name.startswith("mlx_lm.models.") or mod_name.startswith("mlx_vlm.models.")):
            continue
        if hasattr(mod, "scaled_dot_product_attention"):
            func = getattr(mod, "scaled_dot_product_attention")
            if func is original_sdpa or func is not patched_sdpa:
                setattr(mod, "scaled_dot_product_attention", patched_sdpa)

    # Also patch mlx_vlm.models.base if loaded
    try:
        from mlx_vlm.models import base as vlm_base
        if hasattr(vlm_base, "scaled_dot_product_attention"):
            vlm_base.scaled_dot_product_attention = patched_sdpa
    except ImportError:
        pass

    _PATCHED = True
    logger.info("TurboQuant attention patch applied")
    return True
