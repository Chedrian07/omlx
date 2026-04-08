# SPDX-License-Identifier: Apache-2.0
"""Batch-aware uniform KV cache quantization helpers."""

from __future__ import annotations

from typing import Any, Iterable, List

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce

from mlx_lm.models.cache import BatchKVCache, CacheList, QuantizedKVCache, dynamic_roll


def _qtree_slice_prefix(tree, length: int):
    return tree_map(lambda x: x[..., :length, :], tree)


def _qtree_batch_select(tree, batch_indices):
    return tree_map(lambda x: x[batch_indices], tree)


def _qtree_seq_slice(tree, start: int, end: int):
    return tree_map(lambda x: x[..., start:end, :], tree)


def _qtree_pad(tree, pad):
    return tree_map(lambda x: mx.pad(x, pad), tree)


def _qtree_concat(parts: Iterable[Any]):
    parts = list(parts)
    return tree_map(lambda *xs: mx.concatenate(xs, axis=0), *parts)


class BatchQuantizedKVCache:
    """Batch variant of mlx-lm's QuantizedKVCache."""

    step = 256

    def __init__(self, left_padding: List[int], group_size: int = 64, bits: int = 4):
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-l for l in left_padding])
        self._idx = 0
        self._right_padding = None
        self.group_size = group_size
        self.bits = bits

    @classmethod
    def from_batch_kv(
        cls,
        cache: BatchKVCache,
        *,
        group_size: int = 64,
        bits: int = 4,
    ) -> "BatchQuantizedKVCache":
        quant = cls(
            [int(v) for v in cache.left_padding.tolist()],
            group_size=group_size,
            bits=bits,
        )
        quant.offset = cache.offset
        quant.left_padding = cache.left_padding
        quant._idx = cache._idx
        quant._right_padding = cache._right_padding
        if cache.keys is not None:
            quant.keys = mx.quantize(cache.keys, group_size=group_size, bits=bits)
            quant.values = mx.quantize(cache.values, group_size=group_size, bits=bits)
        return quant

    def update_and_fetch(self, keys, values):
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        prev = self._idx

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self.bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (B, n_kv_heads, new_steps)

            def init_quant(dim, dtype):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self.group_size), dtype=dtype),
                    mx.zeros((*shape, dim // self.group_size), dtype=dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = _qtree_slice_prefix(self.keys, prev)
                    self.values = _qtree_slice_prefix(self.values, prev)

                self.keys = tree_map(expand_quant, self.keys)
                self.values = tree_map(expand_quant, self.values)
            else:
                self.keys = init_quant(k_head_dim, keys.dtype)
                self.values = init_quant(v_head_dim, values.dtype)

        self.offset += num_steps
        self._idx += num_steps

        q_keys = mx.quantize(keys, group_size=self.group_size, bits=self.bits)
        q_values = mx.quantize(values, group_size=self.group_size, bits=self.bits)
        for i in range(len(self.keys)):
            self.keys[i][..., prev:self._idx, :] = q_keys[i]
            self.values[i][..., prev:self._idx, :] = q_values[i]

        return _qtree_slice_prefix(self.keys, self._idx), _qtree_slice_prefix(
            self.values, self._idx
        )

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchQuantizedKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self.keys = tree_map(
                lambda x: dynamic_roll(x, padding[:, None], axis=2), self.keys
            )
            self.values = tree_map(
                lambda x: dynamic_roll(x, padding[:, None], axis=2), self.values
            )
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        k = self.keys
        v = self.values
        if k is not None and self._idx < k[0].shape[2]:
            k = _qtree_slice_prefix(k, self._idx)
            v = _qtree_slice_prefix(v, self._idx)
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        if self.keys is not None:
            self._idx = self.keys[0].shape[2]
        else:
            self._idx = 0

    @property
    def meta_state(self):
        return tuple(map(str, (self._idx, self.group_size, self.bits)))

    @meta_state.setter
    def meta_state(self, v):
        self._idx, self.group_size, self.bits = map(int, v[:3])

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return BatchKVCache.make_mask(self, N, return_array=return_array, **kwargs)

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys = _qtree_batch_select(self.keys, batch_indices)
            self.values = _qtree_batch_select(self.values, batch_indices)
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]

        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = _qtree_seq_slice(self.keys, min_left_pad, self._idx)
                self.values = _qtree_seq_slice(self.values, min_left_pad, self._idx)
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        pad_size = max(
            self.keys[0].shape[2] if self.keys is not None else 0,
            other.keys[0].shape[2] if other.keys is not None else 0,
        )

        template_state = self.keys or other.keys
        template_value = self.values or other.values
        assert template_state is not None and template_value is not None

        def empty_like(tree):
            return tree_map(
                lambda x: mx.zeros(
                    (
                        0,
                        x.shape[1],
                        0,
                        x.shape[-1],
                    ),
                    dtype=x.dtype,
                ),
                tree,
            )

        def pad_state(cache, state_tree, value_tree):
            if state_tree is None:
                state_tree = empty_like(template_state)
                value_tree = empty_like(template_value)
            left = max_idx - cache._idx
            right = pad_size - state_tree[0].shape[2] - left
            if right < 0:
                state_tree = _qtree_slice_prefix(state_tree, state_tree[0].shape[2] + right)
                value_tree = _qtree_slice_prefix(value_tree, value_tree[0].shape[2] + right)
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                state_tree = _qtree_pad(state_tree, pad)
                value_tree = _qtree_pad(value_tree, pad)
            left_padding = cache.left_padding + left
            return state_tree, value_tree, cache.offset, left_padding

        parts = [
            pad_state(self, self.keys, self.values),
            pad_state(other, other.keys, other.values),
        ]
        self.keys = _qtree_concat([parts[0][0], parts[1][0]])
        self.values = _qtree_concat([parts[0][1], parts[1][1]])
        self.offset = mx.concatenate([parts[0][2], parts[1][2]])
        self.left_padding = mx.concatenate([parts[0][3], parts[1][3]])
        self._idx = max_idx

    def extract(self, idx):
        cache = QuantizedKVCache(group_size=self.group_size, bits=self.bits)
        padding = self.left_padding[idx].item()
        cache.keys = tree_map(
            lambda x: mx.contiguous(x[idx : idx + 1, :, padding : self._idx, :]),
            self.keys,
        )
        cache.values = tree_map(
            lambda x: mx.contiguous(x[idx : idx + 1, :, padding : self._idx, :]),
            self.values,
        )
        cache.offset = int(cache.keys[0].shape[2]) if cache.keys is not None else 0
        return cache

    def size(self):
        return self._idx

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        return tree_reduce(lambda a, x: a + x.nbytes, (self.keys, self.values), 0)


def quantize_batch_cache(
    cache_obj: Any,
    *,
    bits: int,
    group_size: int,
    quantized_kv_start: int,
) -> Any:
    """Recursively quantize BatchKVCache entries once the threshold is reached."""

    if isinstance(cache_obj, BatchQuantizedKVCache):
        return cache_obj

    if isinstance(cache_obj, BatchKVCache):
        if cache_obj._idx < quantized_kv_start:
            return cache_obj
        return BatchQuantizedKVCache.from_batch_kv(
            cache_obj,
            group_size=group_size,
            bits=bits,
        )

    if isinstance(cache_obj, CacheList):
        cache_obj.caches = tuple(
            quantize_batch_cache(
                sub_cache,
                bits=bits,
                group_size=group_size,
                quantized_kv_start=quantized_kv_start,
            )
            for sub_cache in cache_obj.caches
        )
        return cache_obj

    return cache_obj
