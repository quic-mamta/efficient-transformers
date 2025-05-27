# -----------------------------------------------------------------------------
#
# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------


from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache

from QEfficient.customop import (
    CtxGatherFunc,
    CtxGatherFunc3D,
    CtxGatherFuncCB,
    CtxGatherFuncCB3D,
    CtxScatterFunc,
    CtxScatterFunc3D,
    CtxScatterFuncCB,
    CtxScatterFuncCB3D,
)
from QEfficient.customop.ctx_scatter_gather import CtxGatherH2OFunc


class QEffDynamicCache(DynamicCache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    - Optimized implementation for the Cloud AI 100 to reuse KV Cache.
    - get the position_ids input using kwargs.
    - Use custom Onnxscript ops to write optimized version to generate Onnx model.

    """

    def write_only(self, key_states, value_states, layer_idx, cache_kwargs):
        """
        Write in the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        """
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)

            # Scatter
            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.key_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
                )
                self.value_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
                )
            else:
                self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], position_ids, key_states)
                self.value_cache[layer_idx] = CtxScatterFunc.apply(
                    self.value_cache[layer_idx], position_ids, value_states
                )

    def read_only(self, layer_idx, cache_kwargs):
        """
        Reads the `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]
        position_ids = cache_kwargs.get("position_ids")
        batch_index = cache_kwargs.get("batch_index", None)
        ctx_len = k_out.shape[2]
        ctx_indices = torch.arange(ctx_len)[None, None, ...]
        gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
        invalid_mask = ctx_indices > gather_limit

        if torch.onnx.is_in_onnx_export():
            invalid_idx_value = torch.iinfo(torch.int32).max
        else:
            invalid_idx_value = 0

        ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)

        if batch_index is not None:
            k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices)
            v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices)
        else:
            k_out = CtxGatherFunc.apply(k_out, ctx_indices)
            v_out = CtxGatherFunc.apply(v_out, ctx_indices)

        v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)
        return k_out, v_out

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value form the kwargs

            # Scatter
            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.key_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
                )

                self.value_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
                )
            else:
                self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], position_ids, key_states)
                self.value_cache[layer_idx] = CtxScatterFunc.apply(
                    self.value_cache[layer_idx], position_ids, value_states
                )

            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Gather
            ctx_len = k_out.shape[2]
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit

            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
            else:
                invalid_idx_value = 0

            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
            if batch_index is not None:
                k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices)
                v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices)
            else:
                k_out = CtxGatherFunc.apply(k_out, ctx_indices)
                v_out = CtxGatherFunc.apply(v_out, ctx_indices)
            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)

        return k_out, v_out

    def update3D(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)

            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.key_cache[layer_idx] = CtxScatterFuncCB3D.apply(
                    self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
                )

                self.value_cache[layer_idx] = CtxScatterFuncCB3D.apply(
                    self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
                )

            else:
                self.key_cache[layer_idx] = CtxScatterFunc3D.apply(self.key_cache[layer_idx], position_ids, key_states)
                self.value_cache[layer_idx] = CtxScatterFunc3D.apply(
                    self.value_cache[layer_idx], position_ids, value_states
                )
            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Gather
            ctx_len = k_out.shape[1]
            ctx_indices = torch.arange(ctx_len)[None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values
            invalid_mask = ctx_indices > gather_limit
            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
            else:
                invalid_idx_value = 0
            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
            if batch_index is not None:
                k_out = CtxGatherFuncCB3D.apply(k_out, batch_index, ctx_indices)
                v_out = CtxGatherFuncCB3D.apply(v_out, batch_index, ctx_indices)
            else:
                k_out = CtxGatherFunc3D.apply(k_out, ctx_indices)
                v_out = CtxGatherFunc3D.apply(v_out, ctx_indices)

            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)

        return k_out, v_out


class QEffEncoderDecoderCache(EncoderDecoderCache):
    """
    Updated the `EncoderDecoderCache` to use the `QEffDynamicCache` for both self-attention and cross-attention caches.
    """

    @classmethod
    def from_legacy_cache(
        cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "EncoderDecoderCache":
        """Converts a cache in the legacy cache format into an equivalent `EncoderDecoderCache`."""
        cache = cls(
            self_attention_cache=QEffDynamicCache(),
            cross_attention_cache=QEffDynamicCache(),
        )
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx][:2]
                cache.self_attention_cache.update(key_states, value_states, layer_idx)
                if len(past_key_values[layer_idx]) > 2:
                    key_states, value_states = past_key_values[layer_idx][2:]
                    cache.cross_attention_cache.update(key_states, value_states, layer_idx)
                    cache.is_updated[layer_idx] = True
        return cache


class HHCache(Cache):
    """
    A cache that apply heavy-hitter oracle (https://proceedings.neurips.cc/paper_files/paper/2023/file/6ceefa7b15572587b78ecfcebb2827f8-Paper-Conference.pdf).
    Only the heavy-hitter and the recent tokens are stored in the cache.

    It stores the Key and Value states as a list of tensors, one for each layer. The expected shape for each tensor is
    `[batch_size, num_heads, seq_len, head_dim]`.

    Parameters:
        window_length (`int`):
            The length of the context window.
        num_hh_tokens (`int`):
            The number of heavy hitter tokens. See the original paper for more information.
    """

    def __init__(self, window_length: int, num_hh_tokens: int) -> None:
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_length = window_length
        self.num_hh_tokens = num_hh_tokens
        self.accumulated_attention_scores: List[torch.Tensor] = []
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        """
        Support for backwards-compatible `past_key_value` indexing, e.g. `past_key_value[0][0].shape[2]` to get the
        sequence length.
        """
        if layer_idx < len(self):
            return (
                self.key_cache[layer_idx],
                self.value_cache[layer_idx],
                self.accumulated_attention_scores[layer_idx],
            )
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        """
        Support for backwards-compatible `past_key_value` iteration, e.g. `for x in past_key_value:` to iterate over
        keys and values
        """
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx], self.accumulated_attention_scores[layer_idx])

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.key_cache)

    def initialize(self, key_states, value_states, attn_scores):
        self.key_cache.append(key_states)
        self.value_cache.append(value_states)
        self.accumulated_attention_scores.append(attn_scores)
        k_out, v_out, attn_sc_out = key_states, value_states, attn_scores
        return k_out, v_out, attn_sc_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # Workaround to make 'key_states.shape[-2] + past_key_value.get_seq_length(self.layer_idx)' <= window_length
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.window_length

    def update_kv(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.

        Return:
            A tuple containing the updated key and value states.
        """
        # Update the cache
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            k_out, v_out = key_states, value_states
        else:
            position_ids = cache_kwargs.get("position_ids")
            batch_index = cache_kwargs.get("batch_index", None)  # Check and fetch batch index value form the kwargs

            # Scatter
            if batch_index is not None:
                invalid_scatter_index = torch.iinfo(torch.int32).max
                scatter_position_ids = torch.where(position_ids < 0, invalid_scatter_index, position_ids)

                self.key_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.key_cache[layer_idx], batch_index, scatter_position_ids, key_states
                )

                self.value_cache[layer_idx] = CtxScatterFuncCB.apply(
                    self.value_cache[layer_idx], batch_index, scatter_position_ids, value_states
                )
            else:
                self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], position_ids, key_states)
                self.value_cache[layer_idx] = CtxScatterFunc.apply(
                    self.value_cache[layer_idx], position_ids, value_states
                )

            k_out, v_out = self.key_cache[layer_idx], self.value_cache[layer_idx]

            # Gather
            ctx_len = k_out.shape[2]
            ctx_indices = torch.arange(ctx_len)[None, None, ...]
            gather_limit = position_ids.max(1, keepdim=True).values.unsqueeze(1)
            invalid_mask = ctx_indices > gather_limit

            if torch.onnx.is_in_onnx_export():
                invalid_idx_value = torch.iinfo(torch.int32).max
            else:
                invalid_idx_value = 0

            ctx_indices = torch.where(invalid_mask, invalid_idx_value, ctx_indices)
            if batch_index is not None:
                k_out = CtxGatherFuncCB.apply(k_out, batch_index, ctx_indices)
                v_out = CtxGatherFuncCB.apply(v_out, batch_index, ctx_indices)
            else:
                k_out = CtxGatherFunc.apply(k_out, ctx_indices)
                v_out = CtxGatherFunc.apply(v_out, ctx_indices)
            v_out = torch.where(invalid_mask.unsqueeze(-1), torch.tensor(0.0, dtype=torch.float32), v_out)

        return k_out, v_out

    def update_slimming(
        self,
        attn_weights: torch.Tensor,
        num_kv_groups: int,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Slimming the cache based on accumulated attention scores, only keep heavy-hitters + local tokens.

        Parameters:
            attention_scores (`torch.Tensor`):
                Attention_scores for current steps.
            num_kv_groups (`int`):
                The number of kv groups in repeat kv.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. No additional arguments are used in `DynamicCache`.
        Return:
            A tuple containing the updated key and value states.
        """
        position_ids = cache_kwargs.get("position_ids")
        kv_seq_len = cache_kwargs.get("kv_seq_len")

        attn_scores = attn_weights.sum(2)

        cond = torch.tensor(position_ids.shape[-1] == 1)
        # TODO: try to remove this extra write
        gathered_scores = torch.where(
            cond, self.accumulated_attention_scores[layer_idx], self.accumulated_attention_scores[layer_idx] * 0
        )

        gathered_scores += attn_scores

        ######################
        # making read indices#
        ######################
        select_hh_scores = gathered_scores[:, :, : kv_seq_len // 2 + 1]
        remove_index = torch.argmin(select_hh_scores, dim=-1)
        read_indices = torch.arange(kv_seq_len)
        read_indices_exp = read_indices.expand(32, kv_seq_len)
        remove_index = remove_index.reshape(32, 1)
        manipulate_indices = torch.where(read_indices_exp >= remove_index, torch.tensor(1), torch.tensor(0))
        read_indices_exp = read_indices_exp + manipulate_indices
        if torch.onnx.is_in_onnx_export():
            invalid_idx_value = torch.iinfo(torch.int32).max
        else:
            invalid_idx_value = 0
        read_indices_exp[:, -1] = invalid_idx_value
        read_indices_for_gather = read_indices_exp.unsqueeze(-1).expand(-1, -1, 128).unsqueeze(0)

        ###########
        # where based H2O Magic
        ###########
        k_out = torch.where(
            position_ids.max() >= kv_seq_len,
            CtxGatherH2OFunc.apply(self.key_cache[layer_idx], read_indices_for_gather),
            self.key_cache[layer_idx],
        )
        v_out = torch.where(
            position_ids.max() >= kv_seq_len,
            CtxGatherH2OFunc.apply(self.value_cache[layer_idx], read_indices_for_gather),
            self.value_cache[layer_idx],
        )
        attn_scores = torch.where(
            position_ids.max() >= kv_seq_len,
            CtxGatherH2OFunc.apply(gathered_scores, read_indices_exp.unsqueeze(0)),
            gathered_scores,
        )

        # --------- scatter ------------- #
        bs = self.key_cache[layer_idx].shape[0]
        write_indices = torch.arange(kv_seq_len).reshape(bs, kv_seq_len)
        self.key_cache[layer_idx] = CtxScatterFunc.apply(self.key_cache[layer_idx], write_indices, k_out)
        self.value_cache[layer_idx] = CtxScatterFunc.apply(self.value_cache[layer_idx], write_indices, v_out)
        self.accumulated_attention_scores[layer_idx] = CtxScatterFunc.apply(
            self.accumulated_attention_scores[layer_idx], write_indices, attn_scores
        )

        # Write all updates values

        ######
        # H2O done
        ######

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += (
                [self.key_cache[layer_idx], self.value_cache[layer_idx], self.accumulated_attention_scores[layer_idx]],
            )
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls, window_length: int, num_hh_tokens: int, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> "DynamicCache":
        """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
        cache = cls(window_length, num_hh_tokens)
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states = past_key_values[layer_idx][0]
                value_states = past_key_values[layer_idx][1]
                accumulated_attention_scores = past_key_values[layer_idx][2]
                cache.initialize(key_states, value_states, accumulated_attention_scores)
        return cache
