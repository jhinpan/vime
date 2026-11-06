from __future__ import annotations

import re

import torch

from .common import SafetensorReader, merge_gate_up, strip_mcore_wrappers

_DIRECT = {
    "embedding.word_embeddings.weight": "embed.weight",
    "decoder.final_layernorm.weight": "norm.weight",
    "output_layer.weight": "head.weight",
}

_LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "self_attention.linear_q_down_proj.weight": "attn.wq_a.weight",
    "self_attention.q_layernorm.weight": "attn.q_norm.weight",
    "self_attention.linear_q_up_proj.weight": "attn.wq_b.weight",
    "self_attention.linear_kv_proj.weight": "attn.wkv.weight",
    "self_attention.kv_layernorm.weight": "attn.kv_norm.weight",
    "self_attention.linear_o_group_proj": "attn.wo_a.weight",
    "self_attention.linear_proj.weight": "attn.wo_b.weight",
    "self_attention.core_attention.attn_sink": "attn.attn_sink",
    "self_attention.core_attention.compressor.linear_wkv.weight": "attn.compressor.wkv.weight",
    "self_attention.core_attention.compressor.linear_wgate.weight": "attn.compressor.wgate.weight",
    "self_attention.core_attention.compressor.norm.weight": "attn.compressor.norm.weight",
    "self_attention.core_attention.indexer.linear_wq_b.weight": "attn.indexer.wq_b.weight",
    "self_attention.core_attention.indexer.linear_weights_proj.weight": "attn.indexer.weights_proj.weight",
    "self_attention.core_attention.indexer.linear_wk.weight": "attn.indexer.wk.weight",
    "self_attention.core_attention.indexer.k_norm.weight": "attn.indexer.k_norm.weight",
    "self_attention_hyper_connection.mapping_proj.weight": "hc_attn_fn",
    "self_attention_hyper_connection.bias": "hc_attn_base",
    "mlp_hyper_connection.mapping_proj.weight": "hc_ffn_fn",
    "mlp_hyper_connection.bias": "hc_ffn_base",
    "v41_engram.linear_wkv.weight": "engram.wkv.weight",
    "v41_engram.q_weight": "engram.q_weight",
    "v41_engram.k_weight": "engram.k_weight",
    "pre_mlp_layernorm.weight": "ffn_norm.weight",
    "mlp.router.weight": "ffn.gate.weight",
    "mlp.router.expert_bias": "ffn.gate.bias",
    "mlp.shared_experts.linear_fc2.weight": "ffn.shared_experts.w2.weight",
}

# HF packs the [pre, post, res] alphas of each hyper-connection into one hc_*_scale tensor.
_ALPHA = re.compile(r"(self_attention|mlp)_hyper_connection\.alpha_(pre|post|res)")
_ALPHA_SEGMENT = {"pre": 0, "post": 1, "res": 2}


def deepseek_v41_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    if name in _DIRECT:
        return reader.get_tensor(_DIRECT[name])

    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise KeyError(f"Unsupported DeepSeek-V4.1 Megatron parameter {name!r}")
    prefix, rest = f"layers.{match.group(1)}", match.group(2)

    if rest in _LAYER:
        return reader.get_tensor(f"{prefix}.{_LAYER[rest]}")
    if alpha := _ALPHA.fullmatch(rest):
        branch, part = alpha.groups()
        scale = reader.get_tensor(f"{prefix}.hc_{'attn' if branch == 'self_attention' else 'ffn'}_scale")
        segment = _ALPHA_SEGMENT[part]
        return scale.reshape(-1)[segment : segment + 1].float()
    if rest == "mlp.shared_experts.linear_fc1.weight":
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.ffn.shared_experts.w1.weight"),
            reader.get_tensor(f"{prefix}.ffn.shared_experts.w3.weight"),
        )
    if expert := re.fullmatch(r"mlp\.experts\.linear_fc([12])\.weight(\d+)", rest):
        projection, index = expert.groups()
        if projection == "1":
            return merge_gate_up(
                reader.get_tensor(f"{prefix}.ffn.experts.{index}.w1.weight"),
                reader.get_tensor(f"{prefix}.ffn.experts.{index}.w3.weight"),
            )
        return reader.get_tensor(f"{prefix}.ffn.experts.{index}.w2.weight")
    raise KeyError(f"Unsupported DeepSeek-V4.1 Megatron layer parameter {rest!r}")
