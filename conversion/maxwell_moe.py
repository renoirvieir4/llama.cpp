from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("MaxwellMoeForCausalLM")
@ModelBase.example("maxwell-model/Maxwell-V1-Flash")
class MaxwellMoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MAXWELL_MOE

    _experts: list[dict[str, Tensor]] | None = None
    _hc_scale: dict[str, dict[str, Tensor]] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hparams = self.hparams

        if (act := hparams.get("hidden_act", "silu")) != "silu":
            raise ValueError(f"MaxwellMoe: unsupported hidden_act {act!r}")
        if hparams.get("rope_scaling") is not None:
            raise ValueError("MaxwellMoe: rope_scaling is not supported")
        if not hparams.get("qk_layernorm", False):
            raise ValueError("MaxwellMoe: only qk_layernorm=true is supported")
        if not hparams.get("enable_hyper_connections", False):
            raise ValueError("MaxwellMoe: only enable_hyper_connections=true is supported")
        if (n_hc := hparams.get("num_residual_streams")) != 4:
            raise ValueError(f"MaxwellMoe: num_residual_streams={n_hc!r}, only 4 is supported")
        if (score := hparams.get("router_score_function", "softmax")) != "sigmoid":
            raise ValueError(f"MaxwellMoe: unsupported router_score_function {score!r}")
        if not hparams.get("moe_router_enable_expert_bias", False):
            raise ValueError("MaxwellMoe: only moe_router_enable_expert_bias=true is supported")
        if hparams.get("use_shared_expert_gate", False):
            raise ValueError("MaxwellMoe: use_shared_expert_gate=true is not supported")
        if not hparams.get("shared_expert_intermediate_size"):
            raise ValueError("MaxwellMoe: a shared expert is required")

        # the reference makes every layer in mlp_only_layers dense, llama.cpp expects them to lead
        dense = sorted(hparams.get("mlp_only_layers") or [])
        if dense != list(range(len(dense))):
            raise ValueError(f"MaxwellMoe: mlp_only_layers {dense} is not a leading block")
        self.n_dense_lead = len(dense)

        # the backbone attention is block-sparse GQA with one indexer query head per KV group;
        # the block layout is stored for the sparse path, the dense path ignores it
        if (variant := hparams.get("attention_variant", "block_sparse_gqa")) != "block_sparse_gqa":
            raise ValueError(f"MaxwellMoe: unsupported attention_variant {variant!r}")
        if hparams["block_sparse_gqa_indexer_n_heads"] != hparams["num_key_value_heads"]:
            raise ValueError("MaxwellMoe: expected one indexer head per KV group")
        if any(f != 1 for f in hparams.get("block_sparse_gqa_layer_freq", [])):
            raise ValueError("MaxwellMoe: block_sparse_gqa_layer_freq must select every layer")
        block = hparams["block_sparse_gqa_block_size"]
        for key in ("block_sparse_gqa_prefix_window_size", "block_sparse_gqa_local_window_size"):
            if hparams[key] % block != 0:
                raise ValueError(f"MaxwellMoe: {key} is not a multiple of the block size {block}")

    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        # tokenizer_config.json sets add_bos_token, but the tokenizer has no post-processor that adds <s>,
        # so neither transformers nor the chat template ever put one in front of the prompt
        special_vocab.add_special_token["bos"] = False
        special_vocab.add_to_gguf(self.gguf_writer)

        # eos_token_id is </s>, the chat turn ends with <|maxwell_end|>: stop on it too
        tokenizer_config = self.dir_model / "tokenizer_config.json"
        if tokenizer_config.is_file():
            import json
            with open(tokenizer_config, encoding="utf-8") as f:
                added = json.load(f).get("added_tokens_decoder", {})
            for tid, tok in added.items():
                if tok.get("content") == "<|maxwell_end|>":
                    self.gguf_writer.add_eot_token_id(int(tid))
                    logger.info(f"gguf: eot_token_id = {tid}")
                    break

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        head_dim = hparams["attention_head_size"]
        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(head_dim)
        self.gguf_writer.add_rope_dimension_count(head_dim)

        self.gguf_writer.add_leading_dense_block_count(self.n_dense_lead)
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["shared_expert_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(1)
        # sigmoid scores, expert_bias only for the selection, the chosen scores renormalized
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        self.gguf_writer.add_expert_weights_norm(True)
        self.gguf_writer.add_expert_weights_scale(1.0)

        # mHC, the reference hardcodes eps = 1e-6
        self.gguf_writer.add_hyper_connection_count(hparams["num_residual_streams"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hparams["mhc_sinkhorn_iterations"])
        self.gguf_writer.add_hyper_connection_epsilon(1e-6)

        # block-sparse GQA: fixed prefix and local blocks plus top_k remote blocks per KV group
        block = hparams["block_sparse_gqa_block_size"]
        self.gguf_writer.add_indexer_head_count(hparams["block_sparse_gqa_indexer_n_heads"])
        self.gguf_writer.add_indexer_key_length(hparams["block_sparse_gqa_indexer_head_dim"])
        self.gguf_writer.add_indexer_top_k(hparams["block_sparse_gqa_num_remote_blocks"])
        self.gguf_writer.add_indexer_block_size(block)
        self.gguf_writer.add_indexer_local_blocks(hparams["block_sparse_gqa_local_window_size"] // block)
        self.gguf_writer.add_indexer_prefix_blocks(hparams["block_sparse_gqa_prefix_window_size"] // block)

        if hparams.get("mtp_num_layers", 0):
            logger.warning("MaxwellMoe: dropping %d MTP layer(s), they only run for speculative decoding",
                           hparams["mtp_num_layers"])

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if item[0].startswith("mtp_layers."):
            return None
        return super().filter_tensors(item)

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # the reference runs the mHC mappings in fp32
        if new_name.endswith(("hc_attn_fn.weight", "hc_ffn_fn.weight", "output_hc_fn.weight")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.find("block_sparse_moe.experts.") != -1:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) < n_experts * 3:
                return

            for wid in ("w1", "w2", "w3"):
                datas = [self._experts[bid].pop(f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{wid}.weight")
                         for xid in range(n_experts)]
                merged = f"model.layers.{bid}.block_sparse_moe.experts.{wid}.weight"
                yield from super().modify_tensors(torch.stack(datas, dim=0), merged, bid)
            return

        if name.endswith(".block_sparse_moe.expert_bias"):
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, bid, ".bias"), data_torch
            return

        for hc, fn_scale, fn_base in (("self_attn_hyper_connection", gguf.MODEL_TENSOR.HC_ATTN_SCALE, gguf.MODEL_TENSOR.HC_ATTN_BASE),
                                      ("mlp_hyper_connection",       gguf.MODEL_TENSOR.HC_FFN_SCALE,  gguf.MODEL_TENSOR.HC_FFN_BASE)):
            prefix = f"model.layers.{bid}.{hc}."
            if not name.startswith(prefix):
                continue
            part = name[len(prefix):]
            if part == "bias":
                yield self.format_tensor_name(fn_base, bid), data_torch
                return
            if part in ("alpha_pre", "alpha_post", "alpha_res"):
                # the three gating factors scale the pre, post and residual mixes, in that order
                if self._hc_scale is None:
                    self._hc_scale = {}
                parts = self._hc_scale.setdefault(prefix, {})
                parts[part] = data_torch
                if len(parts) == 3:
                    del self._hc_scale[prefix]
                    scale = torch.cat([parts["alpha_pre"], parts["alpha_post"], parts["alpha_res"]]).reshape(3)
                    yield self.format_tensor_name(fn_scale, bid), scale
                return

        if name in ("model.hc_head.hc_head_fn", "model.hc_head.hc_head_base", "model.hc_head.hc_head_scale"):
            name += ".weight"

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
        if self._hc_scale:
            raise ValueError(f"Unprocessed mHC gating factors: {sorted(self._hc_scale)}")
