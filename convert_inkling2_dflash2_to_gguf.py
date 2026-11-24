#!/usr/bin/env python3
# Converts an Inkling 2 DFlash2 drafter (e.g. incoai/Inkling-2-Mini-DFlash2) to GGUF.
# The vocabulary comes from the target Inkling 2 GGUF, and the drafter shares that model's
# token embeddings and output head at run time.
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(1, str(Path(__file__).parent))
if 'NO_LOCAL_GGUF' not in __import__('os').environ:
    sys.path.insert(1, str(Path(__file__).parent / 'gguf-py'))
import gguf

from conversion.qwen import DFlashModel

logger = logging.getLogger("inkling2-dflash2")


def target_layout(reader: gguf.GGUFReader) -> tuple[int, int, int, int, int]:
    # (physical layers, loop start, loop end, passes, embedding length)
    arch = reader.fields["general.architecture"].contents()
    if arch not in ("inkling2", "inkling2_loop"):
        raise ValueError(f"target is {arch!r}, not an Inkling 2 GGUF")
    n_phys = reader.fields[f"{arch}.block_count"].contents()
    start, end, count = 0, 0, 1
    if arch == "inkling2_loop":
        start = reader.fields[f"{arch}.loop.start"].contents()
        end   = reader.fields[f"{arch}.loop.end"].contents()
        count = reader.fields[f"{arch}.loop.count"].contents()
    return n_phys, start, end, count, reader.fields[f"{arch}.embedding_length"].contents()


def visit_to_layer(v: int, n_phys: int, start: int, end: int, count: int) -> int:
    # execution order: prelude [0, start), the core [start, end) count times, then the tail; n_visits is the final norm
    core = (end - start) * count
    if v < start:
        return v
    if v < start + core:
        return start + (v - start) % (end - start)
    return v - core + (end - start)


class Inkling2DFlash2Model(DFlashModel):
    model_arch = gguf.MODEL_ARCH.DFLASH
    target_gguf: gguf.GGUFReader

    def set_vocab(self):
        # the target's tokenizer keys as they are, so the vocabularies match exactly
        for key, field in self.target_gguf.fields.items():
            if not key.startswith("tokenizer.ggml.") or key == "tokenizer.ggml.mask_token_id":
                continue
            vtype = field.types[0]
            sub_type = field.types[1] if vtype == gguf.GGUFValueType.ARRAY else None
            self.gguf_writer.add_key_value(key, field.contents(), vtype, sub_type)
        self.gguf_writer.add_mask_token_id(self.hparams["dflash_config"]["mask_token_id"])

    def set_gguf_parameters(self):
        dflash_config = self.hparams["dflash_config"]
        layer_ids = dflash_config.pop("target_layer_ids")
        super().set_gguf_parameters()
        dflash_config["target_layer_ids"] = layer_ids

        # inkling2 records the residual state entering visit k as layer input k, and the final-norm
        # state as layer input n_visits: target_visits holds these ids; target_layers holds the
        # physical layer of each visit (n_phys for the final norm), which is what model browsers check
        n_phys, start, end, count, n_embd = target_layout(self.target_gguf)
        n_visits = n_phys + (count - 1) * (end - start)
        if layer_ids != sorted(set(layer_ids)) or layer_ids[0] < 0 or layer_ids[-1] != n_visits:
            raise ValueError(f"target_layer_ids {layer_ids} must be sorted, unique and end at {n_visits}")
        if self.hparams.get("num_target_layers", n_visits) != n_visits or self.hparams["hidden_size"] != n_embd:
            raise ValueError("the drafter was not trained for this target")
        layers = [visit_to_layer(v, n_phys, start, end, count) for v in layer_ids]
        self.gguf_writer.add_target_layers(layers)
        self.gguf_writer.add_array(f"{self.gguf_writer.arch}.target_visits", layer_ids)
        logger.info(f"target visits {layer_ids} -> physical layers {layers}")

        mode = dflash_config.get("selector_softcap_mode", "before_correction")
        if mode not in ("before_correction", "after_correction"):
            raise ValueError(f"unknown selector_softcap_mode {mode!r}")
        if mode == "after_correction":
            self.gguf_writer.add_bool(f"{self.gguf_writer.arch}.selector_softcap_after_correction", True)

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None):
        # the codebooks cover the padded vocabulary; the selector only indexes real tokens
        if name in ("model.candidate_selector.predecessor_codebook", "model.candidate_selector.successor_codebook"):
            n_vocab = len(self.target_gguf.fields["tokenizer.ggml.tokens"].data)
            data_torch = data_torch[:n_vocab]
        yield from super().modify_tensors(data_torch, name, bid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an Inkling 2 DFlash2 drafter to GGUF")
    parser.add_argument("model", type=Path, help="drafter directory (config.json, model.safetensors)")
    parser.add_argument("--target", type=Path, required=True, help="the Inkling 2 GGUF the drafter runs with")
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--outtype", choices=["bf16", "f16", "q8_0"], default="bf16")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    with open(args.model / "config.json", encoding="utf-8") as f:
        if "DFlash2DraftModel" not in json.load(f).get("architectures", []):
            raise ValueError("not a DFlash2 drafter")

    ftype = {
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "f16":  gguf.LlamaFileType.MOSTLY_F16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    }[args.outtype]

    Inkling2DFlash2Model.target_gguf = gguf.GGUFReader(args.target)
    model = Inkling2DFlash2Model(args.model, ftype, args.outfile)
    model.write()
    logger.info(f"wrote {args.outfile}")


if __name__ == "__main__":
    main()
