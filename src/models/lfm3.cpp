#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

void llama_model_lfm3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        hparams.is_recr_impl[il] = hparams.n_head_kv(il) == 0;
    }

    // LFM3 always mixes full and sliding window attention layers
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);
    if (!hparams.is_swa_any()) {
        throw std::runtime_error("LFM3 requires sliding window attention layers");
    }
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa);

    // d1: full attention layers read the prompt both ways, sliding window layers stay causal
    if (!hparams.causal_attn) {
        hparams.non_causal_type = LLAMA_NON_CAUSAL_TYPE_DENSE_ONLY;
    }

    uint32_t version = 0;
    if (ml.get_key("lfm3.loop_mhc.version", version, false)) {
        uint32_t streams, passes, physical, iterations;
        ml.get_key("lfm3.loop_mhc.streams", streams);
        ml.get_key("lfm3.loop_mhc.passes", passes);
        ml.get_key("lfm3.loop_mhc.physical_layers", physical);
        ml.get_key("lfm3.loop_mhc.iterations", iterations);
        if (version != 1 || streams != 4 || passes != 2 || physical != 24 ||
            iterations != 20 || hparams.n_layer() != 48 || !hparams.causal_attn) {
            throw std::runtime_error("Unsupported LFM3 loop/mHC settings");
        }
        loop_mhc = true;
    }

    uint32_t refine_version = 0;
    if (ml.get_key("lfm3.loop_refine.version", refine_version, false)) {
        uint32_t begin, end, passes, streams, physical, rank;
        ml.get_key("lfm3.loop_refine.begin", begin);
        ml.get_key("lfm3.loop_refine.end", end);
        ml.get_key("lfm3.loop_refine.passes", passes);
        ml.get_key("lfm3.loop_refine.streams", streams);
        ml.get_key("lfm3.loop_refine.physical_layers", physical);
        ml.get_key("lfm3.loop_refine.delta_rank", rank);
        // begin and end are physical layers; the extra pass visits them again right after the first pass reaches end
        if (refine_version != 1 || loop_mhc || passes != 2 || streams != 2 || begin >= end || end > physical || rank == 0 ||
            hparams.n_layer() != physical + (end - begin) || !hparams.causal_attn) {
            throw std::runtime_error("Unsupported LFM3 refine loop settings");
        }
        loop_refine  = true;
        refine_begin = end;
        refine_end   = 2 * end - begin;
        refine_rank  = rank;
        LLAMA_LOG_INFO("%s: LFM3 refine loop: layers [%u, %u) twice, %u execution layers, delta rank %u\n", __func__, begin, end, hparams.n_layer(), rank);
    }

    switch (hparams.n_ff()) {
        case 7680: type = LLM_TYPE_1_7B; break;
        default:   type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_lfm3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == NULL) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    int rope_freqs_flag = 0;

    if (loop_refine) {
        refine_visits.assign(n_layer, {});
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];
        if (loop_refine && i >= refine_begin && i < refine_end) {
            // the extra pass reuses the physical weights and adds its own routers and deltas
            layer = layers[i - (refine_end - refine_begin)];
            auto & visit = refine_visits[i];
            const int64_t rank = refine_rank;
            auto delta = [&](llm_tensor type, int64_t n_in, int64_t n_out) {
                llm_lfm_delta d;
                d.down = create_tensor(tn(type, "delta_down", i), {n_in, rank}, 0);
                d.up   = create_tensor(tn(type, "delta_up",   i), {rank, n_out}, 0);
                return d;
            };
            layer.hc_attn_fn   = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,   "weight", i), {2 * n_embd, 8}, 0);
            layer.hc_attn_base = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE, "bias",   i), {8}, 0);
            layer.hc_ffn_fn    = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,    "weight", i), {2 * n_embd, 8}, 0);
            layer.hc_ffn_base  = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,  "bias",   i), {8}, 0);
            visit.ffn_gate = delta(LLM_TENSOR_FFN_GATE, n_embd, n_ff);
            visit.ffn_up   = delta(LLM_TENSOR_FFN_UP,   n_embd, n_ff);
            visit.ffn_down = delta(LLM_TENSOR_FFN_DOWN, n_ff,   n_embd);
            if (hparams.is_recr(i)) {
                visit.conv_in  = delta(LLM_TENSOR_SHORTCONV_INPROJ,  n_embd, 3 * n_embd);
                visit.conv_out = delta(LLM_TENSOR_SHORTCONV_OUTPROJ, n_embd, n_embd);
            } else {
                visit.wq = delta(LLM_TENSOR_ATTN_Q,   n_embd, n_embd_head_k * n_head);
                visit.wk = delta(LLM_TENSOR_ATTN_K,   n_embd, hparams.n_embd_k_gqa(i));
                visit.wv = delta(LLM_TENSOR_ATTN_V,   n_embd, hparams.n_embd_v_gqa(i));
                visit.wo = delta(LLM_TENSOR_ATTN_OUT, n_embd_head_k * n_head, n_embd);
            }
            std::vector<ggml_tensor *> fp32 = {layer.hc_attn_fn, layer.hc_attn_base, layer.hc_ffn_fn, layer.hc_ffn_base};
            if (i == refine_end - 1) {
                loop_gate_logits = create_tensor(tn(LLM_TENSOR_LOOP_GATE, "logits", i), {n_embd}, 0);
                loop_gate_token  = create_tensor(tn(LLM_TENSOR_LOOP_GATE, "token",  i), {n_embd}, 0);
                fp32.push_back(loop_gate_logits);
                fp32.push_back(loop_gate_token);
            }
            for (auto * tensor : fp32) {
                if (tensor->type != GGML_TYPE_F32) {
                    throw std::runtime_error("LFM3 refine routers and gate must be FP32");
                }
            }
            continue;
        }
        if (loop_mhc && i >= 24) {
            layer = layers[i - 24];
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {1}, 0);
            layer.hc_ffn_scale = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE, "weight", i), {1}, 0);
            if (layer.hc_attn_scale->type != GGML_TYPE_F32 || layer.hc_ffn_scale->type != GGML_TYPE_F32) {
                throw std::runtime_error("LFM3 loop gates must be FP32");
            }
            continue;
        }
        if (loop_mhc) {
            layer.hc_attn_fn = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN, "weight", i), {4 * n_embd, 24}, 0);
            layer.hc_attn_base = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE, "bias", i), {24}, 0);
            layer.hc_ffn_fn = create_tensor(tn(LLM_TENSOR_HC_FFN_FN, "weight", i), {4 * n_embd, 24}, 0);
            layer.hc_ffn_base = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE, "bias", i), {24}, 0);
            for (auto * tensor : {layer.hc_attn_fn, layer.hc_attn_base, layer.hc_ffn_fn, layer.hc_ffn_base}) {
                if (tensor->type != GGML_TYPE_F32) {
                    throw std::runtime_error("LFM3 mHC routers must be FP32");
                }
            }
        }

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), {n_embd}, 0);
        // absent when use_sandwich_norm is false
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        if ((layer.attn_post_norm == nullptr) != (layer.ffn_post_norm == nullptr)) {
            throw std::runtime_error("LFM3 requires both post norms or neither");
        }

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        if (!hparams.is_recr(i)) {
            // note: n_embd_head_k * n_head may differ from n_embd
            const int64_t n_embd_q = n_embd_head_k * n_head;
            const int64_t n_embd_k = hparams.n_embd_k_gqa(i);
            const int64_t n_embd_v = hparams.n_embd_v_gqa(i);
            if (n_embd_k != n_embd_v) {
                throw std::runtime_error("LFM3 requires equal key and value sizes");
            }

            layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
            layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

            create_tensor_qkv(layer, i, n_embd, n_embd_q, n_embd_k, n_embd_v, 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd}, 0);

            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head}, 0);

            if (!hparams.is_swa(i)) {
                layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head_k / 2}, rope_freqs_flag);
                rope_freqs_flag = TENSOR_DUPLICATED;
            }
        } else {
            layer.shortconv.conv     = create_tensor(tn(LLM_TENSOR_SHORTCONV_CONV,    "weight", i), {hparams.n_shortconv_l_cache, n_embd}, 0);
            layer.shortconv.in_proj  = create_tensor(tn(LLM_TENSOR_SHORTCONV_INPROJ,  "weight", i), {n_embd, 3 * n_embd}, 0);
            layer.shortconv.out_proj = create_tensor(tn(LLM_TENSOR_SHORTCONV_OUTPROJ, "weight", i), {n_embd, n_embd}, 0);
        }
        // Load scales before later visits alias the physical layer.
        if (loop_mhc || loop_refine) {
            layer.ffn_up_s = create_tensor(tn(LLM_TENSOR_FFN_UP, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            layer.ffn_gate_s = create_tensor(tn(LLM_TENSOR_FFN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            layer.ffn_down_s = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            if (hparams.is_recr(i)) {
                layer.shortconv.in_proj_s = create_tensor(tn(LLM_TENSOR_SHORTCONV_INPROJ, "scale", i), {1}, TENSOR_NOT_REQUIRED);
                layer.shortconv.out_proj_s = create_tensor(tn(LLM_TENSOR_SHORTCONV_OUTPROJ, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            } else {
                layer.wq_s = create_tensor(tn(LLM_TENSOR_ATTN_Q, "scale", i), {1}, TENSOR_NOT_REQUIRED);
                layer.wk_s = create_tensor(tn(LLM_TENSOR_ATTN_K, "scale", i), {1}, TENSOR_NOT_REQUIRED);
                layer.wv_s = create_tensor(tn(LLM_TENSOR_ATTN_V, "scale", i), {1}, TENSOR_NOT_REQUIRED);
                layer.wo_s = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "scale", i), {1}, TENSOR_NOT_REQUIRED);
                layer.wqkv_gate_s = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "scale", i), {1}, TENSOR_NOT_REQUIRED);
            }
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_lfm3::build_arch_graph(const llm_graph_params & params) const {
    if (!hparams.causal_attn) {
        return std::make_unique<graph<true>>(*this, params);
    }
    return std::make_unique<graph<false>>(*this, params);
}

template <bool decision>
llama_model_lfm3::graph<decision>::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_lfm_base(params) {
    auto build_attn_block = [&model, this](ggml_tensor * cur_in,
                                           ggml_tensor * inp_pos,
                                           auto *        inp_attn,
                                           int           il,
                                           const llama_model_lfm3::refine_visit * visit) -> ggml_tensor * {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_head_il   = hparams.n_head(il);
        const int64_t n_head_kv   = hparams.n_head_kv(il);

        const int     n_rot_l      = hparams.n_rot(il);
        const float   freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float   freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = hparams.is_swa(il) ? nullptr : model.layers[il].rope_freqs;

        auto [q, k, v] = build_qkv(model.layers[il], cur_in, n_embd_head, n_head_il, n_head_kv, il);
        if (visit) {
            auto add_delta = [this, cur_in](ggml_tensor * t, const llm_lfm_delta & d) {
                ggml_tensor * dx = ggml_mul_mat(ctx0, d.up, ggml_mul_mat(ctx0, d.down, cur_in));
                return ggml_add(ctx0, t, ggml_reshape_3d(ctx0, dx, t->ne[0], t->ne[1], t->ne[2]));
            };
            q = add_delta(q, visit->wq);
            k = add_delta(k, visit->wk);
            v = add_delta(v, visit->wv);
        }

        q = build_norm(q, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
        cb(q, "model.layers.{}.self_attn.q_layernorm", il);
        k = build_norm(k, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        cb(k, "model.layers.{}.self_attn.k_layernorm", il);

        q = ggml_rope_ext(ctx0, q, inp_pos, rope_factors, n_rot_l, rope_type, n_ctx_orig,
                          freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        k = ggml_rope_ext(ctx0, k, inp_pos, rope_factors, n_rot_l, rope_type, n_ctx_orig,
                          freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);

        ggml_tensor * attn_out = build_attn(inp_attn,
                NULL, NULL, NULL,
                q, k, v, nullptr, nullptr, nullptr, 1.0f / sqrtf(float(n_embd_head)), il);
        cb(attn_out, "attn_out", il);

        ggml_tensor * gate = build_lora_mm(model.layers[il].wqkv_gate, cur_in, model.layers[il].wqkv_gate_s);
        gate = ggml_sigmoid(ctx0, gate);
        cb(gate, "model.layers.{}.self_attn.gate", il);

        ggml_tensor * attn_3d = ggml_reshape_3d(ctx0, attn_out, n_embd_head, n_head_il, n_tokens);
        ggml_tensor * gate_3d = ggml_reshape_3d(ctx0, gate,     1,           n_head_il, n_tokens);
        attn_3d = ggml_mul(ctx0, attn_3d, gate_3d);
        attn_out = ggml_reshape_2d(ctx0, attn_3d, n_embd_head * n_head_il, n_tokens);

        ggml_tensor * out = build_lora_mm(model.layers[il].wo, attn_out, model.layers[il].wo_s);
        if (visit) {
            out = build_lfm_delta(&visit->wo, out, attn_out);
        }
        cb(out, "model.layers.{}.self_attn.out_proj", il);

        return out;
    };

    const bool loop_mhc = static_cast<const llama_model_lfm3 &>(model).loop_mhc;
    struct mhc_state { ggml_tensor * residual; ggml_tensor * post; ggml_tensor * mix; };
    auto round_bf16 = [this](ggml_tensor * x) {
        return ggml_cast(ctx0, ggml_cast(ctx0, x, GGML_TYPE_BF16), GGML_TYPE_F32);
    };
    auto stream = [this](ggml_tensor * x, int i) {
        return ggml_view_2d(ctx0, x, n_embd, x->ne[2], x->nb[2], i * x->nb[1]);
    };
    auto read_mhc = [this, &round_bf16, &stream](ggml_tensor * r, ggml_tensor * w, ggml_tensor * b, mhc_state & state) {
        const int64_t nt = r->ne[2];
        state.residual = r;
        auto * flat = ggml_reshape_2d(ctx0, r, 4 * n_embd, nt);
        auto * norm = ggml_rms_norm(ctx0, flat, 1e-6f);
        ggml_tensor * projection = nullptr;
        for (int row = 0; row < 24; ++row) {
            auto * weight = ggml_view_2d(ctx0, w, 4 * n_embd, 1, w->nb[1], row * w->nb[1]);
            auto * dot = ggml_sum_rows(ctx0, ggml_mul(ctx0, norm, weight));
            projection = projection ? ggml_concat(ctx0, projection, dot, 0) : dot;
        }
        auto * logits = ggml_add(ctx0, projection, b);
        auto * pre = ggml_sigmoid(ctx0, ggml_cont(ctx0, ggml_view_2d(ctx0, logits, 4, nt, logits->nb[1], 0)));
        pre = ggml_div(ctx0, pre, ggml_sum_rows(ctx0, pre));
        state.post = ggml_scale(ctx0, ggml_sigmoid(ctx0, ggml_cont(ctx0,
            ggml_view_2d(ctx0, logits, 4, nt, logits->nb[1], 4 * logits->nb[0]))), 2.0f);
        auto * mix = ggml_cont(ctx0, ggml_view_2d(ctx0, logits, 16, nt, logits->nb[1], 8 * logits->nb[0]));
        mix = ggml_reshape_3d(ctx0, mix, 4, 4, nt);
        for (int i = 0; i < 20; ++i) {
            mix = ggml_log(ctx0, ggml_soft_max(ctx0, mix));
            mix = ggml_cont(ctx0, ggml_permute(ctx0, mix, 1, 0, 2, 3));
            mix = ggml_log(ctx0, ggml_soft_max(ctx0, mix));
            mix = ggml_cont(ctx0, ggml_permute(ctx0, mix, 1, 0, 2, 3));
        }
        mix = ggml_exp(ctx0, mix);
        state.mix = ggml_div(ctx0, mix, ggml_sum_rows(ctx0, mix));
        auto * anchor = stream(r, 0);
        ggml_tensor * delta = nullptr;
        for (int i = 0; i < 4; ++i) {
            auto * coeff = ggml_view_2d(ctx0, pre, 1, nt, pre->nb[1], i * pre->nb[0]);
            auto * term = ggml_mul(ctx0, ggml_sub(ctx0, stream(r, i), anchor), coeff);
            delta = delta ? ggml_add(ctx0, delta, term) : term;
        }
        return round_bf16(ggml_add(ctx0, anchor, delta));
    };
    auto merge_mhc = [this, &round_bf16, &stream](const mhc_state & state, ggml_tensor * branch, ggml_tensor * gate) {
        auto * r = state.residual;
        const int64_t nt = r->ne[2];
        auto * anchor = stream(r, 0);
        ggml_tensor * result = nullptr;
        for (int i = 0; i < 4; ++i) {
            ggml_tensor * delta = nullptr;
            for (int j = 0; j < 4; ++j) {
                auto * coeff = ggml_view_2d(ctx0, state.mix, 1, nt, state.mix->nb[2], i * state.mix->nb[1] + j * state.mix->nb[0]);
                auto * term = ggml_mul(ctx0, ggml_sub(ctx0, stream(r, j), anchor), coeff);
                delta = delta ? ggml_add(ctx0, delta, term) : term;
            }
            auto * post = ggml_view_2d(ctx0, state.post, 1, nt, state.post->nb[1], i * state.post->nb[0]);
            auto * candidate = round_bf16(ggml_add(ctx0, round_bf16(ggml_add(ctx0, anchor, delta)),
                round_bf16(ggml_mul(ctx0, round_bf16(branch), post))));
            if (gate) {
                auto * old = stream(r, i);
                candidate = round_bf16(ggml_add(ctx0, old, ggml_mul(ctx0, ggml_sub(ctx0, candidate, old), ggml_tanh(ctx0, gate))));
            }
            auto * out = ggml_reshape_3d(ctx0, candidate, n_embd, 1, nt);
            result = result ? ggml_concat(ctx0, result, out, 1) : out;
        }
        return result;
    };

    // refine loop: the casts follow the BF16 reference, where the residual streams and the pre, post and mix weights are BF16
    const auto & lfm3 = static_cast<const llama_model_lfm3 &>(model);
    struct refine_state { ggml_tensor * residual; ggml_tensor * post0; ggml_tensor * post1; ggml_tensor * diag; ggml_tensor * off; };
    auto row = [this](ggml_tensor * x, int i) {
        return ggml_cont(ctx0, ggml_view_2d(ctx0, x, 1, x->ne[1], x->nb[1], i * x->nb[0]));
    };
    auto read_refine = [this, &round_bf16, &stream, &row](ggml_tensor * r, ggml_tensor * w, ggml_tensor * b, refine_state & state) {
        state.residual = r;
        auto * norm   = ggml_rms_norm(ctx0, ggml_reshape_2d(ctx0, r, 2 * n_embd, r->ne[2]), 1e-6f);
        auto * logits = ggml_add(ctx0, ggml_mul_mat(ctx0, w, norm), b);
        auto * act    = ggml_sigmoid(ctx0, logits);
        state.post0 = round_bf16(ggml_scale(ctx0, row(act, 2), 2.0f));
        state.post1 = round_bf16(ggml_scale(ctx0, row(act, 3), 2.0f));
        // closed-form Sinkhorn limit of a 2x2 mix: [[diag, off], [off, diag]]
        auto * diag = ggml_sigmoid(ctx0, ggml_scale(ctx0,
            ggml_sub(ctx0, ggml_sub(ctx0, ggml_add(ctx0, row(logits, 4), row(logits, 7)), row(logits, 5)), row(logits, 6)), 0.5f));
        state.diag = round_bf16(diag);
        state.off  = round_bf16(ggml_scale_bias(ctx0, diag, -1.0f, 1.0f));
        auto * branch = ggml_add(ctx0, ggml_mul(ctx0, stream(r, 0), round_bf16(row(act, 0))), ggml_mul(ctx0, stream(r, 1), round_bf16(row(act, 1))));
        return round_bf16(branch);
    };
    auto merge_refine = [this, &round_bf16, &stream](const refine_state & state, ggml_tensor * branch) {
        auto * r = state.residual;
        branch = round_bf16(branch);
        auto merged = [&](ggml_tensor * own, ggml_tensor * other, ggml_tensor * post) {
            auto * mixed = round_bf16(ggml_add(ctx0, ggml_mul(ctx0, own, state.diag), ggml_mul(ctx0, other, state.off)));
            auto * out   = round_bf16(ggml_add(ctx0, mixed, round_bf16(ggml_mul(ctx0, branch, post))));
            return ggml_reshape_3d(ctx0, out, n_embd, 1, r->ne[2]);
        };
        return ggml_concat(ctx0, merged(stream(r, 0), stream(r, 1), state.post0), merged(stream(r, 1), stream(r, 0), state.post1), 1);
    };
    auto refine_ffn = [this](ggml_tensor * x, const llama_layer & layer, const llama_model_lfm3::refine_visit & visit) {
        auto * gate = build_lfm_delta(&visit.ffn_gate, build_lora_mm(layer.ffn_gate, x, layer.ffn_gate_s), x);
        auto * up   = build_lfm_delta(&visit.ffn_up,   build_lora_mm(layer.ffn_up,   x, layer.ffn_up_s),   x);
        auto * h    = ggml_swiglu_split(ctx0, gate, up);
        return build_lfm_delta(&visit.ffn_down, build_lora_mm(layer.ffn_down, h, layer.ffn_down_s), h);
    };
    auto blend_refine = [this, &round_bf16, &lfm3](ggml_tensor * old, ggml_tensor * candidate) {
        auto * signal = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_rms_norm(ctx0, old, 1e-6f), lfm3.loop_gate_token));
        auto * gate   = ggml_sigmoid(ctx0, ggml_add(ctx0, ggml_repeat(ctx0, lfm3.loop_gate_logits, old), signal));
        return round_bf16(ggml_add(ctx0, old, ggml_mul(ctx0, gate, ggml_sub(ctx0, candidate, old))));
    };
    ggml_tensor * pass_input = nullptr;

    ggml_tensor * cur = build_inp_embd(model.tok_embd);
    cb(cur, "model.embed_tokens", -1);

    ggml_build_forward_expand(gf, cur);

    // decision: no memory, the whole prompt is read in one batch
    auto * inp_hybrid = decision ? nullptr : build_inp_mem_hybrid_iswa();
    auto * inp_attn   = decision ? build_attn_inp_no_cache() : nullptr;

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    if (loop_mhc) {
        cur = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), n_embd, 4, n_tokens, 1);
    }
    for (int il = 0; il < n_layer; ++il) {
        if (lfm3.loop_refine && il >= lfm3.refine_begin && il < lfm3.refine_end) {
            const auto & layer = model.layers[il];
            const auto & visit = lfm3.refine_visits[il];
            if (il == lfm3.refine_begin) {
                pass_input = round_bf16(cur);
                cur = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, pass_input, n_embd, 1, n_tokens), n_embd, 2, n_tokens, 1);
            }
            refine_state state{};
            ggml_tensor * branch = read_refine(cur, layer.hc_attn_fn, layer.hc_attn_base, state);
            branch = build_norm(branch, layer.attn_norm, NULL, LLM_NORM_RMS, il);
            if (hparams.is_recr(il)) {
                if constexpr (decision) {
                    branch = build_lfm_shortconv(model, branch, il, &visit.conv_in, &visit.conv_out);
                } else {
                    const auto * mctx_recr = static_cast<const llama_memory_hybrid_iswa_context *>(mctx)->get_recr();
                    branch = build_lfm_shortconv(model, mctx_recr, inp_hybrid->get_recr(), branch, il, &visit.conv_in, &visit.conv_out);
                }
            } else if constexpr (decision) {
                branch = build_attn_block(branch, inp_pos, inp_attn, il, &visit);
            } else {
                branch = build_attn_block(branch, inp_pos, inp_hybrid->get_attn(), il, &visit);
            }
            if (layer.attn_post_norm) {
                branch = build_norm(branch, layer.attn_post_norm, NULL, LLM_NORM_RMS, il);
            }
            cur = merge_refine(state, branch);
            branch = read_refine(cur, layer.hc_ffn_fn, layer.hc_ffn_base, state);
            branch = build_norm(branch, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
            branch = refine_ffn(branch, layer, visit);
            if (layer.ffn_post_norm) {
                branch = build_norm(branch, layer.ffn_post_norm, NULL, LLM_NORM_RMS, il);
            }
            cur = merge_refine(state, branch);
            if (il == lfm3.refine_end - 1) {
                // leave the pass: average the streams, then gate the result into the first-pass state
                cur = blend_refine(pass_input, round_bf16(ggml_scale(ctx0, ggml_add(ctx0, stream(cur, 0), stream(cur, 1)), 0.5f)));
                if (il == n_layer - 1 && inp_out_ids) {
                    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
                }
                cur = build_cvec(cur, il);
            }
            cb(cur, "l_out", il);
            continue;
        }

        mhc_state attn_state{}, ffn_state{};
        if (loop_mhc) {
            cur = read_mhc(cur, model.layers[il].hc_attn_fn, model.layers[il].hc_attn_base, attn_state);
        }
        auto * prev_cur = cur;

        cur = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "model.layers.{}.operator_norm", il);

        if (hparams.is_recr(il)) {
            if constexpr (decision) {
                cur = build_lfm_shortconv(model, cur, il);
            } else {
                const auto * mctx_recr = static_cast<const llama_memory_hybrid_iswa_context *>(mctx)->get_recr();
                cur = build_lfm_shortconv(model, mctx_recr, inp_hybrid->get_recr(), cur, il);
            }
        } else if constexpr (decision) {
            cur = build_attn_block(cur, inp_pos, inp_attn, il, nullptr);
        } else {
            cur = build_attn_block(cur, inp_pos, inp_hybrid->get_attn(), il, nullptr);
        }

        if (model.layers[il].attn_post_norm) {
            cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "model.layers.{}.post_operator_norm", il);
        }

        if (!loop_mhc && il == n_layer - 1 && inp_out_ids) {
            cur      = ggml_get_rows(ctx0, cur, inp_out_ids);
            prev_cur = ggml_get_rows(ctx0, prev_cur, inp_out_ids);
        }

        if (loop_mhc) {
            cur = merge_mhc(attn_state, cur, model.layers[il].hc_attn_scale);
            cur = read_mhc(cur, model.layers[il].hc_ffn_fn, model.layers[il].hc_ffn_base, ffn_state);
        } else {
            cur = ggml_add(ctx0, prev_cur, cur);
        }

        auto * ffn_norm_out = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(ffn_norm_out, "model.layers.{}.ffn_norm", il);

        ggml_tensor * ffn_out = build_lfm_dense_ffn(model, ffn_norm_out, il);

        if (model.layers[il].ffn_post_norm) {
            ffn_out = build_norm(ffn_out, model.layers[il].ffn_post_norm, NULL, LLM_NORM_RMS, il);
            cb(ffn_out, "model.layers.{}.post_ffn_norm", il);
        }

        cur = loop_mhc ? merge_mhc(ffn_state, ffn_out, model.layers[il].hc_ffn_scale) : ggml_add(ctx0, cur, ffn_out);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
    }

    if (loop_mhc) {
        auto * anchor = stream(cur, 0);
        auto * delta = ggml_sub(ctx0, stream(cur, 1), anchor);
        delta = ggml_add(ctx0, delta, ggml_sub(ctx0, stream(cur, 2), anchor));
        delta = ggml_add(ctx0, delta, ggml_sub(ctx0, stream(cur, 3), anchor));
        cur = round_bf16(ggml_add(ctx0, anchor, ggml_scale(ctx0, delta, 0.25f)));
        if (inp_out_ids) { cur = ggml_get_rows(ctx0, cur, inp_out_ids); }
    }
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
}

template struct llama_model_lfm3::graph<false>;
template struct llama_model_lfm3::graph<true>;
