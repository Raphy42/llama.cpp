#include "ffi.h"

#include "common/common.h"

typedef struct s_callbacks {
    OutputCallback output_callback = nullptr;
    ModelReadyCallback model_ready_callback = nullptr;
    LogCallback log_callback = nullptr;
} Callbacks;

static Callbacks* get_callbacks(const Backend* backend) {
    if (backend->baton == nullptr) {
        return nullptr;
    }
    return static_cast<Callbacks *>(backend->baton);
}

BuildInfo init_build_info() {
    return BuildInfo{
        .build_number = LLAMA_BUILD_NUMBER,
        .compiler = LLAMA_COMPILER,
        .commit = LLAMA_COMMIT,
        .build_target = LLAMA_BUILD_TARGET,
    };
}

Backend* init_backend() {
    auto* const backend = static_cast<Backend *>(calloc(1, sizeof(Backend)));
    backend->initialized = false;
    backend->baton = calloc(1, sizeof(Callbacks));
    return backend;
}

void free_backend(Backend* backend) {
    if (backend->baton != nullptr) {
        free(backend->baton);
    }

    if (backend->initialized) {
        llama_backend_free();
    }
}

static llama_model_params model_params_from_prediction_params(const PredictionParams&params) {
    llama_model_params model_params = llama_model_default_params();
    if (params.n_gpu_layers != -1) {
        model_params.n_gpu_layers = params.n_gpu_layers;
    }
    model_params.main_gpu = params.main_gpu;
    model_params.tensor_split = params.tensor_split;
    model_params.use_mmap = params.use_mmap;
    model_params.use_mlock = params.use_mlock;
    return model_params;
}

static llama_context_params context_params_from_prediction_params(const PredictionParams&params) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = params.n_ctx;
    ctx_params.n_batch = params.n_batch;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;
    ctx_params.mul_mat_q = params.mul_mat_q;
    ctx_params.seed = params.seed;
    ctx_params.f16_kv = params.memory_f16;
    ctx_params.logits_all = params.logits_all;
    ctx_params.embedding = params.embedding;
    ctx_params.rope_scaling_type = static_cast<int8_t>(params.rope_scaling_type);
    ctx_params.rope_freq_base = params.rope_freq_base;
    ctx_params.rope_freq_scale = params.rope_freq_scale;
    ctx_params.yarn_ext_factor = params.yarn_ext_factor;
    ctx_params.yarn_attn_factor = params.yarn_attn_factor;
    ctx_params.yarn_beta_fast = params.yarn_beta_fast;
    ctx_params.yarn_beta_slow = params.yarn_beta_slow;
    ctx_params.yarn_orig_ctx = params.yarn_orig_ctx;
    return ctx_params;
}

static void warmup_model(PredictionParams params, llama_context* ctx, llama_model* model) {
    std::vector<llama_token> tokens = {llama_token_bos(model), llama_token_eos(model)};
    llama_decode(ctx, llama_batch_get_one(tokens.data(), std::min(tokens.size(), static_cast<size_t>(params.n_batch)),
                                          0, 0));
    llama_kv_cache_clear(ctx);
    llama_reset_timings(ctx);
}

int backend_start_prediction(Backend* backend, Session* session, PredictionParams params) {
    Callbacks* callbacks;
    if (backend->baton != nullptr) {
        callbacks = get_callbacks(backend);
        if (callbacks->model_ready_callback == nullptr || callbacks->output_callback == nullptr || callbacks->
            log_callback == nullptr) {
            return BACKEND_MISSING_CALLBACKS;
        }
        if (!backend->initialized) {
            llama_backend_init(params.use_numa);
            backend->initialized = true;
        }
    }
    else {
        return BACKEND_NOT_INITIALISED;
    }
    callbacks->log_callback(BackendLogLevel::DEBUG, "initializing backend");
    const llama_model_params m_params = model_params_from_prediction_params(params);
    llama_model* model = llama_load_model_from_file(params.model_filename, m_params);
    if (model == nullptr) {
        return BACKEND_MODEL_LOADING_ERROR;
    }

    const llama_context_params c_params = context_params_from_prediction_params(params);
    llama_context* ctx = llama_new_context_with_model(model, c_params);
    if (ctx == nullptr) {
        llama_free_model(model);
        return BACKEND_CONTEXT_LOADING_ERROR;
    }

    warmup_model(params, ctx, model);

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);
    callbacks->model_ready_callback(params.model_filename, ModelParams{
                                        .name = "test",
                                        .n_ctx = n_ctx,
                                        .n_ctx_train = n_ctx_train,
                                        .n_vocab = llama_n_vocab(model),
                                        .has_metal = !!ggml_cpu_has_metal(),
                                        .has_cublas = !!ggml_cpu_has_cublas(),
                                        .build_info = init_build_info(),
                                    });

    const bool add_bos = llama_should_add_bos_token(model);

    std::vector<llama_token> embd_inp;
    if (strlen(params.prompt) != 0) {
        embd_inp = llama_tokenize(ctx, params.prompt, add_bos, true);
    }
    callbacks->log_callback(BackendLogLevel::DEBUG, "prompt tokenized");

    if (embd_inp.empty()) {
        embd_inp.push_back(llama_token_bos(model));
    }

    if (static_cast<int>(embd_inp.size()) > n_ctx - 4) {
        return BACKEND_PROMPT_TOO_LONG;
    }

    if (params.n_ctx != 0 && params.n_ctx < 8) {
        params.n_ctx = 8;
    }

    if (params.n_keep < 0 || params.n_keep > static_cast<int>(embd_inp.size())) {
        params.n_keep = static_cast<int>(embd_inp.size());
    }

    int n_past = 0;
    int n_remain = params.n_predict;
    int n_consumed = 0;

    bool input_echo = true;

    std::vector<llama_token> embd;

    llama_sampling_params s_params = {};
    callbacks->log_callback(BackendLogLevel::DEBUG, llama_sampling_print(s_params).c_str());
    if (params.grammar != nullptr) {
        puts(s_params.grammar.c_str());
        s_params.grammar = std::string(params.grammar);
        callbacks->log_callback(BackendLogLevel::INFO, "using grammar constraint");
    }
    llama_sampling_context* ctx_sampling = llama_sampling_init(s_params);
    if (ctx_sampling == nullptr) {
        return BACKEND_SAMPLING_CONTEXT_LOADING_ERROR;
    }

    std::string output = {};
    output.reserve(std::min(std::max(0, params.n_predict), 1000));

    callbacks->log_callback(BackendLogLevel::INFO, "prediction started");
    while (n_remain != 0) {
        // predict
        if (!embd.empty()) {
            const int max_embd_size = n_ctx - 4;
            if (static_cast<int>(embd.size()) > max_embd_size) {
                const int skipped_tokens = static_cast<int>(embd.size()) - max_embd_size;
                embd.resize(max_embd_size);
                (void)skipped_tokens;
            }

            // infinite text generation via context swapping
            // if we run out of context:
            // - take the n_keep first tokens from the original prompt (via n_past)
            // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in batches
            if (n_past + static_cast<int>(embd.size()) > n_ctx) {
                if (params.n_predict == -2) {
                    break;
                }

                const int n_left = n_past - params.n_keep - 1;
                const int n_discard = n_left / 2;

                llama_kv_cache_seq_rm(ctx, 0, params.n_keep + 1, params.n_keep + n_discard + 1);
                llama_kv_cache_seq_shift(ctx, 0, params.n_keep + 1 + n_discard, n_past, -n_discard);

                n_past -= n_discard;
            }

            for (int i = 0; i < static_cast<int>(embd.size()); i += params.n_batch) {
                int n_eval = static_cast<int>(embd.size()) - i;
                if (n_eval > params.n_batch) {
                    n_eval = params.n_batch;
                }

                if (llama_decode(ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                    return BACKEND_EVAL_FAILED;
                }

                n_past += n_eval;
            }
        }

        embd.clear();

        if (static_cast<int>(embd_inp.size()) <= n_consumed) {
            const llama_token id = llama_sampling_sample(ctx_sampling, ctx, nullptr);
            llama_sampling_accept(ctx_sampling, ctx, id, true);
            embd.push_back(id);
            input_echo = true;
            --n_remain;
        }
        else {
            while (static_cast<int>(embd_inp.size()) > n_consumed) {
                embd.push_back(embd_inp[n_consumed]);
                llama_sampling_accept(ctx_sampling, ctx, embd_inp[n_consumed], false);
                ++n_consumed;
                if (static_cast<int>(embd.size()) >= params.n_batch) {
                    break;
                }
            }
        }

        if (input_echo) {
            for (const auto id: embd) {
                const std::string token_str = llama_token_to_piece(ctx, id);
                callbacks->output_callback(token_str.c_str(), id == llama_token_eos(model));
                output.append(token_str.c_str());
            }
        }

        if (!embd.empty() && embd.back() == llama_token_eos(model)) {
            break;
        }
    }
    size_t size = output.size() + 1; // +1 for null-terminator
    auto* c_str = static_cast<char *>(std::calloc(size, sizeof(char)));
    if (c_str) {
        std::memcpy(c_str, output.c_str(), size);
    }
    session->output = c_str;
    return 0;
}

void set_model_ready_callback(Backend* backend, const ModelReadyCallback callback) {
    auto* const callbacks = get_callbacks(backend);
    if (callbacks == nullptr) {
        return;
    }
    callbacks->model_ready_callback = callback;
}

void set_output_callback(Backend* backend, const OutputCallback callback) {
    auto* const callbacks = get_callbacks(backend);
    if (callbacks == nullptr) {
        return;
    }
    callbacks->output_callback = callback;
}

void set_log_callback(Backend* backend, const LogCallback callback) {
    auto* const callbacks = get_callbacks(backend);
    if (callbacks == nullptr) {
        return;
    }
    callbacks->log_callback = callback;
}

Session* init_session() {
    auto* const session = static_cast<Session *>(calloc(1, sizeof(Session)));
    session->id = static_cast<int32_t>(time(nullptr));
    return session;
}

void free_session(Session* session) {
    if (session->output != nullptr) {
        free((void *)(session->output));
    }
}
