#pragma once
#include <cstdint>

#include "common/common.h"

#define BACKEND_MISSING_CALLBACKS -1
#define BACKEND_MODEL_LOADING_ERROR -2
#define BACKEND_CONTEXT_LOADING_ERROR -3
#define BACKEND_PROMPT_TOO_LONG -4
#define BACKEND_EVAL_FAILED -5

typedef struct s_build_info {
    int build_number;
    char const* commit;
    char const* compiler;
    char const* build_target;
} BuildInfo;

BuildInfo init_build_info();

typedef struct s_model_params {
    char const* name;
} ModelParams;

typedef struct s_prediction_params {
    uint32_t seed;

    int32_t n_threads                       = get_num_physical_cores();
    int32_t n_threads_batch                 = -1;    // number of threads to use for batch processing (-1 = use n_threads)
    int32_t n_predict                       = -1;    // new tokens to predict
    int32_t n_ctx                           = 512;   // context size
    int32_t n_batch                         = 512;   // batch size for prompt processing (must be >=32 to use BLAS)
    mutable int32_t n_keep                          = 0;     // number of tokens to keep from initial prompt
    int32_t n_draft                         = 16;    // number of tokens to draft during speculative decoding
    int32_t n_chunks                        = -1;    // max number of chunks to process (-1 = unlimited)
    int32_t n_parallel                      = 1;     // number of parallel sequences to decode
    int32_t n_sequences                     = 1;     // number of sequences to decode
    float   p_accept                        = 0.5f;  // speculative decoding accept probability
    float   p_split                         = 0.1f;  // speculative decoding split probability
    int32_t n_gpu_layers                    = -1;    // number of layers to store in VRAM (-1 - use default)
    int32_t n_gpu_layers_draft              = -1;    // number of layers to store in VRAM for the draft model (-1 - use default)
    int32_t main_gpu                        = 0;     // the GPU that is used for scratch and small tensors
    float   tensor_split[LLAMA_MAX_DEVICES] = {0};   // how split tensors should be distributed across GPUs
    int32_t n_beams                         = 0;     // if non-zero then use beam search of given width.
    float   rope_freq_base                  = 0.0f;  // RoPE base frequency
    float   rope_freq_scale                 = 0.0f;  // RoPE frequency scaling factor
    float   yarn_ext_factor                 = -1.0f; // YaRN extrapolation mix factor
    float   yarn_attn_factor                = 1.0f;  // YaRN magnitude scaling factor
    float   yarn_beta_fast                  = 32.0f; // YaRN low correction dim
    float   yarn_beta_slow                  = 1.0f;  // YaRN high correction dim
    int32_t yarn_orig_ctx                   = 0;     // YaRN original context length
    int8_t  rope_scaling_type               = LLAMA_ROPE_SCALING_UNSPECIFIED; // TODO: better to be int32_t for alignment
                                                                              //       pinging @cebtenzzre

    // sampling parameters
    llama_sampling_params sparams;

    bool mul_mat_q         = true;  // if true, use mul_mat_q kernels instead of cuBLAS
    bool memory_f16        = true;  // use f16 instead of f32 for memory kv
    bool random_prompt     = false; // do not randomize prompt if none provided
    bool use_color         = false; // use color to distinguish generations and inputs
    bool interactive       = false; // interactive mode
    bool chatml            = false; // chatml mode (used for models trained on chatml syntax)
    bool prompt_cache_all  = false; // save user input and generations to prompt cache
    bool prompt_cache_ro   = false; // open the prompt cache read-only and do not update it

    bool embedding         = false; // get only sentence embedding
    bool escape            = false; // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool interactive_first = false; // wait for user input immediately
    bool multiline_input   = false; // reverse the usage of `\`
    bool simple_io         = false; // improves compatibility with subprocesses and limited consoles
    bool cont_batching     = false; // insert new sequences for decoding on-the-fly

    bool input_prefix_bos  = false; // prefix BOS to user inputs, preceding input_prefix
    bool ignore_eos        = false; // ignore generated EOS tokens
    bool instruct          = false; // instruction mode (used for Alpaca models)
    bool logits_all        = false; // return logits for all tokens in the batch
    bool use_mmap          = true;  // use mmap for faster loads
    bool use_mlock         = false; // use mlock to keep model in memory
    bool numa              = false; // attempt optimizations that help on some NUMA systems
    bool verbose_prompt    = false; // print prompt tokens before generation
    bool infill            = false; // use infill mode
    bool dump_kv_cache     = false; // dump the KV cache contents for debugging purposes

    char const* model_filename;
    char const* prompt;
    bool use_numa;
} PredictionParams;

typedef struct s_backend {
    bool initialized = false;
    void* baton = nullptr;
} Backend;

Backend init_backend();

void free_backend(Backend backend);

int backend_start_prediction(Backend* backend, const PredictionParams& params);

typedef void (*OutputCallback)(const char* token, bool is_eos);

typedef void (*ModelReadyCallback)(const char* model, ModelParams params);

extern "C" void set_model_ready_callback(const Backend* backend, ModelReadyCallback callback);

extern "C" void set_output_callback(const Backend* backend, OutputCallback callback);
