#pragma once
#include "llama.h"

#ifdef __cplusplus
#define ENUM_TYPE enum class
#else
#define ENUM_TYPE enum
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define BACKEND_NOT_INITIALISED (-1)
#define BACKEND_INVALID_PARAMS (-2)
#define BACKEND_MISSING_CALLBACKS (-10)
#define BACKEND_MODEL_LOADING_ERROR (-20)
#define BACKEND_CONTEXT_LOADING_ERROR (-30)
#define BACKEND_SAMPLING_CONTEXT_LOADING_ERROR (-31)
#define BACKEND_PROMPT_TOO_LONG (-40)
#define BACKEND_EVAL_FAILED (-50)

typedef struct s_build_info {
    int build_number;
    char const* commit;
    char const* compiler;
    char const* build_target;
} BuildInfo;

BuildInfo init_build_info();

typedef struct s_model_params {
    char const* name;
    int32_t n_ctx;
    int32_t n_ctx_train;
    int32_t n_vocab;
    bool has_metal;
    bool has_cublas;
    BuildInfo build_info;
} ModelParams;

typedef struct s_prediction_params {
    uint32_t seed;

    int32_t n_threads;
    int32_t n_threads_batch; // number of threads to use for batch processing (-1 = use n_threads)
    int32_t n_predict; // new tokens to predict
    int32_t n_ctx; // context size
    int32_t n_batch; // batch size for prompt processing (must be >=32 to use BLAS)
    int32_t n_keep; // number of tokens to keep from initial prompt
    int32_t n_draft; // number of tokens to draft during speculative decoding
    int32_t n_chunks; // max number of chunks to process (-1 = unlimited)
    int32_t n_parallel; // number of parallel sequences to decode
    int32_t n_sequences; // number of sequences to decode
    float p_accept; // speculative decoding accept probability
    float p_split; // speculative decoding split probability
    int32_t n_gpu_layers; // number of layers to store in VRAM (-1 - use default)
    int32_t n_gpu_layers_draft; // number of layers to store in VRAM for the draft model (-1 - use default)
    int32_t main_gpu; // the GPU that is used for scratch and small tensors
    float tensor_split[LLAMA_MAX_DEVICES]; // how split tensors should be distributed across GPUs
    int32_t n_beams; // if non-zero then use beam search of given width.
    float rope_freq_base; // RoPE base frequency
    float rope_freq_scale; // RoPE frequency scaling factor
    float yarn_ext_factor; // YaRN extrapolation mix factor
    float yarn_attn_factor; // YaRN magnitude scaling factor
    float yarn_beta_fast; // YaRN low correction dim
    float yarn_beta_slow; // YaRN high correction dim
    int32_t yarn_orig_ctx; // YaRN original context length
    int32_t rope_scaling_type;

    bool mul_mat_q; // if true, use mul_mat_q kernels instead of cuBLAS
    bool memory_f16; // use f16 instead of f32 for memory kv
    bool random_prompt; // do not randomize prompt if none provided
    bool use_color; // use color to distinguish generations and inputs
    bool interactive; // interactive mode
    bool chatml; // chatml mode (used for models trained on chatml syntax)
    bool prompt_cache_all; // save user input and generations to prompt cache
    bool prompt_cache_ro; // open the prompt cache read-only and do not update it

    bool embedding; // get only sentence embedding
    bool escape; // escape "\n", "\r", "\t", "\'", "\"", and "\\"
    bool interactive_first; // wait for user input immediately
    bool multiline_input; // reverse the usage of `\`
    bool simple_io; // improves compatibility with subprocesses and limited consoles
    bool cont_batching; // insert new sequences for decoding on-the-fly

    bool input_prefix_bos; // prefix BOS to user inputs, preceding input_prefix
    bool ignore_eos; // ignore generated EOS tokens
    bool instruct; // instruction mode (used for Alpaca models)
    bool logits_all; // return logits for all tokens in the batch
    bool use_mmap; // use mmap for faster loads
    bool use_mlock; // use mlock to keep model in memory
    bool numa; // attempt optimizations that help on some NUMA systems
    bool verbose_prompt; // print prompt tokens before generation
    bool infill; // use infill mode
    bool dump_kv_cache; // dump the KV cache contents for debugging purposes

    char const* model_filename;
    char const* prompt;
    char const* grammar;
    bool use_numa;
} PredictionParams;

typedef struct s_backend {
    bool initialized;
    void* baton;
} Backend;

Backend* init_backend();

void free_backend(Backend* backend);


typedef struct s_session {
    int32_t id;
    char const* output;
} Session;

Session* init_session();

int backend_start_prediction(Backend* backend, Session* session, PredictionParams params);

void free_session(Session* session);

typedef ENUM_TYPE s_backend_log_level: uint8_t {
    TRACE = 1,
    DEBUG = 2,
    WARN = 3,
    ERROR = 4,
    INFO = 5,
} BackendLogLevel;


typedef void (*OutputCallback)(const char* token, bool is_eos);

typedef void (*ModelReadyCallback)(const char* model, ModelParams params);

typedef void (*LogCallback)(BackendLogLevel level, const char* message);

void set_model_ready_callback(Backend* backend, ModelReadyCallback callback);

void set_output_callback(Backend* backend, OutputCallback callback);

void set_log_callback(Backend* backend, LogCallback callback);

#ifdef __cplusplus
}
#endif
