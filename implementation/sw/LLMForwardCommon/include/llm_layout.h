#ifndef _LLM_LAYOUT_H_
#define _LLM_LAYOUT_H_

#include <stdint.h>

// =======================================================
// Model hyperparameters
// =======================================================
#define LLM_B          1
// Working sequence length for the current workload binaries.
#define LLM_T          2048
// Upper bound reserved for upcoming decode-phase cache growth.
#define LLM_MAX_CTX    2048
#define LLM_D_MODEL    768
#define LLM_D_FF       3072
#define LLM_N_HEAD     6
#define LLM_N_KV_HEAD  6
#define LLM_HEAD_DIM   (LLM_D_MODEL / LLM_N_HEAD)
#define LLM_NUM_LAYERS 12
#define LLM_ELEM_SIZE  2        // fp16 bytes

#if (LLM_D_MODEL % LLM_N_HEAD) != 0
#error "LLM_D_MODEL must be divisible by LLM_N_HEAD"
#endif

#if (LLM_MAX_CTX < LLM_T)
#error "LLM_MAX_CTX must be >= LLM_T"
#endif

// Sizes in bytes
#define BYTES_H        ((uint64_t)LLM_T * (uint64_t)LLM_D_MODEL * (uint64_t)LLM_ELEM_SIZE)
#define BYTES_H_NORM   BYTES_H
#define BYTES_MLP_MID  ((uint64_t)LLM_T * (uint64_t)LLM_D_FF    * (uint64_t)LLM_ELEM_SIZE)
#define BYTES_MLP_OUT  BYTES_H

#define BYTES_HM_HEAD  ((uint64_t)LLM_T * (uint64_t)LLM_HEAD_DIM * (uint64_t)LLM_ELEM_SIZE)
#define BYTES_HM_ALL   ((uint64_t)LLM_N_HEAD * (uint64_t)BYTES_HM_HEAD)

// Persistent KV-cache footprint (head-major: [kv_head][ctx][head_dim]).
#define BYTES_KV_HEAD_CTX       ((uint64_t)LLM_MAX_CTX * (uint64_t)LLM_HEAD_DIM * (uint64_t)LLM_ELEM_SIZE)
#define BYTES_KV_LAYER          ((uint64_t)LLM_N_KV_HEAD * (uint64_t)BYTES_KV_HEAD_CTX)
#define BYTES_KV_ALL_LAYERS     ((uint64_t)LLM_NUM_LAYERS * (uint64_t)BYTES_KV_LAYER)
// Bytes written when appending one token for one layer across all KV heads.
#define BYTES_KV_TOKEN_ALL_HEAD ((uint64_t)LLM_N_KV_HEAD * (uint64_t)LLM_HEAD_DIM * (uint64_t)LLM_ELEM_SIZE)
#define BYTES_KV_TOKEN_PER_HEAD ((uint64_t)LLM_HEAD_DIM * (uint64_t)LLM_ELEM_SIZE)

// =======================================================
// Activation layout in HBM
// =======================================================
// Buffers are laid out contiguously so prefill/decode apps can share the same map.
#define LLM_H_ADDR        ((uint64_t)0xC0000000)
#define LLM_H_NORM_ADDR   (LLM_H_ADDR       + BYTES_H)
#define LLM_MLP_MID_ADDR  (LLM_H_NORM_ADDR  + BYTES_H_NORM)
#define LLM_MLP_OUT_ADDR  (LLM_MLP_MID_ADDR + BYTES_MLP_MID)
#define LLM_ATTN_O_ADDR   (LLM_MLP_OUT_ADDR + BYTES_MLP_OUT)

// Attention scratch/work buffers
#define LLM_QKV_TM_ADDR      (LLM_ATTN_O_ADDR + BYTES_H)
#define LLM_Q_HM_ADDR        (LLM_QKV_TM_ADDR + BYTES_H)
#define LLM_K_HM_ADDR        (LLM_Q_HM_ADDR   + BYTES_HM_ALL)
#define LLM_V_HM_ADDR        (LLM_K_HM_ADDR   + BYTES_HM_ALL)
#define LLM_O_HM_ADDR        (LLM_V_HM_ADDR   + BYTES_HM_ALL)

// Reuse MLP_OUT as attention projection output
#define LLM_ATTN_PROJ_TM_ADDR  (LLM_MLP_OUT_ADDR)

// End of temporary activation/scratch region.
#define LLM_ACT_SCRATCH_END_ADDR ((uint64_t)LLM_O_HM_ADDR + (uint64_t)BYTES_HM_ALL)

// Persistent cache regions (shared by prefill/decode apps).
#define LLM_K_CACHE_BASE_ADDR   (LLM_ACT_SCRATCH_END_ADDR)
#define LLM_V_CACHE_BASE_ADDR   (LLM_K_CACHE_BASE_ADDR + BYTES_KV_ALL_LAYERS)
#define LLM_KV_CACHE_END_ADDR   (LLM_V_CACHE_BASE_ADDR + BYTES_KV_ALL_LAYERS)

// Weight base (HBM)
#define LLM_W_BASE        ((uint64_t)0x9C0000000)

// Compile-time guard: KV cache must end before the weight region starts.
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 201112L)
_Static_assert(LLM_KV_CACHE_END_ADDR <= LLM_W_BASE,
               "KV cache region overlaps weight region; move base addresses or reduce cache footprint.");
#else
typedef char llm_kv_cache_overlap_check[(LLM_KV_CACHE_END_ADDR <= LLM_W_BASE) ? 1 : -1];
#endif

// Layer-major base pointers for persistent K/V cache.
static inline uint64_t llm_k_cache_layer_base(uint32_t layer_id)
{
    return (uint64_t)LLM_K_CACHE_BASE_ADDR + (uint64_t)layer_id * (uint64_t)BYTES_KV_LAYER;
}

static inline uint64_t llm_v_cache_layer_base(uint32_t layer_id)
{
    return (uint64_t)LLM_V_CACHE_BASE_ADDR + (uint64_t)layer_id * (uint64_t)BYTES_KV_LAYER;
}

// Address helpers for one [head_dim] token slice in K/V cache.
static inline uint64_t llm_k_cache_head_token_addr(uint32_t layer_id, uint32_t kv_head, uint32_t token_idx)
{
    return llm_k_cache_layer_base(layer_id)
           + (uint64_t)kv_head * (uint64_t)BYTES_KV_HEAD_CTX
           + (uint64_t)token_idx * (uint64_t)BYTES_KV_TOKEN_PER_HEAD;
}

static inline uint64_t llm_v_cache_head_token_addr(uint32_t layer_id, uint32_t kv_head, uint32_t token_idx)
{
    return llm_v_cache_layer_base(layer_id)
           + (uint64_t)kv_head * (uint64_t)BYTES_KV_HEAD_CTX
           + (uint64_t)token_idx * (uint64_t)BYTES_KV_TOKEN_PER_HEAD;
}

// Lightweight index checks shared by prefill/decode drivers.
static inline uint32_t llm_cache_valid_layer(uint32_t layer_id)
{
    return (layer_id < (uint32_t)LLM_NUM_LAYERS) ? 1u : 0u;
}

static inline uint32_t llm_cache_valid_head(uint32_t kv_head)
{
    return (kv_head < (uint32_t)LLM_N_KV_HEAD) ? 1u : 0u;
}

static inline uint32_t llm_cache_valid_token(uint32_t token_idx, uint32_t max_ctx_len)
{
    return (token_idx < max_ctx_len) ? 1u : 0u;
}

// True when appending append_len tokens will not exceed max_ctx_len.
static inline uint32_t llm_cache_can_append(uint32_t cache_len, uint32_t append_len, uint32_t max_ctx_len)
{
    if (cache_len > max_ctx_len)
        return 0u;
    return (append_len <= (max_ctx_len - cache_len)) ? 1u : 0u;
}

#endif // _LLM_LAYOUT_H_
