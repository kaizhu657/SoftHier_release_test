#ifndef _LLM_COMMON_H_
#define _LLM_COMMON_H_

#include <stdint.h>

#include "llm_layout.h"
#include "llm_weights.h"

typedef struct LLMRuntimeState
{
    uint32_t prompt_len;    // Prefill token count.
    uint32_t decode_steps;  // Number of decode iterations to run.
    uint32_t cache_len;     // Current KV cache length.
    uint32_t max_ctx_len;   // Cap for cache growth.
} LLMRuntimeState;

// Initialize runtime state with optional max-context override.
void llm_common_init_runtime(LLMRuntimeState *state,
                             uint32_t prompt_len,
                             uint32_t decode_steps,
                             uint32_t max_ctx_len);
// Strict runtime-state consistency check against context/cache limits.
uint32_t llm_common_runtime_is_valid(const LLMRuntimeState *state);
// Query whether cache_len can grow by append_tokens.
uint32_t llm_common_cache_can_append(const LLMRuntimeState *state, uint32_t append_tokens);
// Apply cache growth when valid; returns 1 on success, 0 on overflow/invalid state.
uint32_t llm_common_cache_append(LLMRuntimeState *state, uint32_t append_tokens);

// Data/weight initialization helpers for micro-bench style runs.
void llm_common_init_hidden_state(void);
void llm_common_init_dummy_weights(uint32_t layer_id);

// Layer primitives used by prefill/decode applications.
void llm_common_run_prefill_layer(uint32_t layer_id, uint32_t q_len, uint32_t kv_len);
void llm_common_run_decode_layer(uint32_t layer_id, uint32_t q_len, uint32_t kv_len);
// Persist the current layer K/V tensors into shared KV cache layout.
void llm_common_store_prefill_kv_cache(uint32_t layer_id, uint32_t kv_len);
// Append decode-step K/V slices into cache at [cache_pos, cache_pos + append_len).
void llm_common_store_decode_kv_cache(uint32_t layer_id, uint32_t cache_pos, uint32_t append_len);

// Debug utility: dump fp16 values from HBM.
void llm_common_dma_dump_u16(uint64_t hbm_addr, uint32_t n_halfwords);

// Runtime wrappers to avoid duplicate runtime symbol definitions.
void llm_common_barrier_init(void);
void llm_common_barrier(void);
void llm_common_timer_start(void);
void llm_common_timer_end(void);
uint32_t llm_common_is_lead_core(void);
void llm_common_eoc(uint32_t eoc_val);
void llm_common_log_start(void);
void llm_common_log_layer_done(uint32_t layer_id);

#endif
