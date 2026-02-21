#include "llm_common.h"
#include "llm_decode_cfg.h"

int main()
{
    uint32_t eoc_val = 0;

    LLMRuntimeState state;
    llm_common_init_runtime(&state,
                            0u,
                            (uint32_t)LLM_DECODE_STEPS,
                            (uint32_t)LLM_MAX_CTX);

    llm_common_barrier_init();
    llm_common_barrier();

    if (!llm_common_runtime_is_valid(&state))
    {
        eoc_val = 1;
        goto finish;
    }

    // Decode may start from an externally prepared cache length.
    state.cache_len = (uint32_t)LLM_DECODE_INIT_CACHE_LEN;
    if (state.cache_len > state.max_ctx_len)
        state.cache_len = state.max_ctx_len;
    if (state.decode_steps > (state.max_ctx_len - state.cache_len))
        state.decode_steps = state.max_ctx_len - state.cache_len;

    if (!llm_common_runtime_is_valid(&state))
    {
        eoc_val = 2;
        goto finish;
    }

    // Initialize all layers so decode behavior is deterministic.
    for (uint32_t layer = 0; layer < (uint32_t)LLM_NUM_LAYERS; ++layer)
        llm_common_init_dummy_weights(layer);

    llm_common_init_hidden_state();
    llm_common_barrier();
    llm_common_log_start();

    for (uint32_t step = 0; step < state.decode_steps; ++step)
    {
        if (!llm_common_cache_can_append(&state, 1u))
        {
            eoc_val = 3;
            break;
        }

        const uint32_t kv_len = state.cache_len + 1u;

        for (uint32_t layer = 0; layer < (uint32_t)LLM_NUM_LAYERS; ++layer)
        {
            llm_common_barrier();
            llm_common_run_decode_layer(layer, 1u, kv_len);
            llm_common_store_decode_kv_cache(layer, state.cache_len, 1u);
            llm_common_log_layer_done(layer);
        }

        if (!llm_common_cache_append(&state, 1u))
        {
            eoc_val = 4;
            break;
        }
    }

#if LLM_DECODE_DEBUG_DUMP
    llm_common_dma_dump_u16((uint64_t)LLM_H_ADDR, 8);
    if (state.cache_len > 0u)
    {
        const uint32_t last_token = state.cache_len - 1u;
        llm_common_dma_dump_u16(llm_k_cache_head_token_addr(0, 0, last_token), 8);
        llm_common_dma_dump_u16(llm_v_cache_head_token_addr(0, 0, last_token), 8);
    }
#endif

finish:
    llm_common_barrier();
    llm_common_eoc(eoc_val);
    return 0;
}
