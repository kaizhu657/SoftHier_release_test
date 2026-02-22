#include "llm_common.h"

#define LLM_DEBUG_DUMP 0

int main()
{
    uint32_t eoc_val = 0;
    LLMAttentionRuntimeArgs attn_profile;

    uint32_t q_len = (uint32_t)LLM_T;
    uint32_t kv_len = (uint32_t)LLM_T;

    LLMRuntimeState state;
    llm_common_init_runtime(&state, q_len, 0, (uint32_t)LLM_MAX_CTX);
    // Legacy app sticks to default attention profile for compatibility.
    llm_common_attn_profile_default(&attn_profile);

    llm_common_barrier_init();

    // Init weights and activations in HBM (layer 0 deterministic baseline).
    llm_common_init_dummy_weights(0);

#if LLM_DEBUG_DUMP
    llm_common_dma_dump_u16(llm_wq_addr(0), 16);
#endif

    llm_common_init_hidden_state();
    llm_common_barrier();

    llm_common_log_start();

    for (uint32_t layer = 0; layer < (uint32_t)LLM_NUM_LAYERS; ++layer)
    {
        llm_common_barrier();

        llm_common_timer_start();
        llm_common_barrier();

        llm_common_run_prefill_layer(layer, q_len, kv_len, &attn_profile);

        llm_common_barrier();
        if (llm_common_is_lead_core())
        {
            llm_common_timer_end();
            llm_common_log_layer_done(layer);
        }
    }

    llm_common_dma_dump_u16((uint64_t)LLM_H_ADDR, 8);

    llm_common_barrier();
    llm_common_eoc(eoc_val);
    return 0;
}
