#include "flex_runtime.h"
#include "flex_printf.h"
#include "llm_common.h"

// ----------------------
// RMSNorm
// ----------------------
#include "../../RMSNorm/include/norm.h"
#include "../../RMSNorm/include/RMSNorm.h"

// --------------------
// GEMM (SummaGEMM app)
// --------------------
#include "../../SummaGEMM/include/gemm.h"
#include "../../SummaGEMM/include/SummaGEMM.h"

// ----------------------
// Activation (SiLU MLP)
// ----------------------
#include "../../Activation/include/acti.h"
#include "../../Activation/include/Activation.h"

// ----------------------
// Attention (FlatAttention)
// ----------------------
#include "../../FlatAttention/include/attn.h"
#include "../../FlatAttention/include/FlatAttention.h"

// Alias common buffer roles used in the MLP path.
#define LLM_GEMM1_X_ADDR LLM_H_NORM_ADDR
#define LLM_GEMM1_Z_ADDR LLM_MLP_MID_ADDR
#define LLM_GEMM2_X_ADDR LLM_MLP_MID_ADDR
#define LLM_GEMM2_Z_ADDR LLM_MLP_OUT_ADDR
#define LLM_ACTI_INPUT_ADDR LLM_GEMM1_Z_ADDR
#define LLM_ACTI_OUTPUT_ADDR LLM_GEMM1_Z_ADDR

#ifndef LLM_PACK_CHUNK_ROWS
// Chunk rows for TM<->HM repacking; bounded by L1 scratch size.
#define LLM_PACK_CHUNK_ROWS 256
#endif

// Per-core mapping for SPATZ vector units used in residual addition.
static const uint32_t SPATZ_CHECK_LIST[ARCH_NUM_CORE_PER_CLUSTER] = ARCH_SPATZ_ATTACED_CHECK_LIST;
static const uint32_t SPATZ_SID_LIST[ARCH_NUM_CORE_PER_CLUSTER] = ARCH_SPATZ_ATTACED_SID_LIST;

static inline uint16_t fp16_norm_lut(uint32_t k)
{
    // Deterministic fp16 pattern used to seed dummy hidden states.
    static const uint16_t lut[4] = {0x3400, 0x3800, 0x3C00, 0x4000};
    return lut[k & 3];
}

void llm_common_init_runtime(LLMRuntimeState *state,
                             uint32_t prompt_len,
                             uint32_t decode_steps,
                             uint32_t max_ctx_len)
{
    if (state == 0)
        return;

    state->prompt_len = prompt_len;
    state->decode_steps = decode_steps;
    state->cache_len = 0;
    // max_ctx_len=0 means "use build-time default".
    state->max_ctx_len = max_ctx_len == 0 ? (uint32_t)LLM_MAX_CTX : max_ctx_len;

    // Clamp to compile-time cache capacity.
    if (state->max_ctx_len > (uint32_t)LLM_MAX_CTX)
        state->max_ctx_len = (uint32_t)LLM_MAX_CTX;

    // Prompt cannot exceed allowed context.
    if (state->prompt_len > state->max_ctx_len)
        state->prompt_len = state->max_ctx_len;

    // Decode steps are bounded by remaining cache space.
    if (state->decode_steps > (state->max_ctx_len - state->cache_len))
        state->decode_steps = state->max_ctx_len - state->cache_len;
}

uint32_t llm_common_runtime_is_valid(const LLMRuntimeState *state)
{
    // Return 1 only for a self-consistent context/cache configuration.
    if (state == 0)
        return 0u;
    if (state->max_ctx_len == 0 || state->max_ctx_len > (uint32_t)LLM_MAX_CTX)
        return 0u;
    if (state->prompt_len > state->max_ctx_len)
        return 0u;
    if (state->cache_len > state->max_ctx_len)
        return 0u;
    if (state->decode_steps > (state->max_ctx_len - state->cache_len))
        return 0u;
    return 1u;
}

uint32_t llm_common_cache_can_append(const LLMRuntimeState *state, uint32_t append_tokens)
{
    // Fast path used by decode loop to check cache growth safety.
    if (state == 0)
        return 0u;
    if (state->max_ctx_len == 0 || state->max_ctx_len > (uint32_t)LLM_MAX_CTX)
        return 0u;
    return llm_cache_can_append(state->cache_len, append_tokens, state->max_ctx_len);
}

uint32_t llm_common_cache_append(LLMRuntimeState *state, uint32_t append_tokens)
{
    // Update cache_len in-place only when bounds check passes.
    if (!llm_common_cache_can_append(state, append_tokens))
        return 0u;
    state->cache_len += append_tokens;
    return 1u;
}

void llm_common_attn_profile_default(LLMAttentionRuntimeArgs *args)
{
    if (args == 0)
        return;

    // Keep attn.h as baseline defaults; apps can override selected fields.
    args->speculative_length = (uint32_t)ATTN_SPECULATIVE_LENGTH;
    args->head_dimension = (uint32_t)ATTN_HEAD_DIMEMSION;
    args->num_head = (uint32_t)ATTN_NUM_HEAD;
    args->num_head_group = (uint32_t)ATTN_NUM_HEAD_GROUP;
    args->batch_size = (uint32_t)ATTN_BATCH_SIZE;
    args->flatten_scale_x = (uint32_t)ATTN_FLATTEN_SCALE_X;
    args->flatten_scale_y = (uint32_t)ATTN_FLATTEN_SCALE_Y;
    args->flatten_shape_x = (uint32_t)ATTN_FLATTEN_SHAPE_X;
    args->flatten_shape_y = (uint32_t)ATTN_FLATTEN_SHAPE_Y;
    args->async_enable = (uint32_t)ATTN_FLATTEN_ASYNC;
    args->dump_enable = 0u;
}

static LLMAttentionRuntimeArgs llm_common_resolve_attn_args(const LLMAttentionRuntimeArgs *attn_args)
{
    LLMAttentionRuntimeArgs cfg;
    // Always start from defaults so partially overridden profiles stay valid.
    llm_common_attn_profile_default(&cfg);
    if (attn_args != 0)
        cfg = *attn_args;
    return cfg;
}

static uint32_t llm_common_attn_args_valid(const LLMAttentionRuntimeArgs *cfg)
{
    // Lightweight guards before calling flat_attention().
    if (cfg == 0)
        return 0u;
    if (cfg->speculative_length == 0u || cfg->head_dimension == 0u)
        return 0u;
    if (cfg->num_head == 0u || cfg->num_head_group == 0u || cfg->batch_size == 0u)
        return 0u;
    if (cfg->num_head_group > cfg->num_head)
        return 0u;
    if (cfg->flatten_scale_x == 0u || cfg->flatten_scale_y == 0u)
        return 0u;
    if (cfg->flatten_shape_x == 0u || cfg->flatten_shape_y == 0u)
        return 0u;
    if ((cfg->flatten_shape_x % cfg->flatten_scale_x) != 0u)
        return 0u;
    if ((cfg->flatten_shape_y % cfg->flatten_scale_y) != 0u)
        return 0u;
    return 1u;
}

void llm_common_barrier_init(void)
{
    flex_barrier_xy_init();
}

void llm_common_barrier(void)
{
    flex_global_barrier_xy();
}

void llm_common_timer_start(void)
{
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();
}

void llm_common_timer_end(void)
{
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();
}

uint32_t llm_common_is_lead_core(void)
{
    return (flex_get_core_id() == 0 && flex_get_cluster_id() == 0) ? 1u : 0u;
}

void llm_common_eoc(uint32_t eoc_val)
{
    flex_eoc(eoc_val);
}

void llm_common_log_start(void)
{
    if (llm_common_is_lead_core())
    {
        printf("[LLMForward] Starting LLM forward prototype with %d layers (nanochat-tuned dims)\n",
               LLM_NUM_LAYERS);
    }
}

void llm_common_log_layer_done(uint32_t layer_id)
{
    if (llm_common_is_lead_core())
    {
        printf("[LLMForward] ===== Layer %u complete =====\n", layer_id);
    }
}

void llm_common_init_hidden_state(void)
{
    // Seed H with a simple deterministic pattern across tokens.
    const uint32_t num_tokens = (uint32_t)LLM_T;
    const uint32_t hidden = (uint32_t)LLM_D_MODEL;
    const uint32_t elem_bytes = (uint32_t)LLM_ELEM_SIZE;
    const uint32_t token_bytes = hidden * elem_bytes;

    uint32_t base = local(0);
    base = (base + 63) & ~((uint32_t)63);
    uint32_t L1_H = base;

    for (uint32_t t = flex_get_cluster_id(); t < num_tokens; t += ARCH_NUM_CLUSTER)
    {
        if (flex_get_core_id() == 0)
        {
            uint16_t *buf = (uint16_t *)L1_H;
            uint32_t base_idx = t * hidden;
            for (uint32_t j = 0; j < hidden; ++j)
                buf[j] = fp16_norm_lut(base_idx + j);
        }
        flex_intra_cluster_sync();

        if (flex_is_dm_core())
        {
            flex_dma_async_1d((uint64_t)LLM_H_ADDR + (uint64_t)t * token_bytes, (uint64_t)L1_H, token_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
    }
}

static void llm_zero_hbm_region(uint64_t base, uint64_t nbytes)
{
    // Zero from one cluster to avoid duplicated writes.
    flex_global_barrier_xy();
    if (flex_get_cluster_id() != 0)
    {
        flex_global_barrier_xy();
        return;
    }

    uint32_t l1 = local(0);
    l1 = (l1 + 63) & ~((uint32_t)63);

    const uint32_t CHUNK = 4096;
    if (flex_get_core_id() == 0)
    {
        uint8_t *p = (uint8_t *)l1;
        for (uint32_t i = 0; i < CHUNK; ++i)
            p[i] = 0;
    }
    flex_intra_cluster_sync();

    if (flex_is_dm_core())
    {
        for (uint64_t off = 0; off < nbytes; off += CHUNK)
        {
            uint32_t sz = (uint32_t)((nbytes - off) > CHUNK ? CHUNK : (nbytes - off));
            flex_dma_async_1d(base + off, (uint64_t)l1, sz);
            flex_dma_async_wait_all();
        }
    }

    flex_intra_cluster_sync();
    flex_global_barrier_xy();
}

static void llm_init_eye(uint64_t W_addr)
{
    // Initialize a [d_model, d_model] fp16 identity matrix.
    const uint32_t N = (uint32_t)LLM_D_MODEL;
    const uint32_t row_bytes = N * (uint32_t)LLM_ELEM_SIZE;

    uint32_t l1 = local(0);
    l1 = (l1 + 63) & ~((uint32_t)63);

    for (uint32_t r = flex_get_cluster_id(); r < N; r += ARCH_NUM_CLUSTER)
    {
        if (flex_get_core_id() == 0)
        {
            uint16_t *buf = (uint16_t *)l1;
            for (uint32_t j = 0; j < N; ++j)
                buf[j] = 0;
            buf[r] = 0x3C00;
        }
        flex_intra_cluster_sync();

        if (flex_is_dm_core())
        {
            flex_dma_async_1d(W_addr + (uint64_t)r * row_bytes, (uint64_t)l1, row_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
    }

    flex_global_barrier_xy();
}

void llm_common_init_dummy_weights(uint32_t layer_id)
{
    // Build a pass-through attention projection stack for bring-up.
    llm_zero_hbm_region(llm_w_layer_base(layer_id), (uint64_t)LLM_LAYER_W_STRIDE);

    llm_init_eye(llm_wq_addr(layer_id));
    llm_init_eye(llm_wk_addr(layer_id));
    llm_init_eye(llm_wv_addr(layer_id));
    llm_init_eye(llm_wo_addr(layer_id));
}

static void llm_residual_add_spatz(uint64_t H_addr, uint64_t ADD_addr,
                                   uint32_t num_tokens_arg, uint32_t hidden_arg)
{
    // In-place H += ADD; DM cores move data, SPATZ cores do vector add.
    const uint32_t num_tokens = num_tokens_arg;
    const uint32_t hidden = hidden_arg;
    const uint32_t elem_bytes = (uint32_t)LLM_ELEM_SIZE;
    const uint32_t token_bytes = hidden * elem_bytes;

    uint32_t base = local(0);
    base = (base + 63) & ~((uint32_t)63);

    uint32_t L1_H = base;
    uint32_t L1_ADD = (L1_H + token_bytes + 63) & ~((uint32_t)63);

    const uint32_t core_id = flex_get_core_id();
    const uint32_t spatz_attached = SPATZ_CHECK_LIST[core_id];
    const uint32_t spatz_sid = SPATZ_SID_LIST[core_id];
    const uint32_t spatz_num = ARCH_SPATZ_ATTACED_CORES;

    uint32_t t = flex_get_cluster_id();
    while (t < num_tokens)
    {
        if (flex_is_dm_core())
        {
            flex_dma_async_1d(L1_H, H_addr + (uint64_t)t * token_bytes, token_bytes);
            flex_dma_async_1d(L1_ADD, ADD_addr + (uint64_t)t * token_bytes, token_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();

        if (spatz_attached)
        {
            const uint32_t slice_elems = (hidden + spatz_num - 1) / spatz_num;
            const uint32_t start = spatz_sid * slice_elems;
            uint32_t end = start + slice_elems;
            if (end > hidden)
                end = hidden;

            if (end > start)
            {
                const uint32_t off_bytes = start * elem_bytes;
                const uint32_t vlen = end - start;
                vector_lib_bias(L1_ADD + off_bytes, L1_H + off_bytes, vlen);
            }
        }
        flex_intra_cluster_sync();

        if (flex_is_dm_core())
        {
            flex_dma_async_1d(H_addr + (uint64_t)t * token_bytes, L1_H, token_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();

        t += ARCH_NUM_CLUSTER;
    }
}

static void llm_pack_tm_to_hm(uint64_t src_tm, uint64_t dst_hm, uint32_t seq_len)
{
    // Convert [T, D] row-major to head-major [H, T, Dh] for FlatAttention.
    const uint32_t head = flex_get_cluster_id();
    if (head >= (uint32_t)LLM_N_HEAD || seq_len == 0)
        return;

    const uint32_t row_bytes = (uint32_t)(LLM_HEAD_DIM * LLM_ELEM_SIZE);
    const uint32_t src_row_stride = (uint32_t)(LLM_D_MODEL * LLM_ELEM_SIZE);

    uint32_t l1 = local(0);
    l1 = (l1 + 63) & ~((uint32_t)63);

    const uint64_t src_base = src_tm + (uint64_t)head * (uint64_t)row_bytes;
    const uint64_t dst_base = dst_hm + (uint64_t)head * (uint64_t)seq_len * (uint64_t)row_bytes;

    uint32_t t0 = 0;
    while (t0 < seq_len)
    {
        uint32_t chunk_rows = seq_len - t0;
        if (chunk_rows > (uint32_t)LLM_PACK_CHUNK_ROWS)
            chunk_rows = (uint32_t)LLM_PACK_CHUNK_ROWS;
        uint32_t chunk_bytes = chunk_rows * row_bytes;

        uint64_t src = src_base + (uint64_t)t0 * (uint64_t)src_row_stride;
        uint64_t dst = dst_base + (uint64_t)t0 * (uint64_t)row_bytes;

        if (flex_is_dm_core())
        {
            flex_dma_async_2d((uint64_t)l1, src, row_bytes, row_bytes, src_row_stride, chunk_rows);
            flex_dma_async_wait_all();

            flex_dma_async_1d(dst, (uint64_t)l1, chunk_bytes);
            flex_dma_async_wait_all();
        }

        flex_intra_cluster_sync();
        t0 += chunk_rows;
    }
}

static void llm_unpack_hm_to_tm(uint64_t src_hm, uint64_t dst_tm, uint32_t seq_len)
{
    // Convert FlatAttention output [H, T, Dh] back to [T, D].
    const uint32_t head = flex_get_cluster_id();
    if (head >= (uint32_t)LLM_N_HEAD || seq_len == 0)
        return;

    const uint32_t row_bytes = (uint32_t)(LLM_HEAD_DIM * LLM_ELEM_SIZE);
    const uint32_t dst_row_stride = (uint32_t)(LLM_D_MODEL * LLM_ELEM_SIZE);

    uint32_t l1 = local(0);
    l1 = (l1 + 63) & ~((uint32_t)63);

    const uint64_t src_base = src_hm + (uint64_t)head * (uint64_t)seq_len * (uint64_t)row_bytes;
    const uint64_t dst_base = dst_tm + (uint64_t)head * (uint64_t)row_bytes;

    uint32_t t0 = 0;
    while (t0 < seq_len)
    {
        uint32_t chunk_rows = seq_len - t0;
        if (chunk_rows > (uint32_t)LLM_PACK_CHUNK_ROWS)
            chunk_rows = (uint32_t)LLM_PACK_CHUNK_ROWS;
        uint32_t chunk_bytes = chunk_rows * row_bytes;

        uint64_t src = src_base + (uint64_t)t0 * (uint64_t)row_bytes;
        uint64_t dst = dst_base + (uint64_t)t0 * (uint64_t)dst_row_stride;

        if (flex_is_dm_core())
        {
            flex_dma_async_1d((uint64_t)l1, src, chunk_bytes);
            flex_dma_async_wait_all();

            flex_dma_async_2d(dst, (uint64_t)l1, row_bytes, dst_row_stride, row_bytes, chunk_rows);
            flex_dma_async_wait_all();
        }

        flex_intra_cluster_sync();
        t0 += chunk_rows;
    }
}

static void llm_store_hm_to_kv_cache(uint64_t src_hm_addr,
                                     uint64_t dst_layer_base,
                                     uint32_t token_offset,
                                     uint32_t seq_len)
{
    if (seq_len == 0)
        return;

    const uint32_t row_bytes = (uint32_t)(LLM_HEAD_DIM * LLM_ELEM_SIZE);
    const uint32_t src_head_bytes = seq_len * row_bytes;
    const uint64_t dst_token_off = (uint64_t)token_offset * (uint64_t)row_bytes;

    // KV cache layout is [kv_head][token][head_dim]; each cluster handles head stripes.
    for (uint32_t kv_head = flex_get_cluster_id();
         kv_head < (uint32_t)LLM_N_KV_HEAD;
         kv_head += ARCH_NUM_CLUSTER)
    {
        const uint64_t src = src_hm_addr + (uint64_t)kv_head * (uint64_t)src_head_bytes;
        const uint64_t dst = dst_layer_base + (uint64_t)kv_head * (uint64_t)BYTES_KV_HEAD_CTX + dst_token_off;

        if (flex_is_dm_core())
        {
            flex_dma_async_1d(dst, src, src_head_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
    }
}

static void llm_run_gemm(uint64_t X, uint64_t W, uint64_t Z,
                         uint32_t M, uint32_t N, uint32_t K)
{
    // Small wrapper to keep barrier discipline around SummaGEMM calls.
    flex_global_barrier_xy();
    SummaGEMMInfo info = SummaGEMMAnaylze(
        X, W, Z,
        M, N, K,
        (uint32_t)GEMM_M_TILE,
        (uint32_t)GEMM_N_TILE,
        (uint32_t)GEMM_K_TILE,
        (uint32_t)GEMM_SUMMA_SCALE_X,
        (uint32_t)GEMM_SUMMA_SCALE_Y,
        (uint32_t)GEMM_SUMMA_GROUP_NUMBER,
        (uint32_t)GEMM_SUMMA_GROUP_REDUCE,
        (uint32_t)GEMM_SUMMA_GROUP_SPLITK,
        (uint32_t)GEMM_SUMMA_GROUP_SPLITN,
        (uint32_t)GEMM_SUMMA_GROUP_GAP_X,
        (uint32_t)GEMM_SUMMA_GROUP_GAP_W,
        (uint32_t)GEMM_SUMMA_GROUP_GAP_Z);
    flex_global_barrier_xy();
    SummaGEMMRun(&info);
    flex_global_barrier_xy();
}

void llm_common_dma_dump_u16(uint64_t hbm_addr, uint32_t n_halfwords)
{
    flex_global_barrier_xy();

    if (flex_get_cluster_id() != 0)
    {
        flex_global_barrier_xy();
        return;
    }

    uint32_t base = local(0);
    base = (base + 63) & ~((uint32_t)63);
    uint32_t l1_buf = base;

    uint32_t nbytes = n_halfwords * (uint32_t)sizeof(uint16_t);

    if (flex_is_dm_core())
    {
        flex_dma_async_1d(l1_buf, hbm_addr, nbytes);
        flex_dma_async_wait_all();
    }
    flex_intra_cluster_sync();

    if (flex_get_core_id() == 0)
    {
        volatile uint16_t *buf = (volatile uint16_t *)l1_buf;
        printf("[LLMForward] H dump :");
        for (uint32_t i = 0; i < n_halfwords; ++i)
            printf(" 0x%04x", (unsigned)buf[i]);
        printf("\n");
    }

    flex_intra_cluster_sync();
    flex_global_barrier_xy();
}

static void llm_run_transformer_layer(uint32_t layer_id,
                                      uint32_t q_len,
                                      uint32_t kv_len,
                                      const LLMAttentionRuntimeArgs *attn_args)
{
    // Resolve per-phase runtime profile (prefill/decode) for attention call.
    const LLMAttentionRuntimeArgs attn_cfg = llm_common_resolve_attn_args(attn_args);

    // 1) Pre-attention RMSNorm.
    RMSNormInfo norm_attn_info = Dsv3RMSNormAnaylze(
        (uint32_t)q_len,
        (uint32_t)LLM_D_MODEL,
        (uint64_t)LLM_H_ADDR,
        (uint64_t)LLM_H_NORM_ADDR);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();
    flex_global_barrier_xy();

    Dsv3RMSNormRun(&norm_attn_info);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        printf("[LLMForward] Layer %u: 1 -- Pre-Attn RMSNorm\n", layer_id);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();
    flex_global_barrier_xy();

    uint64_t WQ = llm_wq_addr(layer_id);
    uint64_t WK = llm_wk_addr(layer_id);
    uint64_t WV = llm_wv_addr(layer_id);
    uint64_t WO = llm_wo_addr(layer_id);

    // 2) Attention path: Q/K/V GEMMs + repack + FlatAttention + output projection.
    llm_run_gemm((uint64_t)LLM_H_NORM_ADDR, WQ, (uint64_t)LLM_QKV_TM_ADDR,
                 (uint32_t)q_len, (uint32_t)LLM_D_MODEL, (uint32_t)LLM_D_MODEL);
    flex_global_barrier_xy();
    llm_pack_tm_to_hm((uint64_t)LLM_QKV_TM_ADDR, (uint64_t)LLM_Q_HM_ADDR, (uint32_t)q_len);
    flex_global_barrier_xy();

    llm_run_gemm((uint64_t)LLM_H_NORM_ADDR, WK, (uint64_t)LLM_QKV_TM_ADDR,
                 (uint32_t)kv_len, (uint32_t)LLM_D_MODEL, (uint32_t)LLM_D_MODEL);
    flex_global_barrier_xy();
    llm_pack_tm_to_hm((uint64_t)LLM_QKV_TM_ADDR, (uint64_t)LLM_K_HM_ADDR, (uint32_t)kv_len);
    flex_global_barrier_xy();

    llm_run_gemm((uint64_t)LLM_H_NORM_ADDR, WV, (uint64_t)LLM_QKV_TM_ADDR,
                 (uint32_t)kv_len, (uint32_t)LLM_D_MODEL, (uint32_t)LLM_D_MODEL);
    flex_global_barrier_xy();
    llm_pack_tm_to_hm((uint64_t)LLM_QKV_TM_ADDR, (uint64_t)LLM_V_HM_ADDR, (uint32_t)kv_len);
    flex_global_barrier_xy();

    int attn_status = 1;
    if (llm_common_attn_args_valid(&attn_cfg))
    {
        // Runtime profile replaces compile-time ATTN_* constants here.
        attn_status = flat_attention(
            (uint32_t)kv_len,
            (uint32_t)q_len,
            (uint32_t)attn_cfg.speculative_length,
            (uint32_t)attn_cfg.head_dimension,
            (uint32_t)attn_cfg.num_head,
            (uint32_t)attn_cfg.num_head_group,
            (uint32_t)attn_cfg.batch_size,
            (uint32_t)attn_cfg.flatten_scale_x,
            (uint32_t)attn_cfg.flatten_scale_y,
            (uint32_t)attn_cfg.flatten_shape_x,
            (uint32_t)attn_cfg.flatten_shape_y,
            (uint64_t)LLM_Q_HM_ADDR,
            (uint64_t)LLM_K_HM_ADDR,
            (uint64_t)LLM_V_HM_ADDR,
            (uint64_t)LLM_O_HM_ADDR,
            (uint64_t)0,
            (uint32_t)attn_cfg.async_enable,
            (uint32_t)attn_cfg.dump_enable);
    }
    else if (llm_common_is_lead_core())
    {
        // Keep run alive for debugging but mark the layer as invalid profile.
        printf("[LLMForward] invalid runtime attention profile\n");
    }

    flex_global_barrier_xy();
    llm_unpack_hm_to_tm((uint64_t)LLM_O_HM_ADDR, (uint64_t)LLM_ATTN_O_ADDR, (uint32_t)q_len);
    flex_global_barrier_xy();

    llm_run_gemm((uint64_t)LLM_ATTN_O_ADDR, WO, (uint64_t)LLM_ATTN_PROJ_TM_ADDR,
                 (uint32_t)q_len, (uint32_t)LLM_D_MODEL, (uint32_t)LLM_D_MODEL);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        printf("[LLMForward] Layer %u: Attention status=%d\n", layer_id, attn_status);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();
    flex_global_barrier_xy();

    // 3) Residual add: H <- H + AttentionOut.
    llm_residual_add_spatz((uint64_t)LLM_H_ADDR, (uint64_t)LLM_ATTN_PROJ_TM_ADDR, q_len, (uint32_t)LLM_D_MODEL);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        printf("[LLMForward] Layer %u: Attention residual applied (proj)\n", layer_id);

    // 4) Pre-MLP RMSNorm.
    RMSNormInfo norm_mlp_info = Dsv3RMSNormAnaylze(
        (uint32_t)q_len,
        (uint32_t)LLM_D_MODEL,
        (uint64_t)LLM_H_ADDR,
        (uint64_t)LLM_H_NORM_ADDR);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();
    flex_global_barrier_xy();

    Dsv3RMSNormRun(&norm_mlp_info);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        printf("[LLMForward] Layer %u: 4 -- Pre-MLP RMSNorm\n", layer_id);

    // 5) MLP up-projection.
    uint64_t W1 = llm_w1_up_addr(layer_id);
    SummaGEMMInfo gemm1_info = SummaGEMMAnaylze(
        (uint64_t)LLM_GEMM1_X_ADDR,
        (uint64_t)W1,
        (uint64_t)LLM_GEMM1_Z_ADDR,
        (uint32_t)q_len,
        (uint32_t)LLM_D_FF,
        (uint32_t)LLM_D_MODEL,
        (uint32_t)GEMM_M_TILE,
        (uint32_t)GEMM_N_TILE,
        (uint32_t)GEMM_K_TILE,
        (uint32_t)GEMM_SUMMA_SCALE_X,
        (uint32_t)GEMM_SUMMA_SCALE_Y,
        (uint32_t)GEMM_SUMMA_GROUP_NUMBER,
        (uint32_t)GEMM_SUMMA_GROUP_REDUCE,
        (uint32_t)GEMM_SUMMA_GROUP_SPLITK,
        (uint32_t)GEMM_SUMMA_GROUP_SPLITN,
        (uint32_t)GEMM_SUMMA_GROUP_GAP_X,
        (uint32_t)GEMM_SUMMA_GROUP_GAP_W,
        (uint32_t)GEMM_SUMMA_GROUP_GAP_Z);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();
    flex_global_barrier_xy();

    SummaGEMMRun(&gemm1_info);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        printf("[LLMForward] Layer %u: 5 -- GEMM1 (up)\n", layer_id);

    // 6) SiLU activation in-place on intermediate buffer.
    ActivationInfo act_info = ActivationAnaylze(
        (uint32_t)q_len,
        (uint32_t)LLM_D_FF,
        (uint32_t)ACTI_GATE_ENABLE,
        (uint32_t)ACTI_BIAS_ENABLE,
        (uint64_t)LLM_ACTI_INPUT_ADDR,
        (uint64_t)LLM_ACTI_OUTPUT_ADDR,
        (uint64_t)0,
        (uint64_t)0);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();
    flex_global_barrier_xy();

    ActivationRun(&act_info);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        printf("[LLMForward] Layer %u: 6 -- SiLU\n", layer_id);

    // 7) MLP down-projection.
    uint64_t W2 = llm_w2_down_addr(layer_id);
    SummaGEMMInfo gemm2_info = SummaGEMMAnaylze(
        (uint64_t)LLM_GEMM2_X_ADDR,
        (uint64_t)W2,
        (uint64_t)LLM_GEMM2_Z_ADDR,
        (uint32_t)q_len,
        (uint32_t)LLM_D_MODEL,
        (uint32_t)LLM_D_FF,
        (uint32_t)GEMM_M_TILE,
        (uint32_t)GEMM_N_TILE,
        (uint32_t)GEMM_K_TILE,
        (uint32_t)GEMM_SUMMA_SCALE_X,
        (uint32_t)GEMM_SUMMA_SCALE_Y,
        (uint32_t)GEMM_SUMMA_GROUP_NUMBER,
        (uint32_t)GEMM_SUMMA_GROUP_REDUCE,
        (uint32_t)GEMM_SUMMA_GROUP_SPLITK,
        (uint32_t)GEMM_SUMMA_GROUP_SPLITN,
        (uint32_t)GEMM_SUMMA_GROUP_GAP_X,
        (uint32_t)GEMM_SUMMA_GROUP_GAP_W,
        (uint32_t)GEMM_SUMMA_GROUP_GAP_Z);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();
    flex_global_barrier_xy();

    SummaGEMMRun(&gemm2_info);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        printf("[LLMForward] Layer %u: 7 -- GEMM2\n", layer_id);

    // 8) Residual add: H <- H + MLPOut.
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();
    flex_global_barrier_xy();

    llm_residual_add_spatz((uint64_t)LLM_H_ADDR, (uint64_t)LLM_MLP_OUT_ADDR, q_len, (uint32_t)LLM_D_MODEL);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        printf("[LLMForward] Layer %u: 8 -- MLP residual applied\n", layer_id);
}

void llm_common_run_prefill_layer(uint32_t layer_id,
                                  uint32_t q_len,
                                  uint32_t kv_len,
                                  const LLMAttentionRuntimeArgs *attn_args)
{
    // Prefill currently reuses the shared transformer layer primitive.
    llm_run_transformer_layer(layer_id, q_len, kv_len, attn_args);
}

void llm_common_store_prefill_kv_cache(uint32_t layer_id, uint32_t kv_len)
{
    if (!llm_cache_valid_layer(layer_id) || kv_len == 0)
        return;

    // Prefill writes the full prompt window starting at cache token 0.
    if (kv_len > (uint32_t)LLM_MAX_CTX)
        kv_len = (uint32_t)LLM_MAX_CTX;

    flex_global_barrier_xy();
    llm_store_hm_to_kv_cache((uint64_t)LLM_K_HM_ADDR, llm_k_cache_layer_base(layer_id), 0u, kv_len);
    flex_global_barrier_xy();
    llm_store_hm_to_kv_cache((uint64_t)LLM_V_HM_ADDR, llm_v_cache_layer_base(layer_id), 0u, kv_len);
    flex_global_barrier_xy();
}

void llm_common_store_decode_kv_cache(uint32_t layer_id, uint32_t cache_pos, uint32_t append_len)
{
    if (!llm_cache_valid_layer(layer_id) || append_len == 0)
        return;

    if (cache_pos >= (uint32_t)LLM_MAX_CTX)
        return;

    // Decode appends only the newly generated token slice at cache_pos.
    if (!llm_cache_can_append(cache_pos, append_len, (uint32_t)LLM_MAX_CTX))
        append_len = (uint32_t)LLM_MAX_CTX - cache_pos;

    if (append_len == 0)
        return;

    flex_global_barrier_xy();
    llm_store_hm_to_kv_cache((uint64_t)LLM_K_HM_ADDR, llm_k_cache_layer_base(layer_id), cache_pos, append_len);
    flex_global_barrier_xy();
    llm_store_hm_to_kv_cache((uint64_t)LLM_V_HM_ADDR, llm_v_cache_layer_base(layer_id), cache_pos, append_len);
    flex_global_barrier_xy();
}

void llm_common_run_decode_layer(uint32_t layer_id,
                                 uint32_t q_len,
                                 uint32_t kv_len,
                                 const LLMAttentionRuntimeArgs *attn_args)
{
    // Decode currently uses the same primitive; cache-specialized path comes later.
    llm_run_transformer_layer(layer_id, q_len, kv_len, attn_args);
}
