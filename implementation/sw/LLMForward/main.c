// LLMForward main.c
// ------------------
//
// Per layer, we do:
//
//   1. Pre-Attention RMSNorm on hidden state H
//   2. Self-Attention with FlatAttention (Q = K = V = RMSNorm(H))
//   3. Residual add: H <- H + AttnOut
//   4. Pre-MLP RMSNorm on updated H
//   5. MLP block (GEMM1 -> SiLU -> GEMM2)
//   6. Residual add: H <- H + MLP_out
//
// The layer is applied LLM_NUM_LAYERS times in sequence, always updating H in
// place at LLM_H_ADDR.
//
// We reuse kernels:
//
//   - sw/RMSNorm       : Dsv3RMSNormAnaylze + Dsv3RMSNormRun
//   - sw/FlatAttention : flat_attention()
//   - sw/SummaGEMM     : SummaGEMMAnaylze + SummaGEMMRun
//   - sw/Activation    : ActivationAnaylze + ActivationRun

#include "flex_runtime.h"
#include "flex_printf.h"
#include "llm_layout.h"
#include "llm_weights.h"
// ----------------------
// RMSNorm
// ----------------------
#include "../RMSNorm/include/norm.h"    // NORM_M_SIZE, NORM_N_SIZE, etc.
#include "../RMSNorm/include/RMSNorm.h" // RMSNormInfo, Dsv3RMSNormAnaylze, Dsv3RMSNormRun

// --------------------
// GEMM (SummaGEMM app)
// --------------------
#include "../SummaGEMM/include/gemm.h"      // GEMM_* config macros
#include "../SummaGEMM/include/SummaGEMM.h" // SummaGEMMInfo, SummaGEMMAnaylze, SummaGEMMRun

// ----------------------
// Activation (SiLU MLP)
// ----------------------
#include "../Activation/include/acti.h"       // ACTI_M_SIZE, ACTI_N_SIZE, etc.
#include "../Activation/include/Activation.h" // ActivationInfo, ActivationAnaylze, ActivationRun

// ----------------------
// Attention (FlatAttention)
// ----------------------
#include "../FlatAttention/include/attn.h"          // ATTN_* macros
#include "../FlatAttention/include/FlatAttention.h" // flat_attention(...)

// =======================================================
// Actual addresses come from llm_layout.h:
//   - LLM_H_ADDR        : hidden state H [T, d_model]
//   - LLM_H_NORM_ADDR   : normalized hidden state H_norm
//   - LLM_MLP_MID_ADDR  : MLP intermediate [T, d_ff]
//   - LLM_MLP_OUT_ADDR  : MLP output [T, d_model]
//   - LLM_ATTN_O_ADDR   : Attention output [T, d_model] (via [T, n_heads, head_dim])
//   - LLM_W_BASE        : weights base (from SummaGEMM preload region)

// For the first GEMM (MLP projection).
// We treat RMSNorm output as X, GEMM weights at W_ADDR, output at Z_EADDR.
#define LLM_GEMM1_X_ADDR LLM_H_NORM_ADDR  // Input to GEMM1: H_norm
#define LLM_GEMM1_Z_ADDR LLM_MLP_MID_ADDR // Z_EADDR from SummaGEMM preload

// For the second GEMM (MLP down projection).
// Input is activation output, weights are a different matrix (down projection),
// output at LLM_MLP_OUT_ADDR.
#define LLM_GEMM2_X_ADDR LLM_MLP_MID_ADDR // After SiLU (in-place)
#define LLM_GEMM2_Z_ADDR LLM_MLP_OUT_ADDR // Z_GADDR from SummaGEMM preload

// Activation runs in-place on GEMM1 output [M x N], here [128 x 4096] as per acti.h.
#define LLM_ACTI_INPUT_ADDR LLM_GEMM1_Z_ADDR
#define LLM_ACTI_OUTPUT_ADDR LLM_GEMM1_Z_ADDR

// Residual connection for MLP:
//   H_in  = H at LLM_H_ADDR
//   H_add = MLP output at LLM_MLP_OUT_ADDR
//   H_out = H_in + H_add  (written back to LLM_H_ADDR)
#define LLM_RESID_IN_ADDR LLM_H_ADDR
#define LLM_RESID_ADD_ADDR LLM_MLP_OUT_ADDR
#define LLM_RESID_OUT_ADDR LLM_H_ADDR

#define LLM_DEBUG_DUMP 0

static inline uint16_t fp16_norm_lut(uint32_t k)
{
    // Deterministic normal fp16 values:
    // 0.25 = 0x3400, 0.5 = 0x3800, 1.0 = 0x3C00, 2.0 = 0x4000
    static const uint16_t lut[4] = {0x3400, 0x3800, 0x3C00, 0x4000};
    return lut[k & 3];
}

// How many token-rows to move per chunk.
// 256 rows * 128B/row = 32KB L1 scratch, usually safe.
// Can tune later.
#ifndef LLM_PACK_CHUNK_ROWS
#define LLM_PACK_CHUNK_ROWS 256
#endif

// =======================================================
// Helper: initialize hidden state H
// =======================================================

static void llm_init_hidden_state()
{
    const uint32_t num_tokens = (uint32_t)LLM_T;         // 128
    const uint32_t hidden = (uint32_t)LLM_D_MODEL;       // 1024
    const uint32_t elem_bytes = (uint32_t)LLM_ELEM_SIZE; // 2
    const uint32_t token_bytes = hidden * elem_bytes;

    // scratch in L1
    uint32_t base = local(0);
    base = (base + 63) & ~((uint32_t)63);
    uint32_t L1_H = base;

    // cluster stripes tokens
    for (uint32_t t = flex_get_cluster_id(); t < num_tokens; t += ARCH_NUM_CLUSTER)
    {
        // Fill one token in L1 (single core per cluster)
        if (flex_get_core_id() == 0)
        {
            uint16_t *buf = (uint16_t *)L1_H;
            uint32_t base_idx = t * hidden;
            for (uint32_t j = 0; j < hidden; ++j)
                buf[j] = fp16_norm_lut(base_idx + j);
        }
        flex_intra_cluster_sync();

        // DMA out (DM core only)
        if (flex_is_dm_core())
        {
            flex_dma_async_1d((uint64_t)LLM_H_ADDR + (uint64_t)t * token_bytes, (uint64_t)L1_H, token_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
    }
}

// =======================================================
// Helper: force the data to 0 in HBM
// =======================================================
static void llm_zero_hbm_region(uint64_t base, uint64_t nbytes)
{
    // Only cluster0 does the init, everyone else waits.
    flex_global_barrier_xy();
    if (flex_get_cluster_id() != 0)
    {
        flex_global_barrier_xy();
        return;
    }

    // Use a small L1 zero buffer.
    uint32_t l1 = local(0);
    l1 = (l1 + 63) & ~((uint32_t)63);

    const uint32_t CHUNK = 4096; // 4KB
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

// =======================================================
// Initialize a 1024x1024 fp16 identity matrix at W_addr in HBM.
// Work is striped across clusters by row index; within each row, core0 fills an L1
// buffer and the DM core DMAs it to HBM.
// =======================================================
static void llm_init_eye(uint64_t W_addr)
{
    const uint32_t N = (uint32_t)LLM_D_MODEL;               // 1024
    const uint32_t row_bytes = N * (uint32_t)LLM_ELEM_SIZE; // 2048 bytes

    uint32_t l1 = local(0);
    l1 = (l1 + 63) & ~((uint32_t)63);

    for (uint32_t r = flex_get_cluster_id(); r < N; r += ARCH_NUM_CLUSTER)
    {
        if (flex_get_core_id() == 0)
        {
            uint16_t *buf = (uint16_t *)l1;
            for (uint32_t j = 0; j < N; ++j)
                buf[j] = 0;
            buf[r] = 0x3C00; // fp16(1.0)
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

// =======================================================
// Create a deterministic weight set for one layer:
//   1) zero the entire per-layer weight region
//   2) set Wq/Wk/Wv/Wo to identity
//   3) leave MLP weights at zero to isolate and validate the attention path first
// =======================================================

static void llm_init_dummy_weights(uint32_t layer_id)
{
    // zero the whole layer region first
    llm_zero_hbm_region(llm_w_layer_base(layer_id), (uint64_t)LLM_LAYER_W_STRIDE);

    // make attention projections identity
    llm_init_eye(llm_wq_addr(layer_id));
    llm_init_eye(llm_wk_addr(layer_id));
    llm_init_eye(llm_wv_addr(layer_id));
    llm_init_eye(llm_wo_addr(layer_id));

    // keep MLP weights zero for now (clean isolation of the attention workflow)
}

// =======================================================
// Helper: residual add for MLP
// =======================================================
//
// This function performs an elementwise addition over the hidden state:
//
//   H_addr[i] = H_addr[i] + ADD_addr[i],  i = 0..M*N-1
//

// Put at file scope (NOT on stack)
static const uint32_t SPATZ_CHECK_LIST[ARCH_NUM_CORE_PER_CLUSTER] = ARCH_SPATZ_ATTACED_CHECK_LIST;
static const uint32_t SPATZ_SID_LIST[ARCH_NUM_CORE_PER_CLUSTER] = ARCH_SPATZ_ATTACED_SID_LIST;

static void llm_residual_add_spatz(uint64_t H_addr, uint64_t ADD_addr,
                                   uint32_t num_tokens_arg, uint32_t hidden_arg)
{
    const uint32_t num_tokens = num_tokens_arg;
    const uint32_t hidden     = hidden_arg;
    const uint32_t elem_bytes = (uint32_t)LLM_ELEM_SIZE; // 2 for fp16
    const uint32_t token_bytes = hidden * elem_bytes;

    // ---- scratch allocation in L1/TCDM ----
    // Align base to 64B; keep both buffers aligned.
    uint32_t base = local(0);
    base = (base + 63) & ~((uint32_t)63);

    uint32_t L1_H   = base;
    uint32_t L1_ADD = (L1_H + token_bytes + 63) & ~((uint32_t)63); // align second buffer too

    // ---- spatz/core info ----
    const uint32_t core_id        = flex_get_core_id();
    const uint32_t spatz_attached = SPATZ_CHECK_LIST[core_id];
    const uint32_t spatz_sid      = SPATZ_SID_LIST[core_id];
    const uint32_t spatz_num      = ARCH_SPATZ_ATTACED_CORES; // e.g., 4

    // ---- token distribution across clusters ----
    uint32_t t = flex_get_cluster_id();
    while (t < num_tokens)
    {
        // DMA in (DM core only): HBM -> L1
        if (flex_is_dm_core())
        {
            flex_dma_async_1d(L1_H,   H_addr   + (uint64_t)t * token_bytes, token_bytes);
            flex_dma_async_1d(L1_ADD, ADD_addr + (uint64_t)t * token_bytes, token_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();

        // Vector add on Spatz cores: L1_H += L1_ADD (slice per spatz core)
        if (spatz_attached)
        {
            // Ceil-div slice so we handle non-multiple hidden sizes safely.
            const uint32_t slice_elems = (hidden + spatz_num - 1) / spatz_num;
            const uint32_t start = spatz_sid * slice_elems;
            uint32_t end = start + slice_elems;
            if (end > hidden) end = hidden;

            if (end > start)
            {
                const uint32_t off_bytes = start * elem_bytes;
                const uint32_t vlen = end - start;

                // vector_lib_bias(i_addr, o_addr, vlen) does: o += i
                vector_lib_bias(L1_ADD + off_bytes, L1_H + off_bytes, vlen);
            }
        }
        flex_intra_cluster_sync();

        // DMA out (DM core only): L1 -> HBM
        if (flex_is_dm_core())
        {
            flex_dma_async_1d(H_addr + (uint64_t)t * token_bytes, L1_H, token_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();

        t += ARCH_NUM_CLUSTER;
    }
}


// =======================================================
// Transform tm (token-major) to hm (head-major)
// =======================================================

static void llm_pack_tm_to_hm(uint64_t src_tm, uint64_t dst_hm, uint32_t seq_len)
{
    const uint32_t head = flex_get_cluster_id();
    if (head >= (uint32_t)LLM_N_HEAD || seq_len == 0) return;

    const uint32_t row_bytes      = (uint32_t)(LLM_HEAD_DIM * LLM_ELEM_SIZE);  // 64*2=128B
    const uint32_t src_row_stride = (uint32_t)(LLM_D_MODEL  * LLM_ELEM_SIZE);  // 1024*2=2048B

    // L1 scratch
    uint32_t l1 = local(0);
    l1 = (l1 + 63) & ~((uint32_t)63);

    // Token-major source starts at column offset for this head
    // (token 0, column head*head_dim)
    const uint64_t src_base = src_tm + (uint64_t)head * (uint64_t)row_bytes;

    // Head-major destination block for this head: [seq_len x head_dim]
    const uint64_t dst_base = dst_hm + (uint64_t)head * (uint64_t)seq_len * (uint64_t)row_bytes;

    uint32_t t0 = 0;
    while (t0 < seq_len)
    {
        uint32_t chunk_rows  = seq_len - t0;
        if (chunk_rows > (uint32_t)LLM_PACK_CHUNK_ROWS) chunk_rows = (uint32_t)LLM_PACK_CHUNK_ROWS;
        uint32_t chunk_bytes = chunk_rows * row_bytes;

        // addresses for this chunk
        uint64_t src = src_base + (uint64_t)t0 * (uint64_t)src_row_stride;
        uint64_t dst = dst_base + (uint64_t)t0 * (uint64_t)row_bytes;

        if (flex_is_dm_core())
        {
            // HBM (token-major slice) -> L1 contiguous [chunk_rows, head_dim]
            flex_dma_async_2d(
                (uint64_t)l1,       // dst (L1)
                src,                // src (HBM)
                row_bytes,          // size per row
                row_bytes,          // dst stride (packed)
                src_row_stride,     // src stride (next token row)
                chunk_rows          // repeat
            );
            flex_dma_async_wait_all();

            // L1 -> HBM (head-major) contiguous write
            flex_dma_async_1d(dst, (uint64_t)l1, chunk_bytes);
            flex_dma_async_wait_all();
        }

        flex_intra_cluster_sync();
        t0 += chunk_rows;
    }
}

// =======================================================
// Transform hm (head-major) to tm (token-major)
// =======================================================

static void llm_unpack_hm_to_tm(uint64_t src_hm, uint64_t dst_tm, uint32_t seq_len)
{
    const uint32_t head = flex_get_cluster_id();
    if (head >= (uint32_t)LLM_N_HEAD || seq_len == 0) return;

    const uint32_t row_bytes      = (uint32_t)(LLM_HEAD_DIM * LLM_ELEM_SIZE);  // 128B
    const uint32_t dst_row_stride = (uint32_t)(LLM_D_MODEL  * LLM_ELEM_SIZE);  // 2048B

    // L1 scratch
    uint32_t l1 = local(0);
    l1 = (l1 + 63) & ~((uint32_t)63);

    // Head-major source block for this head: [seq_len x head_dim]
    const uint64_t src_base = src_hm + (uint64_t)head * (uint64_t)seq_len * (uint64_t)row_bytes;

    // Token-major destination starts at column offset for this head
    const uint64_t dst_base = dst_tm + (uint64_t)head * (uint64_t)row_bytes;

    uint32_t t0 = 0;
    while (t0 < seq_len)
    {
        uint32_t chunk_rows  = seq_len - t0;
        if (chunk_rows > (uint32_t)LLM_PACK_CHUNK_ROWS) chunk_rows = (uint32_t)LLM_PACK_CHUNK_ROWS;
        uint32_t chunk_bytes = chunk_rows * row_bytes;

        uint64_t src = src_base + (uint64_t)t0 * (uint64_t)row_bytes;
        uint64_t dst = dst_base + (uint64_t)t0 * (uint64_t)dst_row_stride;

        if (flex_is_dm_core())
        {
            // HBM (head-major) -> L1 contiguous
            flex_dma_async_1d((uint64_t)l1, src, chunk_bytes);
            flex_dma_async_wait_all();

            // L1 -> HBM (token-major scatter into the correct columns each row)
            flex_dma_async_2d(
                dst,                // dst (HBM token-major + column offset)
                (uint64_t)l1,       // src (L1 packed)
                row_bytes,          // size per row
                dst_row_stride,     // dst stride (next token row)
                row_bytes,          // src stride (next packed row)
                chunk_rows
            );
            flex_dma_async_wait_all();
        }

        flex_intra_cluster_sync();
        t0 += chunk_rows;
    }
}

// =======================================================
// Wrapper for GEMM
// =======================================================

static void llm_run_gemm(uint64_t X, uint64_t W, uint64_t Z,
                         uint32_t M, uint32_t N, uint32_t K)
{
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

// =======================================================
// For checking the final output
// =======================================================

static void llm_dma_dump_u16(uint64_t hbm_addr, uint32_t n_halfwords)
{
    flex_global_barrier_xy();

    // Only cluster 0 participates in DMA+print
    if (flex_get_cluster_id() != 0)
    {
        flex_global_barrier_xy();
        return;
    }

    // L1 scratch (aligned)
    uint32_t base = local(0);
    base = (base + 63) & ~((uint32_t)63);
    uint32_t l1_buf = base;

    uint32_t nbytes = n_halfwords * (uint32_t)sizeof(uint16_t);

    // DMA HBM -> L1 on DM core
    if (flex_is_dm_core())
    {
        flex_dma_async_1d(l1_buf, hbm_addr, nbytes);
        flex_dma_async_wait_all();
    }
    flex_intra_cluster_sync();

    // Print from core 0 (NO 64-bit printf args)
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

// =======================================================
// One Transformer-like layer forward pass
// =======================================================
//
// This function operates in-place on H in HBM:
//
//   - Reads H from LLM_H_ADDR
//   - Writes H_norm to LLM_H_NORM_ADDR
//   - Writes AttnOut to LLM_ATTN_O_ADDR and adds it back to H
//   - Runs MLP and adds MLP_out back to H
//
static void llm_layer_forward(int layer_id, uint32_t q_len, uint32_t kv_len)
{

    /*******************************************/
    /*  1. Pre-Attention RMSNorm over H        */
    /*******************************************/

    RMSNormInfo norm_attn_info = Dsv3RMSNormAnaylze(
        (uint32_t)q_len,
        (uint32_t)LLM_D_MODEL,
        (uint64_t)LLM_H_ADDR,     // input: H
        (uint64_t)LLM_H_NORM_ADDR // output: H_norm (for attention)
    );

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
    {
        printf("[LLMForward] Layer %d: 1 -- Pre-Attn RMSNorm\n", layer_id);
    }

    /**********************************************/
    /*  2. Self-Attention using FlatAttention     */
    /**********************************************/
    //
    //   Reuse H_norm as Q/K/V. Shapes:
    //   H_norm : [T=128, d_model=1024]
    //   Q/K/V  : [T, num_head=16, head_dim=64] (16*64 = 1024)
    //   O      : [T, num_head, head_dim] (same size as H)
    //   H_norm → (QKV GEMMs) → pack → flat_attention → unpack → Wo GEMM → residual

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();
    flex_global_barrier_xy();

    uint64_t WQ = llm_wq_addr(layer_id);
    uint64_t WK = llm_wk_addr(layer_id);
    uint64_t WV = llm_wv_addr(layer_id);
    uint64_t WO = llm_wo_addr(layer_id);

    // --- Q = H_norm @ Wq  -> QKV_TM (scratch), then pack to Q_HM
    llm_run_gemm((uint64_t)LLM_H_NORM_ADDR, WQ, (uint64_t)LLM_QKV_TM_ADDR,
                 (uint32_t)q_len, (uint32_t)LLM_D_MODEL, (uint32_t)LLM_D_MODEL);

    // For debug
#if LLM_DEBUG_DUMP
    llm_dma_dump_u16((uint64_t)LLM_H_NORM_ADDR, 8);
    llm_dma_dump_u16((uint64_t)LLM_QKV_TM_ADDR, 8);
#endif

    flex_global_barrier_xy();
    llm_pack_tm_to_hm((uint64_t)LLM_QKV_TM_ADDR, (uint64_t)LLM_Q_HM_ADDR, (uint32_t)q_len);
    flex_global_barrier_xy();

    // For debug
#if LLM_DEBUG_DUMP
    llm_dma_dump_u16((uint64_t)LLM_Q_HM_ADDR + 0 * (uint64_t)BYTES_HM_HEAD, 8);
#endif

    // --- K
    llm_run_gemm((uint64_t)LLM_H_NORM_ADDR, WK, (uint64_t)LLM_QKV_TM_ADDR,
                 (uint32_t)kv_len, (uint32_t)LLM_D_MODEL, (uint32_t)LLM_D_MODEL);
    flex_global_barrier_xy();
    llm_pack_tm_to_hm((uint64_t)LLM_QKV_TM_ADDR, (uint64_t)LLM_K_HM_ADDR, (uint32_t)kv_len);
    flex_global_barrier_xy();

    // --- V
    llm_run_gemm((uint64_t)LLM_H_NORM_ADDR, WV, (uint64_t)LLM_QKV_TM_ADDR,
                 (uint32_t)kv_len, (uint32_t)LLM_D_MODEL, (uint32_t)LLM_D_MODEL);
    flex_global_barrier_xy();
    llm_pack_tm_to_hm((uint64_t)LLM_QKV_TM_ADDR, (uint64_t)LLM_V_HM_ADDR, (uint32_t)kv_len);
    flex_global_barrier_xy();

    // --- flat_attention on head-major buffers: O_HM
    int attn_status = flat_attention(
        (uint32_t)kv_len,
        (uint32_t)q_len,
        (uint32_t)ATTN_SPECULATIVE_LENGTH,
        (uint32_t)ATTN_HEAD_DIMEMSION,
        (uint32_t)ATTN_NUM_HEAD,
        (uint32_t)ATTN_NUM_HEAD_GROUP,
        (uint32_t)ATTN_BATCH_SIZE,
        (uint32_t)ATTN_FLATTEN_SCALE_X,
        (uint32_t)ATTN_FLATTEN_SCALE_Y,
        (uint32_t)ATTN_FLATTEN_SHAPE_X,
        (uint32_t)ATTN_FLATTEN_SHAPE_Y,
        (uint64_t)LLM_Q_HM_ADDR,
        (uint64_t)LLM_K_HM_ADDR,
        (uint64_t)LLM_V_HM_ADDR,
        (uint64_t)LLM_O_HM_ADDR,
        (uint64_t)0,
        (uint32_t)ATTN_FLATTEN_ASYNC,
        (uint32_t)0);

    flex_global_barrier_xy();
    // --- unpack O_HM -> O_TM (LLM_ATTN_O_ADDR)
    llm_unpack_hm_to_tm((uint64_t)LLM_O_HM_ADDR, (uint64_t)LLM_ATTN_O_ADDR, (uint32_t)q_len);
    flex_global_barrier_xy();

    // --- AttnProj = O_TM @ Wo -> reuse MLP_OUT as projection buffer
    llm_run_gemm((uint64_t)LLM_ATTN_O_ADDR, WO, (uint64_t)LLM_ATTN_PROJ_TM_ADDR,
                 (uint32_t)q_len, (uint32_t)LLM_D_MODEL, (uint32_t)LLM_D_MODEL);
    flex_global_barrier_xy();

    // For debug
#if LLM_DEBUG_DUMP
    llm_dma_dump_u16((uint64_t)LLM_ATTN_O_ADDR, 8);
    llm_dma_dump_u16((uint64_t)LLM_ATTN_PROJ_TM_ADDR, 8);
#endif

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
    {
        printf("[LLMForward] Layer %d: Attention status=%d\n", layer_id, attn_status);
    }

    /*******************************************/
    /*  3. Attention Residual: H += AttnProj   */
    /*******************************************/
    flex_global_barrier_xy();

    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();
    flex_global_barrier_xy();

    llm_residual_add_spatz((uint64_t)LLM_H_ADDR, (uint64_t)LLM_ATTN_PROJ_TM_ADDR, q_len, (uint32_t)LLM_D_MODEL);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
    {
        printf("[LLMForward] Layer %d: Attention residual applied (proj)\n", layer_id);
    }

    /*******************************************/
    /*  4. Pre-MLP RMSNorm over H              */
    /*******************************************/

    flex_global_barrier_xy();
    RMSNormInfo norm_mlp_info = Dsv3RMSNormAnaylze(
        (uint32_t)q_len,
        (uint32_t)LLM_D_MODEL,
        (uint64_t)LLM_H_ADDR,     // input: H after attention
        (uint64_t)LLM_H_NORM_ADDR // output: H_norm for MLP
    );

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
    {
        printf("[LLMForward] Layer %d: 4 -- Pre-MLP RMSNorm\n", layer_id);
    }

    /*****************************************************/
    /*  5 GEMM #1: H_norm -> MLP mid                    */
    /*     using SummaGEMM                               */
    /*****************************************************/
    //
    // Shapes according to gemm.h:
    //   X: [GEMM_M_SIZE x GEMM_K_SIZE]
    //   W: [GEMM_K_SIZE x GEMM_N_SIZE]
    //   Z: [GEMM_M_SIZE x GEMM_N_SIZE]
    //
    // For the LLM block, we conceptually want:
    //   [128 x 1024] * [1024 x 4096] -> [128 x 4096]
    // but the exact orientation depends on your current gemm.h config.
    // Here we simply use GEMM_* macros as-is.

    uint64_t W1 = llm_w1_up_addr(layer_id);

    flex_global_barrier_xy();
    SummaGEMMInfo gemm1_info = SummaGEMMAnaylze(
        (uint64_t)LLM_GEMM1_X_ADDR, // X_address (H_norm)
        (uint64_t)W1,               // W_address (MLP up weights)
        (uint64_t)LLM_GEMM1_Z_ADDR, // Z_address (MLP_mid)
        (uint32_t)q_len,
        (uint32_t)LLM_D_FF,
        (uint32_t)LLM_D_MODEL, // shared dimension
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
    {
        printf("[LLMForward] Layer %d: 5 -- GEMM1 (up)\n", layer_id);
    }

    /*******************************************/
    /*  6 SiLU Activation on GEMM1 output     */
    /*******************************************/
    //
    // Using the Activation kernel with ACTI_M_SIZE x ACTI_N_SIZE.
    // We run it in-place on the GEMM1 output at LLM_GEMM1_Z_ADDR.

    flex_global_barrier_xy();
    ActivationInfo act_info = ActivationAnaylze(
        (uint32_t)q_len,          // num_total_token
        (uint32_t)LLM_D_FF,          // token_embedded_length
        (uint32_t)ACTI_GATE_ENABLE,     // gate_enable
        (uint32_t)ACTI_BIAS_ENABLE,     // bias_enable
        (uint64_t)LLM_ACTI_INPUT_ADDR,  // input_address
        (uint64_t)LLM_ACTI_OUTPUT_ADDR, // output_address
        (uint64_t)0,                    // gate_address (unused)
        (uint64_t)0                     // bias_address (unused)
    );

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
    {
        printf("[LLMForward] Layer %d: 6 -- SiLU\n", layer_id);
    }

    /*****************************************************/
    /*  7 GEMM #2: MLP mid -> MLP out                   */
    /*****************************************************/
    //
    //
    //   X: LLM_GEMM2_X_ADDR  (MLP_mid, after SiLU)
    //   Z: LLM_GEMM2_Z_ADDR  (MLP_out)
    // Conceptually [LLM_T x LLM_D_FF] * [LLM_D_FF x LLM_D_MODEL] -> [LLM_T x LLM_D_MODEL]
    //              [128 x 4096] * [4096 x 1024] -> [128 x 1024].
    uint64_t W2 = llm_w2_down_addr(layer_id);

    flex_global_barrier_xy();
    SummaGEMMInfo gemm2_info = SummaGEMMAnaylze(
        (uint64_t)LLM_GEMM2_X_ADDR,
        (uint64_t)W2,
        (uint64_t)LLM_GEMM2_Z_ADDR,
        (uint32_t)q_len,       // 128
        (uint32_t)LLM_D_MODEL, // 1024
        (uint32_t)LLM_D_FF,    // 4096
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
    {
        printf("[LLMForward] Layer %d: 7 -- GEMM2\n", layer_id);
    }

    // For debug
#if LLM_DEBUG_DUMP
    llm_dma_dump_u16((uint64_t)LLM_MLP_OUT_ADDR, 8);
#endif
    /**************************************/
    /*  8. Residual: H = H + MLP_out      */
    /**************************************/
    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_start();

    flex_global_barrier_xy();
    // MLP residual: H = H + MLP_out
    llm_residual_add_spatz((uint64_t)LLM_H_ADDR, (uint64_t)LLM_MLP_OUT_ADDR, q_len, (uint32_t)LLM_D_MODEL);

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        flex_timer_end();

    flex_global_barrier_xy();
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
    {
        printf("[LLMForward] Layer %d: 8 -- MLP residual applied\n", layer_id);
    }
}

int main()
{
    uint32_t eoc_val = 0;

    uint32_t q_len  = (uint32_t)LLM_T;
    uint32_t kv_len = (uint32_t)LLM_T;

    // Initialize XY barriers for the Flex clusters and cores.
    flex_barrier_xy_init();
    // Init weights and other data in HBM (only the first layer right now)
    llm_init_dummy_weights(0);

    // For debug
#if LLM_DEBUG_DUMP
    llm_dma_dump_u16(llm_wq_addr(0), 16);
#endif

    // Initialize hidden state H at LLM_H_ADDR
    llm_init_hidden_state();
    flex_global_barrier_xy();

    // Only cluster 0, core 0 prints control messages to avoid clutter.
    if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
    {
        printf("[LLMForward] Starting LLM forward prototype with %d layers\n",
               LLM_NUM_LAYERS);
    }

    // Run LLM_NUM_LAYERS transformer-like layers
    for (int layer = 0; layer < (int)LLM_NUM_LAYERS; ++layer)
    {
        flex_global_barrier_xy();

        if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
            flex_timer_start();
        flex_global_barrier_xy();

        llm_layer_forward(layer, q_len, kv_len);

        flex_global_barrier_xy();
        if (flex_get_core_id() == 0 && flex_get_cluster_id() == 0)
        {
            flex_timer_end();
            printf("[LLMForward] ===== Layer %d complete =====\n", layer);
        }
    }

    // dump a few elements of final H for sanity
    llm_dma_dump_u16((uint64_t)LLM_H_ADDR, 8); // prints H[0..7] as fp16 bit-patterns

    /**************************************/
    /*  Program Execution Region -- Stop  */
    /**************************************/
    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
