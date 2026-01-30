#include "flex_runtime.h"
#include "flex_printf.h"
#include "gemm.h"
#include "SummaGEMM.h"

// Put test matrices in HBM
#define X_ADDR ((uint64_t)0xC0000000)
#define W_ADDR ((uint64_t)0xC0100000)
#define Z_ADDR ((uint64_t)0xC0400000)

static void dma_dump_u16(uint64_t hbm_addr, uint32_t n_halfwords)
{
    flex_global_barrier_xy();
    if (flex_get_cluster_id() != 0) { flex_global_barrier_xy(); return; }

    uint32_t l1 = local(0);
    l1 = (l1 + 63) & ~((uint32_t)63);

    uint32_t nbytes = n_halfwords * 2;
    if (flex_is_dm_core()) {
        flex_dma_async_1d((uint64_t)l1, hbm_addr, nbytes); // HBM->L1
        flex_dma_async_wait_all();
    }
    flex_intra_cluster_sync();

    if (flex_get_core_id() == 0) {
        volatile uint16_t *p = (volatile uint16_t*)l1;
        printf("[DUMP] :");
        for (uint32_t i=0;i<n_halfwords;i++) printf(" 0x%04x", (unsigned)p[i]);
        printf("\n");
    }
    flex_intra_cluster_sync();
    flex_global_barrier_xy();
}

static inline uint16_t fp16_const(uint32_t k)
{
    static const uint16_t lut[4] = {0x3400, 0x3800, 0x3C00, 0x4000}; // 0.25,0.5,1,2
    return lut[k & 3];
}

static void init_X(void)
{
    const uint32_t row_bytes = (uint32_t)(GEMM_K_SIZE * 2); // K columns
    uint32_t l1 = local(0);
    l1 = (l1 + 63) & ~((uint32_t)63);

    for (uint32_t r = flex_get_cluster_id(); r < (uint32_t)GEMM_M_SIZE; r += ARCH_NUM_CLUSTER)
    {
        if (flex_get_core_id() == 0) {
            uint16_t *buf = (uint16_t*)l1;
            for (uint32_t c=0;c<(uint32_t)GEMM_K_SIZE;c++) buf[c] = fp16_const(r + c);
        }
        flex_intra_cluster_sync();

        if (flex_is_dm_core()) {
            flex_dma_async_1d(X_ADDR + (uint64_t)r * row_bytes, (uint64_t)l1, row_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
    }
    flex_global_barrier_xy();
}

static void init_W_identity(void)
{
    const uint32_t row_bytes = (uint32_t)(GEMM_N_SIZE * 2);
    uint32_t l1 = local(0);
    l1 = (l1 + 63) & ~((uint32_t)63);

    for (uint32_t r = flex_get_cluster_id(); r < (uint32_t)GEMM_K_SIZE; r += ARCH_NUM_CLUSTER)
    {
        if (flex_get_core_id() == 0) {
            uint16_t *buf = (uint16_t*)l1;
            for (uint32_t c=0;c<(uint32_t)GEMM_N_SIZE;c++) buf[c] = 0;
            buf[r] = 0x3C00; // 1.0
        }
        flex_intra_cluster_sync();

        if (flex_is_dm_core()) {
            flex_dma_async_1d(W_ADDR + (uint64_t)r * row_bytes, (uint64_t)l1, row_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
    }
    flex_global_barrier_xy();
}

static void zero_Z(void)
{
    // simple zero fill row-by-row
    const uint32_t row_bytes = (uint32_t)(GEMM_N_SIZE * 2);
    uint32_t l1 = local(0);
    l1 = (l1 + 63) & ~((uint32_t)63);

    if (flex_get_core_id() == 0) {
        uint16_t *buf = (uint16_t*)l1;
        for (uint32_t c=0;c<(uint32_t)GEMM_N_SIZE;c++) buf[c] = 0;
    }
    flex_intra_cluster_sync();

    for (uint32_t r = flex_get_cluster_id(); r < (uint32_t)GEMM_M_SIZE; r += ARCH_NUM_CLUSTER)
    {
        if (flex_is_dm_core()) {
            flex_dma_async_1d(Z_ADDR + (uint64_t)r * row_bytes, (uint64_t)l1, row_bytes);
            flex_dma_async_wait_all();
        }
        flex_intra_cluster_sync();
    }
    flex_global_barrier_xy();
}

int main()
{
    uint32_t eoc_val = 0;
    flex_barrier_xy_init();
    flex_global_barrier_xy();

    init_X();
    init_W_identity();
    zero_Z();

    // sanity: dump X row0 and W row0
    dma_dump_u16(X_ADDR, 8); // should be 3400 3800 3c00 4000 ...
    dma_dump_u16(W_ADDR, 16); // should be 3c00 0000 ...

    SummaGEMMInfo info = SummaGEMMAnaylze(
        X_ADDR, W_ADDR, Z_ADDR,
        (uint32_t)GEMM_M_SIZE, (uint32_t)GEMM_N_SIZE, (uint32_t)GEMM_K_SIZE,
        (uint32_t)GEMM_M_TILE, (uint32_t)GEMM_N_TILE, (uint32_t)GEMM_K_TILE,
        (uint32_t)GEMM_SUMMA_SCALE_X, (uint32_t)GEMM_SUMMA_SCALE_Y,
        (uint32_t)GEMM_SUMMA_GROUP_NUMBER,
        (uint32_t)GEMM_SUMMA_GROUP_REDUCE,
        (uint32_t)GEMM_SUMMA_GROUP_SPLITK,
        (uint32_t)GEMM_SUMMA_GROUP_SPLITN,
        (uint32_t)GEMM_SUMMA_GROUP_GAP_X,
        (uint32_t)GEMM_SUMMA_GROUP_GAP_W,
        (uint32_t)GEMM_SUMMA_GROUP_GAP_Z
    );

    flex_global_barrier_xy();
    if (flex_get_cluster_id()==0 && flex_get_core_id()==0) flex_timer_start();
    flex_global_barrier_xy();

    SummaGEMMRun(&info);

    flex_global_barrier_xy();
    if (flex_get_cluster_id()==0 && flex_get_core_id()==0) flex_timer_end();
    flex_global_barrier_xy();

    // result should match X (since W=I)
    dma_dump_u16(Z_ADDR, 8);

    flex_global_barrier_xy();
    flex_eoc(eoc_val);
    return 0;
}
