#ifndef _LLM_LAYOUT_H_
#define _LLM_LAYOUT_H_

#include <stdint.h>

// =======================================================
// Model hyperparameters
// =======================================================
#define LLM_B          1
#define LLM_T          128
#define LLM_D_MODEL    1024
#define LLM_D_FF       4096
#define LLM_N_HEAD     16
#define LLM_HEAD_DIM   64
#define LLM_NUM_LAYERS 1
#define LLM_ELEM_SIZE  2        // fp16 bytes


// Sizes in bytes
#define BYTES_H        (LLM_T * LLM_D_MODEL * LLM_ELEM_SIZE)   // 128*1024*2 = 0x40000
#define BYTES_H_NORM   BYTES_H
#define BYTES_MLP_MID  (LLM_T * LLM_D_FF * LLM_ELEM_SIZE)      // 128*4096*2 = 0x100000
#define BYTES_MLP_OUT  BYTES_H

#define BYTES_HM_HEAD   (LLM_T * LLM_HEAD_DIM * LLM_ELEM_SIZE)      // 128*64*2 = 16384
#define BYTES_HM_ALL    (LLM_N_HEAD * BYTES_HM_HEAD)                // 16 * 16384 = 262144 (same as BYTES_H)


// =======================================================
// Activation layout in HBM
// =======================================================
//
// All of these buffers are reused across layers; only H is persistent.
// Sizes (fp16):
//   H          : B * T * d_model = 1 * 128 * 1024 = 131072 elements ≈ 256 KB
//   H_norm     : same as H
//   MLP_mid    : B * T * d_ff    = 1 * 128 * 4096 = 524288 elements ≈ 1 MB
//   MLP_out    : same as H
//   Q/K/V/O    : B * T * n_heads * head_dim
//              = 1 * 128 * 16 * 64 = 131072 elements ≈ 256 KB each
//
// We'll place them contiguously starting from 0xC0000000.

// Activation layout in 0xC0... space
#define LLM_H_ADDR        ((uint64_t)0xC0000000)                 // [T, d_model], 0x40000 bytes
#define LLM_H_NORM_ADDR   (LLM_H_ADDR       + BYTES_H)           // 0xC0040000
#define LLM_MLP_MID_ADDR  (LLM_H_NORM_ADDR  + BYTES_H_NORM)      // 0xC0080000 (1MB region)
#define LLM_MLP_OUT_ADDR  (LLM_MLP_MID_ADDR + BYTES_MLP_MID)     // 0xC0180000
#define LLM_ATTN_O_ADDR   (LLM_MLP_OUT_ADDR + BYTES_MLP_OUT)     // 0xC01C0000


#define LLM_QKV_TM_ADDR     (LLM_ATTN_O_ADDR + BYTES_H)             // scratch [T,d_model]
#define LLM_Q_HM_ADDR       (LLM_QKV_TM_ADDR + BYTES_H)             // [head,T,head_dim]
#define LLM_K_HM_ADDR       (LLM_Q_HM_ADDR   + BYTES_HM_ALL)
#define LLM_V_HM_ADDR       (LLM_K_HM_ADDR   + BYTES_HM_ALL)
#define LLM_O_HM_ADDR       (LLM_V_HM_ADDR   + BYTES_HM_ALL)

// reuse MLP_OUT as attn projection output
#define LLM_ATTN_PROJ_TM_ADDR  (LLM_MLP_OUT_ADDR)

// Weight base (still using SummaGEMM preload region)
#define LLM_W_BASE        ((uint64_t)0x9C0000000)


#endif // _LLM_LAYOUT_H_