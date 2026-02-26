#ifndef _LLM_DECODE_CFG_H_
#define _LLM_DECODE_CFG_H_

#include "llm_common.h"

// Initial cache length (tokens) when decode starts.
#ifndef LLM_DECODE_INIT_CACHE_LEN
#define LLM_DECODE_INIT_CACHE_LEN 0
#endif

// Number of decode iterations to run.
#ifndef LLM_DECODE_STEPS
#define LLM_DECODE_STEPS 2
#endif

// Dump minimal outputs for quick sanity checks.
#ifndef LLM_DECODE_DEBUG_DUMP
#define LLM_DECODE_DEBUG_DUMP 1
#endif

// Query tokens processed per decode iteration.
// decode semantics
#ifndef LLM_DECODE_QUERY_TOKENS
#define LLM_DECODE_QUERY_TOKENS 1u
#endif

// Decode runtime attention profile.
#ifndef LLM_DECODE_ATTN_SPECULATIVE_LENGTH
#define LLM_DECODE_ATTN_SPECULATIVE_LENGTH 1u
#endif
#ifndef LLM_DECODE_ATTN_HEAD_DIM
#define LLM_DECODE_ATTN_HEAD_DIM LLM_HEAD_DIM
#endif
#ifndef LLM_DECODE_ATTN_NUM_HEAD
#define LLM_DECODE_ATTN_NUM_HEAD LLM_N_HEAD
#endif
#ifndef LLM_DECODE_ATTN_NUM_HEAD_GROUP
#define LLM_DECODE_ATTN_NUM_HEAD_GROUP LLM_N_KV_HEAD
#endif
#ifndef LLM_DECODE_ATTN_BATCH_SIZE
#define LLM_DECODE_ATTN_BATCH_SIZE 1u
#endif
#ifndef LLM_DECODE_ATTN_FLATTEN_SCALE_X
// q_len=1 decode-safe profile: keep flattening minimal.
#define LLM_DECODE_ATTN_FLATTEN_SCALE_X 1u
#endif
#ifndef LLM_DECODE_ATTN_FLATTEN_SCALE_Y
#define LLM_DECODE_ATTN_FLATTEN_SCALE_Y 1u
#endif
#ifndef LLM_DECODE_ATTN_FLATTEN_SHAPE_X
#define LLM_DECODE_ATTN_FLATTEN_SHAPE_X 1u
#endif
#ifndef LLM_DECODE_ATTN_FLATTEN_SHAPE_Y
#define LLM_DECODE_ATTN_FLATTEN_SHAPE_Y 1u
#endif
#ifndef LLM_DECODE_ATTN_ASYNC_ENABLE
#define LLM_DECODE_ATTN_ASYNC_ENABLE 0u
#endif
#ifndef LLM_DECODE_ATTN_DUMP_ENABLE
#define LLM_DECODE_ATTN_DUMP_ENABLE 0u
#endif

static inline void llm_init_attn_decode(LLMAttentionRuntimeArgs *decode_attn)
{
    if (decode_attn == 0)
        return;

    decode_attn->speculative_length = (uint32_t)LLM_DECODE_ATTN_SPECULATIVE_LENGTH;
    decode_attn->head_dimension = (uint32_t)LLM_DECODE_ATTN_HEAD_DIM;
    decode_attn->num_head = (uint32_t)LLM_DECODE_ATTN_NUM_HEAD;
    decode_attn->num_head_group = (uint32_t)LLM_DECODE_ATTN_NUM_HEAD_GROUP;
    decode_attn->batch_size = (uint32_t)LLM_DECODE_ATTN_BATCH_SIZE;
    decode_attn->flatten_scale_x = (uint32_t)LLM_DECODE_ATTN_FLATTEN_SCALE_X;
    decode_attn->flatten_scale_y = (uint32_t)LLM_DECODE_ATTN_FLATTEN_SCALE_Y;
    decode_attn->flatten_shape_x = (uint32_t)LLM_DECODE_ATTN_FLATTEN_SHAPE_X;
    decode_attn->flatten_shape_y = (uint32_t)LLM_DECODE_ATTN_FLATTEN_SHAPE_Y;
    decode_attn->async_enable = (uint32_t)LLM_DECODE_ATTN_ASYNC_ENABLE;
    decode_attn->dump_enable = (uint32_t)LLM_DECODE_ATTN_DUMP_ENABLE;
}

// Decode-only GEMM mapping knobs.
// 0: conservative bring-up on one 8x8 group, 1: full-chip 1024-cluster mapping.
#ifndef LLM_DECODE_GEMM_USE_FULL_CHIP
#define LLM_DECODE_GEMM_USE_FULL_CHIP 1u
#endif

#ifndef LLM_DECODE_GEMM_M_TILE
#define LLM_DECODE_GEMM_M_TILE 32u
#endif
#ifndef LLM_DECODE_GEMM_N_TILE
#define LLM_DECODE_GEMM_N_TILE 64u
#endif
#ifndef LLM_DECODE_GEMM_K_TILE
#define LLM_DECODE_GEMM_K_TILE 64u
#endif
#ifndef LLM_DECODE_GEMM_SUMMA_SCALE_X
// Bring-up mapping: run GEMM on one 8x8 group first (64 clusters).
#if LLM_DECODE_GEMM_USE_FULL_CHIP
#define LLM_DECODE_GEMM_SUMMA_SCALE_X 32u
#else
#define LLM_DECODE_GEMM_SUMMA_SCALE_X 8u
#endif
#endif
#ifndef LLM_DECODE_GEMM_SUMMA_SCALE_Y
#if LLM_DECODE_GEMM_USE_FULL_CHIP
#define LLM_DECODE_GEMM_SUMMA_SCALE_Y 32u
#else
#define LLM_DECODE_GEMM_SUMMA_SCALE_Y 8u
#endif
#endif
#ifndef LLM_DECODE_GEMM_GROUP_NUMBER
// Single full-chip group avoids split-N cross-group synchronization hazards.
#define LLM_DECODE_GEMM_GROUP_NUMBER 1u
#endif
#ifndef LLM_DECODE_GEMM_GROUP_REDUCE
#define LLM_DECODE_GEMM_GROUP_REDUCE 0u
#endif
#ifndef LLM_DECODE_GEMM_GROUP_SPLITK
#define LLM_DECODE_GEMM_GROUP_SPLITK 0u
#endif
#ifndef LLM_DECODE_GEMM_GROUP_SPLITN
// No split-N while using a single 8x8 group.
#define LLM_DECODE_GEMM_GROUP_SPLITN 0u
#endif
#ifndef LLM_DECODE_GEMM_GROUP_GAP_X
#define LLM_DECODE_GEMM_GROUP_GAP_X 0u
#endif
#ifndef LLM_DECODE_GEMM_GROUP_GAP_W
#define LLM_DECODE_GEMM_GROUP_GAP_W 0u
#endif
#ifndef LLM_DECODE_GEMM_GROUP_GAP_Z
#define LLM_DECODE_GEMM_GROUP_GAP_Z 0u
#endif

#endif
