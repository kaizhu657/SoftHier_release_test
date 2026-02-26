#ifndef _LLM_PREFILL_CFG_H_
#define _LLM_PREFILL_CFG_H_

#include "llm_common.h"

// Default prompt length for prefill-only experiments.
#ifndef LLM_PREFILL_PROMPT_LEN
#define LLM_PREFILL_PROMPT_LEN LLM_T
#endif

// Dump minimal outputs for quick sanity checks.
#ifndef LLM_PREFILL_DEBUG_DUMP
#define LLM_PREFILL_DEBUG_DUMP 1
#endif

// Prefill attention profile (32x32-friendly, full-chip group).
#ifndef LLM_PREFILL_ATTN_SPECULATIVE_LENGTH
#define LLM_PREFILL_ATTN_SPECULATIVE_LENGTH 1u
#endif
#ifndef LLM_PREFILL_ATTN_HEAD_DIM
#define LLM_PREFILL_ATTN_HEAD_DIM LLM_HEAD_DIM
#endif
#ifndef LLM_PREFILL_ATTN_NUM_HEAD
#define LLM_PREFILL_ATTN_NUM_HEAD LLM_N_HEAD
#endif
#ifndef LLM_PREFILL_ATTN_NUM_HEAD_GROUP
#define LLM_PREFILL_ATTN_NUM_HEAD_GROUP LLM_N_KV_HEAD
#endif
#ifndef LLM_PREFILL_ATTN_BATCH_SIZE
#define LLM_PREFILL_ATTN_BATCH_SIZE 1u
#endif
#ifndef LLM_PREFILL_ATTN_FLATTEN_SCALE_X
#define LLM_PREFILL_ATTN_FLATTEN_SCALE_X 32u
#endif
#ifndef LLM_PREFILL_ATTN_FLATTEN_SCALE_Y
#define LLM_PREFILL_ATTN_FLATTEN_SCALE_Y 32u
#endif
#ifndef LLM_PREFILL_ATTN_FLATTEN_SHAPE_X
#define LLM_PREFILL_ATTN_FLATTEN_SHAPE_X 1024u
#endif
#ifndef LLM_PREFILL_ATTN_FLATTEN_SHAPE_Y
#define LLM_PREFILL_ATTN_FLATTEN_SHAPE_Y 1024u
#endif
#ifndef LLM_PREFILL_ATTN_ASYNC_ENABLE
#define LLM_PREFILL_ATTN_ASYNC_ENABLE 0u
#endif
#ifndef LLM_PREFILL_ATTN_DUMP_ENABLE
#define LLM_PREFILL_ATTN_DUMP_ENABLE 0u
#endif

static inline void llm_init_attn_prefill(LLMAttentionRuntimeArgs *prefill_attn)
{
    if (prefill_attn == 0)
        return;

    prefill_attn->speculative_length = (uint32_t)LLM_PREFILL_ATTN_SPECULATIVE_LENGTH;
    prefill_attn->head_dimension = (uint32_t)LLM_PREFILL_ATTN_HEAD_DIM;
    prefill_attn->num_head = (uint32_t)LLM_PREFILL_ATTN_NUM_HEAD;
    prefill_attn->num_head_group = (uint32_t)LLM_PREFILL_ATTN_NUM_HEAD_GROUP;
    prefill_attn->batch_size = (uint32_t)LLM_PREFILL_ATTN_BATCH_SIZE;
    prefill_attn->flatten_scale_x = (uint32_t)LLM_PREFILL_ATTN_FLATTEN_SCALE_X;
    prefill_attn->flatten_scale_y = (uint32_t)LLM_PREFILL_ATTN_FLATTEN_SCALE_Y;
    prefill_attn->flatten_shape_x = (uint32_t)LLM_PREFILL_ATTN_FLATTEN_SHAPE_X;
    prefill_attn->flatten_shape_y = (uint32_t)LLM_PREFILL_ATTN_FLATTEN_SHAPE_Y;
    prefill_attn->async_enable = (uint32_t)LLM_PREFILL_ATTN_ASYNC_ENABLE;
    prefill_attn->dump_enable = (uint32_t)LLM_PREFILL_ATTN_DUMP_ENABLE;
}

// Prefill GEMM mapping knobs.
// 0: conservative bring-up on one 8x8 group, 1: full-chip 1024-cluster mapping.
#ifndef LLM_PREFILL_GEMM_USE_FULL_CHIP
#define LLM_PREFILL_GEMM_USE_FULL_CHIP 1u
#endif

#ifndef LLM_PREFILL_GEMM_M_TILE
#define LLM_PREFILL_GEMM_M_TILE 64u
#endif
#ifndef LLM_PREFILL_GEMM_N_TILE
#define LLM_PREFILL_GEMM_N_TILE 24u
#endif
#ifndef LLM_PREFILL_GEMM_K_TILE
#define LLM_PREFILL_GEMM_K_TILE 64u
#endif
#ifndef LLM_PREFILL_GEMM_SUMMA_SCALE_X
// Bring-up mapping: run GEMM on one 8x8 group first (64 clusters).
#if LLM_PREFILL_GEMM_USE_FULL_CHIP
#define LLM_PREFILL_GEMM_SUMMA_SCALE_X 32u
#else
#define LLM_PREFILL_GEMM_SUMMA_SCALE_X 8u
#endif
#endif
#ifndef LLM_PREFILL_GEMM_SUMMA_SCALE_Y
#if LLM_PREFILL_GEMM_USE_FULL_CHIP
#define LLM_PREFILL_GEMM_SUMMA_SCALE_Y 32u
#else
#define LLM_PREFILL_GEMM_SUMMA_SCALE_Y 8u
#endif
#endif
#ifndef LLM_PREFILL_GEMM_GROUP_NUMBER
// Single full-chip group avoids split-N cross-group synchronization hazards.
#define LLM_PREFILL_GEMM_GROUP_NUMBER 1u
#endif
#ifndef LLM_PREFILL_GEMM_GROUP_REDUCE
#define LLM_PREFILL_GEMM_GROUP_REDUCE 0u
#endif
#ifndef LLM_PREFILL_GEMM_GROUP_SPLITK
#define LLM_PREFILL_GEMM_GROUP_SPLITK 0u
#endif
#ifndef LLM_PREFILL_GEMM_GROUP_SPLITN
// No split-N while using a single 8x8 group.
#define LLM_PREFILL_GEMM_GROUP_SPLITN 0u
#endif
#ifndef LLM_PREFILL_GEMM_GROUP_GAP_X
#define LLM_PREFILL_GEMM_GROUP_GAP_X 0u
#endif
#ifndef LLM_PREFILL_GEMM_GROUP_GAP_W
#define LLM_PREFILL_GEMM_GROUP_GAP_W 0u
#endif
#ifndef LLM_PREFILL_GEMM_GROUP_GAP_Z
#define LLM_PREFILL_GEMM_GROUP_GAP_Z 0u
#endif

#endif
