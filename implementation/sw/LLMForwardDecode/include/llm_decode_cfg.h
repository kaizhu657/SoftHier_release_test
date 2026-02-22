#ifndef _LLM_DECODE_CFG_H_
#define _LLM_DECODE_CFG_H_

// Initial cache length (tokens) when decode starts.
#ifndef LLM_DECODE_INIT_CACHE_LEN
#define LLM_DECODE_INIT_CACHE_LEN 0
#endif

// Number of decode iterations to run.
#ifndef LLM_DECODE_STEPS
#define LLM_DECODE_STEPS 8
#endif

// Dump minimal outputs for quick sanity checks.
#ifndef LLM_DECODE_DEBUG_DUMP
#define LLM_DECODE_DEBUG_DUMP 1
#endif

// Decode-safe runtime attention profile.
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
// Decode uses q_len=1, so a minimal 1x1 flatten partition is safer.
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

#endif
