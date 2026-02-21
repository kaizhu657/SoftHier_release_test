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

#endif
