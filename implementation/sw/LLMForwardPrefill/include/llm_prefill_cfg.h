#ifndef _LLM_PREFILL_CFG_H_
#define _LLM_PREFILL_CFG_H_

// Default prompt length for prefill-only experiments.
#ifndef LLM_PREFILL_PROMPT_LEN
#define LLM_PREFILL_PROMPT_LEN LLM_T
#endif

// Dump minimal outputs for quick sanity checks.
#ifndef LLM_PREFILL_DEBUG_DUMP
#define LLM_PREFILL_DEBUG_DUMP 1
#endif

#endif
