#ifndef _LLM_COMMON_DECODE_FORWARD_COMPAT_H_
#define _LLM_COMMON_DECODE_FORWARD_COMPAT_H_

// Decode processes one query token per step; keep decode activations compact.
#ifndef LLM_T
#define LLM_T 1
#endif

#include "../../LLMForwardCommon/include/llm_common.h"

#endif
