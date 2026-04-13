#pragma once

// Detect arch
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #define GWLDCORE_ARCH_X86 1
#else
  #define GWLDCORE_ARCH_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__ARM_ARCH)
  #define GWLDCORE_ARCH_ARM 1
#else
  #define GWLDCORE_ARCH_ARM 0
#endif

// Pull intrinsics only where they exist
#if GWLDCORE_ARCH_X86
  #include <immintrin.h>   // AVX/AVX2/FMA/… (x86 only)
#elif GWLDCORE_ARCH_ARM
  #include <arm_neon.h>    // NEON (ARM only) — optional, see note below
#endif
