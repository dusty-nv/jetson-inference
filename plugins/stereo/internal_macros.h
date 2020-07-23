// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#ifndef REDTAIL_INTERNAL_MACROS_H
#define REDTAIL_INTERNAL_MACROS_H

// -----------------------------------------------------------------
// CHECKL and CHECK macros check for status and log the message
// in case of status != 0. The difference between macros is CHECKL
// takes log argument explicitly while CHECK - assumes log_ variable
// is defined (used in class memebers).
// -----------------------------------------------------------------
#undef  CHECKL
#define CHECKL(status, log) do {                                                           \
    auto res = (status);                                                             \
    if ((int)res != 0)                                                               \
        redtail::tensorrt::reportError(res, __FILE__, __LINE__, __FUNCTION__, log); \
} while(false)

#undef  CHECK
#define CHECK(status) CHECKL(status, log_)

// -----------------------------------------------------------------
// UNUSED is generic version of macro that allows to avoid unused
// variable warning.
// UNUSEDR disables such warning only in Release builds.
// -----------------------------------------------------------------
#undef  UNUSED
#define UNUSED(x) ((void)(x))

#undef  UNUSEDR
#ifdef NDEBUG
    #define UNUSEDR(x) ((void)(x))
#else
    #define UNUSEDR(x)
#endif

#endif